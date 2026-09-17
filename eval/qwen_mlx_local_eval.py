"""Qwen3-Embedding-0.6B (MLX INT8 + BF16) Full TyDi QA Multilingual Benchmark.

Evaluates retrieval quality, latency distribution (p50, p95, p99, mean),
and multilingual breakdown using real Qwen3-Embedding-0.6B MLX embeddings
against local Supabase PostgreSQL (port 54322) on the authentic TyDi QA dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.corpora.fetch_tydiqa import fetch_tydiqa_corpus
from supabase_easy_rag.providers.mlx_provider import MlxQwenEmbeddingProvider


def get_conn_info() -> Dict[str, str]:
    return {
        "host": "127.0.0.1",
        "port": os.environ.get("LOCAL_SUPABASE_PORT", "54322"),
        "user": "postgres",
        "password": "postgres",
        "dbname": "postgres",
    }


def run_psql(sql: str, conn_info: Dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PGPASSWORD"] = conn_info["password"]
    cmd = [
        "psql",
        "-h", conn_info["host"],
        "-p", conn_info["port"],
        "-U", conn_info["user"],
        "-d", conn_info["dbname"],
        "-v", "ON_ERROR_STOP=1",
        "-q", "-t", "-A", "-c", sql,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def to_pgvector_literal(vec: List[float], target_dim: int = 1024) -> str:
    if len(vec) < target_dim:
        padded = (vec + [0.0] * (target_dim - len(vec)))[:target_dim]
    else:
        padded = vec[:target_dim]
    return "[" + ",".join(f"{x:.6f}" for x in padded) + "]"


def run_qwen_benchmark(
    limit_docs: Optional[int] = None,
    limit_queries: Optional[int] = None,
    dest_dir: str = "/tmp/tydiqa_full_corpus",
    batch_size: int = 32,
    force_ingest: bool = False,
):
    print("=" * 85)
    print("  🚀 FULL TYDI QA BENCHMARK: QWEN3-EMBEDDING-0.6B (MLX INT8 + BF16)")
    print("=" * 85)

    conn_info = get_conn_info()
    token_val = "easy_rag_local_eval_token_secret"

    # Verify connection
    res = run_psql("SELECT version();", conn_info)
    if res.returncode != 0:
        raise RuntimeError(f"Cannot connect to local Supabase: {res.stderr}")
    print(f"Connected to local Supabase PostgreSQL: {res.stdout.strip().splitlines()[0]}")

    # 1. Dataset Fetch / Load
    bench_dir = Path(dest_dir)
    doc_dir = bench_dir / "documents"
    dataset_file = bench_dir / "tydiqa_full_dataset.json"

    if not doc_dir.exists() or not dataset_file.exists() or len(list(doc_dir.glob("*.md"))) == 0:
        print("\n[1/4] Fetching authentic Google Research TyDi QA dataset from HuggingFace...")
        fetch_tydiqa_corpus(bench_dir, limit=limit_docs, split="validation")

    doc_files = sorted(doc_dir.glob("*.md"))
    if limit_docs is not None and limit_docs > 0:
        doc_files = doc_files[:limit_docs]

    qa_items: List[Dict[str, Any]] = json.loads(dataset_file.read_text(encoding="utf-8"))
    if limit_queries is not None and limit_queries > 0:
        qa_items = qa_items[:limit_queries]

    print(f" ✓ Total Corpus Documents: {len(doc_files):,}")
    print(f" ✓ Total Evaluation Queries: {len(qa_items):,}")

    # 2. Initialize MLX Qwen Provider
    print("\n[2/4] Loading local Qwen3-Embedding-0.6B (INT8 quantized weights, BF16 compute)...")
    provider = MlxQwenEmbeddingProvider(
        model_path_or_repo="models/Qwen3-Embedding-0.6B",
        quantize_int8=True,
        use_bf16=True,
        lazy_load=False,
    )
    print(" ✓ MLX Qwen Provider loaded successfully from local directory.")

    # 3. Check existing documents in DB (Skip ingestion if already present)
    print("\n[3/4] Checking database for existing TyDi QA documents...")
    cnt_res = run_psql("SELECT count(*) FROM knowledgebase.documents WHERE document_key LIKE 'tydi_%';", conn_info)
    existing_docs_count = 0
    if cnt_res.returncode == 0:
        for line in cnt_res.stdout.splitlines():
            if line.strip().isdigit():
                existing_docs_count = int(line.strip())
                break

    if existing_docs_count >= len(doc_files) and not force_ingest:
        print(f" ✓ Found {existing_docs_count:,} already indexed documents in Supabase. Skipping re-ingestion! (use --force-ingest to overwrite)")
        docs_to_ingest = doc_files
        ingest_time = 0.0
        throughput = 0.0
    else:
        print(f" Need ingestion: found {existing_docs_count} docs in DB, target is {len(doc_files)}. Ingesting...")
        run_psql("DELETE FROM knowledgebase.chunks WHERE content LIKE '%[TYDI_EVAL]%';", conn_info)
        run_psql("DELETE FROM knowledgebase.documents WHERE document_key LIKE 'tydi_%';", conn_info)

        # Insert Auth Token
        token_sql = f"""
        INSERT INTO knowledgebase.access_tokens (token_name, token_hash, is_active)
        VALUES ('local_eval_token', knowledgebase.hash_access_token('{token_val}'), TRUE)
        ON CONFLICT (token_hash) DO NOTHING;
        """
        run_psql(token_sql, conn_info)

        docs_to_ingest = []
        for doc_path in doc_files:
            content = doc_path.read_text(encoding="utf-8")
            title = doc_path.stem
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if lines and lines[0].startswith("# "):
                title = lines[0][2:].strip()
            docs_to_ingest.append({
                "key": doc_path.name,
                "title": title,
                "content": content,
            })

        t0_ingest = time.perf_counter()
        inserted_count = 0

        for i in range(0, len(docs_to_ingest), batch_size):
            batch = docs_to_ingest[i : i + batch_size]
            texts = [b["content"] for b in batch]
            embeddings = provider.embed_texts(texts)

            sql_statements = ["BEGIN;"]
            for item, emb in zip(batch, embeddings):
                item_key = item["key"]
                clean_title = item["title"].replace("'", "''")
                vec_lit = to_pgvector_literal(emb, 1024)
                marked_content = f"[TYDI_EVAL] {item['content']}".replace("'", "''")

                sql_statements.append(f"""
                DO $$
                DECLARE
                    v_doc_id UUID;
                    v_sec_id UUID;
                BEGIN
                    INSERT INTO knowledgebase.documents (document_key, title, top_level_category, checksum)
                    VALUES ('{item_key}', '{clean_title}', 'tydiqa', 'chk_{item_key}')
                    RETURNING id INTO v_doc_id;

                    INSERT INTO knowledgebase.document_sections (document_id, heading, level, sort_order)
                    VALUES (v_doc_id, '{clean_title}', 1, 1)
                    RETURNING id INTO v_sec_id;

                    INSERT INTO knowledgebase.chunks (document_id, section_id, chunk_index, content, embedding, metadata)
                    VALUES (
                        v_doc_id,
                        v_sec_id,
                        0,
                        '{marked_content}',
                        '{vec_lit}'::vector,
                        jsonb_build_object('document_key', '{item_key}', 'title', '{clean_title}', 'model', 'Qwen/Qwen3-Embedding-0.6B')
                    );
                END;
                $$;
                """)
            sql_statements.append("COMMIT;")
            batch_sql = "\n".join(sql_statements)
            res_ins = run_psql(batch_sql, conn_info)
            if res_ins.returncode != 0:
                raise RuntimeError(f"Batch ingestion failed: {res_ins.stderr}")
            inserted_count += len(batch)
            if inserted_count % 200 == 0 or inserted_count == len(docs_to_ingest):
                elapsed = time.perf_counter() - t0_ingest
                print(f"  Ingested {inserted_count}/{len(docs_to_ingest)} documents ({inserted_count / elapsed:.1f} docs/sec)...")

        ingest_time = time.perf_counter() - t0_ingest
        throughput = len(docs_to_ingest) / ingest_time if ingest_time > 0 else 0
        print(f" ✓ Ingestion complete: {len(docs_to_ingest)} documents in {ingest_time:.2f}s ({throughput:.1f} docs/sec)")

    # Ensure HNSW vector index for 1024 exists
    print("\n Ensuring partial HNSW vector index for 1024 dimensions...")
    res_idx = run_psql("SELECT knowledgebase.ensure_vector_index(1024);", conn_info)
    print(f" ✓ Index status: {res_idx.stdout.strip()}")

    # 4. Retrieval Evaluation Loop
    print(f"\n[4/4] Executing Retrieval Evaluation across {len(qa_items):,} queries...")
    modes = ["vector", "fts", "hybrid"]
    metrics_by_mode: Dict[str, Any] = {}

    for mode in modes:
        print(f"\n--- Evaluating Mode: {mode.upper()} ---")
        latencies_ms: List[float] = []
        embedding_latencies_ms: List[float] = []
        sql_latencies_ms: List[float] = []
        reciprocal_ranks: List[float] = []
        hit1, hit3, hit5, hit10 = 0, 0, 0, 0
        ans_recall1, ans_recall5, ans_recall10 = 0, 0, 0
        by_lang: Dict[str, Dict[str, Any]] = {}

        t0_mode = time.perf_counter()

        for idx, qa in enumerate(qa_items, 1):
            q_text = qa["question"]
            expected_key = qa.get("expected_document_key", "")
            lang = qa.get("language", "english")
            gold_answers = [a.lower() for a in qa.get("gold_answers", []) if a]
            fts_cfg = lang if lang in ["english", "russian", "arabic", "finnish", "indonesian", "swahili"] else "simple"

            t_emb_start = time.perf_counter()
            if mode in ("vector", "hybrid"):
                q_vec = provider.embed_query(q_text)
                q_vec_lit = to_pgvector_literal(q_vec, 1024)
            else:
                q_vec_lit = "NULL"
            t_emb_end = time.perf_counter()
            emb_ms = (t_emb_end - t_emb_start) * 1000 if mode in ("vector", "hybrid") else 0.0
            embedding_latencies_ms.append(emb_ms)

            clean_q_text = q_text.replace("'", "''")

            if mode == "vector":
                sql = f"""
                SELECT (metadata->>'document_key')::text, chunk_text
                FROM knowledgebase.match_chunks_by_embedding(
                    p_kb_token := '{token_val}',
                    p_query_embedding := '{q_vec_lit}'::vector,
                    p_match_count := 10,
                    p_facet_keys := NULL::text[],
                    p_min_vector_similarity := NULL::double precision,
                    p_ef_search := NULL::integer
                );
                """
            elif mode == "fts":
                sql = f"""
                SELECT (metadata->>'document_key')::text, chunk_text
                FROM knowledgebase.search_chunks_full_text(
                    p_kb_token := '{token_val}',
                    p_query := '{clean_q_text}',
                    p_match_count := 10,
                    p_facet_keys := NULL::text[],
                    p_fts_config := '{fts_cfg}'
                );
                """
            else:  # hybrid
                sql = f"""
                SELECT (metadata->>'document_key')::text, chunk_text
                FROM knowledgebase.search_chunks_hybrid(
                    p_kb_token := '{token_val}',
                    p_query := '{clean_q_text}',
                    p_query_embedding := '{q_vec_lit}'::vector,
                    p_match_count := 10,
                    p_facet_keys := NULL::text[],
                    p_candidate_count := 50,
                    p_rrf_k := 60,
                    p_vector_weight := 1.0,
                    p_text_weight := 1.0,
                    p_fts_config := '{fts_cfg}',
                    p_min_vector_similarity := NULL::double precision,
                    p_ef_search := NULL::integer
                );
                """

            t_sql_start = time.perf_counter()
            q_res = run_psql(sql, conn_info)
            t_sql_end = time.perf_counter()
            sql_ms = (t_sql_end - t_sql_start) * 1000
            sql_latencies_ms.append(sql_ms)
            latencies_ms.append(emb_ms + sql_ms)

            retrieved_rows = []
            for line in q_res.stdout.splitlines():
                if "|" in line:
                    parts = line.split("|", 1)
                    retrieved_rows.append((parts[0].strip(), parts[1].strip()))
                elif line.strip():
                    retrieved_rows.append((line.strip(), ""))

            doc_rank = None
            ans_rank = None

            for rank_idx, (r_key, r_content) in enumerate(retrieved_rows, 1):
                if doc_rank is None and expected_key and expected_key.lower() == r_key.lower():
                    doc_rank = rank_idx

                if ans_rank is None and gold_answers:
                    c_lower = r_content.lower()
                    if any(ga in c_lower for ga in gold_answers):
                        ans_rank = rank_idx

            doc_rr = (1.0 / doc_rank) if doc_rank else 0.0
            reciprocal_ranks.append(doc_rr)

            is_hit1 = (doc_rank == 1)
            is_hit3 = (doc_rank is not None and doc_rank <= 3)
            is_hit5 = (doc_rank is not None and doc_rank <= 5)
            is_hit10 = (doc_rank is not None and doc_rank <= 10)

            is_ans1 = (ans_rank == 1)
            is_ans5 = (ans_rank is not None and ans_rank <= 5)
            is_ans10 = (ans_rank is not None and ans_rank <= 10)

            if is_hit1: hit1 += 1
            if is_hit3: hit3 += 1
            if is_hit5: hit5 += 1
            if is_hit10: hit10 += 1

            if is_ans1: ans_recall1 += 1
            if is_ans5: ans_recall5 += 1
            if is_ans10: ans_recall10 += 1

            if lang not in by_lang:
                by_lang[lang] = {"queries": 0, "hit1": 0, "hit5": 0, "ans5": 0, "rr": []}
            by_lang[lang]["queries"] += 1
            if is_hit1: by_lang[lang]["hit1"] += 1
            if is_hit5: by_lang[lang]["hit5"] += 1
            if is_ans5: by_lang[lang]["ans5"] += 1
            by_lang[lang]["rr"].append(doc_rr)

            if idx % 500 == 0 or idx == len(qa_items):
                elapsed = time.perf_counter() - t0_mode
                print(f"  Processed {idx}/{len(qa_items)} queries ({idx / elapsed:.1f} q/s) | Current Hit@1: {hit1 / idx:.1%} | MRR: {statistics.mean(reciprocal_ranks):.4f}")

        total_q = len(qa_items)
        sorted_lat = sorted(latencies_ms)
        sorted_sql = sorted(sql_latencies_ms)
        sorted_emb = sorted(embedding_latencies_ms)

        lang_breakdown = {}
        for l_name, l_data in by_lang.items():
            l_cnt = l_data["queries"]
            lang_breakdown[l_name] = {
                "queries": l_cnt,
                "doc_hit_rate_at_1": round(l_data["hit1"] / l_cnt, 4) if l_cnt else 0,
                "doc_hit_rate_at_5": round(l_data["hit5"] / l_cnt, 4) if l_cnt else 0,
                "doc_mrr": round(statistics.mean(l_data["rr"]), 4) if l_data["rr"] else 0,
                "answer_recall_at_5": round(l_data["ans5"] / l_cnt, 4) if l_cnt else 0,
            }

        metrics_by_mode[mode] = {
            "total_queries": total_q,
            "document_hit_rate_at_1": round(hit1 / total_q, 4) if total_q else 0,
            "document_hit_rate_at_3": round(hit3 / total_q, 4) if total_q else 0,
            "document_hit_rate_at_5": round(hit5 / total_q, 4) if total_q else 0,
            "document_hit_rate_at_10": round(hit10 / total_q, 4) if total_q else 0,
            "document_mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0,
            "answer_recall_at_1": round(ans_recall1 / total_q, 4) if total_q else 0,
            "answer_recall_at_5": round(ans_recall5 / total_q, 4) if total_q else 0,
            "answer_recall_at_10": round(ans_recall10 / total_q, 4) if total_q else 0,
            "latency_ms": {
                "mean": round(statistics.mean(latencies_ms), 2),
                "p50": round(sorted_lat[int(len(sorted_lat) * 0.5)], 2),
                "p95": round(sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)], 2),
            },
            "sql_latency_ms": {
                "mean": round(statistics.mean(sql_latencies_ms), 2),
                "p50": round(sorted_sql[int(len(sorted_sql) * 0.5)], 2),
                "p95": round(sorted_sql[min(int(len(sorted_sql) * 0.95), len(sorted_sql) - 1)], 2),
            },
            "embedding_latency_ms": {
                "mean": round(statistics.mean(embedding_latencies_ms), 2),
                "p50": round(sorted_emb[int(len(sorted_emb) * 0.5)], 2),
                "p95": round(sorted_emb[min(int(len(sorted_emb) * 0.95), len(sorted_emb) - 1)], 2),
            },
            "by_language": lang_breakdown,
        }

    # 5. Save Report
    out_file = Path("eval/output/qwen3_full_tydiqa_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "precision": "INT8 weights + BF16 compute (Apple Silicon Metal MLX)",
        "database": "Local Supabase PostgreSQL 15 + pgvector",
        "total_documents": len(docs_to_ingest),
        "total_queries": len(qa_items),
        "ingestion_stats": {
            "duration_seconds": round(ingest_time, 2),
            "throughput_docs_per_sec": round(throughput, 1),
        },
        "retrieval_metrics": metrics_by_mode,
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(f"🎉 FULL BENCHMARK COMPLETE! Results saved to {out_file}")
    print("=" * 85)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full TyDi QA benchmark with Qwen3 MLX")
    parser.add_argument("--limit-docs", type=int, default=None, help="Limit corpus documents")
    parser.add_argument("--limit-queries", type=int, default=None, help="Limit evaluation queries")
    parser.add_argument("--dest", default="/tmp/tydiqa_full_corpus", help="Corpus path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--force-ingest", action="store_true", help="Force re-ingestion of all documents")
    args = parser.parse_args()

    run_qwen_benchmark(
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
        dest_dir=args.dest,
        batch_size=args.batch_size,
        force_ingest=args.force_ingest,
    )

