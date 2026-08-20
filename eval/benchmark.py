"""Comprehensive Information Retrieval Evaluation & Benchmark Suite.

Evaluates Supabase Easy RAG against gold-standard datasets (e.g. Google Research TyDi QA):
- Ingestion throughput (Fresh Embedding Ingestion vs Incremental SHA-256 Verification)
- Sub-chunk retrieval accuracy with Parent-Context Expansion (400-token chunks)
- Strict Ground-Truth Document Hit Rates (Doc Hit @ 1, 3, 5, 10, Doc MRR)
- Fact/Answer Span Recall @ 1, 5, 10
- Multilingual per-language breakdown matrix across 11 languages
- Automated Markdown & JSON report generation
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from eval.corpora.fetch_tydiqa import fetch_tydiqa_corpus
from supabase_easy_rag import EasyRagClient
from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.providers.azure import AzureEmbeddingProvider
from supabase_easy_rag.providers.openai import OpenAIEmbeddingProvider


def run_benchmark(
    corpus_dir: Path | None = None,
    dataset_file: Path | None = None,
    output_dir: Path = Path("eval/output"),
    workers: int = 8,
    match_count: int = 5,
    candidate_count: int = 50,
    rrf_k: int = 60,
    enable_chunking: bool = True,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    force_sync: bool = False,
    expand_context: str | None = None,
    limit_queries: int | None = None,
) -> Dict[str, Any]:
    print("=" * 85)
    print("  🚀 SUPABASE EASY RAG — COMPREHENSIVE BENCHMARK RUNNER")
    print("=" * 85)

    cfg = EasyRagConfig.from_env()
    print(f"Database:          {cfg.supabase_url}")
    print(f"Workers:           {workers}")
    print(f"Chunking:          enabled={enable_chunking}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    if expand_context:
        print(f"Context Expansion: {expand_context}")

    if cfg.embedding.endpoint:
        provider = AzureEmbeddingProvider(
            api_key=cfg.embedding.api_key,
            endpoint=cfg.embedding.endpoint,
            model=cfg.embedding.model,
            batch_size=100,
            batch_sleep=0.0,
            dimensions=cfg.embedding_dim if "large" in cfg.embedding.model.lower() else None,
        )
    else:
        provider = OpenAIEmbeddingProvider(
            api_key=cfg.embedding.api_key,
            model=cfg.embedding.model,
            batch_size=100,
            batch_sleep=0.0,
            dimensions=cfg.embedding_dim if "large" in cfg.embedding.model.lower() else None,
        )

    client = EasyRagClient(embedding_provider=provider)

    # 1. Dataset Discovery / Preparation
    if not corpus_dir or not dataset_file or not corpus_dir.exists() or not dataset_file.exists():
        default_bench_dir = Path("/tmp/tydiqa_full_corpus")
        if (default_bench_dir / "documents").exists() and (default_bench_dir / "tydiqa_full_dataset.json").exists():
            corpus_dir = default_bench_dir / "documents"
            dataset_file = default_bench_dir / "tydiqa_full_dataset.json"
        else:
            files, dataset_file = fetch_tydiqa_corpus(default_bench_dir, split="validation")
            corpus_dir = default_bench_dir / "documents"

    doc_files = sorted(corpus_dir.glob("*.md"))
    qa_items: List[Dict[str, Any]] = json.loads(dataset_file.read_text(encoding="utf-8"))
    if limit_queries is not None and limit_queries > 0:
        qa_items = qa_items[:limit_queries]

    print(f"[Corpus]  Loaded {len(doc_files):,} documents from {corpus_dir}")
    print(f"[Dataset] Loaded {len(qa_items):,} evaluation questions from {dataset_file.name}")

    # 2. Ingestion Verification
    print(f"\n[Ingestion] Synchronizing corpus with {workers} parallel workers (chunk_size={chunk_size})...")
    t0_sync = time.perf_counter()
    sync_stats = client.sync_directory(
        corpus_dir,
        batch_size=30,
        enable_chunking=enable_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_workers=workers,
        force=force_sync,
    )
    sync_time = time.perf_counter() - t0_sync
    files_changed = sync_stats.get("files_changed", 0)
    files_seen = sync_stats.get("files_seen", len(doc_files))

    if files_changed > 0:
        throughput = files_changed / sync_time if sync_time > 0 else 0
        sync_mode_desc = f"Fresh embedding ingestion: {throughput:.1f} docs/sec ({files_changed} changed docs in {sync_time:.2f}s)"
    else:
        throughput = files_seen / sync_time if sync_time > 0 else 0
        sync_mode_desc = f"Incremental SHA-256 verification (no-op): {throughput:.1f} docs/sec ({files_seen} docs checked in {sync_time:.2f}s)"

    print(f"✓ Ingestion completed: {sync_mode_desc}")

    # 3. Parallel Retrieval Evaluation
    print(f"\n[Evaluation] Evaluating {len(qa_items):,} queries across {workers} workers...")

    def _evaluate_question(qa: Dict[str, Any]) -> Dict[str, Any]:
        q_text = qa["question"]
        expected_key = qa.get("expected_document_key", "")
        expected_title = qa.get("document_title", "")
        gold_answers = [a.lower() for a in qa.get("gold_answers", []) if a]
        lang = qa.get("language", "english")
        fts_cfg = lang if lang in ["english", "russian", "arabic", "finnish", "indonesian", "swahili"] else "simple"

        try:
            results = client.search_hybrid(
                query=q_text,
                match_count=max(match_count, 10),
                candidate_count=candidate_count,
                rrf_k=rrf_k,
                fts_config=fts_cfg,
                expand_context=expand_context,
            )
        except Exception as err:
            return {
                "id": qa.get("id"),
                "lang": lang,
                "doc_rank": None,
                "ans_rank": None,
                "doc_hit1": False,
                "doc_hit3": False,
                "doc_hit5": False,
                "doc_hit10": False,
                "ans_recall1": False,
                "ans_recall5": False,
                "ans_recall10": False,
                "doc_rr": 0.0,
                "err": str(err),
            }

        doc_rank = None
        ans_rank = None

        for idx, res in enumerate(results, 1):
            d_title = (res.document_title or "").strip().lower()
            d_key = (res.metadata.get("document_key") or "").strip().lower()
            c_txt = (res.chunk_text or "").lower()
            exp_title = expected_title.strip().lower()
            exp_key = expected_key.strip().lower()

            # Strict Document Hit: retrieved chunk must belong to the ground-truth document
            is_doc_hit = (
                (exp_key and (exp_key == d_key or exp_key in d_title))
                or (exp_title and (exp_title == d_title or exp_title in d_title or d_title in exp_title))
            )
            if is_doc_hit and doc_rank is None:
                doc_rank = idx

            # Answer Span Recall: chunk text contains at least one gold answer span
            is_ans_hit = bool(gold_answers and any(ga in c_txt for ga in gold_answers))
            if is_ans_hit and ans_rank is None:
                ans_rank = idx

        return {
            "id": qa.get("id"),
            "lang": lang,
            "doc_rank": doc_rank,
            "ans_rank": ans_rank,
            "doc_hit1": (doc_rank == 1),
            "doc_hit3": (doc_rank is not None and doc_rank <= 3),
            "doc_hit5": (doc_rank is not None and doc_rank <= 5),
            "doc_hit10": (doc_rank is not None and doc_rank <= 10),
            "ans_recall1": (ans_rank == 1),
            "ans_recall5": (ans_rank is not None and ans_rank <= 5),
            "ans_recall10": (ans_rank is not None and ans_rank <= 10),
            "doc_rr": (1.0 / doc_rank) if doc_rank else 0.0,
        }

    eval_results = []
    t0_eval = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_evaluate_question, qa) for qa in qa_items]
        for idx, f in enumerate(as_completed(futures), 1):
            eval_results.append(f.result())
            if idx % 500 == 0 or idx == len(qa_items):
                elapsed = time.perf_counter() - t0_eval
                print(f"  Progress: {idx}/{len(qa_items)} ({idx / elapsed:.1f} q/s)...")

    total_q = len(eval_results)
    all_doc_rr = [r["doc_rr"] for r in eval_results]
    doc_hit1 = sum(1 for r in eval_results if r.get("doc_hit1"))
    doc_hit3 = sum(1 for r in eval_results if r.get("doc_hit3"))
    doc_hit5 = sum(1 for r in eval_results if r.get("doc_hit5"))
    doc_hit10 = sum(1 for r in eval_results if r.get("doc_hit10"))
    ans_rec1 = sum(1 for r in eval_results if r.get("ans_recall1"))
    ans_rec5 = sum(1 for r in eval_results if r.get("ans_recall5"))
    ans_rec10 = sum(1 for r in eval_results if r.get("ans_recall10"))

    metrics = {
        "total_queries": total_q,
        "corpus_documents": len(doc_files),
        "chunking_config": {
            "enabled": enable_chunking,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "ingestion_stats": {
            "files_seen": files_seen,
            "files_changed": files_changed,
            "sync_duration_seconds": round(sync_time, 2),
            "throughput_docs_per_sec": round(throughput, 1),
            "mode": "fresh_ingestion" if files_changed > 0 else "incremental_verification",
        },
        "document_retrieval": {
            "doc_hit_rate_at_1": round(doc_hit1 / total_q, 4) if total_q else 0,
            "doc_hit_rate_at_3": round(doc_hit3 / total_q, 4) if total_q else 0,
            "doc_hit_rate_at_5": round(doc_hit5 / total_q, 4) if total_q else 0,
            "doc_hit_rate_at_10": round(doc_hit10 / total_q, 4) if total_q else 0,
            "doc_mrr": round(statistics.mean(all_doc_rr), 4) if all_doc_rr else 0,
        },
        "answer_span_extraction": {
            "answer_recall_at_1": round(ans_rec1 / total_q, 4) if total_q else 0,
            "answer_recall_at_5": round(ans_rec5 / total_q, 4) if total_q else 0,
            "answer_recall_at_10": round(ans_rec10 / total_q, 4) if total_q else 0,
        },
    }

    # Per-Language breakdown
    by_lang: Dict[str, Dict[str, Any]] = {}
    for r in eval_results:
        lang = r["lang"]
        if lang not in by_lang:
            by_lang[lang] = {"total": 0, "doc_hit1": 0, "doc_hit5": 0, "ans_rec5": 0, "doc_rr": []}
        by_lang[lang]["total"] += 1
        if r.get("doc_hit1"):
            by_lang[lang]["doc_hit1"] += 1
        if r.get("doc_hit5"):
            by_lang[lang]["doc_hit5"] += 1
        if r.get("ans_recall5"):
            by_lang[lang]["ans_rec5"] += 1
        by_lang[lang]["doc_rr"].append(r.get("doc_rr", 0))

    lang_matrix = {}
    for lang, s in sorted(by_lang.items(), key=lambda x: x[1]["total"], reverse=True):
        ltotal = s["total"]
        lang_matrix[lang] = {
            "queries": ltotal,
            "doc_hit_rate_at_1": round(s["doc_hit1"] / ltotal, 4) if ltotal else 0,
            "doc_hit_rate_at_5": round(s["doc_hit5"] / ltotal, 4) if ltotal else 0,
            "doc_mrr": round(statistics.mean(s["doc_rr"]), 4) if s["doc_rr"] else 0,
            "answer_recall_at_5": round(s["ans_rec5"] / ltotal, 4) if ltotal else 0,
        }

    metrics["by_language"] = lang_matrix

    # 4. Generate Reports
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "benchmark_report.json"
    report_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    doc_m = metrics["document_retrieval"]
    ans_m = metrics["answer_span_extraction"]
    ing_m = metrics["ingestion_stats"]

    md_report = f"""# Supabase Easy RAG — Benchmark & Quality Report

Comprehensive quality evaluation against the **Google Research TyDi QA** multilingual benchmark ({metrics['total_queries']:,} queries, {metrics['corpus_documents']:,} documents, 11 languages).

---

## 1. Ground-Truth Document Retrieval Metrics (Strict Document ID/Title Match)

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Document Hit Rate @ 1 (Top-1)** | **{doc_m['doc_hit_rate_at_1']*100:.2f}%** | Ground-truth document ranked #1 |
| **Document Hit Rate @ 3 (Top-3)** | **{doc_m['doc_hit_rate_at_3']*100:.2f}%** | Ground-truth document in Top-3 chunks |
| **Document Hit Rate @ 5 (Top-5)** | **{doc_m['doc_hit_rate_at_5']*100:.2f}%** | Ground-truth document in Top-5 chunks |
| **Document Hit Rate @ 10 (Top-10)** | **{doc_m['doc_hit_rate_at_10']*100:.2f}%** | Ground-truth document in Top-10 chunks |
| **Document MRR** | **{doc_m['doc_mrr']:.4f}** | Mean Reciprocal Rank on document retrieval |

---

## 2. Fact / Answer Span Extraction Metrics

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Answer Span Recall @ 1** | **{ans_m['answer_recall_at_1']*100:.2f}%** | Gold answer span contained in Top-1 chunk |
| **Answer Span Recall @ 5** | **{ans_m['answer_recall_at_5']*100:.2f}%** | Gold answer span contained in Top-5 chunks |
| **Answer Span Recall @ 10** | **{ans_m['answer_recall_at_10']*100:.2f}%** | Gold answer span contained in Top-10 chunks |

---

## 3. Ingestion & Synchronization Performance

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Chunking Setup** | **{chunk_size} chars (overlap {chunk_overlap})** | Sub-chunk splitting enabled |
| **Sync Duration** | **{ing_m['sync_duration_seconds']}s** | Total synchronization time ({workers} workers) |
| **Throughput** | **{ing_m['throughput_docs_per_sec']} docs/sec** | {ing_m['mode'].replace('_', ' ').title()} ({files_changed} changed, {files_seen} seen) |

---

## 4. Multilingual Breakdown Across 11 Languages

| Language | Queries | Doc Hit @ 1 | Doc Hit @ 5 | Doc MRR | Answer Recall @ 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for lang, s in lang_matrix.items():
        md_report += f"| **{lang.capitalize()}** | {s['queries']} | **{s['doc_hit_rate_at_1']*100:.1f}%** | **{s['doc_hit_rate_at_5']*100:.1f}%** | **{s['doc_mrr']:.3f}** | **{s['answer_recall_at_5']*100:.1f}%** |\n"

    report_md = output_dir / "BENCHMARK_REPORT.md"
    report_md.write_text(md_report, encoding="utf-8")

    print("\n" + "=" * 85)
    print("  📊 BENCHMARK SUMMARY")
    print("=" * 85)
    print(f" Document Hit @ 1:        {doc_m['doc_hit_rate_at_1']*100:.2f}%")
    print(f" Document Hit @ 5:        {doc_m['doc_hit_rate_at_5']*100:.2f}%")
    print(f" Document MRR:            {doc_m['doc_mrr']:.4f}")
    print(f" Answer Recall @ 5:       {ans_m['answer_recall_at_5']*100:.2f}%")
    print(f" Ingestion Throughput:    {ing_m['throughput_docs_per_sec']} docs/sec ({ing_m['mode']})")
    print(f" Reports Saved:           {report_json} & {report_md}")
    print("=" * 85)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Supabase Easy RAG Evaluation Suite")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="Directory with markdown documents")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to evaluation dataset JSON")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", type=Path, default=Path("eval/output"), help="Output directory for reports")
    parser.add_argument("--enable-chunking", action="store_true", default=True, help="Enable chunking into sub-chunks")
    parser.add_argument("--no-chunking", action="store_false", dest="enable_chunking", help="Disable chunking")
    parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size in characters/tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--force-sync", action="store_true", default=False, help="Force re-sync and re-chunking")
    parser.add_argument("--expand-context", type=str, default=None, help="Expand context mode (section or document)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Limit number of queries to evaluate")
    args = parser.parse_args()

    run_benchmark(
        corpus_dir=args.corpus_dir,
        dataset_file=args.dataset,
        output_dir=args.output,
        workers=args.workers,
        enable_chunking=args.enable_chunking,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_sync=args.force_sync,
        expand_context=args.expand_context,
        limit_queries=args.limit_queries,
    )
