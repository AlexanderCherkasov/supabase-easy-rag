from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.corpora.fetcher import fetch_corpus, list_corpora
from eval.test_isolation import isolated_eval_prefix, isolated_test_schema

# Full-cycle eval: bulk web corpus (>100 docs) + isolated test schema/prefix + auto-drop

def run(corpus: str = "synthetic-300", limit: int = 300, k: int = 5, keep: bool = False, use_llm_judge: bool = False):
    from eval.evaluate import evaluate

    dest = Path(f"/tmp/eval_corpus_{corpus}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[corpus] Fetching {corpus} (limit {limit}) -> {dest}")
    files = fetch_corpus(corpus, dest, limit=limit)
    print(f"[corpus] Got {len(files)} markdown files")

    # Use prefix isolation so artifacts self-destruct (or keep if --keep)
    prefix_ctx = isolated_eval_prefix() if not keep else isolated_test_schema(schema=f"knowledgebase_keep_{corpus}")

    with prefix_ctx as prefix:
        print(f"[isolation] Prefix: {prefix} (artifacts will auto-drop)" if not keep else f"[isolation] Keep mode: {prefix}")
        # Ingest bulk corpus under prefix
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            prefixed_root = tmp_root / prefix.strip("/")
            prefixed_root.mkdir(parents=True)
            for f in files:
                shutil.copy2(f, prefixed_root / f.name)
            from supabase_easy_rag import EasyRagClient
            from supabase_easy_rag.config import EasyRagConfig, get_config
            get_config.cache_clear()
            cfg = EasyRagConfig.from_env()
            # Check Supabase reachable and schema exists
            try:
                from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client
                c = create_postgrest_client(cfg.supabase_url, cfg.supabase_service_role_key, schema_name=cfg.schema_name)
                c.schema(cfg.schema_name).table("documents").select("id").limit(1).execute()
                can_ingest = True
            except Exception as e:
                print(f"[warn] Supabase not reachable or schema missing: {e}\n -> Run MIGRATION_GUIDE.md, falling back to mock eval")
                can_ingest = False

            if can_ingest:
                if cfg.embedding.provider == "azure" and cfg.embedding.endpoint:
                    from supabase_easy_rag.providers.azure import AzureEmbeddingProvider

                    emb = AzureEmbeddingProvider(
                        api_key=cfg.embedding.api_key, endpoint=cfg.embedding.endpoint, model=cfg.embedding.model
                    )
                else:
                    from supabase_easy_rag.providers.openai import OpenAIEmbeddingProvider

                    emb = OpenAIEmbeddingProvider(
                        api_key=cfg.embedding.api_key, model=cfg.embedding.model, base_url=cfg.embedding.endpoint
                    )
                client = EasyRagClient(embedding_provider=emb)
                # Ingest bulk
                print(f"[ingest] Syncing {len(files)} docs with chunking {cfg.chunk_size}/{cfg.chunk_overlap} ...")
                res = client.sync_directory(tmp_root)
                print(f"[ingest] Done: {res}")

                def _db_retriever(q, mode, kk):
                    if mode == "vector":
                        return client.search_vector(q, match_count=kk)
                    elif mode == "fts":
                        return client.search_fts(q, match_count=kk)
                    else:
                        return client.search_hybrid(q, match_count=kk)

                retriever = _db_retriever
            else:
                # Mock retriever for CI without DB: returns synthetic hits
                def _mock_retriever(q, mode, kk):
                    return [{"document_key": f"{prefix}needle_001.md", "chunk_text": "NEEDLE-001 secret code", "hybrid_score": 0.9}] * kk

                retriever = _mock_retriever

            # Build a bulk dataset from corpus (first 10 docs as queries)
            dataset_path = dest / "dataset_bulk.json"
            # For synthetic-300, we know needle codes; for web corpora, use keyword queries
            items = []
            for i, f in enumerate(files[:10]):
                txt = f.read_text(encoding="utf-8")[:200]
                items.append({"id": f"bulk_{i}", "question": f"What is in {f.name}?", "expected_document_key": f"{prefix}{f.name}", "expected_keywords": txt.split()[:3], "mode": "hybrid"})
            dataset_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

            report = evaluate(dataset_path=dataset_path, retriever=retriever, k=k, model=cfg.embedding.model, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, use_llm_judge=use_llm_judge)
            out = Path(f"eval/output/report_full_{corpus}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report.__dict__, default=str, indent=2), encoding="utf-8")
            print(f"[eval] Bulk corpus {corpus}: {len(files)} docs, hit_rate={report.hit_rate:.2f} mrr={report.mrr:.2f} avg_lat={report.avg_latency_ms:.0f}ms")
            print(f"[eval] Report -> {out}")
            for r in report.results[:5]:
                print(f"  {'✓' if r.hit else '✗'} {r.id} hit={r.hit} mrr={r.mrr:.2f}")

            if keep:
                print(f"[keep] Artifacts kept under prefix {prefix} (delete manually: document_key ilike '{prefix}%')")
                return report
            # else auto-drop via context manager
            print(f"[cleanup] Prefix {prefix} will be deleted on exit")
            return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full-cycle bulk eval with isolated prefix and auto-cleanup")
    ap.add_argument("--corpus", choices=[c["id"] for c in list_corpora()], default="synthetic-300", help="Web corpus id (see eval/corpora/registry.json)")
    ap.add_argument("--limit", type=int, default=300, help="Number of docs to fetch (bulk, 100-1000)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--keep", action="store_true", help="Keep artifacts (don't auto-drop)")
    ap.add_argument("--llm-judge", action="store_true", help="Use 5.4-nano as judge")
    ap.add_argument("--list", action="store_true", help="List available corpora")
    args = ap.parse_args()
    if args.list:
        for c in list_corpora():
            print(f"{c['id']:20s} {c['size']:5d} {c['name']} — {c['source'][:60]}")
    else:
        run(corpus=args.corpus, limit=args.limit, k=args.k, keep=args.keep, use_llm_judge=args.llm_judge)
