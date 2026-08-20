"""Comprehensive Information Retrieval Evaluation & Benchmark Suite.

Evaluates Supabase Easy RAG against gold-standard datasets (e.g. Google Research TyDi QA):
- Ingestion throughput & verification
- Hybrid Two-Stage RRF retrieval accuracy (Hit Rate @ 1, 3, 5, 10, MRR)
- Fact/Answer Span Recall @ 5
- Multilingual per-language breakdown matrix
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
) -> Dict[str, Any]:
    print("=" * 85)
    print("  🚀 SUPABASE EASY RAG — COMPREHENSIVE BENCHMARK RUNNER")
    print("=" * 85)

    cfg = EasyRagConfig.from_env()
    print(f"Database:          {cfg.supabase_url}")
    print(f"Workers:           {workers}")

    if cfg.embedding.endpoint:
        provider = AzureEmbeddingProvider(
            api_key=cfg.embedding.api_key,
            endpoint=cfg.embedding.endpoint,
            model=cfg.embedding.model,
            batch_size=50,
            batch_sleep=0.05,
        )
    else:
        provider = OpenAIEmbeddingProvider(
            api_key=cfg.embedding.api_key,
            model=cfg.embedding.model,
            batch_size=50,
        )

    client = EasyRagClient(embedding_provider=provider)

    # 1. Dataset Discovery / Preparation
    if not corpus_dir or not dataset_file or not corpus_dir.exists() or not dataset_file.exists():
        default_bench_dir = Path("/tmp/tydiqa_full_corpus")
        if (default_bench_dir / "documents").exists() and (default_bench_dir / "tydiqa_full_dataset.json").exists():
            corpus_dir = default_bench_dir / "documents"
            dataset_file = default_bench_dir / "tydiqa_full_dataset.json"
        else:
            files, dataset_file = fetch_tydiqa_corpus(default_bench_dir, limit=500)
            corpus_dir = default_bench_dir / "documents"

    doc_files = sorted(corpus_dir.glob("*.md"))
    qa_items: List[Dict[str, Any]] = json.loads(dataset_file.read_text(encoding="utf-8"))
    print(f"[Corpus]  Loaded {len(doc_files):,} documents from {corpus_dir}")
    print(f"[Dataset] Loaded {len(qa_items):,} evaluation questions from {dataset_file.name}")

    # 2. Ingestion Verification
    print(f"\n[Ingestion] Synchronizing corpus with {workers} parallel workers...")
    t0_sync = time.perf_counter()
    sync_stats = client.sync_directory(
        corpus_dir,
        batch_size=30,
        enable_chunking=False,
        max_workers=workers,
    )
    sync_time = time.perf_counter() - t0_sync
    throughput = len(doc_files) / sync_time if sync_time > 0 else 0
    print(f"✓ Ingestion completed in {sync_time:.2f}s ({throughput:.1f} docs/sec, changed: {sync_stats.get('files_changed')})")

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
            )
        except Exception as err:
            return {"id": qa.get("id"), "lang": lang, "rank": None, "hit1": False, "hit5": False, "ans5": False, "rr": 0.0, "err": str(err)}

        rank = None
        ans_in_top5 = False
        for idx, res in enumerate(results, 1):
            d_title = res.document_title or ""
            c_txt = res.chunk_text or ""

            is_match = (
                (expected_key and expected_key.lower() in d_title.lower())
                or (expected_title and expected_title.lower() in d_title.lower())
                or (gold_answers and any(ga in c_txt.lower() for ga in gold_answers))
            )
            if is_match and rank is None:
                rank = idx

            if idx <= 5 and gold_answers and any(ga in c_txt.lower() for ga in gold_answers):
                ans_in_top5 = True

        return {
            "id": qa.get("id"),
            "lang": lang,
            "rank": rank,
            "hit1": (rank == 1),
            "hit3": (rank is not None and rank <= 3),
            "hit5": (rank is not None and rank <= 5),
            "hit10": (rank is not None and rank <= 10),
            "ans5": ans_in_top5,
            "rr": (1.0 / rank) if rank else 0.0,
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
    all_rr = [r["rr"] for r in eval_results]
    hit1 = sum(1 for r in eval_results if r.get("hit1"))
    hit3 = sum(1 for r in eval_results if r.get("hit3"))
    hit5 = sum(1 for r in eval_results if r.get("hit5"))
    hit10 = sum(1 for r in eval_results if r.get("hit10"))
    ans5 = sum(1 for r in eval_results if r.get("ans5"))

    metrics = {
        "total_queries": total_q,
        "corpus_documents": len(doc_files),
        "ingestion_throughput_docs_sec": round(throughput, 1),
        "hit_rate_at_1": round(hit1 / total_q, 4) if total_q else 0,
        "hit_rate_at_3": round(hit3 / total_q, 4) if total_q else 0,
        "hit_rate_at_5": round(hit5 / total_q, 4) if total_q else 0,
        "hit_rate_at_10": round(hit10 / total_q, 4) if total_q else 0,
        "mrr": round(statistics.mean(all_rr), 4) if all_rr else 0,
        "answer_span_recall_at_5": round(ans5 / total_q, 4) if total_q else 0,
    }

    # Per-Language breakdown
    by_lang: Dict[str, Dict[str, Any]] = {}
    for r in eval_results:
        lang = r["lang"]
        if lang not in by_lang:
            by_lang[lang] = {"total": 0, "hit1": 0, "hit5": 0, "ans5": 0, "rr": []}
        by_lang[lang]["total"] += 1
        if r.get("hit1"):
            by_lang[lang]["hit1"] += 1
        if r.get("hit5"):
            by_lang[lang]["hit5"] += 1
        if r.get("ans5"):
            by_lang[lang]["ans5"] += 1
        by_lang[lang]["rr"].append(r.get("rr", 0))

    lang_matrix = {}
    for lang, s in sorted(by_lang.items(), key=lambda x: x[1]["total"], reverse=True):
        ltotal = s["total"]
        lang_matrix[lang] = {
            "queries": ltotal,
            "hit_rate_at_1": round(s["hit1"] / ltotal, 4) if ltotal else 0,
            "hit_rate_at_5": round(s["hit5"] / ltotal, 4) if ltotal else 0,
            "mrr": round(statistics.mean(s["rr"]), 4) if s["rr"] else 0,
            "answer_span_recall_at_5": round(s["ans5"] / ltotal, 4) if ltotal else 0,
        }

    metrics["by_language"] = lang_matrix

    # 4. Generate Reports
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "benchmark_report.json"
    report_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    md_report = f"""# Supabase Easy RAG — Benchmark & Quality Report

Comprehensive quality evaluation against the **Google Research TyDi QA** multilingual benchmark.

---

## 1. Global IR Quality Metrics

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Hit Rate @ 1 (Top-1 Accuracy)** | **{metrics['hit_rate_at_1']*100:.2f}%** | Relevant document ranked #1 |
| **Hit Rate @ 3 (Top-3 Accuracy)** | **{metrics['hit_rate_at_3']*100:.2f}%** | Top-3 retrieval recall |
| **Hit Rate @ 5 (Top-5 Accuracy)** | **{metrics['hit_rate_at_5']*100:.2f}%** | Top-5 retrieval recall |
| **Hit Rate @ 10 (Top-10 Accuracy)** | **{metrics['hit_rate_at_10']*100:.2f}%** | Top-10 candidate coverage |
| **MRR (Mean Reciprocal Rank)** | **{metrics['mrr']:.4f}** | Average reciprocal rank |
| **Answer Span Recall @ 5** | **{metrics['answer_span_recall_at_5']*100:.2f}%** | Fact answer contained in top-5 chunks |
| **Ingestion Throughput** | **{metrics['ingestion_throughput_docs_sec']:.1f} docs/sec** | Multi-threaded sync ({workers} workers) |

---

## 2. Multilingual Breakdown

| Language | Queries | Hit Rate @ 1 | Hit Rate @ 5 | MRR | Answer Recall @ 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for lang, s in lang_matrix.items():
        md_report += f"| **{lang.capitalize()}** | {s['queries']} | **{s['hit_rate_at_1']*100:.1f}%** | **{s['hit_rate_at_5']*100:.1f}%** | **{s['mrr']:.3f}** | **{s['answer_span_recall_at_5']*100:.1f}%** |\n"

    report_md = output_dir / "BENCHMARK_REPORT.md"
    report_md.write_text(md_report, encoding="utf-8")

    print("\n" + "=" * 85)
    print("  📊 BENCHMARK SUMMARY")
    print("=" * 85)
    print(f" Hit Rate @ 1:           {metrics['hit_rate_at_1']*100:.2f}%")
    print(f" Hit Rate @ 5:           {metrics['hit_rate_at_5']*100:.2f}%")
    print(f" MRR:                    {metrics['mrr']:.4f}")
    print(f" Answer Recall @ 5:      {metrics['answer_span_recall_at_5']*100:.2f}%")
    print(f" Ingestion Throughput:   {metrics['ingestion_throughput_docs_sec']} docs/sec")
    print(f" Reports Saved:          {report_json} & {report_md}")
    print("=" * 85)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Supabase Easy RAG Evaluation Suite")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="Directory with markdown documents")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to evaluation dataset JSON")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", type=Path, default=Path("eval/output"), help="Output directory for reports")
    args = parser.parse_args()

    run_benchmark(
        corpus_dir=args.corpus_dir,
        dataset_file=args.dataset,
        output_dir=args.output,
        workers=args.workers,
    )
