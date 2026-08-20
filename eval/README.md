# Supabase Easy RAG — Evaluation & Benchmark Suite

This directory contains the standardized evaluation and benchmarking framework for `supabase-easy-rag`.

---

## 🚀 Quick Evaluation Run

To run the complete multilingual benchmark against your connected Supabase PostgreSQL instance:

```bash
uv run python eval/benchmark.py --workers 8
```

This will:
1. Load / fetch authentic validation articles and human questions from the **Google Research TyDi QA** dataset.
2. Synchronize documents in parallel using `EasyRagClient.sync_directory()`.
3. Evaluate hybrid retrieval accuracy across all 11 languages (Arabic, Russian, Finnish, Telugu, Indonesian, Swahili, English, Korean, Japanese, Bengali, Thai).
4. Compute **Hit Rate @ 1, 3, 5, 10**, **Mean Reciprocal Rank (MRR)**, and **Answer Span Recall @ 5**.
5. Save markdown and JSON reports to `eval/output/`.

---

## 📊 CLI Options

```bash
# Run benchmark on custom corpus and dataset
uv run python eval/benchmark.py --corpus-dir ./my_docs --dataset ./my_dataset.json --workers 8

# Specify custom output directory
uv run python eval/benchmark.py --output ./eval/custom_output
```

---

## 📁 Directory Structure

```
eval/
├── README.md                 # This documentation
├── benchmark.py              # Unified evaluation and benchmarking CLI
├── corpora/
│   └── fetch_tydiqa.py       # Google Research TyDi QA dataset fetcher
└── output/
    ├── BENCHMARK_REPORT.md   # Generated Markdown report
    └── benchmark_report.json # Detailed evaluation metrics JSON
```
