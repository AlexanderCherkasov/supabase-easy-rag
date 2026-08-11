# Eval & Benchmark Framework — Portable RAG Evaluation

Portable `eval` framework for evaluating RAG pipelines and Needle-in-a-Haystack (NIAH) benchmarks. Works completely locally (offline `--mock` mode), in CI/CD pipelines, and against live Supabase PostgreSQL + pgvector databases.

---

## 🪡 Needle-in-a-Haystack (NIAH) Benchmark

The benchmark tests how accurately the hybrid/vector retriever retrieves small isolated facts ("needles") hidden inside larger background documents ("haystack").

### Corpus Structure
Files are located in `eval/data/haystack/`:
- `needles/` — Documents containing specific target facts (e.g. activation keys `PASSPHRASE_STARLIGHT_9842`, deal amounts `$74.35 million`, timeouts `POSTGRES_STATEMENT_TIMEOUT_MS=14200`, dosages `35mg/kg`).
- `distractors/` — Background articles (Kubernetes cloud infrastructure, HR policies, B-Tree indexes, LoRA fine-tuning) introducing semantic noise.

### 1. Running NIAH Benchmark (Offline / Mock)
```bash
uv run python eval/run_niah.py --dataset eval/dataset_niah.json --mock
```

### 2. Running Against Live Supabase Instance
```bash
# 1. Sync haystack corpus files to Supabase
uv run python -m supabase_easy_rag.cli sync eval/data/haystack --public

# 2. Run benchmark in live mode
uv run python eval/run_niah.py --dataset eval/dataset_niah.json --live
```

---

## ⚙️ Synthetic Dataset Generator (Depth Variations)

The script `eval/generate_dataset.py` generates datasets by embedding needles at specific relative document depths (0%, 25%, 50%, 75%, 100%) to test for the *Lost in the Middle* effect:

```bash
# Generate dataset with varied needle depth
uv run python eval/generate_dataset.py

# Run benchmark on synthetic dataset
uv run python eval/run_niah.py --dataset eval/dataset_niah_synthetic.json
```

---

## 📊 Standard Eval Metrics (Hit Rate, MRR, Keyword Recall)

### Measured Metrics:
- **Hit Rate / Recall@k** — Target document or fact presence in top-k results.
- **MRR (Mean Reciprocal Rank)** — \(1 / \text{rank}\) of the first relevant chunk retrieved.
- **Keyword Recall** — Percentage of target keywords present in retrieved chunks.
- **Latency** — Query search execution time in milliseconds.

### Running Standard Evaluation:
```bash
# Mock mode
uv run python eval/evaluate.py --mock --output eval/report.json

# Live Supabase mode
uv run python eval/evaluate.py --k 5 --output eval/report.json
cat eval/report.json
```

---

## 📁 Dataset Schema (`dataset_niah.json`)

Record format:
```json
{
  "id": "niah_01",
  "question": "What is the emergency activation passkey for Project Starlight cluster fra-1-edge?",
  "expected_document_key": "needles/starlight_passkey.md",
  "expected_keywords": ["PASSPHRASE_STARLIGHT_9842", "9443", "fra-1-edge"],
  "needle_fact": "PASSPHRASE_STARLIGHT_9842",
  "mode": "hybrid"
}
```

