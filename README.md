# Supabase Easy RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PostgreSQL 16+](https://img.shields.io/badge/PostgreSQL-16%2B%20%2F%2017-336791.svg)](https://supabase.com)
[![pgvector HNSW](https://img.shields.io/badge/pgvector-HNSW-green.svg)](https://github.com/pgvector/pgvector)
[![CI Postgres Live](https://img.shields.io/badge/CI-Postgres%2016%20%2B%20pgvector-success.svg)](.github/workflows/ci.yml)
[![Cloud Supabase Validated](https://img.shields.io/badge/Cloud%20Supabase-Live%20Verified-brightgreen.svg)](CLOUD_EVAL_REPORT.md)

A production-ready, ultra-fast Hybrid RAG engine built natively on PostgreSQL & Supabase. Combines dense vector search (HNSW) and sparse BM25 full-text search with Reciprocal Rank Fusion (RRF), parent-context expansion, and database-native multi-tenant RLS isolation.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install supabase-easy-rag
```

### 2. Apply Database Migrations

#### Option A: Via Supabase CLI (Recommended)
```bash
supabase db push
```

#### Option B: Via SQL Editor or CLI Export
```bash
easy-rag init-sql --output ./migrations --dimensions 1536
```
Apply `01_schema.sql` and `02_functions.sql` in the Supabase SQL Editor.

### 3. Configure Environment Variables (`.env`)

```env
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
SUPABASE_ANON_KEY="your-anon-key"

# Embedding Provider (OpenAI or Azure OpenAI)
OPENAI_API_KEY="sk-..."
```

### 4. 30-Second Python Example

```python
from supabase_easy_rag import EasyRagClient

# Pass the end-user JWT for automatic RLS tenant isolation
client = EasyRagClient(user_jwt="eyJhbGci...")

# 1. Sync markdown knowledge base in parallel (SHA-256 incremental)
client.sync_directory("./docs", max_workers=8)

# 2. Search with Hybrid RRF & Parent-Context Expansion
results = client.search_hybrid(
    query="PostgreSQL connection pooling guidelines",
    match_count=5,
    candidate_count=50,
    rrf_k=60,
    expand_context="section",  # "section" or "document"
)

for item in results:
    print(f"[{item.final_score:.4f}] {item.document_title} > {item.section_title}")
    print(item.effective_text[:200] + "...\n")
```

---

## 📊 Evaluation & Latency Benchmarks

Empirical evaluation against the **Google Research TyDi QA** multilingual gold-standard benchmark (5,077 natural queries, 4,488 authentic Wikipedia articles across 11 typologically diverse languages) and direct execution on PostgreSQL 16 + pgvector:

### 1. Retrieval Accuracy & Quality (TyDi QA Global Benchmark)

| Benchmark Metric | Score | Description |
| :--- | :---: | :--- |
| **Document Hit Rate @ 1 (Top-1)** | **88.46%** | Ground-truth document ranked #1 (Strict Document Match) |
| **Document Hit Rate @ 3 (Top-3)** | **91.61%** | Ground-truth document in Top-3 chunks |
| **Document Hit Rate @ 5 (Top-5)** | **92.34%** | Ground-truth document in Top-5 chunks |
| **Document Hit Rate @ 10 (Top-10)** | **92.75%** | Ground-truth document in Top-10 chunks |
| **Document MRR (Mean Reciprocal Rank)** | **0.9012** | Average reciprocal rank on document retrieval |
| **Answer Span Recall @ 5** | **91.90%** | Exact fact-answer span contained in Top-5 chunks |

---

### 2. Execution Latency & Ingestion Throughput

| Component | Metric | Value | Details |
| :--- | :--- | :---: | :--- |
| **Hybrid SQL Query Latency** | Mean | **21.34 ms** | Single round-trip SQL (pgvector HNSW + FTS GIN + RRF fusion) |
| | p50 | **21.42 ms** | Zero client-side post-processing |
| | p95 | **22.09 ms** | Predictable tail latency on PostgreSQL 16 |
| **Ingestion & Verification** | Throughput | **651.5 docs/sec** | 4,488 documents verified via SHA-256 in **6.89s** (16 workers) |


---

### 3. Multilingual Breakdown Across 11 Languages

| Language | Queries | Doc Hit @ 1 | Doc Hit @ 5 | Doc MRR | Answer Recall @ 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Russian** | 828 | **91.5%** | **93.8%** | **0.926** | **93.2%** |
| **Finnish** | 920 | **90.0%** | **93.0%** | **0.913** | **92.9%** |
| **Arabic** | 964 | **89.2%** | **92.8%** | **0.908** | **92.0%** |
| **Telugu** | 668 | **89.1%** | **92.8%** | **0.906** | **93.1%** |
| **Indonesian** | 499 | **89.4%** | **91.8%** | **0.903** | **91.6%** |
| **Swahili** | 475 | **87.4%** | **92.6%** | **0.895** | **91.6%** |
| **English** | 278 | **87.1%** | **89.9%** | **0.883** | **91.0%** |
| **Japanese** | 139 | **82.0%** | **89.2%** | **0.854** | **88.5%** |
| **Bengali** | 111 | **76.6%** | **91.9%** | **0.834** | **90.1%** |
| **Korean** | 194 | **75.8%** | **85.0%** | **0.795** | **83.0%** |

---

## ⚙️ Engineering & Security Highlights

- ☁️ **Live Cloud Supabase Tested**: Validated against live Supabase Cloud (PostgreSQL 17.6 in `eu-north-1`). See [CLOUD_EVAL_REPORT.md](CLOUD_EVAL_REPORT.md).
- 🐘 **PostgreSQL 16 + pgvector CI**: GitHub Actions integration tests against live `pgvector/pgvector:pg16` containers with real HNSW index scans, GIN full-text index scans, and database triggers.
- 🔒 **Strict RLS Tenant Isolation**: Zero cross-tenant data leakage across vector, FTS, and hybrid retrieval with PostgreSQL `auth.uid()`.
- 🎯 **Filtered ANN Recall Preservation**: Candidate oversampling (`candidate_count`) and iterative scan (`hnsw.iterative_scan = 'relaxed_order'`) prevent candidate starvation under metadata/tenant filters.
- 📖 **Parent-Context Expansion**: Searches granular 400-token chunks for precision, returns full parent sections for LLM context.
- 🌍 **Multilingual FTS & Fallbacks**: Native dictionary stemming and fallback support across 11 languages (English, Russian, Arabic, Finnish, and more).

---

## 💻 Python SDK Usage

### 1. Hybrid Search with Parent-Context Expansion

```python
from supabase_easy_rag import EasyRagClient

# Pass the end-user JWT for automatic RLS tenant isolation
client = EasyRagClient(user_jwt="eyJhbGci...")

# Search by dense chunk, expand to full parent section for LLM context
results = client.search_hybrid(
    query="PostgreSQL connection pooling guidelines",
    match_count=5,
    candidate_count=50,
    rrf_k=60,
    expand_context="section",  # "section" or "document"
)

for item in results:
    print(f"[{item.final_score:.4f}] {item.document_title} > {item.section_title}")
    print(item.effective_text[:200] + "...")
```

### 2. High-Throughput Parallel Directory Ingestion

```python
# Sync a directory of Markdown documents with 8 parallel workers
sync_stats = client.sync_directory(
    directory_path="./knowledgebase_docs",
    batch_size=30,
    max_workers=8,
    enable_chunking=True,
    chunk_size=800,
    chunk_overlap=100,
)

print(f"Synced {sync_stats['total_files']} files in parallel.")
```

### 3. Asynchronous Client (FastAPI / Async Runtimes)

```python
from supabase_easy_rag import AsyncEasyRagClient

async_client = AsyncEasyRagClient()

results = await async_client.search_hybrid(
    query="Distributed consensus election timeouts",
    match_count=3,
    expand_context="section",
)
```

---

## 🧪 Local Docker & CI Testing

Run the full integration test suite against a local PostgreSQL container with pgvector:

```bash
# 1. Start local Postgres container with pgvector and run all 59 tests:
./scripts/run_local_postgres_tests.sh

# Or run via docker-compose manually:
docker compose -f docker-compose.test.yml up -d
POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/postgres" python -m unittest discover tests -v

# Run local Postgres retrieval evaluation:
POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/postgres" python eval/local_postgres_eval.py
```

---

## 🛠️ CLI Commands

```bash
# Export migration files
easy-rag init-sql --dimensions 1536 --output ./migrations

# Sync directory of documents
easy-rag sync ./docs --workers 8

# Execute test query with diagnostics
easy-rag query "What is distributed erasure coding?" --mode hybrid --count 5

# Manage backend API access tokens
easy-rag create-token "Production Ingestion Worker"
easy-rag list-tokens
```

---

## ⚙️ Architecture & Implementation Details

- **Two-Stage Hybrid Fusion**: Independent candidate pools are retrieved via indexed scans (`HNSW` for vector distance, `GIN` for full-text match) and fused using Reciprocal Rank Fusion:
  $$RRF\_Score = \frac{w_v}{k + rank_v} + \frac{w_t}{k + rank_t}$$
  This avoids arbitrary score normalization issues between cosine similarities and BM25/FTS weights.
- **Iterative Graph Traversal**: Employs `hnsw.iterative_scan = 'relaxed_order'` and dynamic `p_ef_search` to prevent candidate starvation when applying strict metadata or tenant filters.
- **Weighted Lexical Search**: Uses PostgreSQL native `setweight()` indexing where document titles receive Weight `A` ($1.0$), section headings receive Weight `B` ($0.4$), and chunk text receives Weight `D` ($0.1$).
- **Multi-Tenant Security**: Enforces database-level isolation via Supabase Auth (`auth.uid()`) using non-recursive Row-Level Security (RLS) policies on documents and chunks.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
