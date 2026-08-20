# Supabase Easy RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PostgreSQL 17+](https://img.shields.io/badge/PostgreSQL-17%2B-336791.svg)](https://supabase.com)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-green.svg)](https://github.com/pgvector/pgvector)

A modular, lightweight Hybrid RAG engine built natively on PostgreSQL & Supabase.

## Why Supabase Easy RAG?

- 🎯 **High-Accuracy Hybrid Search**: Combines semantic vector search with keyword matching — catches both conceptual questions and exact IDs/terms.
- 📖 **Parent-Context Expansion**: Searches precise 400-token chunks for 90%+ Top-1 accuracy, but feeds the full 3,000-token parent section to your LLM.
- 🔒 **Native Multi-Tenant Security**: Out-of-the-box Row-Level Security (RLS) via Supabase Auth (`auth.uid()`) — users only see documents they own or are shared with them.
- ⚡ **High-Speed Parallel Ingestion**: Syncs whole folders of Markdown docs in seconds (700+ docs/sec) with automatic change detection (SHA-256).
- 🌍 **Battle-Tested Multilingual**: Evaluated across 11 languages with automatic text search dictionary fallbacks.
- 🔌 **Zero Framework Overhead**: Clean Python SDK and pure PostgreSQL RPCs. No heavy dependencies.

---



## 📊 Comprehensive Multilingual Benchmark (TyDi QA)

Evaluated against the complete **Google Research TyDi QA** gold-standard validation corpus (4,488 authentic Wikipedia articles, 5,077 real human questions across 11 typologically diverse languages).

### 1. Global Information Retrieval (IR) Quality Metrics

| Benchmark Metric | Score | Description |
| :--- | :---: | :--- |
| **Hit Rate @ 1 (Top-1 Accuracy)** | **89.64%** | Ground-truth relevant document is ranked #1 in 89.6% of queries |
| **Hit Rate @ 3 (Top-3 Accuracy)** | **92.71%** | Top-3 retrieval recall |
| **Hit Rate @ 5 (Top-5 Accuracy)** | **93.52%** | Standard LLM context window budget accuracy |
| **Hit Rate @ 10 (Top-10 Accuracy)** | **93.91%** | Broad candidate recall |
| **MRR (Mean Reciprocal Rank)** | **0.9128** | Average reciprocal rank across all 5,077 queries |
| **Answer Span Recall @ 5** | **92.77%** | Exact fact-answer span is contained in the top-5 retrieved chunks |
| **Parallel Ingestion Throughput** | **732.2 docs/sec** | 4,488 documents with 1536-dim embeddings synchronized in **6.1 seconds** (8 workers) |

---

### 2. Multilingual Breakdown Across 11 Languages

| Language | Language Family | Evaluated Queries | Hit Rate @ 1 | Hit Rate @ 5 | MRR | Answer Recall @ 5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabic** | Afroasiatic (Semitic) | 964 | **90.0%** | **93.8%** | **0.916** | **92.8%** |
| **Finnish** | Uralic (Agglutinative) | 920 | **91.6%** | **94.6%** | **0.929** | **94.0%** |
| **Russian** | Indo-European (Slavic) | 828 | **91.8%** | **94.0%** | **0.928** | **93.2%** |
| **Telugu** | Dravidian | 668 | **90.7%** | **94.8%** | **0.924** | **94.6%** |
| **Indonesian** | Austronesian | 499 | **92.2%** | **94.2%** | **0.929** | **93.8%** |
| **Swahili** | Niger-Congo (Bantu) | 475 | **89.5%** | **94.1%** | **0.913** | **92.8%** |
| **English** | Germanic | 278 | **86.7%** | **90.6%** | **0.883** | **90.3%** |
| **Korean** | Koreanic | 194 | **75.3%** | **85.0%** | **0.793** | **83.0%** |
| **Japanese** | Japonic | 139 | **82.7%** | **89.9%** | **0.863** | **89.2%** |
| **Bengali** | Indo-Aryan | 111 | **77.5%** | **92.8%** | **0.843** | **90.1%** |
| **Thai** | Kra-Dai | 1 | **100.0%** | **100.0%** | **1.000** | **100.0%** |


## Quick Start


### 1. Installation

```bash
pip install supabase-easy-rag
```

### 2. Apply Database Migrations to Supabase

Export and run the SQL migrations:

```bash
easy-rag init-sql --output ./migrations --dimensions 1536
```

Apply `01_schema.sql` and `02_functions.sql` to your Supabase project (via Supabase CLI `supabase db query --file ...` or SQL Editor).

### 3. Configure Environment Variables

```env
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"

# Embedding Provider (OpenAI or Azure OpenAI)
OPENAI_API_KEY="sk-..."
# or Azure:
AZURE_OPENAI_API_KEY="your-azure-key"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
```

---

## 💻 Python SDK Usage

### 1. Hybrid Search with Parent-Context Expansion

```python
from supabase_easy_rag import EasyRagClient
from supabase_easy_rag.providers.azure import AzureEmbeddingProvider

# Initialize provider & client
provider = AzureEmbeddingProvider(
    api_key="...",
    endpoint="https://your-resource.openai.azure.com",
    model="text-embedding-3-small",
)
client = EasyRagClient(embedding_provider=provider)

# Search by dense chunk, expand to full parent section for LLM context
results = client.search_hybrid(
    query="When was the Ottoman Empire established?",
    match_count=5,
    candidate_count=50,
    rrf_k=60,
    expand_context="section",  # "section" or "document"
)

for item in results:
    print(f"[{item.final_score:.4f}] {item.document_title} > {item.section_title}")
    # Effective text contains the full parent section text
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

### 3. Asynchronous Client (FastAPI / High-Concurrency)

```python
from supabase_easy_rag import AsyncEasyRagClient

async_client = AsyncEasyRagClient(embedding_provider=provider)

results = await async_client.search_hybrid(
    query="PostgreSQL connection pooling guidelines",
    match_count=3,
    expand_context="section",
)
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

## ⚙️ Architecture & Implementation Details

- **Two-Stage Hybrid Fusion**: Independent candidate pools are retrieved via indexed scans (`HNSW` for vector distance, `GIN` for full-text match) and fused using Reciprocal Rank Fusion:
  $$RRF\_Score = \frac{w_v}{k + rank_v} + \frac{w_t}{k + rank_t}$$
  This avoids arbitrary score normalization issues between cosine similarities and BM25/FTS weights.
- **Weighted Lexical Search**: Uses PostgreSQL native `setweight()` indexing where document titles receive Weight `A` ($1.0$), section headings receive Weight `B` ($0.4$), and chunk text receives Weight `D` ($0.1$).
- **Candidate Pre-filtering**: Full-text candidate scans limit initial matches to Top-200 before applying `ts_rank` to minimize CPU cycles on frequent keywords.
- **Multi-Tenant Security**: Enforces database-level isolation via Supabase Auth (`auth.uid()`) using Row-Level Security (RLS) policies on documents and chunks.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

