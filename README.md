# Supabase Easy RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

**Supabase Easy RAG** is a production-ready, lightweight Hybrid RAG (Retrieval-Augmented Generation) engine for Supabase PostgreSQL.

It brings enterprise-grade **Hybrid Search (Weighted Vector + Full-Text Search + Title Boosting)**, **Hierarchical Metadata & Path Facets**, **Incremental Markdown Document Syncing**, and **Token Access Security with Audit Trails** to your Supabase application without heavy framework overhead.

---

## ⚡ Features

- **Schema Isolation & Security**: Runs in an isolated `knowledgebase` schema with PostgreSQL RLS and SHA-256 token authentication with audit logging.
- **Hybrid Search Engine**: Combines cosine vector similarity (70%), full-text search (`tsvector` / `websearch_to_tsquery`, 30%), and exact/partial title matching boosts.
- **Hierarchical Markdown Ingestion**: Automatically extracts document titles, section structures (H2–H6), metadata headers, and builds directory/attribute facets.
- **Incremental Sync**: Uses SHA-256 checksums to sync only new or modified documents, saving embedding API costs.
- **Graceful Fallbacks**: If embedding providers fail, search automatically falls back to full-text search without breaking runtime applications.
- **Pluggable Embedding Providers**: Supports OpenAI, Azure OpenAI, and custom callbacks.
- **CLI & Python SDK**: Easy CLI (`easy-rag`) for SQL migrations, syncing, querying, and managing access tokens.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install supabase-easy-rag
```

### 2. Apply Database Migrations to Supabase

Export the SQL migration files using the CLI:

```bash
easy-rag init-sql --output ./supabase_migrations
```

Run `01_schema.sql` and `02_functions.sql` inside your Supabase SQL Editor.

### 3. Set Environment Variables

Create a `.env` file:

```env
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
KNOWLEDGEBASE_ACCESS_TOKEN="kb_live_your_generated_access_token"
OPENAI_API_KEY="sk-..."
```

### 4. Create an Access Token

```bash
easy-rag create-token "My Production Agent Token"
```

---

## 💻 Python Usage

```python
from supabase_easy_rag import EasyRagClient

client = EasyRagClient()

# 1. Sync a directory of Markdown documents
client.sync_directory("./docs")

# 2. Perform Hybrid Search
results = client.search_hybrid(
    query="How do I configure vector indexes?",
    match_count=5,
)

for item in results:
    print(f"[{item.hybrid_score:.4f}] {item.document_title} > {item.section_title or 'Main'}")
    print(item.chunk_text)
    print("-" * 40)
```

---

## 🛠️ CLI Reference

- `easy-rag init-sql` — Generate Supabase SQL migration files.
- `easy-rag sync <directory>` — Sync markdown files to Supabase.
- `easy-rag query "<text>"` — Run hybrid, vector, or FTS search in terminal.
- `easy-rag create-token "<name>"` — Create a new RAG access token.
- `easy-rag list-tokens` — List all registered access tokens.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
