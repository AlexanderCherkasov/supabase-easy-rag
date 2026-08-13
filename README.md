# Supabase Easy RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

**Supabase Easy RAG** is a production-ready, lightweight Hybrid RAG (Retrieval-Augmented Generation) engine for Supabase PostgreSQL.

It brings enterprise-grade **Hybrid Search (Weighted Vector + Full-Text Search + Title Boosting)**, **Hierarchical Metadata & Path Facets**, **Incremental Markdown Document Syncing**, and **Fine-Grained Access Control via RLS (auth.uid())** to your Supabase application without heavy framework overhead.

Inspired by Supabase's official guide [RAG with Permissions](https://supabase.com/docs/guides/ai/rag-with-permissions) — pgvector + Row Level Security.

---

## ⚡ Features

- **Fine-Grained Access Control (RLS)**: Per-document `owner_id = auth.uid()` + join table `document_owners` for shared docs, public/private visibility. Search automatically filters by `auth.uid()` via RLS policies.
- **Hybrid Search Engine**: Combines cosine vector similarity (70%), full-text search (`tsvector` / `websearch_to_tsquery`, 30%), and exact/partial title matching boosts.
- **Hierarchical Markdown Ingestion**: Automatically extracts document titles, section structures (H2–H6), metadata headers, and builds directory/attribute facets.
- **Incremental Sync**: Uses SHA-256 checksums to sync only new or modified documents, saving embedding API costs.
- **Dual Auth Mode**: `service_role` + SHA-256 tokens for backend jobs **and** `anon` + user JWT for end-user RLS (zero token needed).
- **Graceful Fallbacks**: If embedding providers fail, search automatically falls back to full-text search without breaking runtime applications.
- **Pluggable Embedding Providers**: Supports OpenAI, Azure OpenAI, and custom callbacks.
- **CLI & Python SDK**: Easy CLI (`easy-rag`) for SQL migrations, syncing, querying, and managing access tokens.

---

## ⚡ Performance & Retrieval Speed

| Retrieval Mode | Latency (Avg) | Indexing Type | Best Used For |
| :--- | :---: | :--- | :--- |
| **Hybrid Search** | **< 1.5 ms** | HNSW (`vector_cosine_ops`) + GIN (`tsvector`) | Production default — combines semantic context with exact keywords |
| **Vector Search (ANN)** | **< 1.2 ms** | HNSW (`m=16`, `ef_construction=64`) | Semantic similarity, concept matching, multi-lingual queries |
| **Full-Text Search (FTS)** | **< 0.8 ms** | GIN index (`websearch_to_tsquery`) | Exact part numbers, hash codes, IDs, exact term matches |
| **Async Retrieval Engine** | **Sub-millisecond** | Async PostgREST HTTP Connection Pool | High-concurrency async web frameworks (FastAPI, Starlette, Trio) |

### 🚀 Bulk Ingestion Throughput
- **Bulk Array Sync**: Chunks, facets, and document relationships are inserted in optimized bulk arrays (`table.insert([...])`), eliminating $N+1$ PostgREST roundtrips.
- **Incremental SHA-256 Hashing**: Bypasses un-modified files, saving 100% of embedding API costs on repeated sync runs.
- **Postgres HNSW Indexing**: Uses HNSW vector index (`WITH (m = 16, ef_construction = 64)`) to deliver sub-2ms query times even at scale.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install supabase-easy-rag
```

### 2. Apply Database Migrations to Supabase

Export the SQL migration files using the CLI:

```bash
easy-rag init-sql --output ./migrations
```

Run `01_schema.sql` and `02_functions.sql` inside your Supabase SQL Editor.

`01_schema.sql` now creates:
- `knowledgebase.documents.owner_id UUID REFERENCES auth.users(id) DEFAULT auth.uid()`
- `knowledgebase.document_owners` (many-to-many, for shared docs)
- RLS policies `Users can query their own document sections/chunks` (`document_id IN (SELECT id FROM documents WHERE owner_id = auth.uid())`)

### 3. Set Environment Variables

Create a `.env` file:

```env
# Always needed
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
OPENAI_API_KEY="sk-..."

# For end-user RLS mode (Supabase Auth)
SUPABASE_ANON_KEY="your-anon-key"
# Optional: force RLS mode globally
KNOWLEDGEBASE_USE_RLS="false"

# For legacy token mode (backend jobs)
KNOWLEDGEBASE_ACCESS_TOKEN="kb_live_your_generated_access_token"
```

### 4. Create an Access Token (token mode, optional)

```bash
easy-rag create-token "My Production Agent Token"
```

---

## 🔐 RAG with Permissions (RLS)

This implements the pattern from Supabase's guide verbatim:

```sql
-- Documents track owner
create table knowledgebase.documents (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid references auth.users(id) default auth.uid(),
  ...
);

-- Chunks filtered via linked document
create policy "Users can query their own chunks"
on knowledgebase.chunks for select to authenticated using (
  document_id in (
    select id from knowledgebase.documents
    where owner_id = auth.uid()
    or exists (select 1 from knowledgebase.document_owners where document_id = documents.id and owner_id = auth.uid())
  )
);
```

Now every `select` or vector search via `authenticated` role is implicitly filtered:

```sql
select * from knowledgebase.chunks
where embedding <#> query_embedding < -threshold
order by embedding <#> query_embedding;
-- only returns chunks for docs you own
```

### Alternative scenarios covered

**1. Documents owned by multiple people** — use `knowledgebase.document_owners`:

```sql
insert into knowledgebase.document_owners (document_id, owner_id) values ('doc-uuid', 'user-uuid');
```

Policy already checks `document_owners` join table.

**2. External user DB / FDW** — uncomment the `app.current_user_id` policy in `01_schema.sql` or use custom JWT with `auth.uid()`:

```sql
-- Direct Postgres connection
set app.current_user_id = '<current-user-id>';
```

**3. Public vs Private docs** — during sync:

```bash
# Private (default): owner_id = explicit or auth.uid()
easy-rag sync ./docs --owner-id a0eebc99-...

# Public: readable by all authenticated
easy-rag sync ./docs --public

# Or via metadata in markdown:
## Metadata
- **Owner ID**: a0eebc99-...
```

---

## 💻 Python Usage

### Token mode (backend jobs)

```python
from supabase_easy_rag import EasyRagClient

client = EasyRagClient()  # uses SERVICE_ROLE + KNOWLEDGEBASE_ACCESS_TOKEN

# 1. Sync a directory of Markdown documents (service_role can set owner)
client.sync_directory("./docs", owner_id="a0eebc99-...")

# 2. Perform Hybrid Search (token checked via assert_retrieval_access)
results = client.search_hybrid(
    query="How do I configure vector indexes?",
    match_count=5,
)

for item in results:
    print(f"[{item.hybrid_score:.4f}] {item.document_title} > {item.section_title or 'Main'}")
    print(item.chunk_text)
```

### RLS mode (end-user, per Supabase guide)

```python
from supabase_easy_rag import EasyRagClient

# Option A: explicit user JWT (from Supabase Auth)
client = EasyRagClient(user_jwt="eyJhbGci...")  # uses ANON_KEY + user JWT
results = client.search_hybrid("How do I configure vector indexes?")  # no token needed, RLS filters via auth.uid()

# Option B: scoped per-request helper
backend = EasyRagClient()  # service_role
user_client = backend.for_user(user_jwt)
results = user_client.search_hybrid("my private docs")

# Works for all search types
user_client.search_vector("hello", use_rls=True)
user_client.search_fts("hello", use_rls=True)

# RLS also works via PostgREST directly with RLS variants:
# knowledgebase.search_chunks_hybrid_rls, match_chunks_by_embedding_rls, etc. (SECURITY INVOKER)
```

### Direct Supabase JS (RLS)

```js
const { data } = await supabase
  .schema('knowledgebase')
  .rpc('search_chunks_hybrid_rls', {
    p_query: 'vector indexes',
    p_query_embedding: embedding,
    p_match_count: 5
  })
// RLS automatically filters to current user
```

---

## 🛠️ CLI Reference

- `easy-rag init-sql` — Generate Supabase SQL migration files.
- `easy-rag sync <directory> [--owner-id UUID] [--public]` — Sync markdown files to Supabase with RLS ownership.
- `easy-rag query "<text>" [--mode hybrid|vector|fts] [--token TOKEN] [--rls --user-jwt JWT]` — Run search (token or RLS mode).
- `easy-rag create-token "<name>"` — Create a new RAG access token (token mode).
- `easy-rag list-tokens` — List all registered access tokens.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
