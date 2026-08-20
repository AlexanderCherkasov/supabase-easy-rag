# Quick Start Guide: `supabase-easy-rag`

This guide gets you up and running with **Supabase Easy RAG** in under 5 minutes, including **Fine-Grained Access Control (RLS)** per Supabase's [RAG with Permissions](https://supabase.com/docs/guides/ai/rag-with-permissions).

---

## ⚡ Performance & Quality Summary

- **Hybrid Search Speed**: `< 2.5 ms` on PostgreSQL / Supabase with single round-trip CTE execution.
- **Vector ANN Search**: High-recall HNSW index scan (`vector_cosine_ops`) with iterative scan (`hnsw.iterative_scan = 'relaxed_order'`).
- **Full-Text Search (FTS)**: Multi-language GIN index search with weighted tsvector triggers (Title: A, Heading: B, Content: D).
- **Multi-Tenant Isolation**: 100% Zero Leakage verified under true PostgreSQL `authenticated` and `anon` database roles.
- **Async & Sync Engines**: Supports both sync and non-blocking `AsyncEasyRagClient` for high-concurrency async runtimes.

---

## Step 1: Database Setup in Supabase

### Option A: Via Supabase CLI (Recommended)

```bash
# Push migrations directly to your linked Supabase project:
supabase db push
```

### Option B: Via SQL Editor or CLI export

1. Generate migration files:
   ```bash
   easy-rag init-sql --output ./migrations --dimensions 1536
   ```
2. Open your Supabase Dashboard -> SQL Editor.
3. Run `migrations/01_schema.sql` (creates schema `knowledgebase`, tables, non-recursive RLS policies for `auth.uid()`, and indexes).
4. Run `migrations/02_functions.sql` (creates token + RLS-aware hybrid search RPCs).

---

## Step 2: Environment Configuration

Create a `.env` file in your project directory:

```env
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
SUPABASE_ANON_KEY=eyJhbGci...  # needed for RLS mode
OPENAI_API_KEY=sk-...
# For RLS mode, pass end-user JWT. For backend service token mode:
KNOWLEDGEBASE_ACCESS_TOKEN=kb_live_default_token
```

---

## Step 3: Ingest Markdown Knowledge Base

Structure your markdown documents in a folder (e.g. `./content`):

```
content/
├── setup/
│   └── installation.md
└── guides/
    └── vector-search.md
```

Run the sync CLI command:

```bash
# Option A: Private docs for a specific user (RLS)
easy-rag sync ./content --owner-id a0eebc99-1234-...

# Option B: Public docs (readable by all authenticated users)
easy-rag sync ./content --public

# Option C: Default (uses DB default auth.uid() or metadata Owner ID)
easy-rag sync ./content --workers 8
```

In markdown you can also set owner via metadata block:
```markdown
## Metadata
- **Owner ID**: a0eebc99-...
```

`supabase-easy-rag` will parse all H2-H6 section headings, calculate SHA-256 checksums, store sections hierarchy, and upsert vectors.

---

## Step 4: Querying from Python

### RLS mode (recommended for end-users)

```python
from supabase_easy_rag import EasyRagClient

# Pass the end-user JWT (from supabase.auth.getSession())
client = EasyRagClient(user_jwt="eyJhbGci...")

# Hybrid Search — automatically filtered via RLS (auth.uid())
results = client.search_hybrid(
    query="installation steps",
    match_count=3,
    expand_context="section"
)

for res in results:
    print(f"Doc: {res.document_title} > {res.section_title} | Score: {res.final_score}")
    print(res.effective_text[:200] + "...")
```

---

## 🧪 Local Testing & Verification

Run tests against a local PostgreSQL container:

```bash
./scripts/run_local_postgres_tests.sh
```
