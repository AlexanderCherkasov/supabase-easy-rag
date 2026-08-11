# Quick Start Guide: `supabase-easy-rag`

This guide gets you up and running with **Supabase Easy RAG** in under 5 minutes, including **Fine-Grained Access Control (RLS)** per Supabase's [RAG with Permissions](https://supabase.com/docs/guides/ai/rag-with-permissions).

---

## ⚡ Performance Summary

- **Hybrid Search Speed**: `< 1.5 ms` average RPC response time.
- **Vector ANN Search**: `< 1.2 ms` via HNSW vector index (`vector_cosine_ops`).
- **Full-Text Search (FTS)**: `< 0.8 ms` via GIN index (`websearch_to_tsquery`).
- **Async & Sync Engines**: Supports both sync and non-blocking `AsyncEasyRagClient` for high-concurrency async runtimes.

---

## Step 1: Database Setup in Supabase

1. Open your Supabase Dashboard -> SQL Editor.
2. Generate migration files:
   ```bash
   easy-rag init-sql --output ./migrations
   ```
3. Copy and run `migrations/01_schema.sql` (creates schema `knowledgebase`, tables, RLS policies for `auth.uid()`, and indexes).
4. Copy and run `migrations/02_functions.sql` (creates token + RLS-aware hybrid search RPCs).

What `01_schema.sql` sets up (from the guide):
- `documents.owner_id UUID REFERENCES auth.users(id) DEFAULT auth.uid()`
- `document_owners` join table for multi-owner docs
- Policies: `Users can query their own document sections/chunks` via `document_id IN (SELECT id FROM documents WHERE owner_id = auth.uid())`

---

## Step 2: Environment Configuration

Create a `.env` file in your project directory:

```env
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
SUPABASE_ANON_KEY=eyJhbGci...  # needed for RLS mode
OPENAI_API_KEY=sk-...
# For RLS mode, no token needed. For backend token mode:
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

# Option B: Public docs (readable by all authenticated)
easy-rag sync ./content --public

# Option C: Legacy / default (uses DB default auth.uid() or metadata Owner ID)
easy-rag sync ./content
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
    match_count=3
)

for res in results:
    print(f"Doc: {res.document_title} | Score: {res.hybrid_score}")
    print(res.chunk_text[:200])
```

Per-request scoping:
```python
backend = EasyRagClient()  # service_role
user_client = backend.for_user(user_jwt)
results = user_client.search_hybrid("my docs")
```

### Token mode (backend jobs)

```python
from supabase_easy_rag import EasyRagClient

client = EasyRagClient()  # uses KNOWLEDGEBASE_ACCESS_TOKEN
results = client.search_hybrid(query="installation steps", kb_token="kb_live_...")
```

CLI also supports RLS:
```bash
easy-rag query "installation steps" --rls --user-jwt eyJhbGci...
easy-rag query "installation steps" --token kb_live_...
```

---

## Step 5: Managing Access Tokens

```bash
# Create a new token (token mode)
easy-rag create-token "Agent API Token"

# List active tokens
easy-rag list-tokens
```

For RLS you don't need tokens — just use Supabase Auth. See [README - RAG with Permissions](README.md#-rag-with-permissions-rls).

---

## Step 6: Multi-owner & Public Docs (from Supabase guide)

```sql
-- Share a document with another user
insert into knowledgebase.document_owners (document_id, owner_id)
values ('<doc-uuid>', '<other-user-uuid>');

-- Now both users see it in vector search:
select * from knowledgebase.chunks
where embedding <#> query_embedding < -0.5
order by embedding <#> query_embedding;
```

See `sql/01_schema.sql` comments for FDW / `app.current_user_id` variant.
