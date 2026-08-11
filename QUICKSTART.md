# Quick Start Guide: `supabase-easy-rag`

This guide gets you up and running with **Supabase Easy RAG** in under 5 minutes.

---

## Step 1: Database Setup in Supabase

1. Open your Supabase Dashboard -> SQL Editor.
2. Generate migration files:
   ```bash
   easy-rag init-sql --output ./migrations
   ```
3. Copy and run `migrations/01_schema.sql` (creates schema `knowledgebase`, tables, RLS, and indexes).
4. Copy and run `migrations/02_functions.sql` (creates security assertion and hybrid search RPCs).

---

## Step 2: Environment Configuration

Create a `.env` file in your project directory:

```env
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
OPENAI_API_KEY=sk-...
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
easy-rag sync ./content
```

`supabase-easy-rag` will parse all H2-H6 section headings, metadata headers, calculate SHA-256 checksums, and upsert vectors to Supabase.

---

## Step 4: Querying from Python

```python
from supabase_easy_rag import EasyRagClient

client = EasyRagClient()

# Hybrid Search (Vector + FTS + Title Boost)
results = client.search_hybrid(
    query="installation steps",
    match_count=3
)

for res in results:
    print(f"Doc: {res.document_title} | Score: {res.hybrid_score}")
    print(res.chunk_text[:200])
```

---

## Step 5: Managing Access Tokens

```bash
# Create a new token
easy-rag create-token "Agent API Token"

# List active tokens
easy-rag list-tokens
```
