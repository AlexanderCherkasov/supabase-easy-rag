# Migration Guide — Supabase Cloud

If your Supabase database (`https://your-project.supabase.co`) is brand new, the `knowledgebase` schema does not exist yet (resulting in `PGRST106 Invalid schema: knowledgebase`). You need to run the SQL migration scripts once in the Supabase SQL Editor.

## Setup Steps (2 minutes)

1. Open `https://supabase.com/dashboard/project/YOUR_PROJECT_REF/sql/new` (replace `YOUR_PROJECT_REF` with your project reference from `SUPABASE_URL`).
2. Copy and run the entire content of `sql/01_schema.sql` (or generate via `easy-rag init-sql`).
3. Copy and run the entire content of `sql/02_functions.sql`.
4. Verify the database tables:
   ```sql
   select schema_name from information_schema.schemata where schema_name='knowledgebase';
   select table_name from information_schema.tables where table_schema='knowledgebase';
   ```
   You should see 8 tables: `documents`, `document_owners`, `document_sections`, `chunks`, `facets`, `document_facets`, `ingestion_runs`, and `access_tokens`.

5. Expose schema for PostgREST (if not exposed automatically):
   Dashboard -> Settings -> API -> Exposed schemas -> add `knowledgebase` (or execute in SQL Editor):
   ```sql
   -- Only needed if Supabase does not expose the schema automatically
   alter role anon set pgrst.db_schemas = 'public, storage, graphql_public, knowledgebase';
   alter role service_role set pgrst.db_schemas = 'public, storage, graphql_public, knowledgebase';
   select pg_reload_conf();
   ```

6. Test connection:
   ```bash
   uv run python -c "from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client; c=create_postgrest_client('https://your-project.supabase.co','sb_secret_YOUR_SECRET_KEY', schema_name='knowledgebase'); print(c.schema('knowledgebase').table('documents').select('id').limit(1).execute().data)"
   ```
   Should return `[]` without `PGRST106` errors.

## Upgrading from v0.1 to v0.2 (Two-Stage RRF + Weighted FTS)

If you already have an existing `knowledgebase` database and want to upgrade to Two-Stage RRF Hybrid Search and Weighted FTS without losing data, run the following SQL script in your Supabase SQL Editor:

```sql
-- 1. Upgrade search_vector to support weighted FTS (Title: 'A', Heading: 'B', Content: 'D')
ALTER TABLE knowledgebase.chunks DROP COLUMN IF EXISTS search_vector;
ALTER TABLE knowledgebase.chunks ADD COLUMN search_vector tsvector;

-- 2. Create the weighted FTS trigger
CREATE OR REPLACE FUNCTION knowledgebase.chunks_search_vector_trigger()
RETURNS TRIGGER AS $$
DECLARE
    v_doc_title TEXT := '';
    v_sec_heading TEXT := '';
    v_fts_config TEXT := 'english';
BEGIN
    SELECT COALESCE(title, '') INTO v_doc_title
    FROM knowledgebase.documents
    WHERE id = NEW.document_id;

    IF NEW.section_id IS NOT NULL THEN
        SELECT COALESCE(heading, '') INTO v_sec_heading
        FROM knowledgebase.document_sections
        WHERE id = NEW.section_id;
    END IF;

    IF NEW.metadata ? 'fts_config' THEN
        v_fts_config := NEW.metadata ->> 'fts_config';
    END IF;

    NEW.search_vector :=
        setweight(to_tsvector(v_fts_config::regconfig, COALESCE(v_doc_title, '')), 'A') ||
        setweight(to_tsvector(v_fts_config::regconfig, COALESCE(v_sec_heading, '')), 'B') ||
        setweight(to_tsvector(v_fts_config::regconfig, COALESCE(NEW.content, '')), 'D');

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 3. Attach triggers to chunks, documents, and sections
DROP TRIGGER IF EXISTS trigger_kb_chunks_search_vector ON knowledgebase.chunks;
CREATE TRIGGER trigger_kb_chunks_search_vector
BEFORE INSERT OR UPDATE OF document_id, section_id, content, metadata
ON knowledgebase.chunks
FOR EACH ROW EXECUTE FUNCTION knowledgebase.chunks_search_vector_trigger();

-- Backfill search_vector for all existing chunks
UPDATE knowledgebase.chunks SET updated_at = NOW();

-- 4. Recreate GIN Index
DROP INDEX IF EXISTS knowledgebase.idx_kb_chunks_fts;
CREATE INDEX idx_kb_chunks_fts ON knowledgebase.chunks USING gin(search_vector);

-- 5. Run sql/02_functions.sql to update RPC functions to two-stage RRF retrieval.
```

## Context
PostgREST in Supabase Cloud exposes `public, graphql_public` schemas by default. Creating the `knowledgebase` schema makes Supabase expose it dynamically, but a reload might be required. Once migrated, RLS policies (`auth.uid() -> documents.owner_id`) and hybrid RPC functions (`search_chunks_hybrid`, `*_rls`) operate seamlessly.


