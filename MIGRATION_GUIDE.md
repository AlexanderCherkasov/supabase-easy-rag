# Migration Guide — Supabase Cloud

If your Supabase database (`https://your-project.supabase.co`) is brand new, the `knowledgebase` schema does not exist yet (resulting in `PGRST106 Invalid schema: knowledgebase`). You need to run the SQL migration scripts once in the Supabase SQL Editor.

## Setup Steps (2 minutes)

1. Open `https://supabase.com/dashboard/project/YOUR_PROJECT_REF/sql/new` (replace `YOUR_PROJECT_REF` with your project reference from `SUPABASE_URL`).
2. Copy and run the entire content of `sql/01_schema.sql` (or `supabase/migrations/20260811000001_schema.sql`).
3. Copy and run the entire content of `sql/02_functions.sql` (or `supabase/migrations/20260811000002_functions.sql`).
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

## Context
PostgREST in Supabase Cloud exposes `public, graphql_public` schemas by default. Creating the `knowledgebase` schema makes Supabase expose it dynamically, but a reload might be required. Once migrated, RLS policies (`auth.uid() -> documents.owner_id`) and hybrid RPC functions (`search_chunks_hybrid`, `*_rls`) operate seamlessly.

