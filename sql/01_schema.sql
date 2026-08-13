-- ============================================================================
-- Supabase Easy RAG: Database Schema
-- Enables pgvector extension, isolated schema, tables, indexes & token security
-- Implements Fine-Grained Access Control via RLS (see Supabase RAG with Permissions)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS knowledgebase;

-- Utility function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION knowledgebase.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 1. Documents Table
-- owner_id enables RLS per Supabase "RAG with Permissions" guide:
--   documents.owner_id references auth.users(id) default auth.uid()
CREATE TABLE IF NOT EXISTS knowledgebase.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    top_level_category TEXT,
    owner_id UUID REFERENCES auth.users(id) ON DELETE SET NULL DEFAULT auth.uid(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    checksum TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 1b. Document Owners Join Table (many-to-many, alternative scenario from guide)
-- Use this when a document is owned by multiple users.
CREATE TABLE IF NOT EXISTS knowledgebase.document_owners (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES knowledgebase.documents(id) ON DELETE CASCADE,
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, owner_id)
);

-- 2. Document Sections Table (Hierarchical sections: H2 - H6)
CREATE TABLE IF NOT EXISTS knowledgebase.document_sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES knowledgebase.documents(id) ON DELETE CASCADE,
    parent_section_id UUID REFERENCES knowledgebase.document_sections(id) ON DELETE CASCADE,
    heading TEXT NOT NULL,
    level INT NOT NULL,
    sort_order INT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, sort_order)
);

-- 3. Document Chunks Table (Vector Embeddings + Full-Text Search Vector)
CREATE TABLE IF NOT EXISTS knowledgebase.chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES knowledgebase.documents(id) ON DELETE CASCADE,
    section_id UUID REFERENCES knowledgebase.document_sections(id) ON DELETE SET NULL,
    chunk_index INT NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    token_count INT,
    char_count INT,
    embedding VECTOR(1536),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

-- 4. Facets Table (Hierarchical categories & attributes for navigation)
CREATE TABLE IF NOT EXISTS knowledgebase.facets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facet_type TEXT NOT NULL,
    facet_key TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    parent_facet_id UUID REFERENCES knowledgebase.facets(id) ON DELETE SET NULL,
    sort_order INT NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Document-Facet Junction Table
CREATE TABLE IF NOT EXISTS knowledgebase.document_facets (
    document_id UUID NOT NULL REFERENCES knowledgebase.documents(id) ON DELETE CASCADE,
    facet_id UUID NOT NULL REFERENCES knowledgebase.facets(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (document_id, facet_id)
);

-- 6. Ingestion Runs Logging Table
CREATE TABLE IF NOT EXISTS knowledgebase.ingestion_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL,
    source_root TEXT NOT NULL,
    files_seen INT NOT NULL DEFAULT 0,
    files_changed INT NOT NULL DEFAULT 0,
    error_summary TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Access Tokens Table (Granular RAG authentication for service_role / machine-to-machine)
-- For end-user RLS use auth.uid() directly; tokens remain for backend jobs & legacy.
CREATE TABLE IF NOT EXISTS knowledgebase.access_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. Access Token Audit Logs Table
CREATE TABLE IF NOT EXISTS knowledgebase.access_token_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    access_token_id UUID REFERENCES knowledgebase.access_tokens(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    user_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_kb_documents_owner ON knowledgebase.documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_kb_doc_owners_doc ON knowledgebase.document_owners(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_doc_owners_owner ON knowledgebase.document_owners(owner_id);
CREATE INDEX IF NOT EXISTS idx_kb_sections_doc_id ON knowledgebase.document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_sections_parent ON knowledgebase.document_sections(parent_section_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON knowledgebase.chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_section_id ON knowledgebase.chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON knowledgebase.chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_fts ON knowledgebase.chunks USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_kb_facets_parent ON knowledgebase.facets(parent_facet_id);
CREATE INDEX IF NOT EXISTS idx_kb_facets_type ON knowledgebase.facets(facet_type);
CREATE INDEX IF NOT EXISTS idx_kb_doc_facets_facet ON knowledgebase.document_facets(facet_id);
CREATE INDEX IF NOT EXISTS idx_kb_tokens_active ON knowledgebase.access_tokens(is_active, expires_at);
CREATE INDEX IF NOT EXISTS idx_kb_token_audit ON knowledgebase.access_token_audit(access_token_id, created_at DESC);

-- Triggers for updated_at
CREATE TRIGGER update_kb_documents_updated_at BEFORE UPDATE ON knowledgebase.documents FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();
CREATE TRIGGER update_kb_sections_updated_at BEFORE UPDATE ON knowledgebase.document_sections FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();
CREATE TRIGGER update_kb_chunks_updated_at BEFORE UPDATE ON knowledgebase.chunks FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();
CREATE TRIGGER update_kb_facets_updated_at BEFORE UPDATE ON knowledgebase.facets FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();
CREATE TRIGGER update_kb_ingestion_updated_at BEFORE UPDATE ON knowledgebase.ingestion_runs FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();
CREATE TRIGGER update_kb_tokens_updated_at BEFORE UPDATE ON knowledgebase.access_tokens FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

-- ============================================================================
-- Row Level Security (RLS) - Fine-Grained Access Control for RAG
-- Based on https://supabase.com/docs/guides/ai/rag-with-permissions
-- ============================================================================

ALTER TABLE knowledgebase.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.document_owners ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.document_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.facets ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.document_facets ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.ingestion_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.access_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.access_token_audit ENABLE ROW LEVEL SECURITY;

-- Grants: schema usage
GRANT USAGE ON SCHEMA knowledgebase TO authenticated;
GRANT USAGE ON SCHEMA knowledgebase TO service_role;
GRANT USAGE ON SCHEMA knowledgebase TO anon;

-- Grants: service_role bypasses RLS - full access for backend ingestion & token RPCs
GRANT ALL ON ALL TABLES IN SCHEMA knowledgebase TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA knowledgebase TO service_role;

-- Grants: authenticated can SELECT via RLS; anon gets nothing by default
REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM anon;
REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM authenticated;
GRANT SELECT ON knowledgebase.documents TO authenticated;
GRANT SELECT ON knowledgebase.document_owners TO authenticated;
GRANT SELECT ON knowledgebase.document_sections TO authenticated;
GRANT SELECT ON knowledgebase.chunks TO authenticated;
GRANT SELECT ON knowledgebase.facets TO authenticated;
GRANT SELECT ON knowledgebase.document_facets TO authenticated;
-- authenticated can also read facets/navigation without ownership check
GRANT SELECT ON knowledgebase.facets TO anon;

-- Helper: check if current user owns document (single-owner + multi-owner + public fallback)
-- Public documents: owner_id IS NULL and no entry in document_owners => visible to all authenticated

-- Policy: documents - users can read own docs + public docs + shared via document_owners
DROP POLICY IF EXISTS "Users can query their own documents" ON knowledgebase.documents;
CREATE POLICY "Users can query their own documents"
ON knowledgebase.documents FOR SELECT TO authenticated USING (
  owner_id IS NULL
  OR owner_id = auth.uid()
  OR EXISTS (
    SELECT 1 FROM knowledgebase.document_owners do2
    WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
  )
);

DROP POLICY IF EXISTS "Users can insert their own documents" ON knowledgebase.documents;
CREATE POLICY "Users can insert their own documents"
ON knowledgebase.documents FOR INSERT TO authenticated WITH CHECK (
  owner_id = auth.uid() OR owner_id IS NULL
);

DROP POLICY IF EXISTS "Users can update their own documents" ON knowledgebase.documents;
CREATE POLICY "Users can update their own documents"
ON knowledgebase.documents FOR UPDATE TO authenticated USING (
  owner_id = auth.uid() OR EXISTS (
    SELECT 1 FROM knowledgebase.document_owners do2
    WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
  )
);

DROP POLICY IF EXISTS "Users can delete their own documents" ON knowledgebase.documents;
CREATE POLICY "Users can delete their own documents"
ON knowledgebase.documents FOR DELETE TO authenticated USING (
  owner_id = auth.uid() OR EXISTS (
    SELECT 1 FROM knowledgebase.document_owners do2
    WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
  )
);

-- Policy: document_owners - user can see ownership rows for docs they own or are member of
DROP POLICY IF EXISTS "Users can query document_owners" ON knowledgebase.document_owners;
CREATE POLICY "Users can query document_owners"
ON knowledgebase.document_owners FOR SELECT TO authenticated USING (
  owner_id = auth.uid()
  OR document_id IN (SELECT id FROM knowledgebase.documents WHERE owner_id = auth.uid())
);

-- Policy: chunks / document_sections - restrict via linked document ownership (core RAG pattern)
-- This mirrors the Supabase guide: document_sections filtered via documents.owner_id
DROP POLICY IF EXISTS "Users can query their own document sections" ON knowledgebase.document_sections;
CREATE POLICY "Users can query their own document sections"
ON knowledgebase.document_sections FOR SELECT TO authenticated USING (
  document_id IN (
    SELECT id FROM knowledgebase.documents
    WHERE owner_id IS NULL
       OR owner_id = auth.uid()
       OR EXISTS (
         SELECT 1 FROM knowledgebase.document_owners do2
         WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
       )
  )
);

DROP POLICY IF EXISTS "Users can query their own chunks" ON knowledgebase.chunks;
CREATE POLICY "Users can query their own chunks"
ON knowledgebase.chunks FOR SELECT TO authenticated USING (
  document_id IN (
    SELECT id FROM knowledgebase.documents
    WHERE owner_id IS NULL
       OR owner_id = auth.uid()
       OR EXISTS (
         SELECT 1 FROM knowledgebase.document_owners do2
         WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
       )
  )
);

-- Optional: Alternative policy for external user source via FDW or custom JWT claim
-- Uncomment if you use direct Postgres connection with app.current_user_id:
-- CREATE POLICY "Users can query via app.current_user_id"
-- ON knowledgebase.chunks FOR SELECT TO authenticated USING (
--   document_id IN (
--     SELECT id FROM knowledgebase.documents
--     WHERE owner_id::text = current_setting('app.current_user_id', true)
--   )
-- );

-- Facets: public read (no ownership), restrict writes to service_role (no policy for authenticated insert)
DROP POLICY IF EXISTS "Anyone can read facets" ON knowledgebase.facets;
CREATE POLICY "Anyone can read facets"
ON knowledgebase.facets FOR SELECT TO authenticated, anon USING (true);

-- Document facets: restricted via linked document ownership (prevents metadata leakage)
DROP POLICY IF EXISTS "Users can query their own document facets" ON knowledgebase.document_facets;
CREATE POLICY "Users can query their own document facets"
ON knowledgebase.document_facets FOR SELECT TO authenticated USING (
  document_id IN (
    SELECT id FROM knowledgebase.documents
    WHERE owner_id IS NULL
       OR owner_id = auth.uid()
       OR EXISTS (
         SELECT 1 FROM knowledgebase.document_owners do2
         WHERE do2.document_id = documents.id AND do2.owner_id = auth.uid()
       )
  )
);

-- Ingestion / tokens / audit: service_role only (no authenticated policies => no access)
-- Intentionally no policies for authenticated on these tables.
