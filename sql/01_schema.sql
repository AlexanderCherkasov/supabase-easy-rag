-- ============================================================================
-- Supabase Easy RAG: Database Schema
-- Enables pgvector extension, isolated schema, tables, indexes & token security
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
CREATE TABLE IF NOT EXISTS knowledgebase.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    top_level_category TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    checksum TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

-- 7. Access Tokens Table (Granular RAG authentication)
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
CREATE INDEX IF NOT EXISTS idx_kb_sections_doc_id ON knowledgebase.document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_sections_parent ON knowledgebase.document_sections(parent_section_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON knowledgebase.chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_section_id ON knowledgebase.chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON knowledgebase.chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
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

-- Row Level Security (RLS) Configuration
ALTER TABLE knowledgebase.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.document_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.facets ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.document_facets ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.ingestion_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.access_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledgebase.access_token_audit ENABLE ROW LEVEL SECURITY;

GRANT USAGE ON SCHEMA knowledgebase TO authenticated;
GRANT USAGE ON SCHEMA knowledgebase TO service_role;

REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM anon;
REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA knowledgebase TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA knowledgebase TO service_role;
