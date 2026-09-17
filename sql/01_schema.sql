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
    tenant_id UUID,
    scope_id UUID,
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
    embedding VECTOR(1536), -- Configurable dimensions via: easy-rag init-sql --dimensions <DIM>
    search_vector tsvector,
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
    tenant_id UUID,
    user_id UUID DEFAULT auth.uid(),
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
    -- NULL is a global token; non-NULL restricts token RPCs to this owner's documents.
    tenant_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
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

-- Function: Compute Weighted tsvector (A: Title, B: Section Heading, D: Content)
-- Supports dynamic language / fts_config from metadata with safe fallback to 'simple'
CREATE OR REPLACE FUNCTION knowledgebase.chunks_search_vector_trigger()
RETURNS TRIGGER AS $$
DECLARE
    v_doc_title TEXT := '';
    v_sec_heading TEXT := '';
    v_fts_config TEXT := 'english';
    v_regconfig regconfig := 'english'::regconfig;
BEGIN
    SELECT COALESCE(title, '') INTO v_doc_title
    FROM knowledgebase.documents
    WHERE id = NEW.document_id;

    IF NEW.section_id IS NOT NULL THEN
        SELECT COALESCE(heading, '') INTO v_sec_heading
        FROM knowledgebase.document_sections
        WHERE id = NEW.section_id;
    END IF;

    -- Extract language / fts_config from metadata
    IF NEW.metadata ? 'fts_config' THEN
        v_fts_config := NEW.metadata ->> 'fts_config';
    ELSIF NEW.metadata ? 'language' THEN
        v_fts_config := NEW.metadata ->> 'language';
    END IF;

    -- Safe cast to regconfig; fallback to 'simple' for unsupported languages
    BEGIN
        v_regconfig := v_fts_config::regconfig;
    EXCEPTION WHEN OTHERS THEN
        v_regconfig := 'simple'::regconfig;
    END;

    NEW.search_vector :=
        setweight(to_tsvector(v_regconfig, COALESCE(v_doc_title, '')), 'A') ||
        setweight(to_tsvector(v_regconfig, COALESCE(v_sec_heading, '')), 'B') ||
        setweight(to_tsvector(v_regconfig, COALESCE(NEW.content, '')), 'D');

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Cascade search vector update on Document Title change
CREATE OR REPLACE FUNCTION knowledgebase.documents_title_update_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.title IS DISTINCT FROM NEW.title THEN
        UPDATE knowledgebase.chunks
        SET updated_at = NOW()
        WHERE document_id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Cascade search vector update on Section Heading change
CREATE OR REPLACE FUNCTION knowledgebase.sections_heading_update_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.heading IS DISTINCT FROM NEW.heading THEN
        UPDATE knowledgebase.chunks
        SET updated_at = NOW()
        WHERE section_id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_kb_documents_owner ON knowledgebase.documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_kb_documents_tenant_scope ON knowledgebase.documents(tenant_id, scope_id);
CREATE INDEX IF NOT EXISTS idx_kb_ingestion_tenant_user ON knowledgebase.ingestion_runs(tenant_id, user_id);
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
DROP TRIGGER IF EXISTS update_kb_documents_updated_at ON knowledgebase.documents;
CREATE TRIGGER update_kb_documents_updated_at BEFORE UPDATE ON knowledgebase.documents FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

DROP TRIGGER IF EXISTS update_kb_sections_updated_at ON knowledgebase.document_sections;
CREATE TRIGGER update_kb_sections_updated_at BEFORE UPDATE ON knowledgebase.document_sections FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

DROP TRIGGER IF EXISTS update_kb_chunks_updated_at ON knowledgebase.chunks;
CREATE TRIGGER update_kb_chunks_updated_at BEFORE UPDATE ON knowledgebase.chunks FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

DROP TRIGGER IF EXISTS update_kb_facets_updated_at ON knowledgebase.facets;
CREATE TRIGGER update_kb_facets_updated_at BEFORE UPDATE ON knowledgebase.facets FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

DROP TRIGGER IF EXISTS update_kb_ingestion_updated_at ON knowledgebase.ingestion_runs;
CREATE TRIGGER update_kb_ingestion_updated_at BEFORE UPDATE ON knowledgebase.ingestion_runs FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

DROP TRIGGER IF EXISTS update_kb_tokens_updated_at ON knowledgebase.access_tokens;
CREATE TRIGGER update_kb_tokens_updated_at BEFORE UPDATE ON knowledgebase.access_tokens FOR EACH ROW EXECUTE FUNCTION knowledgebase.update_updated_at_column();

-- Triggers for Weighted FTS Search Vector
DROP TRIGGER IF EXISTS trigger_kb_chunks_search_vector ON knowledgebase.chunks;
CREATE TRIGGER trigger_kb_chunks_search_vector
BEFORE INSERT OR UPDATE
ON knowledgebase.chunks
FOR EACH ROW EXECUTE FUNCTION knowledgebase.chunks_search_vector_trigger();


DROP TRIGGER IF EXISTS trigger_kb_documents_title_update ON knowledgebase.documents;
CREATE TRIGGER trigger_kb_documents_title_update
AFTER UPDATE OF title ON knowledgebase.documents
FOR EACH ROW EXECUTE FUNCTION knowledgebase.documents_title_update_trigger();

DROP TRIGGER IF EXISTS trigger_kb_sections_heading_update ON knowledgebase.document_sections;
CREATE TRIGGER trigger_kb_sections_heading_update
AFTER UPDATE OF heading ON knowledgebase.document_sections
FOR EACH ROW EXECUTE FUNCTION knowledgebase.sections_heading_update_trigger();

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

-- Grants: authenticated gets access to documents, sections, chunks, facets & ingestion runs via RLS
REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM anon;
REVOKE ALL ON ALL TABLES IN SCHEMA knowledgebase FROM authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.documents TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.document_owners TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.document_sections TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.chunks TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.document_facets TO authenticated;
GRANT SELECT, INSERT, UPDATE ON knowledgebase.ingestion_runs TO authenticated;
GRANT SELECT ON knowledgebase.facets TO authenticated, anon;

-- Helper Functions for Fine-Grained Access & Multitenancy Integration
CREATE OR REPLACE FUNCTION knowledgebase.can_read_document(p_doc_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_doc RECORD;
    v_role TEXT;
    v_user_id UUID;
    v_has_access BOOLEAN;
BEGIN
    v_role := COALESCE(current_setting('request.jwt.claim.role', true), 'anon');
    IF v_role = 'service_role' THEN
        RETURN TRUE;
    END IF;

    v_user_id := auth.uid();

    SELECT id, tenant_id, scope_id, owner_id, top_level_category
    INTO v_doc
    FROM knowledgebase.documents
    WHERE id = p_doc_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- System Category Isolation:
    -- System documents are restricted to service_role, explicit system claim, or RBAC permission.
    IF v_doc.top_level_category = 'system' THEN
        IF (auth.jwt() -> 'app_metadata' ->> 'is_system_agent')::boolean IS TRUE THEN
            RETURN TRUE;
        END IF;
        IF to_regproc('api.has_permission') IS NOT NULL AND v_doc.tenant_id IS NOT NULL THEN
            BEGIN
                EXECUTE 'SELECT api.has_permission($1, $2)'
                INTO v_has_access
                USING v_doc.tenant_id, 'knowledgebase.system.read';
                IF v_has_access IS TRUE THEN
                    RETURN TRUE;
                END IF;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
        END IF;
        RETURN FALSE;
    END IF;

    -- Global Scope (tenant_id IS NULL):
    IF v_doc.tenant_id IS NULL THEN
        RETURN (
            v_doc.owner_id IS NULL
            OR v_doc.owner_id = v_user_id
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_owners do2
                WHERE do2.document_id = v_doc.id AND do2.owner_id = v_user_id
            )
        );
    END IF;

    -- Tenant Scope with supabase-multitenancy:
    IF to_regproc('api.access_level') IS NOT NULL THEN
        BEGIN
            EXECUTE 'SELECT api.access_level($1, $2, ARRAY[$3])'
            INTO v_role
            USING v_doc.tenant_id, 'knowledgebase.read', v_doc.scope_id;

            IF v_role = 'all' THEN
                RETURN TRUE;
            ELSIF v_role = 'own' THEN
                RETURN (v_doc.owner_id = v_user_id);
            ELSE
                RETURN FALSE;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END IF;

    -- Fallback when supabase-multitenancy is not installed
    RETURN (
        v_doc.owner_id IS NULL
        OR v_doc.owner_id = v_user_id
        OR EXISTS (
            SELECT 1 FROM knowledgebase.document_owners do2
            WHERE do2.document_id = v_doc.id AND do2.owner_id = v_user_id
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.can_write_document(p_doc_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_doc RECORD;
    v_role TEXT;
    v_user_id UUID;
    v_level TEXT;
BEGIN
    v_role := COALESCE(current_setting('request.jwt.claim.role', true), 'anon');
    IF v_role = 'service_role' THEN
        RETURN TRUE;
    END IF;

    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RETURN FALSE;
    END IF;

    SELECT id, tenant_id, scope_id, owner_id, top_level_category
    INTO v_doc
    FROM knowledgebase.documents
    WHERE id = p_doc_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- Regular users cannot modify system category or global documents
    IF v_doc.top_level_category = 'system' OR v_doc.tenant_id IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Tenant Scope with supabase-multitenancy
    IF to_regproc('api.access_level') IS NOT NULL THEN
        BEGIN
            EXECUTE 'SELECT api.access_level($1, $2, ARRAY[$3])'
            INTO v_level
            USING v_doc.tenant_id, 'knowledgebase.ingest', v_doc.scope_id;

            IF v_level = 'all' THEN
                RETURN TRUE;
            ELSIF v_level = 'own' THEN
                RETURN (v_doc.owner_id = v_user_id);
            ELSE
                RETURN FALSE;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END IF;

    -- Fallback when supabase-multitenancy is not installed
    RETURN (
        v_doc.owner_id = v_user_id
        OR EXISTS (
            SELECT 1 FROM knowledgebase.document_owners do2
            WHERE do2.document_id = v_doc.id AND do2.owner_id = v_user_id
        )
    );
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.can_insert_document(
    p_tenant_id UUID,
    p_scope_id UUID,
    p_owner_id UUID,
    p_top_level_category TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_role TEXT;
    v_user_id UUID;
    v_level TEXT;
BEGIN
    v_role := COALESCE(current_setting('request.jwt.claim.role', true), 'anon');
    IF v_role = 'service_role' THEN
        RETURN TRUE;
    END IF;

    IF p_top_level_category = 'system' THEN
        RETURN FALSE;
    END IF;

    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN
        RETURN FALSE;
    END IF;

    IF p_tenant_id IS NOT NULL THEN
        IF to_regproc('api.access_level') IS NOT NULL THEN
            BEGIN
                EXECUTE 'SELECT api.access_level($1, $2, ARRAY[$3])'
                INTO v_level
                USING p_tenant_id, 'knowledgebase.ingest', p_scope_id;

                IF v_level = 'all' THEN
                    RETURN TRUE;
                ELSIF v_level = 'own' THEN
                    RETURN (p_owner_id = v_user_id OR p_owner_id IS NULL);
                ELSE
                    RETURN FALSE;
                END IF;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
        END IF;
        RETURN (p_owner_id = v_user_id OR p_owner_id IS NULL);
    ELSE
        RETURN (p_owner_id = v_user_id);
    END IF;
END;
$$;

-- Policy: documents
DROP POLICY IF EXISTS "Users can query their own documents" ON knowledgebase.documents;
DROP POLICY IF EXISTS "documents_select_scoped" ON knowledgebase.documents;
CREATE POLICY "documents_select_scoped"
ON knowledgebase.documents FOR SELECT TO authenticated USING (
    knowledgebase.can_read_document(id)
);

DROP POLICY IF EXISTS "Users can insert their own documents" ON knowledgebase.documents;
DROP POLICY IF EXISTS "documents_insert_scoped" ON knowledgebase.documents;
CREATE POLICY "documents_insert_scoped"
ON knowledgebase.documents FOR INSERT TO authenticated WITH CHECK (
    knowledgebase.can_insert_document(tenant_id, scope_id, owner_id, top_level_category)
);

DROP POLICY IF EXISTS "Users can update their own documents" ON knowledgebase.documents;
DROP POLICY IF EXISTS "documents_update_scoped" ON knowledgebase.documents;
CREATE POLICY "documents_update_scoped"
ON knowledgebase.documents FOR UPDATE TO authenticated
USING (knowledgebase.can_write_document(id))
WITH CHECK (knowledgebase.can_write_document(id));

DROP POLICY IF EXISTS "Users can delete their own documents" ON knowledgebase.documents;
DROP POLICY IF EXISTS "documents_delete_scoped" ON knowledgebase.documents;
CREATE POLICY "documents_delete_scoped"
ON knowledgebase.documents FOR DELETE TO authenticated
USING (knowledgebase.can_write_document(id));

-- Policy: document_owners
DROP POLICY IF EXISTS "Users can query document_owners" ON knowledgebase.document_owners;
CREATE POLICY "Users can query document_owners"
ON knowledgebase.document_owners FOR SELECT TO authenticated USING (
    owner_id = auth.uid()
);

-- Policy: document_sections
DROP POLICY IF EXISTS "Users can query their own document sections" ON knowledgebase.document_sections;
DROP POLICY IF EXISTS "document_sections_select_scoped" ON knowledgebase.document_sections;
CREATE POLICY "document_sections_select_scoped"
ON knowledgebase.document_sections FOR SELECT TO authenticated
USING (knowledgebase.can_read_document(document_id));

DROP POLICY IF EXISTS "document_sections_insert_scoped" ON knowledgebase.document_sections;
CREATE POLICY "document_sections_insert_scoped"
ON knowledgebase.document_sections FOR INSERT TO authenticated
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "document_sections_update_scoped" ON knowledgebase.document_sections;
CREATE POLICY "document_sections_update_scoped"
ON knowledgebase.document_sections FOR UPDATE TO authenticated
USING (knowledgebase.can_write_document(document_id))
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "document_sections_delete_scoped" ON knowledgebase.document_sections;
CREATE POLICY "document_sections_delete_scoped"
ON knowledgebase.document_sections FOR DELETE TO authenticated
USING (knowledgebase.can_write_document(document_id));

-- Policy: chunks
DROP POLICY IF EXISTS "Users can query their own chunks" ON knowledgebase.chunks;
DROP POLICY IF EXISTS "chunks_select_scoped" ON knowledgebase.chunks;
CREATE POLICY "chunks_select_scoped"
ON knowledgebase.chunks FOR SELECT TO authenticated
USING (knowledgebase.can_read_document(document_id));

DROP POLICY IF EXISTS "chunks_insert_scoped" ON knowledgebase.chunks;
CREATE POLICY "chunks_insert_scoped"
ON knowledgebase.chunks FOR INSERT TO authenticated
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "chunks_update_scoped" ON knowledgebase.chunks;
CREATE POLICY "chunks_update_scoped"
ON knowledgebase.chunks FOR UPDATE TO authenticated
USING (knowledgebase.can_write_document(document_id))
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "chunks_delete_scoped" ON knowledgebase.chunks;
CREATE POLICY "chunks_delete_scoped"
ON knowledgebase.chunks FOR DELETE TO authenticated
USING (knowledgebase.can_write_document(document_id));

-- Facets: public read
DROP POLICY IF EXISTS "Anyone can read facets" ON knowledgebase.facets;
CREATE POLICY "Anyone can read facets"
ON knowledgebase.facets FOR SELECT TO authenticated, anon USING (true);

-- Document Facets: restricted via document access
DROP POLICY IF EXISTS "Users can query their own document facets" ON knowledgebase.document_facets;
DROP POLICY IF EXISTS "document_facets_select_scoped" ON knowledgebase.document_facets;
CREATE POLICY "document_facets_select_scoped"
ON knowledgebase.document_facets FOR SELECT TO authenticated
USING (knowledgebase.can_read_document(document_id));

DROP POLICY IF EXISTS "document_facets_insert_scoped" ON knowledgebase.document_facets;
CREATE POLICY "document_facets_insert_scoped"
ON knowledgebase.document_facets FOR INSERT TO authenticated
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "document_facets_update_scoped" ON knowledgebase.document_facets;
CREATE POLICY "document_facets_update_scoped"
ON knowledgebase.document_facets FOR UPDATE TO authenticated
USING (knowledgebase.can_write_document(document_id))
WITH CHECK (knowledgebase.can_write_document(document_id));

DROP POLICY IF EXISTS "document_facets_delete_scoped" ON knowledgebase.document_facets;
CREATE POLICY "document_facets_delete_scoped"
ON knowledgebase.document_facets FOR DELETE TO authenticated
USING (knowledgebase.can_write_document(document_id));

-- Ingestion Runs: authenticated users manage their own runs
DROP POLICY IF EXISTS "ingestion_runs_scoped" ON knowledgebase.ingestion_runs;
CREATE POLICY "ingestion_runs_scoped"
ON knowledgebase.ingestion_runs FOR ALL TO authenticated
USING (user_id = auth.uid() OR user_id IS NULL)
WITH CHECK (user_id = auth.uid() OR user_id IS NULL);

