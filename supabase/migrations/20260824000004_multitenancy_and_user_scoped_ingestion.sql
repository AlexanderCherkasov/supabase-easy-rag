-- ============================================================================
-- Migration: 20260824000004_multitenancy_and_user_scoped_ingestion.sql
-- Integrates with supabase-multitenancy:
-- 1. Adds tenant_id & scope_id to documents and ingestion_runs
-- 2. Grants INSERT, UPDATE, DELETE on documents, sections, chunks to authenticated
-- 3. Implements can_read_document() and can_write_document() RLS helpers
-- 4. Enforces category isolation (KB token vs User JWT)
-- 5. Upgrades search RPCs with tenant_id, scope_id, include_global, allowed_categories
-- ============================================================================

-- 1. Schema Alterations
ALTER TABLE knowledgebase.documents
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    ADD COLUMN IF NOT EXISTS scope_id UUID;

CREATE INDEX IF NOT EXISTS idx_kb_documents_tenant_scope
    ON knowledgebase.documents(tenant_id, scope_id);

ALTER TABLE knowledgebase.ingestion_runs
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    ADD COLUMN IF NOT EXISTS user_id UUID DEFAULT auth.uid();

CREATE INDEX IF NOT EXISTS idx_kb_ingestion_tenant_user
    ON knowledgebase.ingestion_runs(tenant_id, user_id);

-- Optional foreign key references if multitenancy schema is installed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'multitenancy' AND table_name = 'tenants'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'fk_kb_documents_tenant'
              AND table_schema = 'knowledgebase'
              AND table_name = 'documents'
        ) THEN
            ALTER TABLE knowledgebase.documents
                ADD CONSTRAINT fk_kb_documents_tenant
                FOREIGN KEY (tenant_id) REFERENCES multitenancy.tenants(id) ON DELETE CASCADE;
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'fk_kb_ingestion_tenant'
              AND table_schema = 'knowledgebase'
              AND table_name = 'ingestion_runs'
        ) THEN
            ALTER TABLE knowledgebase.ingestion_runs
                ADD CONSTRAINT fk_kb_ingestion_tenant
                FOREIGN KEY (tenant_id) REFERENCES multitenancy.tenants(id) ON DELETE CASCADE;
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'multitenancy' AND table_name = 'scopes'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = 'fk_kb_documents_scope'
              AND table_schema = 'knowledgebase'
              AND table_name = 'documents'
        ) THEN
            ALTER TABLE knowledgebase.documents
                ADD CONSTRAINT fk_kb_documents_scope
                FOREIGN KEY (tenant_id, scope_id) REFERENCES multitenancy.scopes(tenant_id, id) ON DELETE RESTRICT;
        END IF;
    END IF;
END $$;

-- 2. Grants for authenticated (enables User-Scoped Ingestion via RLS)
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.documents TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.document_sections TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.chunks TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledgebase.document_facets TO authenticated;
GRANT SELECT, INSERT, UPDATE ON knowledgebase.ingestion_runs TO authenticated;

-- 3. RLS Security Helper Functions
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
    -- Readable by authenticated users if public (owner_id IS NULL) or owned by current user or shared.
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

    -- Tenant Scope:
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

-- 4. RLS Policies

-- Documents Policies
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

-- Sections Policies
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

-- Chunks Policies
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

-- Document Facets Policies
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

-- Ingestion Runs Policy (Authenticated Users can manage their own runs)
DROP POLICY IF EXISTS "ingestion_runs_scoped" ON knowledgebase.ingestion_runs;
CREATE POLICY "ingestion_runs_scoped"
ON knowledgebase.ingestion_runs FOR ALL TO authenticated
USING (user_id = auth.uid() OR user_id IS NULL)
WITH CHECK (user_id = auth.uid() OR user_id IS NULL);

-- 5. Upgraded Search RPCs
-- Drop obsolete function overloads from migrations 02 and 03
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding(text, vector, integer, text[], double precision, integer);
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding_rls(vector, integer, text[], double precision, integer);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_full_text(text, text, integer, text[], text);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_full_text_rls(text, integer, text[], text);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid(text, text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid_rls(text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer);

CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding(
    p_kb_token TEXT,
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL,
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_token_id UUID;
    v_tenant_id UUID;
    v_token_meta JSONB;
    v_token_categories TEXT[];
    v_is_rls BOOLEAN;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

    IF v_token_id IS NOT NULL THEN
        SELECT t.tenant_id, t.metadata INTO v_tenant_id, v_token_meta
        FROM knowledgebase.access_tokens t WHERE t.id = v_token_id;

        IF v_token_meta ? 'allowed_categories' THEN
            SELECT ARRAY(SELECT jsonb_array_elements_text(v_token_meta -> 'allowed_categories'))
            INTO v_token_categories;
        END IF;
    END IF;

    IF p_query_embedding IS NULL THEN
        RAISE EXCEPTION 'Query embedding is required';
    END IF;

    BEGIN
        PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);
        PERFORM set_config('hnsw.ef_search', GREATEST(COALESCE(p_ef_search, COALESCE(p_match_count, 5) * 10), 40)::text, true);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    RETURN QUERY
    WITH candidates AS (
        SELECT
            c.id,
            d.id AS doc_id,
            d.title AS doc_title,
            ds.heading AS sec_title,
            c.content,
            c.metadata ->> 'facet_path' AS f_path,
            c.metadata AS c_meta,
            (1 - (c.embedding <=> p_query_embedding)) AS v_score,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_query_embedding) AS v_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.embedding IS NOT NULL
          AND (p_min_vector_similarity IS NULL OR (1 - (c.embedding <=> p_query_embedding)) >= p_min_vector_similarity)
          -- Category isolation
          AND (
            -- 1. KB Token caller
            (v_token_id IS NOT NULL AND (
                v_token_categories IS NULL OR d.top_level_category = ANY(v_token_categories)
            ))
            -- 2. User JWT / RLS caller (exclude system unless explicitly allowed)
            OR (v_is_rls AND (
                (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
                OR (d.top_level_category IS DISTINCT FROM 'system')
            ))
            -- 3. Service role caller without token
            OR (v_token_id IS NULL AND NOT v_is_rls)
          )
          -- Tenant / Scope isolation
          AND (
            -- Service role full access
            (v_token_id IS NULL AND NOT v_is_rls)
            -- Scoped token
            OR (v_token_id IS NOT NULL AND (
                (v_tenant_id IS NULL AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id))
                OR (v_tenant_id IS NOT NULL AND d.tenant_id = v_tenant_id)
            ))
            -- RLS User JWT
            OR (v_is_rls AND (
                (p_include_global AND d.tenant_id IS NULL AND knowledgebase.can_read_document(d.id))
                OR (d.tenant_id IS NOT NULL
                    AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                    AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
                    AND knowledgebase.can_read_document(d.id)
                )
            ))
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1
                FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id
                  AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT GREATEST(COALESCE(p_match_count, 5), 1)
    )
    SELECT
        candidates.id AS chunk_id,
        candidates.doc_id AS document_id,
        candidates.doc_title AS document_title,
        candidates.sec_title AS section_title,
        candidates.content AS chunk_text,
        candidates.f_path AS facet_path,
        candidates.c_meta AS metadata,
        candidates.v_score AS vector_score,
        NULL::REAL AS text_score,
        candidates.v_score AS hybrid_score,
        candidates.v_rank::INT AS vector_rank,
        NULL::INT AS text_rank
    FROM candidates
    ORDER BY candidates.v_score DESC, candidates.doc_title ASC;
END;
$$;

-- RLS-only variant (SECURITY INVOKER)
CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding_rls(
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL,
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
BEGIN
    IF p_query_embedding IS NULL THEN
        RAISE EXCEPTION 'Query embedding is required';
    END IF;

    BEGIN
        PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);
        PERFORM set_config('hnsw.ef_search', GREATEST(COALESCE(p_ef_search, COALESCE(p_match_count, 5) * 10), 40)::text, true);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    RETURN QUERY
    WITH candidates AS (
        SELECT
            c.id, d.id AS doc_id, d.title AS doc_title, ds.heading AS sec_title, c.content,
            c.metadata ->> 'facet_path' AS f_path, c.metadata AS c_meta,
            (1 - (c.embedding <=> p_query_embedding)) AS v_score,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_query_embedding) AS v_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.embedding IS NOT NULL
          AND (p_min_vector_similarity IS NULL OR (1 - (c.embedding <=> p_query_embedding)) >= p_min_vector_similarity)
          AND (
            (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
            OR (d.top_level_category IS DISTINCT FROM 'system')
          )
          AND (
            (p_include_global AND d.tenant_id IS NULL)
            OR (d.tenant_id IS NOT NULL
                AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
            )
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT GREATEST(COALESCE(p_match_count, 5), 1)
    )
    SELECT
        candidates.id, candidates.doc_id, candidates.doc_title, candidates.sec_title, candidates.content,
        candidates.f_path, candidates.c_meta, candidates.v_score, NULL::REAL,
        candidates.v_score, candidates.v_rank::INT, NULL::INT
    FROM candidates
    ORDER BY candidates.v_score DESC, candidates.doc_title ASC;
END;
$$;

-- Full-Text Search RPC
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text(
    p_kb_token TEXT,
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_fts_config TEXT DEFAULT 'english',
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_tenant_id UUID;
    v_token_meta JSONB;
    v_token_categories TEXT[];
    v_is_rls BOOLEAN;
    v_regconfig regconfig;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

    IF v_token_id IS NOT NULL THEN
        SELECT t.tenant_id, t.metadata INTO v_tenant_id, v_token_meta
        FROM knowledgebase.access_tokens t WHERE t.id = v_token_id;

        IF v_token_meta ? 'allowed_categories' THEN
            SELECT ARRAY(SELECT jsonb_array_elements_text(v_token_meta -> 'allowed_categories'))
            INTO v_token_categories;
        END IF;
    END IF;

    IF p_query IS NULL OR btrim(p_query) = '' THEN
        RAISE EXCEPTION 'Full-text query is required';
    END IF;

    BEGIN
        v_regconfig := COALESCE(NULLIF(btrim(p_fts_config), ''), 'english')::regconfig;
    EXCEPTION WHEN OTHERS THEN
        v_regconfig := 'simple'::regconfig;
    END;

    v_tsquery := websearch_to_tsquery(v_regconfig, p_query);
    IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
        v_tsquery := plainto_tsquery(v_regconfig, p_query);
    END IF;
    IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
        v_tsquery := plainto_tsquery('simple'::regconfig, p_query);
    END IF;

    RETURN QUERY
    WITH candidates AS (
        SELECT
            c.id,
            d.id AS doc_id,
            d.title AS doc_title,
            ds.heading AS sec_title,
            c.content,
            c.metadata ->> 'facet_path' AS f_path,
            c.metadata AS c_meta,
            ts_rank_cd(c.search_vector, v_tsquery) AS t_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, c.id ASC) AS t_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.search_vector @@ v_tsquery
          -- Category isolation
          AND (
            (v_token_id IS NOT NULL AND (
                v_token_categories IS NULL OR d.top_level_category = ANY(v_token_categories)
            ))
            OR (v_is_rls AND (
                (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
                OR (d.top_level_category IS DISTINCT FROM 'system')
            ))
            OR (v_token_id IS NULL AND NOT v_is_rls)
          )
          -- Tenant / Scope isolation
          AND (
            (v_token_id IS NULL AND NOT v_is_rls)
            OR (v_token_id IS NOT NULL AND (
                (v_tenant_id IS NULL AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id))
                OR (v_tenant_id IS NOT NULL AND d.tenant_id = v_tenant_id)
            ))
            OR (v_is_rls AND (
                (p_include_global AND d.tenant_id IS NULL AND knowledgebase.can_read_document(d.id))
                OR (d.tenant_id IS NOT NULL
                    AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                    AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
                    AND knowledgebase.can_read_document(d.id)
                )
            ))
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1
                FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id
                  AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, d.title ASC
        LIMIT GREATEST(COALESCE(p_match_count, 5), 1)
    )
    SELECT
        candidates.id AS chunk_id,
        candidates.doc_id AS document_id,
        candidates.doc_title AS document_title,
        candidates.sec_title AS section_title,
        candidates.content AS chunk_text,
        candidates.f_path AS facet_path,
        candidates.c_meta AS metadata,
        NULL::DOUBLE PRECISION AS vector_score,
        candidates.t_score AS text_score,
        candidates.t_score::DOUBLE PRECISION AS hybrid_score,
        NULL::INT AS vector_rank,
        candidates.t_rank::INT AS text_rank
    FROM candidates
    ORDER BY candidates.t_score DESC, candidates.doc_title ASC;
END;
$$;

-- Full-Text Search RLS-only RPC (SECURITY INVOKER)
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text_rls(
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_fts_config TEXT DEFAULT 'english',
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_tsquery tsquery;
    v_regconfig regconfig;
BEGIN
    IF p_query IS NULL OR btrim(p_query) = '' THEN
        RAISE EXCEPTION 'Full-text query is required';
    END IF;

    BEGIN
        v_regconfig := COALESCE(NULLIF(btrim(p_fts_config), ''), 'english')::regconfig;
    EXCEPTION WHEN OTHERS THEN
        v_regconfig := 'simple'::regconfig;
    END;

    v_tsquery := websearch_to_tsquery(v_regconfig, p_query);
    IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
        v_tsquery := plainto_tsquery(v_regconfig, p_query);
    END IF;
    IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
        v_tsquery := plainto_tsquery('simple'::regconfig, p_query);
    END IF;

    RETURN QUERY
    WITH candidates AS (
        SELECT
            c.id, d.id AS doc_id, d.title AS doc_title, ds.heading AS sec_title, c.content,
            c.metadata ->> 'facet_path' AS f_path, c.metadata AS c_meta,
            ts_rank_cd(c.search_vector, v_tsquery) AS t_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, c.id ASC) AS t_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.search_vector @@ v_tsquery
          AND (
            (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
            OR (d.top_level_category IS DISTINCT FROM 'system')
          )
          AND (
            (p_include_global AND d.tenant_id IS NULL)
            OR (d.tenant_id IS NOT NULL
                AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
            )
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, d.title ASC
        LIMIT GREATEST(COALESCE(p_match_count, 5), 1)
    )
    SELECT
        candidates.id, candidates.doc_id, candidates.doc_title, candidates.sec_title, candidates.content,
        candidates.f_path, candidates.c_meta, NULL::DOUBLE PRECISION, candidates.t_score,
        candidates.t_score::DOUBLE PRECISION, NULL::INT, candidates.t_rank::INT
    FROM candidates
    ORDER BY candidates.t_score DESC, candidates.doc_title ASC;
END;
$$;

-- Hybrid Search RPC (Two-Stage Vector + FTS Candidates with RRF Fusion)
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_hybrid(
    p_kb_token TEXT,
    p_query TEXT,
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_candidate_count INT DEFAULT NULL,
    p_rrf_k INT DEFAULT 60,
    p_vector_weight DOUBLE PRECISION DEFAULT 1.0,
    p_text_weight DOUBLE PRECISION DEFAULT 1.0,
    p_fts_config TEXT DEFAULT 'english',
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL,
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_tenant_id UUID;
    v_token_meta JSONB;
    v_token_categories TEXT[];
    v_is_rls BOOLEAN;
    v_candidate_count INT;
    v_rrf_k INT;
    v_vector_weight DOUBLE PRECISION;
    v_text_weight DOUBLE PRECISION;
    v_regconfig regconfig;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

    IF v_token_id IS NOT NULL THEN
        SELECT t.tenant_id, t.metadata INTO v_tenant_id, v_token_meta
        FROM knowledgebase.access_tokens t WHERE t.id = v_token_id;

        IF v_token_meta ? 'allowed_categories' THEN
            SELECT ARRAY(SELECT jsonb_array_elements_text(v_token_meta -> 'allowed_categories'))
            INTO v_token_categories;
        END IF;
    END IF;

    IF p_query_embedding IS NULL AND (p_query IS NULL OR btrim(p_query) = '') THEN
        RAISE EXCEPTION 'Query text or query embedding is required';
    END IF;

    v_candidate_count := LEAST(COALESCE(p_candidate_count, GREATEST(COALESCE(p_match_count, 5) * 10, 50)), 500);
    v_rrf_k := GREATEST(COALESCE(p_rrf_k, 60), 1);
    v_vector_weight := GREATEST(COALESCE(p_vector_weight, 1.0), 0.0);
    v_text_weight := GREATEST(COALESCE(p_text_weight, 1.0), 0.0);
    BEGIN
        v_regconfig := COALESCE(NULLIF(btrim(p_fts_config), ''), 'english')::regconfig;
    EXCEPTION WHEN OTHERS THEN
        v_regconfig := 'simple'::regconfig;
    END;

    BEGIN
        PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);
        PERFORM set_config('hnsw.ef_search', GREATEST(COALESCE(p_ef_search, v_candidate_count * 2), 40)::text, true);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    IF p_query IS NOT NULL AND btrim(p_query) <> '' THEN
        v_tsquery := websearch_to_tsquery(v_regconfig, p_query);
        IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
            v_tsquery := plainto_tsquery(v_regconfig, p_query);
        END IF;
        IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
            v_tsquery := plainto_tsquery('simple'::regconfig, p_query);
        END IF;
    END IF;

    RETURN QUERY
    WITH vector_candidates AS (
        SELECT
            c.id AS chunk_id,
            (1 - (c.embedding <=> p_query_embedding)) AS vector_similarity,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_query_embedding) AS vector_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE p_query_embedding IS NOT NULL
          AND c.embedding IS NOT NULL
          AND (p_min_vector_similarity IS NULL OR (1 - (c.embedding <=> p_query_embedding)) >= p_min_vector_similarity)
          AND (
            (v_token_id IS NOT NULL AND (
                v_token_categories IS NULL OR d.top_level_category = ANY(v_token_categories)
            ))
            OR (v_is_rls AND (
                (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
                OR (d.top_level_category IS DISTINCT FROM 'system')
            ))
            OR (v_token_id IS NULL AND NOT v_is_rls)
          )
          AND (
            (v_token_id IS NULL AND NOT v_is_rls)
            OR (v_token_id IS NOT NULL AND (
                (v_tenant_id IS NULL AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id))
                OR (v_tenant_id IS NOT NULL AND d.tenant_id = v_tenant_id)
            ))
            OR (v_is_rls AND (
                (p_include_global AND d.tenant_id IS NULL AND knowledgebase.can_read_document(d.id))
                OR (d.tenant_id IS NOT NULL
                    AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                    AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
                    AND knowledgebase.can_read_document(d.id)
                )
            ))
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1
                FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id
                  AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT v_candidate_count
    ),
    fts_candidates AS (
        SELECT
            c.id AS chunk_id,
            ts_rank_cd(c.search_vector, v_tsquery) AS fts_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, c.id ASC) AS text_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE v_tsquery IS NOT NULL
          AND c.search_vector @@ v_tsquery
          AND (
            (v_token_id IS NOT NULL AND (
                v_token_categories IS NULL OR d.top_level_category = ANY(v_token_categories)
            ))
            OR (v_is_rls AND (
                (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
                OR (d.top_level_category IS DISTINCT FROM 'system')
            ))
            OR (v_token_id IS NULL AND NOT v_is_rls)
          )
          AND (
            (v_token_id IS NULL AND NOT v_is_rls)
            OR (v_token_id IS NOT NULL AND (
                (v_tenant_id IS NULL AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id))
                OR (v_tenant_id IS NOT NULL AND d.tenant_id = v_tenant_id)
            ))
            OR (v_is_rls AND (
                (p_include_global AND d.tenant_id IS NULL AND knowledgebase.can_read_document(d.id))
                OR (d.tenant_id IS NOT NULL
                    AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                    AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
                    AND knowledgebase.can_read_document(d.id)
                )
            ))
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1
                FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id
                  AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC
        LIMIT v_candidate_count
    ),
    fused AS (
        SELECT
            COALESCE(vc.chunk_id, fc.chunk_id) AS chunk_id,
            vc.vector_similarity,
            fc.fts_score AS text_score,
            vc.vector_rank::INT AS vector_rank,
            fc.text_rank::INT AS text_rank,
            (
                (CASE WHEN vc.vector_rank IS NOT NULL THEN (v_vector_weight / (v_rrf_k + vc.vector_rank)) ELSE 0.0 END)
                +
                (CASE WHEN fc.text_rank IS NOT NULL THEN (v_text_weight / (v_rrf_k + fc.text_rank)) ELSE 0.0 END)
            ) AS rrf_score
        FROM vector_candidates vc
        FULL OUTER JOIN fts_candidates fc ON vc.chunk_id = fc.chunk_id
    )
    SELECT
        c.id AS chunk_id,
        d.id AS document_id,
        d.title AS document_title,
        ds.heading AS section_title,
        c.content AS chunk_text,
        c.metadata ->> 'facet_path' AS facet_path,
        c.metadata AS metadata,
        fused.vector_similarity AS vector_score,
        fused.text_score AS text_score,
        fused.rrf_score AS hybrid_score,
        fused.vector_rank,
        fused.text_rank
    FROM fused
    JOIN knowledgebase.chunks c ON c.id = fused.chunk_id
    JOIN knowledgebase.documents d ON d.id = c.document_id
    LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
    ORDER BY fused.rrf_score DESC, d.title ASC
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

-- Hybrid Search RLS-only RPC (SECURITY INVOKER)
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_hybrid_rls(
    p_query TEXT,
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_candidate_count INT DEFAULT NULL,
    p_rrf_k INT DEFAULT 60,
    p_vector_weight DOUBLE PRECISION DEFAULT 1.0,
    p_text_weight DOUBLE PRECISION DEFAULT 1.0,
    p_fts_config TEXT DEFAULT 'english',
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL,
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    document_title TEXT,
    section_title TEXT,
    chunk_text TEXT,
    facet_path TEXT,
    metadata JSONB,
    vector_score DOUBLE PRECISION,
    text_score REAL,
    hybrid_score DOUBLE PRECISION,
    vector_rank INT,
    text_rank INT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_tsquery tsquery;
    v_candidate_count INT;
    v_rrf_k INT;
    v_vector_weight DOUBLE PRECISION;
    v_text_weight DOUBLE PRECISION;
    v_regconfig regconfig;
BEGIN
    IF p_query_embedding IS NULL AND (p_query IS NULL OR btrim(p_query) = '') THEN
        RAISE EXCEPTION 'Query text or query embedding is required';
    END IF;

    v_candidate_count := LEAST(COALESCE(p_candidate_count, GREATEST(COALESCE(p_match_count, 5) * 10, 50)), 500);
    v_rrf_k := GREATEST(COALESCE(p_rrf_k, 60), 1);
    v_vector_weight := GREATEST(COALESCE(p_vector_weight, 1.0), 0.0);
    v_text_weight := GREATEST(COALESCE(p_text_weight, 1.0), 0.0);
    BEGIN
        v_regconfig := COALESCE(NULLIF(btrim(p_fts_config), ''), 'english')::regconfig;
    EXCEPTION WHEN OTHERS THEN
        v_regconfig := 'simple'::regconfig;
    END;

    BEGIN
        PERFORM set_config('hnsw.iterative_scan', 'relaxed_order', true);
        PERFORM set_config('hnsw.ef_search', GREATEST(COALESCE(p_ef_search, v_candidate_count * 2), 40)::text, true);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;

    IF p_query IS NOT NULL AND btrim(p_query) <> '' THEN
        v_tsquery := websearch_to_tsquery(v_regconfig, p_query);
        IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
            v_tsquery := plainto_tsquery(v_regconfig, p_query);
        END IF;
        IF v_tsquery IS NULL OR length(v_tsquery::text) = 0 THEN
            v_tsquery := plainto_tsquery('simple'::regconfig, p_query);
        END IF;
    END IF;

    RETURN QUERY
    WITH vector_candidates AS (
        SELECT
            c.id AS chunk_id,
            (1 - (c.embedding <=> p_query_embedding)) AS vector_similarity,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> p_query_embedding) AS vector_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE p_query_embedding IS NOT NULL
          AND c.embedding IS NOT NULL
          AND (p_min_vector_similarity IS NULL OR (1 - (c.embedding <=> p_query_embedding)) >= p_min_vector_similarity)
          AND (
            (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
            OR (d.top_level_category IS DISTINCT FROM 'system')
          )
          AND (
            (p_include_global AND d.tenant_id IS NULL)
            OR (d.tenant_id IS NOT NULL
                AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
            )
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY c.embedding <=> p_query_embedding
        LIMIT v_candidate_count
    ),
    fts_candidates AS (
        SELECT
            c.id AS chunk_id,
            ts_rank_cd(c.search_vector, v_tsquery) AS fts_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, c.id ASC) AS text_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE v_tsquery IS NOT NULL
          AND c.search_vector @@ v_tsquery
          AND (
            (p_allowed_categories IS NOT NULL AND d.top_level_category = ANY(p_allowed_categories))
            OR (d.top_level_category IS DISTINCT FROM 'system')
          )
          AND (
            (p_include_global AND d.tenant_id IS NULL)
            OR (d.tenant_id IS NOT NULL
                AND (p_tenant_id IS NULL OR d.tenant_id = p_tenant_id)
                AND (p_scope_id IS NULL OR d.scope_id = p_scope_id)
            )
          )
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC
        LIMIT v_candidate_count
    ),
    fused AS (
        SELECT
            COALESCE(vc.chunk_id, fc.chunk_id) AS chunk_id,
            vc.vector_similarity,
            fc.fts_score AS text_score,
            vc.vector_rank::INT AS vector_rank,
            fc.text_rank::INT AS text_rank,
            (
                (CASE WHEN vc.vector_rank IS NOT NULL THEN (v_vector_weight / (v_rrf_k + vc.vector_rank)) ELSE 0.0 END)
                +
                (CASE WHEN fc.text_rank IS NOT NULL THEN (v_text_weight / (v_rrf_k + fc.text_rank)) ELSE 0.0 END)
            ) AS rrf_score
        FROM vector_candidates vc
        FULL OUTER JOIN fts_candidates fc ON vc.chunk_id = fc.chunk_id
    )
    SELECT
        c.id AS chunk_id,
        d.id AS document_id,
        d.title AS document_title,
        ds.heading AS section_title,
        c.content AS chunk_text,
        c.metadata ->> 'facet_path' AS facet_path,
        c.metadata AS metadata,
        fused.vector_similarity AS vector_score,
        fused.text_score AS text_score,
        fused.rrf_score AS hybrid_score,
        fused.vector_rank,
        fused.text_rank
    FROM fused
    JOIN knowledgebase.chunks c ON c.id = fused.chunk_id
    JOIN knowledgebase.documents d ON d.id = c.document_id
    LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
    ORDER BY fused.rrf_score DESC, d.title ASC
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

-- Ensure chunks search vector trigger fires on all updates including cascaded updated_at
DROP TRIGGER IF EXISTS trigger_kb_chunks_search_vector ON knowledgebase.chunks;
CREATE TRIGGER trigger_kb_chunks_search_vector
BEFORE INSERT OR UPDATE
ON knowledgebase.chunks
FOR EACH ROW EXECUTE FUNCTION knowledgebase.chunks_search_vector_trigger();

