-- ============================================================================
-- Migration: 20260824000005_dynamic_vector_dimensions_and_model_scoping.sql
-- 1. Alters chunks.embedding from VECTOR(1536) to unconstrained VECTOR
-- 2. Adds tenant_id to chunks with auto-sync trigger from parent document
-- 3. Creates partial HNSW index pattern for (tenant_id, dimension) support
-- 4. Provides native index management functions (ensure, drop, list) per tenant & dimension
-- 5. Upgrades search RPCs with vector_dims matching and optional model_name filter
-- ============================================================================

-- 1. Alter chunks.embedding to unconstrained VECTOR & add tenant_id
ALTER TABLE knowledgebase.chunks
    ALTER COLUMN embedding TYPE VECTOR,
    ADD COLUMN IF NOT EXISTS tenant_id UUID;

CREATE INDEX IF NOT EXISTS idx_kb_chunks_tenant
    ON knowledgebase.chunks(tenant_id);

-- Populate existing chunks
UPDATE knowledgebase.chunks c
SET tenant_id = d.tenant_id
FROM knowledgebase.documents d
WHERE d.id = c.document_id AND c.tenant_id IS DISTINCT FROM d.tenant_id;

-- Trigger to auto-sync tenant_id from document on chunk insert/update
CREATE OR REPLACE FUNCTION knowledgebase.chunks_populate_tenant_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.tenant_id IS NULL THEN
        SELECT tenant_id INTO NEW.tenant_id
        FROM knowledgebase.documents
        WHERE id = NEW.document_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_kb_chunks_populate_tenant ON knowledgebase.chunks;
CREATE TRIGGER trigger_kb_chunks_populate_tenant
BEFORE INSERT OR UPDATE OF document_id ON knowledgebase.chunks
FOR EACH ROW EXECUTE FUNCTION knowledgebase.chunks_populate_tenant_trigger();

-- Drop legacy fixed-dimension indexes if present
DROP INDEX IF EXISTS knowledgebase.idx_kb_chunks_embedding;
DROP INDEX IF EXISTS knowledgebase.idx_kb_chunks_hnsw_1536;
DROP INDEX IF EXISTS knowledgebase.idx_kb_chunks_hnsw_1024;

-- Create default global partial indexes for common dimensions (1536 for OpenAI, 1024 for Qwen3)
CREATE INDEX IF NOT EXISTS idx_kb_chunks_hnsw_g_1536
    ON knowledgebase.chunks USING hnsw (((embedding)::vector(1536)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE tenant_id IS NULL AND vector_dims(embedding) = 1536;

CREATE INDEX IF NOT EXISTS idx_kb_chunks_hnsw_g_1024
    ON knowledgebase.chunks USING hnsw (((embedding)::vector(1024)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE tenant_id IS NULL AND vector_dims(embedding) = 1024;

-- 2. Dynamic Vector Index Management Functions (Tenant + Dimension)
CREATE OR REPLACE FUNCTION knowledgebase.ensure_vector_index(
    p_dimension INT,
    p_tenant_id UUID DEFAULT NULL,
    p_m INT DEFAULT 16,
    p_ef_construction INT DEFAULT 64
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_idx_name TEXT;
    v_sql TEXT;
BEGIN
    IF p_dimension IS NULL OR p_dimension <= 0 THEN
        RAISE EXCEPTION 'Dimension must be a positive integer';
    END IF;

    IF p_tenant_id IS NULL THEN
        v_idx_name := 'idx_kb_chunks_hnsw_g_' || p_dimension;
        v_sql := format(
            'CREATE INDEX %I ON knowledgebase.chunks USING hnsw (((embedding)::vector(%s)) vector_cosine_ops) WITH (m = %s, ef_construction = %s) WHERE tenant_id IS NULL AND vector_dims(embedding) = %s',
            v_idx_name,
            p_dimension,
            p_m,
            p_ef_construction,
            p_dimension
        );
    ELSE
        v_idx_name := 'idx_kb_chunks_hnsw_t_' || replace(p_tenant_id::text, '-', '_') || '_' || p_dimension;
        v_sql := format(
            'CREATE INDEX %I ON knowledgebase.chunks USING hnsw (((embedding)::vector(%s)) vector_cosine_ops) WITH (m = %s, ef_construction = %s) WHERE tenant_id = %L AND vector_dims(embedding) = %s',
            v_idx_name,
            p_dimension,
            p_m,
            p_ef_construction,
            p_tenant_id,
            p_dimension
        );
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'knowledgebase' AND c.relname = v_idx_name
    ) THEN
        RETURN 'Index ' || v_idx_name || ' already exists';
    END IF;

    EXECUTE v_sql;
    RETURN 'Created index ' || v_idx_name;
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.drop_vector_index(
    p_dimension INT,
    p_tenant_id UUID DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
DECLARE
    v_idx_name TEXT;
    v_sql TEXT;
BEGIN
    IF p_dimension IS NULL OR p_dimension <= 0 THEN
        RAISE EXCEPTION 'Dimension must be a positive integer';
    END IF;

    IF p_tenant_id IS NULL THEN
        v_idx_name := 'idx_kb_chunks_hnsw_g_' || p_dimension;
    ELSE
        v_idx_name := 'idx_kb_chunks_hnsw_t_' || replace(p_tenant_id::text, '-', '_') || '_' || p_dimension;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'knowledgebase' AND c.relname = v_idx_name
    ) THEN
        RETURN 'Index ' || v_idx_name || ' does not exist';
    END IF;

    v_sql := format('DROP INDEX IF EXISTS knowledgebase.%I', v_idx_name);
    EXECUTE v_sql;
    RETURN 'Dropped index ' || v_idx_name;
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.list_vector_indexes(
    p_tenant_id UUID DEFAULT NULL
)
RETURNS TABLE (
    index_name TEXT,
    tenant_id UUID,
    dimension INT,
    is_global BOOLEAN,
    index_def TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, knowledgebase
AS $$
BEGIN
    RETURN QUERY
    WITH raw_indexes AS (
        SELECT
            c.relname::TEXT AS idx_name,
            CASE
                WHEN c.relname LIKE 'idx_kb_chunks_hnsw_t_%' THEN
                    replace(substring(c.relname from 'idx_kb_chunks_hnsw_t_([0-9a-fA-F_]+)_[0-9]+'), '_', '-')::UUID
                ELSE NULL
            END AS t_id,
            NULLIF(substring(c.relname from 'idx_kb_chunks_hnsw_.*_([0-9]+)$'), '')::INT AS dim,
            (c.relname LIKE 'idx_kb_chunks_hnsw_g_%' OR c.relname NOT LIKE 'idx_kb_chunks_hnsw_t_%') AS is_glob,
            pg_get_indexdef(c.oid)::TEXT AS def
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'knowledgebase'
          AND c.relname LIKE 'idx_kb_chunks_hnsw_%'
    )
    SELECT
        raw_indexes.idx_name AS index_name,
        raw_indexes.t_id AS tenant_id,
        raw_indexes.dim AS dimension,
        raw_indexes.is_glob AS is_global,
        raw_indexes.def AS index_def
    FROM raw_indexes
    WHERE (p_tenant_id IS NULL OR raw_indexes.t_id = p_tenant_id)
    ORDER BY raw_indexes.idx_name;
END;
$$;

GRANT EXECUTE ON FUNCTION knowledgebase.ensure_vector_index(INT, UUID, INT, INT) TO service_role, authenticated;
GRANT EXECUTE ON FUNCTION knowledgebase.drop_vector_index(INT, UUID) TO service_role, authenticated;
GRANT EXECUTE ON FUNCTION knowledgebase.list_vector_indexes(UUID) TO service_role, authenticated, anon;

-- 3. Upgraded Search RPCs with dynamic dimension safety & model scoping
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding(text, vector, integer, text[], double precision, integer, uuid, uuid, boolean, text[]);
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding(text, vector, integer, text[], double precision, integer, uuid, uuid, boolean, text[], text);
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding_rls(vector, integer, text[], double precision, integer, uuid, uuid, boolean, text[]);
DROP FUNCTION IF EXISTS knowledgebase.match_chunks_by_embedding_rls(vector, integer, text[], double precision, integer, uuid, uuid, boolean, text[], text);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid(text, text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer, uuid, uuid, boolean, text[]);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid(text, text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer, uuid, uuid, boolean, text[], text);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid_rls(text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer, uuid, uuid, boolean, text[]);
DROP FUNCTION IF EXISTS knowledgebase.search_chunks_hybrid_rls(text, vector, integer, text[], integer, integer, double precision, double precision, text, double precision, integer, uuid, uuid, boolean, text[], text);

-- 3.1 Vector Search RPC (Service Role / Token)
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
    p_allowed_categories TEXT[] DEFAULT NULL,
    p_model_name TEXT DEFAULT NULL
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
    v_query_dim INT;
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

    v_query_dim := vector_dims(p_query_embedding);

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
          AND vector_dims(c.embedding) = v_query_dim
          AND (p_model_name IS NULL OR (c.metadata ->> 'model') = p_model_name)
          AND (p_min_vector_similarity IS NULL OR (1 - (c.embedding <=> p_query_embedding)) >= p_min_vector_similarity)
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

-- 3.2 Vector Search RLS-only RPC (SECURITY INVOKER)
CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding_rls(
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL,
    p_tenant_id UUID DEFAULT NULL,
    p_scope_id UUID DEFAULT NULL,
    p_include_global BOOLEAN DEFAULT TRUE,
    p_allowed_categories TEXT[] DEFAULT NULL,
    p_model_name TEXT DEFAULT NULL
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
    v_query_dim INT;
BEGIN
    IF p_query_embedding IS NULL THEN
        RAISE EXCEPTION 'Query embedding is required';
    END IF;

    v_query_dim := vector_dims(p_query_embedding);

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
          AND vector_dims(c.embedding) = v_query_dim
          AND (p_model_name IS NULL OR (c.metadata ->> 'model') = p_model_name)
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

-- 3.3 Hybrid Search RPC (Service Role / Token)
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
    p_allowed_categories TEXT[] DEFAULT NULL,
    p_model_name TEXT DEFAULT NULL
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
    v_query_dim INT;
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

    IF p_query_embedding IS NOT NULL THEN
        v_query_dim := vector_dims(p_query_embedding);
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
          AND vector_dims(c.embedding) = v_query_dim
          AND (p_model_name IS NULL OR (c.metadata ->> 'model') = p_model_name)
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

-- 3.4 Hybrid Search RLS-only RPC (SECURITY INVOKER)
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
    p_allowed_categories TEXT[] DEFAULT NULL,
    p_model_name TEXT DEFAULT NULL
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
    v_query_dim INT;
BEGIN
    IF p_query_embedding IS NULL AND (p_query IS NULL OR btrim(p_query) = '') THEN
        RAISE EXCEPTION 'Query text or query embedding is required';
    END IF;

    IF p_query_embedding IS NOT NULL THEN
        v_query_dim := vector_dims(p_query_embedding);
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
          AND vector_dims(c.embedding) = v_query_dim
          AND (p_model_name IS NULL OR (c.metadata ->> 'model') = p_model_name)
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
