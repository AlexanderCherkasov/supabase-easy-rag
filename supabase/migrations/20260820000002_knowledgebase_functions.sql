-- ============================================================================
-- Supabase Easy RAG: PostgreSQL RPC Functions
-- Security Token Audit + Vector, Full-Text, and Hybrid Search RPCs
-- Supports TWO auth modes (see 01_schema.sql RLS guide):
--   1) Service-role / Machine token (p_kb_token) via assert_retrieval_access
--   2) Fine-grained RLS via auth.uid() — no token needed (Supabase RAG with Permissions)
-- ============================================================================

-- Helper function: Hash Access Token (SHA-256)
CREATE OR REPLACE FUNCTION knowledgebase.hash_access_token(p_token TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT encode(sha256(p_token::bytea), 'hex');
$$;

-- Security Assertion Function: Validates Token & Audit Trail
CREATE OR REPLACE FUNCTION knowledgebase.assert_retrieval_access(p_kb_token TEXT)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_access_token_id UUID;
    v_role TEXT;
BEGIN
    v_role := COALESCE(current_setting('request.jwt.claim.role', true), 'service_role');

    IF (p_kb_token IS NULL OR btrim(p_kb_token) = '') AND auth.uid() IS NOT NULL THEN
        RETURN NULL;
    END IF;

    IF p_kb_token IS NULL OR btrim(p_kb_token) = '' THEN
        INSERT INTO knowledgebase.access_token_audit (access_token_id, event_type, user_id, metadata)
        VALUES (
            NULL,
            'failed_validation',
            auth.uid(),
            jsonb_build_object(
                'reason', 'missing_token',
                'actor_role', v_role
            )
        );
        RAISE EXCEPTION 'Knowledgebase token is required (or authenticate via Supabase Auth for RLS)';
    END IF;

    SELECT id
    INTO v_access_token_id
    FROM knowledgebase.access_tokens
    WHERE token_hash = knowledgebase.hash_access_token(p_kb_token)
      AND is_active = TRUE
      AND (expires_at IS NULL OR expires_at > NOW())
    LIMIT 1;

    IF v_access_token_id IS NULL THEN
        INSERT INTO knowledgebase.access_token_audit (access_token_id, event_type, user_id, metadata)
        VALUES (
            NULL,
            'failed_validation',
            auth.uid(),
            jsonb_build_object(
                'reason', 'invalid_token',
                'actor_role', v_role
            )
        );
        RAISE EXCEPTION 'Invalid knowledgebase token';
    END IF;

    UPDATE knowledgebase.access_tokens
    SET last_used_at = NOW(), updated_at = NOW()
    WHERE id = v_access_token_id;

    INSERT INTO knowledgebase.access_token_audit (access_token_id, event_type, user_id, metadata)
    VALUES (
        v_access_token_id,
        'used',
        auth.uid(),
        jsonb_build_object(
            'actor_role', v_role
        )
    );

    RETURN v_access_token_id;
END;
$$;

-- Helper: check if request is RLS-authenticated
CREATE OR REPLACE FUNCTION knowledgebase.is_rls_authenticated()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT auth.uid() IS NOT NULL;
$$;

-- 1. Vector Search RPC
CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding(
    p_kb_token TEXT,
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL
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
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_token_id UUID;
    v_is_rls BOOLEAN;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

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
          AND (
            NOT v_is_rls
            OR d.owner_id IS NULL
            OR d.owner_id = auth.uid()
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_owners do2
                WHERE do2.document_id = d.id AND do2.owner_id = auth.uid()
            )
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

CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding_rls(
    p_query_embedding VECTOR,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_min_vector_similarity DOUBLE PRECISION DEFAULT NULL,
    p_ef_search INT DEFAULT NULL
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
SET search_path = knowledgebase, public
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

-- 2. Full-Text Search (FTS) RPC
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text(
    p_kb_token TEXT,
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_fts_config TEXT DEFAULT 'english'
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
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_is_rls BOOLEAN;
    v_regconfig regconfig;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

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
            ts_rank(c.search_vector, v_tsquery) AS t_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, c.id ASC) AS t_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.search_vector @@ v_tsquery
          AND (
            NOT v_is_rls
            OR d.owner_id IS NULL
            OR d.owner_id = auth.uid()
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_owners do2
                WHERE do2.document_id = d.id AND do2.owner_id = auth.uid()
            )
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
        ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, d.title ASC
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

CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text_rls(
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL,
    p_fts_config TEXT DEFAULT 'english'
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
SET search_path = knowledgebase, public
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
            ts_rank(c.search_vector, v_tsquery) AS t_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, c.id ASC) AS t_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE c.search_vector @@ v_tsquery
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, d.title ASC
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

-- 3. Hybrid Search RPC
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
    p_ef_search INT DEFAULT NULL
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
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_is_rls BOOLEAN;
    v_candidate_count INT;
    v_rrf_k INT;
    v_vector_weight DOUBLE PRECISION;
    v_text_weight DOUBLE PRECISION;
    v_regconfig regconfig;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

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
            NOT v_is_rls
            OR d.owner_id IS NULL
            OR d.owner_id = auth.uid()
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_owners do2
                WHERE do2.document_id = d.id AND do2.owner_id = auth.uid()
            )
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
            ts_rank(c.search_vector, v_tsquery) AS fts_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, c.id ASC) AS text_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE v_tsquery IS NOT NULL
          AND c.search_vector @@ v_tsquery
          AND (
            NOT v_is_rls
            OR d.owner_id IS NULL
            OR d.owner_id = auth.uid()
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_owners do2
                WHERE do2.document_id = d.id AND do2.owner_id = auth.uid()
            )
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
        ORDER BY ts_rank(c.search_vector, v_tsquery) DESC
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
    p_ef_search INT DEFAULT NULL
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
SET search_path = knowledgebase, public
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
            ts_rank(c.search_vector, v_tsquery) AS fts_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_vector, v_tsquery) DESC, c.id ASC) AS text_rank
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        WHERE v_tsquery IS NOT NULL
          AND c.search_vector @@ v_tsquery
          AND (
            p_facet_keys IS NULL
            OR EXISTS (
                SELECT 1 FROM knowledgebase.document_facets df
                JOIN knowledgebase.facets f ON f.id = df.facet_id
                WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
            )
          )
        ORDER BY ts_rank(c.search_vector, v_tsquery) DESC
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

-- 4. Navigation Facets RPC
CREATE OR REPLACE FUNCTION knowledgebase.get_navigation_facets(
    p_kb_token TEXT,
    p_facet_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    facet_id UUID,
    facet_type TEXT,
    facet_key TEXT,
    label TEXT,
    parent_facet_id UUID,
    sort_order INT,
    document_count BIGINT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_token_id UUID;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    RETURN QUERY
    SELECT
        f.id,
        f.facet_type,
        f.facet_key,
        f.label,
        f.parent_facet_id,
        f.sort_order,
        COUNT(df.document_id) AS document_count
    FROM knowledgebase.facets f
    LEFT JOIN knowledgebase.document_facets df ON df.facet_id = f.id
    WHERE p_facet_type IS NULL OR f.facet_type = p_facet_type
    GROUP BY f.id, f.facet_type, f.facet_key, f.label, f.parent_facet_id, f.sort_order
    ORDER BY f.facet_type ASC, f.sort_order ASC, f.label ASC;
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.get_navigation_facets_rls(
    p_facet_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    facet_id UUID,
    facet_type TEXT,
    facet_key TEXT,
    label TEXT,
    parent_facet_id UUID,
    sort_order INT,
    document_count BIGINT
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = knowledgebase, public
AS $$
BEGIN
    RETURN QUERY
    SELECT f.id, f.facet_type, f.facet_key, f.label, f.parent_facet_id, f.sort_order, COUNT(df.document_id)
    FROM knowledgebase.facets f LEFT JOIN knowledgebase.document_facets df ON df.facet_id = f.id
    WHERE p_facet_type IS NULL OR f.facet_type = p_facet_type
    GROUP BY f.id, f.facet_type, f.facet_key, f.label, f.parent_facet_id, f.sort_order
    ORDER BY f.facet_type ASC, f.sort_order ASC, f.label ASC;
END;
$$;

-- Grants
REVOKE ALL ON FUNCTION knowledgebase.hash_access_token(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION knowledgebase.assert_retrieval_access(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION knowledgebase.is_rls_authenticated() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION knowledgebase.hash_access_token(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.assert_retrieval_access(TEXT) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.is_rls_authenticated() TO authenticated, service_role;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA knowledgebase TO authenticated, service_role;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA knowledgebase FROM anon;
GRANT EXECUTE ON FUNCTION knowledgebase.get_navigation_facets_rls(TEXT) TO anon;
