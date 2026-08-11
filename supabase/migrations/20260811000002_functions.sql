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
    SELECT encode(extensions.digest(p_token, 'sha256'), 'hex');
$$;

-- Security Assertion Function: Validates Token & Audit Trail
-- If p_kb_token is null/empty but auth.uid() exists -> RLS mode, no token check.
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

    -- RLS mode: no token, but authenticated user via JWT -> skip token check
    -- Caller relies on RLS policies (documents.owner_id = auth.uid())
    IF (p_kb_token IS NULL OR btrim(p_kb_token) = '') AND auth.uid() IS NOT NULL THEN
        RETURN NULL; -- signal RLS path
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

-- Helper: check if request is RLS-authenticated (auth.uid() exists) vs token
CREATE OR REPLACE FUNCTION knowledgebase.is_rls_authenticated()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT auth.uid() IS NOT NULL;
$$;

-- ============================================================================
-- 1. Vector Search RPC — supports token OR RLS
-- ============================================================================
CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding(
    p_kb_token TEXT,
    p_query_embedding VECTOR(1536),
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
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

    RETURN QUERY
    SELECT
        c.id,
        d.id,
        d.title,
        ds.heading,
        c.content,
        c.metadata ->> 'facet_path',
        c.metadata,
        1 - (c.embedding <=> p_query_embedding) AS vector_score,
        NULL::REAL AS text_score,
        1 - (c.embedding <=> p_query_embedding) AS hybrid_score
    FROM knowledgebase.chunks c
    JOIN knowledgebase.documents d ON d.id = c.document_id
    LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
    WHERE c.embedding IS NOT NULL
      -- RLS enforcement when in RLS mode (explicit filter, since SECURITY DEFINER bypasses RLS)
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
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

-- RLS-only variant (SECURITY INVOKER — respects RLS natively, no token needed)
CREATE OR REPLACE FUNCTION knowledgebase.match_chunks_by_embedding_rls(
    p_query_embedding VECTOR(1536),
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = knowledgebase, public
AS $$
BEGIN
    IF p_query_embedding IS NULL THEN
        RAISE EXCEPTION 'Query embedding is required';
    END IF;
    -- RLS policies on chunks/documents enforce access automatically
    RETURN QUERY
    SELECT
        c.id, d.id, d.title, ds.heading, c.content,
        c.metadata ->> 'facet_path', c.metadata,
        1 - (c.embedding <=> p_query_embedding), NULL::REAL,
        1 - (c.embedding <=> p_query_embedding)
    FROM knowledgebase.chunks c
    JOIN knowledgebase.documents d ON d.id = c.document_id
    LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
    WHERE c.embedding IS NOT NULL
      AND (
        p_facet_keys IS NULL
        OR EXISTS (
            SELECT 1 FROM knowledgebase.document_facets df
            JOIN knowledgebase.facets f ON f.id = df.facet_id
            WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)
        )
      )
    ORDER BY c.embedding <=> p_query_embedding
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

-- ============================================================================
-- 2. Full-Text Search (FTS) RPC — supports token OR RLS
-- ============================================================================
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text(
    p_kb_token TEXT,
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_is_rls BOOLEAN;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

    IF p_query IS NULL OR btrim(p_query) = '' THEN
        RAISE EXCEPTION 'Full-text query is required';
    END IF;

    v_tsquery := websearch_to_tsquery('english', p_query);

    RETURN QUERY
    SELECT
        c.id,
        d.id,
        d.title,
        ds.heading,
        c.content,
        c.metadata ->> 'facet_path',
        c.metadata,
        NULL::DOUBLE PRECISION AS vector_score,
        ts_rank_cd(c.search_vector, v_tsquery) AS text_score,
        ts_rank_cd(c.search_vector, v_tsquery)::DOUBLE PRECISION AS hybrid_score
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
    ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, d.title ASC
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_full_text_rls(
    p_query TEXT,
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
BEGIN
    IF p_query IS NULL OR btrim(p_query) = '' THEN
        RAISE EXCEPTION 'Full-text query is required';
    END IF;
    v_tsquery := websearch_to_tsquery('english', p_query);
    RETURN QUERY
    SELECT c.id, d.id, d.title, ds.heading, c.content,
           c.metadata ->> 'facet_path', c.metadata,
           NULL::DOUBLE PRECISION, ts_rank_cd(c.search_vector, v_tsquery),
           ts_rank_cd(c.search_vector, v_tsquery)::DOUBLE PRECISION
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
    ORDER BY ts_rank_cd(c.search_vector, v_tsquery) DESC, d.title ASC
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

-- ============================================================================
-- 3. Hybrid Search RPC (Weighted Vector + FTS + Title Boost) — token OR RLS
-- ============================================================================
CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_hybrid(
    p_kb_token TEXT,
    p_query TEXT,
    p_query_embedding VECTOR(1536),
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
    v_token_id UUID;
    v_is_rls BOOLEAN;
BEGIN
    v_token_id := knowledgebase.assert_retrieval_access(p_kb_token);
    v_is_rls := (v_token_id IS NULL AND knowledgebase.is_rls_authenticated());

    IF p_query_embedding IS NULL AND (p_query IS NULL OR btrim(p_query) = '') THEN
        RAISE EXCEPTION 'Query text or query embedding is required';
    END IF;

    IF p_query IS NOT NULL AND btrim(p_query) <> '' THEN
        v_tsquery := websearch_to_tsquery('english', p_query);
    END IF;

    RETURN QUERY
    WITH scored AS (
        SELECT
            c.id AS chunk_id,
            d.id AS document_id,
            d.title AS document_title,
            ds.heading AS section_title,
            c.content AS chunk_text,
            c.metadata ->> 'facet_path' AS facet_path,
            c.metadata AS metadata,
            CASE
                WHEN p_query_embedding IS NOT NULL AND c.embedding IS NOT NULL
                    THEN 1 - (c.embedding <=> p_query_embedding)
                ELSE NULL::DOUBLE PRECISION
            END AS vector_score,
            CASE
                WHEN v_tsquery IS NOT NULL AND c.search_vector @@ v_tsquery
                    THEN ts_rank_cd(c.search_vector, v_tsquery)
                ELSE NULL::REAL
            END AS text_score,
            CASE
                WHEN lower(d.title) = lower(COALESCE(btrim(p_query), '')) THEN 0.15
                WHEN COALESCE(btrim(p_query), '') <> '' AND lower(d.title) LIKE '%' || lower(btrim(p_query)) || '%' THEN 0.05
                ELSE 0.0
            END AS title_boost,
            d.owner_id AS owner_id
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE (
            (p_query_embedding IS NOT NULL AND c.embedding IS NOT NULL)
            OR (v_tsquery IS NOT NULL AND c.search_vector @@ v_tsquery)
        )
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
    )
    SELECT
        scored.chunk_id,
        scored.document_id,
        scored.document_title,
        scored.section_title,
        scored.chunk_text,
        scored.facet_path,
        scored.metadata,
        scored.vector_score,
        scored.text_score,
        (
            COALESCE(scored.vector_score, 0.0) * 0.7
            + COALESCE(scored.text_score, 0.0)::DOUBLE PRECISION * 0.3
            + scored.title_boost
        ) AS hybrid_score
    FROM scored
    ORDER BY hybrid_score DESC, document_title ASC
    LIMIT GREATEST(COALESCE(p_match_count, 5), 1);
END;
$$;

CREATE OR REPLACE FUNCTION knowledgebase.search_chunks_hybrid_rls(
    p_query TEXT,
    p_query_embedding VECTOR(1536),
    p_match_count INT DEFAULT 5,
    p_facet_keys TEXT[] DEFAULT NULL
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
    hybrid_score DOUBLE PRECISION
)
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = knowledgebase, public
AS $$
DECLARE
    v_tsquery tsquery;
BEGIN
    IF p_query_embedding IS NULL AND (p_query IS NULL OR btrim(p_query) = '') THEN
        RAISE EXCEPTION 'Query text or query embedding is required';
    END IF;
    IF p_query IS NOT NULL AND btrim(p_query) <> '' THEN
        v_tsquery := websearch_to_tsquery('english', p_query);
    END IF;
    RETURN QUERY
    WITH scored AS (
        SELECT c.id AS chunk_id, d.id AS document_id, d.title AS document_title,
               ds.heading AS section_title, c.content AS chunk_text,
               c.metadata ->> 'facet_path' AS facet_path, c.metadata AS metadata,
               CASE WHEN p_query_embedding IS NOT NULL AND c.embedding IS NOT NULL THEN 1 - (c.embedding <=> p_query_embedding) ELSE NULL END AS vector_score,
               CASE WHEN v_tsquery IS NOT NULL AND c.search_vector @@ v_tsquery THEN ts_rank_cd(c.search_vector, v_tsquery) ELSE NULL END AS text_score,
               CASE WHEN lower(d.title) = lower(COALESCE(btrim(p_query), '')) THEN 0.15 WHEN COALESCE(btrim(p_query), '') <> '' AND lower(d.title) LIKE '%' || lower(btrim(p_query)) || '%' THEN 0.05 ELSE 0.0 END AS title_boost
        FROM knowledgebase.chunks c
        JOIN knowledgebase.documents d ON d.id = c.document_id
        LEFT JOIN knowledgebase.document_sections ds ON ds.id = c.section_id
        WHERE ((p_query_embedding IS NOT NULL AND c.embedding IS NOT NULL) OR (v_tsquery IS NOT NULL AND c.search_vector @@ v_tsquery))
          AND (p_facet_keys IS NULL OR EXISTS (SELECT 1 FROM knowledgebase.document_facets df JOIN knowledgebase.facets f ON f.id = df.facet_id WHERE df.document_id = d.id AND f.facet_key = ANY (p_facet_keys)))
    )
    SELECT scored.chunk_id, scored.document_id, scored.document_title, scored.section_title, scored.chunk_text, scored.facet_path, scored.metadata, scored.vector_score, scored.text_score,
           (COALESCE(scored.vector_score,0)*0.7 + COALESCE(scored.text_score,0)::DOUBLE PRECISION*0.3 + scored.title_boost) AS hybrid_score
    FROM scored ORDER BY hybrid_score DESC, document_title ASC LIMIT GREATEST(COALESCE(p_match_count,5),1);
END;
$$;

-- ============================================================================
-- 4. Navigation Facets RPC — token OR RLS
-- ============================================================================
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
    -- facets are public; no extra RLS filter needed
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

GRANT EXECUTE ON FUNCTION knowledgebase.match_chunks_by_embedding(TEXT, VECTOR, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.search_chunks_full_text(TEXT, TEXT, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.search_chunks_hybrid(TEXT, TEXT, VECTOR, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.get_navigation_facets(TEXT, TEXT) TO authenticated, service_role;

GRANT EXECUTE ON FUNCTION knowledgebase.match_chunks_by_embedding_rls(VECTOR, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.search_chunks_full_text_rls(TEXT, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.search_chunks_hybrid_rls(TEXT, VECTOR, INT, TEXT[]) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION knowledgebase.get_navigation_facets_rls(TEXT) TO authenticated, service_role;
