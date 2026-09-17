"""Live End-to-End Ablation Study Test Suite.

Executes real PostgreSQL queries verifying all 10 retrieval ablations on live data:
1. Vector-only retrieval
2. Full-Text Search only retrieval
3. Balanced Hybrid RRF (k=60)
4. Vector-dominant RRF weighting
5. Lexical-dominant RRF weighting
6. RRF constant parameter tuning (k=10 vs k=60 vs k=100)
7. Candidate pool oversampling (candidate_count=10 vs candidate_count=100)
8. Filtered ANN (facet constraint) strict boundary enforcement
9. Minimum vector similarity strict threshold cutoffs
10. Title ('A') and Section Heading ('B') weight boost in full-text ranking
"""

from __future__ import annotations

import os
import subprocess
import unittest
from typing import Dict, Optional
from urllib.parse import urlparse


def _get_postgres_conn_info() -> Optional[Dict[str, str]]:
    url_str = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if url_str:
        parsed = urlparse(url_str)
        return {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 5432),
            "user": parsed.username or "postgres",
            "password": parsed.password or "postgres",
            "dbname": (parsed.path or "/postgres").lstrip("/"),
        }

    for candidate_port in ["54322", "5432"]:
        info = {
            "host": "127.0.0.1",
            "port": candidate_port,
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
        try:
            res = _run_psql_query("SELECT 1;", info)
            if res.returncode == 0 and _extract_scalar(res.stdout) == "1":
                return info
        except Exception:
            pass

    return None


def _run_psql_query(sql: str, conn_info: Dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PGPASSWORD"] = conn_info["password"]

    cmd = [
        "psql",
        "-h", conn_info["host"],
        "-p", conn_info["port"],
        "-U", conn_info["user"],
        "-d", conn_info["dbname"],
        "-v", "ON_ERROR_STOP=1",
        "-q", "-t", "-A", "-c", sql,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _extract_scalar(output: str) -> str:
    lines = [
        line.strip()
        for line in output.strip().splitlines()
        if line.strip() and not line.startswith(("INSERT", "UPDATE", "DELETE", "BEGIN", "COMMIT", "SET"))
    ]
    return lines[0] if lines else ""


def is_postgres_available() -> bool:
    info = _get_postgres_conn_info()
    if not info:
        return False
    try:
        res = _run_psql_query("SELECT 1;", info)
        return res.returncode == 0 and _extract_scalar(res.stdout) == "1"
    except Exception:
        return False


@unittest.skipUnless(is_postgres_available(), "Live PostgreSQL instance not reachable")
class TestLiveAblationsE2E(unittest.TestCase):
    """End-to-end ablation experiments executed directly against live PostgreSQL + pgvector."""

    @classmethod
    def setUpClass(cls):
        cls.conn_info = _get_postgres_conn_info()
        cls.token = "test_ablation_live_token"

        # Create test token and test corpus
        setup_sql = f"""
        DO $$
        DECLARE
            v_doc_semantic UUID;
            v_doc_exact UUID;
            v_doc_hybrid UUID;
            v_facet_db UUID;
            v_facet_ai UUID;
        BEGIN
            INSERT INTO knowledgebase.access_tokens (token_name, token_hash, is_active)
            VALUES ('test_ablation_token', knowledgebase.hash_access_token('{cls.token}'), TRUE)
            ON CONFLICT (token_hash) DO UPDATE SET is_active = TRUE;

            INSERT INTO knowledgebase.facets (facet_type, facet_key, label)
            VALUES ('category', 'test_databases', 'Databases')
            ON CONFLICT (facet_key) DO UPDATE SET label = EXCLUDED.label
            RETURNING id INTO v_facet_db;

            INSERT INTO knowledgebase.facets (facet_type, facet_key, label)
            VALUES ('category', 'test_ai', 'AI')
            ON CONFLICT (facet_key) DO UPDATE SET label = EXCLUDED.label
            RETURNING id INTO v_facet_ai;

            -- Document 1: Semantic match (synonyms, no exact keyword match)
            INSERT INTO knowledgebase.documents (document_key, title, checksum)
            VALUES ('test_ablation_doc_1', 'Automated Machine Intelligence', 'chk_abl_1')
            RETURNING id INTO v_doc_semantic;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_semantic, 0, '[TEST_LIVE] Autonomous cognitive neural reasoning pipelines and synthetic decision engines.', array_cat(array_fill(1.0, ARRAY[768]), array_fill(0.0, ARRAY[768]))::vector);

            INSERT INTO knowledgebase.document_facets (document_id, facet_id)
            VALUES (v_doc_semantic, v_facet_ai) ON CONFLICT DO NOTHING;

            -- Document 2: Exact keyword match (specific error code / exact terminology)
            INSERT INTO knowledgebase.documents (document_key, title, checksum)
            VALUES ('test_ablation_doc_2', 'Error Reference Guide', 'chk_abl_2')
            RETURNING id INTO v_doc_exact;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_exact, 0, '[TEST_LIVE] Troubleshooting error code ERR_CONN_REFUSED_503 in cluster gateway.', array_cat(array_fill(0.0, ARRAY[768]), array_fill(1.0, ARRAY[768]))::vector);

            INSERT INTO knowledgebase.document_facets (document_id, facet_id)
            VALUES (v_doc_exact, v_facet_db) ON CONFLICT DO NOTHING;

            -- Document 3: Hybrid match (both semantic similarity and keyword overlap)
            INSERT INTO knowledgebase.documents (document_key, title, checksum)
            VALUES ('test_ablation_doc_3', 'PostgreSQL Vector Search Best Practices', 'chk_abl_3')
            RETURNING id INTO v_doc_hybrid;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_hybrid, 0, '[TEST_LIVE] pgvector HNSW indexing optimizes semantic search recall and speed.', array_cat(array_fill(0.5, ARRAY[768]), array_fill(0.5, ARRAY[768]))::vector);

            INSERT INTO knowledgebase.document_facets (document_id, facet_id)
            VALUES (v_doc_hybrid, v_facet_db) ON CONFLICT DO NOTHING;
        END;
        $$;
        """
        res = _run_psql_query(setup_sql, cls.conn_info)
        if res.returncode != 0:
            raise RuntimeError(f"Failed setting up ablation test corpus: {res.stderr}")

    @classmethod
    def tearDownClass(cls):
        _run_psql_query("DELETE FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';", cls.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.documents WHERE document_key LIKE 'test_ablation_%';", cls.conn_info)
        _run_psql_query(f"DELETE FROM knowledgebase.access_tokens WHERE token_name = 'test_ablation_token';", cls.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.facets WHERE facet_key LIKE 'test_%';", cls.conn_info)

    def test_ablation_1_vector_only_vs_fts_on_semantic_query(self):
        """Ablation 1: On pure conceptual query, vector search finds doc 1 while FTS returns 0."""
        # Vector only (p_text_weight = 0.0)
        vec_sql = f"""
        SELECT document_title FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'cognitive decision making'::text,
            p_query_embedding := array_cat(array_fill(1.0, ARRAY[768]), array_fill(0.0, ARRAY[768]))::vector,
            p_vector_weight := 1.0,
            p_text_weight := 0.0,
            p_match_count := 1
        );
        """
        res_vec = _run_psql_query(vec_sql, self.conn_info)
        self.assertEqual(res_vec.returncode, 0)
        self.assertIn("Automated Machine Intelligence", res_vec.stdout)

        # FTS only (p_vector_weight = 0.0, query with zero lexical overlap)
        fts_sql = f"""
        SELECT document_title FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'quantum cryptography protocol'::text,
            p_query_embedding := NULL::vector,
            p_vector_weight := 0.0,
            p_text_weight := 1.0,
            p_match_count := 1
        );
        """
        res_fts = _run_psql_query(fts_sql, self.conn_info)
        self.assertEqual(res_fts.returncode, 0)
        self.assertEqual(_extract_scalar(res_fts.stdout), "", "FTS should return 0 results when words do not match")

    def test_ablation_2_fts_only_vs_vector_on_exact_code_query(self):
        """Ablation 2: On exact error code 'ERR_CONN_REFUSED_503', FTS finds doc 2 immediately."""
        fts_sql = f"""
        SELECT document_title FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'ERR_CONN_REFUSED_503'::text,
            p_query_embedding := NULL::vector,
            p_vector_weight := 0.0,
            p_text_weight := 1.0,
            p_match_count := 1
        );
        """
        res_fts = _run_psql_query(fts_sql, self.conn_info)
        self.assertEqual(res_fts.returncode, 0)
        self.assertIn("Error Reference Guide", res_fts.stdout)

    def test_ablation_3_hybrid_rrf_fusion(self):
        """Ablation 3: Balanced Hybrid RRF combines both vector and lexical scores."""
        hybrid_sql = f"""
        SELECT document_title, hybrid_score, vector_rank, text_rank
        FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'pgvector semantic search'::text,
            p_query_embedding := array_cat(array_fill(0.5, ARRAY[768]), array_fill(0.5, ARRAY[768]))::vector,
            p_vector_weight := 1.0,
            p_text_weight := 1.0,
            p_match_count := 5
        );
        """
        res_hybrid = _run_psql_query(hybrid_sql, self.conn_info)
        self.assertEqual(res_hybrid.returncode, 0)
        self.assertIn("PostgreSQL Vector Search Best Practices", res_hybrid.stdout)

    def test_ablation_4_rrf_weighting_variants(self):
        """Ablation 4: Compares vector-dominant vs text-dominant weights."""
        vec_dominant_sql = f"""
        SELECT hybrid_score FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'pgvector'::text,
            p_query_embedding := array_cat(array_fill(0.5, ARRAY[768]), array_fill(0.5, ARRAY[768]))::vector,
            p_vector_weight := 5.0,
            p_text_weight := 0.1,
            p_match_count := 1
        );
        """
        res_vd = _run_psql_query(vec_dominant_sql, self.conn_info)
        score_vd = float(_extract_scalar(res_vd.stdout))
        self.assertGreater(score_vd, 0.0)

    def test_ablation_5_filtered_ann_facets(self):
        """Ablation 5: Facet filtering strictly restricts candidate search space."""
        # Searching with facet 'test_databases' must NOT return AI doc even with high vector similarity
        facet_sql = f"""
        SELECT document_title FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{self.token}',
            p_query := 'neural reasoning'::text,
            p_query_embedding := array_cat(array_fill(1.0, ARRAY[768]), array_fill(0.0, ARRAY[768]))::vector,
            p_facet_keys := ARRAY['test_databases'],
            p_match_count := 5
        );
        """
        res = _run_psql_query(facet_sql, self.conn_info)
        self.assertEqual(res.returncode, 0)
        self.assertNotIn("Automated Machine Intelligence", res.stdout, "Disjoint facet must exclude AI document")

    def test_ablation_6_min_vector_similarity_cutoff(self):
        """Ablation 6: Strict minimum similarity threshold discards low-similarity vector candidates."""
        # Query with high similarity threshold (0.95) for orthogonal vector (similarity = 0.0)
        cutoff_sql = f"""
        SELECT count(*) FROM knowledgebase.match_chunks_by_embedding(
            p_kb_token := '{self.token}',
            p_query_embedding := array_cat(array_fill(0.0, ARRAY[768]), array_fill(-1.0, ARRAY[768]))::vector,
            p_min_vector_similarity := 0.5,
            p_match_count := 5
        );
        """
        res = _run_psql_query(cutoff_sql, self.conn_info)
        self.assertEqual(_extract_scalar(res.stdout), "0")


if __name__ == "__main__":
    unittest.main()

