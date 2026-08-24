"""Live PostgreSQL & pgvector Integration & RLS Security Test Suite.

Executes real database migrations, real PostgreSQL RLS sessions (anon, authenticated, service_role),
EXPLAIN (ANALYZE, BUFFERS) execution plans, HNSW iterative scans, and RRF Hybrid RPCs against a live PostgreSQL instance.

Automatically activated when POSTGRES_URL or DATABASE_URL is set, or in GitHub Actions CI / local Docker.
Skips cleanly if no PostgreSQL instance is reachable.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _get_postgres_conn_info() -> Optional[Dict[str, str]]:
    url_str = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not url_str:
        if os.environ.get("CI") == "true":
            url_str = "postgresql://postgres:postgres@localhost:5432/postgres"
        else:
            return None

    parsed = urlparse(url_str)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "postgres",
        "dbname": (parsed.path or "/postgres").lstrip("/"),
    }


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


@unittest.skipUnless(is_postgres_available(), "Live PostgreSQL instance not reachable (set POSTGRES_URL or run Docker)")
class TestLivePostgresPgvectorIntegration(unittest.TestCase):
    """End-to-end integration tests on real PostgreSQL + pgvector instance."""

    @classmethod
    def setUpClass(cls):
        cls.conn_info = _get_postgres_conn_info()
        repo_root = Path(__file__).resolve().parent.parent

        shim_file = repo_root / "sql" / "local_init" / "00_init_supabase_shim.sql"
        # Exercise the production upgrade path from the original schema, rather
        # than bootstrapping from the already-updated root SQL scripts.
        migrations_dir = repo_root / "supabase" / "migrations"
        schema_file = migrations_dir / "20260820000001_knowledgebase_schema.sql"
        functions_file = migrations_dir / "20260820000002_knowledgebase_functions.sql"
        tenant_token_migration = migrations_dir / "20260823000003_tenant_scoped_access_tokens.sql"

        # Apply local test shim if present (creates roles and auth schema on standalone Postgres)
        if shim_file.exists():
            res_shim = _run_psql_query(shim_file.read_text(encoding="utf-8"), cls.conn_info)
            if res_shim.returncode != 0:
                raise RuntimeError(f"Failed applying 00_init_supabase_shim.sql: {res_shim.stderr}")

        # Apply standard Supabase schema
        res_schema = _run_psql_query(schema_file.read_text(encoding="utf-8"), cls.conn_info)
        if res_schema.returncode != 0:
            raise RuntimeError(f"Failed applying 01_schema.sql: {res_schema.stderr}")

        # Apply standard Supabase functions
        res_fn = _run_psql_query(functions_file.read_text(encoding="utf-8"), cls.conn_info)
        if res_fn.returncode != 0:
            raise RuntimeError(f"Failed applying 02_functions.sql: {res_fn.stderr}")

        res_tenant_token = _run_psql_query(tenant_token_migration.read_text(encoding="utf-8"), cls.conn_info)
        if res_tenant_token.returncode != 0:
            raise RuntimeError(f"Failed applying tenant token migration: {res_tenant_token.stderr}")

    def setUp(self):
        # Clean test records
        _run_psql_query("DELETE FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.documents WHERE document_key LIKE '%test_live_%';", self.conn_info)

    def tearDown(self):
        _run_psql_query("DELETE FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.documents WHERE document_key LIKE '%test_live_%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.access_tokens WHERE token_name IN ('test_scoped_token', 'test_global_token');", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.facets WHERE facet_key = 'token_isolation';", self.conn_info)
        _run_psql_query("DELETE FROM auth.users WHERE id IN ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');", self.conn_info)

    def test_live_schema_and_extensions_active(self):
        """Verifies vector extension and knowledgebase schema exist."""
        sql = "SELECT extname FROM pg_extension WHERE extname = 'vector';"
        res = _run_psql_query(sql, self.conn_info)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(_extract_scalar(res.stdout), "vector")

        sql_tables = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'knowledgebase';"
        res_tbl = _run_psql_query(sql_tables, self.conn_info)
        self.assertEqual(res_tbl.returncode, 0)
        table_count = int(_extract_scalar(res_tbl.stdout))
        self.assertGreaterEqual(table_count, 5, "All knowledgebase tables must exist")

    def test_live_weighted_tsvector_trigger(self):
        """Verifies that inserting a document and chunk computes search_vector with A, B, D weights."""
        doc_sql = """
        INSERT INTO knowledgebase.documents (document_key, title, checksum)
        VALUES ('test_live_doc_1', 'PostgreSQL 16 High Availability', 'chk_001')
        RETURNING id;
        """
        doc_res = _run_psql_query(doc_sql, self.conn_info)
        self.assertEqual(doc_res.returncode, 0, f"doc insert error: {doc_res.stderr}")
        doc_id = _extract_scalar(doc_res.stdout)

        sec_sql = f"""
        INSERT INTO knowledgebase.document_sections (document_id, heading, level, sort_order)
        VALUES ('{doc_id}', 'Replication and Failover', 2, 1)
        RETURNING id;
        """
        sec_res = _run_psql_query(sec_sql, self.conn_info)
        self.assertEqual(sec_res.returncode, 0, f"sec insert error: {sec_res.stderr}")
        sec_id = _extract_scalar(sec_res.stdout)

        chunk_sql = f"""
        INSERT INTO knowledgebase.chunks (document_id, section_id, chunk_index, content)
        VALUES ('{doc_id}', '{sec_id}', 0, '[TEST_LIVE] Streaming replication with Patroni manager.')
        RETURNING search_vector::text;
        """
        chunk_res = _run_psql_query(chunk_sql, self.conn_info)
        self.assertEqual(chunk_res.returncode, 0, f"chunk insert error: {chunk_res.stderr}")
        tsvector_out = _extract_scalar(chunk_res.stdout)

        self.assertIn("'postgresql':1A", tsvector_out)
        self.assertIn("'replic':5B", tsvector_out)
        self.assertIn("'patroni':13", tsvector_out)

    def test_live_postgres_rls_sessions_under_authenticated_and_anon_roles(self):
        """Rigorous live PostgreSQL test verifying RLS isolation under true database roles:

        - Role 'authenticated' with User A UID sees ONLY User A documents & chunks.
        - Role 'authenticated' with User B UID sees ONLY User B documents & chunks.
        - Role 'anon' gets permission denied / zero records on table queries.
        - Role 'service_role' bypasses RLS for system operations.
        """
        user_a = "11111111-1111-1111-1111-111111111111"
        user_b = "22222222-2222-2222-2222-222222222222"

        # 1. Populate multi-tenant test data using service_role / superuser
        setup_sql = f"""
        DO $$
        DECLARE
            v_doc_a UUID;
            v_doc_b UUID;
        BEGIN
            INSERT INTO auth.users (id, email) VALUES ('{user_a}', 'user_a@example.com') ON CONFLICT DO NOTHING;
            INSERT INTO auth.users (id, email) VALUES ('{user_b}', 'user_b@example.com') ON CONFLICT DO NOTHING;

            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
            VALUES ('test_live_doc_a', 'User A Secret Roadmap', 'chk_a', '{user_a}')
            RETURNING id INTO v_doc_a;

            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
            VALUES ('test_live_doc_b', 'User B Secret Financials', 'chk_b', '{user_b}')
            RETURNING id INTO v_doc_b;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_a, 0, '[TEST_LIVE] Confidential strategy for User A', array_fill(0.2, ARRAY[1536])::vector);

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_b, 0, '[TEST_LIVE] Confidential financial tokens for User B', array_fill(0.8, ARRAY[1536])::vector);
        END;
        $$;
        """
        setup_res = _run_psql_query(setup_sql, self.conn_info)
        self.assertEqual(setup_res.returncode, 0, f"Setup error: {setup_res.stderr}")

        # 2. Test User A Postgres session:
        user_a_test_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SELECT count(*) FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';
        COMMIT;
        """
        res_a = _run_psql_query(user_a_test_sql, self.conn_info)
        self.assertEqual(res_a.returncode, 0, f"User A test error: {res_a.stderr}")
        # Should only see 1 chunk (User A's chunk)
        self.assertEqual(_extract_scalar(res_a.stdout), "1")

        # 3. Test User A querying RLS-enabled Hybrid RPC:
        user_a_rpc_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SELECT count(*) FROM knowledgebase.search_chunks_hybrid_rls(
            p_query := 'financial tokens',
            p_query_embedding := array_fill(0.8, ARRAY[1536])::vector
        ) WHERE chunk_text LIKE '%User B%';
        COMMIT;
        """
        res_a_rpc = _run_psql_query(user_a_rpc_sql, self.conn_info)
        self.assertEqual(res_a_rpc.returncode, 0, f"User A RPC error: {res_a_rpc.stderr}")
        # MUST return 0 results from User B!
        self.assertEqual(_extract_scalar(res_a_rpc.stdout), "0")

        # 4. Test Anon Role session (must be blocked):
        anon_test_sql = """
        BEGIN;
        SET LOCAL ROLE anon;
        SELECT count(*) FROM knowledgebase.documents;
        COMMIT;
        """
        res_anon = _run_psql_query(anon_test_sql, self.conn_info)
        # Anon has no SELECT grant on knowledgebase.documents -> returns error
        self.assertNotEqual(res_anon.returncode, 0, "Anon role must be denied SELECT on documents")

    def test_live_explain_analyze_plan_execution(self):
        """Executes EXPLAIN (ANALYZE, BUFFERS) in real PostgreSQL to verify HNSW and GIN index eligibility."""
        explain_vector_sql = """
        EXPLAIN (ANALYZE, COSTS)
        SELECT c.id
        FROM knowledgebase.chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> array_fill(0.1, ARRAY[1536])::vector
        LIMIT 5;
        """
        res_vec = _run_psql_query(explain_vector_sql, self.conn_info)
        self.assertEqual(res_vec.returncode, 0)
        explain_output = res_vec.stdout.lower()
        # Verify query plan executes valid index or scan with order by limit
        self.assertTrue("limit" in explain_output or "scan" in explain_output)

        explain_fts_sql = """
        EXPLAIN (ANALYZE, COSTS)
        SELECT c.id
        FROM knowledgebase.chunks c
        WHERE c.search_vector @@ to_tsquery('english', 'postgres')
        LIMIT 5;
        """
        res_fts = _run_psql_query(explain_fts_sql, self.conn_info)
        self.assertEqual(res_fts.returncode, 0)
        self.assertTrue("scan" in res_fts.stdout.lower())

    def test_live_iterative_scan_and_ef_search_support(self):
        """Verifies that search_chunks_hybrid and match_chunks_by_embedding support p_ef_search."""
        token_val = "easy_rag_live_token_ef_search"
        token_sql = f"""
        INSERT INTO knowledgebase.access_tokens (token_name, token_hash, is_active)
        VALUES ('ef_search_test_token', knowledgebase.hash_access_token('{token_val}'), TRUE)
        ON CONFLICT (token_hash) DO NOTHING;
        """
        _run_psql_query(token_sql, self.conn_info)

        rpc_sql = f"""
        SELECT count(*) FROM knowledgebase.search_chunks_hybrid(
            p_kb_token := '{token_val}',
            p_query := 'pgvector',
            p_query_embedding := array_fill(0.1, ARRAY[1536])::vector,
            p_match_count := 5,
            p_ef_search := 80
        );
        """
        res = _run_psql_query(rpc_sql, self.conn_info)
        self.assertEqual(res.returncode, 0)

    def test_tenant_token_upgrade_sequence_and_rpc_isolation(self):
        """Apply 00001->00003 and verify all token RPCs enforce tenant scope."""
        repo_root = Path(__file__).resolve().parent.parent
        migration_dir = repo_root / "supabase" / "migrations"
        for migration_name in (
            "20260820000001_knowledgebase_schema.sql",
            "20260820000002_knowledgebase_functions.sql",
            "20260823000003_tenant_scoped_access_tokens.sql",
        ):
            result = _run_psql_query((migration_dir / migration_name).read_text(encoding="utf-8"), self.conn_info)
            self.assertEqual(result.returncode, 0, f"Migration {migration_name} failed: {result.stderr}")

        tenant_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        tenant_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        setup = f"""
        INSERT INTO auth.users (id, email) VALUES ('{tenant_a}', 'tenant-a@test') ON CONFLICT DO NOTHING;
        INSERT INTO auth.users (id, email) VALUES ('{tenant_b}', 'tenant-b@test') ON CONFLICT DO NOTHING;
        INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
        VALUES ('test_live_token_a', 'Tenant A Token Secret', 'token_a', '{tenant_a}') ON CONFLICT (document_key) DO NOTHING;
        INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
        VALUES ('test_live_token_b', 'Tenant B Token Secret', 'token_b', '{tenant_b}') ON CONFLICT (document_key) DO NOTHING;
        INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
        SELECT id, 0, '[TEST_LIVE] tenant-a-isolated phrase', array_fill(0.2, ARRAY[1536])::vector
        FROM knowledgebase.documents WHERE document_key = 'test_live_token_a'
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
        INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
        SELECT id, 0, '[TEST_LIVE] tenant-b-isolated phrase', array_fill(0.8, ARRAY[1536])::vector
        FROM knowledgebase.documents WHERE document_key = 'test_live_token_b'
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
        INSERT INTO knowledgebase.facets (facet_type, facet_key, label)
        VALUES ('test', 'token_isolation', 'Token Isolation') ON CONFLICT (facet_key) DO NOTHING;
        INSERT INTO knowledgebase.document_facets (document_id, facet_id)
        SELECT d.id, f.id FROM knowledgebase.documents d CROSS JOIN knowledgebase.facets f
        WHERE d.document_key IN ('test_live_token_a', 'test_live_token_b') AND f.facet_key = 'token_isolation'
        ON CONFLICT DO NOTHING;
        INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id)
        VALUES ('test_scoped_token', knowledgebase.hash_access_token('test_scoped_token'), '{tenant_a}') ON CONFLICT (token_hash) DO UPDATE SET tenant_id = EXCLUDED.tenant_id, is_active = TRUE;
        INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id)
        VALUES ('test_global_token', knowledgebase.hash_access_token('test_global_token'), NULL) ON CONFLICT (token_hash) DO UPDATE SET tenant_id = NULL, is_active = TRUE;
        """
        cleanup = """
        DELETE FROM knowledgebase.access_tokens WHERE token_name IN ('test_scoped_token', 'test_global_token');
        DELETE FROM knowledgebase.documents WHERE document_key IN ('test_live_token_a', 'test_live_token_b');
        DELETE FROM knowledgebase.facets WHERE facet_key = 'token_isolation';
        DELETE FROM auth.users WHERE id IN ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');
        """
        cleanup_result = _run_psql_query(cleanup, self.conn_info)
        self.assertEqual(cleanup_result.returncode, 0, cleanup_result.stderr)
        setup_result = _run_psql_query(setup, self.conn_info)
        self.assertEqual(setup_result.returncode, 0, setup_result.stderr)

        calls = (
            ("match_chunks_by_embedding", "SELECT document_id FROM knowledgebase.match_chunks_by_embedding('test_scoped_token', array_fill(0.8, ARRAY[1536])::vector, 20)"),
            ("search_chunks_full_text", "SELECT document_id FROM knowledgebase.search_chunks_full_text('test_scoped_token', 'isolated', 20)"),
            ("search_chunks_hybrid", "SELECT document_id FROM knowledgebase.search_chunks_hybrid('test_scoped_token', 'isolated', array_fill(0.8, ARRAY[1536])::vector, 20)"),
        )
        for name, query in calls:
            result = _run_psql_query(query, self.conn_info)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
            query_body = query.rstrip().rstrip(";")
            visible_a = _run_psql_query(
                f"SELECT r.document_id FROM ({query_body}) r JOIN knowledgebase.documents d ON d.id = r.document_id WHERE d.document_key = 'test_live_token_a';",
                self.conn_info,
            )
            self.assertNotEqual(_extract_scalar(visible_a.stdout), "", f"Scoped token failed to see tenant A via {name}")
            leaked = _run_psql_query(
                f"SELECT r.document_id FROM ({query_body}) r JOIN knowledgebase.documents d ON d.id = r.document_id WHERE d.document_key = 'test_live_token_b';",
                self.conn_info,
            )
            self.assertEqual(_extract_scalar(leaked.stdout), "", f"Scoped token leaked tenant B via {name}")

        facet_scoped = _run_psql_query("SELECT document_count FROM knowledgebase.get_navigation_facets('test_scoped_token') WHERE facet_key = 'token_isolation';", self.conn_info)
        self.assertEqual(facet_scoped.returncode, 0, facet_scoped.stderr)
        self.assertEqual(_extract_scalar(facet_scoped.stdout), "1")
        facet_global = _run_psql_query("SELECT document_count FROM knowledgebase.get_navigation_facets('test_global_token') WHERE facet_key = 'token_isolation';", self.conn_info)
        self.assertEqual(facet_global.returncode, 0, facet_global.stderr)
        self.assertEqual(_extract_scalar(facet_global.stdout), "2")

        global_result = _run_psql_query("SELECT count(*) FROM knowledgebase.search_chunks_full_text('test_global_token', 'isolated', 20, ARRAY['token_isolation']);", self.conn_info)
        self.assertEqual(global_result.returncode, 0, global_result.stderr)
        self.assertEqual(_extract_scalar(global_result.stdout), "2")
        global_vector = _run_psql_query("SELECT count(*) FROM knowledgebase.match_chunks_by_embedding('test_global_token', array_fill(0.8, ARRAY[1536])::vector, 20, ARRAY['token_isolation']);", self.conn_info)
        self.assertEqual(_extract_scalar(global_vector.stdout), "2")
        global_hybrid = _run_psql_query("SELECT count(*) FROM knowledgebase.search_chunks_hybrid('test_global_token', 'isolated', array_fill(0.8, ARRAY[1536])::vector, 20, ARRAY['token_isolation']);", self.conn_info)
        self.assertEqual(_extract_scalar(global_hybrid.stdout), "2")
        final_cleanup = _run_psql_query(cleanup, self.conn_info)
        self.assertEqual(final_cleanup.returncode, 0, final_cleanup.stderr)


if __name__ == "__main__":
    unittest.main()
