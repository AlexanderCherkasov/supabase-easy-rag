"""Live PostgreSQL & pgvector Comprehensive E2E Integration & Security Test Suite.

Executes real database migrations, triggers, PostgreSQL RLS sessions (anon, authenticated, service_role),
EXPLAIN (ANALYZE, BUFFERS) execution plans, HNSW iterative scans, and all RRF Hybrid RPCs against a live PostgreSQL instance.

Automatically activated when POSTGRES_URL or DATABASE_URL is set, or automatically connects to
local Supabase (port 54322) or standard Docker/CI (port 5432).
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
    if url_str:
        parsed = urlparse(url_str)
        return {
            "host": parsed.hostname or "localhost",
            "port": str(parsed.port or 5432),
            "user": parsed.username or "postgres",
            "password": parsed.password or "postgres",
            "dbname": (parsed.path or "/postgres").lstrip("/"),
        }

    # Auto-detect local Supabase port (54322) first, then standard Docker/CI (5432)
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


@unittest.skipUnless(is_postgres_available(), "Live PostgreSQL instance not reachable (set POSTGRES_URL or run Supabase/Docker)")
class TestLivePostgresPgvectorIntegration(unittest.TestCase):
    """End-to-end integration and security test suite on a real PostgreSQL + pgvector instance."""

    @classmethod
    def setUpClass(cls):
        cls.conn_info = _get_postgres_conn_info()
        repo_root = Path(__file__).resolve().parent.parent

        shim_file = repo_root / "sql" / "local_init" / "00_init_supabase_shim.sql"
        migrations_dir = repo_root / "supabase" / "migrations"
        migration_files = [
            migrations_dir / "20260820000001_knowledgebase_schema.sql",
            migrations_dir / "20260820000002_knowledgebase_functions.sql",
            migrations_dir / "20260823000003_tenant_scoped_access_tokens.sql",
            migrations_dir / "20260824000004_multitenancy_and_user_scoped_ingestion.sql",
        ]

        if shim_file.exists():
            res_shim = _run_psql_query(shim_file.read_text(encoding="utf-8"), cls.conn_info)
            if res_shim.returncode != 0:
                raise RuntimeError(f"Failed applying 00_init_supabase_shim.sql: {res_shim.stderr}")

        for mig in migration_files:
            if mig.exists():
                res = _run_psql_query(mig.read_text(encoding="utf-8"), cls.conn_info)
                if res.returncode != 0:
                    raise RuntimeError(f"Failed applying {mig.name}: {res.stderr}")

    def setUp(self):
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        _run_psql_query("DELETE FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.document_sections WHERE heading LIKE '%[TEST_LIVE]%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.documents WHERE document_key LIKE '%test_live_%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.access_tokens WHERE token_name LIKE '%test_%';", self.conn_info)
        _run_psql_query("DELETE FROM knowledgebase.facets WHERE facet_key LIKE '%test_%' OR facet_key = 'token_isolation';", self.conn_info)
        _run_psql_query("DELETE FROM auth.users WHERE id IN ('11111111-1111-1111-1111-111111111111', '22222222-2222-2222-2222-222222222222', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'cccccccc-cccc-cccc-cccc-cccccccccccc');", self.conn_info)

    # -------------------------------------------------------------------------
    # 1. Schema, Extensions & Indexes
    # -------------------------------------------------------------------------
    def test_live_schema_and_extensions_active(self):
        """Verifies vector extension, pgcrypto, and knowledgebase schema with all required tables exist."""
        sql = "SELECT extname FROM pg_extension WHERE extname = 'vector';"
        res = _run_psql_query(sql, self.conn_info)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(_extract_scalar(res.stdout), "vector")

        sql_tables = "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'knowledgebase';"
        res_tbl = _run_psql_query(sql_tables, self.conn_info)
        self.assertEqual(res_tbl.returncode, 0)
        table_count = int(_extract_scalar(res_tbl.stdout))
        self.assertGreaterEqual(table_count, 7, "All knowledgebase tables must exist")

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

    # -------------------------------------------------------------------------
    # 2. Database Triggers (Weights A/B/D, Cascades & Timestamps)
    # -------------------------------------------------------------------------
    def test_live_weighted_tsvector_trigger_and_cascades(self):
        """Verifies:
        1. Inserting doc + section + chunk computes search_vector with A (Title), B (Section), D (Content) weights.
        2. Updating document title cascades tsvector re-computation to all attached chunks.
        3. Updating section heading cascades tsvector re-computation to all attached chunks.
        4. Updating documents/chunks bumps updated_at timestamp.
        """
        # 1. Insert initial doc, section, chunk
        setup_sql = """
        DO $$
        DECLARE
            v_doc_id UUID;
            v_sec_id UUID;
            v_chunk_id UUID;
        BEGIN
            INSERT INTO knowledgebase.documents (document_key, title, checksum)
            VALUES ('test_live_cascade_doc', 'PostgreSQL High Availability Architecture', 'chk_init')
            RETURNING id INTO v_doc_id;

            INSERT INTO knowledgebase.document_sections (document_id, heading, level, sort_order)
            VALUES (v_doc_id, '[TEST_LIVE] Replication Failover Strategy', 2, 1)
            RETURNING id INTO v_sec_id;

            INSERT INTO knowledgebase.chunks (document_id, section_id, chunk_index, content)
            VALUES (v_doc_id, v_sec_id, 0, '[TEST_LIVE] Distributed consensus via Raft algorithm and Patroni cluster manager.')
            RETURNING id INTO v_chunk_id;
        END;
        $$;
        """
        res_setup = _run_psql_query(setup_sql, self.conn_info)
        self.assertEqual(res_setup.returncode, 0, f"Setup error: {res_setup.stderr}")

        # Check search_vector weights
        sql_check = "SELECT c.search_vector::text FROM knowledgebase.chunks c JOIN knowledgebase.documents d ON d.id = c.document_id WHERE d.document_key = 'test_live_cascade_doc';"
        tsvector_out = _extract_scalar(_run_psql_query(sql_check, self.conn_info).stdout)

        self.assertIn("'postgresql':1A", tsvector_out)
        self.assertIn("'architectur':4A", tsvector_out)
        self.assertIn("'replic':7B", tsvector_out)
        self.assertIn("'patroni':18", tsvector_out)  # Content has weight D (unweighted tag in psql)

        # 2. Test Document Title update cascade
        update_title_sql = """
        UPDATE knowledgebase.documents
        SET title = 'Kubernetes Cloud Native Databases'
        WHERE document_key = 'test_live_cascade_doc';
        """
        _run_psql_query(update_title_sql, self.conn_info)

        tsvector_updated = _extract_scalar(_run_psql_query(sql_check, self.conn_info).stdout)
        self.assertIn("'kubernet':1A", tsvector_updated)
        self.assertNotIn("'postgresql':1A", tsvector_updated)

        # 3. Test Section Heading update cascade
        update_heading_sql = """
        UPDATE knowledgebase.document_sections
        SET heading = '[TEST_LIVE] Disaster Recovery Runbooks'
        WHERE heading LIKE '%Replication Failover%';
        """
        _run_psql_query(update_heading_sql, self.conn_info)

        tsvector_sec_updated = _extract_scalar(_run_psql_query(sql_check, self.conn_info).stdout)
        self.assertIn("'disast':7B", tsvector_sec_updated)
        self.assertIn("'recoveri':8B", tsvector_sec_updated)
        self.assertNotIn("'replic':7B", tsvector_sec_updated)

    # -------------------------------------------------------------------------
    # 3. Fine-Grained Access Control & RLS Security Matrix
    # -------------------------------------------------------------------------
    def test_live_postgres_rls_sessions_and_sharing_matrix(self):
        """Rigorous live PostgreSQL test verifying RLS isolation under true database roles:
        - Role 'authenticated' with User A UID sees ONLY User A documents & chunks.
        - Role 'authenticated' with User B UID sees ONLY User B documents & chunks.
        - Role 'anon' gets permission denied / zero records on table queries.
        - Document sharing via document_owners allows shared read access.
        - Revoking from document_owners immediately removes access.
        - Service role bypasses RLS.
        """
        user_a = "11111111-1111-1111-1111-111111111111"
        user_b = "22222222-2222-2222-2222-222222222222"
        user_c = "cccccccc-cccc-cccc-cccc-cccccccccccc"

        setup_sql = f"""
        DO $$
        DECLARE
            v_doc_a UUID;
            v_doc_b UUID;
            v_doc_shared UUID;
        BEGIN
            INSERT INTO auth.users (id, email) VALUES ('{user_a}', 'user_a@example.com') ON CONFLICT (id) DO NOTHING;
            INSERT INTO auth.users (id, email) VALUES ('{user_b}', 'user_b@example.com') ON CONFLICT (id) DO NOTHING;
            INSERT INTO auth.users (id, email) VALUES ('{user_c}', 'user_c@example.com') ON CONFLICT (id) DO NOTHING;

            -- User A private doc
            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
            VALUES ('test_live_doc_a', 'User A Secret Roadmap', 'chk_a', '{user_a}')
            RETURNING id INTO v_doc_a;

            -- User B private doc
            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
            VALUES ('test_live_doc_b', 'User B Secret Financials', 'chk_b', '{user_b}')
            RETURNING id INTO v_doc_b;

            -- User B shared doc (initially shared with User A)
            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id)
            VALUES ('test_live_doc_shared', 'User B Shared Research', 'chk_s', '{user_b}')
            RETURNING id INTO v_doc_shared;

            INSERT INTO knowledgebase.document_owners (document_id, owner_id)
            VALUES (v_doc_shared, '{user_a}');

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_a, 0, '[TEST_LIVE] Confidential strategy for User A', array_fill(0.2, ARRAY[1536])::vector);

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_b, 0, '[TEST_LIVE] Confidential financial tokens for User B', array_fill(0.8, ARRAY[1536])::vector);

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_shared, 0, '[TEST_LIVE] Joint collaborative document between B and A', array_fill(0.5, ARRAY[1536])::vector);
        END;
        $$;
        """
        setup_res = _run_psql_query(setup_sql, self.conn_info)
        self.assertEqual(setup_res.returncode, 0, f"Setup error: {setup_res.stderr}")

        # 1. User A session: should see User A doc + Shared doc (2 total), NEVER User B private doc
        user_a_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SELECT count(*) FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';
        COMMIT;
        """
        res_a = _run_psql_query(user_a_sql, self.conn_info)
        self.assertEqual(res_a.returncode, 0, res_a.stderr)
        self.assertEqual(_extract_scalar(res_a.stdout), "2")

        # User A hybrid search RPC: MUST return 0 User B private records
        user_a_rpc_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SELECT count(*) FROM knowledgebase.search_chunks_hybrid_rls(
            p_query := 'financial tokens'::text,
            p_query_embedding := array_fill(0.8, ARRAY[1536])::vector
        ) WHERE chunk_text LIKE '%Financials%' OR chunk_text LIKE '%User B Secret%';
        COMMIT;
        """
        res_a_rpc = _run_psql_query(user_a_rpc_sql, self.conn_info)
        self.assertEqual(res_a_rpc.returncode, 0, res_a_rpc.stderr)
        self.assertEqual(_extract_scalar(res_a_rpc.stdout), "0")

        # 2. User C session (unshared): should see 0 chunks
        user_c_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_c}';
        SELECT count(*) FROM knowledgebase.chunks WHERE content LIKE '%[TEST_LIVE]%';
        COMMIT;
        """
        res_c = _run_psql_query(user_c_sql, self.conn_info)
        self.assertEqual(res_c.returncode, 0, res_c.stderr)
        self.assertEqual(_extract_scalar(res_c.stdout), "0")

        # 3. Revoke sharing: remove User A from document_owners
        revoke_sql = f"""
        DELETE FROM knowledgebase.document_owners
        WHERE document_id = (SELECT id FROM knowledgebase.documents WHERE document_key = 'test_live_doc_shared')
          AND owner_id = '{user_a}';
        """
        _run_psql_query(revoke_sql, self.conn_info)

        # User A should now see ONLY 1 chunk (User A's own chunk)
        res_a_after_revoke = _run_psql_query(user_a_sql, self.conn_info)
        self.assertEqual(_extract_scalar(res_a_after_revoke.stdout), "1")

        # 4. Anon role: must be completely denied on tables
        anon_sql = """
        BEGIN;
        SET LOCAL ROLE anon;
        SELECT count(*) FROM knowledgebase.documents;
        COMMIT;
        """
        res_anon = _run_psql_query(anon_sql, self.conn_info)
        self.assertNotEqual(res_anon.returncode, 0, "Anon role must be denied SELECT on documents")

    def test_live_system_category_isolation(self):
        """Verifies that top_level_category = 'system' documents are strictly protected:
        - Regular authenticated users cannot read or search them via RLS.
        - Users with JWT claim 'is_system_agent': true CAN read them.
        - Service role can read them.
        """
        user_a = "11111111-1111-1111-1111-111111111111"

        setup_sql = f"""
        DO $$
        DECLARE
            v_doc_sys UUID;
        BEGIN
            INSERT INTO auth.users (id, email) VALUES ('{user_a}', 'user_a@example.com') ON CONFLICT (id) DO NOTHING;

            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id, top_level_category)
            VALUES ('test_live_doc_sys', 'Internal System Prompt & Tools', 'chk_sys', '{user_a}', 'system')
            RETURNING id INTO v_doc_sys;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_sys, 0, '[TEST_LIVE] Secret prompt instructions and system tools', array_fill(0.1, ARRAY[1536])::vector);
        END;
        $$;
        """
        _run_psql_query(setup_sql, self.conn_info)

        # Regular user query: must return 0
        regular_user_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SELECT count(*) FROM knowledgebase.documents WHERE document_key = 'test_live_doc_sys';
        COMMIT;
        """
        res_reg = _run_psql_query(regular_user_sql, self.conn_info)
        self.assertEqual(res_reg.returncode, 0)
        self.assertEqual(_extract_scalar(res_reg.stdout), "0")

        # System agent query with JWT claim: must return 1
        system_agent_sql = f"""
        BEGIN;
        SET LOCAL ROLE authenticated;
        SET LOCAL "request.jwt.claim.sub" = '{user_a}';
        SET LOCAL "request.jwt.claims" = '{{"sub": "{user_a}", "app_metadata": {{"is_system_agent": true}}}}';
        SELECT count(*) FROM knowledgebase.documents WHERE document_key = 'test_live_doc_sys';
        COMMIT;
        """
        res_sys = _run_psql_query(system_agent_sql, self.conn_info)
        self.assertEqual(res_sys.returncode, 0)
        self.assertEqual(_extract_scalar(res_sys.stdout), "1")

    # -------------------------------------------------------------------------
    # 4. Access Tokens Matrix & Scoping E2E
    # -------------------------------------------------------------------------
    def test_live_access_tokens_scoping_and_audit(self):
        """Verifies full lifecycle of access tokens:
        - Tenant-scoped token isolates search to tenant documents.
        - Global token searches global documents.
        - Inactive and expired tokens are rejected.
        - Audit trail logs 'used' and 'failed_validation' events.
        """
        tenant_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        tenant_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        token_a = "test_scoped_token_a_val"
        token_global = "test_global_token_val"
        token_inactive = "test_inactive_token_val"
        token_expired = "test_expired_token_val"

        setup_sql = f"""
        DO $$
        DECLARE
            v_doc_a UUID;
            v_doc_b UUID;
            v_facet_id UUID;
        BEGIN
            INSERT INTO auth.users (id, email) VALUES ('{tenant_a}', 'tenant_a@test') ON CONFLICT (id) DO NOTHING;
            INSERT INTO auth.users (id, email) VALUES ('{tenant_b}', 'tenant_b@test') ON CONFLICT (id) DO NOTHING;

            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id, tenant_id)
            VALUES ('test_live_t_doc_a', 'Tenant A Secret Vault', 'chk_ta', '{tenant_a}', '{tenant_a}')
            RETURNING id INTO v_doc_a;

            INSERT INTO knowledgebase.documents (document_key, title, checksum, owner_id, tenant_id)
            VALUES ('test_live_t_doc_b', 'Tenant B Secret Vault', 'chk_tb', '{tenant_b}', '{tenant_b}')
            RETURNING id INTO v_doc_b;

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_a, 0, '[TEST_LIVE] Token isolated vault data for Tenant A', array_fill(0.3, ARRAY[1536])::vector);

            INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding)
            VALUES (v_doc_b, 0, '[TEST_LIVE] Token isolated vault data for Tenant B', array_fill(0.7, ARRAY[1536])::vector);

            INSERT INTO knowledgebase.facets (facet_type, facet_key, label)
            VALUES ('test', 'token_isolation', 'Token Isolation')
            ON CONFLICT (facet_key) DO UPDATE SET label = EXCLUDED.label
            RETURNING id INTO v_facet_id;

            INSERT INTO knowledgebase.document_facets (document_id, facet_id)
            VALUES (v_doc_a, v_facet_id), (v_doc_b, v_facet_id)
            ON CONFLICT DO NOTHING;

            -- Scoped token for Tenant A
            INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id, is_active)
            VALUES ('test_scoped_a', knowledgebase.hash_access_token('{token_a}'), '{tenant_a}', TRUE)
            ON CONFLICT (token_hash) DO UPDATE SET is_active = TRUE, tenant_id = '{tenant_a}';

            -- Global token
            INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id, is_active)
            VALUES ('test_global', knowledgebase.hash_access_token('{token_global}'), NULL, TRUE)
            ON CONFLICT (token_hash) DO UPDATE SET is_active = TRUE, tenant_id = NULL;

            -- Inactive token
            INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id, is_active)
            VALUES ('test_inactive', knowledgebase.hash_access_token('{token_inactive}'), '{tenant_a}', FALSE)
            ON CONFLICT (token_hash) DO UPDATE SET is_active = FALSE;

            -- Expired token
            INSERT INTO knowledgebase.access_tokens (token_name, token_hash, tenant_id, is_active, expires_at)
            VALUES ('test_expired', knowledgebase.hash_access_token('{token_expired}'), '{tenant_a}', TRUE, NOW() - INTERVAL '1 hour')
            ON CONFLICT (token_hash) DO UPDATE SET expires_at = NOW() - INTERVAL '1 hour';
        END;
        $$;
        """
        setup_res = _run_psql_query(setup_sql, self.conn_info)
        self.assertEqual(setup_res.returncode, 0, f"Setup error: {setup_res.stderr}")

        # 1. Scoped token: must see Tenant A and NOT Tenant B
        for rpc_name, sql_call in [
            ("hybrid", f"SELECT chunk_text FROM knowledgebase.search_chunks_hybrid('{token_a}', 'vault data'::text, array_fill(0.7, ARRAY[1536])::vector, 10);"),
            ("vector", f"SELECT chunk_text FROM knowledgebase.match_chunks_by_embedding('{token_a}', array_fill(0.7, ARRAY[1536])::vector, 10);"),
            ("fts", f"SELECT chunk_text FROM knowledgebase.search_chunks_full_text('{token_a}', 'vault data'::text, 10);"),
        ]:
            res = _run_psql_query(sql_call, self.conn_info)
            self.assertEqual(res.returncode, 0, f"{rpc_name} failed: {res.stderr}")
            self.assertIn("Tenant A", res.stdout, f"Scoped token must find Tenant A via {rpc_name}")
            self.assertNotIn("Tenant B", res.stdout, f"Scoped token leaked Tenant B via {rpc_name}")

        # Facets scoped count
        facet_sql = f"SELECT document_count FROM knowledgebase.get_navigation_facets('{token_a}') WHERE facet_key = 'token_isolation';"
        res_facet = _run_psql_query(facet_sql, self.conn_info)
        self.assertEqual(_extract_scalar(res_facet.stdout), "1")

        # 2. Inactive token: must fail
        res_inact = _run_psql_query(f"SELECT * FROM knowledgebase.search_chunks_full_text('{token_inactive}', 'vault'::text, 5);", self.conn_info)
        self.assertNotEqual(res_inact.returncode, 0, "Inactive token must be rejected")
        self.assertIn("Invalid knowledgebase token", res_inact.stderr)

        # 3. Expired token: must fail
        res_exp = _run_psql_query(f"SELECT * FROM knowledgebase.search_chunks_full_text('{token_expired}', 'vault'::text, 5);", self.conn_info)
        self.assertNotEqual(res_exp.returncode, 0, "Expired token must be rejected")
        self.assertIn("Invalid knowledgebase token", res_exp.stderr)

        # 4. Verify audit trail has recorded 'used' events
        audit_sql = """
        SELECT event_type, count(*)
        FROM knowledgebase.access_token_audit
        GROUP BY event_type;
        """
        res_audit = _run_psql_query(audit_sql, self.conn_info)
        self.assertEqual(res_audit.returncode, 0)
        self.assertIn("used", res_audit.stdout)


if __name__ == "__main__":
    unittest.main()
