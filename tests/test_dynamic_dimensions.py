import asyncio
import json
import os
import subprocess
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.core.models import ParsedDocument
from supabase_easy_rag.indexes.manager import AsyncIndexManager, IndexManager
from supabase_easy_rag.ingestion.syncer import DocumentSyncer
from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.retrieval.engine import RetrievalEngine


class DummyProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "custom-model"):
        self._model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


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
            if res.returncode == 0:
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
        "-c", sql,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


class TestDynamicVectorDimensionsUnit(unittest.TestCase):
    def test_providers_export_model_name(self):
        provider = DummyProvider("test-model-0.6b")
        self.assertEqual(provider.model_name, "test-model-0.6b")

    def test_syncer_stamps_model_in_chunk_metadata(self):
        mock_postgrest = MagicMock()
        mock_table = MagicMock()
        mock_postgrest.schema.return_value.table.return_value = mock_table

        provider = DummyProvider("Qwen/Qwen3-Embedding-0.6B")
        syncer = DocumentSyncer(
            postgrest_client=mock_postgrest,
            embedding_provider=provider,
        )

        with patch("pathlib.Path.rglob") as mock_rglob, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("supabase_easy_rag.ingestion.syncer.parse_markdown_document") as mock_parse:

            fake_file = Path("/fake/docs/test.md")
            mock_rglob.return_value = [fake_file]

            parsed_doc = ParsedDocument(
                document_key="test_doc",
                title="Test Doc",
                top_level_category="general",
                metadata={"custom": "val"},
                checksum="abc12345",
                content="This is test content for dynamic dimensions.",
                sections=[],
                facets=[],
                facet_path=None,
                token_count=10,
                char_count=45,
            )
            mock_parse.return_value = parsed_doc

            mock_run_res = MagicMock()
            mock_run_res.data = [{"id": "run-001"}]
            mock_table.insert.return_value.execute.return_value = mock_run_res
            mock_table.select.return_value.eq.return_value.execute.return_value.data = []

            # Mock document insert
            mock_table.upsert.return_value.execute.return_value.data = [{"id": "doc-001"}]

            syncer.sync_directory(source_root=Path("/fake/docs"))

            # Check chunk payload includes model
            chunk_upsert_calls = [
                call for call in mock_table.upsert.call_args_list
            ]
            self.assertTrue(len(chunk_upsert_calls) >= 1)
            chunk_payloads = None
            for call in chunk_upsert_calls:
                args = call[0]
                if isinstance(args[0], list) and len(args[0]) > 0 and "embedding" in args[0][0]:
                    chunk_payloads = args[0]
                    break

            self.assertIsNotNone(chunk_payloads)
            self.assertEqual(chunk_payloads[0]["metadata"]["model"], "Qwen/Qwen3-Embedding-0.6B")

    def test_retrieval_engine_forwards_model_name(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        provider = DummyProvider("Qwen/Qwen3-Embedding-0.6B")
        engine = RetrievalEngine(
            postgrest_client=mock_postgrest,
            embedding_provider=provider,
        )

        engine.search_vector(
            query="test query",
            kb_token="token-xyz",
            model_name="custom-override-model",
        )

        called_params = mock_postgrest.schema.return_value.rpc.call_args[0][1]
        self.assertEqual(called_params.get("p_model_name"), "custom-override-model")

    def test_index_manager_methods_sync_and_async(self):
        # Sync
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = [
            {"index_name": "idx_kb_chunks_hnsw_g_1024", "tenant_id": None, "dimension": 1024, "is_global": True, "index_def": "..."}
        ]

        mgr = IndexManager(postgrest_client=mock_postgrest)
        indexes = mgr.list_vector_indexes()
        self.assertEqual(len(indexes), 1)
        self.assertEqual(indexes[0]["dimension"], 1024)

        mock_rpc.execute.return_value.data = "Created index idx_kb_chunks_hnsw_g_768"
        res_ensure = mgr.ensure_vector_index(768)
        self.assertIn("Created index", res_ensure)

        mock_rpc.execute.return_value.data = "Created index idx_kb_chunks_hnsw_t_11111111_1111_1111_1111_111111111111_768"
        test_t_id = "11111111-1111-1111-1111-111111111111"
        res_ensure_tenant = mgr.ensure_vector_index(768, tenant_id=test_t_id)
        self.assertIn("Created index", res_ensure_tenant)

        mock_rpc.execute.return_value.data = "Dropped index idx_kb_chunks_hnsw_g_768"
        res_drop = mgr.drop_vector_index(768)
        self.assertIn("Dropped index", res_drop)

        # Async
        async def run_async_test():
            mock_async_postgrest = MagicMock()
            mock_async_rpc = MagicMock()
            mock_async_postgrest.schema.return_value.rpc.return_value = mock_async_rpc
            mock_async_rpc.execute = AsyncMock(return_value=MagicMock(data=[
                {"index_name": "idx_kb_chunks_hnsw_g_1024", "dimension": 1024}
            ]))

            async_mgr = AsyncIndexManager(postgrest_client=mock_async_postgrest)
            res_async = await async_mgr.list_vector_indexes()
            self.assertEqual(len(res_async), 1)

            mock_async_rpc.execute = AsyncMock(return_value=MagicMock(data="Created index idx_kb_chunks_hnsw_g_384"))
            res_async_ensure = await async_mgr.ensure_vector_index(384)
            self.assertIn("Created index", res_async_ensure)

            mock_async_rpc.execute = AsyncMock(return_value=MagicMock(data="Dropped index idx_kb_chunks_hnsw_g_384"))
            res_async_drop = await async_mgr.drop_vector_index(384)
            self.assertIn("Dropped index", res_async_drop)

        asyncio.run(run_async_test())


class TestDynamicVectorDimensionsLivePostgres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = _get_postgres_conn_info()
        if not cls.conn:
            raise unittest.SkipTest("No PostgreSQL instance available for live tests")

    def test_live_dynamic_index_creation_and_listing(self):
        tenant_uuid = str(uuid.uuid4())

        # 1. Ensure global index 768
        res = _run_psql_query(
            "SELECT knowledgebase.ensure_vector_index(768);",
            self.conn,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("idx_kb_chunks_hnsw_g_768", res.stdout)

        # 2. Ensure tenant index 768
        res_tenant = _run_psql_query(
            f"SELECT knowledgebase.ensure_vector_index(768, '{tenant_uuid}'::uuid);",
            self.conn,
        )
        self.assertEqual(res_tenant.returncode, 0)
        tenant_sanitized = tenant_uuid.replace("-", "_")
        self.assertIn(f"idx_kb_chunks_hnsw_t_{tenant_sanitized}_768", res_tenant.stdout)

        # 3. List indexes
        res_list = _run_psql_query(
            "SELECT index_name, tenant_id, dimension, is_global FROM knowledgebase.list_vector_indexes();",
            self.conn,
        )
        self.assertEqual(res_list.returncode, 0)
        self.assertIn("idx_kb_chunks_hnsw_g_768", res_list.stdout)
        self.assertIn(f"idx_kb_chunks_hnsw_t_{tenant_sanitized}_768", res_list.stdout)

        # 4. Drop tenant index 768
        res_drop_tenant = _run_psql_query(
            f"SELECT knowledgebase.drop_vector_index(768, '{tenant_uuid}'::uuid);",
            self.conn,
        )
        self.assertEqual(res_drop_tenant.returncode, 0)
        self.assertIn(f"Dropped index idx_kb_chunks_hnsw_t_{tenant_sanitized}_768", res_drop_tenant.stdout)

        # 5. Drop global index 768
        res_drop = _run_psql_query(
            "SELECT knowledgebase.drop_vector_index(768);",
            self.conn,
        )
        self.assertEqual(res_drop.returncode, 0)
        self.assertIn("Dropped index idx_kb_chunks_hnsw_g_768", res_drop.stdout)

    def test_live_multi_dimension_isolation_and_model_scoping(self):
        # Clean up existing test docs and create a test token
        setup_sql = """
        DELETE FROM knowledgebase.documents WHERE document_key IN ('test_openai_1536', 'test_qwen_1024');
        DELETE FROM knowledgebase.access_tokens WHERE token_name = 'test_dim_token';

        INSERT INTO knowledgebase.access_tokens (token_name, token_hash, is_active)
        VALUES ('test_dim_token', knowledgebase.hash_access_token('test_dim_secret'), TRUE)
        ON CONFLICT (token_hash) DO NOTHING;
        """
        _run_psql_query(setup_sql, self.conn)

        # Insert 1536-dim document and chunk
        vec_1536 = "[" + ", ".join(["0.01"] * 1536) + "]"
        sql_1536 = f"""
        INSERT INTO knowledgebase.documents (id, document_key, title, top_level_category, checksum)
        VALUES ('11111111-1111-1111-1111-111111111111', 'test_openai_1536', 'OpenAI 1536 Doc', 'docs', 'chk1')
        ON CONFLICT (document_key) DO NOTHING;

        INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding, metadata)
        VALUES (
            '11111111-1111-1111-1111-111111111111',
            0,
            'This chunk was embedded using OpenAI 1536-dimensional vector.',
            '{vec_1536}'::vector,
            jsonb_build_object('model', 'text-embedding-3-small')
        )
        ON CONFLICT (document_id, chunk_index) DO NOTHING;
        """
        res_ins1 = _run_psql_query(sql_1536, self.conn)
        self.assertEqual(res_ins1.returncode, 0, f"Failed to insert 1536-dim: {res_ins1.stderr}")

        # Insert 1024-dim document and chunk
        vec_1024 = "[" + ", ".join(["0.02"] * 1024) + "]"
        sql_1024 = f"""
        INSERT INTO knowledgebase.documents (id, document_key, title, top_level_category, checksum)
        VALUES ('22222222-2222-2222-2222-222222222222', 'test_qwen_1024', 'Qwen 1024 Doc', 'docs', 'chk2')
        ON CONFLICT (document_key) DO NOTHING;

        INSERT INTO knowledgebase.chunks (document_id, chunk_index, content, embedding, metadata)
        VALUES (
            '22222222-2222-2222-2222-222222222222',
            0,
            'This chunk was embedded using Qwen 1024-dimensional vector.',
            '{vec_1024}'::vector,
            jsonb_build_object('model', 'Qwen/Qwen3-Embedding-0.6B')
        )
        ON CONFLICT (document_id, chunk_index) DO NOTHING;
        """
        res_ins2 = _run_psql_query(sql_1024, self.conn)
        self.assertEqual(res_ins2.returncode, 0, f"Failed to insert 1024-dim: {res_ins2.stderr}")

        # 1. Search with 1024-dim query: MUST succeed and return ONLY the 1024-dim document
        q_1024 = "[" + ", ".join(["0.02"] * 1024) + "]"
        sql_search_1024 = f"""
        SELECT document_title, vector_score FROM knowledgebase.match_chunks_by_embedding(
            p_kb_token => 'test_dim_secret',
            p_query_embedding => '{q_1024}'::vector,
            p_match_count => 5
        );
        """
        res_s1024 = _run_psql_query(sql_search_1024, self.conn)
        self.assertEqual(res_s1024.returncode, 0, f"1024 search failed: {res_s1024.stderr}")
        self.assertIn("Qwen 1024 Doc", res_s1024.stdout)
        self.assertNotIn("OpenAI 1536 Doc", res_s1024.stdout)

        # 2. Search with 1536-dim query: MUST succeed and return ONLY the 1536-dim document
        q_1536 = "[" + ", ".join(["0.01"] * 1536) + "]"
        sql_search_1536 = f"""
        SELECT document_title, vector_score FROM knowledgebase.match_chunks_by_embedding(
            p_kb_token => 'test_dim_secret',
            p_query_embedding => '{q_1536}'::vector,
            p_match_count => 5
        );
        """
        res_s1536 = _run_psql_query(sql_search_1536, self.conn)
        self.assertEqual(res_s1536.returncode, 0, f"1536 search failed: {res_s1536.stderr}")
        self.assertIn("OpenAI 1536 Doc", res_s1536.stdout)
        self.assertNotIn("Qwen 1024 Doc", res_s1536.stdout)

        # 3. Model filter isolation: when filtering by model 'text-embedding-3-small', 1024 query returns 0 rows
        sql_model_filter = f"""
        SELECT document_title FROM knowledgebase.match_chunks_by_embedding(
            p_kb_token => 'test_dim_secret',
            p_query_embedding => '{q_1024}'::vector,
            p_match_count => 5,
            p_model_name => 'text-embedding-3-small'
        );
        """
        res_mfilter = _run_psql_query(sql_model_filter, self.conn)
        self.assertEqual(res_mfilter.returncode, 0)
        self.assertNotIn("Qwen 1024 Doc", res_mfilter.stdout)
        self.assertNotIn("OpenAI 1536 Doc", res_mfilter.stdout)

        # 4. Hybrid Search with RRF on mixed database
        sql_hybrid = f"""
        SELECT document_title, hybrid_score FROM knowledgebase.search_chunks_hybrid(
            p_kb_token => 'test_dim_secret',
            p_query => 'Qwen 1024',
            p_query_embedding => '{q_1024}'::vector,
            p_match_count => 5
        );
        """
        res_hyb = _run_psql_query(sql_hybrid, self.conn)
        self.assertEqual(res_hyb.returncode, 0, f"Hybrid search failed: {res_hyb.stderr}")
        self.assertIn("Qwen 1024 Doc", res_hyb.stdout)

        # 5. RLS Variant search
        sql_rls = f"""
        SELECT document_title, vector_score FROM knowledgebase.match_chunks_by_embedding_rls(
            p_query_embedding => '{q_1024}'::vector,
            p_match_count => 5
        );
        """
        res_rls = _run_psql_query(sql_rls, self.conn)
        self.assertEqual(res_rls.returncode, 0, f"RLS search failed: {res_rls.stderr}")
        self.assertIn("Qwen 1024 Doc", res_rls.stdout)

        # Clean up test rows
        _run_psql_query(
            "DELETE FROM knowledgebase.documents WHERE document_key IN ('test_openai_1536', 'test_qwen_1024'); DELETE FROM knowledgebase.access_tokens WHERE token_name = 'test_dim_token';",
            self.conn,
        )


if __name__ == "__main__":
    unittest.main()
