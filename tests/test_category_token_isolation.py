import unittest
from unittest.mock import MagicMock

from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.retrieval.engine import RetrievalEngine


class TestCategoryAndTokenIsolation(unittest.TestCase):
    """Verifies category isolation between User JWT and KB tokens.

    - System documents (top_level_category = 'system') are hidden from standard User JWT.
    - KB tokens with explicit allowed_categories (e.g. ['system']) can access system documents.
    - Global documents (tenant_id IS NULL) are accessible when include_global is True.
    """

    def setUp(self):
        self.tenant_id = "tenant-111"
        self.scope_id = "scope-222"

    def test_retrieval_engine_passes_category_and_multitenancy_parameters(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1, 0.2]

        engine = RetrievalEngine(postgrest_client=mock_postgrest, embedding_provider=mock_provider)

        # 1. Search with User RLS mode (tenant-scoped, no system category by default)
        engine.search_hybrid(
            query="test query",
            use_rls=True,
            tenant_id=self.tenant_id,
            scope_id=self.scope_id,
            include_global=True,
            allowed_categories=["general", "knowledge"],
        )

        mock_postgrest.schema.return_value.rpc.assert_called_with(
            "search_chunks_hybrid_rls",
            {
                "p_query": "test query",
                "p_query_embedding": [0.1, 0.2],
                "p_match_count": 5,
                "p_facet_keys": None,
                "p_candidate_count": None,
                "p_rrf_k": 60,
                "p_vector_weight": 1.0,
                "p_text_weight": 1.0,
                "p_fts_config": "english",
                "p_min_vector_similarity": None,
                "p_tenant_id": self.tenant_id,
                "p_scope_id": self.scope_id,
                "p_include_global": True,
                "p_allowed_categories": ["general", "knowledge"],
            },
        )

        # 2. Search with KB token (can access system category)
        mock_postgrest.reset_mock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        engine.search_hybrid(
            query="admin query",
            kb_token="kb_sys_token_999",
            use_rls=False,
            allowed_categories=["system"],
        )

        mock_postgrest.schema.return_value.rpc.assert_called_with(
            "search_chunks_hybrid",
            {
                "p_kb_token": "kb_sys_token_999",
                "p_query": "admin query",
                "p_query_embedding": [0.1, 0.2],
                "p_match_count": 5,
                "p_facet_keys": None,
                "p_candidate_count": None,
                "p_rrf_k": 60,
                "p_vector_weight": 1.0,
                "p_text_weight": 1.0,
                "p_fts_config": "english",
                "p_min_vector_similarity": None,
                "p_allowed_categories": ["system"],
            },
        )

    def test_client_search_forwards_category_and_scope(self):
        mock_provider = MagicMock()
        mock_retrieval = MagicMock()

        from supabase_easy_rag.config import EasyRagConfig, ProviderConfig

        mock_config = EasyRagConfig(
            supabase_url="https://example.supabase.co",
            supabase_service_role_key="srv-key",
            supabase_anon_key="anon-key-123",
            knowledgebase_access_token="",
            schema_name="knowledgebase",
            embedding=ProviderConfig("dummy", "m", None, "k"),
            chat_nano=ProviderConfig("dummy", "m", None, "k"),
            chat_mini=ProviderConfig("dummy", "m", None, "k"),
            azure_nano=ProviderConfig("dummy", "m", None, "k"),
            azure_mini=ProviderConfig("dummy", "m", None, "k"),
            azure_embedding=ProviderConfig("dummy", "m", None, "k"),
            embedding_model="m",
            embedding_dim=1536,
            batch_size=10,
            default_match_count=5,
            use_rls=True,
            enable_chunking=True,
            chunk_size=500,
            chunk_overlap=50,
            openai_api_key="",
            openai_endpoint=None,
            openai_api_version=None,
        )

        client = EasyRagClient(
            supabase_url="https://example.supabase.co",
            supabase_key="anon-key-123",
            embedding_provider=mock_provider,
            config=mock_config,
            tenant_id=self.tenant_id,
            scope_id=self.scope_id,
            include_global=False,
            use_rls=True,
        )
        client.retrieval = mock_retrieval

        client.search_vector(
            query="find docs",
            allowed_categories=["guides"],
        )

        mock_retrieval.search_vector.assert_called_with(
            query="find docs",
            kb_token=None,
            match_count=5,
            facet_keys=None,
            min_vector_similarity=None,
            use_rls=True,
            expand_context=None,
            tenant_id=self.tenant_id,
            scope_id=self.scope_id,
            include_global=False,
            allowed_categories=["guides"],
        )


if __name__ == "__main__":
    unittest.main()
