import asyncio
import unittest
from unittest.mock import MagicMock

from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.core.client import AsyncEasyRagClient, EasyRagClient
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.retrieval.engine import (
    AsyncRetrievalEngine,
    RetrievalEngine,
    _parse_search_results,
)


class TestClientParametersAndParsing(unittest.TestCase):
    def test_parse_search_results_with_diagnostic_fields(self):
        raw_data = [
            {
                "chunk_id": "c-101",
                "document_id": "d-202",
                "document_title": "Architecture Overview",
                "section_title": "Database Layer",
                "chunk_text": "PostgreSQL with pgvector and RRF fusion.",
                "facet_path": "docs/architecture",
                "metadata": {"tags": ["database", "rag"]},
                "vector_score": 0.88,
                "text_score": 0.45,
                "hybrid_score": 0.0245,
                "vector_rank": 1,
                "text_rank": 3,
            }
        ]
        results = _parse_search_results(raw_data)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertIsInstance(res, SearchResult)
        self.assertEqual(res.chunk_id, "c-101")
        self.assertEqual(res.document_id, "d-202")
        self.assertEqual(res.document_title, "Architecture Overview")
        self.assertEqual(res.section_title, "Database Layer")
        self.assertEqual(res.chunk_text, "PostgreSQL with pgvector and RRF fusion.")
        self.assertEqual(res.facet_path, "docs/architecture")
        self.assertEqual(res.metadata, {"tags": ["database", "rag"]})
        self.assertEqual(res.vector_score, 0.88)
        self.assertEqual(res.text_score, 0.45)
        self.assertEqual(res.hybrid_score, 0.0245)
        self.assertEqual(res.vector_rank, 1)
        self.assertEqual(res.text_rank, 3)
        self.assertEqual(res.final_score, 0.0245)
        self.assertEqual(res.vector_similarity, 0.88)

    def test_sync_engine_search_hybrid_parameter_forwarding(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1, 0.2, 0.3]

        engine = RetrievalEngine(postgrest_client=mock_postgrest, embedding_provider=mock_provider)
        engine.search_hybrid(
            query="hybrid search query",
            kb_token="token_abc",
            match_count=7,
            facet_keys=["category/databases"],
            candidate_count=70,
            rrf_k=50,
            vector_weight=1.5,
            text_weight=0.8,
            fts_config="russian",
            min_vector_similarity=0.65,
            use_rls=False,
        )

        mock_postgrest.schema.assert_called_with("knowledgebase")
        mock_postgrest.schema.return_value.rpc.assert_called_once_with(
            "search_chunks_hybrid",
            {
                "p_kb_token": "token_abc",
                "p_query": "hybrid search query",
                "p_query_embedding": [0.1, 0.2, 0.3],
                "p_match_count": 7,
                "p_facet_keys": ["category/databases"],
                "p_candidate_count": 70,
                "p_rrf_k": 50,
                "p_vector_weight": 1.5,
                "p_text_weight": 0.8,
                "p_fts_config": "russian",
                "p_min_vector_similarity": 0.65,
            },
        )

    def test_sync_engine_search_hybrid_rls_parameter_forwarding(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.4, 0.5, 0.6]

        engine = RetrievalEngine(postgrest_client=mock_postgrest, embedding_provider=mock_provider)
        engine.search_hybrid(
            query="rls query",
            kb_token=None,
            match_count=10,
            use_rls=True,
            candidate_count=100,
            rrf_k=60,
            vector_weight=1.0,
            text_weight=1.0,
            fts_config="portuguese",
            min_vector_similarity=0.7,
        )

        mock_postgrest.schema.return_value.rpc.assert_called_once_with(
            "search_chunks_hybrid_rls",
            {
                "p_query": "rls query",
                "p_query_embedding": [0.4, 0.5, 0.6],
                "p_match_count": 10,
                "p_facet_keys": None,
                "p_candidate_count": 100,
                "p_rrf_k": 60,
                "p_vector_weight": 1.0,
                "p_text_weight": 1.0,
                "p_fts_config": "portuguese",
                "p_min_vector_similarity": 0.7,
            },
        )

    def test_async_engine_search_hybrid_parameter_forwarding(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc

        async def dummy_execute():
            res = MagicMock()
            res.data = []
            return res

        mock_rpc.execute.side_effect = dummy_execute

        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.11, 0.22]

        engine = AsyncRetrievalEngine(postgrest_client=mock_postgrest, embedding_provider=mock_provider)
        asyncio.run(
            engine.search_hybrid(
                query="async query",
                kb_token="async_token",
                match_count=3,
                candidate_count=30,
                rrf_k=40,
                vector_weight=2.0,
                text_weight=0.5,
                fts_config="simple",
                min_vector_similarity=0.5,
                use_rls=False,
            )
        )

        mock_postgrest.schema.return_value.rpc.assert_called_once_with(
            "search_chunks_hybrid",
            {
                "p_kb_token": "async_token",
                "p_query": "async query",
                "p_query_embedding": [0.11, 0.22],
                "p_match_count": 3,
                "p_facet_keys": None,
                "p_candidate_count": 30,
                "p_rrf_k": 40,
                "p_vector_weight": 2.0,
                "p_text_weight": 0.5,
                "p_fts_config": "simple",
                "p_min_vector_similarity": 0.5,
            },
        )

    def test_sync_engine_search_vector_with_min_similarity(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1, 0.2]

        engine = RetrievalEngine(postgrest_client=mock_postgrest, embedding_provider=mock_provider)
        engine.search_vector(
            query="vector threshold query",
            kb_token="token_xyz",
            match_count=4,
            min_vector_similarity=0.75,
            use_rls=False,
        )

        mock_postgrest.schema.return_value.rpc.assert_called_once_with(
            "match_chunks_by_embedding",
            {
                "p_kb_token": "token_xyz",
                "p_query_embedding": [0.1, 0.2],
                "p_match_count": 4,
                "p_facet_keys": None,
                "p_min_vector_similarity": 0.75,
            },
        )

    def test_sync_engine_search_fts_with_config(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = []

        engine = RetrievalEngine(postgrest_client=mock_postgrest)
        engine.search_fts(
            query="fts query",
            kb_token="token_xyz",
            match_count=8,
            fts_config="russian",
            use_rls=False,
        )

        mock_postgrest.schema.return_value.rpc.assert_called_once_with(
            "search_chunks_full_text",
            {
                "p_kb_token": "token_xyz",
                "p_query": "fts query",
                "p_match_count": 8,
                "p_facet_keys": None,
                "p_fts_config": "russian",
            },
        )


if __name__ == "__main__":
    unittest.main()
