import unittest
from unittest.mock import MagicMock

from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.retrieval.engine import RetrievalEngine, _parse_search_results


class TestRetrieval(unittest.TestCase):
    def test_parse_search_results(self):
        data = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "document_title": "Doc 1",
                "section_title": "Sec 1",
                "chunk_text": "Sample text",
                "facet_path": "path/to",
                "metadata": {"key": "val"},
                "hybrid_score": 0.85,
            }
        ]
        results = _parse_search_results(data)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertEqual(results[0].hybrid_score, 0.85)

    def test_retrieval_engine_search_fts(self):
        mock_postgrest = MagicMock()
        mock_rpc = MagicMock()
        mock_postgrest.schema.return_value.rpc.return_value = mock_rpc
        mock_rpc.execute.return_value.data = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "document_title": "Test Doc",
                "chunk_text": "Matching content",
                "text_score": 0.5,
            }
        ]

        engine = RetrievalEngine(postgrest_client=mock_postgrest)
        res = engine.search_fts(query="test", kb_token="test_token")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].document_title, "Test Doc")
        mock_postgrest.schema.assert_called_with("knowledgebase")


if __name__ == "__main__":
    unittest.main()
