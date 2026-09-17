import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.ingestion.syncer import DocumentSyncer


class TestMultitenancyScopedIngestion(unittest.TestCase):
    def setUp(self):
        self.tenant_id = "11111111-2222-3333-4444-555555555555"
        self.scope_id = "66666666-7777-8888-9999-000000000000"
        self.user_id = "user-123"

    def test_syncer_accepts_and_forwards_multitenancy_scope(self):
        from supabase_easy_rag.core.models import ParsedDocument

        mock_postgrest = MagicMock()
        mock_table = MagicMock()
        mock_postgrest.schema.return_value.table.return_value = mock_table

        mock_provider = MagicMock()
        mock_provider.embed_texts.return_value = [[0.1, 0.2, 0.3]]

        syncer = DocumentSyncer(
            postgrest_client=mock_postgrest,
            embedding_provider=mock_provider,
        )

        with patch("pathlib.Path.rglob") as mock_rglob, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("supabase_easy_rag.ingestion.syncer.parse_markdown_document") as mock_parse:

            fake_file = Path("/fake/docs/user_guide.md")
            mock_rglob.return_value = [fake_file]

            parsed_doc = ParsedDocument(
                document_key="user_guide",
                title="User Guide",
                top_level_category="docs",
                metadata={},
                checksum="hash-123",
                content="# User Guide\nContent",
                sections=[],
                facets=[],
                facet_path=None,
                token_count=10,
                char_count=20,
            )
            mock_parse.return_value = parsed_doc

            # Mock ingestion_runs insert
            mock_run_res = MagicMock()
            mock_run_res.data = [{"id": "run-999"}]
            mock_table.insert.return_value.execute.return_value = mock_run_res
            mock_table.select.return_value.eq.return_value.execute.return_value.data = []

            res = syncer.sync_directory(
                source_root=Path("/fake/docs"),
                owner_id=self.user_id,
                tenant_id=self.tenant_id,
                scope_id=self.scope_id,
            )

            self.assertEqual(res["status"], "completed")

    def test_client_for_user_preserves_and_overrides_scope(self):
        client = EasyRagClient(
            supabase_url="https://example.supabase.co",
            supabase_key="anon-key-123",
            tenant_id=self.tenant_id,
            scope_id=self.scope_id,
            include_global=True,
            use_rls=True,
        )

        self.assertEqual(client.tenant_id, self.tenant_id)
        self.assertEqual(client.scope_id, self.scope_id)
        self.assertTrue(client.include_global)

        # for_user inheriting scope
        user_client = client.for_user(user_jwt="jwt.token.abc")
        self.assertEqual(user_client.tenant_id, self.tenant_id)
        self.assertEqual(user_client.scope_id, self.scope_id)
        self.assertTrue(user_client.include_global)
        self.assertEqual(user_client.user_jwt, "jwt.token.abc")

        # for_user overriding scope
        custom_tenant = "99999999-9999-9999-9999-999999999999"
        custom_user_client = client.for_user(
            user_jwt="jwt.token.abc",
            tenant_id=custom_tenant,
            scope_id=None,
            include_global=False,
        )
        self.assertEqual(custom_user_client.tenant_id, custom_tenant)
        self.assertIsNone(custom_user_client.scope_id)
        self.assertFalse(custom_user_client.include_global)


if __name__ == "__main__":
    unittest.main()
