import unittest
import uuid
from unittest.mock import MagicMock

from supabase_easy_rag.config import EasyRagConfig, ProviderConfig
from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.core.exceptions import EasyRagConfigurationError
from supabase_easy_rag.security.tokens import TokenManager, generate_secure_token, hash_token


class TestSecurity(unittest.TestCase):
    def test_token_generation_and_hashing(self):
        token = generate_secure_token(prefix="test_")
        self.assertTrue(token.startswith("test_"))
        self.assertGreater(len(token), 20)

        token_hash = hash_token(token)
        self.assertEqual(len(token_hash), 64)
        self.assertEqual(token_hash, hash_token(token))

    def test_revoke_token_by_uuid(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.schema.return_value.table.return_value = mock_table
        mock_update = MagicMock()
        mock_table.update.return_value = mock_update
        mock_eq = MagicMock()
        mock_update.eq.return_value = mock_eq
        mock_eq.execute.return_value.data = [{"id": "123e4567-e89b-12d3-a456-426614174000"}]

        manager = TokenManager(postgrest_client=mock_client)
        test_uuid = str(uuid.uuid4())
        res = manager.revoke_token(test_uuid)

        self.assertTrue(res)
        mock_table.update.assert_called_once_with({"is_active": False})
        mock_update.eq.assert_called_once_with("id", test_uuid)

    def test_revoke_token_by_name_and_injection_safety(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.schema.return_value.table.return_value = mock_table
        mock_update = MagicMock()
        mock_table.update.return_value = mock_update
        mock_eq = MagicMock()
        mock_update.eq.return_value = mock_eq
        mock_eq.execute.return_value.data = [{"id": "some-id"}]

        manager = TokenManager(postgrest_client=mock_client)
        # Even with injection syntax containing comma and operators
        injection_attempt = "prod_token,is_active.eq.true"
        res = manager.revoke_token(injection_attempt)

        self.assertTrue(res)
        mock_table.update.assert_called_once_with({"is_active": False})
        mock_update.eq.assert_called_once_with("token_name", injection_attempt)

    def test_client_rls_requires_anon_key(self):
        dummy_provider = ProviderConfig(provider="openai_like", model="test", endpoint="http://localhost", api_key="sk-test")
        cfg = EasyRagConfig(
            supabase_url="http://localhost:54321",
            supabase_service_role_key="secret-service-role-key",
            supabase_anon_key="",  # empty anon key
            knowledgebase_access_token="",
            schema_name="knowledgebase",
            embedding=dummy_provider,
            chat_nano=dummy_provider,
            chat_mini=dummy_provider,
            azure_nano=dummy_provider,
            azure_mini=dummy_provider,
            azure_embedding=dummy_provider,
            embedding_model="test",
            embedding_dim=1536,
            batch_size=20,
            default_match_count=5,
            use_rls=False,
            enable_chunking=True,
            chunk_size=800,
            chunk_overlap=100,
            openai_api_key="sk-test",
            openai_endpoint=None,
            openai_api_version=None,
        )
        # Enabling RLS mode without anon_key must raise EasyRagConfigurationError
        with self.assertRaises(EasyRagConfigurationError):
            EasyRagClient(config=cfg, use_rls=True)


if __name__ == "__main__":
    unittest.main()
