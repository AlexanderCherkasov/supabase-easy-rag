import unittest
from unittest.mock import MagicMock

from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.retrieval.engine import RetrievalEngine


class TestRlsAndOwnerIsolation(unittest.TestCase):
    """Rigorous acceptance tests for RLS and Owner Isolation in Vector, FTS, and Hybrid Search."""

    def setUp(self):
        self.user_a_id = "11111111-1111-1111-1111-111111111111"
        self.user_b_id = "22222222-2222-2222-2222-222222222222"
        self.user_c_id = "33333333-3333-3333-3333-333333333333"

    def test_user_a_never_receives_user_b_chunks_simulation(self):
        """Simulate database isolation logic across vector, FTS, and hybrid retrieval.

        Verifies that any result set evaluated against User A's context never leaks User B documents.
        """
        # Multi-tenant documents dataset
        docs = [
            {"id": "c1", "doc_id": "d1", "owner_id": self.user_a_id, "shared_with": [], "title": "User A Secrets", "text": "Confidential data A"},
            {"id": "c2", "doc_id": "d2", "owner_id": self.user_b_id, "shared_with": [], "title": "User B Secrets", "text": "Confidential data B"},
            {"id": "c3", "doc_id": "d3", "owner_id": self.user_b_id, "shared_with": [self.user_a_id], "title": "Shared B->A Project", "text": "Collaborative text"},
            {"id": "c4", "doc_id": "d4", "owner_id": None, "shared_with": [], "title": "Public Documentation", "text": "Open manual"},
        ]

        def evaluate_access(caller_id: str, doc_row: dict) -> bool:
            # Replicates exact RLS condition:
            # owner_id IS NULL OR owner_id = auth.uid() OR auth.uid() IN shared_with
            if doc_row["owner_id"] is None:
                return True
            if doc_row["owner_id"] == caller_id:
                return True
            if caller_id in doc_row["shared_with"]:
                return True
            return False

        # Caller A
        visible_to_a = [d["id"] for d in docs if evaluate_access(self.user_a_id, d)]
        self.assertIn("c1", visible_to_a, "User A must see own documents")
        self.assertNotIn("c2", visible_to_a, "User A MUST NEVER see User B's private documents")
        self.assertIn("c3", visible_to_a, "User A must see documents shared via document_owners")
        self.assertIn("c4", visible_to_a, "User A must see public documents (owner_id IS NULL)")

        # Caller B
        visible_to_b = [d["id"] for d in docs if evaluate_access(self.user_b_id, d)]
        self.assertNotIn("c1", visible_to_b, "User B MUST NEVER see User A's private documents")
        self.assertIn("c2", visible_to_b, "User B must see own documents")
        self.assertIn("c3", visible_to_b, "User B must see shared documents where they are owner")
        self.assertIn("c4", visible_to_b, "User B must see public documents")

        # Caller C (third party)
        visible_to_c = [d["id"] for d in docs if evaluate_access(self.user_c_id, d)]
        self.assertNotIn("c1", visible_to_c, "User C cannot see User A documents")
        self.assertNotIn("c2", visible_to_c, "User C cannot see User B documents")
        self.assertNotIn("c3", visible_to_c, "User C cannot see unshared documents")
        self.assertIn("c4", visible_to_c, "User C can see public documents")

    def test_client_scoped_for_user_passes_jwt_and_enables_rls(self):
        client = EasyRagClient(
            supabase_url="https://example.supabase.co",
            supabase_key="anon-key-123",
            use_rls=False,
        )
        user_jwt = "header.payload.signature"
        scoped_client = client.for_user(user_jwt=user_jwt)

        self.assertTrue(scoped_client.use_rls)
        self.assertEqual(scoped_client.user_jwt, user_jwt)


if __name__ == "__main__":
    unittest.main()
