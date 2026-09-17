"""Live End-to-End Client Test Suite for EasyRagClient and AsyncEasyRagClient.

Tests real PostgREST HTTP calls, directory ingestion, incremental sync, hybrid search,
context expansion (section/document), user scoping, access token management, and facets.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid
from collections.abc import Sequence
from pathlib import Path

from supabase_easy_rag.config import EasyRagConfig, ProviderConfig
from supabase_easy_rag.core.client import AsyncEasyRagClient, EasyRagClient
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.providers.base import BaseEmbeddingProvider


class MockDeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic embedding provider for repeatable E2E tests."""

    def __init__(self, dimension: int = 1536):
        self._dim = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            val = float(abs(hash(text)) % 1000) / 1000.0
            vec = [val] * self._dim
            norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings


LOCAL_SUPABASE_URL = "http://127.0.0.1:54321"
LOCAL_SERVICE_ROLE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
)
LOCAL_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0"
)


def _get_live_supabase_config():
    supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("API_URL") or LOCAL_SUPABASE_URL
    if "127.0.0.1" in supabase_url or "localhost" in supabase_url:
        return supabase_url, LOCAL_SERVICE_ROLE_KEY, LOCAL_ANON_KEY
    service_role_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SERVICE_ROLE_KEY")
        or LOCAL_SERVICE_ROLE_KEY
    )
    anon_key = (
        os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("ANON_KEY")
        or LOCAL_ANON_KEY
    )
    return supabase_url, service_role_key, anon_key


def _is_postgrest_reachable() -> bool:
    try:
        import httpx

        url, _, _ = _get_live_supabase_config()
        r = httpx.get(f"{url}/rest/v1/", timeout=1.0)
        return r.status_code in (200, 401, 404)
    except Exception:
        return False


@unittest.skipUnless(_is_postgrest_reachable(), "Live Supabase PostgREST endpoint not reachable")
class TestLiveClientE2E(unittest.IsolatedAsyncioTestCase):
    """Live E2E test suite for EasyRagClient and AsyncEasyRagClient."""

    @classmethod
    def setUpClass(cls):
        cls.url, cls.service_key, cls.anon_key = _get_live_supabase_config()
        cls.provider = MockDeterministicEmbeddingProvider(dimension=1536)
        cls.config = EasyRagConfig(
            supabase_url=cls.url,
            supabase_service_role_key=cls.service_key,
            supabase_anon_key=cls.anon_key,
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
            use_rls=False,
            enable_chunking=True,
            chunk_size=500,
            chunk_overlap=50,
            openai_api_key="",
            openai_endpoint=None,
            openai_api_version=None,
        )

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="easy_rag_e2e_")
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)
        self.client = EasyRagClient(
            supabase_url=self.url,
            supabase_key=self.service_key,
            embedding_provider=self.provider,
            config=self.config,
            use_rls=False,
        )

    def test_live_token_lifecycle_and_search(self):
        """Create token, search with token, and revoke token via PostgREST."""
        raw_token, token_row = self.client.tokens.create_token(
            name=f"test_live_client_tok_{uuid.uuid4().hex[:6]}",
            metadata={"purpose": "e2e_testing"},
        )
        self.assertIn("id", token_row)
        self.assertTrue(raw_token.startswith("kb_live_"))

        results = self.client.search_hybrid(
            query="knowledgebase test query",
            kb_token=raw_token,
            match_count=5,
        )
        self.assertIsInstance(results, list)

        revoked = self.client.tokens.revoke_token(token_row["id"])
        self.assertTrue(revoked)

    def test_live_sync_directory_incremental_and_expansion(self):
        """Creates sample markdown docs, syncs them live, checks incremental skipping, and tests context expansion."""
        raw_token, token_row = self.client.tokens.create_token(
            name=f"test_sync_expansion_tok_{uuid.uuid4().hex[:6]}",
            metadata={"purpose": "sync_expansion"},
        )

        u_id = uuid.uuid4().hex[:6]
        doc1 = Path(self.test_dir) / f"database_guide_{u_id}.md"
        doc1.write_text(
            f"""---
title: PostgreSQL Distributed Guide {u_id}
category: databases
---

# PostgreSQL Distributed Guide {u_id}

Overview of distributed relational databases and consensus.

## Raft Consensus Architecture

Raft consensus ensures strong consistency across all cluster replicas.
When a leader node fails, the candidate election begins immediately.

## Failover Mechanisms

Failover is coordinated via heartbeats and split-brain prevention guards.
""",
            encoding="utf-8",
        )

        doc2 = Path(self.test_dir) / f"ai_agents_guide_{u_id}.md"
        doc2.write_text(
            f"""---
title: Autonomous AI Agent Architecture {u_id}
category: ai
---

# Autonomous AI Agent Architecture {u_id}

Design principles for multi-agent reasoning and tool use.

## Planning and Reflection

Agents decompose complex goals into discrete execution steps.
Reflection allows error correction without human intervention.
""",
            encoding="utf-8",
        )

        # 1. Initial Sync
        run_stats = self.client.sync_directory(
            source_dir=self.test_dir,
            batch_size=10,
            visibility="public",
        )
        self.assertEqual(run_stats["files_seen"], 2)
        self.assertEqual(run_stats["files_changed"], 2)

        # 2. Incremental Sync (No changes -> files_changed == 0)
        run_stats_2 = self.client.sync_directory(
            source_dir=self.test_dir,
            batch_size=10,
        )
        self.assertEqual(run_stats_2["files_seen"], 2)
        self.assertEqual(run_stats_2["files_changed"], 0)

        # 3. Test Search Hybrid
        results = self.client.search_hybrid(
            query="Raft consensus and leader election",
            kb_token=raw_token,
            match_count=5,
        )
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertIn("Raft", top_match.chunk_text)
        self.assertIn("PostgreSQL Distributed Guide", top_match.document_title)

        # 4. Test Context Expansion (Section vs Document)
        sec_results = self.client.search_hybrid(
            query="Raft consensus",
            kb_token=raw_token,
            match_count=1,
            expand_context="section",
        )
        self.assertEqual(len(sec_results), 1)
        self.assertIsNotNone(sec_results[0].expanded_text)
        self.assertIn("Raft consensus ensures strong consistency", sec_results[0].expanded_text)

        doc_results = self.client.search_hybrid(
            query="Raft consensus",
            kb_token=raw_token,
            match_count=1,
            expand_context="document",
        )
        self.assertEqual(len(doc_results), 1)
        self.assertIsNotNone(doc_results[0].expanded_text)
        self.assertIn("Failover Mechanisms", doc_results[0].expanded_text)

        self.client.tokens.revoke_token(token_row["id"])

    def test_live_facets_navigation(self):
        """Verifies facets retrieval over live PostgREST API."""
        raw_token, token_row = self.client.tokens.create_token(
            name=f"test_facets_nav_tok_{uuid.uuid4().hex[:6]}",
            metadata={"purpose": "facets"},
        )
        facets = self.client.retrieval.get_facets(kb_token=raw_token)
        self.assertIsInstance(facets, list)
        self.client.tokens.revoke_token(token_row["id"])

    async def test_async_live_search_hybrid_and_facets(self):
        """Async hybrid search and facets execution over PostgREST."""
        raw_token, token_row = self.client.tokens.create_token(
            name=f"test_async_tok_{uuid.uuid4().hex[:6]}",
            metadata={"purpose": "async_search"},
        )
        async_client = AsyncEasyRagClient(
            supabase_url=self.url,
            supabase_key=self.service_key,
            embedding_provider=self.provider,
            config=self.config,
            use_rls=False,
        )
        try:
            results = await async_client.search_hybrid(
                query="PostgreSQL",
                kb_token=raw_token,
                match_count=3,
            )
            self.assertIsInstance(results, list)

            facets = await async_client.retrieval.get_facets(kb_token=raw_token)
            self.assertIsInstance(facets, list)
        finally:
            self.client.tokens.revoke_token(token_row["id"])


if __name__ == "__main__":
    unittest.main()


