from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from postgrest._sync.client import (
    SyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)

from supabase_easy_rag.core.exceptions import EasyRagAccessError, EasyRagError
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.providers.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def _parse_search_results(data: Any) -> list[SearchResult]:
    if not isinstance(data, list):
        return []
    results: list[SearchResult] = []
    for row in data:
        if isinstance(row, dict):
            results.append(
                SearchResult(
                    chunk_id=str(row.get("chunk_id", "")),
                    document_id=str(row.get("document_id", "")),
                    document_title=str(row.get("document_title", "")),
                    section_title=row.get("section_title"),
                    chunk_text=str(row.get("chunk_text", "")),
                    facet_path=row.get("facet_path"),
                    metadata=row.get("metadata") or {},
                    vector_score=row.get("vector_score"),
                    text_score=row.get("text_score"),
                    hybrid_score=row.get("hybrid_score"),
                )
            )
    return results


class RetrievalEngine:
    """RAG Search Retrieval Engine over Supabase RPCs."""

    def __init__(
        self,
        postgrest_client: SyncPostgrestClient,
        embedding_provider: BaseEmbeddingProvider | None = None,
        schema_name: str = "knowledgebase",
    ):
        self.client = postgrest_client
        self.provider = embedding_provider
        self.schema_name = schema_name

    def _rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        try:
            response = self.client.schema(self.schema_name).rpc(function_name, params).execute()
            return getattr(response, "data", None)
        except Exception as exc:
            msg = str(exc).lower()
            if "invalid knowledgebase token" in msg or "token is required" in msg:
                raise EasyRagAccessError(f"Access denied: {exc}") from exc
            raise EasyRagError(f"RPC {function_name} failed: {exc}") from exc

    def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        if not self.provider:
            raise EasyRagError("Embedding provider is required for vector search")
        query_vector = self.provider.embed_query(query)
        # RLS mode: call _rls variant without token (auth.uid() enforced)
        if use_rls or kb_token is None:
            data = self._rpc(
                "match_chunks_by_embedding_rls",
                {
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = self._rpc(
                "match_chunks_by_embedding",
                {
                    "p_kb_token": kb_token,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        if use_rls or kb_token is None:
            data = self._rpc(
                "search_chunks_full_text_rls",
                {
                    "p_query": query,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = self._rpc(
                "search_chunks_full_text",
                {
                    "p_kb_token": kb_token,
                    "p_query": query,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        query_vector: list[float] | None = None
        if self.provider:
            try:
                query_vector = self.provider.embed_query(query)
            except Exception as exc:
                logger.warning(
                    "Embedding generation failed (%s). Falling back to Full-Text Search.", exc
                )

        if use_rls or kb_token is None:
            data = self._rpc(
                "search_chunks_hybrid_rls",
                {
                    "p_query": query,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = self._rpc(
                "search_chunks_hybrid",
                {
                    "p_kb_token": kb_token,
                    "p_query": query,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    def get_facets(
        self,
        kb_token: str | None = None,
        facet_type: str | None = None,
        use_rls: bool = False,
    ) -> list[dict[str, Any]]:
        if use_rls or kb_token is None:
            data = self._rpc(
                "get_navigation_facets_rls",
                {"p_facet_type": facet_type},
            )
        else:
            data = self._rpc(
                "get_navigation_facets",
                {"p_kb_token": kb_token, "p_facet_type": facet_type},
            )
        return data if isinstance(data, list) else []


class AsyncRetrievalEngine:
    """Asynchronous RAG Search Retrieval Engine over Supabase RPCs."""

    def __init__(
        self,
        postgrest_client: Any,
        embedding_provider: BaseEmbeddingProvider | None = None,
        schema_name: str = "knowledgebase",
    ):
        self.client = postgrest_client
        self.provider = embedding_provider
        self.schema_name = schema_name

    async def _rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        try:
            response = await self.client.schema(self.schema_name).rpc(function_name, params).execute()
            return getattr(response, "data", None)
        except Exception as exc:
            msg = str(exc).lower()
            if "invalid knowledgebase token" in msg or "token is required" in msg:
                raise EasyRagAccessError(f"Access denied: {exc}") from exc
            raise EasyRagError(f"RPC {function_name} failed: {exc}") from exc

    async def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        if not self.provider:
            raise EasyRagError("Embedding provider is required for vector search")
        query_vector = self.provider.embed_query(query)
        if use_rls or kb_token is None:
            data = await self._rpc(
                "match_chunks_by_embedding_rls",
                {
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = await self._rpc(
                "match_chunks_by_embedding",
                {
                    "p_kb_token": kb_token,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    async def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        if use_rls or kb_token is None:
            data = await self._rpc(
                "search_chunks_full_text_rls",
                {
                    "p_query": query,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = await self._rpc(
                "search_chunks_full_text",
                {
                    "p_kb_token": kb_token,
                    "p_query": query,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    async def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool = False,
    ) -> list[SearchResult]:
        query_vector: list[float] | None = None
        if self.provider:
            try:
                query_vector = self.provider.embed_query(query)
            except Exception as exc:
                logger.warning("Embedding generation failed (%s). Falling back to Full-Text Search.", exc)

        if use_rls or kb_token is None:
            data = await self._rpc(
                "search_chunks_hybrid_rls",
                {
                    "p_query": query,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        else:
            data = await self._rpc(
                "search_chunks_hybrid",
                {
                    "p_kb_token": kb_token,
                    "p_query": query,
                    "p_query_embedding": query_vector,
                    "p_match_count": match_count,
                    "p_facet_keys": list(facet_keys) if facet_keys else None,
                },
            )
        return _parse_search_results(data)

    async def get_facets(
        self,
        kb_token: str | None = None,
        facet_type: str | None = None,
        use_rls: bool = False,
    ) -> list[dict[str, Any]]:
        if use_rls or kb_token is None:
            data = await self._rpc(
                "get_navigation_facets_rls",
                {"p_facet_type": facet_type},
            )
        else:
            data = await self._rpc(
                "get_navigation_facets",
                {"p_kb_token": kb_token, "p_facet_type": facet_type},
            )
        return data if isinstance(data, list) else []

