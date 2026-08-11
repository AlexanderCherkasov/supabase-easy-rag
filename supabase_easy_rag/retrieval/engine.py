from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from postgrest import SyncPostgrestClient

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
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
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
        kb_token: str,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        if not self.provider:
            raise EasyRagError("Embedding provider is required for vector search")
        query_vector = self.provider.embed_query(query)
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
        kb_token: str,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
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
        kb_token: str,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        query_vector: Optional[list[float]] = None
        if self.provider:
            try:
                query_vector = self.provider.embed_query(query)
            except Exception as exc:
                logger.warning(
                    "Embedding generation failed (%s). Falling back to Full-Text Search.", exc
                )

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
        kb_token: str,
        facet_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        data = self._rpc(
            "get_navigation_facets",
            {"p_kb_token": kb_token, "p_facet_type": facet_type},
        )
        return data if isinstance(data, list) else []
