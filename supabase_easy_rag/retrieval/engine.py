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
                    vector_rank=row.get("vector_rank"),
                    text_rank=row.get("text_rank"),
                    section_id=str(row.get("section_id")) if row.get("section_id") else None,
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

    def _expand_results_sync(self, results: list[SearchResult], mode: str) -> list[SearchResult]:
        if not results or mode not in ("section", "document"):
            return results

        if mode == "document":
            doc_ids = list({r.document_id for r in results if r.document_id})
            if not doc_ids:
                return results
            resp = self.client.schema(self.schema_name).table("chunks").select("document_id,chunk_index,content").in_("document_id", doc_ids).order("chunk_index").execute()
            doc_texts: dict[str, list[str]] = {}
            for row in resp.data or []:
                if isinstance(row, dict):
                    doc_texts.setdefault(row["document_id"], []).append(row.get("content", ""))
            doc_full: dict[str, str] = {d_id: "\n\n".join(parts) for d_id, parts in doc_texts.items()}
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    section_title=r.section_title,
                    chunk_text=r.chunk_text,
                    facet_path=r.facet_path,
                    metadata=r.metadata,
                    vector_score=r.vector_score,
                    text_score=r.text_score,
                    hybrid_score=r.hybrid_score,
                    vector_rank=r.vector_rank,
                    text_rank=r.text_rank,
                    section_id=r.section_id,
                    expanded_text=doc_full.get(r.document_id, r.chunk_text),
                )
                for r in results
            ]

        if mode == "section":
            # For section expansion: find chunks sharing the same document_id and section_title / section_id
            doc_ids = list({r.document_id for r in results if r.document_id})
            if not doc_ids:
                return results
            resp = self.client.schema(self.schema_name).table("chunks").select("document_id,section_id,chunk_index,content").in_("document_id", doc_ids).order("chunk_index").execute()
            sec_texts: dict[tuple[str, str | None], list[str]] = {}
            for row in resp.data or []:
                if isinstance(row, dict):
                    key = (row["document_id"], row.get("section_id"))
                    sec_texts.setdefault(key, []).append(row.get("content", ""))
            sec_full: dict[tuple[str, str | None], str] = {k: "\n\n".join(parts) for k, parts in sec_texts.items()}
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    section_title=r.section_title,
                    chunk_text=r.chunk_text,
                    facet_path=r.facet_path,
                    metadata=r.metadata,
                    vector_score=r.vector_score,
                    text_score=r.text_score,
                    hybrid_score=r.hybrid_score,
                    vector_rank=r.vector_rank,
                    text_rank=r.text_rank,
                    section_id=r.section_id,
                    expanded_text=sec_full.get((r.document_id, r.section_id), r.chunk_text),
                )
                for r in results
            ]
        return results

    def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        if not self.provider:
            raise EasyRagError("Embedding provider is required for vector search")
        query_vector = self.provider.embed_query(query)
        params: dict[str, Any] = {
            "p_query_embedding": query_vector,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_min_vector_similarity": min_vector_similarity,
        }
        # RLS mode: call _rls variant without token (auth.uid() enforced)
        if use_rls or kb_token is None:
            data = self._rpc("match_chunks_by_embedding_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = self._rpc("match_chunks_by_embedding", params)
        res = _parse_search_results(data)
        if expand_context:
            res = self._expand_results_sync(res, expand_context)
        return res

    def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        fts_config: str = "english",
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        params: dict[str, Any] = {
            "p_query": query,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_fts_config": fts_config,
        }
        if use_rls or kb_token is None:
            data = self._rpc("search_chunks_full_text_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = self._rpc("search_chunks_full_text", params)
        res = _parse_search_results(data)
        if expand_context:
            res = self._expand_results_sync(res, expand_context)
        return res

    def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        candidate_count: int | None = None,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        text_weight: float = 1.0,
        fts_config: str = "english",
        min_vector_similarity: float | None = None,
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        query_vector: list[float] | None = None
        if self.provider:
            try:
                query_vector = self.provider.embed_query(query)
            except Exception as exc:
                logger.warning(
                    "Embedding generation failed (%s). Falling back to Full-Text Search.", exc
                )

        params: dict[str, Any] = {
            "p_query": query,
            "p_query_embedding": query_vector,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_candidate_count": candidate_count,
            "p_rrf_k": rrf_k,
            "p_vector_weight": vector_weight,
            "p_text_weight": text_weight,
            "p_fts_config": fts_config,
            "p_min_vector_similarity": min_vector_similarity,
        }

        if use_rls or kb_token is None:
            data = self._rpc("search_chunks_hybrid_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = self._rpc("search_chunks_hybrid", params)
        res = _parse_search_results(data)
        if expand_context:
            res = self._expand_results_sync(res, expand_context)
        return res

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

    async def _expand_results_async(self, results: list[SearchResult], mode: str) -> list[SearchResult]:
        if not results or mode not in ("section", "document"):
            return results

        if mode == "document":
            doc_ids = list({r.document_id for r in results if r.document_id})
            if not doc_ids:
                return results
            resp = await self.client.schema(self.schema_name).table("chunks").select("document_id,chunk_index,content").in_("document_id", doc_ids).order("chunk_index").execute()
            doc_texts: dict[str, list[str]] = {}
            for row in resp.data or []:
                if isinstance(row, dict):
                    doc_texts.setdefault(row["document_id"], []).append(row.get("content", ""))
            doc_full: dict[str, str] = {d_id: "\n\n".join(parts) for d_id, parts in doc_texts.items()}
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    section_title=r.section_title,
                    chunk_text=r.chunk_text,
                    facet_path=r.facet_path,
                    metadata=r.metadata,
                    vector_score=r.vector_score,
                    text_score=r.text_score,
                    hybrid_score=r.hybrid_score,
                    vector_rank=r.vector_rank,
                    text_rank=r.text_rank,
                    section_id=r.section_id,
                    expanded_text=doc_full.get(r.document_id, r.chunk_text),
                )
                for r in results
            ]

        if mode == "section":
            doc_ids = list({r.document_id for r in results if r.document_id})
            if not doc_ids:
                return results
            resp = await self.client.schema(self.schema_name).table("chunks").select("document_id,section_id,chunk_index,content").in_("document_id", doc_ids).order("chunk_index").execute()
            sec_texts: dict[tuple[str, str | None], list[str]] = {}
            for row in resp.data or []:
                if isinstance(row, dict):
                    key = (row["document_id"], row.get("section_id"))
                    sec_texts.setdefault(key, []).append(row.get("content", ""))
            sec_full: dict[tuple[str, str | None], str] = {k: "\n\n".join(parts) for k, parts in sec_texts.items()}
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_title=r.document_title,
                    section_title=r.section_title,
                    chunk_text=r.chunk_text,
                    facet_path=r.facet_path,
                    metadata=r.metadata,
                    vector_score=r.vector_score,
                    text_score=r.text_score,
                    hybrid_score=r.hybrid_score,
                    vector_rank=r.vector_rank,
                    text_rank=r.text_rank,
                    section_id=r.section_id,
                    expanded_text=sec_full.get((r.document_id, r.section_id), r.chunk_text),
                )
                for r in results
            ]
        return results

    async def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        if not self.provider:
            raise EasyRagError("Embedding provider is required for vector search")
        query_vector = self.provider.embed_query(query)
        params: dict[str, Any] = {
            "p_query_embedding": query_vector,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_min_vector_similarity": min_vector_similarity,
        }
        if use_rls or kb_token is None:
            data = await self._rpc("match_chunks_by_embedding_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = await self._rpc("match_chunks_by_embedding", params)
        res = _parse_search_results(data)
        if expand_context:
            res = await self._expand_results_async(res, expand_context)
        return res

    async def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        fts_config: str = "english",
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        params: dict[str, Any] = {
            "p_query": query,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_fts_config": fts_config,
        }
        if use_rls or kb_token is None:
            data = await self._rpc("search_chunks_full_text_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = await self._rpc("search_chunks_full_text", params)
        res = _parse_search_results(data)
        if expand_context:
            res = await self._expand_results_async(res, expand_context)
        return res

    async def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        candidate_count: int | None = None,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        text_weight: float = 1.0,
        fts_config: str = "english",
        min_vector_similarity: float | None = None,
        use_rls: bool = False,
        expand_context: str | None = None,
    ) -> list[SearchResult]:
        query_vector: list[float] | None = None
        if self.provider:
            try:
                query_vector = self.provider.embed_query(query)
            except Exception as exc:
                logger.warning("Embedding generation failed (%s). Falling back to Full-Text Search.", exc)

        params: dict[str, Any] = {
            "p_query": query,
            "p_query_embedding": query_vector,
            "p_match_count": match_count,
            "p_facet_keys": list(facet_keys) if facet_keys else None,
            "p_candidate_count": candidate_count,
            "p_rrf_k": rrf_k,
            "p_vector_weight": vector_weight,
            "p_text_weight": text_weight,
            "p_fts_config": fts_config,
            "p_min_vector_similarity": min_vector_similarity,
        }

        if use_rls or kb_token is None:
            data = await self._rpc("search_chunks_hybrid_rls", params)
        else:
            params["p_kb_token"] = kb_token
            data = await self._rpc("search_chunks_hybrid", params)
        res = _parse_search_results(data)
        if expand_context:
            res = await self._expand_results_async(res, expand_context)
        return res


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


