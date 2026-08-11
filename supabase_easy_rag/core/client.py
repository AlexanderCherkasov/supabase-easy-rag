from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.ingestion.syncer import DocumentSyncer
from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.providers.openai_provider import OpenAIEmbeddingProvider
from supabase_easy_rag.retrieval.engine import RetrievalEngine
from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client
from supabase_easy_rag.security.tokens import TokenManager


class EasyRagClient:
    """High-level client for Supabase Easy RAG operations."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        config: Optional[EasyRagConfig] = None,
    ):
        self.config = config or EasyRagConfig.from_env()
        self.url = supabase_url or self.config.supabase_url
        self.key = supabase_key or self.config.supabase_service_role_key

        self.postgrest = create_postgrest_client(
            supabase_url=self.url,
            supabase_key=self.key,
            schema_name=self.config.schema_name,
        )

        if embedding_provider:
            self.provider = embedding_provider
        elif self.config.openai_api_key:
            self.provider = OpenAIEmbeddingProvider(
                api_key=self.config.openai_api_key,
                model=self.config.embedding_model,
                endpoint=self.config.openai_endpoint,
                api_version=self.config.openai_api_version,
            )
        else:
            self.provider = None

        self.retrieval = RetrievalEngine(
            postgrest_client=self.postgrest,
            embedding_provider=self.provider,
            schema_name=self.config.schema_name,
        )
        self.syncer = (
            DocumentSyncer(
                postgrest_client=self.postgrest,
                embedding_provider=self.provider,
                schema_name=self.config.schema_name,
            )
            if self.provider
            else None
        )
        self.tokens = TokenManager(
            postgrest_client=self.postgrest,
            schema_name=self.config.schema_name,
        )

    def search_hybrid(
        self,
        query: str,
        kb_token: Optional[str] = None,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_hybrid(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
        )

    def search_vector(
        self,
        query: str,
        kb_token: Optional[str] = None,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_vector(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
        )

    def search_fts(
        self,
        query: str,
        kb_token: Optional[str] = None,
        match_count: int = 5,
        facet_keys: Optional[Sequence[str]] = None,
    ) -> list[SearchResult]:
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_fts(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
        )

    def sync_directory(
        self,
        source_dir: Path | str,
        pattern: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.syncer:
            raise RuntimeError("DocumentSyncer requires an active embedding provider")
        return self.syncer.sync_directory(
            source_root=Path(source_dir),
            pattern=pattern,
            limit=limit,
        )
