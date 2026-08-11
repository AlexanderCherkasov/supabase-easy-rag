from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.ingestion.syncer import DocumentSyncer
from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.retrieval.engine import RetrievalEngine
from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client
from supabase_easy_rag.security.tokens import TokenManager


class EasyRagClient:
    """Core RAG client — no vendor branching, pure injection.

    Provider is injected explicitly (see examples/):
        from supabase_easy_rag.providers.azure import AzureEmbeddingProvider
        connector = AzureEmbeddingProvider(api_key="...", endpoint="...", model="gpt-5.4-nano")
        client = EasyRagClient(embedding_provider=connector)

    Supports RLS (auth.uid()) and token modes via Supabase config.
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        config: EasyRagConfig | None = None,
        user_jwt: str | None = None,
        use_rls: bool | None = None,
    ) -> None:
        self.config: EasyRagConfig = config or EasyRagConfig.from_env()
        self.url: str = supabase_url or self.config.supabase_url
        default_key: str = self.config.supabase_anon_key if (use_rls or self.config.use_rls) else self.config.supabase_service_role_key
        self.key: str = supabase_key or default_key or self.config.supabase_service_role_key
        self.user_jwt: str | None = user_jwt
        self.use_rls: bool = use_rls if use_rls is not None else self.config.use_rls
        if self.user_jwt:
            self.use_rls = True

        self.postgrest = create_postgrest_client(
            supabase_url=self.url,
            supabase_key=self.key,
            schema_name=self.config.schema_name,
            user_jwt=self.user_jwt,
        )

        self.provider: BaseEmbeddingProvider | None = embedding_provider

        self.retrieval: RetrievalEngine = RetrievalEngine(
            postgrest_client=self.postgrest,
            embedding_provider=self.provider,
            schema_name=self.config.schema_name,
        )
        self.syncer: DocumentSyncer | None = (
            DocumentSyncer(
                postgrest_client=self.postgrest,
                embedding_provider=self.provider,
                schema_name=self.config.schema_name,
            )
            if self.provider
            else None
        )
        self.tokens: TokenManager = TokenManager(
            postgrest_client=self.postgrest,
            schema_name=self.config.schema_name,
        )

    def for_user(self, user_jwt: str) -> EasyRagClient:
        anon_key: str = self.config.supabase_anon_key or self.key
        return EasyRagClient(
            supabase_url=self.url,
            supabase_key=anon_key,
            embedding_provider=self.provider,
            config=self.config,
            user_jwt=user_jwt,
            use_rls=True,
        )

    def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_hybrid(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token: str = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_hybrid(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

    def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_vector(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_vector(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

    def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_fts(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_fts(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

    def sync_directory(
        self,
        source_dir: Path | str,
        pattern: str | None = None,
        limit: int | None = None,
        owner_id: str | None = None,
        visibility: str = "private",
        enable_chunking: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        if not self.syncer:
            raise RuntimeError("DocumentSyncer requires an explicit embedding_provider (inject via connectors)")
        return self.syncer.sync_directory(
            source_root=Path(source_dir),
            pattern=pattern,
            limit=limit,
            owner_id=owner_id,
            visibility=visibility,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


class AsyncEasyRagClient:
    """Asynchronous RAG client — supports non-blocking operations in FastAPI/Starlette."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        config: EasyRagConfig | None = None,
        user_jwt: str | None = None,
        use_rls: bool | None = None,
    ) -> None:
        from supabase_easy_rag.retrieval.engine import AsyncRetrievalEngine
        from supabase_easy_rag.retrieval.postgrest_client import create_async_postgrest_client

        self.config: EasyRagConfig = config or EasyRagConfig.from_env()
        self.url: str = supabase_url or self.config.supabase_url
        default_key: str = self.config.supabase_anon_key if (use_rls or self.config.use_rls) else self.config.supabase_service_role_key
        self.key: str = supabase_key or default_key or self.config.supabase_service_role_key
        self.user_jwt: str | None = user_jwt
        self.use_rls: bool = use_rls if use_rls is not None else self.config.use_rls
        if self.user_jwt:
            self.use_rls = True

        self.postgrest = create_async_postgrest_client(
            supabase_url=self.url,
            supabase_key=self.key,
            schema_name=self.config.schema_name,
            user_jwt=self.user_jwt,
        )

        self.provider: BaseEmbeddingProvider | None = embedding_provider

        self.retrieval: AsyncRetrievalEngine = AsyncRetrievalEngine(
            postgrest_client=self.postgrest,
            embedding_provider=self.provider,
            schema_name=self.config.schema_name,
        )

    def for_user(self, user_jwt: str) -> AsyncEasyRagClient:
        anon_key: str = self.config.supabase_anon_key or self.key
        return AsyncEasyRagClient(
            supabase_url=self.url,
            supabase_key=anon_key,
            embedding_provider=self.provider,
            config=self.config,
            user_jwt=user_jwt,
            use_rls=True,
        )

    async def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_hybrid(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token: str = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_hybrid(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

    async def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_vector(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_vector(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

    async def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        use_rls: bool | None = None,
    ) -> list[SearchResult]:
        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_fts(query=query, kb_token=None, match_count=match_count, facet_keys=facet_keys, use_rls=True)
        token = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_fts(query=query, kb_token=token, match_count=match_count, facet_keys=facet_keys, use_rls=False)

