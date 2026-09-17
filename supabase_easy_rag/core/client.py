from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.core.exceptions import EasyRagConfigurationError
from supabase_easy_rag.core.models import SearchResult
from supabase_easy_rag.ingestion.syncer import DocumentSyncer
from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.retrieval.engine import RetrievalEngine
from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client
from supabase_easy_rag.security.tokens import TokenManager


_UNSET = object()


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
        tenant_id: Any = _UNSET,
        scope_id: Any = _UNSET,
        include_global: Any = _UNSET,
    ) -> None:
        self.config: EasyRagConfig = config or EasyRagConfig.from_env()
        self.url: str = supabase_url or self.config.supabase_url
        self.user_jwt: str | None = user_jwt
        self.use_rls: bool = use_rls if use_rls is not None else self.config.use_rls
        if self.user_jwt:
            self.use_rls = True

        self.tenant_id: str | None = (
            tenant_id if tenant_id is not _UNSET else self.config.tenant_id
        )
        self.scope_id: str | None = (
            scope_id if scope_id is not _UNSET else self.config.scope_id
        )
        self.include_global: bool = (
            include_global if include_global is not _UNSET else self.config.include_global
        )

        if self.use_rls:
            resolved_key = supabase_key or self.config.supabase_anon_key
            if not resolved_key:
                raise EasyRagConfigurationError(
                    "SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY) is required when RLS mode is enabled. "
                    "Using the service_role key in RLS mode bypasses security policies."
                )
            self.key = resolved_key
        else:
            self.key = supabase_key or self.config.supabase_service_role_key or self.config.supabase_anon_key

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
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                enable_chunking=self.config.enable_chunking,
            )
            if self.provider
            else None
        )
        self.tokens: TokenManager = TokenManager(
            postgrest_client=self.postgrest,
            schema_name=self.config.schema_name,
        )

    def for_user(
        self,
        user_jwt: str,
        tenant_id: Any = _UNSET,
        scope_id: Any = _UNSET,
        include_global: Any = _UNSET,
    ) -> EasyRagClient:
        anon_key: str = self.config.supabase_anon_key or self.key
        resolved_tenant = tenant_id if tenant_id is not _UNSET else self.tenant_id
        resolved_scope = scope_id if scope_id is not _UNSET else self.scope_id
        resolved_global = include_global if include_global is not _UNSET else self.include_global
        return EasyRagClient(
            supabase_url=self.url,
            supabase_key=anon_key,
            embedding_provider=self.provider,
            config=self.config,
            user_jwt=user_jwt,
            use_rls=True,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
        )

    def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        candidate_count: int | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        fts_config: str | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_cand = candidate_count if candidate_count is not None else self.config.candidate_count
        resolved_rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k
        resolved_v_weight = vector_weight if vector_weight is not None else self.config.vector_weight
        resolved_t_weight = text_weight if text_weight is not None else self.config.text_weight
        resolved_fts_cfg = fts_config if fts_config is not None else self.config.fts_config
        resolved_min_sim = min_vector_similarity if min_vector_similarity is not None else self.config.min_vector_similarity
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_hybrid(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                candidate_count=resolved_cand,
                rrf_k=resolved_rrf_k,
                vector_weight=resolved_v_weight,
                text_weight=resolved_t_weight,
                fts_config=resolved_fts_cfg,
                min_vector_similarity=resolved_min_sim,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token: str = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_hybrid(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            candidate_count=resolved_cand,
            rrf_k=resolved_rrf_k,
            vector_weight=resolved_v_weight,
            text_weight=resolved_t_weight,
            fts_config=resolved_fts_cfg,
            min_vector_similarity=resolved_min_sim,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )

    def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_min_sim = min_vector_similarity if min_vector_similarity is not None else self.config.min_vector_similarity
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_vector(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                min_vector_similarity=resolved_min_sim,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_vector(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            min_vector_similarity=resolved_min_sim,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )

    def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        fts_config: str | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_fts_cfg = fts_config if fts_config is not None else self.config.fts_config
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return self.retrieval.search_fts(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                fts_config=resolved_fts_cfg,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token = kb_token or self.config.knowledgebase_access_token
        return self.retrieval.search_fts(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            fts_config=resolved_fts_cfg,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )

    def sync_directory(
        self,
        source_dir: Path | str,
        pattern: str | None = None,
        limit: int | None = None,
        batch_size: int = 20,
        owner_id: str | None = None,
        visibility: str = "private",
        enable_chunking: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_workers: int = 4,
        force: bool = False,
        tenant_id: str | None = None,
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.syncer:
            raise RuntimeError("DocumentSyncer requires an explicit embedding_provider (inject via connectors)")
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        return self.syncer.sync_directory(
            source_root=Path(source_dir),
            pattern=pattern,
            limit=limit,
            batch_size=batch_size,
            owner_id=owner_id,
            visibility=visibility,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_workers=max_workers,
            force=force,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
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
        tenant_id: Any = _UNSET,
        scope_id: Any = _UNSET,
        include_global: Any = _UNSET,
    ) -> None:
        from supabase_easy_rag.retrieval.engine import AsyncRetrievalEngine
        from supabase_easy_rag.retrieval.postgrest_client import create_async_postgrest_client

        self.config: EasyRagConfig = config or EasyRagConfig.from_env()
        self.url: str = supabase_url or self.config.supabase_url
        self.user_jwt: str | None = user_jwt
        self.use_rls: bool = use_rls if use_rls is not None else self.config.use_rls
        if self.user_jwt:
            self.use_rls = True

        self.tenant_id: str | None = (
            tenant_id if tenant_id is not _UNSET else self.config.tenant_id
        )
        self.scope_id: str | None = (
            scope_id if scope_id is not _UNSET else self.config.scope_id
        )
        self.include_global: bool = (
            include_global if include_global is not _UNSET else self.config.include_global
        )

        if self.use_rls:
            resolved_key = supabase_key or self.config.supabase_anon_key
            if not resolved_key:
                raise EasyRagConfigurationError(
                    "SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY) is required when RLS mode is enabled. "
                    "Using the service_role key in RLS mode bypasses security policies."
                )
            self.key = resolved_key
        else:
            self.key = supabase_key or self.config.supabase_service_role_key or self.config.supabase_anon_key

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

    def for_user(
        self,
        user_jwt: str,
        tenant_id: Any = _UNSET,
        scope_id: Any = _UNSET,
        include_global: Any = _UNSET,
    ) -> AsyncEasyRagClient:
        anon_key: str = self.config.supabase_anon_key or self.key
        resolved_tenant = tenant_id if tenant_id is not _UNSET else self.tenant_id
        resolved_scope = scope_id if scope_id is not _UNSET else self.scope_id
        resolved_global = include_global if include_global is not _UNSET else self.include_global
        return AsyncEasyRagClient(
            supabase_url=self.url,
            supabase_key=anon_key,
            embedding_provider=self.provider,
            config=self.config,
            user_jwt=user_jwt,
            use_rls=True,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
        )

    async def search_hybrid(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        candidate_count: int | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        fts_config: str | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_cand = candidate_count if candidate_count is not None else self.config.candidate_count
        resolved_rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k
        resolved_v_weight = vector_weight if vector_weight is not None else self.config.vector_weight
        resolved_t_weight = text_weight if text_weight is not None else self.config.text_weight
        resolved_fts_cfg = fts_config if fts_config is not None else self.config.fts_config
        resolved_min_sim = min_vector_similarity if min_vector_similarity is not None else self.config.min_vector_similarity
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_hybrid(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                candidate_count=resolved_cand,
                rrf_k=resolved_rrf_k,
                vector_weight=resolved_v_weight,
                text_weight=resolved_t_weight,
                fts_config=resolved_fts_cfg,
                min_vector_similarity=resolved_min_sim,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token: str = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_hybrid(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            candidate_count=resolved_cand,
            rrf_k=resolved_rrf_k,
            vector_weight=resolved_v_weight,
            text_weight=resolved_t_weight,
            fts_config=resolved_fts_cfg,
            min_vector_similarity=resolved_min_sim,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )

    async def search_vector(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        min_vector_similarity: float | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_min_sim = min_vector_similarity if min_vector_similarity is not None else self.config.min_vector_similarity
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_vector(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                min_vector_similarity=resolved_min_sim,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_vector(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            min_vector_similarity=resolved_min_sim,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )

    async def search_fts(
        self,
        query: str,
        kb_token: str | None = None,
        match_count: int = 5,
        facet_keys: Sequence[str] | None = None,
        fts_config: str | None = None,
        use_rls: bool | None = None,
        expand_context: str | None = None,
        tenant_id: str | None = None,
        scope_id: str | None = None,
        include_global: bool | None = None,
        allowed_categories: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        resolved_fts_cfg = fts_config if fts_config is not None else self.config.fts_config
        resolved_tenant = tenant_id if tenant_id is not None else self.tenant_id
        resolved_scope = scope_id if scope_id is not None else self.scope_id
        resolved_global = include_global if include_global is not None else self.include_global

        if self.use_rls or use_rls or (kb_token is None and not self.config.knowledgebase_access_token):
            return await self.retrieval.search_fts(
                query=query,
                kb_token=None,
                match_count=match_count,
                facet_keys=facet_keys,
                fts_config=resolved_fts_cfg,
                use_rls=True,
                expand_context=expand_context,
                tenant_id=resolved_tenant,
                scope_id=resolved_scope,
                include_global=resolved_global,
                allowed_categories=allowed_categories,
            )
        token = kb_token or self.config.knowledgebase_access_token
        return await self.retrieval.search_fts(
            query=query,
            kb_token=token,
            match_count=match_count,
            facet_keys=facet_keys,
            fts_config=resolved_fts_cfg,
            use_rls=False,
            expand_context=expand_context,
            tenant_id=resolved_tenant,
            scope_id=resolved_scope,
            include_global=resolved_global,
            allowed_categories=allowed_categories,
        )



