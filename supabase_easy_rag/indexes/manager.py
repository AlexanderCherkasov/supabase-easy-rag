from __future__ import annotations

import logging
from typing import Any
import uuid

from postgrest._async.client import (
    AsyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)
from postgrest._sync.client import (
    SyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)

from supabase_easy_rag.core.exceptions import EasyRagError

logger = logging.getLogger(__name__)


def _sanitize_tenant_id(tenant_id: str | uuid.UUID | None) -> str | None:
    if tenant_id is None:
        return None
    tid_str = str(tenant_id).strip()
    if not tid_str:
        return None
    try:
        return str(uuid.UUID(tid_str))
    except ValueError as exc:
        raise ValueError(f"tenant_id must be a valid UUID, got '{tenant_id}'") from exc


class IndexManager:
    """Management of dynamic vector indexes in PostgreSQL / pgvector."""

    def __init__(
        self,
        postgrest_client: SyncPostgrestClient,
        schema_name: str = "knowledgebase",
    ) -> None:
        self.client = postgrest_client
        self.schema_name = schema_name

    def list_vector_indexes(self, tenant_id: str | uuid.UUID | None = None) -> list[dict[str, Any]]:
        """List all partial HNSW vector indexes created on knowledgebase.chunks."""
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {}
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = self.client.schema(self.schema_name).rpc("list_vector_indexes", params).execute()
            data = getattr(resp, "data", None)
            return data if isinstance(data, list) else []
        except Exception as exc:
            raise EasyRagError(f"Failed to list vector indexes: {exc}") from exc

    def ensure_vector_index(
        self,
        dimension: int,
        tenant_id: str | uuid.UUID | None = None,
        m: int = 16,
        ef_construction: int = 64,
    ) -> str:
        """Create or ensure an HNSW index for the specified vector dimension and tenant."""
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {
            "p_dimension": dimension,
            "p_m": m,
            "p_ef_construction": ef_construction,
        }
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = (
                self.client.schema(self.schema_name)
                .rpc(
                    "ensure_vector_index",
                    params,
                )
                .execute()
            )
            return str(getattr(resp, "data", ""))
        except Exception as exc:
            raise EasyRagError(f"Failed to ensure vector index for dimension {dimension}: {exc}") from exc

    def drop_vector_index(
        self,
        dimension: int,
        tenant_id: str | uuid.UUID | None = None,
    ) -> str:
        """Drop an HNSW index for the specified vector dimension and tenant."""
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {"p_dimension": dimension}
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = (
                self.client.schema(self.schema_name)
                .rpc("drop_vector_index", params)
                .execute()
            )
            return str(getattr(resp, "data", ""))
        except Exception as exc:
            raise EasyRagError(f"Failed to drop vector index for dimension {dimension}: {exc}") from exc


class AsyncIndexManager:
    """Async management of dynamic vector indexes in PostgreSQL / pgvector."""

    def __init__(
        self,
        postgrest_client: AsyncPostgrestClient,
        schema_name: str = "knowledgebase",
    ) -> None:
        self.client = postgrest_client
        self.schema_name = schema_name

    async def list_vector_indexes(self, tenant_id: str | uuid.UUID | None = None) -> list[dict[str, Any]]:
        """List all partial HNSW vector indexes created on knowledgebase.chunks asynchronously."""
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {}
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = await self.client.schema(self.schema_name).rpc("list_vector_indexes", params).execute()
            data = getattr(resp, "data", None)
            return data if isinstance(data, list) else []
        except Exception as exc:
            raise EasyRagError(f"Failed to list vector indexes: {exc}") from exc

    async def ensure_vector_index(
        self,
        dimension: int,
        tenant_id: str | uuid.UUID | None = None,
        m: int = 16,
        ef_construction: int = 64,
    ) -> str:
        """Create or ensure an HNSW index for the specified vector dimension and tenant asynchronously."""
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {
            "p_dimension": dimension,
            "p_m": m,
            "p_ef_construction": ef_construction,
        }
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = await (
                self.client.schema(self.schema_name)
                .rpc(
                    "ensure_vector_index",
                    params,
                )
                .execute()
            )
            return str(getattr(resp, "data", ""))
        except Exception as exc:
            raise EasyRagError(f"Failed to ensure vector index for dimension {dimension}: {exc}") from exc

    async def drop_vector_index(
        self,
        dimension: int,
        tenant_id: str | uuid.UUID | None = None,
    ) -> str:
        """Drop an HNSW index for the specified vector dimension and tenant asynchronously."""
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        clean_tenant = _sanitize_tenant_id(tenant_id)
        params: dict[str, Any] = {"p_dimension": dimension}
        if clean_tenant is not None:
            params["p_tenant_id"] = clean_tenant
        try:
            resp = await (
                self.client.schema(self.schema_name)
                .rpc("drop_vector_index", params)
                .execute()
            )
            return str(getattr(resp, "data", ""))
        except Exception as exc:
            raise EasyRagError(f"Failed to drop vector index for dimension {dimension}: {exc}") from exc
