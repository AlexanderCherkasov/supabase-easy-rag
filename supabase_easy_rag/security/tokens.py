from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import Any

from postgrest._sync.client import (
    SyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_secure_token(prefix: str = "kb_live_") -> str:
    return f"{prefix}{secrets.token_hex(24)}"


class TokenManager:
    """Manager for creating and revoking RAG access tokens in Supabase."""

    def __init__(self, postgrest_client: SyncPostgrestClient, schema_name: str = "knowledgebase"):
        self.client = postgrest_client
        self.schema_name = schema_name

    def _table(self):
        return self.client.schema(self.schema_name).table("access_tokens")

    def create_token(
        self,
        name: str,
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        raw_token = generate_secure_token()
        token_hash = hash_token(raw_token)

        payload = {
            "token_name": name,
            "token_hash": token_hash,
            "is_active": True,
            "expires_at": expires_at,
            "metadata": metadata or {},
        }
        response = self._table().insert(payload).execute()
        row = (response.data or [{}])[0]
        return raw_token, row

    def revoke_token(self, token_name_or_id: str) -> bool:
        if not token_name_or_id or not str(token_name_or_id).strip():
            return False
        clean_target = str(token_name_or_id).strip()
        try:
            valid_uuid = str(uuid.UUID(clean_target))
            response = self._table().update({"is_active": False}).eq("id", valid_uuid).execute()
        except ValueError:
            response = self._table().update({"is_active": False}).eq("token_name", clean_target).execute()
        return bool(response.data)

    def list_tokens(self) -> list[dict[str, Any]]:
        response = self._table().select("id,token_name,is_active,expires_at,last_used_at,created_at").execute()
        return response.data or []
