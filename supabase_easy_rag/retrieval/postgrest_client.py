from __future__ import annotations

from typing import Optional
from httpx import Client as HttpClient
from postgrest import SyncPostgrestClient

from supabase_easy_rag.core.exceptions import EasyRagConfigurationError


def create_postgrest_client(
    supabase_url: str,
    supabase_key: str,
    access_token: Optional[str] = None,
    schema_name: str = "knowledgebase",
) -> SyncPostgrestClient:
    """Create a lightweight PostgREST client for Supabase database RPCs."""
    if not supabase_url or not supabase_key:
        raise EasyRagConfigurationError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) are required")

    url = supabase_url.rstrip("/")
    base_rest_url = f"{url}/rest/v1"

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token or supabase_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    return SyncPostgrestClient(
        base_url=base_rest_url,
        schema=schema_name,
        headers=headers,
        http_client=HttpClient(
            base_url=base_rest_url,
            headers=headers,
            follow_redirects=True,
            http2=True,
        ),
    )
