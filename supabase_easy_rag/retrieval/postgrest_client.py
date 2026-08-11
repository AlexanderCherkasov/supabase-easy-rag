from __future__ import annotations

from postgrest._sync.client import (
    SyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)

from supabase_easy_rag.core.exceptions import EasyRagConfigurationError


def create_postgrest_client(
    supabase_url: str,
    supabase_key: str,
    access_token: str | None = None,
    schema_name: str = "knowledgebase",
    user_jwt: str | None = None,
) -> SyncPostgrestClient:
    """Create a lightweight PostgREST client for Supabase database RPCs.

    Two modes:
      - service_role / token mode: supabase_key = service_role, Authorization = token or service_role
      - RLS mode (Supabase guide): supabase_key = anon key, Authorization = user JWT (auth.uid())
        In RLS mode pass user_jwt (from supabase.auth.getSession()) and anon key as supabase_key.
    """
    if not supabase_url or not supabase_key:
        raise EasyRagConfigurationError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) are required")

    url = supabase_url.rstrip("/")
    base_rest_url = f"{url}/rest/v1"

    # In RLS mode, apikey is anon key, Authorization is user JWT
    # In token mode, both are service_role (or token)
    auth_bearer = user_jwt or access_token or supabase_key
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {auth_bearer}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    return SyncPostgrestClient(
        base_url=base_rest_url,
        schema=schema_name,
        headers=headers,
    )


def create_rls_client(
    supabase_url: str,
    supabase_anon_key: str,
    user_jwt: str,
    schema_name: str = "knowledgebase",
) -> SyncPostgrestClient:
    """Convenience helper for RLS mode per Supabase 'RAG with Permissions' guide.

    Usage:
        client = create_rls_client(url, anon_key, user_jwt)
        # then: client.schema('knowledgebase').rpc('search_chunks_hybrid_rls', {...})
        # RLS policies will filter chunks via documents.owner_id = auth.uid()
    """
    return create_postgrest_client(
        supabase_url=supabase_url,
        supabase_key=supabase_anon_key,
        schema_name=schema_name,
        user_jwt=user_jwt,
    )


def create_async_postgrest_client(
    supabase_url: str,
    supabase_key: str,
    access_token: str | None = None,
    schema_name: str = "knowledgebase",
    user_jwt: str | None = None,
):
    """Create an asynchronous PostgREST client for Supabase database RPCs."""
    from postgrest._async.client import (
        AsyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
    )

    if not supabase_url or not supabase_key:
        raise EasyRagConfigurationError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) are required")

    url = supabase_url.rstrip("/")
    base_rest_url = f"{url}/rest/v1"

    auth_bearer = user_jwt or access_token or supabase_key
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {auth_bearer}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    return AsyncPostgrestClient(
        base_url=base_rest_url,
        schema=schema_name,
        headers=headers,
    )

