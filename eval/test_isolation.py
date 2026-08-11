from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from supabase_easy_rag.config import EasyRagConfig

# Isolated eval: test schema/prefix + auto-cleanup — artifacts self-destruct
# No haystack, no hidden stack: explicit document_key prefix + delete after.

@contextmanager
def isolated_eval_prefix(prefix: Optional[str] = None) -> Generator[str, None, None]:
    """Yield a unique document_key prefix for this eval run. Caller uses it for ingested docs."""
    run_id = uuid.uuid4().hex[:8]
    pfx = prefix or f"__eval_{run_id}/"
    try:
        yield pfx
    finally:
        # Auto-cleanup: delete all documents with this prefix via postgrest
        try:
            cfg = EasyRagConfig.from_env()
            if not cfg.supabase_url or not cfg.supabase_service_role_key:
                pass
            else:
                from supabase_easy_rag.retrieval.postgrest_client import create_postgrest_client
                client = create_postgrest_client(cfg.supabase_url, cfg.supabase_service_role_key, schema_name=cfg.schema_name)
                client.schema(cfg.schema_name).table("documents").delete().ilike("document_key", f"{pfx}%").execute()
                print(f"[isolation] Cleaned up prefix {pfx}")
        except Exception as e:
            print(f"[isolation] Cleanup failed for {pfx}: {e}")


def ingest_with_prefix(source_dir: Path, prefix: str, **sync_kwargs):
    """Copy source docs under a prefixed document_key by using a temp dir with prefix subfolder."""
    import shutil
    import tempfile
    # Create temp dir that mirrors source but under prefix
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / prefix.rstrip("/")
        tmp_path.mkdir(parents=True, exist_ok=True)
        # Copy all md files preserving structure under prefix
        for f in Path(source_dir).rglob("*.md"):
            rel = f.relative_to(source_dir)
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        # Now sync the temp dir's parent so document_key includes prefix
        from supabase_easy_rag import EasyRagClient
        client = EasyRagClient()
        # source_root is tmp, so keys become __eval_xxx/... 
        result = client.sync_directory(Path(tmp), **sync_kwargs)
        return result


@contextmanager
def isolated_test_schema(schema: Optional[str] = None):
    """Legacy: isolated schema via CREATE SCHEMA + migrations, then DROP SCHEMA.
    Requires direct DB access; if not available, falls back to prefix isolation.
    """
    # For Supabase Cloud without direct DDL, we fallback to prefix.
    # This context still yields a schema name for API compatibility.
    cfg = EasyRagConfig.from_env()
    test_schema = schema or f"knowledgebase_test_{uuid.uuid4().hex[:6]}"
    print(f"[isolation] Using prefix isolation (schema DDL not available via REST, using {test_schema} as logical namespace)")
    # Yield a prefix instead of real schema
    with isolated_eval_prefix(prefix=f"{test_schema}/") as pfx:
        yield pfx
    # After yield, prefix already cleaned. If we had real schema, we'd do DROP SCHEMA
    # try: client.rpc("exec_sql", {"sql": f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"}).execute()
