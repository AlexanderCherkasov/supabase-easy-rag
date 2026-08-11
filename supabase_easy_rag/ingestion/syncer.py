from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from postgrest import SyncPostgrestClient

from supabase_easy_rag.core.exceptions import EasyRagIngestionError
from supabase_easy_rag.core.models import ParsedDocument
from supabase_easy_rag.ingestion.parser import parse_markdown_document
from supabase_easy_rag.providers.base import BaseEmbeddingProvider

SELECT_BATCH_SIZE = 100
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY_SECONDS = 1.0


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "'code': 502",
            "'code': 503",
            "'code': 504",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
            "connection error",
            "timed out",
        )
    )


def run_with_retry(operation_name: str, operation: Callable[[], Any]) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not _is_transient_error(exc) or attempt == RETRY_ATTEMPTS:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * attempt
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise EasyRagIngestionError(f"Unexpected retry state for {operation_name}")


class DocumentSyncer:
    """Incremental Markdown Document Syncer into Supabase."""

    def __init__(
        self,
        postgrest_client: SyncPostgrestClient,
        embedding_provider: BaseEmbeddingProvider,
        schema_name: str = "knowledgebase",
    ):
        self.client = postgrest_client
        self.provider = embedding_provider
        self.schema_name = schema_name

    def _table(self, name: str):
        return self.client.schema(self.schema_name).table(name)

    def sync_directory(
        self,
        source_root: Path,
        pattern: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        source_root = source_root.resolve()
        markdown_files = sorted(p for p in source_root.rglob("*.md") if p.is_file())
        if pattern:
            lowered = pattern.lower()
            markdown_files = [f for f in markdown_files if lowered in f.as_posix().lower()]
        if limit is not None and limit >= 0:
            markdown_files = markdown_files[:limit]

        if not markdown_files:
            return {"files_seen": 0, "files_changed": 0, "status": "completed"}

        # 1. Log ingestion run start
        run_response = run_with_retry(
            "start_ingestion_run",
            lambda: self._table("ingestion_runs")
            .insert(
                {
                    "status": "running",
                    "source_root": str(source_root),
                    "files_seen": len(markdown_files),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .execute(),
        )
        run_id = (run_response.data or [{}])[0].get("id")

        files_changed = 0
        try:
            # 2. Parse all documents
            parsed_docs: list[ParsedDocument] = [
                parse_markdown_document(f, source_root) for f in markdown_files
            ]
            doc_keys = [doc.document_key for doc in parsed_docs]

            # 3. Check existing document checksums
            existing_map: dict[str, dict[str, Any]] = {}
            for start in range(0, len(doc_keys), SELECT_BATCH_SIZE):
                chunk = doc_keys[start : start + SELECT_BATCH_SIZE]
                resp = run_with_retry(
                    "fetch_existing_docs",
                    lambda: self._table("documents")
                    .select("id,document_key,checksum")
                    .in_("document_key", chunk)
                    .execute(),
                )
                for row in resp.data or []:
                    if isinstance(row, dict):
                        existing_map[row["document_key"]] = row

            # 4. Filter only changed or new documents
            changed_docs = [
                doc
                for doc in parsed_docs
                if doc.document_key not in existing_map
                or existing_map[doc.document_key].get("checksum") != doc.checksum
            ]

            files_changed = len(changed_docs)

            # 5. Process changed documents in batches
            for i in range(0, len(changed_docs), batch_size):
                batch = changed_docs[i : i + batch_size]
                self._process_batch(batch, existing_map)

            # 6. Mark ingestion run completed
            if run_id:
                self._table("ingestion_runs").update(
                    {
                        "status": "completed",
                        "files_changed": files_changed,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).eq("id", run_id).execute()

            return {
                "files_seen": len(markdown_files),
                "files_changed": files_changed,
                "status": "completed",
            }
        except Exception as exc:
            if run_id:
                self._table("ingestion_runs").update(
                    {
                        "status": "failed",
                        "error_summary": str(exc),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).eq("id", run_id).execute()
            raise EasyRagIngestionError(f"Directory sync failed: {exc}") from exc

    def _process_batch(
        self,
        docs: Sequence[ParsedDocument],
        existing_map: dict[str, dict[str, Any]],
    ) -> None:
        if not docs:
            return

        # Generate embeddings for batch
        contents = [doc.content for doc in docs]
        embeddings = self.provider.embed_texts(contents)

        for doc, embedding in zip(docs, embeddings):
            # Upsert document
            doc_payload = {
                "document_key": doc.document_key,
                "title": doc.title,
                "top_level_category": doc.top_level_category,
                "metadata": doc.metadata,
                "checksum": doc.checksum,
            }
            if doc.document_key in existing_map:
                doc_id = existing_map[doc.document_key]["id"]
                run_with_retry(
                    "update_document",
                    lambda: self._table("documents").update(doc_payload).eq("id", doc_id).execute(),
                )
                # Clear old chunks/sections/facets
                run_with_retry(
                    "clear_doc_sections",
                    lambda: self._table("document_sections").delete().eq("document_id", doc_id).execute(),
                )
                run_with_retry(
                    "clear_doc_chunks",
                    lambda: self._table("chunks").delete().eq("document_id", doc_id).execute(),
                )
            else:
                resp = run_with_retry(
                    "insert_document",
                    lambda: self._table("documents").insert(doc_payload).execute(),
                )
                doc_id = (resp.data or [{}])[0]["id"]

            # Insert Chunks
            chunk_payload = {
                "document_id": doc_id,
                "chunk_index": 0,
                "content": doc.content,
                "metadata": doc.metadata,
                "token_count": doc.token_count,
                "char_count": doc.char_count,
                "embedding": list(embedding),
            }
            run_with_retry(
                "insert_chunk",
                lambda: self._table("chunks").insert(chunk_payload).execute(),
            )

            # Insert Facets
            for facet in doc.facets:
                facet_payload = {
                    "facet_type": facet.facet_type,
                    "facet_key": facet.facet_key,
                    "label": facet.label,
                    "sort_order": facet.sort_order,
                    "metadata": facet.metadata,
                }
                facet_resp = run_with_retry(
                    "upsert_facet",
                    lambda: self._table("facets")
                    .upsert(facet_payload, on_conflict="facet_key")
                    .execute(),
                )
                facet_id = (facet_resp.data or [{}])[0].get("id")
                if facet_id:
                    run_with_retry(
                        "link_doc_facet",
                        lambda: self._table("document_facets")
                        .upsert(
                            {"document_id": doc_id, "facet_id": facet_id},
                            on_conflict="document_id,facet_id",
                        )
                        .execute(),
                    )
