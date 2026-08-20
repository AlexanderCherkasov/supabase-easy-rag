from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from postgrest._sync.client import (
    SyncPostgrestClient,  # type: ignore[reportPrivateImportUsage]
)

from supabase_easy_rag.config import EasyRagConfig
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
    last_error: Exception | None = None
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
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        enable_chunking: bool | None = None,
    ):
        self.client = postgrest_client
        self.provider = embedding_provider
        self.schema_name = schema_name
        cfg = EasyRagConfig.from_env()
        self.enable_chunking = enable_chunking if enable_chunking is not None else cfg.enable_chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap

    def _table(self, name: str):
        return self.client.schema(self.schema_name).table(name)

    def sync_directory(
        self,
        source_root: Path,
        pattern: str | None = None,
        limit: int | None = None,
        batch_size: int = 20,
        owner_id: str | None = None,
        visibility: str = "private",
        enable_chunking: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """Sync markdown files. Owner handling per RAG with Permissions guide:

        - owner_id: explicit UUID to assign to all synced docs (overrides metadata)
        - visibility: 'private' (default, assign owner), 'public' (owner_id = NULL, readable by all authenticated)
        - enable_chunking: True to split into chunks, False to store whole doc as single chunk
        - max_workers: Number of parallel worker threads for batch ingestion
        """
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

            # 5. Process changed documents in parallel batches
            batches = [changed_docs[i : i + batch_size] for i in range(0, len(changed_docs), batch_size)]

            if max_workers > 1 and len(batches) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _worker_task(batch_docs: list[ParsedDocument]) -> None:
                    self._process_batch(
                        batch_docs,
                        existing_map,
                        owner_id=owner_id,
                        visibility=visibility,
                        enable_chunking=enable_chunking,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_worker_task, b) for b in batches]
                    for f in as_completed(futures):
                        f.result()
            else:
                for batch in batches:
                    self._process_batch(
                        batch,
                        existing_map,
                        owner_id=owner_id,
                        visibility=visibility,
                        enable_chunking=enable_chunking,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

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
        owner_id: str | None = None,
        visibility: str = "private",
        enable_chunking: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        if not docs:
            return

        # Chunk each doc (portable, eval-stable)
        from supabase_easy_rag.ingestion.chunker import chunk_text as _chunk

        use_chunking = enable_chunking if enable_chunking is not None else self.enable_chunking
        c_size = chunk_size or self.chunk_size
        c_overlap = chunk_overlap or self.chunk_overlap

        doc_chunks: dict[str, list] = {}
        all_texts: list[str] = []
        for doc in docs:
            chunks = _chunk(
                doc.content,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                enable_chunking=use_chunking,
            )
            # fallback: if chunking produced 0 or enable_chunking is False, keep doc content (capped safely)
            if not chunks:
                from supabase_easy_rag.ingestion.chunker import Chunk as _C

                safe_content = doc.content[:4000] if len(doc.content) > 4000 else doc.content
                chunks = [_C(content=safe_content, chunk_index=0, char_count=len(safe_content), token_count=min(doc.token_count or 0, 1000))]
            doc_chunks[doc.document_key] = chunks
            all_texts.extend([c.content[:4000] for c in chunks])


        embeddings = self.provider.embed_texts(all_texts)
        # map back per doc
        embed_idx = 0

        for doc in docs:
            # Resolve owner_id per RLS guide: explicit param > document metadata > NULL (public)
            effective_owner = owner_id or doc.owner_id
            if effective_owner is None and visibility == "public":
                effective_owner = None  # public doc, readable by all authenticated
            # If visibility is private and no owner, rely on DB default auth.uid()
            doc_payload: dict[str, Any] = {
                "document_key": doc.document_key,
                "title": doc.title,
                "top_level_category": doc.top_level_category,
                "metadata": doc.metadata,
                "checksum": doc.checksum,
            }
            # Only set owner_id if explicitly provided; otherwise DB default (auth.uid()) applies on insert
            # On update we preserve existing owner unless override given
            if effective_owner is not None:
                doc_payload["owner_id"] = effective_owner
            elif visibility == "public":
                doc_payload["owner_id"] = None
            if doc.document_key in existing_map:
                doc_id = existing_map[doc.document_key]["id"]
                run_with_retry(
                    "update_document",
                    lambda: self._table("documents").update(doc_payload).eq("id", doc_id).execute(),
                )
            else:
                resp = run_with_retry(
                    "upsert_document",
                    lambda: self._table("documents").upsert(doc_payload, on_conflict="document_key").execute(),
                )
                doc_id = (resp.data or [{}])[0]["id"]

            # Insert Sections (bulk if possible)
            section_id_map: dict[str, str] = {}  # old temp id -> new db id
            if doc.sections:
                sec_payloads = [
                    {
                        "document_id": doc_id,
                        "heading": sec.heading,
                        "level": sec.level,
                        "sort_order": sec.sort_order,
                        "metadata": sec.metadata,
                    }
                    for sec in doc.sections
                ]
                resp_sec = run_with_retry(
                    "bulk_upsert_sections",
                    lambda p=sec_payloads: self._table("document_sections").upsert(p, on_conflict="document_id,sort_order").execute(),  # type: ignore[misc]
                )
                for sec, row in zip(doc.sections, resp_sec.data or []):
                    if isinstance(row, dict) and row.get("id"):
                        section_id_map[sec.id] = row["id"]

            # Insert Chunks (bulk upsert to guarantee idempotency across parallel threads)
            chunks = doc_chunks[doc.document_key]
            first_section_db_id = next(iter(section_id_map.values()), None)
            chunk_payloads = []
            for ch in chunks:
                emb = embeddings[embed_idx]
                embed_idx += 1
                chunk_payloads.append(
                    {
                        "document_id": doc_id,
                        "section_id": ch.section_id or first_section_db_id,
                        "chunk_index": ch.chunk_index,
                        "content": ch.content,
                        "metadata": {**doc.metadata, "facet_path": doc.facet_path} if doc.facet_path else doc.metadata,
                        "token_count": ch.token_count,
                        "char_count": ch.char_count,
                        "embedding": list(emb),
                    }
                )
            if chunk_payloads:
                run_with_retry(
                    "bulk_upsert_chunks",
                    lambda payload=chunk_payloads: self._table("chunks").upsert(payload, on_conflict="document_id,chunk_index").execute(),  # type: ignore[misc]
                )


            # Insert Facets (bulk upsert & link)
            if doc.facets:
                facet_payloads = [
                    {
                        "facet_type": f.facet_type,
                        "facet_key": f.facet_key,
                        "label": f.label,
                        "sort_order": f.sort_order,
                        "metadata": f.metadata,
                    }
                    for f in doc.facets
                ]
                facet_resp = run_with_retry(
                    "upsert_facets",
                    lambda payload=facet_payloads: self._table("facets")  # type: ignore[misc]
                    .upsert(payload, on_conflict="facet_key")
                    .execute(),
                )
                facet_rows = facet_resp.data or []
                doc_facet_links = [
                    {"document_id": doc_id, "facet_id": f_row["id"]}
                    for f_row in facet_rows
                    if isinstance(f_row, dict) and f_row.get("id")
                ]
                if doc_facet_links:
                    run_with_retry(
                        "link_doc_facets",
                        lambda links=doc_facet_links: self._table("document_facets")  # type: ignore[misc]
                        .upsert(links, on_conflict="document_id,facet_id")
                        .execute(),
                    )
