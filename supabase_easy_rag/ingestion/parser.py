from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from supabase_easy_rag.core.models import (
    FacetDefinition,
    ParsedDocument,
    SectionDefinition,
)

_METADATA_LINE_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+?)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SECTION_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*\S)\s*$")
_SPLIT_LIST_RE = re.compile(r"\s*,\s*")


def normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def display_label(value: str) -> str:
    trimmed = re.sub(r"^\d+[_\-\s]*", "", value.strip())
    cleaned = re.sub(r"[_\-]+", " ", trimmed).strip()
    return cleaned or value.strip()


def document_key_for_path(file_path: Path, source_root: Path) -> str:
    return file_path.resolve().relative_to(source_root.resolve()).as_posix().lower()


def checksum_for_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_heading(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return display_label(Path(fallback).stem)


def extract_metadata_block(markdown: str) -> dict[str, Any]:
    lines = markdown.splitlines()
    inside_metadata = False
    metadata: dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## metadata":
            inside_metadata = True
            continue
        if inside_metadata and stripped.startswith("## "):
            break
        if inside_metadata and stripped == "---":
            if metadata:
                break
            continue
        if inside_metadata:
            match = _METADATA_LINE_RE.match(stripped)
            if not match:
                continue
            raw_key, raw_value = match.groups()
            metadata[normalize_key(raw_key)] = raw_value.strip()

    return metadata


def extract_sections(markdown: str) -> list[SectionDefinition]:
    sections: list[SectionDefinition] = []
    parent_stack: dict[int, str] = {}

    for line in markdown.splitlines():
        match = _SECTION_HEADING_RE.match(line.strip())
        if not match:
            continue
        heading_level = len(match.group(1))
        heading = match.group(2).strip()
        if heading.lower() == "metadata":
            continue
        section_id = str(uuid.uuid4())
        parent_section_id = None
        for candidate_level in range(heading_level - 1, 1, -1):
            if candidate_level in parent_stack:
                parent_section_id = parent_stack[candidate_level]
                break
        parent_stack[heading_level] = section_id
        for candidate_level in list(parent_stack.keys()):
            if candidate_level > heading_level:
                parent_stack.pop(candidate_level, None)
        sections.append(
            SectionDefinition(
                id=section_id,
                heading=heading,
                level=heading_level,
                sort_order=len(sections),
                parent_section_id=parent_section_id,
                metadata={"anchor": normalize_key(heading)},
            )
        )

    return sections


def parse_markdown_document(
    file_path: Path, source_root: Path
) -> ParsedDocument:
    from supabase_easy_rag.ingestion.facets import build_facets_for_file

    content = file_path.read_text(encoding="utf-8")
    metadata = extract_metadata_block(content)
    all_facets, top_level_category, facet_path = build_facets_for_file(
        file_path=file_path, source_root=source_root, metadata=metadata
    )
    chunk_metadata = dict(metadata)
    if facet_path:
        chunk_metadata["facet_path"] = facet_path

    return ParsedDocument(
        document_key=document_key_for_path(file_path, source_root),
        title=first_heading(content, file_path.name),
        top_level_category=top_level_category,
        metadata=chunk_metadata,
        checksum=checksum_for_text(content),
        content=content,
        sections=extract_sections(content),
        facets=all_facets,
        facet_path=facet_path,
        token_count=len(re.findall(r"\S+", content)),
        char_count=len(content),
    )
