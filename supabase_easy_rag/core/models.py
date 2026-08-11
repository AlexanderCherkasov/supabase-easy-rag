from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FacetDefinition:
    facet_type: str
    facet_key: str
    label: str
    parent_facet_key: str | None = None
    sort_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionDefinition:
    id: str
    heading: str
    level: int
    sort_order: int
    parent_section_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    document_key: str
    title: str
    top_level_category: str | None
    metadata: dict[str, Any]
    checksum: str
    content: str
    sections: list[SectionDefinition]
    facets: list[FacetDefinition]
    facet_path: str | None
    token_count: int
    char_count: int
    owner_id: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    document_title: str
    section_title: str | None
    chunk_text: str
    facet_path: str | None
    metadata: dict[str, Any]
    vector_score: float | None = None
    text_score: float | None = None
    hybrid_score: float | None = None
