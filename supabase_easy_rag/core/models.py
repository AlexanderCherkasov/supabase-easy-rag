from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FacetDefinition:
    facet_type: str
    facet_key: str
    label: str
    parent_facet_key: Optional[str] = None
    sort_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionDefinition:
    id: str
    heading: str
    level: int
    sort_order: int
    parent_section_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    document_key: str
    title: str
    top_level_category: Optional[str]
    metadata: dict[str, Any]
    checksum: str
    content: str
    sections: list[SectionDefinition]
    facets: list[FacetDefinition]
    facet_path: Optional[str]
    token_count: int
    char_count: int


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    document_title: str
    section_title: Optional[str]
    chunk_text: str
    facet_path: Optional[str]
    metadata: dict[str, Any]
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    hybrid_score: Optional[float] = None
