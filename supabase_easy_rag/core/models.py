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
    vector_rank: int | None = None
    text_rank: int | None = None
    section_id: str | None = None
    expanded_text: str | None = None

    @property
    def final_score(self) -> float | None:
        """Alias for hybrid_score / primary result score."""
        return self.hybrid_score if self.hybrid_score is not None else (self.vector_score if self.vector_score is not None else self.text_score)

    @property
    def effective_text(self) -> str:
        """Returns expanded_text if available, otherwise chunk_text."""
        return self.expanded_text or self.chunk_text

    @property
    def vector_similarity(self) -> float | None:
        """Alias for vector_score."""
        return self.vector_score
