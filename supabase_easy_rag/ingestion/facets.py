from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from supabase_easy_rag.core.models import FacetDefinition

_SPLIT_LIST_RE = re.compile(r"\s*,\s*")


def normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def display_label(value: str) -> str:
    trimmed = re.sub(r"^\d+[_\-\s]*", "", value.strip())
    cleaned = re.sub(r"[_\-]+", " ", trimmed).strip()
    return cleaned or value.strip()


def path_facets_for_file(
    file_path: Path, source_root: Path
) -> tuple[list[FacetDefinition], str | None, str | None]:
    relative_path = file_path.resolve().relative_to(source_root.resolve())
    folder_parts = list(relative_path.parts[:-1])
    if not folder_parts:
        return [], None, None

    facets: list[FacetDefinition] = []
    labels: list[str] = []
    parent_facet_key: str | None = None
    normalized_parts: list[str] = []

    for index, part in enumerate(folder_parts):
        label = display_label(part)
        labels.append(label)
        normalized_parts.append(normalize_key(part))
        facet_key = f"path:{'/'.join(normalized_parts)}"
        facets.append(
            FacetDefinition(
                facet_type="path",
                facet_key=facet_key,
                label=label,
                parent_facet_key=parent_facet_key,
                sort_order=index,
                metadata={"depth": index},
            )
        )
        parent_facet_key = facet_key

    return facets, display_label(folder_parts[0]), " / ".join(labels)


def metadata_facets(metadata: dict[str, Any]) -> list[FacetDefinition]:
    facets: list[FacetDefinition] = []
    candidate_fields = ("classification", "category", "author", "tags", "topic")

    for field_name in candidate_fields:
        raw_value = metadata.get(field_name)
        if not raw_value or not isinstance(raw_value, str):
            continue
        values = (
            [raw_value]
            if field_name not in ("tags", "topic")
            else [v for v in _SPLIT_LIST_RE.split(raw_value) if v]
        )
        for index, value in enumerate(values):
            facets.append(
                FacetDefinition(
                    facet_type=field_name,
                    facet_key=f"{field_name}:{normalize_key(value)}",
                    label=value.strip(),
                    sort_order=index,
                )
            )

    return facets


def dedupe_facets(facets: Iterable[FacetDefinition]) -> list[FacetDefinition]:
    deduped: dict[str, FacetDefinition] = {}
    for facet in facets:
        deduped[facet.facet_key] = facet
    return list(deduped.values())


def build_facets_for_file(
    file_path: Path, source_root: Path, metadata: dict[str, Any]
) -> tuple[list[FacetDefinition], str | None, str | None]:
    p_facets, top_level_category, facet_path = path_facets_for_file(file_path, source_root)
    m_facets = metadata_facets(metadata)
    all_facets = dedupe_facets([*p_facets, *m_facets])
    return all_facets, top_level_category, facet_path
