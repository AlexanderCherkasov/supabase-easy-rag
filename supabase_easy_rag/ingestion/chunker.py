from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    content: str
    chunk_index: int
    char_count: int
    token_count: int
    section_id: str | None = None


def _find_best_break(text: str, start: int, end: int) -> int:
    """Find best break point near end, preferring markdown boundaries."""
    window = text[start:end]
    # Prefer in order: double newline, heading, single newline, sentence end
    for sep in ["\n\n", "\n## ", "\n### ", "\n", ". ", " "]:
        idx = window.rfind(sep)
        # Only use separator if it's not too far from end (at least 50% of chunk)
        if idx != -1 and idx > len(window) * 0.5:
            return start + idx + len(sep)
    return end


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    section_id: str | None = None,
    enable_chunking: bool = True,
) -> list[Chunk]:
    if not text or not text.strip():
        return []
    chunk_size = max(10, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not enable_chunking or len(text) <= chunk_size:
        return [Chunk(content=text, chunk_index=0, char_count=len(text), token_count=len(re.findall(r"\S+", text)), section_id=section_id)]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_best_break(text, start, end)
        if end <= start:
            end = min(start + chunk_size, len(text))
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(Chunk(content=chunk_str, chunk_index=idx, char_count=len(chunk_str), token_count=len(re.findall(r"\S+", chunk_str)), section_id=section_id))
            idx += 1
        if end >= len(text):
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_document(
    content: str,
    sections: list,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    enable_chunking: bool = True,
) -> list[Chunk]:
    section_id = getattr(sections[0], "id", None) if sections else None
    return chunk_text(content, chunk_size, chunk_overlap, section_id=section_id, enable_chunking=enable_chunking)

