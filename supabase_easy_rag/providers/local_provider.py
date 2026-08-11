from __future__ import annotations

from typing import Callable, Sequence

from supabase_easy_rag.providers.base import BaseEmbeddingProvider


class CustomEmbeddingProvider(BaseEmbeddingProvider):
    """Custom embedding provider wrapping a callback function."""

    def __init__(self, embed_fn: Callable[[Sequence[str]], list[list[float]]]):
        self._embed_fn = embed_fn

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_fn(texts)
