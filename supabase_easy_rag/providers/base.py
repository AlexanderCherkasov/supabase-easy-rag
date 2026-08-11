from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of text strings."""
        pass

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single query string."""
        results = self.embed_texts([query])
        if not results:
            raise RuntimeError("Embedding provider returned empty result for query")
        return results[0]
