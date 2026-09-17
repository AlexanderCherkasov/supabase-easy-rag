from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of text strings."""

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding vector for a single query string."""
        results = self.embed_texts([query])
        if not results:
            raise RuntimeError("Embedding provider returned empty result for query")
        return results[0]

    @property
    def model_name(self) -> str | None:
        """Name or identifier of the embedding model."""
        return getattr(self, "_model_name", None) or getattr(self, "model", None) or getattr(self, "model_path_or_repo", None)


