from __future__ import annotations

import time
from typing import Optional, Sequence

from openai import AzureOpenAI, OpenAI

from supabase_easy_rag.core.exceptions import EasyRagConfigurationError
from supabase_easy_rag.providers.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI & Azure OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        endpoint: Optional[str] = None,
        api_version: Optional[str] = "2024-02-15-preview",
        batch_size: int = 20,
        batch_sleep: float = 0.2,
    ):
        if not api_key:
            raise EasyRagConfigurationError("API key is required for OpenAIEmbeddingProvider")

        self.model = model
        self.batch_size = batch_size
        self.batch_sleep = batch_sleep

        if endpoint:
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        else:
            self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
            if start + self.batch_size < len(texts):
                time.sleep(self.batch_sleep)
        return embeddings
