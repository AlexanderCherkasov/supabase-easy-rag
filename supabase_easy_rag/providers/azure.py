from __future__ import annotations

import time
from collections.abc import Sequence

from openai import AzureOpenAI

from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.providers.chat_base import BaseChatProvider


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Explicit Azure OpenAI connector — no branching, developer picks in example."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: str = "2024-02-15-preview",
        batch_size: int = 100,
        batch_sleep: float = 0.0,
        dimensions: int | None = None,
    ) -> None:
        if not api_key or not endpoint or not model:
            raise ValueError("AzureEmbeddingProvider requires api_key, endpoint, model")
        self.model: str = model
        self.batch_size: int = batch_size
        self.batch_sleep: float = batch_sleep
        self.dimensions: int | None = dimensions
        self.client: AzureOpenAI = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            # Guard against exceeding 8192 OpenAI token limit on giant passages (especially non-latin scripts)
            batch = [t[:6000] if len(t) > 6000 else t for t in texts[start : start + self.batch_size]]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            out.extend([item.embedding for item in resp.data])
            if start + self.batch_size < len(texts) and self.batch_sleep > 0:
                time.sleep(self.batch_sleep)
        return out


class AzureChatProvider(BaseChatProvider):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: str = "2024-02-15-preview",
    ) -> None:
        if not api_key or not endpoint or not model:
            raise ValueError("AzureChatProvider requires api_key, endpoint, model")
        self.model: str = model
        self.client: AzureOpenAI = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    def chat(self, prompt: str, system: str = "You are helpful.", max_tokens: int = 512, temperature: float | None = None) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "temperature" in str(e).lower() and "only the default" in str(e).lower():
                kwargs["temperature"] = 1.0
                resp = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
                return resp.choices[0].message.content or ""
            raise
