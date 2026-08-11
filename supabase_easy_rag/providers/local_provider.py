from __future__ import annotations

from collections.abc import Callable, Sequence

from supabase_easy_rag.providers.base import BaseEmbeddingProvider


class CustomEmbeddingProvider(BaseEmbeddingProvider):
    """Custom embedding provider wrapping a callback function."""

    def __init__(self, embed_fn: Callable[[Sequence[str]], list[list[float]]]):
        self._embed_fn = embed_fn

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_fn(texts)


class CustomChatProvider:
    """Custom chat provider wrapping any callable LLM function (Claude, Gemini, Ollama, etc.)."""

    def __init__(self, chat_fn: Callable[[str, str], str]):
        self._chat_fn = chat_fn

    def chat(self, prompt: str, system: str = "You are helpful.", max_tokens: int = 512, temperature: float | None = None) -> str:
        return self._chat_fn(prompt, system)

