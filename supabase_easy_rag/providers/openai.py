from __future__ import annotations

import time
from collections.abc import Sequence

from openai import OpenAI

from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.providers.chat_base import BaseChatProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Explicit OpenAI (or compatible) connector."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        batch_size: int = 20,
        batch_sleep: float = 0.2,
    ) -> None:
        if not api_key or not model:
            raise ValueError("OpenAIEmbeddingProvider requires api_key, model")
        self.model: str = model
        self.batch_size: int = batch_size
        self.batch_sleep: float = batch_sleep
        if base_url:
            self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([item.embedding for item in resp.data])
            if start + self.batch_size < len(texts):
                time.sleep(self.batch_sleep)
        return out


class OpenAIChatProvider(BaseChatProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        if not api_key or not model:
            raise ValueError("OpenAIChatProvider requires api_key, model")
        self.model: str = model
        if base_url:
            self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, prompt: str, system: str = "You are helpful.", max_tokens: int = 512, temperature: float | None = None) -> str:
        # 1. Fallback for new OpenAI Responses API (client.responses.create)
        if hasattr(self.client, "responses") and callable(getattr(self.client.responses, "create", None)):
            try:
                resp = getattr(self.client.responses, "create")(
                    model=self.model,
                    input=[
                        {"role": "developer", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text:
                    return str(resp.output_text)
            except Exception:
                pass  # Fallback to chat.completions below

        # 2. Standard chat completions with graceful fallback for reasoning models (o1, o3, etc.)
        is_reasoning_model = any(self.model.startswith(prefix) for prefix in ("o1", "o3", "o-"))
        system_role = "developer" if is_reasoning_model else "system"
        messages = [{"role": system_role, "content": system}, {"role": "user", "content": prompt}]

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        if is_reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

        try:
            resp = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if "max_tokens" in err_str and "max_completion_tokens" in err_str:
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = max_tokens
            if "temperature" in err_str:
                kwargs.pop("temperature", None)
            resp = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
            return resp.choices[0].message.content or ""
