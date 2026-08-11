from __future__ import annotations

from abc import ABC, abstractmethod


class BaseChatProvider(ABC):
    """Abstract chat provider — model-agnostic, no vendor lock-in."""

    @abstractmethod
    def chat(self, prompt: str, system: str = "You are helpful.", max_tokens: int = 512, temperature: float | None = None) -> str:
        pass

    def generate(self, question: str, context: str, system: str | None = None) -> str:
        sys = system or "You are RAG assistant. Answer only from context. If not in context, say 'Not found in knowledge base'."
        prompt = f"Context:\n{context[:6000]}\n\nQuestion: {question}\nAnswer:"
        return self.chat(prompt=prompt, system=sys, max_tokens=512)
