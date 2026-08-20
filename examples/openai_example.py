from __future__ import annotations

"""OpenAI (or OpenAI-compatible) example — explicit connector injection."""

from supabase_easy_rag import EasyRagClient
from supabase_easy_rag.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider


def main() -> None:
    # Explicit OpenAI connector — developer chooses
    embedding = OpenAIEmbeddingProvider(
        api_key="sk-...",
        model="text-embedding-3-large",
        base_url="https://api.openai.com/v1",
    )
    chat = OpenAIChatProvider(
        api_key="sk-...",
        model="gpt-5.4-nano",
        base_url="https://api.openai.com/v1",
    )

    client = EasyRagClient(embedding_provider=embedding)
    results = client.search_hybrid("hello", match_count=5)
    print(results)

    print(chat.chat(prompt="hi", max_tokens=10))


if __name__ == "__main__":
    main()
