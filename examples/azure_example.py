from __future__ import annotations

"""Azure example — developer explicitly chooses connector and injects into core.

No branching in lib: example decides which connector to use.
"""

from supabase_easy_rag import EasyRagClient
from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.providers.azure import AzureChatProvider, AzureEmbeddingProvider


def main() -> None:
    cfg = EasyRagConfig.from_env()

    # Explicit Azure connector — model concrete in example (gpt-5.4-nano)
    embedding = AzureEmbeddingProvider(
        api_key=cfg.embedding.api_key,
        endpoint=cfg.embedding.endpoint or "",
        model=cfg.embedding.model,  # text-embedding-3-large
    )
    chat = AzureChatProvider(
        api_key=cfg.chat_nano.api_key,
        endpoint=cfg.chat_nano.endpoint or "",
        model=cfg.chat_nano.model,  # gpt-5.4-nano
    )

    client = EasyRagClient(embedding_provider=embedding)

    # Sync (uses embedding connector)
    # client.sync_directory("./docs")

    # Search (hybrid)
    results = client.search_hybrid("How does RLS work?", match_count=5)
    for r in results:
        print(f"{r.document_title} | {r.hybrid_score}")

    # Chat (separate, explicit)
    answer = chat.chat(prompt="Explain RLS in one sentence", system="You are helpful.")
    print(answer)


if __name__ == "__main__":
    main()
