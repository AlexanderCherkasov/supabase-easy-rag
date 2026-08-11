from __future__ import annotations

"""Providers — abstract interfaces + explicit vendor connectors (like openai-agents).

Core (BaseEmbeddingProvider/BaseChatProvider) has no vendor code.
Concrete connectors: supabase_easy_rag.providers.azure / openai — each explicit, imported only in examples.
"""

from supabase_easy_rag.providers.base import BaseEmbeddingProvider
from supabase_easy_rag.providers.chat_base import BaseChatProvider
from supabase_easy_rag.providers.local_provider import CustomChatProvider, CustomEmbeddingProvider

__all__ = ["BaseChatProvider", "BaseEmbeddingProvider", "CustomChatProvider", "CustomEmbeddingProvider"]
