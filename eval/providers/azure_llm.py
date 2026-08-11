from __future__ import annotations

import os
from typing import Optional

from supabase_easy_rag.config import EasyRagConfig


def get_client(which: str = "nano"):
    """Get OpenAI or Azure client — model concrete from config."""
    cfg = EasyRagConfig.from_env()
    prov = cfg.chat_nano if which == "nano" else cfg.chat_mini
    if not prov.api_key:
        raise RuntimeError(f"Chat {which} not configured: set KNOWLEDGEBASE_CHAT_* or OPENAI_API_KEY in .env (model={prov.model})")
    if prov.provider == "azure" and prov.endpoint:
        from supabase_easy_rag.providers.azure import AzureChatProvider
        chat_prov = AzureChatProvider(api_key=prov.api_key, endpoint=prov.endpoint, model=prov.model, api_version=prov.api_version or "2024-02-15-preview")
    else:
        from supabase_easy_rag.providers.openai import OpenAIChatProvider
        chat_prov = OpenAIChatProvider(api_key=prov.api_key, model=prov.model, base_url=prov.endpoint)
    return chat_prov.client, prov.model


def chat(which: str = "nano", prompt: str = "hello", system: str = "You are helpful.", temperature: float | None = None, max_tokens: int = 512) -> str:
    cfg = EasyRagConfig.from_env()
    prov = cfg.chat_nano if which == "nano" else cfg.chat_mini
    if prov.provider == "azure" and prov.endpoint:
        from supabase_easy_rag.providers.azure import AzureChatProvider
        chat_prov = AzureChatProvider(api_key=prov.api_key, endpoint=prov.endpoint, model=prov.model, api_version=prov.api_version or "2024-02-15-preview")
    else:
        from supabase_easy_rag.providers.openai import OpenAIChatProvider
        chat_prov = OpenAIChatProvider(api_key=prov.api_key, model=prov.model, base_url=prov.endpoint)
    return chat_prov.chat(prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature)


def judge_faithfulness(question: str, context: str) -> float:
    """LLM judge (nano) returns 0..1 faithfulness.
    Uses gpt-5-nano strictly as requested.
    """
    prompt = f"Question: {question}\nContext: {context[:4000]}\n\nRate if context contains answer to question (0.0 to 1.0). Respond with only number."
    try:
        out = chat("nano", prompt, system="You are an eval judge. Reply only with float 0-1.", temperature=0.0, max_tokens=10)
        return float(out.strip().split()[0])
    except Exception:
        return 0.0


def generate_answer(which: str = "mini", question: str = "", context: str = "") -> str:
    """Generate RAG answer using mini (strictly mini for tests)."""
    system = "You are RAG assistant. Answer only from context. If not in context, say 'Not found in knowledge base'."
    prompt = f"Context:\n{context[:6000]}\n\nQuestion: {question}\nAnswer:"
    return chat(which, prompt, system=system, temperature=0.0)
