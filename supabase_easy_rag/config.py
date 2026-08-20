from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    for candidate in (".env.local", ".env"):
        env_file = Path(candidate)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    endpoint: str | None
    api_key: str
    api_version: str | None = None


AzureConfig = ProviderConfig


@dataclass(frozen=True)
class EasyRagConfig:
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str
    knowledgebase_access_token: str
    schema_name: str
    embedding: ProviderConfig
    chat_nano: ProviderConfig
    chat_mini: ProviderConfig
    azure_nano: ProviderConfig
    azure_mini: ProviderConfig
    azure_embedding: ProviderConfig
    embedding_model: str
    embedding_dim: int
    batch_size: int
    default_match_count: int
    use_rls: bool
    enable_chunking: bool
    chunk_size: int
    chunk_overlap: int
    openai_api_key: str
    openai_endpoint: str | None
    openai_api_version: str | None
    fts_config: str = "english"
    rrf_k: int = 60
    vector_weight: float = 1.0
    text_weight: float = 1.0
    candidate_count: int | None = None
    min_vector_similarity: float | None = None

    @classmethod
    def from_env(cls) -> EasyRagConfig:
        _load_env()
        use_rls_raw = os.getenv("KNOWLEDGEBASE_USE_RLS", os.getenv("USE_RLS", "false"))
        enable_chunking_raw = os.getenv("KNOWLEDGEBASE_ENABLE_CHUNKING", os.getenv("ENABLE_CHUNKING", "true"))
        service_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        anon_key = os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

        # --- Simplified: LLM_* is primary, fallback to AZURE_*/KNOWLEDGEBASE_* for compat ---
        llm_endpoint = (
            os.getenv("LLM_ENDPOINT")
            or os.getenv("KNOWLEDGEBASE_CHAT_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT_NANO")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
        )
        llm_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("KNOWLEDGEBASE_CHAT_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY_NANO")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        llm_model = (
            os.getenv("LLM_MODEL")
            or os.getenv("KNOWLEDGEBASE_CHAT_MODEL")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NANO")
            or "gpt-5.4-nano"
        )
        llm_mini_model = os.getenv("LLM_MINI_MODEL") or os.getenv("KNOWLEDGEBASE_CHAT_MODEL_MINI") or os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI") or (llm_model.replace("nano", "mini") if "nano" in llm_model else llm_model)
        llm_version = os.getenv("AZURE_OPENAI_API_VERSION_NANO") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        chat_nano = ProviderConfig(provider="azure" if llm_endpoint and "azure" in llm_endpoint else "openai_like", model=llm_model, endpoint=llm_endpoint, api_key=llm_key, api_version=llm_version)
        chat_mini = ProviderConfig(provider=chat_nano.provider, model=llm_mini_model, endpoint=llm_endpoint, api_key=llm_key, api_version=llm_version)

        # Embedding: re-use LLM endpoint/key if not set separately
        emb_model = os.getenv("EMBEDDING_MODEL") or os.getenv("KNOWLEDGEBASE_EMBEDDING_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING") or "text-embedding-3-small"
        emb_endpoint = os.getenv("EMBEDDING_ENDPOINT") or os.getenv("KNOWLEDGEBASE_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING") or llm_endpoint
        emb_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("KNOWLEDGEBASE_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING") or llm_key
        emb_provider = "azure" if emb_endpoint and "azure" in emb_endpoint else "openai_like"
        embedding = ProviderConfig(provider=emb_provider, model=emb_model, endpoint=emb_endpoint, api_key=emb_key, api_version=llm_version)

        cand_count_env = os.getenv("KNOWLEDGEBASE_CANDIDATE_COUNT")
        cand_count = int(cand_count_env) if cand_count_env and cand_count_env.isdigit() else None

        min_sim_env = os.getenv("KNOWLEDGEBASE_MIN_VECTOR_SIMILARITY")
        min_sim = float(min_sim_env) if min_sim_env is not None and min_sim_env != "" else None

        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_service_role_key=service_key,
            supabase_anon_key=anon_key,
            knowledgebase_access_token=os.getenv("KNOWLEDGEBASE_ACCESS_TOKEN", ""),
            schema_name=os.getenv("KNOWLEDGEBASE_SCHEMA", "knowledgebase"),
            embedding=embedding,
            chat_nano=chat_nano,
            chat_mini=chat_mini,
            azure_nano=chat_nano,
            azure_mini=chat_mini,
            azure_embedding=embedding,
            embedding_model=emb_model,
            embedding_dim=int(os.getenv("KNOWLEDGEBASE_EMBEDDING_DIM", "1536")),
            batch_size=int(os.getenv("KNOWLEDGEBASE_PROCESS_BATCH_SIZE", "20")),
            default_match_count=int(os.getenv("KNOWLEDGEBASE_DEFAULT_MATCH_COUNT", "5")),
            use_rls=use_rls_raw.lower() in ("1", "true", "yes", "on"),
            enable_chunking=enable_chunking_raw.lower() in ("1", "true", "yes", "on"),
            chunk_size=int(os.getenv("KNOWLEDGEBASE_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("KNOWLEDGEBASE_CHUNK_OVERLAP", "100")),
            openai_api_key=llm_key,
            openai_endpoint=llm_endpoint,
            openai_api_version=llm_version,
            fts_config=os.getenv("KNOWLEDGEBASE_FTS_CONFIG", "english"),
            rrf_k=int(os.getenv("KNOWLEDGEBASE_RRF_K", "60")),
            vector_weight=float(os.getenv("KNOWLEDGEBASE_VECTOR_WEIGHT", "1.0")),
            text_weight=float(os.getenv("KNOWLEDGEBASE_TEXT_WEIGHT", "1.0")),
            candidate_count=cand_count,
            min_vector_similarity=min_sim,
        )


@lru_cache(maxsize=1)
def get_config() -> EasyRagConfig:
    return EasyRagConfig.from_env()
