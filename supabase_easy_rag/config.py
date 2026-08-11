from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env() -> None:
    for candidate in (".env.local", ".env"):
        env_file = Path(candidate)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)


@dataclass(frozen=True)
class EasyRagConfig:
    supabase_url: str
    supabase_service_role_key: str
    knowledgebase_access_token: str
    schema_name: str
    openai_api_key: str
    openai_endpoint: Optional[str]
    openai_api_version: Optional[str]
    embedding_model: str
    embedding_dim: int
    batch_size: int
    default_match_count: int

    @classmethod
    def from_env(cls) -> EasyRagConfig:
        _load_env()
        return cls(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            knowledgebase_access_token=os.getenv("KNOWLEDGEBASE_ACCESS_TOKEN", ""),
            schema_name=os.getenv("KNOWLEDGEBASE_SCHEMA", "knowledgebase"),
            openai_api_key=os.getenv("OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY", "")),
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", None),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            embedding_model=os.getenv("KNOWLEDGEBASE_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=int(os.getenv("KNOWLEDGEBASE_EMBEDDING_DIM", "1536")),
            batch_size=int(os.getenv("KNOWLEDGEBASE_PROCESS_BATCH_SIZE", "20")),
            default_match_count=int(os.getenv("KNOWLEDGEBASE_DEFAULT_MATCH_COUNT", "5")),
        )


@lru_cache(maxsize=1)
def get_config() -> EasyRagConfig:
    return EasyRagConfig.from_env()
