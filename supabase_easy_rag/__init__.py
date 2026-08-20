"""
supabase-easy-rag: Production-ready Hybrid RAG engine for Supabase.
"""

from supabase_easy_rag.config import EasyRagConfig, get_config
from supabase_easy_rag.core.client import AsyncEasyRagClient, EasyRagClient
from supabase_easy_rag.core.exceptions import (
    EasyRagAccessError,
    EasyRagConfigurationError,
    EasyRagError,
)
from supabase_easy_rag.core.models import (
    FacetDefinition,
    ParsedDocument,
    SearchResult,
    SectionDefinition,
)

__version__ = "0.1.2"


__all__ = [
    "AsyncEasyRagClient",
    "EasyRagAccessError",
    "EasyRagClient",
    "EasyRagConfig",
    "EasyRagConfigurationError",
    "EasyRagError",
    "FacetDefinition",
    "ParsedDocument",
    "SearchResult",
    "SectionDefinition",
    "get_config",
]

