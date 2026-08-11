"""
supabase-easy-rag: Production-ready Hybrid RAG engine for Supabase.
"""

from supabase_easy_rag.config import EasyRagConfig, get_config
from supabase_easy_rag.core.client import EasyRagClient
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

__version__ = "0.1.0"

__all__ = [
    "EasyRagConfig",
    "get_config",
    "EasyRagClient",
    "EasyRagError",
    "EasyRagAccessError",
    "EasyRagConfigurationError",
    "FacetDefinition",
    "ParsedDocument",
    "SearchResult",
    "SectionDefinition",
]
