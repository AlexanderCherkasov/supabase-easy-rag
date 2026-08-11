class EasyRagError(Exception):
    """Base exception for supabase-easy-rag."""


class EasyRagConfigurationError(EasyRagError):
    """Raised when environment variables or configurations are missing or invalid."""


class EasyRagAccessError(EasyRagError):
    """Raised when token authentication or PostgREST access fails."""


class EasyRagIngestionError(EasyRagError):
    """Raised during document parsing or ingestion failure."""
