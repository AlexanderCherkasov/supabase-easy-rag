from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from supabase_easy_rag.config import EasyRagConfig
from supabase_easy_rag.core.client import EasyRagClient
from supabase_easy_rag.providers.base import BaseEmbeddingProvider

app = typer.Typer(
    name="easy-rag",
    help="Supabase Easy RAG - Production-ready Hybrid Vector + Full-Text Search Engine",
    add_completion=False,
)
console = Console()


def _make_client(user_jwt: str | None = None, use_rls: bool = False) -> EasyRagClient:
    """Execution-layer helper — explicitly chooses connector (only place with vendor branching)."""
    cfg = EasyRagConfig.from_env()
    # Choose connector explicitly based on config endpoint — example of developer choice
    emb: BaseEmbeddingProvider | None = None
    if cfg.embedding.api_key and cfg.embedding.model and cfg.embedding.endpoint:
        # Explicit Azure vs OpenAI — developer decides, not lib
        if "openai.azure.com" in cfg.embedding.endpoint or "services.ai.azure.com" in cfg.embedding.endpoint:
            from supabase_easy_rag.providers.azure import AzureEmbeddingProvider

            emb = AzureEmbeddingProvider(api_key=cfg.embedding.api_key, endpoint=cfg.embedding.endpoint, model=cfg.embedding.model)
        else:
            from supabase_easy_rag.providers.openai import OpenAIEmbeddingProvider

            emb = OpenAIEmbeddingProvider(api_key=cfg.embedding.api_key, model=cfg.embedding.model, base_url=cfg.embedding.endpoint)
    return EasyRagClient(embedding_provider=emb, user_jwt=user_jwt, use_rls=use_rls or bool(user_jwt))


@app.command("init-sql")
def init_sql(
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Directory to save SQL migration scripts to"
    ),
    dimensions: int = typer.Option(
        1536, "--dimensions", "-d", help="pgvector embedding dimensionality (e.g. 384, 768, 1536, 3072)"
    ),
):
    sql_dir = Path(__file__).resolve().parent / "sql"
    if not sql_dir.exists():
        sql_dir = Path(__file__).resolve().parent.parent / "sql"
    schema_sql = (sql_dir / "01_schema.sql").read_text(encoding="utf-8")

    if dimensions != 1536:
        schema_sql = schema_sql.replace("VECTOR(1536)", f"VECTOR({dimensions})")
    functions_sql = (sql_dir / "02_functions.sql").read_text(encoding="utf-8")
    combined_sql = f"{schema_sql}\n\n{functions_sql}"

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "01_schema.sql").write_text(schema_sql, encoding="utf-8")
        (output_dir / "02_functions.sql").write_text(functions_sql, encoding="utf-8")
        rprint(f"[bold green]✓ SQL migrations saved to {output_dir} (dimensions={dimensions})[/bold green]")
    else:
        rprint(combined_sql)


@app.command("sync")
def sync_docs(
    directory: Path = typer.Argument(..., help="Path to markdown files directory"),
    pattern: str | None = typer.Option(None, "--pattern", "-p", help="Filter files by substring"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
    owner_id: str | None = typer.Option(None, "--owner-id", help="Owner UUID for RLS (documents.owner_id). Overrides metadata."),
    public: bool = typer.Option(False, "--public", help="Make documents public (owner_id=NULL, readable by all authenticated)"),
):
    """Sync a directory of Markdown files into Supabase RAG database.

    RLS tip (per Supabase RAG with Permissions): use --owner-id to assign ownership,
    or --public for shared knowledge base. Without flags, uses auth.uid() default or metadata Owner ID.
    Use --no-chunking to ingest full documents without splitting into sub-chunks.
    """
    rprint(f"[bold blue]Starting sync for directory:[/bold blue] {directory}")
    client = _make_client()
    result = client.sync_directory(
        source_dir=directory,
        pattern=pattern,
        limit=limit,
        owner_id=owner_id,
        visibility="public" if public else "private",
    )

    rprint("[bold green]✓ Sync Completed![/bold green]")
    rprint(f"Files seen: {result.get('files_seen')}")
    rprint(f"Files changed/synced: {result.get('files_changed')}")


@app.command("query")
def query_rag(
    query_string: str = typer.Argument(..., help="Search query string"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, fts"),
    match_count: int = typer.Option(5, "--count", "-c", help="Number of results"),
    kb_token: str | None = typer.Option(None, "--token", "-t", help="Access token (token mode)"),
    use_rls: bool = typer.Option(False, "--rls", help="Use RLS mode (auth.uid() via SUPABASE_ANON_KEY + user JWT)"),
    user_jwt: str | None = typer.Option(None, "--user-jwt", help="User JWT for RLS mode"),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF smoothing constant k (default 60)"),
    vector_weight: float = typer.Option(1.0, "--vector-weight", help="RRF vector component weight"),
    text_weight: float = typer.Option(1.0, "--text-weight", help="RRF lexical component weight"),
    fts_config: str = typer.Option("english", "--fts-config", help="Text search config (e.g. english, russian, simple)"),
    min_similarity: float | None = typer.Option(None, "--min-similarity", help="Minimum cosine vector similarity threshold"),
    candidate_count: int | None = typer.Option(None, "--candidate-count", help="Oversampling candidate count"),
):
    """Query the Supabase RAG engine directly from terminal with RRF rank fusion."""
    client = _make_client(user_jwt=user_jwt, use_rls=use_rls or bool(user_jwt))
    # If RLS, token is None -> _rls RPC; else use token
    token: str | None = kb_token or client.config.knowledgebase_access_token
    if use_rls or user_jwt:
        token = None

    if mode == "vector":
        results = client.search_vector(
            query_string,
            kb_token=token,
            match_count=match_count,
            min_vector_similarity=min_similarity,
            use_rls=bool(use_rls or user_jwt),
        )
    elif mode == "fts":
        results = client.search_fts(
            query_string,
            kb_token=token,
            match_count=match_count,
            fts_config=fts_config,
            use_rls=bool(use_rls or user_jwt),
        )
    else:
        results = client.search_hybrid(
            query_string,
            kb_token=token,
            match_count=match_count,
            candidate_count=candidate_count,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
            fts_config=fts_config,
            min_vector_similarity=min_similarity,
            use_rls=bool(use_rls or user_jwt),
        )

    table = Table(title=f"RAG Search Results ({mode.upper()})")
    table.add_column("Score", style="cyan", no_wrap=True)
    table.add_column("Ranks (V/T)", style="blue", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Section", style="green")
    table.add_column("Excerpt", style="yellow")

    for item in results:
        score = f"{item.hybrid_score or item.vector_score or item.text_score or 0.0:.4f}"
        vr = str(item.vector_rank) if item.vector_rank is not None else "-"
        tr = str(item.text_rank) if item.text_rank is not None else "-"
        table.add_row(
            score,
            f"V:{vr} / T:{tr}",
            item.document_title,
            item.section_title or "-",
            item.chunk_text[:100] + "...",
        )

    console.print(table)


@app.command("create-token")
def create_token(
    name: str = typer.Argument(..., help="Token descriptive name"),
    expires_in_days: int | None = typer.Option(None, "--expires-days", help="Expiry in days"),
):
    """Generate a new RAG access token and save it to database."""
    client = EasyRagClient()
    raw_token, row = client.tokens.create_token(name=name)

    rprint("[bold green]✓ Token Created Successfully![/bold green]")
    rprint(f"[bold yellow]Token Secret (save this):[/bold yellow] {raw_token}")
    rprint(f"Token ID: {row.get('id')}")


@app.command("list-tokens")
def list_tokens():
    """List all registered access tokens."""
    client = EasyRagClient()
    tokens = client.tokens.list_tokens()

    table = Table(title="Access Tokens")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Active", style="green")
    table.add_column("Last Used", style="yellow")

    for tok in tokens:
        table.add_row(
            str(tok.get("id")),
            str(tok.get("token_name")),
            "Yes" if tok.get("is_active") else "No",
            str(tok.get("last_used_at") or "Never"),
        )

    console.print(table)


if __name__ == "__main__":
    app()
