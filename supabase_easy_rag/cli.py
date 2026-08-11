from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from supabase_easy_rag.core.client import EasyRagClient

app = typer.Typer(
    name="easy-rag",
    help="Supabase Easy RAG - Production-ready Hybrid Vector + Full-Text Search Engine",
    add_completion=False,
)
console = Console()


@app.command("init-sql")
def init_sql(
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Directory to save SQL migration scripts to"
    )
):
    """Output or save SQL schema migrations for Supabase."""
    sql_dir = Path(__file__).resolve().parent.parent / "sql"
    schema_sql = (sql_dir / "01_schema.sql").read_text(encoding="utf-8")
    functions_sql = (sql_dir / "02_functions.sql").read_text(encoding="utf-8")
    combined_sql = f"{schema_sql}\n\n{functions_sql}"

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "01_schema.sql").write_text(schema_sql, encoding="utf-8")
        (output_dir / "02_functions.sql").write_text(functions_sql, encoding="utf-8")
        rprint(f"[bold green]✓ SQL migrations saved to {output_dir}[/bold green]")
    else:
        rprint(combined_sql)


@app.command("sync")
def sync_docs(
    directory: Path = typer.Argument(..., help="Path to markdown files directory"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Filter files by substring"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
):
    """Sync a directory of Markdown files into Supabase RAG database."""
    rprint(f"[bold blue]Starting sync for directory:[/bold blue] {directory}")
    client = EasyRagClient()
    result = client.sync_directory(source_dir=directory, pattern=pattern, limit=limit)

    rprint(f"[bold green]✓ Sync Completed![/bold green]")
    rprint(f"Files seen: {result.get('files_seen')}")
    rprint(f"Files changed/synced: {result.get('files_changed')}")


@app.command("query")
def query_rag(
    query_string: str = typer.Argument(..., help="Search query string"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, fts"),
    match_count: int = typer.Option(5, "--count", "-c", help="Number of results"),
    kb_token: Optional[str] = typer.Option(None, "--token", "-t", help="Access token"),
):
    """Query the Supabase RAG engine directly from terminal."""
    client = EasyRagClient()
    token = kb_token or client.config.knowledgebase_access_token

    if mode == "vector":
        results = client.search_vector(query_string, kb_token=token, match_count=match_count)
    elif mode == "fts":
        results = client.search_fts(query_string, kb_token=token, match_count=match_count)
    else:
        results = client.search_hybrid(query_string, kb_token=token, match_count=match_count)

    table = Table(title=f"RAG Search Results ({mode.upper()})")
    table.add_column("Score", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Section", style="green")
    table.add_column("Excerpt", style="yellow")

    for item in results:
        score = f"{item.hybrid_score or item.vector_score or item.text_score or 0.0:.4f}"
        table.add_row(
            score,
            item.document_title,
            item.section_title or "-",
            item.chunk_text[:100] + "...",
        )

    console.print(table)


@app.command("create-token")
def create_token(
    name: str = typer.Argument(..., help="Token descriptive name"),
    expires_in_days: Optional[int] = typer.Option(None, "--expires-days", help="Expiry in days"),
):
    """Generate a new RAG access token and save it to database."""
    client = EasyRagClient()
    raw_token, row = client.tokens.create_token(name=name)

    rprint(f"[bold green]✓ Token Created Successfully![/bold green]")
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
