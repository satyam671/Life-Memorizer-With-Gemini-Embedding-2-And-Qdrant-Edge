"""`life-memorizer` command line interface."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .embeddings import build_embedder
from .ingest import Ingestor
from .mock_data import seed_store
from .models import Modality
from .rag import LocalRAG
from .recall import RecallEngine
from .store import MemoryStore

app = typer.Typer(
    add_completion=False,
    help="Offline multi-modal life memorizer (Gemini 2.0 + Qdrant Edge).",
)
console = Console()


def _bootstrap():
    settings = get_settings()
    embedder = build_embedder(settings)
    store = MemoryStore(settings)
    return settings, embedder, store


@app.command()
def init() -> None:
    """Initialize (or verify) the local Qdrant Edge collection."""
    settings, _embedder, store = _bootstrap()
    store.ensure_collection()
    console.print(
        f"[green]Ready[/green] collection=[bold]{settings.collection}[/bold] "
        f"path=[bold]{settings.db_path}[/bold] dim=[bold]{settings.embed_dim}[/bold] "
        f"quantization=[bold]{settings.quantization.value}[/bold]"
    )


@app.command()
def ingest(
    video: str = typer.Option(..., "--video", help="Path to a video clip to ingest."),
    location: str = typer.Option("Unknown", "--location", help="location_context tag."),
) -> None:
    """Ingest a video clip (frames + audio + OCR) into local memory."""
    _settings, embedder, store = _bootstrap()
    store.ensure_collection()
    ingestor = Ingestor(_settings, embedder, store)
    moments = ingestor.ingest_video(video, location=location)
    console.print(
        f"[green]Ingested[/green] {len(moments)} moments from "
        f"[bold]{video}[/bold] @ [bold]{location}[/bold]"
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="Natural-language query (or file path)."),
    modality: Modality = typer.Option(
        Modality.text, "--query-modality", help="Modality of the query input."
    ),
    target: Optional[Modality] = typer.Option(
        None, "--modality", help="Which memory space to search (visual/audio/text)."
    ),
    location: Optional[str] = typer.Option(None, "--location", help="Filter by location_context."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Fuse all vector spaces with weights."),
    limit: int = typer.Option(5, "--limit", min=1, help="Max results."),
) -> None:
    """Recall moments from local memory."""
    settings, embedder, store = _bootstrap()
    store.ensure_collection()
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall(
        query=query,
        modality=modality,
        target=target,
        limit=limit,
        location_context=location,
        hybrid=hybrid,
    )
    if not hits:
        console.print("[yellow]No matching moments found.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"Recall: {query!r}")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("When")
    table.add_column("Where")
    table.add_column("Matched")
    table.add_column("Note / Media", overflow="fold")
    for hit in hits:
        m = hit.moment
        note = m.ocr_text or m.transcript or (m.media_file_path or "")
        table.add_row(
            f"{hit.score:.3f}",
            m.timestamp.strftime("%Y-%m-%d %H:%M"),
            m.location_context,
            hit.matched_vector,
            note,
        )
    console.print(table)


@app.command()
def seed() -> None:
    """Load a realistic mock memory session (lost keys, Sarah's list, cafe menu).

    Useful for trying the three tutorial recall scenarios without recording video.
    Works fully offline with LIFE_MEMORIZER_FAKE_EMBEDDINGS=1.
    """
    _settings, embedder, store = _bootstrap()
    store.ensure_collection(recreate=True)
    written = seed_store(embedder, store)
    console.print(
        f"[green]Seeded[/green] {written} realistic moments. Try:\n"
        "  life-memorizer recall \"where did I leave my keys?\" --modality image\n"
        "  life-memorizer recall \"what did Sarah say to buy?\" --modality audio\n"
        "  life-memorizer recall \"the cafe menu\" --location Cafe --hybrid"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="A natural-language question about your memories."),
    location: Optional[str] = typer.Option(None, "--location", help="Filter by location_context."),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Search all vector spaces."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Memories used as context."),
    show_sources: bool = typer.Option(
        True, "--show-sources/--no-sources", help="Print the memories used to answer."
    ),
) -> None:
    """Ask a question and get an answer grounded in your local memories (local RAG)."""
    settings, embedder, store = _bootstrap()
    store.ensure_collection()
    rag = LocalRAG(settings, embedder, store)
    result = rag.ask(question, location_context=location, hybrid=hybrid, limit=limit)

    console.print(f"[bold cyan]Answer:[/bold cyan] {result.answer}")
    if show_sources and result.sources:
        table = Table(title="Grounded on")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("When")
        table.add_column("Where")
        table.add_column("Memory", overflow="fold")
        for hit in result.sources:
            m = hit.moment
            note = m.ocr_text or m.transcript or (m.media_file_path or "")
            table.add_row(
                f"{hit.score:.3f}",
                m.timestamp.strftime("%Y-%m-%d %H:%M"),
                m.location_context,
                note,
            )
        console.print(table)


@app.command()
def stats() -> None:
    """Show how many moments are stored locally."""
    settings, _embedder, store = _bootstrap()
    store.ensure_collection()
    console.print(
        f"[bold]{store.count()}[/bold] moments in "
        f"[bold]{settings.collection}[/bold] ([bold]{settings.db_path}[/bold])"
    )


@app.command()
def prune(
    ttl_days: Optional[int] = typer.Option(
        None, "--ttl-days", help="Override TTL in days (0 disables)."
    ),
    summarize: bool = typer.Option(
        True,
        "--summarize/--no-summarize",
        help="Summarize expired moments into compact digests before deletion "
        "(keeps storage roughly constant) instead of dropping them.",
    ),
) -> None:
    """Age out moments older than the TTL (optionally summarizing them first)."""
    _settings, _embedder, store = _bootstrap()
    store.ensure_collection()
    if summarize:
        removed = store.summarize_expired(ttl_days)
        console.print(
            f"[green]Summarized[/green] expired moments into digests; "
            f"net {removed} points removed."
        )
    else:
        removed = store.prune_expired(ttl_days)
        console.print(f"[green]Pruned[/green] {removed} expired moments.")


if __name__ == "__main__":  # pragma: no cover
    app()
