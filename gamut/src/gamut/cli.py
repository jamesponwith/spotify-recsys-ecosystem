"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from .config import ARTIFACTS, AuditConfig

app = typer.Typer(add_completion=False, help="Gamut -- catalog exposure audit.")
console = Console()


@app.command()
def collect(n_queries: int = 400, depth: int = 100) -> None:
    """Run Cadence over the held-out title-only battery and cache what it surfaced."""
    from .collect import collect as run_collect

    cfg = AuditConfig(n_queries=n_queries, depth=depth)
    console.print(f"wrote {run_collect(cfg).save()}")


@app.command()
def audit() -> None:
    """Per-channel exposure attribution, then the accuracy/exposure frontier."""
    from .audit import run

    report = run()
    b = report["baseline"]
    console.print(
        f"\n[bold]{b['track_coverage']:.2%}[/bold] of the catalog ever surfaced · "
        f"artist gini [bold]{b['artist_gini']:.3f}[/bold] · "
        f"long tail [bold]{b['tail_share']:.1%}[/bold] of recommendations"
    )


@app.command("report")
def report_cmd() -> None:
    """Print the stored audit report."""
    path = ARTIFACTS / "audit_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `gamut audit` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
