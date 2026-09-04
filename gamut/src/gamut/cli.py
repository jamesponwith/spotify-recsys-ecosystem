"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from .config import ARTIFACTS, RETRIEVE_DEPTH, AuditConfig

app = typer.Typer(add_completion=False, help="Gamut -- catalog exposure audit.")
console = Console()


# `depth` defaults to Cadence's real pool depth rather than a literal: this flag
# was a second hand-copy of the constant that drifted, so a fix in config.py
# alone would have left `gamut collect` still caching a 100-deep window.
@app.command()
def collect(n_queries: int = 400, depth: int = RETRIEVE_DEPTH) -> None:
    """Run Cadence over the held-out title-only battery and cache what it surfaced."""
    from .collect import collect as run_collect

    cfg = AuditConfig(n_queries=n_queries, depth=depth)
    console.print(f"wrote {run_collect(cfg).save()}")


# `--depth` mirrors `collect`'s: a shallower cache can still be audited, it just
# has to be *labelled* at the depth it was collected at. Without this the two
# commands could not agree, since `audit` would always demand the full pool.
@app.command()
def audit(depth: int = RETRIEVE_DEPTH) -> None:
    """Per-channel exposure attribution, then the accuracy/exposure frontier."""
    from .audit import run

    report = run(AuditConfig(depth=depth))
    b = report["baseline"]
    console.print(
        f"\n[bold]{b['track_coverage']:.2%}[/bold] of the catalog ever surfaced · "
        f"artist gini [bold]{b['artist_gini']:.3f}[/bold] · "
        f"long tail [bold]{b['tail_share']:.1%}[/bold] of recommendations"
    )


@app.command()
def demo(index: int = 0, penalty: float = 0.3, n: int = 10) -> None:
    """Show one query: what ships today vs what the exposure-aware re-rank shows."""
    from rich.table import Table

    from .demo import build

    out = build(index=index, penalty=penalty, n=n)
    console.print(f'\n[bold]"{out["query"]}"[/bold]  ({out["held_out"]} tracks withheld)')
    for label, key in (("as it ships today", "before"), (f"penalty {penalty}", "after")):
        side = out[key]
        table = Table(
            title=f"{label} — {side['tail_share']:.0%} long tail, "
            f"{side['distinct_artists']} artists, "
            f"median {side['median_playlists']:.0f} playlists"
        )
        table.add_column("#", justify="right")
        table.add_column("Track", max_width=34, overflow="ellipsis")
        table.add_column("Artist", max_width=20, overflow="ellipsis")
        table.add_column("Playlists", justify="right")
        for r in side["rows"]:
            mark = "[green]+[/green] " if r.get("new") else "  "
            tail = "[dim]tail[/dim]" if r["tail"] else ""
            table.add_row(
                str(r["rank"]), mark + r["name"], r["artist"], f"{r['playlists']:,} {tail}"
            )
        console.print(table)


@app.command("report")
def report_cmd() -> None:
    """Print the stored audit report."""
    path = ARTIFACTS / "audit_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `gamut audit` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
