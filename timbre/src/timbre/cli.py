"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .config import ARTIFACTS, Phase0Config

app = typer.Typer(add_completion=False, help="Timbre -- content-based cold start for Cadence.")
console = Console()


@app.command()
def phase0(
    test_fraction: float = 0.2,
    recall_k: int = 100,
    mlp_max_iter: int = typer.Option(
        60, help="Cap on MLP epochs. The dominant cost of this command by far."
    ),
    skip_mlp: bool = typer.Option(
        False, "--skip-mlp", help="Ridge only. Reaches a Gate 0 verdict in ~5 minutes."
    ),
) -> None:
    """Run the Phase 0 falsification test and rule on Gate 0."""
    from .phase0.run import run

    cfg = Phase0Config(
        test_fraction=test_fraction,
        recall_k=recall_k,
        mlp_max_iter=mlp_max_iter,
        skip_mlp=skip_mlp,
    )
    report = run(cfg)
    g = report["gate_0"]
    console.print()
    console.print(
        f"[bold]{'PASS' if g['passed'] else 'FAIL'}[/bold]  "
        f"oracle recovery ratio {g['oracle_recovery_ratio']:.1%}"
    )


@app.command()
def demo(
    query: str = typer.Argument(..., help="Natural-language playlist request."),
    n_cold: int = typer.Option(20, help="How many of Cadence's own picks to freeze out."),
    top_n: int = typer.Option(200, help="Candidate depth counted as 'recovered'."),
) -> None:
    """Freeze out Cadence's picks, then hand them back an audio-only embedding."""
    from .demo import run_demo

    result = run_demo(query, n_cold=n_cold, top_n=top_n)
    by = result.by_name

    table = Table(title=f'"{result.query}"  --  {result.n_cold} tracks frozen out')
    table.add_column("Track", overflow="ellipsis", max_width=38)
    table.add_column("Artist", overflow="ellipsis", max_width=22)
    table.add_column("warm", justify="right")
    table.add_column("cold", justify="right")
    table.add_column("timbre", justify="right")
    for t in result.cold_tracks:

        def fmt(r: int) -> str:
            return "[dim]--[/dim]" if r == 0 else str(r)

        table.add_row(
            t["name"], t["artist"], str(t["warm_rank"]), fmt(t["cold_rank"]), fmt(t["timbre_rank"])
        )
    console.print(table)
    console.print(
        f"in top {result.top_n}:  warm {by['warm'].in_top_n}/{result.n_cold}   "
        f"cold {by['cold'].in_top_n}/{result.n_cold}   "
        f"[bold]timbre {by['timbre'].in_top_n}/{result.n_cold}[/bold]"
    )


@app.command()
def regate() -> None:
    """Re-apply the Gate 0 rule to the stored measurements.

    The rule is a pure function of the recalls, so re-deriving it costs
    milliseconds and cannot disagree with what a fresh run would produce.
    Useful when the thresholds change, or after a fix to the rule itself.
    """
    import json

    from .phase0.gate import rule

    path = ARTIFACTS / "phase0_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `timbre phase0` first.")
    report = json.loads(path.read_text())
    report["gate_0"] = rule(report["retrieval"], Phase0Config())
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    g = report["gate_0"]
    console.print(
        f"[bold]{'PASS' if g['passed'] else 'FAIL'}[/bold]  "
        f"oracle recovery ratio {g['oracle_recovery_ratio']:.1%}"
    )


@app.command("demo-report")
def demo_report(
    n_cold: int = typer.Option(20),
    top_n: int = typer.Option(200),
) -> None:
    """Run the joint demo across a query battery and persist the aggregate."""
    from .report import build

    r = build(n_cold=n_cold, top_n=top_n)
    a = r["aggregate"]
    console.print()
    console.print(
        f"[bold]{a['recovered_with_timbre']}/{a['total_frozen']}[/bold] frozen tracks recovered "
        f"with Timbre ({a['recovery_rate_with']:.0%}), "
        f"vs {a['recovered_without_timbre']} ({a['recovery_rate_without']:.0%}) without."
    )


@app.command("report")
def report_cmd() -> None:
    """Print the stored Phase 0 report."""
    path = ARTIFACTS / "phase0_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `timbre phase0` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
