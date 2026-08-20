"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from .config import ARTIFACTS, SimConfig

app = typer.Typer(add_completion=False, help="Ostinato -- feedback-loop simulation.")
console = Console()


@app.command()
def simulate(rounds: int = 5, queries: int = 150, penalty: float = 0.3, dose: int = 25) -> None:
    """Run all three arms and record how exposure moves each round."""
    from .simulate import run

    cfg = SimConfig(rounds=rounds, queries_per_round=queries, penalty=penalty, dose=dose)
    report = run(cfg)
    console.print(f"\nwrote {ARTIFACTS / 'sim_report.json'} in {report['seconds']}s")


@app.command("report")
def report_cmd() -> None:
    """Print the stored simulation report."""
    path = ARTIFACTS / "sim_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `ostinato simulate` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
