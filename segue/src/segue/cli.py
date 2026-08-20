"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .config import ARTIFACTS, SegueConfig

app = typer.Typer(add_completion=False, help="Segue -- sequence-aware playlist continuation.")
console = Console()


@app.command()
def build() -> None:
    """Rebuild true playlist order from the raw MPD slices."""
    from .sequences import build as build_sequences

    path = build_sequences().save()
    console.print(f"wrote {path}")


@app.command()
def train(
    window: int = 5,
    horizon: int = 10,
    max_positions: int = 400_000,
) -> None:
    """Fit the transition operator on training playlists.

    `--horizon 1` recovers the naive next-track objective, which is the
    ablation that motivated the wider default.
    """
    from .demo import Bundle
    from .evaluate import holdout_rows
    from .model import train as fit

    cfg = SegueConfig(window=window, horizon=horizon, max_train_positions=max_positions)
    bundle = Bundle.load()
    model = fit(bundle.sequences, bundle.embeddings, holdout_rows(cfg), cfg)
    path = model.save()
    console.print(
        f"horizon={model.horizon}  alpha={model.alpha:,.0f}  "
        f"valid cosine={model.valid_cosine:.4f}  {model.seconds}s  ->  {path}"
    )


@app.command()
def evaluate() -> None:
    """Head-to-head against popularity / last-track / centroid, on ordered challenges."""
    from .demo import Bundle
    from .evaluate import evaluate as run_eval
    from .model import SegueModel

    cfg = SegueConfig()
    bundle = Bundle.load()
    report = run_eval(
        bundle.sequences,
        SegueModel.load(),
        bundle.embeddings,
        bundle.popularity,
        bundle.artists,
        cfg,
    )
    console.print(f"\nwrote {ARTIFACTS / 'eval_report.json'} in {report['seconds']}s")


@app.command()
def demo(k: int = 5, n: int = 10, index: int = 0) -> None:
    """Continue one held-out playlist and show every system's next-n."""
    from .demo import build_demo

    out = build_demo(indices=(index,), k=k, n=n)
    case = out["cases"][0]
    console.print(f'\n[bold]"{case["title"]}"[/bold]  ({case["seed_count"]} seeds shown)')
    for s in case["seeds"]:
        console.print(f"  · {s['label']}")

    table = Table(title=f"next {n}")
    table.add_column("#", justify="right")
    table.add_column("actually played next", max_width=40, overflow="ellipsis")
    for name in ("centroid", "segue"):
        table.add_column(name, max_width=40, overflow="ellipsis")
    for i in range(n):
        row = [str(i + 1)]
        actual = case["actual_next"]
        row.append(actual[i]["label"] if i < len(actual) else "")
        for name in ("centroid", "segue"):
            p = case["predictions"][name]
            if i < len(p):
                mark = "[green]✓[/green] " if p[i]["hit"] else "  "
                row.append(mark + p[i]["label"])
            else:
                row.append("")
        table.add_row(*row)
    console.print(table)


@app.command("report")
def report_cmd() -> None:
    """Print the stored evaluation report."""
    path = ARTIFACTS / "eval_report.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `segue evaluate` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
