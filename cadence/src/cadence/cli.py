"""Command-line interface.

cadence build            build the catalog from raw MPD slices
cadence splits           freeze the evaluation split
cadence train            fit the retrieval spaces
cadence train-reranker   fit the learned reranker
cadence evaluate         run the offline harness
cadence eval-ab          price a retrieval knob with a paired A/B
cadence play "..."       generate a playlist and print it
cadence serve            start the HTTP API
cadence pipeline         everything above, in order
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ARTIFACTS, DATA_PROCESSED, DATA_RAW

app = typer.Typer(add_completion=False, help="Cadence — natural-language playlist generation")
console = Console()


def _engine(provider: str | None = None):
    from .catalog import Catalog
    from .engine import CadenceEngine
    from .models.reranker import Reranker
    from .planner.base import get_planner

    catalog = Catalog.load()
    reranker = None
    path = ARTIFACTS / "reranker.pkl"
    if path.exists():
        reranker = Reranker.load(path)
    return CadenceEngine(catalog, planner=get_planner(provider), reranker=reranker)


@app.command()
def build(
    raw: Path = typer.Option(DATA_RAW, help="directory containing mpd.slice.*.json"),
    out: Path = typer.Option(DATA_PROCESSED),
    max_slices: int = typer.Option(0, help="0 = use every slice"),
    min_tag_playlists: int = typer.Option(5, help="min playlists a title token must appear on"),
):
    """Build the catalog, interaction matrix and folksonomy tag matrix."""
    from .config import BuildConfig
    from .data.build import build as build_catalog

    cfg = BuildConfig(
        max_slices=max_slices or None,
        min_tag_playlists=min_tag_playlists,
    )
    build_catalog(raw_dir=raw, out_dir=out, cfg=cfg)


@app.command()
def splits(n_eval: int = typer.Option(2000), processed: Path = typer.Option(DATA_PROCESSED)):
    """Freeze the held-out evaluation challenges."""
    from .eval.splits import make_splits

    meta = make_splits(processed_dir=processed, n_eval=n_eval)
    console.print_json(json.dumps({k: v for k, v in meta.items() if k != "holdout_rows"}))


@app.command()
def train(
    processed: Path = typer.Option(DATA_PROCESSED),
    out: Path = typer.Option(ARTIFACTS),
    holdout: bool = typer.Option(True, help="exclude evaluation playlists from training"),
):
    """Fit the collaborative, tag and lexical spaces."""
    import numpy as np

    from .eval.splits import load_splits
    from .models.train import train as train_spaces

    rows = np.zeros(0, dtype=np.int64)
    if holdout and (processed / "splits.json").exists():
        rows, _ = load_splits(processed)
    train_spaces(processed_dir=processed, out_dir=out, holdout_rows=rows)


@app.command("train-reranker")
def train_reranker_cmd(
    n_queries: int = typer.Option(1500),
    processed: Path = typer.Option(DATA_PROCESSED),
):
    """Fit the learned reranker on held-in playlists."""
    from .models.reranker import train_reranker

    engine = _engine("offline")
    engine.reranker = None
    train_reranker(engine, processed_dir=processed, n_queries=n_queries)


@app.command()
def evaluate(
    limit: int = typer.Option(400, help="challenges per cell; 0 = all"),
    ablations: bool = typer.Option(True),
    out: Path = typer.Option(ARTIFACTS / "eval_report.json"),
):
    """Run the offline evaluation harness."""
    from .eval.run_eval import run

    run(limit=limit or None, ablations=ablations, out_path=out)


@app.command("eval-ab")
def eval_ab_cmd(
    k: int = typer.Option(0, "--k", help="seed count to run both arms over"),
    limit: int = typer.Option(400, help="challenges in the cell; 0 = all"),
    arm: list[str] = typer.Option(
        [], "--arm", help="KEY=VALUE override for arm B; repeat for several"
    ),
    base: list[str] = typer.Option(
        [], "--base", help="KEY=VALUE override for arm A (default: the shipped config)"
    ),
    reranker: bool = typer.Option(False, help="score both arms through the learned reranker"),
    out: Path = typer.Option(ARTIFACTS / "eval_ab.json"),
):
    """Price one retrieval knob against another arm with a paired band.

    Both arms see the same challenges and the same planned intents, so the delta
    is paired and resolves differences the unpaired band in `evaluate` calls
    noise. Only rrf_k and the seven channel weights are settable — the assembly
    knobs live past this harness's last stage.
    """
    from .eval.eval_ab import parse_overrides, run

    try:
        arm_overrides = parse_overrides(arm)
        base_overrides = parse_overrides(base)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    run(
        k=k,
        limit=limit or None,
        arm=arm_overrides,
        base=base_overrides,
        use_reranker=reranker,
        out_path=out,
    )


@app.command("eval-affinity")
def eval_affinity_cmd():
    """Sweep audio_affinity_weight against tag adherence and mood error."""
    from .eval.affinity_sweep import main

    main()


@app.command("audit-lexicon")
def audit_lexicon_cmd(
    processed: Path = typer.Option(DATA_PROCESSED),
    out: Path = typer.Option(ARTIFACTS / "lexicon_calibration.json"),
):
    """Compare MOOD_LEXICON's audio targets with where humans file each word."""
    from .eval.lexicon_audit import main

    main(processed_dir=processed, out=out)


@app.command("eval-constraints")
def eval_constraints_cmd():
    """Measure how often the assembly stage honours stated requirements."""
    from .eval.constraints_eval import main

    main()


@app.command()
def play(
    query: str = typer.Argument(..., help="free-text playlist request"),
    n: int = typer.Option(0, help="number of tracks; 0 = let the request decide"),
    provider: str = typer.Option("offline", help="offline | anthropic"),
    show_reasons: bool = typer.Option(True),
    as_json: bool = typer.Option(False, "--json"),
):
    """Generate a playlist and print it."""
    engine = _engine(provider)
    playlist = engine.generate(query, n_tracks=n or None)

    if as_json:
        console.print_json(playlist.model_dump_json())
        return

    console.print(
        Panel(
            f"[bold]{playlist.title}[/bold]\n[dim]{playlist.description}[/dim]",
            title=f"“{query}”",
            border_style="green",
        )
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", width=3, justify="right")
    table.add_column("Track", max_width=38)
    table.add_column("Artist", max_width=22)
    table.add_column("BPM", width=5, justify="right")
    table.add_column("Nrg", width=4, justify="right")
    if show_reasons:
        table.add_column("Why", max_width=46)
    for t in playlist.tracks:
        bpm = f"{t.track.tempo:.0f}" if t.track.tempo else "—"
        nrg = f"{t.track.energy:.2f}" if t.track.energy is not None else "—"
        row = [str(t.position), t.track.name, t.track.artist, bpm, nrg]
        if show_reasons:
            row.append("; ".join(t.reasons[:2]))
        table.add_row(*row)
    console.print(table)

    s = playlist.stats
    console.print(
        f"[dim]{s.n_tracks} tracks · {s.total_duration_s / 60:.0f} min · "
        f"{s.n_artists} artists · long-tail {s.long_tail_share:.0%} · "
        f"{engine.arc_summary(playlist)}[/dim]"
    )
    if playlist.constraint_report:
        ok = all(playlist.constraint_report.values())
        colour = "green" if ok else "red"
        console.print(f"[{colour}]constraints: {playlist.constraint_report}[/{colour}]")
    console.print(f"[dim]latency: {playlist.timings_ms.get('total', 0):.0f} ms[/dim]")
    for w in playlist.warnings:
        if w:
            console.print(f"[yellow]note:[/yellow] {w}")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)):
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("cadence.service.api:app", host=host, port=port, log_level="info")


@app.command()
def pipeline(
    max_slices: int = typer.Option(0),
    n_eval: int = typer.Option(2000),
    n_queries: int = typer.Option(1500),
    eval_limit: int = typer.Option(400),
):
    """Run the whole offline pipeline end to end."""
    from .config import BuildConfig
    from .data.build import build as build_catalog
    from .eval.run_eval import run
    from .eval.splits import load_splits, make_splits
    from .models.reranker import train_reranker
    from .models.train import train as train_spaces

    console.rule("build")
    build_catalog(cfg=BuildConfig(max_slices=max_slices or None, min_tag_playlists=5))
    console.rule("splits")
    make_splits(n_eval=n_eval)
    console.rule("train")
    rows, _ = load_splits()
    train_spaces(holdout_rows=rows)
    console.rule("reranker")
    engine = _engine("offline")
    engine.reranker = None
    train_reranker(engine, processed_dir=DATA_PROCESSED, n_queries=n_queries)
    console.rule("evaluate")
    run(limit=eval_limit or None)


@app.command()
def info():
    """Print catalog and model provenance."""
    from .catalog import Catalog

    catalog = Catalog.load()
    console.print_json(json.dumps(catalog.meta, indent=2))


if __name__ == "__main__":
    app()
