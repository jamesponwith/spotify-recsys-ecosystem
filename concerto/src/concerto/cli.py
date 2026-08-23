"""Typer entry points."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .config import ARM_BY_KEY, ARTIFACTS, Scenario, SensitivityGrid

app = typer.Typer(add_completion=False, help="Concerto -- what actually stops a scalper.")
console = Console()


def _fmt(value: float, spec: str) -> str:
    return "--" if value != value else format(value, spec)


@app.command()
def simulate(trials: int = 24, demand: float = 8.0) -> None:
    """Every allocation arm, solved to equilibrium, over paired populations."""
    from dataclasses import replace

    from .simulate import compare, save

    scn = replace(Scenario(), n_trials=trials, demand_multiple=demand)
    report = compare(scn)
    save(report, "simulation.json")

    table = Table(
        title=f"{scn.on_sale:,} seats · {scn.demand_multiple:g}x demand · ${scn.face_price:.0f} face"
    )
    table.add_column("Arm")
    for col in (
        "Broker",
        "At face",
        "Fan pays",
        "Artist/seat",
        "Superfans",
        "Low income",
        "Harm",
    ):
        table.add_column(col, justify="right")
    for r in report["arms"]:
        table.add_row(
            r["label"],
            f"{r['broker_capture']:.1%}",
            f"{r['face_access']:.1%}",
            f"{r['price_multiple']:.2f}x",
            f"${r['artist_per_seat']:.0f}",
            f"{r['superfan_served']:.1%}",
            f"{r['low_income_served']:.1%}",
            f"{r['customer_harm']:.1%}",
        )
    console.print(table)
    console.print(
        "[dim]Fan pays = mean outlay per ticket including fees, over face. "
        "Harm = parties split by the cap + fans wrongly rejected + turned away at the gate.[/dim]"
    )


@app.command()
def sensitivity(trials: int = 8) -> None:
    """Re-run every arm across the assumption grid and test the stated claims."""
    from .simulate import save
    from .simulate import sensitivity as run

    grid = SensitivityGrid(trials=trials)
    report = run(Scenario(), grid)
    save(report, "sensitivity.json")

    table = Table(title=f"Claims tested across {report['n_cells']} parameter cells")
    table.add_column("Claim")
    table.add_column("Holds", justify="right")
    table.add_column("", justify="left")
    for c in report["claims"]:
        mark = "[green]survives[/green]" if c["survives"] else "[yellow]breaks[/yellow]"
        table.add_row(c["statement"], f"{c['held']}/{c['of']}", mark)
    console.print(table)


@app.command()
def calibrate(target_markup: float = 3.0, trials: int = 6) -> None:
    """Fit the one free parameter, and show the sweep it was fitted on."""
    from .simulate import calibrate as run
    from .simulate import save

    report = run(Scenario(), target_markup=target_markup, trials=trials)
    save(report, "calibration.json")
    table = Table(title=f"Identity-cost constant fitted to a {target_markup:g}x resale markup")
    table.add_column("c0", justify="right")
    table.add_column("Markup", justify="right")
    table.add_column("Broker capture", justify="right")
    table.add_column("Identities", justify="right")
    for row in report["sweep"]:
        table.add_row(
            f"{row['c0']:.4f}",
            f"{row['markup']:.2f}x",
            f"{row['broker_capture']:.1%}",
            f"{row['broker_identities']:,.0f}",
        )
    console.print(table)
    console.print(f"\nfitted [bold]c0 = {report['fitted_c0']:.4f}[/bold] (shipped default)")


@app.command()
def ledger(seats: int = 0, buyers: int = 40_000) -> None:
    """Cardano enforcement ladder, and the eUTxO contention a real drop hits."""
    from .ledger import contention_curve, leak_ladder
    from .simulate import save

    scn = Scenario()
    report = {
        "leak_ladder": leak_ladder(scn),
        "contention": contention_curve(seats or scn.on_sale, buyers),
    }
    save(report, "ledger.json")

    table = Table(title="What each rung of on-chain enforcement leaves the broker")
    table.add_column("Enforcement")
    table.add_column("Spread kept", justify="right")
    table.add_column("Broker capture", justify="right")
    table.add_column("Fan pays", justify="right")
    for r in report["leak_ladder"]["rungs"]:
        table.add_row(
            r["label"],
            f"{r['spread_retained']:.1%}",
            f"{r['broker_capture']:.1%}",
            f"{r['price_multiple']:.2f}x",
        )
    console.print(table)

    table = Table(
        title=f"eUTxO contention: {seats or scn.on_sale:,} seats, {buyers:,} concurrent buyers"
    )
    table.add_column("Inventory UTxOs", justify="right")
    table.add_column("Settled/block", justify="right")
    table.add_column("Tx success", justify="right")
    table.add_column("Attempts per buy", justify="right")
    table.add_column("Minutes", justify="right")
    for r in report["contention"]["rows"]:
        table.add_row(
            f"{r['shards']:,}",
            f"{r['settled_per_block']:,.0f}",
            f"{r['success_rate']:.2%}",
            f"{r['attempts_per_purchase']:,.0f}",
            f"{r['minutes_to_clear']:.1f}",
        )
    console.print(table)


@app.command()
def demo(left: str = "queue", right: str = "affinity_bound", trial: int = 0) -> None:
    """Follow twelve people across two policies."""
    from .demo import build

    for key in (left, right):
        if key not in ARM_BY_KEY:
            raise typer.BadParameter(f"unknown arm {key!r}; try {', '.join(ARM_BY_KEY)}")
    out = build(left, right, trial=trial)
    console.print(
        f"\n[bold]{out['on_sale']:,} seats[/bold] at ${out['face_price']:.0f} face, "
        f"{out['demand_multiple']:g}x oversubscribed\n"
    )
    table = Table()
    table.add_column("Who", max_width=22)
    table.add_column("Affinity", justify="right")
    table.add_column("Income", justify="right")
    table.add_column("Wants", justify="right")
    table.add_column(out["left"]["label"], justify="right", max_width=18)
    table.add_column(out["right"]["label"], justify="right", max_width=18)
    for r in out["rows"]:

        def cell(side: dict, wanted: int) -> str:
            if side["tickets"] < 0.01:
                return "[dim]shut out[/dim]"
            # Share of what they asked for, then the price. Showing the raw
            # expected ticket count made a 5% chance of a pair read as cheap
            # rather than as almost certainly nothing.
            return f"{side['tickets'] / wanted:.0%} of {wanted} @ {_fmt(side['multiple'], '.2f')}x"

        table.add_row(
            r["profile"],
            f"{r['affinity_pct']:.0%}",
            f"{r['income_x']:.2f}x",
            str(r["wants"]),
            cell(r[out["left"]["key"]], r["wants"]),
            cell(r[out["right"]["key"]], r["wants"]),
        )
    console.print(table)
    console.print(
        "[dim]A lottery offers a probability, not a seat: '10% of 2' is a one-in-ten "
        "chance of a pair. '@ 2.4x' is what they pay per ticket, over face, all-in.[/dim]"
    )
    for side in ("left", "right"):
        console.print(f"\n[bold]{out[side]['label']}[/bold] — {out[side]['note']}")


@app.command("report")
def report_cmd() -> None:
    """Print the stored simulation report."""
    path = ARTIFACTS / "simulation.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found -- run `concerto simulate` first.")
    console.print_json(json.dumps(json.loads(path.read_text())))


if __name__ == "__main__":
    app()
