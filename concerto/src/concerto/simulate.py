"""Run every arm, then try to break the result.

`compare` is the headline table. `sensitivity` is the part that decides whether
the headline is allowed to be a headline: the same comparison re-run across a
grid over the four assumptions that could plausibly overturn it, reporting not
the numbers but whether the *ordering* held. A simulation with invented
parameters can claim anything; a simulation that reports where its claim breaks
can be argued with.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np

from .config import ARM_BY_KEY, ARMS, ARTIFACTS, Arm, Scenario, SensitivityGrid
from .market import solve
from .metrics import aggregate, summarise
from .population import draw_for


def run_arm(arm: Arm, scn: Scenario, *, trials: int | None = None) -> dict:
    """One arm, over independent fan populations drawn from the same scenario."""
    n = trials if trials is not None else scn.n_trials
    rows = []
    for t in range(n):
        # Every arm sees the *same* population on trial t. The comparison is
        # paired, exactly as in ostinato: without this, a difference between two
        # arms could be a difference between two crowds.
        pop = draw_for(scn, t)
        rows.append(summarise(solve(arm, scn, pop), arm, scn, pop))
    out = aggregate(rows)
    out["note"] = arm.note
    return out


def compare(scn: Scenario, arms: tuple[Arm, ...] = ARMS, *, trials: int | None = None) -> dict:
    results = [run_arm(a, scn, trials=trials) for a in arms]
    return {
        "scenario": {
            "capacity": scn.capacity,
            "on_sale": scn.on_sale,
            "face_price": scn.face_price,
            "demand_multiple": scn.demand_multiple,
            "primary_fee": scn.primary_fee,
            "resale_fee": scn.resale_fee,
            "off_platform_leak": scn.off_platform_leak,
            "identity_cost_gamma": scn.identity_cost_gamma,
            "n_trials": trials if trials is not None else scn.n_trials,
            "seed": scn.seed,
        },
        "arms": results,
    }


def _rank(results: list[dict], key: str, *, higher_is_better: bool = True) -> list[str]:
    order = sorted(results, key=lambda r: r[key], reverse=higher_is_better)
    return [r["arm"] for r in order]


def sensitivity(scn: Scenario, grid: SensitivityGrid) -> dict:
    """Re-run the comparison across the assumption grid.

    Reports, for each cell, the ordering of arms on the two metrics the
    conclusion actually rests on -- how much of the house a broker takes, and
    whether the people who care most get in. A claim survives only if its
    ordering is the same in every cell.
    """
    arms = tuple(ARM_BY_KEY[k] for k in grid.arms)
    cells = []
    combos = list(
        itertools.product(
            grid.demand_multiple,
            grid.identity_cost_gamma,
            grid.off_platform_leak,
            grid.affinity_forge_multiplier,
        )
    )
    for demand, gamma, leak, forge in combos:
        cell_scn = replace(
            scn,
            demand_multiple=demand,
            identity_cost_gamma=gamma,
            off_platform_leak=leak,
            affinity_forge_multiplier=forge,
            n_trials=grid.trials,
        )
        # The affinity arms carry the forging cost, so the swept parameter has
        # to reach them -- sweeping a constant that nothing reads is the classic
        # way to produce a robustness section that proves nothing.
        cell_arms = tuple(
            replace(a, identity_cost_multiplier=forge) if a.rule == "affinity" else a for a in arms
        )
        rows = [run_arm(a, cell_scn, trials=grid.trials) for a in cell_arms]
        by_arm = {r["arm"]: r for r in rows}
        cells.append(
            {
                "demand_multiple": demand,
                "identity_cost_gamma": gamma,
                "off_platform_leak": leak,
                "affinity_forge_multiplier": forge,
                "capture_rank": _rank(rows, "broker_capture", higher_is_better=False),
                "superfan_rank": _rank(rows, "superfan_served"),
                "metrics": {
                    k: {
                        "broker_capture": v["broker_capture"],
                        "superfan_served": v["superfan_served"],
                        "low_income_served": v["low_income_served"],
                        "artist_per_seat": v["artist_per_seat"],
                        "price_multiple": v["price_multiple"],
                    }
                    for k, v in by_arm.items()
                },
            }
        )

    claims = _check_claims(cells, grid)
    return {"grid": grid.arms, "n_cells": len(cells), "claims": claims, "cells": cells}


def _check_claims(cells: list[dict], grid: SensitivityGrid) -> list[dict]:
    """The specific statements the report wants to make, tested cell by cell.

    Each claim is written down *before* the grid runs and is allowed to fail.
    A claim that holds in 42 of 180 cells is reported as holding in 42 of 180.
    """
    checks = [
        (
            "verified_beats_queue_on_capture",
            "Verified fan takes less of the house than an open queue",
            lambda m: m["verified"]["broker_capture"] < m["queue"]["broker_capture"],
        ),
        (
            "margin_beats_identity",
            "Capping resale beats hardening identity, on broker capture",
            lambda m: m["capped"]["broker_capture"] < m["verified"]["broker_capture"],
        ),
        (
            "clearing_kills_capture",
            "Market clearing leaves brokers less of the house than any face-price arm",
            lambda m: (
                m["clearing"]["broker_capture"]
                <= min(m[k]["broker_capture"] for k in ("queue", "verified", "capped", "bound"))
            ),
        ),
        (
            "clearing_costs_the_poor",
            "Market clearing serves the bottom income quartile worse than the capped arm",
            lambda m: m["clearing"]["low_income_served"] < m["capped"]["low_income_served"],
        ),
        (
            "affinity_serves_superfans",
            "Affinity rationing serves superfans better than every price-rationed arm",
            lambda m: (
                m["affinity"]["superfan_served"]
                > max(m[k]["superfan_served"] for k in ("queue", "verified", "capped", "clearing"))
            ),
        ),
        (
            "affinity_needs_a_closed_resale_channel",
            "...and it only holds once resale is closed as well",
            lambda m: (
                m["affinity_bound"]["superfan_served"]
                > max(m[k]["superfan_served"] for k in ("queue", "verified", "capped", "clearing"))
            ),
        ),
        (
            "no_arm_is_cheap",
            "No arm gets the median fan in at close to face once demand is hot",
            lambda m: min(v["price_multiple"] for v in m.values()) > 1.2,
        ),
    ]
    out = []
    for key, statement, fn in checks:
        held = [bool(fn(c["metrics"])) for c in cells]
        failures = [
            {
                "demand_multiple": c["demand_multiple"],
                "identity_cost_gamma": c["identity_cost_gamma"],
                "off_platform_leak": c["off_platform_leak"],
                "affinity_forge_multiplier": c["affinity_forge_multiplier"],
            }
            for c, ok in zip(cells, held, strict=True)
            if not ok
        ]
        out.append(
            {
                "key": key,
                "statement": statement,
                "held": int(sum(held)),
                "of": len(held),
                "survives": all(held),
                "region": _failure_region(failures),
                "failures": failures[:12],
            }
        )
    del grid
    return out


def _failure_region(failures: list[dict]) -> dict | None:
    """Where a claim breaks, as a box rather than a list of cells.

    A list of ten failing cells is not a finding. "Every failure has demand
    >= 15x and leakage >= 0.65" is, because it names the condition under which
    the recommendation would be wrong -- which is the only part of a robustness
    check anyone should act on.
    """
    if not failures:
        return None
    keys = (
        "demand_multiple",
        "identity_cost_gamma",
        "off_platform_leak",
        "affinity_forge_multiplier",
    )
    return {k: [min(f[k] for f in failures), max(f[k] for f in failures)] for k in keys}


def calibrate(scn: Scenario, *, target_markup: float = 3.0, trials: int = 6) -> dict:
    """Fit the identity-cost constant, and publish the sweep it was fitted on.

    The whole model has exactly one free parameter, and hiding it inside a
    default would make every number downstream unfalsifiable. So it is fitted
    here, in the open, by bisection on a quantity anyone can check against a
    resale listing -- and the sweep around the fit is written out with it, so a
    reader can see how much of the conclusion moves if they disagree with the
    anchor.
    """
    arm = ARM_BY_KEY["queue"]

    def markup_at(c0: float) -> dict:
        return run_arm(arm, replace(scn, identity_cost_c0=c0), trials=trials)

    lo, hi = 0.01, 5.0
    for _ in range(30):
        mid = float(np.sqrt(lo * hi))
        if markup_at(mid)["markup"] < target_markup:
            lo = mid
        else:
            hi = mid
    fitted = float(np.sqrt(lo * hi))

    sweep = []
    for c0 in (0.05, 0.1, fitted, 0.3, 0.5, 1.0, 2.0):
        r = markup_at(c0)
        sweep.append(
            {
                "c0": c0,
                "fitted": abs(c0 - fitted) < 1e-9,
                "markup": r["markup"],
                "broker_capture": r["broker_capture"],
                "broker_identities": r["broker_identities"],
                "price_multiple": r["price_multiple"],
            }
        )
    sweep.sort(key=lambda r: r["c0"])
    return {
        "target_markup": target_markup,
        "fitted_c0": fitted,
        "shipped_c0": scn.identity_cost_c0,
        "anchor": "Resale markup on the unrestricted arm, the only quantity in "
        "the model that is publicly observable. Broker capture is an output.",
        "sweep": sweep,
    }


def save(payload: dict, name: str) -> str:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return str(path)
