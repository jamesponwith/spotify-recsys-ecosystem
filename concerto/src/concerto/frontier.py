"""Three knobs the documents asserted a value for before anything measured one.

Writing the policy document first turned out to be a useful mistake. It
recommended holding back "20% of the house" as an open lottery, shipping "15%"
of the seats at a clearing price, and it called adding tour dates the strongest
intervention available while leaving it out of the arm set entirely. All three
were assertions with a number attached and nothing behind them, in a project
whose whole discipline is not doing that.

This module measures them. Where the measurement disagreed with the assertion,
the documents were changed rather than the sweep.
"""

from __future__ import annotations

from dataclasses import replace

from .config import ARM_BY_KEY, Arm, Scenario
from .market import solve
from .metrics import aggregate, summarise
from .population import draw_for

RESERVE_SHARES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
CLEARING_SHARES = (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0)
NIGHTS = (1, 2, 3, 4, 6, 8)
CAPS = (1, 2, 3, 4, 6, 8)


def _run(arm: Arm, scn: Scenario, trials: int) -> dict:
    rows = []
    for t in range(trials):
        pop = draw_for(scn, t)
        rows.append(summarise(solve(arm, scn, pop), arm, scn, pop))
    return aggregate(rows)


def reserve_curve(scn: Scenario, *, arm_key: str = "affinity_bound", trials: int = 24) -> dict:
    """What holding seats back for an open draw buys, and what it costs.

    Affinity rationing serves 96% of superfan demand and *zero* casual demand --
    a merit cut is absolute, so a fan one place below it has no chance at all
    rather than a small one. The reserve is the only thing that gives them one,
    and it is taken directly out of the superfans' share.
    """
    base = ARM_BY_KEY[arm_key]
    rows = []
    for share in RESERVE_SHARES:
        r = _run(replace(base, reserve_share=share), scn, trials)
        rows.append(
            {
                "reserve_share": share,
                "superfan_served": r["superfan_served"],
                "casual_served": r["casual_served"],
                "low_income_served": r["low_income_served"],
                "broker_capture": r["broker_capture"],
                "price_multiple": r["price_multiple"],
                "income_ratio": r["income_ratio"],
            }
        )
    return {"arm": arm_key, "rows": rows, "knee": _knee(rows)}


def _knee(rows: list[dict]) -> dict:
    """Where the trade stops being cheap.

    Defined as the largest reserve at which each point of casual access still
    costs less than one point of superfan access -- i.e. the last share where
    the exchange rate is better than one-for-one. Beyond it the reserve is
    buying casual access at a premium, which may still be the right call but is
    no longer free.
    """
    best = rows[0]
    for prev, cur in zip(rows, rows[1:], strict=False):
        gained = cur["casual_served"] - prev["casual_served"]
        lost = prev["superfan_served"] - cur["superfan_served"]
        if gained <= 0 or lost > gained:
            break
        best = cur
    return {
        "reserve_share": best["reserve_share"],
        "superfan_served": best["superfan_served"],
        "casual_served": best["casual_served"],
    }


def clearing_curve(scn: Scenario, *, arm_key: str = "affinity_bound", trials: int = 24) -> dict:
    """The artist's revenue against the fan's ability to pay for it.

    Every point of the house moved to a clearing price is money the artist keeps
    instead of a broker -- and a fan priced out. This is the trade the `hybrid`
    arm makes at 15%, a number that was picked before it was measured.
    """
    base = ARM_BY_KEY[arm_key]
    rows = []
    for share in CLEARING_SHARES:
        r = _run(replace(base, clearing_share=share), scn, trials)
        rows.append(
            {
                "clearing_share": share,
                "artist_per_seat": r["artist_per_seat"],
                "price_multiple": r["price_multiple"],
                "low_income_served": r["low_income_served"],
                "superfan_served": r["superfan_served"],
                "face_access": r["face_access"],
                "income_ratio": r["income_ratio"],
            }
        )
    base_poor = rows[0]["low_income_served"]
    for r in rows:
        # Share of the low-income access the face-price arm delivered that this
        # much clearing still leaves standing.
        r["poor_access_kept"] = r["low_income_served"] / max(base_poor, 1e-9)
    return {"arm": arm_key, "rows": rows}


def nights_curve(scn: Scenario, *, arm_key: str = "queue", trials: int = 24) -> dict:
    """The intervention that is not a policy: play the city more than once.

    Total demand is held fixed and the house is played `n` times, so the
    oversubscription multiple falls as 1/n. Nothing about the allocation rule
    changes. This is the comparison the arm set cannot make, and the reason the
    documents name it before they name any mechanism.

    The pairing here is unusually clean, and by construction rather than luck.
    Capacity is multiplied by `n` and the demand multiple divided by it, so the
    total ticket demand is identical and the population `draw` sizes itself off
    that total -- which means every row is scored against *literally the same
    14,849 people*, with the same party sizes and the same willingness to pay.
    The only thing that differs between one night and eight is how many seats
    there are. `tests/test_frontier.py` asserts it, because if a future change
    to `draw` broke it the rows would still look plausible.
    """
    base = ARM_BY_KEY[arm_key]
    rows = []
    for n in NIGHTS:
        cell = replace(
            scn,
            capacity=scn.capacity * n,
            demand_multiple=scn.demand_multiple / n,
        )
        r = _run(base, cell, trials)
        rows.append(
            {
                "nights": n,
                "demand_multiple": cell.demand_multiple,
                "seats": cell.on_sale,
                "markup": r["markup"],
                "price_multiple": r["price_multiple"],
                "broker_capture": r["broker_capture"],
                "face_access": r["face_access"],
                "superfan_served": r["superfan_served"],
                "low_income_served": r["low_income_served"],
                # Per seat is flat at face; what the artist gains is volume.
                "artist_total": r["artist_per_seat"] * cell.on_sale,
                "identity_burn_total": r["identity_burn"],
            }
        )
    one = rows[0]
    for r in rows:
        r["artist_vs_one_night"] = r["artist_total"] / max(one["artist_total"], 1e-9)
    return {"arm": arm_key, "rows": rows}


def cap_curve(
    scn: Scenario, *, arm_keys: tuple[str, ...] = ("queue", "capped"), trials: int = 24
) -> dict:
    """What a per-identity purchase cap costs a family, against what it saves.

    The policy document asserted that caps below four do more damage to families
    than to brokers, on the reasoning that a broker's binding constraint is what
    an identity costs rather than how many seats one identity may buy.

    That is half right, and the half it gets wrong is the half that matters. The
    trade depends entirely on whether the resale margin has already been closed,
    so this sweep runs against two arms rather than one -- an unprotected queue
    and the capped exchange -- because a single number here would have confirmed
    the assertion on the arm that happened to be measured.
    """
    out: dict = {"arms": {}}
    for arm_key in arm_keys:
        base = ARM_BY_KEY[arm_key]
        rows = []
        for cap in CAPS:
            r = _run(replace(base, cap=cap), scn, trials)
            rows.append(
                {
                    "cap": cap,
                    "party_truncated": r["party_truncated"],
                    "broker_capture": r["broker_capture"],
                    "broker_identities": r["broker_identities"],
                    "price_multiple": r["price_multiple"],
                    "face_access": r["face_access"],
                    "superfan_served": r["superfan_served"],
                    "low_income_served": r["low_income_served"],
                    "customer_harm": r["customer_harm"],
                }
            )
        loose = next(r for r in rows if r["cap"] == 8)
        tight = next(r for r in rows if r["cap"] == 2)
        cost = tight["party_truncated"] - loose["party_truncated"]
        saved = loose["broker_capture"] - tight["broker_capture"]
        out["arms"][arm_key] = {
            "rows": rows,
            # Points of family demand truncated per point of broker capture
            # prevented, going from a cap of 8 to a cap of 2. Below 1.0 the cap
            # is a good trade; above it, an expensive one.
            "cost_per_point_saved": cost / max(saved, 1e-9),
            "party_cost": cost,
            "capture_saved": saved,
        }
    return out


def build(scn: Scenario, *, trials: int = 24) -> dict:
    """Default trials match `concerto simulate`, deliberately.

    The one-night row of `nights_curve` *is* the queue arm of the main table, so
    at a matching trial count it reproduces that number exactly rather than
    differing from it by sampling noise. A reader who spots 2.33x in one table
    and 2.34x in another has found a discrepancy, and will be right to wonder
    which one the conclusion rests on.
    """
    return {
        "trials": trials,
        "reserve": reserve_curve(scn, trials=trials),
        "cap": cap_curve(scn, trials=trials),
        "clearing": clearing_curve(scn, trials=trials),
        "nights": nights_curve(scn, trials=trials),
    }


__all__ = [
    "build",
    "cap_curve",
    "clearing_curve",
    "nights_curve",
    "reserve_curve",
]
