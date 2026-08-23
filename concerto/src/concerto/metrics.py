"""What to count, and why the obvious statistic is the wrong one.

"Percentage of tickets that went to real fans" is the number every platform
quotes and it is nearly useless on its own, for two reasons this module exists
to work around:

  - It is satisfied by a broker selling a ticket to a fan at four times face.
    The fan is a fan. The ticket reached them. The statistic is 100%.
  - It says nothing about *which* fans. A policy that admits only the richest
    third of the demand curve scores identically to one that admits a random
    third, and those are not the same product.

So the metrics here are in four groups: who got in, at what price, where the
money went, and who was shut out. An arm has to be read across all four -- the
result of this project is that no arm wins all of them, and that the trade-off
is the actual decision a promoter is making whether they know it or not.
"""

from __future__ import annotations

import numpy as np

from .config import Arm, Scenario
from .market import Outcome, broker_revenue
from .population import Population


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini. 0 = every unit identical, 1 = one unit holds everything."""
    mask = weights > 1e-12
    v, w = values[mask], weights[mask]
    if v.size == 0 or w.sum() <= 0:
        return 0.0
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    # Brown's formula over the Lorenz curve, and the leading zero is not
    # optional: the first trapezoid runs from the origin to the first point.
    # Dropping it made a perfectly equal distribution score 0.11 rather than 0,
    # which is small enough to look like a real result.
    cw = np.concatenate(([0.0], np.cumsum(w)))
    cvw = np.concatenate(([0.0], np.cumsum(v * w)))
    if cvw[-1] <= 0:
        return 0.0
    return float(1.0 - np.sum(np.diff(cw) * (cvw[:-1] + cvw[1:])) / (cvw[-1] * cw[-1]))


def summarise(outcome: Outcome, arm: Arm, scn: Scenario, pop: Population) -> dict:
    on_sale = float(scn.on_sale)
    held = outcome.fan_tickets
    total_held = float(held.sum())
    primary_to_fans = float((outcome.fan_primary + outcome.fan_cleared).sum())

    resale_volume = float(outcome.fan_resale.sum()) / on_sale

    # --- price -----------------------------------------------------------
    fan_outlay = float(outcome.fan_spend.sum())
    price_paid = np.divide(outcome.fan_spend, held, out=np.zeros_like(held), where=held > 1e-12)
    at_face = float(held[(price_paid <= scn.face_price * (1.0 + scn.primary_fee) * 1.02)].sum())

    # --- money -----------------------------------------------------------
    cleared_rev = (
        float(outcome.fan_cleared.sum()) * outcome.clearing_price
        if outcome.seats_cleared > 0
        else 0.0
    )
    face_seats_sold = primary_to_fans - float(outcome.fan_cleared.sum()) + outcome.broker_tickets
    artist_revenue = cleared_rev + face_seats_sold * scn.face_price
    broker_gross = broker_revenue(outcome, arm, scn)
    identity_cost = (
        scn.identity_cost_c0
        * arm.identity_cost_multiplier
        * outcome.identities**scn.identity_cost_gamma
    )
    broker_profit = (
        broker_gross
        - outcome.broker_tickets * scn.face_price * (1.0 + scn.primary_fee)
        - identity_cost
    )
    # Rent dissipation, and it is most of the damage. At equilibrium the sector's
    # revenue is exactly `gamma` times its identity spend, so at gamma = 1.55
    # brokers burn about 64% of what they gross on proxies, aged accounts and
    # payment instruments. That money does not go to a scalper either -- it
    # leaves the fan and reaches nobody who made the music. Reporting only
    # broker *profit* makes the queue look four times less wasteful than it is.
    identity_burn = identity_cost

    # --- who was shut out -------------------------------------------------
    # Superfans are the top decile of verified listening depth. This is the
    # group every arm claims to be protecting and the group the report shows
    # most arms do not protect at all.
    cut = float(np.quantile(pop.affinity, 0.90))
    superfan = pop.affinity >= cut
    want = pop.party.astype(np.float64)
    superfan_served = float(held[superfan].sum() / max(want[superfan].sum(), 1e-9))

    # The other end of the affinity distribution, and the group affinity
    # rationing costs. Under a lottery someone who found the artist last month
    # has a small chance; under an absolute merit cut they have none. Reporting
    # only `superfan_served` would show the benefit of merit rationing and hide
    # what it is taken from.
    casual = pop.affinity <= float(np.quantile(pop.affinity, 0.50))
    casual_served = float(held[casual].sum() / max(want[casual].sum(), 1e-9))

    # The same question asked of money rather than fandom: bottom income
    # quartile. A clearing price is invisible in every metric above and is
    # extremely visible in this one.
    poor = pop.income <= float(np.quantile(pop.income, 0.25))
    poor_served = float(held[poor].sum() / max(want[poor].sum(), 1e-9))

    return {
        "arm": arm.key,
        "label": arm.label,
        "converged": bool(outcome.converged),
        "iterations": int(outcome.iterations),
        # who got in
        "fan_fill": total_held / on_sale,
        "fan_fill_primary": primary_to_fans / on_sale,
        "broker_capture": outcome.broker_tickets / on_sale,
        "face_access": at_face / on_sale,
        # at what price
        "clearing_price": (
            float(outcome.clearing_price) if outcome.seats_cleared > 0 else float("nan")
        ),
        # Reported only when enough tickets change hands for the price to mean
        # something. A resale price is the clearing price of the broker's
        # inventory, so as that inventory goes to zero the price runs off to the
        # very top of the demand curve -- a real property of the model and a
        # meaningless statistic about half a dozen seats. Suppressing it below
        # 0.5% of the house keeps a restrictive arm from being credited with a
        # spectacular markup it does not actually charge anyone.
        "resale_volume": resale_volume,
        "resale_price": float(outcome.resale_price) if resale_volume >= 0.005 else float("nan"),
        "markup": (
            float(outcome.resale_price) / scn.face_price if resale_volume >= 0.005 else float("nan")
        ),
        "mean_price_paid": fan_outlay / max(total_held, 1e-9),
        "price_multiple": (fan_outlay / max(total_held, 1e-9)) / scn.face_price,
        # where the money went
        "fan_outlay": fan_outlay,
        "artist_revenue": artist_revenue,
        "artist_per_seat": artist_revenue / on_sale,
        "broker_profit": broker_profit,
        "identity_burn": identity_burn,
        "identity_burn_per_seat": identity_burn / on_sale,
        "broker_identities": float(outcome.identities),
        "unit_margin": float(outcome.unit_margin),
        "leak_to_brokers": broker_profit / max(fan_outlay, 1e-9),
        # who was shut out
        "superfan_served": superfan_served,
        "casual_served": casual_served,
        "low_income_served": poor_served,
        "admitted_affinity": float(np.average(pop.affinity, weights=held))
        if total_held > 0
        else 0.0,
        "admitted_income": float(np.average(pop.income, weights=held)) if total_held > 0 else 0.0,
        # Mean income of who got in, over mean income of everyone who wanted in.
        # 1.0 is an income-blind allocation. This is the metric `access_gini`
        # looks like it is and is not -- see the note on AGGREGATE_KEYS.
        "income_ratio": (
            float(np.average(pop.income, weights=held)) / float(pop.income.mean())
            if total_held > 0
            else 1.0
        ),
        "access_gini": gini(pop.income, held),
        # --- what the policy cost the people it was for ---------------------
        "party_truncated": float(outcome.party_truncated),
        "false_rejected": float(outcome.false_rejected),
        "gate_denied": scn.gate_id_failure if arm.transfer == "bound" else 0.0,
        "customer_harm": (
            float(outcome.party_truncated)
            + float(outcome.false_rejected)
            + (scn.gate_id_failure if arm.transfer == "bound" else 0.0)
        ),
    }


# `access_gini` is kept deliberately, and deliberately not featured. It is the
# obvious equity metric and it reads backwards: the most exclusive arm scores
# *best* on it, because an allocation that admits only rich people admits a set
# of people who are all equally rich, and dispersion within the admitted set is
# not the same question as who was admitted. It is in the report as a worked
# example of a metric that answers a question nobody asked -- the same shape of
# error as measuring exposure over a candidate pool that re-ranking cannot
# change. `income_ratio` is the metric it is usually mistaken for.
AGGREGATE_KEYS = (
    "fan_fill",
    "fan_fill_primary",
    "broker_capture",
    "face_access",
    "clearing_price",
    "resale_volume",
    "resale_price",
    "markup",
    "mean_price_paid",
    "price_multiple",
    "fan_outlay",
    "artist_revenue",
    "artist_per_seat",
    "broker_profit",
    "identity_burn",
    "identity_burn_per_seat",
    "broker_identities",
    "unit_margin",
    "leak_to_brokers",
    "superfan_served",
    "casual_served",
    "low_income_served",
    "admitted_affinity",
    "admitted_income",
    "income_ratio",
    "access_gini",
    "party_truncated",
    "false_rejected",
    "gate_denied",
    "customer_harm",
)


def aggregate(rows: list[dict]) -> dict:
    """Mean and standard deviation across trials, so a difference can be read."""
    out: dict = {
        "arm": rows[0]["arm"],
        "label": rows[0]["label"],
        "n_trials": len(rows),
        "converged": all(r["converged"] for r in rows),
    }
    out["iterations"] = int(np.max([r["iterations"] for r in rows]))
    for key in AGGREGATE_KEYS:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        # `clearing_price` is legitimately absent on every face-price arm, so an
        # all-NaN column is expected rather than a bug. Guarding here keeps the
        # distinction: NaN means "this arm has no such price", never "the run
        # produced nothing".
        out[key] = float(finite.mean()) if finite.size else float("nan")
        out[key + "_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return out
