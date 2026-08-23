"""One drop, two policies, and the same twelve people under both.

Aggregate tables hide the thing this project is actually about. "Broker capture
fell from 61% to 0%" does not tell you that the person it fell in favour of was
a hedge fund analyst who has played the artist twice, while the person who has
played them four thousand times still did not get in. So the demo picks real
rows out of one simulated population and follows them across arms.

The twelve are chosen to span the two axes that the whole model turns on --
how much money, how much affinity -- rather than sampled at random, because a
random twelve out of thirty thousand are all median and show nothing.
"""

from __future__ import annotations

import numpy as np

from .config import ARM_BY_KEY, Scenario
from .market import Outcome, solve
from .population import Population, draw

# (label, affinity percentile, income percentile). Twelve distinct points rather
# than three draws around four centres: nearest-neighbour picks cluster, and
# three near-identical rows spend a table on one fact.
PROFILES = (
    ("lifer, broke", 0.99, 0.08),
    ("lifer, comfortable", 0.98, 0.55),
    ("lifer, wealthy", 0.97, 0.96),
    ("deep fan, broke", 0.85, 0.12),
    ("deep fan, middling", 0.84, 0.50),
    ("deep fan, wealthy", 0.83, 0.93),
    ("knows the hits, broke", 0.50, 0.10),
    ("knows the hits, middling", 0.49, 0.52),
    ("knows the hits, wealthy", 0.48, 0.95),
    ("heard one song, broke", 0.12, 0.15),
    ("heard one song, middling", 0.11, 0.54),
    ("heard one song, wealthy", 0.10, 0.97),
)


def _pick(pop: Population) -> list[tuple[str, int]]:
    """The fan closest to each profile's target percentiles."""
    a_rank = np.argsort(np.argsort(pop.affinity)) / max(pop.n_fans - 1, 1)
    i_rank = np.argsort(np.argsort(pop.income)) / max(pop.n_fans - 1, 1)
    picked: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, a_target, i_target in PROFILES:
        d = (a_rank - a_target) ** 2 + (i_rank - i_target) ** 2
        for idx in np.argsort(d):
            if int(idx) not in used:
                picked.append((label, int(idx)))
                used.add(int(idx))
                break
    return picked


def _fan_view(out: Outcome, scn: Scenario, i: int) -> dict:
    got = float(out.fan_tickets[i])
    spend = float(out.fan_spend[i])
    return {
        "tickets": got,
        "primary": float(out.fan_primary[i] + out.fan_cleared[i]),
        "resale": float(out.fan_resale[i]),
        "price": spend / got if got > 1e-9 else float("nan"),
        "multiple": (spend / got) / scn.face_price if got > 1e-9 else float("nan"),
    }


def build(
    left: str = "queue",
    right: str = "affinity_bound",
    *,
    scn: Scenario | None = None,
    trial: int = 0,
) -> dict:
    scn = scn or Scenario()
    pop = draw(scn, np.random.default_rng(scn.seed + trial))
    arms = {k: ARM_BY_KEY[k] for k in (left, right)}
    outs = {k: solve(a, scn, pop) for k, a in arms.items()}

    rows = []
    for label, i in _pick(pop):
        rows.append(
            {
                "profile": label,
                "affinity_pct": float((pop.affinity < pop.affinity[i]).mean()),
                "income_x": float(pop.income[i]),
                "wants": int(pop.party[i]),
                "wtp": float(pop.wtp[i]),
                left: _fan_view(outs[left], scn, i),
                right: _fan_view(outs[right], scn, i),
            }
        )
    return {
        "left": {"key": left, "label": arms[left].label, "note": arms[left].note},
        "right": {"key": right, "label": arms[right].label, "note": arms[right].note},
        "face_price": scn.face_price,
        "on_sale": scn.on_sale,
        "demand_multiple": scn.demand_multiple,
        "rows": rows,
        "totals": {
            k: {
                "broker_capture": outs[k].broker_tickets / scn.on_sale,
                "resale_price": outs[k].resale_price,
                "identities": outs[k].identities,
            }
            for k in (left, right)
        },
    }
