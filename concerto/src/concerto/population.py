"""The people who want in, and the one modelling choice the argument rests on.

Willingness to pay is not drawn directly. It is *built*:

    wtp = face x income_factor x (0.5 + affinity)

because that is the claim the whole project is testing. A price rations on the
product of those two terms. A fan believes the ticket should ration on the
second one. Every result downstream is a consequence of the fact that those are
different orderings, and the gap between them is exactly the income term.

Drawing WTP directly from one distribution would have made the affinity arm
untestable by construction -- there would have been nothing for it to ration on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

from .config import Scenario


@dataclass(frozen=True)
class Population:
    """One realisation of the demand side of an on-sale."""

    wtp: np.ndarray  # per fan, in currency units
    affinity: np.ndarray  # per fan, in [0, 1]; verified listening depth
    income: np.ndarray  # per fan, multiplier on face
    party: np.ndarray  # per fan, tickets wanted (1..4)

    @cached_property
    def by_wtp(self) -> np.ndarray:
        """Indices in descending willingness to pay.

        Cached because the equilibrium solver re-prices the same population
        forty-odd times per arm and the ordering never changes. Re-sorting
        inside the bisection made the sensitivity grid roughly nine times
        slower for no difference in the answer.
        """
        return np.argsort(-self.wtp, kind="stable")

    @cached_property
    def by_affinity(self) -> np.ndarray:
        """Indices in descending verified listening depth."""
        return np.argsort(-self.affinity, kind="stable")

    @property
    def n_fans(self) -> int:
        return int(self.wtp.size)

    @property
    def demand_tickets(self) -> int:
        return int(self.party.sum())


def draw(scn: Scenario, rng: np.random.Generator) -> Population:
    """Draw a fan population whose total ticket demand hits the demand multiple."""
    mean_party = float(np.dot(np.arange(1, 5), scn.party_probs))
    target_tickets = scn.demand_multiple * scn.on_sale
    n_fans = max(1, int(round(target_tickets / mean_party)))

    party = rng.choice(np.arange(1, 5), size=n_fans, p=np.asarray(scn.party_probs))

    # Affinity: most people who want a ticket like the artist a normal amount,
    # a thin tail have listened for years. Beta(2, 5) is that shape.
    affinity = rng.beta(2.0, 5.0, size=n_fans)

    # Income factor: lognormal, median pinned to `income_median_factor` so the
    # median fan's WTP sits at a sane multiple of face rather than wherever the
    # exponential happens to land.
    income = scn.income_median_factor * np.exp(rng.normal(0.0, scn.income_sigma, size=n_fans))

    wtp = scn.face_price * income * (0.5 + affinity)
    return Population(wtp=wtp, affinity=affinity, income=income, party=party)


def eligible(pop: Population, false_reject: float, rng: np.random.Generator) -> np.ndarray:
    """Mask of fans who clear the identity check.

    A false reject is a real person turned away by an anti-bot measure. It is
    counted here rather than assumed to be zero because it is the direct
    customer cost of every identity-hardening policy, and no platform that ships
    one publishes it.
    """
    if false_reject <= 0.0:
        return np.ones(pop.n_fans, dtype=bool)
    return rng.random(pop.n_fans) >= false_reject


@lru_cache(maxsize=64)
def _cached(key: tuple, scn: Scenario, trial: int) -> Population:
    del key
    return draw(scn, np.random.default_rng(scn.seed + trial))


def draw_for(scn: Scenario, trial: int) -> Population:
    """The population for trial `t`, shared across every arm that asks for it.

    Two things at once. The pairing -- every arm is scored against the same
    crowd, so a difference between arms cannot be a difference between
    populations -- and the cache, keyed only on the parameters `draw` actually
    reads. The sensitivity grid varies broker and enforcement parameters that do
    not touch the demand side at all, so without this it redraws fifty thousand
    identical fans for every cell.
    """
    key = (
        scn.demand_multiple,
        scn.party_probs,
        scn.income_sigma,
        scn.income_median_factor,
        scn.capacity,
        scn.public_share,
        scn.face_price,
        scn.seed,
        trial,
    )
    return _cached(key, scn, trial)
