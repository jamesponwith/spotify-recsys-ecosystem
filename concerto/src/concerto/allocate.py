"""Rationing rules, and the demand curve they leave behind.

Allocation is computed as an *expectation* over each fan rather than by drawing
lottery winners. Two reasons, both about being able to trust the output:

  - The comparison between arms is then exact given a population, so the trial
    variance in the report is population variance and nothing else. Sampling the
    lottery as well would add noise that looks like a policy effect.
  - A fractional allocation is the honest object anyway. "This fan has a 12%
    chance of a pair" is what the lottery actually offers them.

Every metric downstream is therefore a weighted statistic, with each fan's weight
being the probability they hold a ticket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import Population


@dataclass(frozen=True)
class DemandCurve:
    """Unmet fan demand, ordered by willingness to pay."""

    wtp: np.ndarray  # descending
    cumulative: np.ndarray  # tickets available at or above each price

    # Index into the *original* fan array, in curve order. Kept so a caller can
    # write an allocation back to the fans it belongs to without re-sorting.
    index: np.ndarray

    @classmethod
    def build(
        cls, wtp: np.ndarray, tickets: np.ndarray, order: np.ndarray | None = None
    ) -> DemandCurve:
        """`order` is descending-WTP indices, precomputed once per population.

        Passing it in is not a micro-optimisation. The solver re-prices the same
        population on every bisection step, so sorting here put an O(n log n)
        inside the inner loop of every arm of every sensitivity cell.
        """
        if order is None:
            order = np.argsort(-wtp, kind="stable")
        t = tickets[order]
        keep = t > 1e-12
        idx = order[keep]
        return cls(wtp=wtp[idx], cumulative=np.cumsum(t[keep]), index=idx)

    def price_for(self, supply: float) -> float:
        """The price at which exactly `supply` tickets are taken.

        Zero when demand runs out first -- unsold inventory is a real outcome and
        the model must be able to express it, or a restrictive policy will look
        good by making tickets worthless in a way nobody notices.
        """
        if self.cumulative.size == 0:
            return 0.0
        if supply <= 0.0:
            # The limit as supply -> 0 is the top of the curve, not infinity.
            # Returning infinity here would hand a broker sector holding no
            # inventory an unbounded expected margin, and the solver would
            # diverge on the first iteration of every restrictive arm.
            return float(self.wtp[0])
        if supply > self.cumulative[-1] + 1e-9:
            # Strictly beyond total demand. At *exactly* total demand there is
            # still a marginal buyer -- the last one -- and collapsing the price
            # to zero there would report a sold-out house as worthless.
            return 0.0
        idx = int(np.searchsorted(self.cumulative, supply, side="left"))
        return float(self.wtp[min(idx, self.wtp.size - 1)])

    def take(self, supply: float) -> np.ndarray:
        """Tickets each rank buys when `supply` is on offer, top of the curve down."""
        if self.cumulative.size == 0 or supply <= 0.0:
            return np.zeros_like(self.cumulative)
        prev = np.concatenate(([0.0], self.cumulative[:-1]))
        return np.clip(supply - prev, 0.0, self.cumulative - prev)


def sorted_take(
    values: np.ndarray,
    tickets: np.ndarray,
    supply: float,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Serve requests in descending order of `values` until `supply` runs out.

    The shared implementation behind both deterministic rules: market clearing
    sorts on willingness to pay, affinity rationing sorts on listening depth.
    That they are the same operation on different keys is the whole argument --
    the mechanism is identical and only the ordering differs.
    """
    out = np.zeros_like(tickets, dtype=np.float64)
    if supply <= 0.0:
        return out
    if order is None:
        order = np.argsort(-values, kind="stable")
    wanted = tickets[order]
    cum = np.cumsum(wanted)
    prev = np.concatenate(([0.0], cum[:-1]))
    granted = np.clip(supply - prev, 0.0, wanted)
    out[order] = granted
    return out


def lottery_probability(seats: float, requests: float) -> float:
    """Chance one identity is admitted, when `requests` tickets are asked for.

    Admission is all-or-nothing *per identity* -- that is what a queue is: reach
    the front and you buy your whole allocation -- while the denominator is in
    tickets, so that expected seats sold equals the seats on offer. Both halves
    matter. Modelling admission per ticket instead would quietly erase the effect
    of a purchase cap, which only exists because identities buy in blocks.
    """
    if requests <= 0.0:
        return 0.0
    return float(min(1.0, seats / requests))


def party_truncation(pop: Population, cap: int | None) -> float:
    """Share of fan ticket demand a per-identity cap refuses to serve.

    The direct customer cost of a tight cap, and the reason the model does not
    simply set the cap to 1 and declare scalping solved. A cap of 2 turns away a
    third of a family.
    """
    if cap is None:
        return 0.0
    wanted = pop.party.astype(np.float64)
    served = np.minimum(wanted, cap)
    total = wanted.sum()
    return float((wanted - served).sum() / total) if total else 0.0
