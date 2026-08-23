"""One on-sale, solved to equilibrium.

The order of operations is the model:

  1. A share of the house is sold at a clearing price to whoever will pay it.
  2. Brokers decide how many identities to buy, against an expected resale price.
  3. What is left of the house is rationed by the arm's rule.
  4. Whatever brokers took is resold into the demand the primary market missed,
     which sets the resale price -- which was step 2's input.

Steps 2 and 4 are circular, so they are solved to a fixed point rather than
assumed. That is the difference between measuring a policy and advertising one.
Under fixed broker demand every restriction works, because the brokers cannot
respond. Under rational-expectations entry they do respond -- and how much of
each restriction survives that response turns out to differ by an order of
magnitude between restrictions that raise the *cost* of capture and ones that
remove the *reward*. That difference is the result; see `broker.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocate import DemandCurve, lottery_probability, party_truncation, sorted_take
from .broker import best_response, margin, resale_realisation
from .config import Arm, Scenario
from .population import Population


@dataclass
class Outcome:
    """Everything one solved on-sale produced, before it is turned into metrics."""

    arm: str
    converged: bool
    iterations: int

    seats_cleared: float
    seats_face: float
    clearing_price: float
    resale_price: float
    unit_margin: float

    identities: float
    broker_tickets: float

    fan_cleared: np.ndarray = field(repr=False)
    fan_primary: np.ndarray = field(repr=False)
    fan_resale: np.ndarray = field(repr=False)
    fan_spend: np.ndarray = field(repr=False)

    party_truncated: float = 0.0
    false_rejected: float = 0.0

    @property
    def fan_tickets(self) -> np.ndarray:
        return self.fan_cleared + self.fan_primary + self.fan_resale


def _resale_prices(p_market: float, arm: Arm, scn: Scenario) -> list[tuple[float, float]]:
    """How broker inventory reaches fans: (share of inventory, price the fan pays).

    Deliberately expressed from the *buyer's* side. A capped exchange is usually
    reported as a restriction on sellers; what it is is a set of prices fans pay,
    and the split between them is the policy.
    """
    if arm.transfer == "open":
        return [(1.0, p_market)]
    if arm.transfer == "capped":
        cap_price = scn.face_price * (1.0 + scn.exchange_cap_uplift)
        leak = scn.off_platform_leak
        return [
            (1.0 - leak, min(p_market, cap_price)),
            (leak, p_market * (1.0 - scn.off_platform_trust_discount)),
        ]
    if arm.transfer == "bound":
        leak = scn.off_platform_leak * scn.bound_leak_factor
        return [
            # Returned to the exchange and re-sold to the queue at face. The seat
            # still reaches a fan -- it just reaches them at the original price.
            (1.0 - leak, scn.face_price),
            (leak, p_market * (1.0 - scn.bound_trust_discount)),
        ]
    raise ValueError(f"unknown transfer regime {arm.transfer!r}")


def _broker_identities(
    unit_margin: float,
    arm: Arm,
    scn: Scenario,
    merit_seats: float,
    reserve_seats: float,
    fan_tickets: float,
) -> tuple[float, float]:
    """Identities the sector buys, and the share of its requests that win.

    A broker's optimal identity count depends on its win probability, which
    depends on how many identities the sector bought. Rather than iterate that
    with damping -- which oscillates badly wherever identities are expensive --
    it is solved directly: the fixed point is the root of a monotone scalar
    function, so bisection finds it exactly and always.
    """
    seats = merit_seats + reserve_seats
    if unit_margin <= 0.0 or seats <= 0.0:
        return 0.0, 0.0
    cap = float(arm.cap) if arm.cap is not None else 8.0

    def win(m: float) -> float:
        asked = m * cap
        if asked <= 0.0:
            return 0.0
        if arm.rule != "affinity":
            return min(1.0, seats / max(fan_tickets + asked, 1e-12))
        # Forged histories clear the cut, so merit seats are a sure thing and
        # brokers take them first. Only the capacity they cannot place there
        # goes into the open draw, where it competes with the fans the merit
        # round already turned away.
        merit_won = min(asked, merit_seats)
        left_over = max(asked - merit_seats, 0.0)
        fans_left = max(fan_tickets - (merit_seats - merit_won), 0.0)
        q_res = (
            min(1.0, reserve_seats / max(fans_left + left_over, 1e-12))
            if reserve_seats > 0.0
            else 0.0
        )
        return (merit_won + left_over * q_res) / asked

    hi = best_response(1.0, unit_margin, arm, scn)
    if hi <= 0.0:
        return 0.0, 0.0
    if best_response(win(hi), unit_margin, arm, scn) >= hi:
        return hi, win(hi)

    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if best_response(win(mid), unit_margin, arm, scn) >= mid:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    return m, win(m)


def solve(arm: Arm, scn: Scenario, pop: Population) -> Outcome:
    """Solve one on-sale under one policy."""
    on_sale = float(scn.on_sale)
    cap = arm.cap
    weight = 1.0 - arm.false_reject
    request = np.minimum(pop.party.astype(np.float64), cap if cap is not None else np.inf) * weight

    # --- 1. the cleared slice -------------------------------------------
    seats_cleared = float(np.round(arm.clearing_share * on_sale))
    seats_face = on_sale - seats_cleared
    curve_all = DemandCurve.build(pop.wtp, request, pop.by_wtp)
    clearing_price = curve_all.price_for(seats_cleared) if seats_cleared > 0 else float("nan")
    fan_cleared = (
        sorted_take(pop.wtp, request, seats_cleared, pop.by_wtp)
        if seats_cleared > 0
        else np.zeros_like(request)
    )
    remaining = request - fan_cleared
    remaining_total = float(remaining.sum())
    cap_f = float(cap) if cap is not None else 8.0

    # --- 1b. the reserved lottery block ----------------------------------
    # Under a merit rule the cut is absolute: a fan one place below it has no
    # chance, not a small one. Holding some of the face-priced seats back for an
    # open draw is the only thing that gives them one. It is a no-op under a
    # rule that is already a lottery.
    reserve_seats = (
        float(np.round(arm.reserve_share * seats_face)) if arm.rule == "affinity" else 0.0
    )
    merit_seats = seats_face - reserve_seats

    def state(p_resale: float) -> tuple[float, float, np.ndarray, float]:
        """Everything the market does, given a belief about the resale price."""
        unit_margin = margin(p_resale, scn.face_price, arm, scn)
        identities, _ = _broker_identities(
            unit_margin, arm, scn, merit_seats, reserve_seats, remaining_total
        )
        if arm.rule == "affinity":
            # Merit seats first -- a forged history clears the cut, so those are
            # a sure thing -- then whatever identity capacity is left over goes
            # into the open draw alongside everyone else.
            asked = identities * cap_f
            broker_merit = min(asked, merit_seats)
            fan_merit = sorted_take(
                pop.affinity, remaining, merit_seats - broker_merit, pop.by_affinity
            )
            left_over = max(asked - merit_seats, 0.0)
            after_merit = remaining - fan_merit
            q_res = lottery_probability(reserve_seats, float(after_merit.sum()) + left_over)
            fan_primary = fan_merit + after_merit * q_res
            broker_tickets = broker_merit + left_over * q_res
        else:
            q = lottery_probability(seats_face, remaining_total + identities * cap_f)
            fan_primary = remaining * q
            broker_tickets = identities * cap_f * q
        return unit_margin, identities, fan_primary, min(broker_tickets, seats_face)

    def residual(p_resale: float) -> float:
        """Belief minus outcome. Decreasing in the belief, so it has one root."""
        _, _, fan_primary, broker_tickets = state(p_resale)
        cleared = DemandCurve.build(pop.wtp, remaining - fan_primary, pop.by_wtp).price_for(
            broker_tickets
        )
        return cleared - p_resale

    # --- 2-4. the fixed point, by bisection ------------------------------
    # At a belief of zero the trade never pays, so no broker enters, so the
    # first resold ticket would fetch the top of the curve: the residual is
    # positive. At the top of the curve the trade pays enormously, the sector
    # floods in, and the residual is negative. The root between them is the
    # rational-expectations equilibrium.
    lo, hi = 0.0, float(pop.wtp.max())
    iters = 0
    if residual(hi) > 0.0:
        # Demand outruns anything a broker can supply -- the market never clears
        # below the top of the curve. Rare, but it is a real corner.
        p_resale = hi
    else:
        while iters < scn.max_iterations and (hi - lo) / max(hi, 1.0) >= scn.tolerance:
            iters += 1
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        p_resale = 0.5 * (lo + hi)

    converged = True
    unit_margin, identities, fan_primary, broker_tickets = state(p_resale)

    # --- the resale, from the buyer's side ------------------------------
    unmet = remaining - fan_primary
    resale_curve = DemandCurve.build(pop.wtp, unmet, pop.by_wtp)

    fan_resale = np.zeros_like(remaining)
    fan_spend = np.zeros_like(remaining)
    taken = resale_curve.take(broker_tickets)
    if taken.size:
        blended = sum(share * price for share, price in _resale_prices(p_resale, arm, scn))
        fan_resale[resale_curve.index] = taken
        fan_spend[resale_curve.index] += taken * blended

    fee = 1.0 + scn.primary_fee
    fan_spend += fan_primary * scn.face_price * fee
    if seats_cleared > 0:
        fan_spend += fan_cleared * clearing_price * fee

    return Outcome(
        arm=arm.key,
        converged=converged,
        iterations=iters,
        seats_cleared=seats_cleared,
        seats_face=seats_face,
        clearing_price=clearing_price,
        resale_price=p_resale,
        unit_margin=unit_margin,
        identities=identities,
        broker_tickets=broker_tickets,
        fan_cleared=fan_cleared,
        fan_primary=fan_primary,
        fan_resale=fan_resale,
        fan_spend=fan_spend,
        party_truncated=party_truncation(pop, cap),
        false_rejected=arm.false_reject,
    )


def broker_revenue(outcome: Outcome, arm: Arm, scn: Scenario) -> float:
    """What the broker sector grossed, net of marketplace fees but before identities."""
    gross = resale_realisation(outcome.resale_price, arm, scn) * (1.0 - scn.resale_fee)
    return outcome.broker_tickets * gross
