"""Broker economics -- the one modelling decision that changes every result.

The tempting way to model a scalper is as fixed demand: brokers want N tickets,
a policy makes buying harder, they get fewer. Every anti-bot measure works under
that model, which is why every anti-bot measure has a press release.

Brokers are not fixed demand. They are an *investment*: an identity costs money,
and a broker buys identities up to the point where the last one stops paying for
itself. Make identities twelve times dearer and a broker buys fewer of them --
but the ones they do buy now face less competition, so each wins more often, and
the sector re-enters against a reward the policy never touched.

How much comes back was the first thing this model was built to answer, and the
first guess was wrong. Twelve-times-dearer identities cut the sector's
*unconstrained* best response by 92x, because the response elasticity is
1/(gamma-1) = 1.8, not 1. The win-rate offset then competes about four fifths of
that back -- but four fifths of 92x still leaves a 20x reduction. Identity
hardening is not the theatre it was expected to be. It is simply much weaker
than its own arithmetic implies, and it loses to attacking the margin instead
(`tests/test_broker.py` pins both halves).

So brokerage here is endogenous. Identities are chosen by first-order condition
against an expected margin, that choice moves the resale price, the resale price
moves the margin, and `market.solve` finds the fixed point of the two by
bisection.

    profit(m) = m x cap x q x mu  -  c0 x k_policy x m^gamma
    dprofit/dm = 0  =>  m* = ( cap x q x mu / (gamma x c0 x k_policy) )^(1/(gamma-1))

`gamma > 1` -- rising marginal cost of a usable identity -- is the only
structural assumption. It is what makes the sector finite. `c0` is the model's
single calibration constant, fitted by bisection so the unrestricted arm
reproduces a 3.0x resale markup: the one quantity here that is publicly
observable, since resale listings are visible and broker inventory is not.
Broker capture is an output of that fit, never a target of it. See
`simulate.calibrate`, which regenerates both the fit and the sweep around it.
"""

from __future__ import annotations

import numpy as np

from .config import Arm, Scenario


def resale_realisation(p_market: float, arm: Arm, scn: Scenario) -> float:
    """Gross per-ticket revenue a broker expects, before marketplace fees.

    This is where a transfer restriction actually bites. It does not stop a sale;
    it lowers what the sale is worth, in three different ways depending on the
    regime.
    """
    if arm.transfer == "open":
        return p_market

    if arm.transfer == "capped":
        # Legal route: an official exchange that will not list above face + uplift.
        # Illegal route: anywhere else, at market -- discounted by what a buyer
        # charges for taking counterparty risk on a stranger's ticket.
        legal = min(p_market, scn.face_price * (1.0 + scn.exchange_cap_uplift))
        leak = scn.off_platform_leak
        return (1.0 - leak) * legal + leak * p_market * (1.0 - scn.off_platform_trust_discount)

    if arm.transfer == "bound":
        # No transfer exists. Unwanted inventory goes back at face, so the
        # broker's downside is bounded -- and so is the upside, which is the
        # entire point. What leaks is not the ticket but the *account*, and an
        # identity check at the gate is what decides how much of that survives.
        leak = scn.off_platform_leak * scn.bound_leak_factor
        off = p_market * (1.0 - scn.bound_trust_discount)
        return (1.0 - leak) * scn.face_price + leak * off

    raise ValueError(f"unknown transfer regime {arm.transfer!r}")


def margin(p_market: float, primary_price: float, arm: Arm, scn: Scenario) -> float:
    """Expected net profit per ticket a broker manages to buy.

    Negative margin is the desired state and is reported as such: it means the
    trade does not pay, so the identities are never bought and the tickets never
    leave the primary market. Deterrence is invisible in a capture statistic --
    it shows up as brokers who are not there.
    """
    gross = resale_realisation(p_market, arm, scn) * (1.0 - scn.resale_fee)
    cost = primary_price * (1.0 + scn.primary_fee)
    return gross - cost


def best_response(
    win_prob: float,
    unit_margin: float,
    arm: Arm,
    scn: Scenario,
    *,
    max_identities: float = 5e6,
) -> float:
    """Identities the broker sector buys, given what it expects to make.

    Returns a continuum, not an integer: this is an aggregate over many brokers,
    and rounding it would put a spurious step function underneath every result.
    """
    if unit_margin <= 0.0 or win_prob <= 0.0:
        return 0.0
    cap = float(arm.cap) if arm.cap is not None else 8.0
    k = scn.identity_cost_c0 * arm.identity_cost_multiplier
    gamma = scn.identity_cost_gamma
    revenue_per_identity = cap * win_prob * unit_margin
    m = (revenue_per_identity / (gamma * k)) ** (1.0 / (gamma - 1.0))
    return float(np.clip(m, 0.0, max_identities))
