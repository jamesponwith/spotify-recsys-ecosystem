import numpy as np

from concerto.broker import best_response, margin, resale_realisation
from concerto.config import ARM_BY_KEY, Scenario


def test_no_entry_when_the_trade_does_not_pay():
    """Deterrence is invisible in a capture statistic: it shows up as brokers
    who are not there. The model has to be able to produce zero."""
    scn = Scenario()
    assert best_response(0.5, -1.0, ARM_BY_KEY["queue"], scn) == 0.0
    assert best_response(0.0, 100.0, ARM_BY_KEY["queue"], scn) == 0.0


def test_dearer_identities_buy_fewer_identities():
    scn = Scenario()
    cheap = best_response(0.1, 100.0, ARM_BY_KEY["lottery"], scn)
    dear = best_response(0.1, 100.0, ARM_BY_KEY["verified"], scn)
    assert 0.0 < dear < cheap


def test_identity_hardening_is_mostly_absorbed_but_not_futile():
    """The first hypothesis this model killed, kept as the test that killed it.

    The guess was that making identities twelve times dearer would reduce broker
    capture by *less* than twelve times -- that competition among the survivors
    would eat the gain. It does not. The response elasticity is 1/(gamma-1) =
    1.8, so a 12x cost rise cuts the unconstrained best response ~92x, and the
    win-rate offset competes roughly four fifths of that back. Four fifths of
    92x is still 20x.

    So the test pins both halves: the offset is large and real, and the policy
    still works. What sinks identity hardening is not futility, it is that
    capping the margin beats it outright -- see the sensitivity grid.
    """
    scn = Scenario()
    lottery, verified = ARM_BY_KEY["lottery"], ARM_BY_KEY["verified"]
    seats, fan_tickets = 4232.0, 34000.0

    def settle(arm):
        # solve m = BR(seats / (fan_tickets + m*cap), mu) by bisection
        lo, hi = 0.0, best_response(1.0, 120.0, arm, scn)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            q = min(1.0, seats / (fan_tickets + mid * arm.cap))
            if best_response(q, 120.0, arm, scn) >= mid:
                lo = mid
            else:
                hi = mid
        m = 0.5 * (lo + hi)
        return m * arm.cap * min(1.0, seats / (fan_tickets + m * arm.cap))

    tickets_cheap, tickets_dear = settle(lottery), settle(verified)
    cost_ratio = verified.identity_cost_multiplier / lottery.identity_cost_multiplier
    capture_ratio = tickets_cheap / tickets_dear

    # Without any offset, identities alone would fall by cost_ratio ** (1/(gamma-1)).
    raw = cost_ratio ** (1.0 / (scn.identity_cost_gamma - 1.0))
    assert capture_ratio < 0.35 * raw, "the win-rate offset should absorb most of it"
    assert capture_ratio > cost_ratio, "...but not enough to make the policy useless"


def test_capping_resale_is_worth_less_than_the_cap_suggests():
    """A face-value cap does not deliver face value: whatever share leaks
    off-platform still sells at market, minus the buyer's risk premium."""
    scn = Scenario()
    market = 400.0
    capped = resale_realisation(market, ARM_BY_KEY["capped"], scn)
    ceiling = scn.face_price * (1.0 + scn.exchange_cap_uplift)
    assert ceiling < capped < market


def test_binding_the_ticket_bounds_the_upside():
    scn = Scenario()
    bound = resale_realisation(1000.0, ARM_BY_KEY["bound"], scn)
    assert bound < resale_realisation(1000.0, ARM_BY_KEY["capped"], scn)
    # Downside is bounded too -- unsold inventory comes back at face.
    assert resale_realisation(0.0, ARM_BY_KEY["bound"], scn) > 0.0


def test_margin_is_negative_for_a_bound_ticket_at_any_plausible_price():
    """Why the bound arm shows zero brokers rather than a few. If this ever
    flips, the headline changes and the test should be the thing that says so."""
    scn = Scenario()
    for p in np.linspace(0.0, 800.0, 40):
        assert margin(float(p), scn.face_price, ARM_BY_KEY["bound"], scn) < 0.0
