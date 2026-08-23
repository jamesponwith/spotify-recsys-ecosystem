import numpy as np

from concerto.config import ARM_BY_KEY, Scenario
from concerto.market import solve
from concerto.metrics import summarise
from concerto.population import draw_for


def run(key, scn=None, trial=0):
    scn = scn or Scenario()
    pop = draw_for(scn, trial)
    arm = ARM_BY_KEY[key]
    return summarise(solve(arm, scn, pop), arm, scn, pop)


def test_the_house_is_never_oversold():
    scn = Scenario()
    for key in ARM_BY_KEY:
        pop = draw_for(scn, 0)
        out = solve(ARM_BY_KEY[key], scn, pop)
        sold = float(out.fan_primary.sum() + out.fan_cleared.sum()) + out.broker_tickets
        assert sold <= scn.on_sale + 1e-6, key


def test_every_arm_reaches_a_fixed_point():
    """Bisection replaced a damped iteration that failed to settle in roughly
    one sensitivity cell in eight. Non-convergence used to be silent."""
    scn = Scenario()
    for key in ARM_BY_KEY:
        pop = draw_for(scn, 0)
        assert solve(ARM_BY_KEY[key], scn, pop).converged, key


def test_the_equilibrium_belief_is_self_consistent():
    """The resale price a broker expected has to be the price they actually get.

    This is the property that separates a solved model from one that assumed its
    own answer, and it is worth testing directly rather than trusting the
    solver: re-derive the clearing price from the allocation the solver
    produced, and check it is the price the solver fed back in.
    """
    from concerto.allocate import DemandCurve

    scn = Scenario()
    pop = draw_for(scn, 0)
    for key in ("queue", "lottery", "verified", "capped"):
        arm = ARM_BY_KEY[key]
        out = solve(arm, scn, pop)
        weight = 1.0 - arm.false_reject
        request = np.minimum(pop.party.astype(float), arm.cap) * weight
        unmet = request - out.fan_cleared - out.fan_primary
        realised = DemandCurve.build(pop.wtp, unmet, pop.by_wtp).price_for(out.broker_tickets)
        assert abs(realised - out.resale_price) / max(out.resale_price, 1.0) < 0.01, key


def test_fan_fill_is_useless_on_its_own():
    """Every arm scores ~100% on 'tickets that reached a real fan', including
    the one where a broker sold every seat at three times face. It is the number
    the industry quotes and it is the reason this repo does not lead with it."""
    fills = {k: run(k)["fan_fill"] for k in ("queue", "capped", "bound", "affinity_bound")}
    assert min(fills.values()) > 0.99
    # ...and yet what fans paid differs by a factor of nearly two.
    assert run("queue")["price_multiple"] / run("bound")["price_multiple"] > 1.7


def test_binding_transfer_removes_the_broker_entirely():
    assert run("queue")["broker_capture"] > 0.4
    assert run("bound")["broker_capture"] < 0.01


def test_market_clearing_shuts_out_the_bottom_income_quartile():
    """The counterweight to the clearing arm's clean capture number. It does not
    stop scalping so much as do it in-house."""
    clearing = run("clearing")
    assert clearing["broker_capture"] < 0.01
    assert clearing["low_income_served"] < 0.02
    assert run("bound")["low_income_served"] > 5 * clearing["low_income_served"]


def test_clearing_costs_the_average_fan_more_than_the_queue_it_replaces():
    """A result worth being suspicious of, so it is pinned. Pricing at market
    charges everyone the market price; the queue at least let a lucky third in
    at face. The surplus moves to the artist -- it does not come back to fans."""
    assert run("clearing")["price_multiple"] > run("queue")["price_multiple"]
    assert run("clearing")["artist_per_seat"] > 2 * run("queue")["artist_per_seat"]


def test_affinity_rationing_serves_superfans_and_shuts_out_casuals():
    """Both halves. The second one is the cost and it is not usually stated."""
    aff, queue = run("affinity_bound"), run("queue")
    assert aff["superfan_served"] > 4 * queue["superfan_served"]
    assert aff["income_ratio"] < 1.1  # income-blind
    # The casual fan's lottery ticket is gone: they had a small chance, now none.
    scn = Scenario()
    pop = draw_for(scn, 0)
    casual = pop.affinity <= np.quantile(pop.affinity, 0.30)
    got_queue = solve(ARM_BY_KEY["queue"], scn, pop).fan_tickets[casual].sum()
    got_aff = solve(ARM_BY_KEY["affinity_bound"], scn, pop).fan_tickets[casual].sum()
    assert got_queue > 0.0
    assert got_aff < 0.01 * got_queue


def test_a_hotter_show_is_worse_for_everyone_but_the_broker():
    from dataclasses import replace

    scn = Scenario()
    cool = run("queue", replace(scn, demand_multiple=3.0))
    hot = run("queue", replace(scn, demand_multiple=30.0))
    assert hot["markup"] > cool["markup"]
    assert hot["price_multiple"] > cool["price_multiple"]
