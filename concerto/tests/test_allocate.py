import numpy as np

from concerto.allocate import DemandCurve, lottery_probability, party_truncation, sorted_take
from concerto.config import Scenario
from concerto.population import draw_for


def curve(wtp, tickets):
    return DemandCurve.build(np.asarray(wtp, float), np.asarray(tickets, float))


def test_price_for_zero_supply_is_the_top_of_the_curve_not_infinity():
    """Infinity here made every restrictive arm diverge on the first iteration.

    A broker sector holding no inventory faces the price the *next* ticket would
    fetch, which is the top of the demand curve. Returning infinity handed it an
    unbounded expected margin and the solver never recovered.
    """
    assert curve([100.0, 50.0], [1.0, 1.0]).price_for(0.0) == 100.0


def test_price_for_supply_beyond_demand_is_zero():
    """Unsold inventory is a real outcome. A model that cannot express it will
    score a policy well for making tickets worthless."""
    assert curve([100.0, 50.0], [1.0, 1.0]).price_for(5.0) == 0.0


def test_price_for_is_the_marginal_buyer():
    c = curve([100.0, 80.0, 60.0], [1.0, 1.0, 1.0])
    assert c.price_for(1.0) == 100.0
    assert c.price_for(2.0) == 80.0
    assert c.price_for(3.0) == 60.0


def test_take_never_hands_out_more_than_is_on_offer():
    c = curve([100.0, 80.0, 60.0], [2.0, 2.0, 2.0])
    assert abs(c.take(3.0).sum() - 3.0) < 1e-12
    assert abs(c.take(99.0).sum() - 6.0) < 1e-12


def test_take_serves_the_top_of_the_curve_first():
    c = curve([10.0, 90.0, 50.0], [1.0, 1.0, 1.0])
    taken = c.take(1.0)
    # index 1 is the 90 -- the highest willingness to pay, wherever it sat
    assert c.index[0] == 1 and taken[0] == 1.0


def test_precomputed_order_gives_the_same_curve():
    """The cached ordering is a speed fix, and a speed fix that changes an
    answer is a bug. This is the guard."""
    scn = Scenario()
    pop = draw_for(scn, 0)
    tickets = pop.party.astype(float)
    a = DemandCurve.build(pop.wtp, tickets)
    b = DemandCurve.build(pop.wtp, tickets, pop.by_wtp)
    assert np.allclose(a.cumulative, b.cumulative)
    assert abs(a.price_for(1000.0) - b.price_for(1000.0)) < 1e-12


def test_sorted_take_matches_precomputed_order():
    scn = Scenario()
    pop = draw_for(scn, 0)
    tickets = pop.party.astype(float)
    a = sorted_take(pop.affinity, tickets, 500.0)
    b = sorted_take(pop.affinity, tickets, 500.0, pop.by_affinity)
    assert np.allclose(a, b)


def test_lottery_probability_is_capped_at_one():
    assert lottery_probability(100.0, 10.0) == 1.0
    assert lottery_probability(10.0, 100.0) == 0.1
    assert lottery_probability(10.0, 0.0) == 0.0


def test_a_tight_cap_splits_parties():
    """The reason the model does not set the cap to 1 and declare victory."""
    scn = Scenario()
    pop = draw_for(scn, 0)
    assert party_truncation(pop, None) == 0.0
    assert party_truncation(pop, 4) == 0.0
    assert party_truncation(pop, 2) > 0.15
    assert party_truncation(pop, 1) > party_truncation(pop, 2)
