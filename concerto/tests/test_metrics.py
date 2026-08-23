import numpy as np

from concerto.config import ARM_BY_KEY, Scenario
from concerto.market import solve
from concerto.metrics import aggregate, gini, summarise
from concerto.population import draw_for


def test_gini_is_zero_when_every_holder_is_identical():
    v = np.full(10, 2.0)
    assert gini(v, np.ones(10)) < 1e-9


def test_gini_approaches_one_when_one_holder_takes_everything():
    v = np.concatenate((np.zeros(99), [1000.0]))
    assert gini(v, np.ones(100)) > 0.95


def test_gini_ignores_people_with_no_ticket():
    """Weights are ticket counts, so the metric is over holders, not applicants."""
    v = np.array([1.0, 1.0, 1.0, 100.0])
    assert gini(v, np.array([1.0, 1.0, 1.0, 0.0])) < 1e-9


def test_access_gini_reads_backwards_and_is_kept_anyway():
    """The trap this repo collects, in its ticketing form.

    `access_gini` is the obvious equity metric and it scores the *most*
    exclusive policy best: an allocation that admits only rich people admits a
    set of people who are all equally rich, so dispersion within the admitted
    set is low. Dispersion among the admitted is not the same question as who
    was admitted. It is the same shape of error as measuring exposure over a
    candidate pool that re-ranking cannot change.

    The test pins the paradox so nobody quietly 'fixes' the number later.
    """
    scn = Scenario()
    pop = draw_for(scn, 0)
    rows = {}
    for key in ("clearing", "bound"):
        arm = ARM_BY_KEY[key]
        rows[key] = summarise(solve(arm, scn, pop), arm, scn, pop)

    # Market clearing shuts out the bottom income quartile entirely...
    assert rows["clearing"]["low_income_served"] < rows["bound"]["low_income_served"]
    # ...and scores *better* on the inequality metric for doing it.
    assert rows["clearing"]["access_gini"] < rows["bound"]["access_gini"]
    # `income_ratio` is the metric it is usually mistaken for, and it reads right.
    assert rows["clearing"]["income_ratio"] > 2.0
    assert abs(rows["bound"]["income_ratio"] - 1.0) < 0.05


def test_markup_is_suppressed_when_almost_nothing_is_resold():
    """Otherwise the strictest arm gets credited with a 22x markup on six seats,
    because the clearing price of a vanishing inventory runs to the top of the
    demand curve."""
    scn = Scenario()
    pop = draw_for(scn, 0)
    arm = ARM_BY_KEY["bound"]
    row = summarise(solve(arm, scn, pop), arm, scn, pop)
    assert row["resale_volume"] < 0.005
    assert row["markup"] != row["markup"]  # NaN


def test_aggregate_keeps_an_all_nan_column_as_nan():
    """A NaN here means 'this arm has no such price', never 'the run failed'."""
    scn = Scenario()
    rows = []
    arm = ARM_BY_KEY["queue"]
    for t in range(3):
        pop = draw_for(scn, t)
        rows.append(summarise(solve(arm, scn, pop), arm, scn, pop))
    agg = aggregate(rows)
    assert agg["clearing_price"] != agg["clearing_price"]
    assert agg["broker_capture"] > 0.0
    assert agg["n_trials"] == 3


def test_customer_harm_counts_the_gate():
    scn = Scenario()
    pop = draw_for(scn, 0)
    bound = ARM_BY_KEY["bound"]
    row = summarise(solve(bound, scn, pop), bound, scn, pop)
    assert row["gate_denied"] == scn.gate_id_failure
    assert row["customer_harm"] >= row["false_rejected"] + row["gate_denied"]
