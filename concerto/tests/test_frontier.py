import numpy as np

from concerto.config import ARM_BY_KEY, Scenario
from concerto.frontier import cap_curve, clearing_curve, nights_curve, reserve_curve
from concerto.simulate import run_arm

SCN = Scenario()


def test_affinity_rationing_shuts_casual_fans_out_completely():
    """The cost the reserve exists to buy back. A merit cut is absolute: a fan
    one place below it has no chance at all, not a small one.

    The figure is 0.003% rather than a clean zero, and the residue is not a
    chance anyone has -- it is the single fan sitting exactly on the median
    affinity boundary, who takes a partial allocation because the cut lands
    mid-fan. Asserted at 0.1% so the test measures the shutout rather than the
    arithmetic of where a quantile falls.
    """
    rows = reserve_curve(SCN, trials=4)["rows"]
    assert rows[0]["reserve_share"] == 0.0
    assert rows[0]["casual_served"] < 0.001
    assert rows[0]["superfan_served"] > 0.9


def test_the_reserve_is_free_up_to_the_point_superfan_demand_stops():
    """The reason 20% happened to be a good guess, which was not the reason
    given when it was guessed.

    Top-decile demand is about 78% of the house, so there is slack. Reserving
    seats costs superfans exactly nothing until the reserve eats into that
    slack -- and then it costs them roughly one-for-one.
    """
    curve = reserve_curve(SCN, trials=4)
    rows = {r["reserve_share"]: r for r in curve["rows"]}
    assert abs(rows[0.20]["superfan_served"] - rows[0.0]["superfan_served"]) < 1e-6
    assert rows[0.50]["superfan_served"] < 0.8 * rows[0.0]["superfan_served"]
    assert curve["knee"]["reserve_share"] >= 0.20


def test_a_full_reserve_is_just_a_lottery():
    """At 100% the merit rule is gone, so every group should be served at the
    same rate. A drift here means the reserve path and the lottery path have
    diverged."""
    row = reserve_curve(SCN, trials=4)["rows"][-1]
    assert row["reserve_share"] == 1.0
    assert abs(row["superfan_served"] - row["casual_served"]) < 0.01
    assert abs(row["superfan_served"] - row["low_income_served"]) < 0.01


def test_the_reserve_buys_back_only_a_fraction_of_what_merit_took():
    """Honest sizing. A 20% reserve is a lottery over 20% of the house shared
    with everyone still unserved, so it does not restore casual access -- it
    gets a fifth of the way back and should be described that way."""
    rows = {r["reserve_share"]: r for r in reserve_curve(SCN, trials=4)["rows"]}
    lottery_rate = rows[1.0]["casual_served"]
    assert 0.1 < rows[0.20]["casual_served"] / lottery_rate < 0.35


def test_a_purchase_cap_is_only_worth_it_before_the_margin_is_closed():
    """The assertion this sweep half-falsified, pinned in both directions.

    The policy document claimed tight caps cost families more than they cost
    brokers. On the capped-exchange arm that is right and then some -- 20 points
    of truncated family demand to save under 4 points of broker capture. On an
    unprotected queue it is flatly wrong: the same 20 points buys back over 30
    points of capture, which is one of the better trades available.

    The cap is not good or bad. It is worth something exactly to the extent the
    resale margin is still open, and a single-arm sweep would have confirmed
    whichever answer the arm happened to give.
    """
    curves = cap_curve(SCN, trials=4)["arms"]
    queue, capped = curves["queue"], curves["capped"]

    # Same cap, same population, so the cost to families is identical.
    assert abs(queue["party_cost"] - capped["party_cost"]) < 1e-9
    # What differs is what it buys.
    assert queue["cost_per_point_saved"] < 1.0
    assert capped["cost_per_point_saved"] > 3.0
    assert queue["capture_saved"] > 5 * capped["capture_saved"]


def test_tightening_a_cap_does_not_reduce_broker_volume_proportionally():
    """Brokers buy their way around it: halving the cap does not halve the
    identities, because the sector re-optimises against the same margin."""
    rows = {r["cap"]: r for r in cap_curve(SCN, trials=4)["arms"]["queue"]["rows"]}
    assert rows[2]["broker_identities"] / rows[8]["broker_identities"] > 0.5


def test_a_cap_at_party_size_costs_families_nothing():
    rows = {r["cap"]: r for r in cap_curve(SCN, trials=4)["arms"]["capped"]["rows"]}
    assert rows[4]["party_truncated"] == 0.0
    assert rows[2]["party_truncated"] > 0.15
    assert rows[1]["party_truncated"] > 0.5


def test_clearing_trades_artist_revenue_against_low_income_access_monotonically():
    rows = clearing_curve(SCN, trials=4)["rows"]
    revenue = [r["artist_per_seat"] for r in rows]
    poor = [r["low_income_served"] for r in rows]
    assert revenue == sorted(revenue)
    assert poor == sorted(poor, reverse=True)


def test_the_clearing_curve_has_no_knee():
    """The result that contradicts how the 15% was chosen.

    There is no share at which the trade suddenly turns. Revenue gained per
    point of low-income access lost falls monotonically from the very first
    seat, so the sweep does not identify an optimum -- it prices a choice. Any
    number picked here is a judgement about how much artist revenue is worth,
    and should be defended as one rather than as a discovered optimum.
    """
    rows = [r for r in clearing_curve(SCN, trials=4)["rows"] if r["clearing_share"] > 0]
    base_rev, base_poor = 95.0, 1.0
    efficiency = [
        (r["artist_per_seat"] - base_rev) / max(base_poor - r["poor_access_kept"], 1e-9)
        for r in rows
    ]
    assert efficiency == sorted(efficiency, reverse=True)


def test_a_second_night_is_worth_about_what_verified_fan_is_worth():
    """The headline of the frontier work, and the reason adding dates is named
    before any mechanism is.

    One extra night moves what the average fan pays about as far as the entire
    verified-fan identity apparatus does -- and it doubles the artist's revenue
    instead of wrongly rejecting 4% of applicants.
    """
    nights = {r["nights"]: r for r in nights_curve(SCN, trials=4)["rows"]}
    verified = run_arm(ARM_BY_KEY["verified"], SCN, trials=4)
    assert abs(nights[2]["price_multiple"] - verified["price_multiple"]) < 0.06
    assert nights[2]["artist_vs_one_night"] > 1.9
    assert verified["customer_harm"] > 0.03
    assert nights[2]["low_income_served"] > 2 * nights[1]["low_income_served"]


def test_enough_nights_removes_the_broker_without_any_policy_at_all():
    rows = nights_curve(SCN, trials=4)["rows"]
    assert rows[0]["broker_capture"] > 0.5
    last = rows[-1]
    assert abs(last["demand_multiple"] - 1.0) < 1e-9
    assert last["broker_capture"] < 0.01
    assert last["face_access"] > 0.95
    assert last["identity_burn_total"] < 1.0


def test_every_nights_row_faces_the_identical_crowd():
    """The pairing the nights comparison rests on, asserted rather than assumed.

    Capacity scales by n and the demand multiple by 1/n, so total ticket demand
    is constant and the population sizes itself off that total. Every row is
    therefore scored against the same people -- the only difference between one
    night and eight is how many seats exist. If a change to `draw` ever broke
    that, the rows would still look entirely plausible.
    """
    from dataclasses import replace

    from concerto.population import draw_for

    ref = draw_for(SCN, 0)
    for n in (2, 3, 4, 6, 8):
        cell = replace(SCN, capacity=SCN.capacity * n, demand_multiple=SCN.demand_multiple / n)
        pop = draw_for(cell, 0)
        assert cell.on_sale == n * SCN.on_sale
        assert pop.n_fans == ref.n_fans
        assert np.array_equal(pop.wtp, ref.wtp)
        assert np.array_equal(pop.party, ref.party)


def test_nights_scale_artist_revenue_linearly():
    """Sanity guard on the construction: demand is held fixed and the house is
    played n times, so the artist's take should be n times one night's."""
    for r in nights_curve(SCN, trials=4)["rows"]:
        assert abs(r["artist_vs_one_night"] - r["nights"]) < 0.05


def test_identity_burn_falls_faster_than_capture():
    """Rent dissipation is convex in how hot the show is. Halving the
    oversubscription cuts broker capture by about a seventh and the money burned
    on identities by about two fifths."""
    rows = {r["nights"]: r for r in nights_curve(SCN, trials=4)["rows"]}
    capture_ratio = rows[1]["broker_capture"] / max(rows[2]["broker_capture"], 1e-9)
    burn_ratio = rows[1]["identity_burn_total"] / max(rows[2]["identity_burn_total"], 1e-9)
    assert burn_ratio > capture_ratio
    assert np.isfinite(burn_ratio)
