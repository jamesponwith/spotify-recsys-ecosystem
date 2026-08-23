import pytest

from concerto.config import Scenario
from concerto.ledger import LADDER, contention, contention_curve, leak_ladder


def test_the_two_rungs_that_differ_most_on_chain_are_identical_off_it():
    """The headline of the Cardano work.

    A validator-capped resale and a fully soulbound token are completely
    different contracts -- one permits a constrained transfer, the other permits
    none at all -- and they produce the same broker economics, because the
    channel that neither of them can see is the one that was already carrying
    the volume. The chain enforces what it observes; a key sale is not observed.
    """
    rungs = {r["key"]: r for r in leak_ladder(Scenario(), trials=3)["rungs"]}
    a, b = rungs["capped_script"], rungs["soulbound"]
    assert abs(a["spread_retained"] - b["spread_retained"]) < 0.01
    assert abs(a["broker_capture"] - b["broker_capture"]) < 0.01


def test_a_royalty_taxes_the_spread_without_closing_it():
    rungs = {r["key"]: r for r in leak_ladder(Scenario(), trials=3)["rungs"]}
    assert rungs["royalty"]["spread_retained"] < rungs["plain_nft"]["spread_retained"]
    assert rungs["royalty"]["spread_retained"] > 0.5


def test_only_the_off_chain_rung_closes_it():
    rungs = {r["key"]: r for r in leak_ladder(Scenario(), trials=3)["rungs"]}
    assert rungs["soulbound_gate"]["spread_retained"] < 0.01
    assert rungs["soulbound_gate"]["broker_capture"] < 0.005


def test_the_ladder_is_monotone():
    """Each rung is at least as strong as the one before it. If this ever fails,
    a rung has been mis-parameterised rather than the world having changed."""
    rows = leak_ladder(Scenario(), trials=3)["rungs"]
    retained = [r["spread_retained"] for r in rows]
    assert retained == sorted(retained, reverse=True)
    assert [r["key"] for r in rows] == [r.key for r in LADDER]


def test_one_inventory_utxo_sells_one_seat_per_block():
    """The whole eUTxO problem in a single assertion."""
    r = contention(seats=4232, concurrent_buyers=40_000, shards=1)
    assert abs(r["settled_per_block"] - 1.0) < 1e-9
    assert r["minutes_to_clear"] > 500


def test_sharding_trades_settlement_time_for_a_retry_storm():
    """`minutes_to_clear` looks respectable at 100 shards. What it hides is that
    every buyer submits a signed transaction every block and has it rejected
    ~400 times first. A waiting room is a queue; this is 400 error dialogs."""
    r = contention(seats=4232, concurrent_buyers=40_000, shards=100)
    assert r["minutes_to_clear"] < 10
    assert r["attempts_per_purchase"] > 300


def test_more_shards_always_helps_and_never_exceeds_the_bid_count():
    rows = contention_curve(4232, 40_000)["rows"]
    settled = [r["settled_per_block"] for r in rows]
    assert settled == sorted(settled)
    assert all(r["settled_per_block"] <= r["concurrent_buyers"] + 1e-9 for r in rows)


def test_contention_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        contention(seats=100, concurrent_buyers=10, shards=0)
