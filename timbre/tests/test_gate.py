"""Gate 0's rule, including the case that produced invalid JSON."""

from __future__ import annotations

import json

from timbre.config import Phase0Config
from timbre.phase0.gate import rule


def _results(content: float, floor: float, oracle: float) -> dict:
    return {
        "random": {"recall_at_100": floor},
        "mean": {"recall_at_100": 0.0},
        "content_ridge": {"recall_at_100": 0.0},
        "content_mlp": {"recall_at_100": content},
        "oracle": {"recall_at_100": oracle},
    }


def test_zero_floor_yields_null_not_infinity():
    """`float("inf")` serialises to `Infinity`, which is not valid JSON."""
    g = rule(_results(0.004, 0.0, 0.03), Phase0Config())
    assert g["random_multiple"] is None
    assert g["random_criterion_vacuous"] is True
    json.dumps(g, allow_nan=False)  # raises on inf/nan


def test_zero_floor_criterion_falls_back_to_beating_zero():
    cfg = Phase0Config()
    assert rule(_results(0.004, 0.0, 0.03), cfg)["beats_random_floor"] is True
    assert rule(_results(0.0, 0.0, 0.03), cfg)["beats_random_floor"] is False


def test_verdict_needs_both_criteria():
    cfg = Phase0Config()
    # clears the floor but not the oracle ratio -> the real Phase 0 outcome
    assert rule(_results(0.004, 0.0, 0.03), cfg)["passed"] is False
    # clears both
    assert rule(_results(0.010, 0.001, 0.03), cfg)["passed"] is True


def test_best_content_system_is_the_stronger_one():
    g = rule(_results(0.004, 0.0, 0.03), Phase0Config())
    assert g["best_content_system"] == "content_mlp"
