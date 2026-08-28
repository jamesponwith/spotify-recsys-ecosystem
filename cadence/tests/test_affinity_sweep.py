"""Unit tests for the affinity sweep's metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cadence.eval.affinity_sweep import build_battery, mood_error, tag_adherence


class _Cat:
    def __init__(self, matrix, cols=None):
        self.tag_matrix_csc = matrix.tocsc()
        self._cols = cols or {}

    def col(self, name):
        return self._cols[name]


@pytest.fixture
def catalog():
    #  track:      tag0  tag1
    #    0          900     0    <- filed under tag0 constantly
    #    1            1     0    <- filed there once
    #    2            0     0    <- not at all
    return _Cat(sparse.csr_matrix(np.array([[900.0, 0.0], [1.0, 0.0], [0.0, 0.0]])))


def test_share_cannot_tell_a_strong_tag_from_a_token_one(catalog):
    """Why the graded metric exists: both tracks clear the binary bar."""
    strong, _ = tag_adherence(catalog, np.array([0]), [0])
    weak, _ = tag_adherence(catalog, np.array([1]), [0])
    assert strong == weak == 1.0


def test_strength_does_tell_them_apart(catalog):
    _, strong = tag_adherence(catalog, np.array([0]), [0])
    _, weak = tag_adherence(catalog, np.array([1]), [0])
    assert strong > weak * 5


def test_a_track_with_no_requested_tag_lowers_both(catalog):
    share, strength = tag_adherence(catalog, np.array([0, 2]), [0])
    assert share == 0.5
    assert 0 < strength < np.log1p(900)


def test_no_requested_tag_returns_none_not_zero(catalog):
    """A mood-only query must not contribute a meaningless zero to the mean."""
    assert tag_adherence(catalog, np.array([0]), []) is None


def test_mood_error_is_none_when_nothing_was_stated():
    cat = _Cat(sparse.csr_matrix(np.zeros((2, 1))), {"energy": np.array([0.5, 0.5])})
    assert mood_error(cat, np.array([0, 1]), {}) is None


def test_mood_error_measures_distance_from_the_target():
    cat = _Cat(sparse.csr_matrix(np.zeros((2, 1))), {"energy": np.array([0.4, 0.6])})
    assert mood_error(cat, np.array([0, 1]), {"energy": 0.5}) == pytest.approx(0.0)
    assert mood_error(cat, np.array([0, 1]), {"energy": 0.8}) == pytest.approx(0.3)


def test_battery_covers_three_families_and_is_deterministic():
    a, b = build_battery(), build_battery()
    assert [x["query"] for x in a] == [x["query"] for x in b]
    assert {x["family"] for x in a} == {"mood", "tag", "mixed"}
