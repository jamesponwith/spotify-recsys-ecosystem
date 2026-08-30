"""Unit tests for the lexicon calibration audit, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cadence.eval.lexicon_audit import audit
from cadence.planner.lexicon import MoodEntry


@pytest.fixture
def world():
    #  track:  energy  valence  has_audio   tags: sleep  party
    #    0      0.20    0.5     yes                1      0
    #    1      0.60    0.5     yes                1      0
    #    2      0.90    0.5     yes                0      1
    #    3      0.99    0.5     no                 1      1   <- excluded everywhere
    frame = pd.DataFrame(
        {
            "energy": [0.20, 0.60, 0.90, 0.99],
            "valence": [0.5, 0.5, 0.5, 0.5],
            "has_audio": [True, True, True, False],
        }
    )
    tags = sparse.csr_matrix(np.array([[1, 0], [3, 0], [0, 2], [1, 1]], dtype=np.float32))
    return frame, tags, ["sleep", "party"]


def test_means_are_over_filed_tracks_with_audio_only(world):
    frame, tags, vocab = world
    report = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.1})})
    (pair,) = report["pairs"]
    assert pair["n_tracks"] == 2  # track 3 is filed but has no audio
    assert pair["folksonomy_mean"] == pytest.approx(0.40)
    assert pair["catalog_mean"] == pytest.approx((0.20 + 0.60 + 0.90) / 3, abs=1e-4)


def test_a_target_further_than_the_catalog_mean_is_flagged(world):
    frame, tags, vocab = world
    # humans 0.40, catalog 0.567 (gap 0.167): 0.10 is worse, 0.30 is better.
    worse = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.10})})
    better = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.30})})
    assert worse["n_target_worse_than_catalog"] == 1
    assert better["n_target_worse_than_catalog"] == 0


def test_a_word_that_is_not_a_tag_is_listed_not_dropped_silently(world):
    frame, tags, vocab = world
    lexicon = {"sleep": MoodEntry({"energy": 0.1}), "aggressive": MoodEntry({"energy": 0.9})}
    report = audit(frame, tags, vocab, lexicon)
    assert report["n_pairs"] == 1
    assert report["words_not_tags"] == ["aggressive"]


def test_every_asserted_dimension_is_a_pair_and_a_bare_theme_is_not(world):
    frame, tags, vocab = world
    lexicon = {
        "party": MoodEntry({"energy": 0.8, "valence": 0.7}),
        "sleep": MoodEntry({}, themes=("sleep",)),
    }
    report = audit(frame, tags, vocab, lexicon)
    assert report["n_pairs"] == 2
    assert report["n_words_audited"] == 1
    assert report["words_not_tags"] == []


def test_pairs_are_ordered_most_damning_first(world):
    frame, tags, vocab = world
    lexicon = {"sleep": MoodEntry({"energy": 0.35, "valence": 0.0})}
    report = audit(frame, tags, vocab, lexicon)
    assert [p["dimension"] for p in report["pairs"]] == ["valence", "energy"]
