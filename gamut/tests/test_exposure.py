import numpy as np
import pandas as pd

from gamut.exposure import CatalogFacts, measure


def _facts(n=10):
    frame = pd.DataFrame(
        {
            "artist_uri": [f"a{i // 2}" for i in range(n)],  # 2 tracks per artist
            "n_playlists": np.arange(1, n + 1, dtype=float),
        }
    )
    return CatalogFacts.build(frame, tail_percentile=50.0, head_percentile=90.0)


def test_padding_is_not_counted_as_exposure():
    """-1 is the pad value; counting it would both crash and inflate coverage."""
    facts = _facts()
    block = np.array([[0, 1, -1, -1], [2, -1, -1, -1]], dtype=np.int32)
    r = measure(block, facts)
    assert r.n_recommendations == 3
    assert abs(r.track_coverage - 3 / 10) < 1e-9


def test_gini_is_zero_when_every_item_shown_equally():
    facts = _facts()
    block = np.arange(10, dtype=np.int32).reshape(1, 10)
    assert measure(block, facts).track_gini < 1e-9


def test_gini_approaches_one_when_one_item_takes_everything():
    facts = _facts()
    block = np.zeros((10, 10), dtype=np.int32)
    assert measure(block, facts).track_gini > 0.85


def test_tail_lift_above_one_means_the_tail_is_over_served():
    facts = _facts()
    # tracks 0-4 are the bottom half by n_playlists
    block = np.array([[0, 1, 2, 3]], dtype=np.int32)
    r = measure(block, facts)
    assert r.tail_share == 1.0
    assert r.tail_lift > 1.0


def test_empty_block_does_not_divide_by_zero():
    r = measure(np.full((2, 3), -1, dtype=np.int32), _facts())
    assert r.n_recommendations == 0 and r.track_gini == 0.0
