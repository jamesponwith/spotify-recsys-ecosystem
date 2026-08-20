"""Integration tests. Skipped wholesale when Cadence's artifacts are absent."""

from __future__ import annotations

import numpy as np
import pytest

from timbre.config import CADENCE_ARTIFACTS, CADENCE_PROCESSED

pytestmark = pytest.mark.skipif(
    not (CADENCE_ARTIFACTS / "spaces.npz").exists()
    or not (CADENCE_PROCESSED / "tracks.parquet").exists(),
    reason="Cadence artifacts not built; run `make all` in ../cadence first.",
)


@pytest.fixture(scope="module")
def loaded():
    from timbre.config import Phase0Config
    from timbre.phase0 import data as data_mod

    return data_mod.load(Phase0Config())


def test_split_is_disjoint_and_covers_the_usable_pool(loaded):
    assert np.intersect1d(loaded.train_idx, loaded.test_idx).size == 0
    combined = np.union1d(loaded.train_idx, loaded.test_idx)
    assert np.array_equal(combined, np.flatnonzero(loaded.usable))


def test_no_query_has_an_unanswerable_relevant_track(loaded):
    """Every relevant track must be one `oracle` could actually retrieve.

    A zero-embedding track is masked to -inf by the index, so leaving one in a
    relevant set would cap the ceiling while leaving `content` free to hit it.
    """
    dead = np.linalg.norm(loaded.y_raw, axis=1) == 0
    for rel in loaded.queries.relevant:
        assert not dead[rel].any()


def test_relevant_sets_are_test_split_only(loaded):
    is_test = np.zeros(loaded.n_tracks, dtype=bool)
    is_test[loaded.test_idx] = True
    for rel in loaded.queries.relevant:
        assert is_test[rel].all()


def test_split_is_reproducible_from_the_seed(loaded):
    from timbre.config import Phase0Config
    from timbre.phase0 import data as data_mod

    again = data_mod.load(Phase0Config())
    assert np.array_equal(again.test_idx, loaded.test_idx)
