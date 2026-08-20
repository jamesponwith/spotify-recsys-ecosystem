"""Selection tests use a hand-built stub catalog so the constraint behaviour is
verified in isolation, not tangled up with retrieval quality."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.assemble.select import select


class StubCatalog:
    """Minimal surface the selector touches: collab vectors, artists, durations."""

    def __init__(self, n=40, n_artists=8, dim=6, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n, dim)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        class _Index:
            vectors = v

        self.collab = _Index()
        self.artist_ids = np.arange(n) % n_artists
        self._duration = np.full(n, 200_000, dtype=np.int64)

    def col(self, name):
        if name == "duration_ms":
            return self._duration
        raise KeyError(name)


@pytest.fixture
def stub():
    return StubCatalog()


def test_selects_the_requested_number(stub):
    out = select(stub, np.arange(40), np.linspace(1, 0, 40).astype(np.float32), n_tracks=10)
    assert len(out.indices) == 10


def test_artist_cap_is_enforced(stub):
    out = select(
        stub, np.arange(40), np.linspace(1, 0, 40).astype(np.float32), n_tracks=16, max_per_artist=2
    )
    counts = np.bincount(stub.artist_ids[out.indices])
    assert counts.max() <= 2


def test_artist_cap_of_one_gives_unique_artists(stub):
    out = select(
        stub, np.arange(40), np.linspace(1, 0, 40).astype(np.float32), n_tracks=8, max_per_artist=1
    )
    assert len(set(stub.artist_ids[out.indices].tolist())) == len(out.indices)


def test_duration_target_overrides_track_count(stub):
    # 200 s per track; ask for ~20 minutes and expect roughly six tracks.
    out = select(
        stub,
        np.arange(40),
        np.linspace(1, 0, 40).astype(np.float32),
        n_tracks=99,
        target_duration_s=1200,
        max_per_artist=10,
    )
    assert 1000 <= out.total_duration_s <= 1400


def test_lower_lambda_increases_diversity(stub):
    idx = np.arange(40)
    scores = np.linspace(1, 0, 40).astype(np.float32)
    relevance_first = select(stub, idx, scores, n_tracks=8, mmr_lambda=1.0, max_per_artist=10)
    diverse = select(stub, idx, scores, n_tracks=8, mmr_lambda=0.1, max_per_artist=10)

    def spread(sel):
        v = stub.collab.vectors[sel.indices]
        sims = v @ v.T
        iu = np.triu_indices(len(sel.indices), k=1)
        return 1.0 - sims[iu].mean()

    assert spread(diverse) >= spread(relevance_first)


def test_pure_relevance_follows_the_input_order(stub):
    out = select(
        stub,
        np.arange(40),
        np.linspace(1, 0, 40).astype(np.float32),
        n_tracks=5,
        mmr_lambda=1.0,
        max_per_artist=10,
    )
    assert out.indices.tolist() == [0, 1, 2, 3, 4]


def test_empty_candidates_are_handled(stub):
    out = select(stub, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), n_tracks=5)
    assert len(out.indices) == 0


def test_requesting_more_than_available_returns_what_exists(stub):
    out = select(stub, np.arange(4), np.ones(4, dtype=np.float32), n_tracks=20, max_per_artist=10)
    assert len(out.indices) == 4
