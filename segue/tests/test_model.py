"""Model-level invariants, on synthetic sequences."""

from __future__ import annotations

import numpy as np

from segue.model import _targets


def _fixture():
    # 8 tracks, 3-d embeddings; one playlist covering positions 0..7
    emb = np.eye(8, dtype=np.float32)[:, :8]
    tracks = np.arange(8, dtype=np.int32)
    ends = np.array([8], dtype=np.int64)
    owners = np.zeros(1, dtype=np.int64)
    return emb, tracks, ends, owners


def test_horizon_one_is_the_next_track_exactly():
    emb, tracks, ends, owners = _fixture()
    y = _targets(np.array([3]), owners, tracks, ends, emb, horizon=1)
    assert np.allclose(y[0], emb[3])


def test_horizon_averages_the_next_h_tracks():
    emb, tracks, ends, owners = _fixture()
    y = _targets(np.array([2]), owners, tracks, ends, emb, horizon=3)
    expect = emb[2:5].mean(axis=0)
    expect = expect / np.linalg.norm(expect)
    assert np.allclose(y[0], expect, atol=1e-6)


def test_horizon_is_clipped_at_the_playlist_end():
    """Reading past the end would pull in the *next playlist's* tracks and
    invent adjacencies no human created."""
    emb, tracks, ends, owners = _fixture()
    y = _targets(np.array([6]), owners, tracks, ends, emb, horizon=10)
    expect = emb[6:8].mean(axis=0)
    expect = expect / np.linalg.norm(expect)
    assert np.allclose(y[0], expect, atol=1e-6)


def test_targets_are_unit_norm():
    emb, tracks, ends, owners = _fixture()
    y = _targets(np.array([0, 3, 5]), np.zeros(3, dtype=np.int64), tracks, ends, emb, horizon=4)
    assert np.allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-6)
