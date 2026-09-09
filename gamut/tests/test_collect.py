"""The fusion-boundary sentinel is the one place Gamut can silently measure
nothing: fusion marks "channel never saw this" as channel_depth + 1, which
passes a >= 0 filter and turns every channel block into the whole pool."""

import json
from pathlib import Path

import numpy as np
import pytest

from gamut.audit import _channel_block
from gamut.collect import ABSENT, Collected, strip_sentinel


def _collected(channel_ranks: np.ndarray) -> Collected:
    n_queries, depth = channel_ranks.shape[1:]
    indices = np.tile(np.arange(depth, dtype=np.int32) + 10, (n_queries, 1))
    scores = np.zeros((n_queries, depth), dtype=np.float32)
    truth: list[set[int]] = [set() for _ in range(n_queries)]
    titles = [f"q{i}" for i in range(n_queries)]
    return Collected(indices, scores, channel_ranks, truth, titles)


def test_strip_sentinel_maps_absent_candidates_to_absent():
    # channel depth 3 -> sentinel 4, shared by every candidate it never returned
    cr = np.array([2.0, 4.0, 0.0, 4.0, 1.0, 4.0], dtype=np.float32)
    assert strip_sentinel(cr, depth=3).tolist() == [2, ABSENT, 0, ABSENT, 1, ABSENT]


def test_strip_sentinel_catches_a_single_absent_candidate():
    cr = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    assert strip_sentinel(cr, depth=3).tolist() == [0, 1, 2, ABSENT]


def test_strip_sentinel_keeps_a_deep_real_rank():
    # A channel deeper than the fused pool that covers the whole row: its
    # largest rank is real and must survive. This is exactly the case a
    # depth-free heuristic (duplicated or out-of-range maximum) gets wrong.
    cr = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert strip_sentinel(cr, depth=5).tolist() == [2, 3, 4]


def test_strip_sentinel_leaves_full_coverage_alone():
    # a channel that returned the whole pool: ranks are a permutation, no sentinel
    cr = np.array([3.0, 0.0, 2.0, 1.0], dtype=np.float32)
    assert strip_sentinel(cr, depth=4).tolist() == [3, 0, 2, 1]


def test_channel_block_never_contains_a_sentinel_ranked_candidate():
    # Pool of 5; the channel returned two of them (ranks 0 and 1, depth 2), so
    # fusion stamps the other three with sentinel 3. Pre-fix, all five survived
    # the audit filter and the "channel" block was the pool re-sorted.
    raw = np.array([3.0, 1.0, 3.0, 0.0, 3.0], dtype=np.float32)
    ranks = np.full((1, 1, 5), ABSENT, dtype=np.int32)
    ranks[0, 0] = strip_sentinel(raw, depth=2)
    collected = _collected(ranks)
    block = _channel_block(collected, 0, depth=5)[0]
    returned = block[block != ABSENT]
    sentinel_candidates = collected.indices[0][raw == raw.max()]
    assert not np.isin(returned, sentinel_candidates).any()
    # and the block is the channel's own ordering, nothing more
    assert returned.tolist() == [collected.indices[0][3], collected.indices[0][1]]


def test_load_refuses_a_cache_collected_before_the_fix(tmp_path):
    # a pre-fix cache lacks the sentinel_stripped marker and carries fusion's
    # sentinel verbatim in its rank rows
    path = tmp_path / "collected.npz"
    np.savez_compressed(
        path,
        indices=np.zeros((1, 5), dtype=np.int32),
        scores=np.zeros((1, 5), dtype=np.float32),
        channel_ranks=np.array([[[3, 1, 3, 0, 3]]], dtype=np.int32),
        titles=np.array(["q0"], dtype=object),
        truth=np.array([json.dumps([])], dtype=object),
    )
    with pytest.raises(ValueError, match="gamut collect"):
        Collected.load(path)


def test_save_load_round_trips_a_clean_cache(tmp_path):
    ranks = np.full((1, 1, 5), ABSENT, dtype=np.int32)
    ranks[0, 0] = np.array([ABSENT, 1, ABSENT, 0, ABSENT], dtype=np.int32)
    path = _collected(ranks).save(tmp_path / "collected.npz")
    loaded = Collected.load(path)
    assert loaded.channel_ranks.tolist() == ranks.tolist()


def _clean_cache(tmp_path, depth: int) -> Path:
    ranks = np.full((1, 1, depth), ABSENT, dtype=np.int32)
    return _collected(ranks).save(tmp_path / "collected.npz")


def test_load_refuses_a_cache_shallower_than_the_depth_being_reported(tmp_path):
    # The shipped condition exactly: a 100-deep cache audited as a 1500-deep
    # pool, every figure from it labelled with a window it never covered.
    with pytest.raises(ValueError, match="collected 100 deep"):
        Collected.load(_clean_cache(tmp_path, 100), min_depth=1500)


def test_load_accepts_a_cache_exactly_as_deep_as_reported(tmp_path):
    assert Collected.load(_clean_cache(tmp_path, 100), min_depth=100).indices.shape[1] == 100


def test_load_without_a_depth_claim_is_unchanged(tmp_path):
    # `demo` reads the cache without publishing a funnel, so it must not be
    # forced to re-collect the whole pool to show one query.
    assert Collected.load(_clean_cache(tmp_path, 100)).indices.shape[1] == 100


def test_load_refuses_a_cache_whose_blocks_disagree_on_width(tmp_path):
    # Hand-assembled: a 1500-wide pool carrying 100-wide channel ranks would
    # publish every channel row under a depth it was never measured at.
    path = tmp_path / "collected.npz"
    np.savez_compressed(
        path,
        indices=np.zeros((1, 1500), dtype=np.int32),
        scores=np.zeros((1, 1500), dtype=np.float32),
        channel_ranks=np.full((1, 1, 100), ABSENT, dtype=np.int32),
        titles=np.array(["q0"], dtype=object),
        truth=np.array([json.dumps([])], dtype=object),
        sentinel_stripped=np.True_,
    )
    with pytest.raises(ValueError, match="disagreeing width"):
        Collected.load(path, min_depth=1500)
