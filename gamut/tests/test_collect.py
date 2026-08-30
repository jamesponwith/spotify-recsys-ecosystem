"""The fusion-boundary sentinel is the one place Gamut can silently measure
nothing: fusion marks "channel never saw this" as channel_depth + 1, which
passes a >= 0 filter and turns every channel block into the whole pool."""

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
    assert strip_sentinel(cr).tolist() == [2, ABSENT, 0, ABSENT, 1, ABSENT]


def test_strip_sentinel_catches_a_single_absent_candidate():
    # depth 3, one absence: the sentinel 4 appears once but exceeds the pool size
    cr = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    assert strip_sentinel(cr).tolist() == [0, 1, 2, ABSENT]


def test_strip_sentinel_leaves_full_coverage_alone():
    # a channel that returned the whole pool: ranks are a permutation, no sentinel
    cr = np.array([3.0, 0.0, 2.0, 1.0], dtype=np.float32)
    assert strip_sentinel(cr).tolist() == [3, 0, 2, 1]


def test_channel_block_never_contains_a_sentinel_ranked_candidate():
    # Pool of 5; the channel returned two of them (ranks 0 and 1, depth 2), so
    # fusion stamps the other three with sentinel 3. Pre-fix, all five survived
    # the audit filter and the "channel" block was the pool re-sorted.
    raw = np.array([3.0, 1.0, 3.0, 0.0, 3.0], dtype=np.float32)
    ranks = np.full((1, 1, 5), ABSENT, dtype=np.int32)
    ranks[0, 0] = strip_sentinel(raw)
    collected = _collected(ranks)
    block = _channel_block(collected, 0, depth=5)[0]
    returned = block[block != ABSENT]
    sentinel_candidates = collected.indices[0][raw == raw.max()]
    assert not np.isin(returned, sentinel_candidates).any()
    # and the block is the channel's own ordering, nothing more
    assert returned.tolist() == [collected.indices[0][3], collected.indices[0][1]]


def test_load_refuses_a_contaminated_cache(tmp_path):
    # rows written before the fix carry the duplicated sentinel verbatim
    ranks = np.full((1, 1, 5), ABSENT, dtype=np.int32)
    ranks[0, 0] = np.array([3, 1, 3, 0, 3], dtype=np.int32)
    path = _collected(ranks).save(tmp_path / "collected.npz")
    with pytest.raises(ValueError, match="gamut collect"):
        Collected.load(path)


def test_load_accepts_a_clean_cache(tmp_path):
    ranks = np.full((1, 1, 5), ABSENT, dtype=np.int32)
    ranks[0, 0] = np.array([ABSENT, 1, ABSENT, 0, ABSENT], dtype=np.int32)
    path = _collected(ranks).save(tmp_path / "collected.npz")
    loaded = Collected.load(path)
    assert loaded.channel_ranks.tolist() == ranks.tolist()
