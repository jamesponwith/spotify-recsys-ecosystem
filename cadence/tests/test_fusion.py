"""The per-channel rank contract downstream consumers depend on: a real rank
is < the channel's depth, an absent candidate carries depth + 1, and
channel_depths says where the line is (Gamut's audit strips on it)."""

import numpy as np

from cadence.retrieval.channels import ChannelResult
from cadence.retrieval.fusion import reciprocal_rank_fusion


def _res(name: str, indices: list[int]) -> ChannelResult:
    idx = np.asarray(indices, dtype=np.int64)
    return ChannelResult(name, idx, np.ones(len(idx), dtype=np.float32))


def test_channel_depths_report_what_each_channel_returned():
    fused = reciprocal_rank_fusion([_res("a", [1, 2, 3]), _res("b", [3, 4])], weights={})
    assert fused.channel_depths == {"a": 3, "b": 2}


def test_absent_candidates_sit_just_past_the_channel_depth():
    fused = reciprocal_rank_fusion([_res("a", [1, 2, 3]), _res("b", [3, 4])], weights={})
    depth = fused.channel_depths["b"]
    ranks = fused.channel_ranks["b"]
    absent = ~np.isin(fused.indices, [3, 4])
    assert (ranks[absent] == depth + 1).all()
    assert (ranks[~absent] < depth).all()
