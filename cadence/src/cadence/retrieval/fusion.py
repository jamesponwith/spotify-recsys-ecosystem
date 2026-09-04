"""Reciprocal-rank fusion of heterogeneous channels.

RRF combines rankings without needing the channels' scores to be comparable —
cosine similarity, TF-IDF dot products and negative-exponential distances live
on wildly different scales, and calibrating them against each other is a
research project in itself. RRF only needs the *order*, which every channel
agrees on.

    score(d) = sum_c  w_c / (k + rank_c(d))

The per-channel ranks and raw scores are preserved on the way out because they
become features for the learned reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channels import ChannelResult


@dataclass
class FusedCandidates:
    indices: np.ndarray
    scores: np.ndarray
    channel_ranks: dict[str, np.ndarray]  # channel -> rank per fused index (0 = best)
    channel_scores: dict[str, np.ndarray]
    channels_present: list[str] = field(default_factory=list)
    # How many candidates each channel actually returned. A rank in
    # channel_ranks is real iff it is < the channel's depth; the absent
    # sentinel below is depth + 1. Consumers that need "did this channel see
    # this candidate" (Gamut's audit) resolve it from here rather than
    # guessing which values are sentinels.
    channel_depths: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)


def reciprocal_rank_fusion(
    results: list[ChannelResult],
    weights: dict[str, float],
    *,
    k: float = 60.0,
    top_n: int = 1500,
) -> FusedCandidates:
    live = [r for r in results if len(r) > 0]
    if not live:
        return FusedCandidates(
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {}, {}, []
        )

    fused: dict[int, float] = {}
    ranks: dict[str, dict[int, int]] = {}
    raw: dict[str, dict[int, float]] = {}

    for res in live:
        w = float(weights.get(res.name, 1.0))
        ranks[res.name] = {}
        raw[res.name] = {}
        for rank, (idx, score) in enumerate(zip(res.indices, res.scores, strict=True)):
            i = int(idx)
            fused[i] = fused.get(i, 0.0) + w / (k + rank + 1.0)
            ranks[res.name][i] = rank
            raw[res.name][i] = float(score)

    items = np.fromiter(fused.keys(), dtype=np.int64, count=len(fused))
    scores = np.fromiter(fused.values(), dtype=np.float32, count=len(fused))
    order = np.argsort(-scores, kind="stable")[:top_n]
    items = items[order]
    scores = scores[order]

    # Missing entries get a sentinel rank just past the channel's depth, so the
    # reranker can distinguish "ranked last" from "this channel never saw it".
    channel_ranks: dict[str, np.ndarray] = {}
    channel_scores: dict[str, np.ndarray] = {}
    for res in live:
        depth = len(res)
        r = ranks[res.name]
        s = raw[res.name]
        channel_ranks[res.name] = np.array(
            [r.get(int(i), depth + 1) for i in items], dtype=np.float32
        )
        channel_scores[res.name] = np.array([s.get(int(i), 0.0) for i in items], dtype=np.float32)

    return FusedCandidates(
        indices=items,
        scores=scores,
        channel_ranks=channel_ranks,
        channel_scores=channel_scores,
        channels_present=[r.name for r in live],
        channel_depths={r.name: len(r) for r in live},
    )
