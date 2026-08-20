"""The intervention: an exposure-aware re-ranking, and its price.

    score'(t) = rank_norm(fused(t)) - penalty * pop_norm(t)

`rank_norm` rather than min-max, for the reason Cadence learned the hard way in
`assemble/select.py`: fused RRF scores are heavily skewed, so min-max compresses
almost every candidate into a narrow band and the penalty term silently
dominates at any strength that does anything at all. Rank-normalising makes the
penalty's units interpretable -- a penalty of 1.0 can move a track across the
entire candidate list, and 0.1 across a tenth of it.

The point is not that this particular penalty is the best possible intervention.
It is the simplest one whose strength is a single readable number, which is what
makes the trade-off curve a decision rather than a black box.
"""

from __future__ import annotations

import numpy as np


def rank_normalize(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map scores to evenly spaced values in [0, 1], best first, per row."""
    out = np.zeros_like(scores, dtype=np.float32)
    for i in range(scores.shape[0]):
        n = int(valid[i].sum())
        if n == 0:
            continue
        order = np.argsort(-scores[i, :n], kind="stable")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.linspace(1.0, 0.0, n, dtype=np.float32)
        out[i, :n] = ranks
    return out


def popularity_norm(play_counts: np.ndarray) -> np.ndarray:
    """Log play counts scaled to [0, 1] across the catalog."""
    lp = np.log1p(play_counts.astype(np.float64))
    span = lp.max() - lp.min()
    return (
        ((lp - lp.min()) / span).astype(np.float32) if span > 0 else np.zeros_like(lp, np.float32)
    )


def rerank(
    indices: np.ndarray, scores: np.ndarray, pop_norm: np.ndarray, penalty: float
) -> np.ndarray:
    """Return re-ordered indices, same shape, -1 padding preserved."""
    valid = indices >= 0
    base = rank_normalize(scores, valid)
    out = np.full_like(indices, -1)
    for i in range(indices.shape[0]):
        n = int(valid[i].sum())
        if n == 0:
            continue
        row = indices[i, :n].astype(np.int64)
        adjusted = base[i, :n] - penalty * pop_norm[row]
        out[i, :n] = row[np.argsort(-adjusted, kind="stable")]
    return out


def apply_artist_cap(block: np.ndarray, artists: np.ndarray, cap: int) -> np.ndarray:
    """Greedily drop a track once its artist already holds `cap` slots in the row.

    A popularity penalty and an artist cap look like the same kind of knob and are
    not. The penalty asks "is this track popular?"; the cap asks "has this artist
    already been heard?" Concentration in a recommender is driven mostly by a few
    artists owning many *catalog entries*, which a per-track popularity term
    cannot see -- so the two interventions move different metrics, and the audit
    reports both rather than assuming one stands in for the other.
    """
    out = np.full_like(block, -1)
    for i in range(block.shape[0]):
        row = block[i][block[i] >= 0].astype(np.int64)
        if row.size == 0:
            continue
        seen: dict[int, int] = {}
        kept = []
        for t in row:
            a = int(artists[t])
            if seen.get(a, 0) >= cap:
                continue
            seen[a] = seen.get(a, 0) + 1
            kept.append(t)
        kept_arr = np.asarray(kept[: block.shape[1]], dtype=block.dtype)
        out[i, : kept_arr.size] = kept_arr
    return out
