"""Turn a ranked candidate list into a playlist that satisfies hard constraints.

A playlist is not a top-k list. Twenty near-duplicate songs by three artists is
a perfect ranking and a bad playlist, so relevance is traded against diversity
via Maximal Marginal Relevance, and per-artist caps and duration targets are
enforced during selection rather than patched afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..catalog import Catalog


@dataclass
class Selection:
    indices: np.ndarray
    scores: np.ndarray
    dropped: dict[str, int]
    total_duration_s: float


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.ones_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map scores to evenly spaced values in [0, 1] by rank, best = 1.

    Min-max normalisation is the obvious choice and the wrong one here: the
    reranker emits calibrated probabilities whose distribution is heavily
    skewed (median ~0.002, max ~0.7), so min-max leaves almost every candidate
    near zero and a handful near one. Any term blended against that — diversity,
    audio affinity — is then numerically irrelevant regardless of its weight.

    Ranking discards score magnitude, which is a real loss, but it makes the MMR
    tradeoff mean the same thing whatever the upstream scorer's calibration.
    """
    if x.size == 0:
        return x
    if x.size == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(-np.asarray(x, dtype=np.float64), kind="stable")
    ranks = np.empty(x.size, dtype=np.float32)
    ranks[order] = np.linspace(1.0, 0.0, x.size, dtype=np.float32)
    return ranks


def select(
    catalog: Catalog,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    n_tracks: int,
    max_per_artist: int = 2,
    mmr_lambda: float = 0.72,
    target_duration_s: float | None = None,
    pool_size: int = 500,
    affinity: np.ndarray | None = None,
    affinity_weight: float = 0.0,
) -> Selection:
    """Greedy MMR selection under artist and duration constraints.

    ``target_duration_s`` overrides ``n_tracks`` when set: the listener asked for
    "about 45 minutes", not "about 12 songs", and the two are not the same
    request.

    ``affinity`` (0..1 per candidate, higher = closer to the listener's stated
    audio targets) is blended into relevance at ``affinity_weight``. Retrieval
    ranks by "would a human put this here"; when someone explicitly asks for
    *calm*, a track measured at 0.9 energy is wrong however well it co-occurs.
    Off by default so that ranking quality is measured without it.
    """
    idx = np.asarray(candidate_indices, dtype=np.int64)[:pool_size]
    raw = np.asarray(candidate_scores, dtype=np.float32)[:pool_size]
    dropped = {"artist_cap": 0, "duration": 0}
    if idx.size == 0:
        return Selection(idx, raw, dropped, 0.0)

    relevance = _rank_normalize(raw)
    if affinity is not None and affinity_weight > 0:
        aff = _minmax(np.asarray(affinity, dtype=np.float32)[: idx.size])
        relevance = ((1.0 - affinity_weight) * relevance + affinity_weight * aff).astype(np.float32)
    vectors = catalog.collab.vectors[idx]  # already unit-normalised
    artists = catalog.artist_ids[idx]
    durations = catalog.col("duration_ms")[idx] / 1000.0

    selected: list[int] = []
    artist_counts: dict[int, int] = {}
    total_duration = 0.0
    # Running max similarity of each candidate to the selected set.
    max_sim = np.zeros(idx.size, dtype=np.float32)
    available = np.ones(idx.size, dtype=bool)

    limit = n_tracks if target_duration_s is None else 10_000

    while len(selected) < limit and available.any():
        if target_duration_s is not None and total_duration >= target_duration_s:
            break
        mmr = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
        mmr[~available] = -np.inf
        pick = int(np.argmax(mmr))
        if not np.isfinite(mmr[pick]):
            break

        artist = int(artists[pick])
        if artist_counts.get(artist, 0) >= max_per_artist:
            available[pick] = False
            dropped["artist_cap"] += 1
            continue

        # Do not overshoot a duration target by more than half a track.
        if target_duration_s is not None and selected:
            projected = total_duration + durations[pick]
            if projected > target_duration_s + 0.5 * durations[pick]:
                available[pick] = False
                dropped["duration"] += 1
                continue

        selected.append(pick)
        available[pick] = False
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        total_duration += float(durations[pick])
        sims = vectors @ vectors[pick]
        np.maximum(max_sim, sims, out=max_sim)

    order = np.asarray(selected, dtype=np.int64)
    return Selection(
        indices=idx[order],
        scores=raw[order],
        dropped=dropped,
        total_duration_s=total_duration,
    )
