"""Ranking and beyond-accuracy metrics.

The three accuracy metrics are the official RecSys Challenge 2018 definitions
for the Million Playlist Dataset, so numbers here are directly comparable with
the published leaderboard rather than being a bespoke score only this repo
understands.

    R-precision  overlap of the top-|G| predictions with the withheld tracks.
                 The official variant also gives 0.25 credit for matching the
                 *artist* of a withheld track, recognising that recommending a
                 different song by the right artist is a near miss, not a
                 total miss. Both variants are reported.

    NDCG         position-discounted gain over the withheld tracks.

    Clicks       how many "refresh 10 more" presses a listener would need
                 before hitting a withheld track: floor(rank_first_hit / 10).
                 Capped at 51 when nothing relevant appears in 500 results.
                 This is the most product-shaped of the three.

Beyond-accuracy metrics matter as much for playlists: a recommender that only
returns the global top-40 can score respectably on NDCG while being useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_CLICKS = 51
CLICK_WINDOW = 10


def r_precision(predicted: np.ndarray, ground_truth: set[int]) -> float:
    if not ground_truth:
        return 0.0
    r = len(ground_truth)
    top = predicted[:r]
    return float(sum(1 for p in top if int(p) in ground_truth) / r)


def r_precision_artist_aware(
    predicted: np.ndarray,
    ground_truth: set[int],
    track_artists: np.ndarray,
    *,
    artist_credit: float = 0.25,
) -> float:
    """Official MPD variant: full credit for a track hit, partial for an artist
    hit. Artist credit is consumed per withheld artist, so recommending ten
    songs by one withheld artist cannot farm the score."""
    if not ground_truth:
        return 0.0
    r = len(ground_truth)
    top = [int(p) for p in predicted[:r]]
    hits: float = float(sum(1 for p in top if p in ground_truth))

    gt_artists: dict[int, int] = {}
    for t in ground_truth:
        a = int(track_artists[t])
        gt_artists[a] = gt_artists.get(a, 0) + 1
    matched_tracks = {p for p in top if p in ground_truth}
    for p in top:
        if p in matched_tracks:
            continue
        a = int(track_artists[p])
        if gt_artists.get(a, 0) > 0:
            gt_artists[a] -= 1
            hits += artist_credit
    return float(hits / r)


def ndcg(predicted: np.ndarray, ground_truth: set[int], k: int | None = None) -> float:
    if not ground_truth:
        return 0.0
    preds = predicted if k is None else predicted[:k]
    gains = np.array([1.0 if int(p) in ground_truth else 0.0 for p in preds], dtype=np.float64)
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_n = min(len(ground_truth), len(preds))
    idcg = float((1.0 / np.log2(np.arange(2, ideal_n + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def clicks(predicted: np.ndarray, ground_truth: set[int], max_rank: int = 500) -> float:
    """Number of 10-track 'show more' pages before the first relevant track."""
    for i, p in enumerate(predicted[:max_rank]):
        if int(p) in ground_truth:
            return float(i // CLICK_WINDOW)
    return float(MAX_CLICKS)


def recall_at_k(predicted: np.ndarray, ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    top = {int(p) for p in predicted[:k]}
    return float(len(top & ground_truth) / len(ground_truth))


# ---- beyond accuracy ----------------------------------------------------


def gini(counts: np.ndarray) -> float:
    """Inequality of exposure across the catalog. 0 = every item recommended
    equally often, 1 = one item takes everything."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def catalog_coverage(recommended: list[np.ndarray], n_items: int) -> float:
    seen: set[int] = set()
    for r in recommended:
        seen.update(int(x) for x in r)
    return float(len(seen) / n_items) if n_items else 0.0


def intra_list_distance(vectors: np.ndarray) -> float:
    """Mean pairwise cosine distance within one playlist. Higher = more varied."""
    if vectors.shape[0] < 2:
        return 0.0
    sims = vectors @ vectors.T
    iu = np.triu_indices(vectors.shape[0], k=1)
    return float(1.0 - sims[iu].mean())


@dataclass
class MetricAccumulator:
    """Aggregates per-playlist metrics into means with standard errors."""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))

    def update(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.add(k, v)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, vals in self.values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[name] = float(arr.mean())
            # Standard error, so a difference between two runs can be read as
            # meaningful or not instead of being eyeballed.
            out[f"{name}_se"] = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        out["n"] = float(len(next(iter(self.values.values()), [])))
        return out


def evaluate_ranking(
    predicted: np.ndarray,
    ground_truth: set[int],
    track_artists: np.ndarray,
) -> dict[str, float]:
    return {
        "r_precision": r_precision(predicted, ground_truth),
        "r_precision_artist": r_precision_artist_aware(predicted, ground_truth, track_artists),
        "ndcg_100": ndcg(predicted, ground_truth, k=100),
        "ndcg_500": ndcg(predicted, ground_truth, k=500),
        "clicks": clicks(predicted, ground_truth),
        "recall_100": recall_at_k(predicted, ground_truth, 100),
        "recall_500": recall_at_k(predicted, ground_truth, 500),
    }
