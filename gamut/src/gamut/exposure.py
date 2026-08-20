"""Exposure metrics: who gets heard, and how unequally.

Accuracy metrics are computed per query and averaged. Exposure metrics are the
opposite shape -- they are properties of the *whole* run, because concentration
only exists across queries. A system that gives every listener a perfectly
diverse playlist can still route every one of them to the same 500 artists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from cadence.eval.metrics import gini


@dataclass
class CatalogFacts:
    """Static properties of the catalog the audit is measured against."""

    n_tracks: int
    artists: np.ndarray  # dense artist id per track
    n_artists: int
    play_counts: np.ndarray  # playlists per track, the popularity proxy
    tail_mask: np.ndarray  # bottom `tail_percentile` by play count
    head_mask: np.ndarray  # top (100 - head_percentile)
    tail_share_of_catalog: float = field(default=0.0)

    @classmethod
    def build(cls, frame, tail_percentile: float, head_percentile: float) -> CatalogFacts:
        uris = frame["artist_uri"].to_numpy(dtype=object)
        lookup: dict[str, int] = {}
        artists = np.empty(len(uris), dtype=np.int32)
        for i, u in enumerate(uris):
            artists[i] = lookup.setdefault(u, len(lookup))
        counts = frame["n_playlists"].to_numpy(dtype=np.float64)
        tail_cut = np.percentile(counts, tail_percentile)
        head_cut = np.percentile(counts, head_percentile)
        tail = counts <= tail_cut
        return cls(
            n_tracks=len(frame),
            artists=artists,
            n_artists=len(lookup),
            play_counts=counts,
            tail_mask=tail,
            head_mask=counts >= head_cut,
            tail_share_of_catalog=float(tail.mean()),
        )


@dataclass
class ExposureReport:
    n_recommendations: int
    track_coverage: float
    artist_coverage: float
    track_gini: float
    artist_gini: float
    tail_share: float
    head_share: float
    tail_lift: float  # tail share relative to the tail's share of the catalog
    mean_log_popularity: float
    top1pct_artist_share: float


def measure(indices: np.ndarray, facts: CatalogFacts, depth: int | None = None) -> ExposureReport:
    """Aggregate exposure over an (n_queries, depth) recommendation block."""
    block = indices if depth is None else indices[:, :depth]
    flat = block.reshape(-1)
    flat = flat[flat >= 0].astype(np.int64)
    if flat.size == 0:
        return ExposureReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    track_counts = np.bincount(flat, minlength=facts.n_tracks).astype(np.float64)
    artist_counts = np.bincount(facts.artists[flat], minlength=facts.n_artists).astype(np.float64)

    tail = float(facts.tail_mask[flat].mean())
    # Sorting once and slicing beats np.percentile on the counts vector, which
    # would answer a different question -- we want the share of *exposure* held
    # by the top 1% of artists, not the count at that percentile.
    top_n = max(1, int(round(0.01 * facts.n_artists)))
    top_share = float(np.sort(artist_counts)[-top_n:].sum() / artist_counts.sum())

    return ExposureReport(
        n_recommendations=int(flat.size),
        track_coverage=float((track_counts > 0).sum() / facts.n_tracks),
        artist_coverage=float((artist_counts > 0).sum() / facts.n_artists),
        track_gini=gini(track_counts),
        artist_gini=gini(artist_counts),
        tail_share=tail,
        head_share=float(facts.head_mask[flat].mean()),
        tail_lift=float(tail / facts.tail_share_of_catalog) if facts.tail_share_of_catalog else 0.0,
        mean_log_popularity=float(np.log1p(facts.play_counts[flat]).mean()),
        top1pct_artist_share=top_share,
    )
