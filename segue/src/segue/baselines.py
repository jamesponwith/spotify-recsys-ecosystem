"""The systems Segue has to beat.

`centroid` is the one that matters: it is exactly what Cadence's collaborative
channel does today -- sum the seed embeddings, search the neighbourhood -- so the
gap between it and Segue is the entire value of modelling order.

`last` is included because it is the cheap way to be order-aware, and a sequence
model that cannot beat "just look at the most recent track" has not earned its
extra parameters.
"""

from __future__ import annotations

import numpy as np

from .features import l2_normalize


def _rank(scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    s = scores.astype(np.float32, copy=True)
    s[exclude] = -np.inf
    k = int(min(k, np.isfinite(s).sum()))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    top = np.argpartition(-s, k - 1)[:k]
    return top[np.argsort(-s[top], kind="stable")].astype(np.int64)


def by_vector(vector: np.ndarray, matrix: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    q = l2_normalize(np.asarray(vector, dtype=np.float32).ravel())
    if not np.any(q):
        return np.zeros(0, dtype=np.int64)
    return _rank(matrix @ q, exclude, k)


def popularity(popularity_scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    return _rank(popularity_scores, exclude, k)


def last_track(seeds: np.ndarray, matrix: np.ndarray, raw: np.ndarray, k: int) -> np.ndarray:
    if seeds.size == 0:
        return np.zeros(0, dtype=np.int64)
    return by_vector(raw[seeds[-1]], matrix, seeds, k)


def centroid(seeds: np.ndarray, matrix: np.ndarray, raw: np.ndarray, k: int) -> np.ndarray:
    """Cadence's collaborative channel, reproduced: an order-free sum of seeds."""
    if seeds.size == 0:
        return np.zeros(0, dtype=np.int64)
    return by_vector(raw[seeds].sum(axis=0), matrix, seeds, k)
