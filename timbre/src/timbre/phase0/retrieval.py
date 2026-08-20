"""Batched retrieval scoring that mirrors Cadence's serving path exactly.

``cadence.retrieval.ann.DenseIndex`` scores one query at a time; 2,000 queries
across five systems is 10,000 full-catalog passes, so this does the same
arithmetic as one blocked matmul. Two behaviours must be preserved or the gate
measures the wrong thing:

* rows are L2-normalised, so the score is cosine and not a dot product;
* all-zero rows are forced to ``-inf`` rather than tying at 0.0.

The second one is the trap. A predicted embedding is never exactly zero, so a
naive harness would let ``content`` compete for candidate slots that ``oracle``
is structurally barred from -- inflating the very ratio Gate 0 divides by.
"""

from __future__ import annotations

import numpy as np

from .data import l2_normalize


class CatalogIndex:
    """Cosine index over a track-embedding matrix, with dead rows masked out."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.live = np.linalg.norm(vectors, axis=1) > 0
        self.vectors = np.ascontiguousarray(l2_normalize(vectors.astype(np.float32)))

    def top_k_hits(
        self, queries: np.ndarray, relevant: list[np.ndarray], k: int, block: int
    ) -> np.ndarray:
        """Per-query recall@k: ``|top_k ∩ relevant| / |relevant|``."""
        q = np.ascontiguousarray(l2_normalize(queries.astype(np.float32)))
        dead = ~self.live
        out = np.zeros(len(relevant), dtype=np.float64)

        for start in range(0, q.shape[0], block):
            stop = min(start + block, q.shape[0])
            scores = self.vectors @ q[start:stop].T  # (n_tracks, n_block)
            scores[dead, :] = -np.inf
            # argpartition over the track axis is O(n) per query against the
            # O(n log n) of a full sort, and only the identity of the top k
            # matters -- their internal order does not.
            part = np.argpartition(-scores, k - 1, axis=0)[:k]
            for j in range(stop - start):
                rel = relevant[start + j]
                hits = np.intersect1d(part[:, j], rel, assume_unique=False).size
                out[start + j] = hits / rel.size
        return out


def substitute(base: np.ndarray, rows: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    """Copy ``base`` with the cold rows overwritten.

    Only the test split is replaced. Every other track keeps its true embedding
    under every system, which is what makes this a cold-start simulation rather
    than a wholesale catalog swap: the rest of the catalog stays known.
    """
    out = base.copy()
    out[rows] = replacement
    return out
