"""Exact cosine search over a dense embedding matrix.

At catalog scale (10^5-10^6 x 10^2 dims) a blocked float32 matmul answers a
query in single-digit milliseconds, which is well inside the latency budget.
Exact search also removes recall/precision ambiguity from the evaluation: any
number the eval harness reports is a property of the *model*, not of an ANN
index's approximation error. Swapping in HNSW later is a drop-in change behind
this interface.
"""

from __future__ import annotations

import numpy as np

from ..models.factorize import l2_normalize


class DenseIndex:
    """Cosine-similarity index over unit-normalised row vectors."""

    def __init__(self, vectors: np.ndarray, *, block: int = 65536) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        self.vectors = np.ascontiguousarray(l2_normalize(vectors.astype(np.float32)))
        self.block = block
        # Rows that are all-zero carry no signal; excluding them stops them from
        # tying at score 0.0 and crowding out real results.
        self.live = np.linalg.norm(vectors, axis=1) > 0

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def scores(self, query: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Cosine score of every row against a single query vector."""
        q = np.asarray(query, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return np.zeros(len(self), dtype=np.float32)
        q = q / norm
        out = np.empty(len(self), dtype=np.float32)
        for start in range(0, len(self), self.block):
            stop = min(start + self.block, len(self))
            out[start:stop] = self.vectors[start:stop] @ q
        out[~self.live] = -np.inf
        if mask is not None:
            out[~mask] = -np.inf
        return out

    def search(
        self, query: np.ndarray, k: int, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (indices, scores), highest first."""
        s = self.scores(query, mask=mask)
        k = int(min(k, np.isfinite(s).sum()))
        if k <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]
