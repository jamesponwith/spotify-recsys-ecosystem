"""Reference systems the main pipeline has to beat.

A retrieval number without a baseline is unreadable. These three bracket the
problem:

popularity   rank by how many playlists a track appears on. Strong on MPD —
             playlists really are popularity-skewed — and the honest floor any
             recommender must clear.
item_knn     classic neighbourhood CF: gather the playlists containing the
             seed tracks, count what else is on them. This is the method the
             system's collaborative channel is a learned version of, so it is
             the comparison that says whether the embedding earned its keep.
lexical      match the playlist title against track/artist/album text only.
             Isolates how much of the task is solved by string matching, which
             is the fair floor for the natural-language claim.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class PopularityBaseline:
    name = "popularity"

    def __init__(self, interactions: sparse.csr_matrix) -> None:
        self.counts = np.asarray(interactions.sum(axis=0)).ravel()
        self.order = np.argsort(-self.counts)

    def recommend(self, seeds: np.ndarray, title: str, k: int = 500) -> np.ndarray:
        if len(seeds) == 0:
            return self.order[:k]
        blocked = {int(s) for s in seeds}
        out = [i for i in self.order[: k + len(blocked)] if int(i) not in blocked]
        return np.asarray(out[:k], dtype=np.int64)


class ItemKNNBaseline:
    name = "item_knn"

    def __init__(self, interactions: sparse.csr_matrix, damping: float = 0.5) -> None:
        self.x = interactions.tocsr()
        self.xt = interactions.T.tocsr()
        counts = np.asarray(interactions.sum(axis=0)).ravel()
        # Popularity damping stops globally huge tracks from winning every
        # neighbourhood vote regardless of the seeds.
        self.norm = np.power(np.maximum(counts, 1.0), damping)
        self.fallback = PopularityBaseline(interactions)

    def recommend(self, seeds: np.ndarray, title: str, k: int = 500) -> np.ndarray:
        if len(seeds) == 0:
            return self.fallback.recommend(seeds, title, k)
        rows = np.unique(np.concatenate([self.xt[int(s)].indices for s in seeds]))
        if rows.size == 0:
            return self.fallback.recommend(seeds, title, k)
        scores = np.asarray(self.x[rows].sum(axis=0)).ravel() / self.norm
        scores[np.asarray(seeds, dtype=np.int64)] = -np.inf
        top = np.argpartition(-scores, min(k, scores.size - 1))[:k]
        return top[np.argsort(-scores[top])].astype(np.int64)


class LexicalBaseline:
    name = "lexical_title"

    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self.fallback_order = np.argsort(-catalog.col("n_playlists"))

    def recommend(self, seeds: np.ndarray, title: str, k: int = 500) -> np.ndarray:
        cat = self.catalog
        q = cat.lexical_vectorizer.transform([title or ""])
        if q.nnz == 0:
            return self.fallback_order[:k]
        scores = np.asarray((cat.lexical @ q.T).todense()).ravel()
        if len(seeds):
            scores[np.asarray(seeds, dtype=np.int64)] = -np.inf
        top = np.argpartition(-scores, min(k, scores.size - 1))[:k]
        return top[np.argsort(-scores[top])].astype(np.int64)
