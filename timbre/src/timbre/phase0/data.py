"""Load Cadence's artifacts and assemble the Phase 0 experiment.

Nothing here is rebuilt. Phase 0's entire claim to being cheap rests on both
halves already existing: Spotify's descriptors in Cadence's catalog, and the
128-d folksonomy embedding in its trained spaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from cadence.text import title_tokens
from scipy import sparse

from ..config import CADENCE_ARTIFACTS, CADENCE_PROCESSED, Phase0Config
from .features import encode, standardizer


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


@dataclass
class Queries:
    """One query per usable holdout playlist."""

    vectors: np.ndarray  # (n_queries, tag_dim) mean of the title's tag vectors
    relevant: list[np.ndarray]  # test-split track indices genuinely in that playlist
    titles: list[str]
    n_titles_no_vocab: int
    n_titles_no_relevant: int

    def __len__(self) -> int:
        return len(self.titles)


@dataclass
class Phase0Data:
    x: np.ndarray  # (n_tracks, n_features) standardised design matrix
    y: np.ndarray  # (n_tracks, tag_dim) L2-normalised target embedding
    y_raw: np.ndarray  # untouched tag_track_vectors, for the oracle system
    usable: np.ndarray  # bool mask: has audio AND a non-zero tag embedding
    train_idx: np.ndarray
    test_idx: np.ndarray
    queries: Queries
    feature_names: list[str]
    mu: np.ndarray  # standardiser fitted on the training split only
    sigma: np.ndarray
    n_tracks: int
    excluded_no_audio: int
    excluded_zero_embedding: int


def load(cfg: Phase0Config | None = None) -> Phase0Data:
    cfg = cfg or Phase0Config()
    rng = np.random.default_rng(cfg.seed)

    frame = pd.read_parquet(CADENCE_PROCESSED / "tracks.parquet")
    spaces = np.load(CADENCE_ARTIFACTS / "spaces.npz")
    y_raw = spaces["tag_track_vectors"].astype(np.float32)
    vocab: list[str] = json.loads((CADENCE_ARTIFACTS / "tag_vocab.json").read_text())
    tag_vectors = spaces["tag_vectors"].astype(np.float32)
    n_tracks = len(frame)

    # --- exclusions -------------------------------------------------------
    # Two independent reasons a track cannot take part, kept separate so the
    # report can state both counts rather than one merged number.
    has_audio = frame["has_audio"].to_numpy(dtype=bool)
    finite = np.isfinite(frame[["danceability", "tempo", "loudness"]].to_numpy()).all(axis=1)
    audio_ok = has_audio & finite
    embedding_ok = np.linalg.norm(y_raw, axis=1) > 0
    usable = audio_ok & embedding_ok

    # --- features and targets --------------------------------------------
    x_all, names = encode(frame)
    x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
    y = l2_normalize(y_raw)

    # --- split by track ----------------------------------------------------
    # Only usable tracks are split. Unusable ones stay in the catalog with their
    # true embedding under every system, so they neither help nor hurt anyone.
    pool = np.flatnonzero(usable)
    perm = rng.permutation(pool.size)
    n_test = int(round(pool.size * cfg.test_fraction))
    test_idx = np.sort(pool[perm[:n_test]])
    train_idx = np.sort(pool[perm[n_test:]])

    mu, sigma = standardizer(x_all[train_idx])
    x = ((x_all - mu) / sigma).astype(np.float32)

    queries = _build_queries(vocab, tag_vectors, test_idx, n_tracks)

    return Phase0Data(
        x=x,
        y=y,
        y_raw=y_raw,
        usable=usable,
        train_idx=train_idx,
        test_idx=test_idx,
        queries=queries,
        feature_names=names,
        mu=mu,
        sigma=sigma,
        n_tracks=n_tracks,
        excluded_no_audio=int((~audio_ok).sum()),
        excluded_zero_embedding=int((~embedding_ok).sum()),
    )


def _build_queries(
    vocab: list[str], tag_vectors: np.ndarray, test_idx: np.ndarray, n_tracks: int
) -> Queries:
    """Turn each holdout playlist title into a query vector plus a relevant set.

    Holdout playlists were excluded from Cadence's tag factorisation, so their
    titles never shaped the space any system is scored in. That matters most for
    ``oracle``: without it, the ceiling would be reading its own answer key.

    Query construction mirrors ``cadence.retrieval.channels.tag_channel`` exactly
    -- title tokens to tag columns to the mean of their vectors. Reimplementing
    the tokeniser here instead of importing it would let the two drift, and any
    drift is silent recall loss.
    """
    playlists = pd.read_parquet(CADENCE_PROCESSED / "playlists.parquet")
    interactions = sparse.load_npz(CADENCE_PROCESSED / "interactions.npz").tocsr()
    splits = json.loads((CADENCE_PROCESSED / "splits.json").read_text())
    holdout_rows = np.asarray(splits["holdout_rows"], dtype=np.int64)

    is_test = np.zeros(n_tracks, dtype=bool)
    is_test[test_idx] = True
    tag_to_col = {t: i for i, t in enumerate(vocab)}
    names = playlists["name"].to_numpy()

    vectors: list[np.ndarray] = []
    relevant: list[np.ndarray] = []
    titles: list[str] = []
    no_vocab = 0
    no_relevant = 0

    for row in holdout_rows:
        title = str(names[row])
        cols = [tag_to_col[t] for t in title_tokens(title) if t in tag_to_col]
        if not cols:
            # Unanswerable: the tag channel has nothing to embed. Scoring it as a
            # miss would understate every system equally, but it would still be
            # measuring the tokeniser rather than the thesis.
            no_vocab += 1
            continue
        tracks = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
        rel = tracks[is_test[tracks]].astype(np.int64)
        if rel.size == 0:
            no_relevant += 1
            continue
        vectors.append(tag_vectors[np.asarray(cols, dtype=np.int64)].mean(axis=0))
        relevant.append(np.unique(rel))
        titles.append(title)

    return Queries(
        vectors=np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 1), np.float32),
        relevant=relevant,
        titles=titles,
        n_titles_no_vocab=no_vocab,
        n_titles_no_relevant=no_relevant,
    )
