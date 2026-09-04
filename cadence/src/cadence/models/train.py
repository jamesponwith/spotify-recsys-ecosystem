"""Train every retrieval space and persist them as a single artifact bundle.

Three learned spaces plus one derived table:

collaborative  SPPMI(playlist x track) -> track vectors. Encodes "these songs
               get put on the same playlists", i.e. classic item-item CF.
tag            SPPMI(track x title-token) -> track *and* tag vectors in one
               space. This is the natural-language bridge: a query phrase is
               embedded through its tag vectors and compared directly to tracks.
lexical        TF-IDF over track/artist/album/genre strings. Handles exact
               entity mentions ("play some Bon Iver") that a latent space blurs.
audio          Standardised Spotify audio features + per-column availability,
               used for mood/tempo targeting and for sequencing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import ARTIFACTS, DATA_PROCESSED, SEED, EmbeddingConfig, TagConfig
from .factorize import factorize

# Feature columns used for mood/energy targeting. `key`/`mode` are excluded:
# they are categorical (pitch class), and Euclidean distance over them is
# meaningless. They are used by the sequencer instead, via the Camelot wheel.
AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def _lexical_corpus(catalog: pd.DataFrame) -> list[str]:
    genre = catalog["genre"].fillna("").str.replace("|", " ", regex=False)
    return (
        catalog["name"].fillna("")
        + " "
        + catalog["artist"].fillna("")
        + " "
        + catalog["album"].fillna("")
        + " "
        + genre
    ).tolist()


def train(
    processed_dir: Path = DATA_PROCESSED,
    out_dir: Path = ARTIFACTS,
    emb_cfg: EmbeddingConfig | None = None,
    tag_cfg: TagConfig | None = None,
    *,
    holdout_rows: np.ndarray | None = None,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Fit all spaces.

    ``holdout_rows`` removes evaluation playlists from the training matrices.
    Without it, every reported retrieval metric is contaminated: the model would
    have already seen the exact playlist it is being asked to reconstruct.
    """
    emb_cfg = emb_cfg or EmbeddingConfig()
    tag_cfg = tag_cfg or TagConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    catalog = pd.read_parquet(processed_dir / "tracks.parquet")
    interactions = sparse.load_npz(processed_dir / "interactions.npz").tocsr()
    tags = sparse.load_npz(processed_dir / "tags.npz").tocsr()
    vocab: list[str] = json.loads((processed_dir / "tag_vocab.json").read_text())

    n_playlists, n_tracks = interactions.shape
    if holdout_rows is not None and len(holdout_rows) > 0:
        keep = np.ones(n_playlists, dtype=bool)
        keep[np.asarray(holdout_rows, dtype=np.int64)] = False
        train_interactions = interactions[keep]
        if verbose:
            print(f"holding out {int((~keep).sum()):,} playlists from training")
    else:
        train_interactions = interactions

    # ---- collaborative space -------------------------------------------
    t = time.perf_counter()
    collab = factorize(
        train_interactions.T.tocsr(),
        emb_cfg.dim,
        shift_k=emb_cfg.shift_k,
        popularity_damping=emb_cfg.popularity_damping,
        seed=seed,
    )
    collab_vectors = collab.rows  # (n_tracks, dim)
    if verbose:
        print(
            f"collaborative: {collab_vectors.shape} energy={collab.explained_energy:.3f} "
            f"({time.perf_counter() - t:.1f}s)"
        )

    # ---- tag space ------------------------------------------------------
    # Rebuild the tag matrix from *training* playlists only, for the same
    # contamination reason as above.
    t = time.perf_counter()
    if holdout_rows is not None and len(holdout_rows) > 0:
        tag_matrix = _rebuild_tags(processed_dir, keep, n_tracks, len(vocab))
    else:
        tag_matrix = tags
    tag_fact = factorize(
        tag_matrix,
        tag_cfg.dim,
        shift_k=tag_cfg.shift_k,
        popularity_damping=1.0,
        seed=seed,
    )
    if verbose:
        print(
            f"tag: tracks={tag_fact.rows.shape} tags={tag_fact.cols.shape} "
            f"energy={tag_fact.explained_energy:.3f} ({time.perf_counter() - t:.1f}s)"
        )

    # ---- lexical space --------------------------------------------------
    t = time.perf_counter()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        token_pattern=r"[a-z0-9]+",
        ngram_range=(1, 2),
        min_df=2,
        max_features=400_000,
        sublinear_tf=True,
    )
    lexical = vectorizer.fit_transform(_lexical_corpus(catalog)).astype(np.float32)
    if verbose:
        print(f"lexical: {lexical.shape} nnz={lexical.nnz:,} ({time.perf_counter() - t:.1f}s)")

    # ---- audio space ----------------------------------------------------
    audio_raw = catalog[AUDIO_FEATURE_COLS].to_numpy(dtype=np.float32)
    has_audio = catalog["has_audio"].to_numpy()
    finite = np.isfinite(audio_raw).all(axis=1) & has_audio
    mu = np.nanmean(audio_raw[finite], axis=0)
    sigma = np.nanstd(audio_raw[finite], axis=0)
    sigma[sigma < 1e-6] = 1.0
    audio_z = np.where(np.isfinite(audio_raw), (audio_raw - mu) / sigma, 0.0).astype(np.float32)
    audio_z[~finite] = 0.0

    # ---- popularity prior ------------------------------------------------
    pop_counts = np.asarray(train_interactions.sum(axis=0)).ravel().astype(np.float32)
    log_pop = np.log1p(pop_counts)
    popularity = (log_pop / max(log_pop.max(), 1e-9)).astype(np.float32)

    np.savez_compressed(
        out_dir / "spaces.npz",
        collab_vectors=collab_vectors,
        collab_singular=collab.singular_values,
        tag_track_vectors=tag_fact.rows,
        tag_vectors=tag_fact.cols,
        tag_singular=tag_fact.singular_values,
        audio_z=audio_z,
        audio_mu=mu.astype(np.float32),
        audio_sigma=sigma.astype(np.float32),
        audio_valid=finite,
        popularity=popularity,
        pop_counts=pop_counts,
    )
    sparse.save_npz(out_dir / "lexical.npz", lexical)
    # Served co-occurrence must exclude evaluation playlists, or the neighbourhood
    # channel gets to read the answer key at eval time.
    sparse.save_npz(out_dir / "train_interactions.npz", train_interactions.tocsr())
    # Same for served tag counts: without this, a held-out "rock" challenge can
    # credit a track for "rock" via the very playlist being scored against.
    sparse.save_npz(out_dir / "tags_train.npz", tag_matrix.tocsr())
    import pickle

    (out_dir / "lexical_vectorizer.pkl").write_bytes(pickle.dumps(vectorizer))
    (out_dir / "tag_vocab.json").write_text(json.dumps(vocab))

    meta = {
        "n_tracks": int(n_tracks),
        "n_train_playlists": int(train_interactions.shape[0]),
        "n_holdout_playlists": int(len(holdout_rows)) if holdout_rows is not None else 0,
        "collab_dim": int(collab_vectors.shape[1]),
        "collab_explained_energy": round(collab.explained_energy, 4),
        "tag_dim": int(tag_fact.rows.shape[1]),
        "tag_explained_energy": round(tag_fact.explained_energy, 4),
        "n_tags": len(vocab),
        "lexical_features": int(lexical.shape[1]),
        "audio_valid_fraction": round(float(finite.mean()), 4),
        "train_seconds": round(time.perf_counter() - t0, 1),
        "seed": seed,
        "embedding_config": dict(emb_cfg.__dict__),
        "tag_config": dict(tag_cfg.__dict__),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(json.dumps(meta, indent=2))
    return meta


def _rebuild_tags(
    processed_dir: Path, keep: np.ndarray, n_tracks: int, n_tags: int
) -> sparse.csr_matrix:
    """Recompute track x tag counts using only training playlists."""
    from ..text import title_tokens

    playlists = pd.read_parquet(processed_dir / "playlists.parquet")
    interactions = sparse.load_npz(processed_dir / "interactions.npz").tocsr()
    vocab: list[str] = json.loads((processed_dir / "tag_vocab.json").read_text())
    tag_to_col = {t: i for i, t in enumerate(vocab)}

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    names = playlists["name"].tolist()
    for prow in np.flatnonzero(keep):
        toks = title_tokens(names[prow])
        c = [tag_to_col[t] for t in toks if t in tag_to_col]
        if not c:
            continue
        trs = interactions.indices[interactions.indptr[prow] : interactions.indptr[prow + 1]]
        rows.append(np.tile(trs, len(c)))
        cols.append(np.repeat(np.asarray(c, dtype=np.int32), len(trs)))
    if not rows:
        return sparse.csr_matrix((n_tracks, n_tags), dtype=np.float32)
    r = np.concatenate(rows)
    cc = np.concatenate(cols)
    m = sparse.coo_matrix(
        (np.ones(r.size, dtype=np.float32), (r, cc)), shape=(n_tracks, n_tags)
    ).tocsr()
    m.sum_duplicates()
    return m
