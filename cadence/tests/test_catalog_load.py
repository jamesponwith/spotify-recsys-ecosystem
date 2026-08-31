"""The served tag matrix must not include held-out playlists.

data/processed/tags.npz is built over *every* playlist, including the ones held
out for evaluation. Serving it means a held-out "rock" challenge can credit a
track for "rock" via the very playlist being scored against. `train` therefore
persists the holdout-free rebuild as ``tags_train.npz`` and `Catalog.load`
prefers it, exactly mirroring ``train_interactions.npz``.

Hermetic: builds a tiny processed dir from scratch, trains real (tiny) spaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cadence.catalog import Catalog
from cadence.config import EmbeddingConfig, TagConfig
from cadence.models.train import AUDIO_FEATURE_COLS, train
from cadence.text import title_tokens

N_TRACKS = 30
VOCAB = ["rock", "chill", "indie", "summer", "gym", "sad"]

# Playlist titles and members. Rows 16-19 are the evaluation holdout; tracks
# 28 and 29 appear *only* there, so any "rock" credit they carry is
# self-supplied by the playlists they would be scored against.
PLAYLISTS: list[tuple[str, list[int]]] = (
    [("chill beats", [3 * i, 3 * i + 1, 3 * i + 2]) for i in range(5)]
    + [("indie summer", [15 + i, 16 + i, 17 + i]) for i in range(5)]
    + [("rock riffs", [20 + i, 21 + i, 22 + i]) for i in range(4)]
    + [("gym rock", [i, i + 5, 24 + i]) for i in range(2)]
    + [("rock anthems", [5 + i, 10 + i, 28, 29]) for i in range(4)]
)
HOLDOUT_ROWS = np.arange(16, 20)


def _tag_counts(playlists: list[tuple[str, list[int]]]) -> sparse.csr_matrix:
    """Track x tag counts over the given playlists, as data.build constructs them."""
    tag_to_col = {t: i for i, t in enumerate(VOCAB)}
    m = np.zeros((N_TRACKS, len(VOCAB)), dtype=np.float32)
    for name, tracks in playlists:
        for tok in title_tokens(name):
            col = tag_to_col.get(tok)
            if col is not None:
                m[tracks, col] += 1.0
    return sparse.csr_matrix(m)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    processed = tmp_path_factory.mktemp("processed")
    artifacts = tmp_path_factory.mktemp("artifacts")

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "name": [f"song {i}" for i in range(N_TRACKS)],
            "artist": [f"artist {i % 5}" for i in range(N_TRACKS)],
            "album": [f"album {i % 7}" for i in range(N_TRACKS)],
            "genre": ["rock|indie" if i % 2 else None for i in range(N_TRACKS)],
            "has_audio": [True] * N_TRACKS,
        }
        | {c: rng.random(N_TRACKS).astype(np.float32) for c in AUDIO_FEATURE_COLS}
    )
    frame.to_parquet(processed / "tracks.parquet")

    pd.DataFrame({"name": [name for name, _ in PLAYLISTS]}).to_parquet(
        processed / "playlists.parquet"
    )
    rows = [p for p, (_, tracks) in enumerate(PLAYLISTS) for _ in tracks]
    cols = [t for _, tracks in PLAYLISTS for t in tracks]
    interactions = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(PLAYLISTS), N_TRACKS),
    )
    sparse.save_npz(processed / "interactions.npz", interactions)
    sparse.save_npz(processed / "tags.npz", _tag_counts(PLAYLISTS))
    (processed / "tag_vocab.json").write_text(json.dumps(VOCAB))
    (processed / "build_meta.json").write_text(json.dumps({"n_playlists": len(PLAYLISTS)}))

    train(
        processed_dir=processed,
        out_dir=artifacts,
        emb_cfg=EmbeddingConfig(dim=8),
        tag_cfg=TagConfig(dim=4),
        holdout_rows=HOLDOUT_ROWS,
        verbose=False,
    )
    return processed, artifacts


def test_train_persists_holdout_free_tags(trained: tuple[Path, Path]) -> None:
    processed, artifacts = trained
    assert (artifacts / "tags_train.npz").exists()

    got = sparse.load_npz(artifacts / "tags_train.npz").tocsr()
    keep = np.ones(len(PLAYLISTS), dtype=bool)
    keep[HOLDOUT_ROWS] = False
    want = _tag_counts([p for i, p in enumerate(PLAYLISTS) if keep[i]])
    assert got.shape == want.shape
    assert (got != want).nnz == 0

    # Tracks that only ever appeared on held-out "rock anthems" playlists must
    # carry no rock credit, while the full matrix says they do.
    full = sparse.load_npz(processed / "tags.npz").tocsr()
    rock = VOCAB.index("rock")
    assert full[28, rock] > 0 and full[29, rock] > 0
    assert got[28, rock] == 0 and got[29, rock] == 0


def test_catalog_load_prefers_tags_train(trained: tuple[Path, Path]) -> None:
    processed, artifacts = trained
    catalog = Catalog.load(processed_dir=processed, artifacts_dir=artifacts)
    want = sparse.load_npz(artifacts / "tags_train.npz").tocsr()
    assert (catalog.tag_matrix != want).nnz == 0


def test_catalog_load_falls_back_to_processed_tags(trained: tuple[Path, Path]) -> None:
    processed, artifacts = trained
    path = artifacts / "tags_train.npz"
    hidden = path.with_suffix(".hidden")
    path.rename(hidden)
    try:
        catalog = Catalog.load(processed_dir=processed, artifacts_dir=artifacts)
    finally:
        hidden.rename(path)
    want = sparse.load_npz(processed / "tags.npz").tocsr()
    assert (catalog.tag_matrix != want).nnz == 0
