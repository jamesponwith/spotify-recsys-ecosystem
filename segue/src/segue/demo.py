"""Continue a real playlist and show what each system would have played next.

The demo is deliberately built on *held-out* playlists rather than typed-in seeds,
because a continuation has a ground truth only when the rest of the playlist
actually exists. Every row it prints can be checked against what a human really
put next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baselines import by_vector, centroid, last_track, popularity
from .config import ARTIFACTS, CADENCE_ARTIFACTS, CADENCE_PROCESSED, SegueConfig
from .evaluate import holdout_rows, make_challenges
from .features import l2_normalize
from .model import SegueModel
from .sequences import Sequences


@dataclass
class Bundle:
    """Everything the demo and the evaluation both need, loaded once."""

    sequences: Sequences
    embeddings: np.ndarray
    matrix: np.ndarray
    popularity: np.ndarray
    artists: np.ndarray
    names: np.ndarray
    artist_names: np.ndarray
    titles: np.ndarray

    @classmethod
    def load(cls) -> Bundle:
        spaces = np.load(CADENCE_ARTIFACTS / "spaces.npz")
        emb = spaces["collab_vectors"].astype(np.float32)
        frame = pd.read_parquet(CADENCE_PROCESSED / "tracks.parquet")
        playlists = pd.read_parquet(CADENCE_PROCESSED / "playlists.parquet")
        uris = frame["artist_uri"].to_numpy(dtype=object)
        lookup: dict[str, int] = {}
        artists = np.empty(len(uris), dtype=np.int32)
        for i, u in enumerate(uris):
            artists[i] = lookup.setdefault(u, len(lookup))
        return cls(
            sequences=Sequences.load(),
            embeddings=emb,
            matrix=np.ascontiguousarray(l2_normalize(emb)),
            popularity=spaces["popularity"].astype(np.float32),
            artists=artists,
            names=frame["name"].to_numpy(dtype=object),
            artist_names=frame["artist"].to_numpy(dtype=object),
            titles=playlists["name"].to_numpy(dtype=object),
        )

    def label(self, i: int) -> str:
        return f"{self.names[i]} — {self.artist_names[i]}"


def continue_playlist(
    bundle: Bundle, model: SegueModel, *, k: int = 5, n: int = 10, index: int = 0
) -> dict[str, Any]:
    """Take a held-out playlist, show its first k tracks, predict the next n."""
    cfg = SegueConfig()
    challenges = make_challenges(bundle.sequences, holdout_rows(cfg), k)
    ch = challenges[index % len(challenges)]
    seeds = ch["seeds"].astype(np.int64)
    truth = ch["held_out"]

    row = ch["row"]
    seq = next(
        bundle.sequences[i]
        for i in range(len(bundle.sequences))
        if int(bundle.sequences.rows[i]) == row
    )
    actual_next = [int(t) for t in seq[k : k + n]]

    preds = {
        "popularity": popularity(bundle.popularity, seeds, n),
        "last": last_track(seeds, bundle.matrix, bundle.embeddings, n),
        "centroid": centroid(seeds, bundle.matrix, bundle.embeddings, n),
        "segue": by_vector(model.predict([seeds], bundle.embeddings)[0], bundle.matrix, seeds, n),
    }
    return {
        "title": str(bundle.titles[row]),
        "seed_count": k,
        "seeds": [{"index": int(t), "label": bundle.label(int(t))} for t in seeds],
        "actual_next": [{"index": t, "label": bundle.label(t)} for t in actual_next],
        "predictions": {
            name: [
                {"index": int(t), "label": bundle.label(int(t)), "hit": int(t) in truth} for t in p
            ]
            for name, p in preds.items()
        },
        "n_held_out": len(truth),
    }


def build_demo(indices: tuple[int, ...] = (0, 1, 2), k: int = 5, n: int = 10) -> dict[str, Any]:
    bundle = Bundle.load()
    model = SegueModel.load()
    cases = [continue_playlist(bundle, model, k=k, n=n, index=i) for i in indices]
    out = {"cases": cases, "k": k, "n": n}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "demo.json").write_text(json.dumps(out, indent=2))
    return out
