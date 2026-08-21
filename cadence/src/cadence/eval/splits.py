"""Held-out challenge splits in the shape of the RecSys Challenge 2018 task.

For each evaluation playlist we expose the title plus the genuinely first ``k``
tracks (see ``_load_order``: the interaction matrix cannot supply them) and
withhold the rest. Sweeping ``k`` from 0 upward is the whole point:

* ``k = 0`` is the pure cold-start natural-language task — title only, nothing
  for collaborative filtering to work with. This is the case this project is
  really about, and the one where the folksonomy channel has to carry the load.
* ``k >= 1`` progressively hands the CF channel real signal, which is how you
  see the channels trade off instead of guessing.

The split is drawn once from a fixed seed and written to disk so that training
(which must exclude these playlists) and evaluation cannot disagree about which
rows are held out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import DATA_PROCESSED, SEED

SEED_COUNTS = (0, 1, 5, 10, 25)


@dataclass
class Challenge:
    row: int
    pid: int
    title: str
    seed_tracks: list[int]
    held_out: list[int]
    n_seed: int


def _load_order(processed_dir: Path, interactions: sparse.csr_matrix) -> list[np.ndarray]:
    """Per-playlist track sequence, in the order a human arranged it.

    This exists because the interaction matrix cannot supply it. SciPy keeps CSR
    column indices sorted within a row, so ``interactions.indices[start:stop]``
    comes back in ascending track-id order. Slicing the first k of that yields
    *the k lowest-numbered tracks in the playlist*, which is what this module used
    to do while its docstring promised the first k. Track ids are assigned in
    corpus-wide first-seen order during the build, so low id correlates with
    popular-and-early: the old seeds were biased toward the catalog's head rather
    than being a neutral prefix.

    ``order.npz`` is written by the build from the same pass that fills the
    matrix, so the two cannot disagree about which tracks a playlist contains.
    """
    path = processed_dir / "order.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Playlist order is not recoverable from "
            "interactions.npz -- rebuild the catalog with `cadence build` to "
            "emit it. Refusing to fall back to track-id order, which silently "
            "produces head-biased seeds."
        )
    z = np.load(path)
    tracks, offsets = z["tracks"], z["offsets"]
    if offsets.size - 1 != interactions.shape[0]:
        raise ValueError(
            f"order.npz has {offsets.size - 1} playlists but interactions.npz has "
            f"{interactions.shape[0]}; rebuild both with `cadence build`."
        )
    return [tracks[offsets[r] : offsets[r + 1]].astype(np.int64) for r in range(offsets.size - 1)]


def make_splits(
    processed_dir: Path = DATA_PROCESSED,
    n_eval: int = 2000,
    seed: int = SEED,
    seed_counts: tuple[int, ...] = SEED_COUNTS,
    min_held_out: int = 5,
) -> dict:
    """Draw the evaluation playlists and materialise one challenge per seed count."""
    playlists = pd.read_parquet(processed_dir / "playlists.parquet")
    interactions = sparse.load_npz(processed_dir / "interactions.npz").tocsr()
    order = _load_order(processed_dir, interactions)
    rng = np.random.default_rng(seed)

    lengths = np.diff(interactions.indptr)
    # Only playlists long enough to still have a meaningful target after the
    # largest seed prefix is removed are eligible.
    eligible = np.flatnonzero(lengths >= max(seed_counts) + min_held_out)
    # Require a non-empty title: a title-only (k=0) challenge with no title is
    # unanswerable, and silently scoring 0 on it would understate the system.
    titles = playlists["name"].to_numpy()
    eligible = np.array([r for r in eligible if str(titles[r]).strip()], dtype=np.int64)
    if eligible.size < n_eval:
        n_eval = int(eligible.size)
    chosen = rng.choice(eligible, size=n_eval, replace=False)
    chosen.sort()

    challenges: dict[int, list[Challenge]] = {}
    for k in seed_counts:
        items: list[Challenge] = []
        for row in chosen:
            trs = order[row]
            seed_tracks = trs[:k].tolist()
            held = trs[k:].tolist()
            if len(held) < min_held_out:
                continue
            items.append(
                Challenge(
                    row=int(row),
                    pid=int(playlists["pid"].iloc[row]),
                    title=str(titles[row]),
                    seed_tracks=seed_tracks,
                    held_out=held,
                    n_seed=k,
                )
            )
        challenges[k] = items

    out = {
        "seed": seed,
        "n_eval": int(n_eval),
        "holdout_rows": chosen.tolist(),
        "seed_counts": list(seed_counts),
        "counts": {str(k): len(v) for k, v in challenges.items()},
    }
    (processed_dir / "splits.json").write_text(json.dumps(out))
    payload = {str(k): [asdict(c) for c in v] for k, v in challenges.items()}
    (processed_dir / "challenges.json").write_text(json.dumps(payload))
    return out


def load_splits(
    processed_dir: Path = DATA_PROCESSED,
) -> tuple[np.ndarray, dict[int, list[Challenge]]]:
    meta = json.loads((processed_dir / "splits.json").read_text())
    raw = json.loads((processed_dir / "challenges.json").read_text())
    challenges = {int(k): [Challenge(**c) for c in v] for k, v in raw.items()}
    return np.asarray(meta["holdout_rows"], dtype=np.int64), challenges
