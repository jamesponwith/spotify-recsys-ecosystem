"""Per-playlist track sequence, in the order a human arranged it.

This exists because the interaction matrix cannot supply it. SciPy keeps CSR
column indices sorted within a row, so ``interactions.indices[start:stop]``
comes back in ascending track-id order; slicing the first *k* of that yields
*the k lowest-numbered tracks in the playlist*, not its first k. Track ids are
assigned in corpus-wide first-seen order during the build, so low id correlates
with popular-and-early -- the resulting seeds are biased toward the catalog's
head rather than being a neutral prefix.

It lives here rather than in ``eval/splits.py``, where it started, because two
callers need it and they sit on opposite sides of the package: the evaluation
harness builds *scoring* seeds and the reranker builds *training* seeds. When it
was private to one of them, the 2026-08-21 fix reached the evaluation path and
left the training path on the old behaviour for three weeks -- the reranker,
which roughly doubles k=0 R-precision, supervised on prefixes that did not mean
what its own harness meant by the word.

``order.npz`` is written by the build from the same pass that fills the matrix,
so the two cannot disagree about which tracks a playlist contains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from ..config import DATA_PROCESSED


def load_order(
    processed_dir: Path = DATA_PROCESSED,
    interactions: sparse.csr_matrix | None = None,
) -> list[np.ndarray]:
    """Ordered catalog indices per playlist row.

    Raises rather than falling back to track-id order when the file is absent.
    A silent fallback is how the original defect survived: the code kept working
    and quietly meant something else.
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
    if interactions is not None and offsets.size - 1 != interactions.shape[0]:
        raise ValueError(
            f"order.npz has {offsets.size - 1} playlists but interactions.npz has "
            f"{interactions.shape[0]}; rebuild both with `cadence build`."
        )
    return [tracks[offsets[r] : offsets[r + 1]].astype(np.int64) for r in range(offsets.size - 1)]
