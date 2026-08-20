"""Turn an ordered prefix into a design vector.

The whole project turns on one question -- does the *order* of a playlist prefix
carry information the bag-of-seeds centroid throws away? -- so the featurisation
is deliberately the smallest thing that can answer it: the last `window` tracks
kept in their own slots, plus the order-free summary the baseline already uses.

If the learned weights on the ordered slots were uninformative, the model would
collapse onto that summary and match the centroid baseline. It does not have to
be told which to prefer; that is the experiment.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def feature_dim(window: int, emb_dim: int) -> int:
    # ordered slots + per-slot presence flags + order-free mean + intercept
    return window * emb_dim + window + emb_dim + 1


def encode(
    prefixes: list[np.ndarray], embeddings: np.ndarray, window: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Encode a batch of prefixes. Row i corresponds to ``prefixes[i]``.

    Slot 0 is the most recent track, slot 1 the one before it, and so on. Short
    prefixes leave later slots zeroed, and the presence flags tell the model that
    a zero means *absent* rather than *an embedding that happens to be zero*.
    """
    emb_dim = embeddings.shape[1]
    dim = feature_dim(window, emb_dim)
    x = np.zeros((len(prefixes), dim), dtype=np.float32) if out is None else out
    x[:] = 0.0
    flag_base = window * emb_dim
    mean_base = flag_base + window

    for i, prefix in enumerate(prefixes):
        if prefix.size == 0:
            x[i, -1] = 1.0
            continue
        recent = prefix[::-1][:window]
        vecs = l2_normalize(embeddings[recent])
        for slot in range(len(recent)):
            x[i, slot * emb_dim : (slot + 1) * emb_dim] = vecs[slot]
            x[i, flag_base + slot] = 1.0
        x[i, mean_base : mean_base + emb_dim] = l2_normalize(
            l2_normalize(embeddings[prefix]).mean(axis=0)
        )
        x[i, -1] = 1.0
    return x
