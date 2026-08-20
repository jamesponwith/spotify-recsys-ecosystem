"""Encode Spotify's 11 descriptors into a regression design matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import AUDIO_CATEGORICAL_COLS, AUDIO_CONTINUOUS_COLS

N_PITCH_CLASSES = 12


def encode(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return ``(X, feature_names)``.

    Nine continuous descriptors are passed through as-is (standardisation is left
    to the caller, which must fit it on the training split only). ``key`` is
    one-hot encoded across the twelve pitch classes rather than passed as an
    integer: pitch class is categorical and 11 is not "more" than 0 -- it is a
    semitone below it. ``mode`` is already binary.

    Sentinel values survive untouched. ``tempo == 0`` and ``loudness == -60``
    are Spotify's markers for "could not be determined", and a model is entitled
    to learn that undetectable tempo is itself informative.
    """
    cont = frame[AUDIO_CONTINUOUS_COLS].to_numpy(dtype=np.float32)

    key = frame["key"].to_numpy()
    onehot = np.zeros((len(frame), N_PITCH_CLASSES), dtype=np.float32)
    valid = np.isfinite(key) & (key >= 0) & (key < N_PITCH_CLASSES)
    onehot[np.flatnonzero(valid), key[valid].astype(np.int64)] = 1.0

    mode = np.nan_to_num(frame["mode"].to_numpy(dtype=np.float32), nan=0.0).reshape(-1, 1)

    x = np.hstack([cont, onehot, mode]).astype(np.float32)
    names = (
        list(AUDIO_CONTINUOUS_COLS)
        + [f"key_{i}" for i in range(N_PITCH_CLASSES)]
        + [AUDIO_CATEGORICAL_COLS[1]]
    )
    return x, names


def standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std fitted on the training split only, with a zero-variance guard.

    The one-hot columns are included deliberately: standardising them costs
    nothing and keeps the ridge penalty comparable across all coordinates.
    """
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)
