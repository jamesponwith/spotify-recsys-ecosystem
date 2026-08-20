import numpy as np
import pandas as pd

from timbre.phase0.features import N_PITCH_CLASSES, encode, standardizer


def _frame(**over):
    base = {
        "danceability": [0.5, 0.9],
        "energy": [0.4, 0.8],
        "speechiness": [0.05, 0.1],
        "acousticness": [0.2, 0.7],
        "instrumentalness": [0.0, 0.9],
        "liveness": [0.1, 0.3],
        "valence": [0.6, 0.2],
        "loudness": [-5.0, -12.0],
        "tempo": [120.0, 90.0],
        "key": [0, 11],
        "mode": [1, 0],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_key_is_one_hot_not_ordinal():
    """Pitch class 11 is a semitone below 0, not eleven units above it."""
    x, names = encode(_frame())
    key_cols = [i for i, n in enumerate(names) if n.startswith("key_")]
    assert len(key_cols) == N_PITCH_CLASSES
    block = x[:, key_cols]
    assert block.sum(axis=1).tolist() == [1.0, 1.0]
    assert block[0].argmax() == 0
    assert block[1].argmax() == 11


def test_shape_is_nine_continuous_plus_twelve_keys_plus_mode():
    x, names = encode(_frame())
    assert x.shape == (2, 22)
    assert len(names) == 22
    assert names[-1] == "mode"


def test_missing_key_produces_all_zero_onehot_not_a_false_c():
    """A NaN key must not silently become pitch class 0."""
    x, names = encode(_frame(key=[np.nan, 11]))
    key_cols = [i for i, n in enumerate(names) if n.startswith("key_")]
    assert x[0, key_cols].sum() == 0.0
    assert x[1, key_cols].sum() == 1.0


def test_standardizer_guards_zero_variance():
    x = np.array([[1.0, 5.0], [1.0, 9.0]], dtype=np.float32)
    mu, sigma = standardizer(x)
    assert sigma[0] == 1.0  # constant column must not divide by ~0
    assert np.isfinite((x - mu) / sigma).all()
