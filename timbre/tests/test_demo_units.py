"""Hermetic tests for the demo's catalog surgery."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from timbre.demo import _zero_cols, _zero_rows


def test_zero_rows_removes_only_named_rows():
    m = sparse.csr_matrix(np.arange(1, 13, dtype=np.float32).reshape(3, 4))
    out = _zero_rows(m, np.array([1]))
    assert out[0].toarray().tolist() == [[1, 2, 3, 4]]
    assert out[1].toarray().sum() == 0
    assert out[2].toarray().tolist() == [[9, 10, 11, 12]]


def test_zero_cols_removes_only_named_cols():
    """A cold track is a *column* of the playlist x track matrix.

    Zeroing the wrong axis would silently delete playlists instead of tracks and
    still produce plausible-looking numbers.
    """
    m = sparse.csr_matrix(np.arange(1, 13, dtype=np.float32).reshape(3, 4))
    out = _zero_cols(m, np.array([2]))
    assert out.toarray()[:, 2].sum() == 0
    assert out.toarray()[:, 0].tolist() == [1, 5, 9]


def test_zeroed_row_is_dead_in_a_dense_index():
    """The whole freeze-out depends on this: a zeroed row must score -inf."""
    from cadence.retrieval.ann import DenseIndex

    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    vectors[2] = 0.0
    idx = DenseIndex(vectors)
    scores = idx.scores(np.array([1.0, 1.0], dtype=np.float32))
    assert np.isneginf(scores[2])
    assert np.isfinite(scores[:2]).all()
