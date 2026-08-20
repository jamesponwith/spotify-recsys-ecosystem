import numpy as np
import pytest
from scipy import sparse

from cadence.models.factorize import factorize, l2_normalize, sppmi
from cadence.retrieval.ann import DenseIndex


def test_sppmi_is_non_negative(synthetic_interactions):
    m = sppmi(synthetic_interactions, shift_k=5.0)
    assert m.data.min() >= 0.0
    assert m.nnz <= synthetic_interactions.nnz


def test_larger_shift_prunes_more(skewed_interactions):
    low = sppmi(skewed_interactions, shift_k=1.0)
    mid = sppmi(skewed_interactions, shift_k=3.0)
    high = sppmi(skewed_interactions, shift_k=12.0)
    assert low.nnz > mid.nnz > high.nnz


def test_popularity_damping_lifts_rare_item_scores(skewed_interactions):
    plain = sppmi(skewed_interactions, shift_k=2.0, popularity_damping=1.0)
    damped = sppmi(skewed_interactions, shift_k=2.0, popularity_damping=0.6)
    # Damping the column marginal reduces the penalty applied to popular items,
    # so more pairs survive the shift.
    assert damped.nnz >= plain.nnz


def test_sppmi_rejects_empty_matrix():
    with pytest.raises(ValueError):
        sppmi(sparse.csr_matrix((5, 5), dtype=np.float32))


def test_factorization_recovers_cluster_structure(synthetic_interactions):
    f = factorize(synthetic_interactions.T.tocsr(), dim=12, seed=0)
    index = DenseIndex(f.rows)
    # Items 0-29 form one taste cluster; neighbours of item 5 must stay inside it.
    ids, _ = index.search(f.rows[5], k=6)
    assert all(i < 30 for i in ids)


def test_factorization_is_deterministic(synthetic_interactions):
    a = factorize(synthetic_interactions, dim=8, seed=7).rows
    b = factorize(synthetic_interactions, dim=8, seed=7).rows
    np.testing.assert_allclose(a, b)


def test_explained_energy_in_unit_interval(synthetic_interactions):
    f = factorize(synthetic_interactions, dim=8, seed=0)
    assert 0.0 <= f.explained_energy <= 1.0


def test_l2_normalize_leaves_zero_rows_finite():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(x)
    assert np.isfinite(out).all()
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6
    assert np.linalg.norm(out[1]) == 0.0


def test_dense_index_excludes_zero_rows():
    vectors = np.array([[1.0, 0.0], [0.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    index = DenseIndex(vectors)
    ids, _ = index.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert 1 not in ids.tolist()


def test_dense_index_respects_mask():
    vectors = np.eye(4, dtype=np.float32)
    index = DenseIndex(vectors)
    mask = np.array([False, True, True, True])
    ids, _ = index.search(vectors[0], k=3, mask=mask)
    assert 0 not in ids.tolist()
