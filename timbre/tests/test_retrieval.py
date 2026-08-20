import numpy as np

from timbre.phase0.retrieval import CatalogIndex, substitute


def test_dead_rows_are_unreachable():
    """An all-zero row must score -inf, not tie at 0.0.

    This is the guard that keeps `content` from competing for slots `oracle` is
    structurally barred from -- the ratio Gate 0 divides by depends on it.
    """
    vectors = np.zeros((4, 3), dtype=np.float32)
    vectors[0] = [1.0, 0.0, 0.0]
    vectors[1] = [0.0, 1.0, 0.0]
    # rows 2 and 3 stay dead
    index = CatalogIndex(vectors)
    assert index.live.tolist() == [True, True, False, False]

    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    recall = index.top_k_hits(q, [np.array([2])], k=2, block=8)
    assert recall[0] == 0.0  # dead row never retrieved even with k == live rows


def test_recall_is_fraction_of_relevant_found():
    vectors = np.eye(5, dtype=np.float32)
    index = CatalogIndex(vectors)
    q = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # top-2 must be rows 0 and 1; two of the three relevant items are there.
    recall = index.top_k_hits(q, [np.array([0, 1, 4])], k=2, block=8)
    assert abs(recall[0] - 2 / 3) < 1e-9


def test_substitute_only_touches_named_rows():
    base = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = substitute(base, np.array([1, 3]), np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(out[0], base[0])
    assert np.array_equal(out[2], base[2])
    assert not out[1].any() and not out[3].any()
    assert base.any()  # the original is untouched
