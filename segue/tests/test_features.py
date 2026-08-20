import numpy as np

from segue.features import encode, feature_dim, l2_normalize


def _emb(n=6, d=4):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_slot_zero_is_the_most_recent_track():
    """Order is the whole thesis; getting the direction backwards would invert it."""
    emb = _emb()
    x = encode([np.array([3, 4, 5])], emb, window=2)
    assert np.allclose(x[0, 0:4], l2_normalize(emb[5]))  # slot 0 = last track
    assert np.allclose(x[0, 4:8], l2_normalize(emb[4]))  # slot 1 = the one before


def test_presence_flags_distinguish_absent_from_zero():
    emb = _emb()
    x = encode([np.array([2])], emb, window=3)
    flags = x[0, 3 * 4 : 3 * 4 + 3]
    assert flags.tolist() == [1.0, 0.0, 0.0]


def test_mean_feature_is_order_free():
    """The order-free block must be identical under permutation -- it is the
    part the centroid baseline also sees."""
    emb = _emb()
    a = encode([np.array([1, 2, 3])], emb, window=2)
    b = encode([np.array([3, 1, 2])], emb, window=2)
    base = 2 * 4 + 2
    assert np.allclose(a[0, base : base + 4], b[0, base : base + 4])
    # ...while the ordered slots must differ
    assert not np.allclose(a[0, :8], b[0, :8])


def test_dim_matches_declared():
    emb = _emb()
    x = encode([np.array([1, 2])], emb, window=5)
    assert x.shape[1] == feature_dim(5, 4)


def test_empty_prefix_is_intercept_only():
    x = encode([np.array([], dtype=np.int64)], _emb(), window=3)
    assert x[0, -1] == 1.0
    assert not x[0, :-1].any()
