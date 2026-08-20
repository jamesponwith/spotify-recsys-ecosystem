import numpy as np

from segue.baselines import by_vector, centroid, last_track, popularity
from segue.features import l2_normalize


def _setup():
    """Non-orthogonal on purpose.

    With an identity basis every non-seed track scores exactly 0 against any
    seed, so the ranking is decided by tie-breaking and the order-sensitivity
    test passes or fails for reasons unrelated to what it claims to check.
    """
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    return emb, np.ascontiguousarray(l2_normalize(emb))


def test_seeds_are_never_recommended_back():
    emb, mat = _setup()
    seeds = np.array([0, 1])
    for pred in (
        last_track(seeds, mat, emb, 5),
        centroid(seeds, mat, emb, 5),
        popularity(np.arange(5, dtype=np.float32), seeds, 3),
    ):
        assert not set(pred.tolist()) & set(seeds.tolist())


def test_centroid_ignores_order_and_last_track_does_not():
    """The two baselines must differ exactly where order lives."""
    emb, mat = _setup()
    a, b = np.array([0, 1]), np.array([1, 0])
    assert centroid(a, mat, emb, 3).tolist() == centroid(b, mat, emb, 3).tolist()
    assert last_track(a, mat, emb, 3).tolist() != last_track(b, mat, emb, 3).tolist()


def test_by_vector_returns_ranked_order():
    emb, mat = _setup()
    out = by_vector(np.array([1.0, 0.0], dtype=np.float32), mat, np.array([], dtype=np.int64), 3)
    assert out.tolist() == [0, 2, 4]  # exact, near, midpoint


def test_zero_query_returns_nothing_rather_than_arbitrary_ties():
    emb, mat = _setup()
    assert by_vector(np.zeros(2, dtype=np.float32), mat, np.array([], dtype=np.int64), 3).size == 0
