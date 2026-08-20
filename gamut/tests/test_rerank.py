import numpy as np

from gamut.rerank import apply_artist_cap, popularity_norm, rank_normalize, rerank


def test_rank_normalize_ignores_padding():
    scores = np.array([[9.0, 5.0, 1.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, True, True, False]])
    out = rank_normalize(scores, valid)
    assert out[0, 0] == 1.0 and out[0, 2] == 0.0
    assert out[0, 3] == 0.0  # padded slot untouched


def test_zero_penalty_preserves_the_original_order():
    idx = np.array([[3, 1, 2]], dtype=np.int32)
    scores = np.array([[9.0, 5.0, 1.0]], dtype=np.float32)
    pop = np.zeros(5, dtype=np.float32)
    assert rerank(idx, scores, pop, 0.0)[0].tolist() == [3, 1, 2]


def test_penalty_demotes_the_popular_track():
    idx = np.array([[0, 1]], dtype=np.int32)
    scores = np.array([[1.0, 0.9]], dtype=np.float32)
    pop = np.array([1.0, 0.0], dtype=np.float32)  # track 0 is maximally popular
    assert rerank(idx, scores, pop, 2.0)[0].tolist() == [1, 0]


def test_artist_cap_keeps_first_n_per_artist_in_rank_order():
    block = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    artists = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    out = apply_artist_cap(block, artists, cap=2)[0]
    kept = out[out >= 0].tolist()
    assert kept == [0, 1, 3, 4, 5]  # track 2 dropped: artist 0 already had two


def test_popularity_norm_spans_zero_to_one():
    pn = popularity_norm(np.array([0, 5, 100], dtype=float))
    assert pn.min() == 0.0 and pn.max() == 1.0
