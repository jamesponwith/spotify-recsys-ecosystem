import numpy as np
import pytest

from cadence.eval.metrics import (
    MetricAccumulator,
    catalog_coverage,
    clicks,
    gini,
    intra_list_distance,
    ndcg,
    r_precision,
    r_precision_artist_aware,
    recall_at_k,
)


def test_r_precision_uses_ground_truth_size_as_cutoff():
    # |G| = 3, so only the top 3 predictions count; one of them is a hit.
    assert r_precision(np.array([1, 3, 5, 7, 9, 11]), {3, 7, 11}) == 1 / 3


def test_r_precision_perfect_and_empty():
    assert r_precision(np.array([1, 2, 3]), {1, 2, 3}) == 1.0
    assert r_precision(np.array([9, 8]), {1, 2}) == 0.0
    assert r_precision(np.array([1]), set()) == 0.0


def test_artist_aware_gives_partial_credit():
    artists = np.arange(20)
    artists[5] = artists[7]  # track 5 shares an artist with withheld track 7
    got = r_precision_artist_aware(np.array([1, 3, 5, 7]), {3, 7, 11}, artists)
    assert abs(got - (1 + 0.25) / 3) < 1e-9


def test_artist_credit_is_consumed_once_per_withheld_artist():
    # Two candidates share the one withheld artist; only one may claim credit.
    # Every other track keeps a distinct artist so nothing else can earn credit.
    artists = np.arange(20)
    artists[[5, 6]] = 3
    artists[7] = 3
    got = r_precision_artist_aware(np.array([5, 6, 4]), {7, 11, 12}, artists)
    assert abs(got - 0.25 / 3) < 1e-9


def test_clicks_counts_ten_track_pages():
    assert clicks(np.arange(100), {0}) == 0.0
    assert clicks(np.arange(100), {10}) == 1.0
    assert clicks(np.arange(100), {25}) == 2.0


def test_clicks_caps_when_nothing_relevant():
    assert clicks(np.arange(100), {9999}) == 51.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg(np.array([1, 2, 3, 4]), {1})
    late = ndcg(np.array([4, 3, 2, 1]), {1})
    assert early > late
    assert early == 1.0


def test_recall_at_k():
    assert recall_at_k(np.array([1, 2, 3, 4]), {1, 4}, k=2) == 0.5
    assert recall_at_k(np.array([1, 2, 3, 4]), {1, 4}, k=4) == 1.0


def test_gini_bounds():
    assert gini(np.ones(50)) == 0.0
    assert gini(np.array([1.0] + [0.0] * 99)) > 0.95


def test_catalog_coverage():
    assert catalog_coverage([np.array([0, 1]), np.array([1, 2])], n_items=10) == 0.3


def test_intra_list_distance_of_identical_vectors_is_zero():
    v = np.tile(np.array([[1.0, 0.0]]), (4, 1))
    assert abs(intra_list_distance(v)) < 1e-6


def test_accumulator_reports_mean_and_standard_error():
    acc = MetricAccumulator()
    for v in (0.0, 1.0):
        acc.add("m", v)
    s = acc.summary()
    assert s["m"] == 0.5
    assert s["m_se"] > 0
    assert s["n"] == 2


def _acc(**metrics: list[float]) -> MetricAccumulator:
    acc = MetricAccumulator()
    for name, vals in metrics.items():
        for v in vals:
            acc.add(name, v)
    return acc


def test_paired_deltas_mean_se_and_n_changed():
    ref = _acc(m=[0.1, 0.2, 0.3, 0.4])
    arm = _acc(m=[0.1, 0.25, 0.3, 0.5])
    d = arm.paired_deltas(ref)
    diffs = np.array([0.0, 0.05, 0.0, 0.1])
    assert abs(d["m_delta"] - diffs.mean()) < 1e-12
    assert abs(d["m_delta_se"] - diffs.std(ddof=1) / 2) < 1e-12
    assert d["m_n_changed"] == 2


def test_paired_deltas_sign_is_arm_minus_reference():
    ref = _acc(m=[0.5, 0.5])
    arm = _acc(m=[0.4, 0.4])
    assert abs(arm.paired_deltas(ref)["m_delta"] - (-0.1)) < 1e-12


def test_paired_deltas_requires_aligned_vectors():
    ref = _acc(m=[0.1, 0.2, 0.3])
    arm = _acc(m=[0.1, 0.2])
    with pytest.raises(ValueError):
        arm.paired_deltas(ref)


def test_paired_deltas_only_covers_shared_metrics():
    ref = _acc(m=[0.1, 0.2])
    arm = _acc(m=[0.1, 0.3], extra=[1.0, 2.0])
    d = arm.paired_deltas(ref)
    assert "m_delta" in d
    assert not any(k.startswith("extra") for k in d)


def test_paired_deltas_does_not_change_summary():
    arm = _acc(m=[0.1, 0.3])
    before = arm.summary()
    arm.paired_deltas(_acc(m=[0.2, 0.2]))
    assert arm.summary() == before


def test_paired_band_beats_unpaired_band_on_correlated_arms():
    # The situation the harness is in: two arms score the same challenges and
    # differ by a small per-challenge nudge, so between-challenge variance is
    # shared and cancels under pairing (rho ~ 0.99 in the real report).
    rng = np.random.default_rng(20260815)
    base = rng.uniform(0.0, 1.0, size=400)
    arm_vals = base + rng.normal(0.0, 0.005, size=400)
    ref, arm = _acc(m=list(base)), _acc(m=list(arm_vals))
    unpaired_band = 2 * np.hypot(ref.summary()["m_se"], arm.summary()["m_se"])
    paired_band = 2 * arm.paired_deltas(ref)["m_delta_se"]
    assert paired_band < unpaired_band / 10
