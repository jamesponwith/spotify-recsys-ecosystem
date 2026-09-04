"""Tests for the paired A/B harness.

The arithmetic tests below are built from synthetic per-challenge vectors whose
paired and unpaired statistics are pinned to the values the bead measured on the
real split, so the *decision rule* is verified hermetically. Reproducing those
numbers from the real catalog needs the built artifacts and lives in the
``requires_artifacts`` test at the bottom.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import requires_artifacts
from typer.testing import CliRunner

from cadence.cli import app
from cadence.config import DEFAULT
from cadence.eval.eval_ab import (
    apply_overrides,
    compare,
    format_report,
    label,
    parse_overrides,
    run,
)
from cadence.eval.metrics import MetricAccumulator
from cadence.retrieval.channels import ChannelResult
from cadence.retrieval.fusion import reciprocal_rank_fusion

# ---- override parsing ----------------------------------------------------


def test_parse_overrides_reads_rrf_k_and_channel_weights():
    got = parse_overrides(["rrf_k=30", "tag_exact=0.9"])
    assert got == {"rrf_k": 30.0, "tag_exact": 0.9}


def test_parse_overrides_accepts_every_shipped_channel_weight():
    for name, value in DEFAULT.retrieval.channel_weights.items():
        assert parse_overrides([f"{name}={value}"]) == {name: value}


def test_parse_overrides_refuses_assembly_knobs_by_name():
    # The whole point: this harness stops at fusion, so a null on mmr_lambda
    # would be a fact about the harness, not about the knob.
    with pytest.raises(ValueError, match="stops at fusion"):
        parse_overrides(["mmr_lambda=0.5"])
    with pytest.raises(ValueError, match="stops at fusion"):
        parse_overrides(["w_tempo=2.0"])


def test_parse_overrides_rejects_unknown_and_malformed():
    with pytest.raises(ValueError, match="unknown knob"):
        parse_overrides(["nonsense=1"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_overrides(["rrf_k"])
    with pytest.raises(ValueError, match="needs a number"):
        parse_overrides(["rrf_k=lots"])
    with pytest.raises(ValueError, match="finite"):
        parse_overrides(["rrf_k=inf"])


def test_parse_overrides_rejects_non_positive_rrf_k():
    # k <= 0 puts a zero in the denominator of w / (k + rank + 1).
    with pytest.raises(ValueError, match="must be > 0"):
        parse_overrides(["rrf_k=0"])


def test_parse_overrides_rejects_conflicting_repeats():
    with pytest.raises(ValueError, match="given twice"):
        parse_overrides(["rrf_k=30", "rrf_k=45"])
    assert parse_overrides(["rrf_k=30", "rrf_k=30"]) == {"rrf_k": 30.0}


# ---- override application ------------------------------------------------


def test_apply_overrides_sets_rrf_k_and_weights():
    cfg = apply_overrides(DEFAULT, {"rrf_k": 30.0, "audio": 1.5})
    assert cfg.retrieval.rrf_k == 30.0
    assert cfg.retrieval.channel_weights["audio"] == 1.5
    # Untouched weights keep their shipped values.
    assert cfg.retrieval.channel_weights["cooccurrence"] == 1.3


def test_apply_overrides_does_not_mutate_the_shared_default():
    before = dict(DEFAULT.retrieval.channel_weights)
    apply_overrides(DEFAULT, {"tag_exact": 0.9})
    assert DEFAULT.retrieval.channel_weights == before
    assert DEFAULT.retrieval.rrf_k == 60.0


def test_apply_overrides_with_nothing_returns_the_config_unchanged():
    assert apply_overrides(DEFAULT, {}) is DEFAULT


def test_label_names_the_arm():
    assert label({}) == "shipped"
    assert label({"rrf_k": 30.0}) == "rrf_k=30"
    assert label({"tag_exact": 0.9, "rrf_k": 30.0}) == "rrf_k=30,tag_exact=0.9"


def test_rrf_k_override_actually_changes_the_fused_order():
    """The knob has to reach the thing being measured, or every delta is zero."""
    results = [
        ChannelResult("collaborative", np.array([1, 2]), np.array([1.0, 0.9])),
        ChannelResult("tag", np.array([2, 1]), np.array([1.0, 0.9])),
    ]
    weights = {"collaborative": 1.0, "tag": 0.5}
    at_60 = reciprocal_rank_fusion(results, weights, k=60.0, top_n=10)
    at_1 = reciprocal_rank_fusion(results, weights, k=1.0, top_n=10)
    assert not np.array_equal(at_60.scores, at_1.scores)


# ---- the paired comparison -----------------------------------------------


def _vectors(mean_a: float, sd_a: float, delta: float, delta_sd: float, n: int, seed: int):
    """Arm A and arm B vectors whose paired delta and spread are exactly the
    requested values, so a verdict can be asserted against known statistics."""
    rng = np.random.default_rng(seed)

    def pin(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
        return (x - x.mean()) / x.std(ddof=1) * sd + mean

    a = pin(rng.normal(size=n), mean_a, sd_a)
    d = pin(rng.normal(size=n), delta, delta_sd)
    return a, a + d


def _acc(name: str, values: np.ndarray) -> MetricAccumulator:
    acc = MetricAccumulator()
    for v in values:
        acc.add(name, float(v))
    return acc


def test_compare_reports_means_delta_and_paired_band():
    a, b = _vectors(mean_a=0.1429, sd_a=0.149, delta=-0.00109, delta_sd=0.0185, n=400, seed=1)
    m = compare(_acc("r_precision", a), _acc("r_precision", b))["r_precision"]
    assert abs(m["mean_a"] - 0.1429) < 1e-12
    assert abs(m["mean_b"] - (0.1429 - 0.00109)) < 1e-12
    assert abs(m["delta"] - (-0.00109)) < 1e-12
    # band is 2 x SE of the difference: 2 * 0.0185 / sqrt(400).
    assert abs(m["band"] - 0.00185) < 1e-12
    assert m["n_changed"] == 400


def test_r_precision_at_the_measured_rrf_k_effect_is_not_detectable():
    # The bead's first acceptance number: delta -0.00109 inside a band of
    # +/-0.00185, so the paired harness must also decline to call it.
    a, b = _vectors(mean_a=0.1429, sd_a=0.149, delta=-0.00109, delta_sd=0.0185, n=400, seed=2)
    m = compare(_acc("r_precision", a), _acc("r_precision", b))["r_precision"]
    assert abs(m["delta"]) < m["band"]
    assert m["detectable"] is False


def test_ndcg_at_the_measured_rrf_k_effect_is_detectable_only_when_paired():
    # The bead's second number: delta -0.00579 against a paired band of
    # +/-0.00222 is significant, while the unpaired band the shipped harness
    # publishes is an order of magnitude wider and calls the same effect null.
    a, b = _vectors(mean_a=0.1889, sd_a=0.149, delta=-0.00579, delta_sd=0.0222, n=400, seed=3)
    m = compare(_acc("ndcg_100", a), _acc("ndcg_100", b))["ndcg_100"]
    assert abs(m["band"] - 0.00222) < 1e-12
    # The effect clears the paired band and is swallowed by the unpaired one.
    assert m["band"] < abs(m["delta"]) < m["unpaired_band"]
    assert m["detectable"] is True
    assert m["unpaired_detectable"] is False


def test_compare_covers_every_metric_both_arms_scored():
    acc_a, acc_b = MetricAccumulator(), MetricAccumulator()
    for name in ("r_precision", "ndcg_100", "clicks"):
        for v in (0.1, 0.2, 0.3):
            acc_a.add(name, v)
            acc_b.add(name, v + 0.01)
    got = compare(acc_a, acc_b)
    assert set(got) == {"r_precision", "ndcg_100", "clicks"}
    assert all(g["delta"] > 0 for g in got.values())


def test_compare_refuses_to_pair_arms_of_different_length():
    with pytest.raises(ValueError, match="different challenges"):
        compare(_acc("m", np.array([0.1, 0.2, 0.3])), _acc("m", np.array([0.1, 0.2])))


def test_one_challenge_is_never_detectable():
    # A single challenge leaves no spread to estimate a band from; a band of
    # zero would make any nonzero delta look perfectly resolved.
    m = compare(_acc("m", np.array([0.1])), _acc("m", np.array([0.9])))["m"]
    assert m["band"] == 0.0
    assert m["detectable"] is False
    assert m["unpaired_detectable"] is False


def test_identical_arms_report_a_zero_delta_and_no_verdict():
    vals = np.array([0.1, 0.4, 0.25, 0.7])
    m = compare(_acc("m", vals), _acc("m", vals))["m"]
    assert m["delta"] == 0.0
    assert m["n_changed"] == 0
    assert m["detectable"] is False


# ---- the printed report --------------------------------------------------


def _meta(n: int = 400) -> dict:
    return {
        "k": 0,
        "n": n,
        "depth": 500,
        "reranker": False,
        "arm_a": {"label": label({})},
        "arm_b": {"label": label({"rrf_k": 30.0})},
    }


def test_format_report_prints_both_bands_and_the_verdict():
    a, b = _vectors(mean_a=0.1889, sd_a=0.149, delta=-0.00579, delta_sd=0.0222, n=400, seed=4)
    got = compare(_acc("ndcg_100", a), _acc("ndcg_100", b))
    text = format_report(got, _meta())
    assert "arm A  shipped" in text
    assert "arm B  rrf_k=30" in text
    m = got["ndcg_100"]
    assert "-0.00579" in text  # the delta, signed
    assert f"{m['band']:.5f}" in text  # the paired band
    assert f"{m['unpaired_band']:.5f}" in text  # the band `evaluate` would use
    assert "DETECTABLE" in text


def test_format_report_columns_line_up_under_the_header():
    a, b = _vectors(mean_a=0.1889, sd_a=0.149, delta=-0.00579, delta_sd=0.0222, n=400, seed=6)
    acc_a, acc_b = _acc("ndcg_100", a), _acc("ndcg_100", b)
    # A second metric on a wholly different scale: clicks runs 0-51, not 0-1.
    ca, cb = _vectors(mean_a=8.9, sd_a=14.0, delta=0.31, delta_sd=2.4, n=400, seed=7)
    for x, y in zip(ca, cb, strict=True):
        acc_a.add("clicks", float(x))
        acc_b.add("clicks", float(y))
    lines = format_report(compare(acc_a, acc_b), _meta()).splitlines()
    header = next(line for line in lines if line.startswith("metric"))
    rows = [line for line in lines if line.startswith(("ndcg_100", "clicks"))]
    assert len(rows) == 2
    assert all(len(row) == len(header) for row in rows)


def test_format_report_says_how_many_metrics_pairing_rescued():
    a, b = _vectors(mean_a=0.1889, sd_a=0.149, delta=-0.00579, delta_sd=0.0222, n=400, seed=5)
    text = format_report(compare(_acc("ndcg_100", a), _acc("ndcg_100", b)), _meta())
    assert "1 of 1 metrics are detectable paired" in text


def test_format_report_stays_quiet_when_pairing_changed_no_verdict():
    vals = np.array([0.1, 0.4, 0.25, 0.7])
    text = format_report(compare(_acc("m", vals), _acc("m", vals)), _meta(n=4))
    assert "not detectable" in text
    assert "rescued" not in text
    assert "detectable paired" not in text


# ---- command wiring ------------------------------------------------------


def test_run_refuses_two_identical_arms_before_touching_the_catalog():
    with pytest.raises(ValueError, match="nothing to compare"):
        run(k=0, arm={"rrf_k": 30.0}, base={"rrf_k": 30.0})


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Typer renders through rich, which colours and boxes the output; strip
    that so an assertion is about the message and not about the frame."""
    return " ".join(ANSI.sub("", text).replace("│", " ").split())


def _cli(*args: str):
    # A wide terminal keeps rich from wrapping the message mid-sentence.
    return CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}).invoke(
        app, ["eval-ab", *args]
    )


def test_cli_rejects_an_assembly_knob_without_running_anything():
    result = _cli("--k", "0", "--arm", "mmr_lambda=0.5")
    # Exit 2 is a usage error: it never reached the catalog load.
    assert result.exit_code == 2
    assert "stops at fusion" in _plain(result.output)


def test_cli_exposes_the_command():
    result = _cli("--help")
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "--arm" in out
    assert "--base" in out


@requires_artifacts
def test_eval_ab_runs_end_to_end_on_the_real_split(tmp_path):
    """Both arms see the same challenges; only fusion differs. Small limit —
    the bead's published numbers come from `--limit 400` on the data host."""
    out = tmp_path / "eval_ab.json"
    report = run(k=0, limit=8, arm={"rrf_k": 30.0}, out_path=out, verbose=False)
    assert out.exists()
    meta = report["meta"]
    assert meta["n"] == 8
    assert meta["arm_a"]["rrf_k"] == DEFAULT.retrieval.rrf_k
    assert meta["arm_b"]["rrf_k"] == 30.0
    assert meta["arm_a"]["channel_weights"] == meta["arm_b"]["channel_weights"]
    for name, m in report["metrics"].items():
        assert abs(m["delta"] - (m["mean_b"] - m["mean_a"])) < 1e-9, name
        assert m["band"] == pytest.approx(2.0 * m["delta_se"])
        assert m["unpaired_band"] >= m["band"]
