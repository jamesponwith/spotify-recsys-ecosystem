"""The per-challenge sidecar written next to eval_report.json.

These tests pin the contract the report now carries: every mean, SE and paired
field must be re-derivable from the sidecar vectors to 1e-9 — including after a
round trip through JSON, which is how any downstream reader will see them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cadence.eval.metrics import MetricAccumulator
from cadence.eval.run_eval import (
    PAIRED_REFERENCE,
    build_cell,
    sidecar_path,
    verify_vectors,
)


def _accs(rng: np.random.Generator, n: int = 40) -> dict[str, MetricAccumulator]:
    base = rng.uniform(0.0, 1.0, size=(2, n))
    out: dict[str, MetricAccumulator] = {}
    for arm in (PAIRED_REFERENCE, "no_cooccurrence", "popularity"):
        acc = MetricAccumulator()
        noise = rng.normal(0.0, 0.01, size=(2, n)) if arm != PAIRED_REFERENCE else 0.0
        vals = base + noise
        for name, row in zip(("r_precision", "ndcg_100"), vals, strict=True):
            for v in row:
                acc.add(name, v)
        out[arm] = acc
    return out


def _report_and_sidecar(accs: dict[str, MetricAccumulator]) -> tuple[dict, dict]:
    cell, vectors = build_cell(accs)
    # Non-derivable, set-level numbers ride along in the real report and must
    # not trip verification.
    cell[PAIRED_REFERENCE]["coverage_100"] = 0.5
    cell[PAIRED_REFERENCE]["latency_p50_ms"] = 12.0
    report = {"results": {"5": cell}}
    sidecar = {"meta": {"reference": PAIRED_REFERENCE}, "vectors": {"5": vectors}}
    return report, sidecar


def test_build_cell_preserves_summary_and_adds_paired_fields():
    accs = _accs(np.random.default_rng(0))
    cell, _ = build_cell(accs)
    for arm, acc in accs.items():
        for key, want in acc.summary().items():
            assert cell[arm][key] == want
    assert not any(k.endswith("_delta") for k in cell[PAIRED_REFERENCE])
    for arm in ("no_cooccurrence", "popularity"):
        for m in ("r_precision", "ndcg_100"):
            assert f"{m}_delta" in cell[arm]
            assert f"{m}_delta_se" in cell[arm]
            assert f"{m}_n_changed" in cell[arm]


def test_verify_vectors_passes_after_json_round_trip():
    report, sidecar = _report_and_sidecar(_accs(np.random.default_rng(1)))
    report = json.loads(json.dumps(report))
    sidecar = json.loads(json.dumps(sidecar))
    verify_vectors(report, sidecar)


def test_verify_vectors_rejects_a_drifted_mean():
    report, sidecar = _report_and_sidecar(_accs(np.random.default_rng(2)))
    report["results"]["5"]["no_cooccurrence"]["r_precision"] += 1e-6
    with pytest.raises(ValueError, match="r_precision"):
        verify_vectors(report, sidecar)


def test_verify_vectors_rejects_a_drifted_delta():
    report, sidecar = _report_and_sidecar(_accs(np.random.default_rng(3)))
    report["results"]["5"]["popularity"]["ndcg_100_delta"] += 1e-6
    with pytest.raises(ValueError, match="ndcg_100_delta"):
        verify_vectors(report, sidecar)


def test_verify_vectors_rejects_a_missing_paired_field():
    report, sidecar = _report_and_sidecar(_accs(np.random.default_rng(4)))
    del report["results"]["5"]["no_cooccurrence"]["r_precision_delta_se"]
    with pytest.raises(ValueError, match="r_precision_delta_se"):
        verify_vectors(report, sidecar)


def test_sidecar_path_sits_next_to_the_report(tmp_path):
    assert sidecar_path(tmp_path / "eval_report.json") == tmp_path / "eval_report_vectors.json"
