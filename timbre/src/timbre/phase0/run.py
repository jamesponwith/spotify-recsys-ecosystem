"""Run the falsification test and rule on Gate 0."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from ..config import ARTIFACTS, Phase0Config
from ..predictor import TagPredictor
from . import data as data_mod
from . import gate
from .fit import fit_mlp, fit_ridge
from .retrieval import CatalogIndex, substitute


def run(cfg: Phase0Config | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or Phase0Config()
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)

    d = data_mod.load(cfg)
    if verbose:
        print(
            f"tracks={d.n_tracks:,}  usable={int(d.usable.sum()):,}  "
            f"train={d.train_idx.size:,}  test={d.test_idx.size:,}  "
            f"queries={len(d.queries):,}"
        )

    x_tr, y_tr = d.x[d.train_idx], d.y[d.train_idx]
    x_te, y_te = d.x[d.test_idx], d.y[d.test_idx]

    fits = []
    for fit in (fit_ridge,) if cfg.skip_mlp else (fit_ridge, fit_mlp):
        f = fit(x_tr, y_tr, x_te, y_te, cfg)
        fits.append(f)
        if verbose:
            print(
                f"{f.name}: mean cosine={f.mean_cosine:.4f} ({f.seconds}s) {f.detail}", flush=True
            )

    # --- the five systems --------------------------------------------------
    dim = d.y.shape[1]
    n_test = d.test_idx.size
    random_vecs = data_mod.l2_normalize(rng.standard_normal((n_test, dim)).astype(np.float32))
    # The no-information floor: one vector, repeated. Every cold track ties
    # exactly, so whether any lands in a top-100 is decided by index order --
    # which is the correct behaviour for a floor, and worth stating plainly.
    mean_vec = data_mod.l2_normalize(d.y[d.train_idx].mean(axis=0)[None, :])
    systems: list[tuple[str, np.ndarray]] = [
        ("random", random_vecs),
        ("mean", np.repeat(mean_vec, n_test, axis=0)),
        *[(f"content_{f.name}", f.prediction) for f in fits],
        ("oracle", d.y[d.test_idx]),
    ]

    results: dict[str, dict[str, float]] = {}
    for name, replacement in systems:
        matrix = substitute(d.y, d.test_idx, replacement)
        index = CatalogIndex(matrix)
        per_query = index.top_k_hits(
            d.queries.vectors, d.queries.relevant, cfg.recall_k, cfg.query_block
        )
        results[name] = {
            "recall_at_100": float(per_query.mean()),
            "recall_std": float(per_query.std()),
            "queries_with_any_hit": float((per_query > 0).mean()),
        }
        if verbose:
            print(f"{name:16s} recall@100={results[name]['recall_at_100']:.4f}")

    # --- Gate 0 -----------------------------------------------------------
    g0 = gate.rule(results, cfg)

    report: dict[str, Any] = {
        "config": {
            "seed": cfg.seed,
            "test_fraction": cfg.test_fraction,
            "recall_k": cfg.recall_k,
        },
        "data": {
            "n_tracks": d.n_tracks,
            "n_usable": int(d.usable.sum()),
            "n_train": int(d.train_idx.size),
            "n_test": int(d.test_idx.size),
            "excluded_no_audio": d.excluded_no_audio,
            "excluded_zero_embedding": d.excluded_zero_embedding,
            "n_queries": len(d.queries),
            "queries_dropped_no_vocab": d.queries.n_titles_no_vocab,
            "queries_dropped_no_relevant": d.queries.n_titles_no_relevant,
            "n_features": int(d.x.shape[1]),
            "tag_dim": dim,
        },
        "fits": {
            f.name: {"mean_cosine": f.mean_cosine, "seconds": f.seconds, **f.detail} for f in fits
        },
        "retrieval": results,
        "gate_0": g0,
        "seconds": round(time.perf_counter() - t0, 1),
    }

    # Persist the stronger regressor. Phase 0's verdict is the point, but the
    # fitted model is what lets Cadence and Timbre be demonstrated together
    # without waiting for Phase 1's encoder.
    winner = next(f for f in fits if f"content_{f.name}" == g0["best_content_system"])
    predictor = TagPredictor(
        model=winner.model,
        mu=d.mu,
        sigma=d.sigma,
        kind=winner.name,
        mean_cosine=winner.mean_cosine,
    )
    report["predictor"] = {"kind": winner.name, "path": str(predictor.save())}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "phase0_report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        mult = (
            "floor is 0 (criterion vacuous)"
            if g0["random_criterion_vacuous"]
            else f"{g0['random_multiple']:.1f}x random, need {cfg.gate_random_multiple}x"
        )
        print(
            f"\nGate 0: {'PASS' if g0['passed'] else 'FAIL'}  ({mult}; "
            f"{g0['oracle_recovery_ratio']:.1%} of oracle, need {cfg.gate_oracle_fraction:.0%})"
        )
    return report
