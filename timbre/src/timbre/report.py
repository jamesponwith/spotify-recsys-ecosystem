"""Aggregate the joint demo across several queries."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from cadence.catalog import Catalog

from .config import ARTIFACTS
from .demo import run_demo
from .predictor import TagPredictor

# Chosen to span the query types Cadence handles differently: a mood, an
# activity, a genre+era compound, a texture, and one with an explicit acoustic
# constraint. A single query would not distinguish "Timbre works" from "Timbre
# works on the one query I picked".
DEMO_QUERIES = [
    "rainy day study",
    "high energy gym workout",
    "90s alternative rock",
    "late night jazz for reading",
    "upbeat indie pop for a road trip",
    "mellow acoustic sunday morning",
]


def build(
    queries: list[str] | None = None, *, n_cold: int = 20, top_n: int = 200
) -> dict[str, Any]:
    queries = queries or DEMO_QUERIES
    t0 = time.perf_counter()
    catalog = Catalog.load()
    predictor = TagPredictor.load()

    runs: list[dict[str, Any]] = []
    for q in queries:
        r = run_demo(q, n_cold=n_cold, top_n=top_n, catalog=catalog, predictor=predictor)
        by = r.by_name
        runs.append(
            {
                "query": r.query,
                "n_cold": r.n_cold,
                "tracks": r.cold_tracks,
                "cold_in_top_n": by["cold"].in_top_n,
                "timbre_in_top_n": by["timbre"].in_top_n,
                "cold_fraction": by["cold"].recovered_fraction,
                "timbre_fraction": by["timbre"].recovered_fraction,
                "timbre_median_rank": by["timbre"].median_rank,
            }
        )
        print(
            f"{r.query:38s} cold {by['cold'].in_top_n:3d}/{r.n_cold}   "
            f"timbre {by['timbre'].in_top_n:3d}/{r.n_cold}",
            flush=True,
        )

    total_cold = sum(r["n_cold"] for r in runs)
    report = {
        "top_n": top_n,
        "n_cold_per_query": n_cold,
        "n_queries": len(runs),
        "runs": runs,
        "aggregate": {
            "total_frozen": total_cold,
            "recovered_without_timbre": sum(r["cold_in_top_n"] for r in runs),
            "recovered_with_timbre": sum(r["timbre_in_top_n"] for r in runs),
            "recovery_rate_without": sum(r["cold_in_top_n"] for r in runs) / total_cold,
            "recovery_rate_with": sum(r["timbre_in_top_n"] for r in runs) / total_cold,
            "median_timbre_rank": float(
                np.median([r["timbre_median_rank"] for r in runs if r["timbre_median_rank"]])
            )
            if any(r["timbre_median_rank"] for r in runs)
            else None,
        },
        "predictor": {"kind": predictor.kind, "mean_cosine": predictor.mean_cosine},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "demo_report.json").write_text(json.dumps(report, indent=2))
    return report
