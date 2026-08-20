"""Run the whole audit: per-channel attribution, then the intervention frontier."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from cadence.eval.metrics import ndcg, r_precision

from .collect import Collected
from .config import ARTIFACTS, CADENCE_PROCESSED, CHANNELS, AuditConfig
from .exposure import CatalogFacts, ExposureReport, measure
from .rerank import apply_artist_cap, popularity_norm, rerank


def _accuracy(block: np.ndarray, truth: list[set[int]], depth: int) -> dict[str, float]:
    rp, nd = [], []
    for row, gt in zip(block, truth, strict=True):
        pred = row[row >= 0][:depth].astype(np.int64)
        rp.append(r_precision(pred, gt))
        nd.append(ndcg(pred, gt, k=100))
    return {"r_precision": float(np.mean(rp)), "ndcg": float(np.mean(nd))}


def _channel_block(collected: Collected, ci: int, depth: int) -> np.ndarray:
    """Re-order each query's candidates by one channel's own ranking.

    A channel that never returned a candidate for a query contributes an empty
    row rather than a padded one, so its coverage is not inflated by other
    channels' finds sitting in the shared candidate pool.
    """
    ranks = collected.channel_ranks[ci]
    out = np.full((ranks.shape[0], depth), -1, dtype=np.int32)
    for i in range(ranks.shape[0]):
        present = np.flatnonzero(ranks[i] >= 0)
        if present.size == 0:
            continue
        ordered = present[np.argsort(ranks[i][present], kind="stable")][:depth]
        out[i, : ordered.size] = collected.indices[i][ordered]
    return out


def _as_dict(r: ExposureReport) -> dict[str, float]:
    return {k: float(v) for k, v in r.__dict__.items()}


def run(cfg: AuditConfig | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or AuditConfig()
    t0 = time.perf_counter()
    collected = Collected.load()
    frame = pd.read_parquet(CADENCE_PROCESSED / "tracks.parquet")
    facts = CatalogFacts.build(frame, cfg.tail_percentile, cfg.head_percentile)
    pop = popularity_norm(facts.play_counts)

    report: dict[str, Any] = {
        "config": {
            "n_queries": len(collected.titles),
            "depth": cfg.depth,
            "cut": cfg.cut,
            "tail_percentile": cfg.tail_percentile,
            "seed": cfg.seed,
        },
        "catalog": {
            "n_tracks": facts.n_tracks,
            "n_artists": facts.n_artists,
            "tail_share_of_catalog": facts.tail_share_of_catalog,
            "median_playlists": float(np.median(facts.play_counts)),
        },
    }

    # --- what the system does today -------------------------------------
    baseline = measure(collected.indices, facts, cfg.cut)
    report["baseline"] = {
        **_as_dict(baseline),
        **_accuracy(collected.indices, collected.truth, cfg.cut),
        "pool": _as_dict(measure(collected.indices, facts, cfg.depth)),
    }
    if verbose:
        print(
            f"baseline  coverage={baseline.track_coverage:.3%}  "
            f"artist gini={baseline.artist_gini:.3f}  tail={baseline.tail_share:.1%}",
            flush=True,
        )

    # --- which channel concentrates exposure -----------------------------
    channels: dict[str, Any] = {}
    for ci, name in enumerate(CHANNELS):
        block = _channel_block(collected, ci, cfg.depth)
        exp = measure(block, facts, cfg.cut)
        if exp.n_recommendations == 0:
            continue
        channels[name] = {**_as_dict(exp), **_accuracy(block, collected.truth, cfg.cut)}
        if verbose:
            print(
                f"  {name:14s} cov={exp.track_coverage:.3%}  gini={exp.artist_gini:.3f}  "
                f"tail={exp.tail_share:.1%}  R-prec={channels[name]['r_precision']:.4f}",
                flush=True,
            )
    report["channels"] = channels

    # --- the price of intervening ----------------------------------------
    frontier = []
    for penalty in cfg.penalties:
        block = rerank(collected.indices, collected.scores, pop, penalty)
        exp = measure(block, facts, cfg.cut)
        acc = _accuracy(block, collected.truth, cfg.cut)
        frontier.append({"penalty": penalty, **_as_dict(exp), **acc})
        if verbose:
            print(
                f"penalty={penalty:<5} tail={exp.tail_share:.1%}  gini={exp.artist_gini:.3f}  "
                f"R-prec={acc['r_precision']:.4f}",
                flush=True,
            )
    report["frontier"] = frontier

    # --- the other knob ---------------------------------------------------
    # The popularity penalty barely moves artist concentration, because
    # concentration is about how many catalog entries an artist owns rather than
    # how popular any one of them is. A per-artist cap attacks that directly.
    caps = []
    for cap in (1, 2, 3, 5):
        block = apply_artist_cap(collected.indices, facts.artists, cap)
        exp = measure(block, facts, cfg.cut)
        acc = _accuracy(block, collected.truth, cfg.cut)
        caps.append({"cap": cap, **_as_dict(exp), **acc})
        if verbose:
            print(
                f"cap={cap:<5}    tail={exp.tail_share:.1%}  gini={exp.artist_gini:.3f}  "
                f"artists={exp.artist_coverage:.3%}  R-prec={acc['r_precision']:.4f}",
                flush=True,
            )
    report["artist_caps"] = caps
    report["seconds"] = round(time.perf_counter() - t0, 1)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "audit_report.json").write_text(json.dumps(report, indent=2))
    return report
