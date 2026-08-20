"""Offline evaluation harness.

Runs the full system, the ablations and the baselines over the same held-out
challenges and writes one JSON report. Everything is driven from the frozen
split so that any two runs are comparable.

Sweeping the seed count is the core experiment. k=0 is the natural-language
cold-start case this project exists for; k=25 is a conventional
continue-this-playlist task. A system that only reports one of them is hiding
half its behaviour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from ..catalog import Catalog
from ..config import ARTIFACTS, DATA_PROCESSED
from ..engine import CadenceEngine
from .baselines import ItemKNNBaseline, LexicalBaseline, PopularityBaseline
from .metrics import MetricAccumulator, catalog_coverage, evaluate_ranking, gini
from .splits import Challenge, load_splits

DEPTH = 500

ALL_CHANNELS = {
    "collaborative",
    "cooccurrence",
    "tag",
    "tag_exact",
    "lexical",
    "audio",
    "popularity",
}
ALL = ALL_CHANNELS


@dataclass
class RunConfig:
    name: str
    channels: set[str] | None = None
    use_reranker: bool = False


def _run_engine_config(
    engine: CadenceEngine,
    challenges: list[Challenge],
    cfg: RunConfig,
    artist_ids: np.ndarray,
    n_items: int,
    limit: int | None,
) -> dict:
    acc = MetricAccumulator()
    recommended: list[np.ndarray] = []
    exposure = np.zeros(n_items, dtype=np.int64)
    latencies: list[float] = []
    items = challenges[:limit] if limit else challenges

    for ch in items:
        seeds = np.asarray(ch.seed_tracks, dtype=np.int64)
        truth = set(ch.held_out)
        t0 = time.perf_counter()
        intent = engine.planner.plan(ch.title, engine.known_tags).intent
        trace = engine.retrieve(
            intent,
            extra_seed_indices=seeds,
            exclude=seeds,
            top_n=DEPTH,
            channels=cfg.channels,
        )
        preds = trace.candidates.indices
        if cfg.use_reranker and engine.reranker is not None and len(preds):
            scores = engine.reranker.score(engine.catalog, intent, trace)
            preds = preds[np.argsort(-scores, kind="stable")]
        latencies.append((time.perf_counter() - t0) * 1000)

        if len(preds) == 0:
            preds = np.zeros(0, dtype=np.int64)
        acc.update(evaluate_ranking(preds, truth, artist_ids))
        recommended.append(preds[:100])
        if len(preds):
            exposure[preds[:100]] += 1

    out = acc.summary()
    out["coverage_100"] = catalog_coverage(recommended, n_items)
    out["gini_100"] = gini(exposure[exposure > 0]) if exposure.any() else 0.0
    out["latency_p50_ms"] = float(np.percentile(latencies, 50)) if latencies else 0.0
    out["latency_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else 0.0
    return out


def _run_baseline(
    baseline, challenges: list[Challenge], artist_ids: np.ndarray, n_items: int, limit: int | None
) -> dict:
    acc = MetricAccumulator()
    recommended: list[np.ndarray] = []
    items = challenges[:limit] if limit else challenges
    for ch in items:
        seeds = np.asarray(ch.seed_tracks, dtype=np.int64)
        preds = baseline.recommend(seeds, ch.title, k=DEPTH)
        acc.update(evaluate_ranking(preds, set(ch.held_out), artist_ids))
        recommended.append(preds[:100])
    out = acc.summary()
    out["coverage_100"] = catalog_coverage(recommended, n_items)
    return out


def run(
    processed_dir: Path = DATA_PROCESSED,
    artifacts_dir: Path = ARTIFACTS,
    *,
    limit: int | None = 400,
    seed_counts: tuple[int, ...] | None = None,
    ablations: bool = True,
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    catalog = Catalog.load(processed_dir, artifacts_dir)
    engine = CadenceEngine(catalog)
    reranker = None
    rr_path = artifacts_dir / "reranker.pkl"
    if rr_path.exists():
        from ..models.reranker import Reranker

        reranker = Reranker.load(rr_path)
        engine.reranker = reranker

    _, challenges = load_splits(processed_dir)
    interactions = sparse.load_npz(processed_dir / "interactions.npz").tocsr()
    holdout, _ = load_splits(processed_dir)
    keep = np.ones(interactions.shape[0], dtype=bool)
    keep[holdout] = False
    train_interactions = interactions[keep]

    artist_ids = catalog.artist_ids
    n_items = len(catalog)

    configs = [RunConfig("full_fusion")]
    if reranker is not None:
        configs.append(RunConfig("full_reranked", use_reranker=True))
    if ablations:
        configs += [
            RunConfig("no_collaborative", channels=ALL - {"collaborative"}),
            RunConfig("no_cooccurrence", channels=ALL - {"cooccurrence"}),
            RunConfig("no_tag", channels=ALL - {"tag", "tag_exact"}),
            RunConfig("no_lexical", channels=ALL - {"lexical"}),
            RunConfig("no_audio", channels=ALL - {"audio"}),
            RunConfig("only_collaborative", channels={"collaborative"}),
            RunConfig("only_cooccurrence", channels={"cooccurrence"}),
            RunConfig("only_tag", channels={"tag", "tag_exact"}),
        ]

    baselines = {
        "popularity": PopularityBaseline(train_interactions),
        "item_knn": ItemKNNBaseline(train_interactions),
        "lexical_title": LexicalBaseline(catalog),
    }

    ks = seed_counts or tuple(sorted(challenges))
    report: dict = {
        "meta": {
            "n_tracks": n_items,
            "depth": DEPTH,
            "limit_per_cell": limit,
            "seed_counts": list(ks),
            "reranker": reranker is not None,
            "build": catalog.meta["build"],
            "train": catalog.meta["train"],
        },
        "results": {},
    }

    for k in ks:
        items = challenges.get(k, [])
        if not items:
            continue
        cell: dict[str, dict] = {}
        for cfg in configs:
            t0 = time.perf_counter()
            cell[cfg.name] = _run_engine_config(engine, items, cfg, artist_ids, n_items, limit)
            if verbose:
                m = cell[cfg.name]
                print(
                    f"k={k:<3} {cfg.name:<20} R-prec={m['r_precision']:.4f} "
                    f"NDCG@100={m['ndcg_100']:.4f} clicks={m['clicks']:.2f} "
                    f"({time.perf_counter() - t0:.0f}s)"
                )
        for name, bl in baselines.items():
            cell[name] = _run_baseline(bl, items, artist_ids, n_items, limit)
            if verbose:
                m = cell[name]
                print(
                    f"k={k:<3} {name:<20} R-prec={m['r_precision']:.4f} "
                    f"NDCG@100={m['ndcg_100']:.4f} clicks={m['clicks']:.2f}"
                )
        report["results"][str(k)] = cell

    out_path = out_path or artifacts_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"\nwrote {out_path}")
    return report
