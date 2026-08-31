"""Offline evaluation harness.

Runs the full system, the ablations and the baselines over the same held-out
challenges and writes one JSON report. Everything is driven from the frozen
split so that any two runs are comparable.

Sweeping the seed count is the core experiment. k=0 is the natural-language
cold-start case this project exists for; k=25 is a conventional
continue-this-playlist task. A system that only reports one of them is hiding
half its behaviour.

Every arm scores the identical challenge list in identical order, so each cell
also carries paired deltas against the full_fusion reference, and the raw
per-challenge vectors are persisted to a `*_vectors.json` sidecar next to the
report. verify_vectors() re-derives every published mean, SE and delta from
that sidecar before either file is written.
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
PAIRED_REFERENCE = "full_fusion"

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
) -> tuple[dict, MetricAccumulator]:
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

    extras = {
        "coverage_100": catalog_coverage(recommended, n_items),
        "gini_100": gini(exposure[exposure > 0]) if exposure.any() else 0.0,
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }
    return extras, acc


def _run_baseline(
    baseline, challenges: list[Challenge], artist_ids: np.ndarray, n_items: int, limit: int | None
) -> tuple[dict, MetricAccumulator]:
    acc = MetricAccumulator()
    recommended: list[np.ndarray] = []
    items = challenges[:limit] if limit else challenges
    for ch in items:
        seeds = np.asarray(ch.seed_tracks, dtype=np.int64)
        preds = baseline.recommend(seeds, ch.title, k=DEPTH)
        acc.update(evaluate_ranking(preds, set(ch.held_out), artist_ids))
        recommended.append(preds[:100])
    return {"coverage_100": catalog_coverage(recommended, n_items)}, acc


def build_cell(
    accs: dict[str, MetricAccumulator],
) -> tuple[dict[str, dict], dict[str, dict[str, list[float]]]]:
    """Collapse each arm's accumulator to its summary, attach paired deltas
    against PAIRED_REFERENCE, and keep the per-challenge vectors the summaries
    collapse — the pairing is only valid because every arm scored the same
    challenges in the same order."""
    reference = accs.get(PAIRED_REFERENCE)
    cell: dict[str, dict] = {}
    vectors: dict[str, dict[str, list[float]]] = {}
    for name, acc in accs.items():
        out = acc.summary()
        if reference is not None and acc is not reference:
            out.update(acc.paired_deltas(reference))
        cell[name] = out
        vectors[name] = {m: list(v) for m, v in acc.values.items()}
    return cell, vectors


def sidecar_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_vectors" + out_path.suffix)


def verify_vectors(report: dict, sidecar: dict, tol: float = 1e-9) -> None:
    """Re-derive every mean, SE and paired field in the report from the
    sidecar's per-challenge vectors; raise ValueError on the first field that
    is missing or disagrees beyond tol. Set-level numbers (coverage, gini,
    latency) have no per-challenge vector and are not checked."""
    try:
        reference = sidecar["meta"]["reference"]
        all_vectors = sidecar["vectors"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"not an eval vectors sidecar: missing {e}") from e
    for k, cell in report["results"].items():
        arms = all_vectors.get(k, {})
        for arm, published in cell.items():
            vecs = arms.get(arm)
            if vecs is None:
                raise ValueError(f"sidecar has no vectors for results[{k!r}][{arm!r}]")
            # Every published mean with an _se companion came from a
            # per-challenge vector, so a vector the sidecar lost is a hole in
            # the guarantee, not a key to skip.
            missing = {
                key[: -len("_se")]
                for key in published
                if key.endswith("_se") and not key.endswith("_delta_se")
            } - set(vecs)
            if missing:
                raise ValueError(
                    f"sidecar vectors for results[{k!r}][{arm!r}] lack {sorted(missing)}"
                )
            acc = MetricAccumulator(values={m: [float(x) for x in v] for m, v in vecs.items()})
            derived = acc.summary()
            if arm != reference and reference in arms:
                ref_acc = MetricAccumulator(
                    values={m: [float(x) for x in v] for m, v in arms[reference].items()}
                )
                derived.update(acc.paired_deltas(ref_acc))
            for key, want in derived.items():
                got = published.get(key)
                # `not <=` rather than `>` so a NaN on either side fails
                # instead of slipping through every comparison.
                if got is None or not abs(float(got) - want) <= tol:
                    raise ValueError(
                        f"results[{k!r}][{arm!r}][{key!r}]: report has {got!r}, "
                        f"vectors re-derive {want!r}"
                    )


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
            "paired_reference": PAIRED_REFERENCE,
            "build": catalog.meta["build"],
            "train": catalog.meta["train"],
        },
        "results": {},
    }
    vectors_by_k: dict[str, dict[str, dict[str, list[float]]]] = {}

    for k in ks:
        items = challenges.get(k, [])
        if not items:
            continue
        accs: dict[str, MetricAccumulator] = {}
        extras: dict[str, dict] = {}
        for cfg in configs:
            t0 = time.perf_counter()
            extras[cfg.name], accs[cfg.name] = _run_engine_config(
                engine, items, cfg, artist_ids, n_items, limit
            )
            if verbose:
                m = accs[cfg.name].summary()
                print(
                    f"k={k:<3} {cfg.name:<20} R-prec={m['r_precision']:.4f} "
                    f"NDCG@100={m['ndcg_100']:.4f} clicks={m['clicks']:.2f} "
                    f"({time.perf_counter() - t0:.0f}s)"
                )
        for name, bl in baselines.items():
            extras[name], accs[name] = _run_baseline(bl, items, artist_ids, n_items, limit)
            if verbose:
                m = accs[name].summary()
                print(
                    f"k={k:<3} {name:<20} R-prec={m['r_precision']:.4f} "
                    f"NDCG@100={m['ndcg_100']:.4f} clicks={m['clicks']:.2f}"
                )
        cell, vectors = build_cell(accs)
        for name, ex in extras.items():
            cell[name].update(ex)
        report["results"][str(k)] = cell
        vectors_by_k[str(k)] = vectors

    out_path = out_path or artifacts_dir / "eval_report.json"
    sidecar = {
        "meta": {"reference": PAIRED_REFERENCE, "report": out_path.name},
        "vectors": vectors_by_k,
    }
    verify_vectors(report, sidecar)
    out_path.write_text(json.dumps(report, indent=2))
    vec_path = sidecar_path(out_path)
    vec_path.write_text(json.dumps(sidecar))
    if verbose:
        print(f"\nwrote {out_path}")
        print(f"wrote {vec_path}")
    return report
