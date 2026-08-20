"""Head-to-head evaluation on genuinely ordered challenges.

The task and the three accuracy metrics are the RecSys Challenge 2018
definitions, reused from `cadence.eval.metrics` rather than reimplemented -- two
copies of a scoring rule is two chances to disagree with the leaderboard.

The challenges are rebuilt here, from the same 2,000 held-out playlists Cadence
uses, because Cadence's own `seed_tracks` are the k *lowest track ids* rather
than the first k tracks (docs/FINDINGS.md). For a sequence model that
distinction is the entire experiment.

Scoring is batched. Every system reduces to "one query vector per challenge,
ranked against the catalog", and 40,000 separate 159k x 160 mat-vecs is the same
arithmetic as a handful of GEMMs with an order of magnitude worse throughput.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from cadence.eval.metrics import clicks, ndcg, r_precision, r_precision_artist_aware

from .config import ARTIFACTS, CADENCE_PROCESSED, SegueConfig
from .features import l2_normalize

SYSTEMS = ("popularity", "last", "centroid", "segue", "segue_shuffled")


def holdout_rows(cfg: SegueConfig) -> set[int]:
    splits = json.loads((CADENCE_PROCESSED / "splits.json").read_text())
    return {int(r) for r in splits["holdout_rows"]}


def make_challenges(sequences, rows: set[int], k: int, min_held_out: int = 5) -> list[dict]:
    """Ordered prefix of length k, everything after it withheld."""
    out = []
    for i in range(len(sequences)):
        if int(sequences.rows[i]) not in rows:
            continue
        seq = sequences[i]
        if seq.size < k + min_held_out:
            continue
        out.append(
            {"row": int(sequences.rows[i]), "seeds": seq[:k], "held_out": set(map(int, seq[k:]))}
        )
    return out


def _query_vectors(
    challenges: list[dict], model, embeddings: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """One query vector per challenge, per vector-based system."""
    seeds = [c["seeds"].astype(np.int64) for c in challenges]
    return {
        "last": l2_normalize(np.vstack([embeddings[s[-1]] for s in seeds])),
        "centroid": l2_normalize(np.vstack([embeddings[s].sum(axis=0) for s in seeds])),
        "segue": model.predict(seeds, embeddings),
        # The causal check: same tracks, same model, order destroyed. If this
        # matches `segue`, the model was never using order and the premise of
        # this project is decoration.
        "segue_shuffled": model.predict([rng.permutation(s) for s in seeds], embeddings),
    }


def _rank_batch(
    queries: np.ndarray,
    matrix: np.ndarray,
    seed_lists: list[np.ndarray],
    k: int,
    block: int = 128,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for start in range(0, len(seed_lists), block):
        stop = min(start + block, len(seed_lists))
        scores = matrix @ queries[start:stop].T  # (n_tracks, n_block)
        for j in range(stop - start):
            col = scores[:, j]
            col[seed_lists[start + j]] = -np.inf
            top = np.argpartition(-col, k - 1)[:k]
            out.append(top[np.argsort(-col[top], kind="stable")].astype(np.int64))
    return out


def evaluate(
    sequences,
    model,
    embeddings: np.ndarray,
    popularity_scores: np.ndarray,
    artists: np.ndarray,
    cfg: SegueConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    matrix = np.ascontiguousarray(l2_normalize(embeddings.astype(np.float32)))
    rows = holdout_rows(cfg)
    pop_order = np.argsort(-popularity_scores, kind="stable")

    report: dict[str, Any] = {
        "config": {"top_k": cfg.top_k, "window": cfg.window, "seed": cfg.seed},
        "seed_counts": {},
    }

    for k in cfg.seed_counts:
        rng = np.random.default_rng(cfg.seed)  # reset per k so shuffles are reproducible
        challenges = make_challenges(sequences, rows, k)
        seed_lists = [c["seeds"].astype(np.int64) for c in challenges]
        queries = _query_vectors(challenges, model, embeddings, rng)

        preds: dict[str, list[np.ndarray]] = {
            name: _rank_batch(q, matrix, seed_lists, cfg.top_k) for name, q in queries.items()
        }
        # Popularity needs no query vector: one global order, minus this
        # challenge's seeds.
        preds["popularity"] = [pop_order[~np.isin(pop_order, s)][: cfg.top_k] for s in seed_lists]

        stats: dict[str, dict[str, float]] = {}
        for name in SYSTEMS:
            rp, rpa, nd, cl = [], [], [], []
            for pred, ch in zip(preds[name], challenges, strict=True):
                truth = ch["held_out"]
                rp.append(r_precision(pred, truth))
                rpa.append(r_precision_artist_aware(pred, truth, artists))
                nd.append(ndcg(pred, truth, k=100))
                cl.append(clicks(pred, truth))
            stats[name] = {
                "r_precision": float(np.mean(rp)),
                "r_precision_artist": float(np.mean(rpa)),
                "ndcg": float(np.mean(nd)),
                "clicks": float(np.mean(cl)),
            }

        report["seed_counts"][str(k)] = {"n_challenges": len(challenges), "systems": stats}
        if verbose:
            line = "  ".join(f"{s}={stats[s]['r_precision']:.4f}" for s in SYSTEMS)
            print(f"k={k:<3} n={len(challenges):<5} R-prec  {line}", flush=True)

    report["seconds"] = round(time.perf_counter() - t0, 1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, indent=2))
    return report
