"""Close the loop: recommend, let a listener respond, retrain on the response.

Each round does one full turn of the wheel that a deployed recommender turns
continuously:

    embeddings -> recommendations -> accepted tracks -> new playlists -> embeddings

The listener is a position-based acceptance model, which is the crude part and
also the load-bearing part: a listener keeps what is near the top far more often
than what is near the bottom, and that is precisely the mechanism that converts a
*ranking* bias into a *data* bias, and then into a training bias on the next
round. Nothing here is a claim about real listener behaviour. It is a claim about
what a ranking bias does to a corpus once the corpus is downstream of the ranking.

Three arms run from the same starting state:

    organic         same acceptance model, but the ranking is drawn from the
                    catalog's popularity distribution instead of from a query.
                    The control: it adds the same volume of new data, so drift
                    cannot be blamed on simply having more of it.
    closed_loop     the system as it ships
    exposure_aware  the system with Gamut's popularity penalty applied before
                    the listener ever sees the list
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from cadence.catalog import Catalog
from cadence.config import EmbeddingConfig, TagConfig
from cadence.engine import CadenceEngine
from cadence.models.factorize import factorize
from cadence.retrieval.ann import DenseIndex
from cadence.text import title_tokens
from gamut.exposure import CatalogFacts, measure
from gamut.rerank import popularity_norm
from scipy import sparse

from .config import ARMS, ARTIFACTS, CADENCE_PROCESSED, SimConfig


def _accept(rng: np.random.Generator, ranked: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Which of a ranked list the simulated listener keeps."""
    pos = np.arange(ranked.size)
    p = cfg.accept_base * np.exp(-cfg.position_decay * pos)
    return ranked[rng.random(ranked.size) < p]


def _refit(
    interactions: sparse.csr_matrix, tags: sparse.csr_matrix, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e, t = EmbeddingConfig(), TagConfig()
    collab = factorize(
        interactions.T.tocsr(),
        e.dim,
        shift_k=e.shift_k,
        popularity_damping=e.popularity_damping,
        seed=seed,
    )
    tag = factorize(tags, t.dim, shift_k=t.shift_k, popularity_damping=1.0, seed=seed)
    return collab.rows, tag.rows, tag.cols


def _rank(
    engine: CadenceEngine, intent, k: int, pop_norm: np.ndarray, penalty: float
) -> np.ndarray:
    """Top-k for one query, optionally exposure-penalised before it is shown."""
    fused = engine.retrieve(intent).candidates
    idx = fused.indices[: max(k * 5, k)]
    if idx.size == 0:
        return idx.astype(np.int64)
    if penalty > 0:
        # Rank-normalise, then penalise: raw RRF scores are far too skewed for a
        # subtractive term to mean anything stable across queries.
        ranks = np.linspace(1.0, 0.0, idx.size, dtype=np.float32)
        idx = idx[np.argsort(-(ranks - penalty * pop_norm[idx]), kind="stable")]
    return idx[:k].astype(np.int64)


def run(cfg: SimConfig | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or SimConfig()
    t0 = time.perf_counter()
    base = Catalog.load()
    facts = CatalogFacts.build(base.frame, cfg.tail_percentile, cfg.head_percentile)
    vocab_index = {t: i for i, t in enumerate(base.tag_vocab)}

    playlists = pd.read_parquet(CADENCE_PROCESSED / "playlists.parquet")
    all_titles = [str(t) for t in playlists["name"].to_numpy() if str(t).strip()]

    start_inter = base.interactions.tocsr()
    start_tags = base.tag_matrix.tocsr()
    n_tracks = start_inter.shape[1]

    report: dict[str, Any] = {
        "config": dataclasses.asdict(cfg),
        "catalog": {"n_tracks": facts.n_tracks, "n_artists": facts.n_artists},
        "arms": {},
    }

    for arm in ARMS:
        rng = np.random.default_rng(cfg.seed)
        inter, tags = start_inter.copy(), start_tags.copy()
        collab_v, tag_tracks_v, tag_cols_v = (
            base.collab.vectors.copy(),
            base.tag_tracks.vectors.copy(),
            base.tag_vectors.copy(),
        )
        history = []

        for rnd in range(cfg.rounds + 1):
            cat = dataclasses.replace(
                base,
                collab=DenseIndex(collab_v),
                tag_tracks=DenseIndex(tag_tracks_v),
                tag_vectors=tag_cols_v,
                interactions=inter,
                tag_matrix=tags,
            )
            engine = CadenceEngine(cat)
            counts = np.asarray(inter.sum(axis=0)).ravel().astype(np.float64)
            pop_norm = popularity_norm(counts)

            chosen = rng.choice(len(all_titles), size=cfg.queries_per_round, replace=False)
            shown, new_rows, new_cols, new_tagr, new_tagc = [], [], [], [], []
            row_id = 0
            for qi in chosen:
                title = all_titles[qi]
                if arm == "organic":
                    # Same acceptance model, popularity-shaped ranking.
                    p = counts / counts.sum() if counts.sum() else None
                    ranked = rng.choice(n_tracks, size=cfg.playlist_len, replace=False, p=p).astype(
                        np.int64
                    )
                else:
                    intent = engine.planner.plan(title, engine.known_tags).intent
                    ranked = _rank(
                        engine,
                        intent,
                        cfg.playlist_len,
                        pop_norm,
                        cfg.penalty if arm == "exposure_aware" else 0.0,
                    )
                if ranked.size == 0:
                    continue
                shown.append(
                    np.pad(ranked, (0, cfg.playlist_len - ranked.size), constant_values=-1)
                )
                kept = _accept(rng, ranked, cfg)
                if kept.size == 0:
                    continue
                new_rows.extend([row_id] * kept.size)
                new_cols.extend(kept.tolist())
                cols = [vocab_index[t] for t in title_tokens(title) if t in vocab_index]
                for c in cols:
                    new_tagr.extend(kept.tolist())
                    new_tagc.extend([c] * kept.size)
                row_id += 1

            block = np.vstack(shown) if shown else np.full((1, cfg.playlist_len), -1)
            exp = measure(block, facts, cfg.playlist_len)
            history.append(
                {
                    "round": rnd,
                    "added_playlists": row_id,
                    "added_interactions": len(new_rows),
                    "weighted_interactions": len(new_rows) * cfg.dose,
                    "total_interactions": int(inter.nnz),
                    "track_coverage": exp.track_coverage,
                    "artist_coverage": exp.artist_coverage,
                    "artist_gini": exp.artist_gini,
                    "track_gini": exp.track_gini,
                    "tail_share": exp.tail_share,
                    "top1pct_artist_share": exp.top1pct_artist_share,
                    "mean_log_popularity": exp.mean_log_popularity,
                }
            )
            if verbose:
                print(
                    f"{arm:15s} r{rnd}  gini={exp.artist_gini:.4f}  tail={exp.tail_share:.1%}  "
                    f"cov={exp.track_coverage:.3%}  top1%={exp.top1pct_artist_share:.1%}  "
                    f"+{len(new_rows):,} interactions",
                    flush=True,
                )

            if rnd == cfg.rounds:
                break

            # --- fold the response back into the corpus and refit -------------
            if new_rows:
                add = sparse.coo_matrix(
                    (
                        np.full(len(new_rows), float(cfg.dose), np.float32),
                        (new_rows, new_cols),
                    ),
                    shape=(row_id, n_tracks),
                ).tocsr()
                inter = sparse.vstack([inter, add]).tocsr()
                if new_tagr:
                    tadd = sparse.coo_matrix(
                        (
                            np.full(len(new_tagr), float(cfg.dose), np.float32),
                            (new_tagr, new_tagc),
                        ),
                        shape=tags.shape,
                    ).tocsr()
                    tags = (tags + tadd).tocsr()
            collab_v, tag_tracks_v, tag_cols_v = _refit(inter, tags, cfg.seed)

        report["arms"][arm] = history

    report["seconds"] = round(time.perf_counter() - t0, 1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "sim_report.json").write_text(json.dumps(report, indent=2))
    return report
