"""Learned reranker over fused candidates.

Reciprocal-rank fusion is a fixed, hand-set prior: it says every query should
trust the tag channel exactly as much as the collaborative one. That is plainly
wrong per query — "songs like Radiohead" is a CF question, "rainy day study" is
a tag question — and the right weighting is something to learn, not to argue
about.

Supervision comes free from the corpus: hide part of a real playlist, retrieve
against its title and remaining seeds, and label each candidate by whether a
human actually put it on that playlist. That is a genuine relevance label, not
a proxy.

Model: gradient-boosted trees on ~25 features. Chosen over a neural ranker
because the feature set is small and tabular, it trains in seconds on CPU, and
`feature_importance` is directly inspectable — which matters more here than the
last point of AUC.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ARTIFACTS, SEED
from ..models.train import AUDIO_FEATURE_COLS

CHANNELS = (
    "collaborative",
    "cooccurrence",
    "tag",
    "tag_exact",
    "lexical",
    "audio",
    "popularity",
)

FEATURE_NAMES: list[str] = (
    ["fusion_score"]
    + [f"{c}_{suffix}" for c in CHANNELS for suffix in ("rank", "score", "present")]
    + [
        "log_popularity",
        "has_audio",
        "audio_distance",
        "audio_target_set",
        "tag_hits",
        "log_tag_mass",
        "tempo_in_range",
        "tempo_known",
        "duration_s",
        "n_seeds",
        "n_tag_cols",
    ]
)


def extract_features(catalog, intent, trace) -> np.ndarray:
    """Build the (n_candidates, n_features) design matrix.

    Shared verbatim by training and serving. If these ever diverge, offline
    metrics stop predicting online behaviour and nothing downstream reveals it.
    """
    cand = trace.candidates
    idx = cand.indices
    n = len(idx)
    cols: list[np.ndarray] = [cand.scores.astype(np.float32)]

    for ch in CHANNELS:
        ranks = cand.channel_ranks.get(ch)
        scores = cand.channel_scores.get(ch)
        if ranks is None:
            cols.append(np.full(n, 9999.0, dtype=np.float32))
            cols.append(np.zeros(n, dtype=np.float32))
            cols.append(np.zeros(n, dtype=np.float32))
        else:
            cols.append(np.log1p(ranks).astype(np.float32))
            cols.append(scores.astype(np.float32))
            cols.append((ranks < 9990).astype(np.float32))

    cols.append(np.log1p(catalog.col("n_playlists")[idx]).astype(np.float32))
    cols.append(catalog.col("has_audio")[idx].astype(np.float32))

    active = intent.audio.active()
    if active:
        dims = [AUDIO_FEATURE_COLS.index(k) for k in active if k in AUDIO_FEATURE_COLS]
        vals = np.array([active[AUDIO_FEATURE_COLS[d]] for d in dims], dtype=np.float32)
        z_target = (vals - catalog.audio_mu[dims]) / catalog.audio_sigma[dims]
        dist = np.linalg.norm(catalog.audio_z[idx][:, dims] - z_target[None, :], axis=1)
        dist = dist / np.sqrt(max(len(dims), 1))
        cols.append(dist.astype(np.float32))
        cols.append(np.ones(n, dtype=np.float32))
    else:
        cols.append(np.zeros(n, dtype=np.float32))
        cols.append(np.zeros(n, dtype=np.float32))

    if trace.tag_cols:
        sub = catalog.tag_matrix_csc[:, np.asarray(trace.tag_cols, dtype=np.int64)][idx]
        mass = np.asarray(sub.sum(axis=1)).ravel()
        hits = np.asarray((sub > 0).sum(axis=1)).ravel()
        cols.append(hits.astype(np.float32))
        cols.append(np.log1p(mass).astype(np.float32))
    else:
        cols.append(np.zeros(n, dtype=np.float32))
        cols.append(np.zeros(n, dtype=np.float32))

    tempo = catalog.col("tempo")[idx]
    known = np.isfinite(tempo)
    in_range = np.ones(n, dtype=bool)
    if intent.tempo.min_bpm is not None:
        in_range &= tempo >= intent.tempo.min_bpm
    if intent.tempo.max_bpm is not None:
        in_range &= tempo <= intent.tempo.max_bpm
    cols.append((in_range & known).astype(np.float32))
    cols.append(known.astype(np.float32))

    cols.append((catalog.col("duration_ms")[idx] / 1000.0).astype(np.float32))
    cols.append(np.full(n, float(len(trace.seed_indices)), dtype=np.float32))
    cols.append(np.full(n, float(len(trace.tag_cols)), dtype=np.float32))

    return np.column_stack(cols).astype(np.float32)


@dataclass
class TrainReport:
    n_queries: int
    n_rows: int
    n_positive: int
    train_auc: float
    valid_auc: float
    valid_ndcg_uplift: float
    seconds: float
    feature_importance: dict[str, float]


class Reranker:
    """Thin wrapper so the engine only needs ``.score(...)``."""

    def __init__(self, model=None, feature_names: list[str] | None = None) -> None:
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES)

    # ---- serving ---------------------------------------------------------
    def score(self, catalog, intent, trace) -> np.ndarray:
        if self.model is None:
            return trace.candidates.scores
        x = extract_features(catalog, intent, trace)
        return self.model.predict_proba(x)[:, 1].astype(np.float32)

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path = ARTIFACTS / "reranker.pkl") -> None:
        path.write_bytes(pickle.dumps({"model": self.model, "features": self.feature_names}))

    @classmethod
    def load(cls, path: Path = ARTIFACTS / "reranker.pkl") -> Reranker:
        payload = pickle.loads(path.read_bytes())
        return cls(model=payload["model"], feature_names=payload["features"])


def train_reranker(
    engine,
    processed_dir: Path,
    *,
    n_queries: int = 1500,
    seed_counts: tuple[int, ...] = (0, 5),
    candidates: int = 600,
    negatives_per_positive: int = 8,
    seed: int = SEED,
    out_dir: Path = ARTIFACTS,
    verbose: bool = True,
) -> TrainReport:
    """Generate supervision from training playlists and fit the ranker."""
    import pandas as pd
    from scipy import sparse
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    from ..eval.splits import load_splits

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    playlists = pd.read_parquet(processed_dir / "playlists.parquet")
    interactions = sparse.load_npz(processed_dir / "interactions.npz").tocsr()
    holdout, _ = load_splits(processed_dir)

    # Never train the reranker on an evaluation playlist.
    eligible = np.setdiff1d(np.arange(interactions.shape[0]), holdout)
    lengths = np.diff(interactions.indptr)
    eligible = eligible[lengths[eligible] >= 15]
    titles = playlists["name"].to_numpy()
    eligible = np.array([r for r in eligible if str(titles[r]).strip()], dtype=np.int64)
    chosen = rng.choice(eligible, size=min(n_queries, eligible.size), replace=False)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []

    for qi, row in enumerate(chosen):
        k = int(seed_counts[qi % len(seed_counts)])
        trs = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
        trs = trs.astype(np.int64)
        if len(trs) <= k + 5:
            continue
        seeds, truth = trs[:k], set(trs[k:].tolist())

        intent = engine.planner.plan(str(titles[row]), engine.known_tags).intent
        trace = engine.retrieve(intent, extra_seed_indices=seeds, exclude=seeds, top_n=candidates)
        if len(trace.candidates) == 0:
            continue

        x = extract_features(engine.catalog, intent, trace)
        y = np.array([1 if int(i) in truth else 0 for i in trace.candidates.indices], dtype=np.int8)
        pos = np.flatnonzero(y == 1)
        if pos.size == 0:
            continue
        neg = np.flatnonzero(y == 0)
        take = min(neg.size, pos.size * negatives_per_positive)
        neg = rng.choice(neg, size=take, replace=False)
        keep = np.concatenate([pos, neg])
        xs.append(x[keep])
        ys.append(y[keep])
        groups.extend([qi] * keep.size)

        if verbose and (qi + 1) % 250 == 0:
            print(f"  {qi + 1}/{len(chosen)} queries, {sum(len(a) for a in xs):,} rows")

    if not xs:
        raise RuntimeError("no training rows generated for the reranker")

    x_all = np.vstack(xs)
    y_all = np.concatenate(ys)
    g_all = np.asarray(groups)

    # Split by query, not by row: rows from one playlist are highly correlated
    # and a random row split would leak the answer across the boundary.
    uniq = np.unique(g_all)
    rng.shuffle(uniq)
    cut = int(len(uniq) * 0.8)
    train_q, valid_q = set(uniq[:cut].tolist()), set(uniq[cut:].tolist())
    tr = np.array([g in train_q for g in g_all])
    va = np.array([g in valid_q for g in g_all])

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(x_all[tr], y_all[tr])

    train_auc = float(roc_auc_score(y_all[tr], model.predict_proba(x_all[tr])[:, 1]))
    valid_auc = float(roc_auc_score(y_all[va], model.predict_proba(x_all[va])[:, 1]))

    # Uplift against the fusion score alone, measured on the same validation rows.
    fusion_auc = float(roc_auc_score(y_all[va], x_all[va][:, 0]))

    imp: dict[str, float] = {}
    try:
        sample = np.flatnonzero(va)
        sample = rng.choice(sample, size=min(6000, sample.size), replace=False)
        r = permutation_importance(
            model, x_all[sample], y_all[sample], n_repeats=3, random_state=seed, scoring="roc_auc"
        )
        order = np.argsort(-r.importances_mean)[:12]
        imp = {FEATURE_NAMES[i]: round(float(r.importances_mean[i]), 5) for i in order}
    except Exception:  # noqa: BLE001 - importance is diagnostic, never load-bearing
        imp = {}

    reranker = Reranker(model=model)
    out_dir.mkdir(parents=True, exist_ok=True)
    reranker.save(out_dir / "reranker.pkl")

    report = TrainReport(
        n_queries=int(len(np.unique(g_all))),
        n_rows=int(x_all.shape[0]),
        n_positive=int(y_all.sum()),
        train_auc=round(train_auc, 4),
        valid_auc=round(valid_auc, 4),
        valid_ndcg_uplift=round(valid_auc - fusion_auc, 4),
        seconds=round(time.perf_counter() - t0, 1),
        feature_importance=imp,
    )
    (out_dir / "reranker_meta.json").write_text(json.dumps(report.__dict__, indent=2))
    if verbose:
        print(json.dumps(report.__dict__, indent=2))
    return report
