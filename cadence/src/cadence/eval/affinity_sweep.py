"""Retune ``audio_affinity_weight`` against a metric that can see both sides.

The weight trades audio adherence against tag relevance, and until now only one
side of it was measured. `constraints_eval` reports mood error, which falls
monotonically as the weight rises, so nothing in the harness ever pushed back --
the term looked free at any strength. It was not: at 0.35 it was overturning a
hundred places of correct retrieval, which is how "90s alternative rock" came
back as One Direction.

Two things are needed to fix that, and the metric is the harder one.

**Tag adherence** is the missing counterweight: of the tracks delivered, what
share actually appear on playlists carrying a tag the listener asked for? It is
the folksonomy's own answer to "is this really 90s alternative rock", measured
against the same human behaviour the retrieval side is built on, rather than
against a genre string nobody curated.

**A battery that separates the cases.** A pure mood request should lean on audio
and a pure genre or era request should lean on tags, so a single global optimum
would be an average over two different questions. The battery is generated in
three families -- mood-led, tag-led, and mixed -- and every sweep is reported per
family so a query-dependent weight can be argued for or ruled out on evidence.

Retrieval does not depend on the weight, so it runs once per query and only the
selection stage is repeated. That makes a full sweep minutes rather than an hour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..assemble.select import select
from ..config import ARTIFACTS, DEFAULT
from ..models.train import AUDIO_FEATURE_COLS

# Mood words drawn from the planner's own lexicon, so the audio target the query
# implies is exactly the one the planner will resolve.
MOODS = [
    "chill",
    "mellow",
    "calm",
    "upbeat",
    "energetic",
    "sad",
    "happy",
    "romantic",
    "dreamy",
    "aggressive",
    "melancholy",
    "peaceful",
]
ACTIVITIES = [
    "for studying",
    "for a workout",
    "for a road trip",
    "for a dinner party",
    "for sleeping",
    "for a rainy morning",
    "for cleaning the house",
]
# Genres and eras that carry real folksonomy mass -- a tag nobody used cannot
# report adherence either way.
GENRES = ["rock", "rap", "pop", "country", "hip hop", "indie", "edm", "house", "soul", "metal"]
ERAS = ["1990s", "1980s", "2000s", "2010s", "70s"]


def build_battery(seed: int = 20260815) -> list[dict[str, Any]]:
    """Three families: mood-led, tag-led, and the contested mix."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []

    for m in MOODS:
        for a in ACTIVITIES:
            out.append({"query": f"{m} songs {a}", "family": "mood"})
    for g in GENRES:
        for e in ERAS:
            out.append({"query": f"{e} {g}", "family": "tag"})
        out.append({"query": f"{g} music", "family": "tag"})
    for m in MOODS:
        for g in GENRES:
            out.append({"query": f"{m} {g}", "family": "mixed"})
    for e in ERAS:
        for m in MOODS[:6]:
            out.append({"query": f"{m} {e} music", "family": "mixed"})

    rng.shuffle(out)
    return out


@dataclass
class Case:
    query: str
    family: str
    indices: np.ndarray
    scores: np.ndarray
    affinity: np.ndarray | None
    tag_cols: list[int]
    targets: dict[str, float]
    n_tracks: int
    max_per_artist: int


def prepare(engine, battery: list[dict[str, Any]], verbose: bool = True) -> list[Case]:
    """Retrieve once per query. Nothing here depends on the weight."""
    cases: list[Case] = []
    acfg = DEFAULT.assembly
    for i, item in enumerate(battery):
        intent = engine.planner.plan(item["query"], engine.known_tags).intent
        trace = engine.retrieve(intent)
        cand = trace.candidates
        if len(cand.indices) == 0:
            continue
        scores = cand.scores
        if engine.reranker is not None:
            scores = engine.reranker.score(engine.catalog, intent, trace)
        cases.append(
            Case(
                query=item["query"],
                family=item["family"],
                indices=cand.indices,
                scores=np.asarray(scores, dtype=np.float32),
                affinity=engine._audio_affinity(intent, cand.indices),
                tag_cols=trace.tag_cols,
                targets=dict(intent.audio.active()),
                n_tracks=intent.constraints.track_count or acfg.default_length,
                max_per_artist=intent.constraints.max_per_artist or acfg.max_tracks_per_artist,
            )
        )
        if verbose and (i + 1) % 40 == 0:
            print(f"  retrieved {i + 1}/{len(battery)}", flush=True)
    return cases


def tag_adherence(catalog, picked: np.ndarray, tag_cols: list[int]) -> tuple[float, float] | None:
    """How well the delivered tracks carry the tags the listener asked for.

    Two readings, because the obvious one saturates. *Share* is the fraction of
    tracks carrying at least one requested tag -- a low bar, since a track on one
    `rock` playlist out of five hundred clears it, and in practice it sits near
    1.0 whatever the weight, which makes it useless for tuning.

    *Strength* is the mean ``log1p`` of each track's requested-tag count, which is
    exactly what `sparse_tag_channel` ranks on. It distinguishes a track the
    crowd filed under `rock` two thousand times from one filed there once, and it
    keeps moving after share has pinned.

    Returns None when the query named no tag, so mood-only requests do not
    contribute a meaningless zero to the average.
    """
    if not tag_cols or picked.size == 0:
        return None
    sub = catalog.tag_matrix_csc[:, np.asarray(tag_cols, dtype=np.int64)]
    counts = np.asarray(sub[picked].sum(axis=1)).ravel()
    share = float((counts > 0).mean())
    strength = float(np.log1p(counts).mean())
    return share, strength


def mood_error(catalog, picked: np.ndarray, targets: dict[str, float]) -> float | None:
    """Mean absolute distance from the stated audio target, over stated dims."""
    if not targets or picked.size == 0:
        return None
    errs = []
    for k, v in targets.items():
        if k not in AUDIO_FEATURE_COLS:
            continue
        col = catalog.col(k)[picked]
        col = col[np.isfinite(col)]
        if col.size:
            errs.append(abs(float(col.mean()) - v))
    return float(np.mean(errs)) if errs else None


def sweep(
    catalog,
    cases: list[Case],
    weights: tuple[float, ...],
    verbose: bool = True,
) -> dict[str, Any]:
    acfg = DEFAULT.assembly
    families = sorted({c.family for c in cases})
    report: dict[str, Any] = {
        "n_cases": len(cases),
        "by_family": {f: sum(1 for c in cases if c.family == f) for f in families},
        "weights": list(weights),
        "results": [],
    }

    for w in weights:
        rows: dict[str, dict[str, list[float]]] = {
            f: {"tag_share": [], "tag_strength": [], "mood_error": []} for f in families
        }
        for c in cases:
            picked = select(
                catalog,
                c.indices,
                c.scores,
                n_tracks=c.n_tracks,
                max_per_artist=c.max_per_artist,
                mmr_lambda=acfg.mmr_lambda,
                affinity=c.affinity,
                affinity_weight=w if c.affinity is not None else 0.0,
            ).indices
            ta = tag_adherence(catalog, picked, c.tag_cols)
            me = mood_error(catalog, picked, c.targets)
            if ta is not None:
                rows[c.family]["tag_share"].append(ta[0])
                rows[c.family]["tag_strength"].append(ta[1])
            if me is not None:
                rows[c.family]["mood_error"].append(me)

        entry: dict[str, Any] = {"weight": w, "families": {}}
        for f in families:
            entry["families"][f] = {
                k: (float(np.mean(v)) if v else None) for k, v in rows[f].items()
            }
            entry["families"][f]["n_tag"] = len(rows[f]["tag_share"])
            entry["families"][f]["n_mood"] = len(rows[f]["mood_error"])
        alls = [x for f in families for x in rows[f]["tag_share"]]
        allg = [x for f in families for x in rows[f]["tag_strength"]]
        allm = [x for f in families for x in rows[f]["mood_error"]]
        entry["overall"] = {
            "tag_share": float(np.mean(alls)) if alls else None,
            "tag_strength": float(np.mean(allg)) if allg else None,
            "mood_error": float(np.mean(allm)) if allm else None,
        }
        report["results"].append(entry)
        if verbose:
            o = entry["overall"]
            print(
                f"  w={w:<5} tag_share={o['tag_share']:.4f}  "
                f"tag_strength={o['tag_strength']:.4f}  mood_error={o['mood_error']:.4f}",
                flush=True,
            )
    return report


def main(out: Path = ARTIFACTS / "affinity_sweep.json", verbose: bool = True) -> dict[str, Any]:
    from ..catalog import Catalog
    from ..engine import CadenceEngine
    from ..models.reranker import Reranker

    t0 = time.perf_counter()
    catalog = Catalog.load()
    # Same reranker the serving path uses; the sweep must measure the system as
    # it ships, not a fusion-only stand-in for it.
    rr_path = ARTIFACTS / "reranker.pkl"
    reranker = Reranker.load(rr_path) if rr_path.exists() else None
    engine = CadenceEngine(catalog, reranker=reranker)
    battery = build_battery()
    if verbose:
        print(f"battery: {len(battery)} queries", flush=True)
    cases = prepare(engine, battery, verbose)
    if verbose:
        print(f"prepared {len(cases)} cases in {time.perf_counter() - t0:.0f}s", flush=True)
    weights = (0.0, 0.1, 0.2, 0.3, 0.35, 0.45, 0.6, 0.8)
    report = sweep(catalog, cases, weights, verbose)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report
