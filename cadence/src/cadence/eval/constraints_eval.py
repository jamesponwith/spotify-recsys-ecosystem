"""Constraint-satisfaction evaluation for the assembly stage.

The ranking harness in ``run_eval`` measures retrieval quality and stops there:
it never calls selection or sequencing. So the stages that actually enforce the
listener's *requirements* — artist caps, track counts, duration targets, tempo
windows, the explicit filter — would otherwise go unmeasured, and "the playlist
respects what you asked for" is at least as much of a product promise as "the
playlist is relevant".

This runs a fixed battery of constrained requests end to end and reports how
often each requirement is actually met, plus the mood-adherence error: how far
the delivered playlist sits from the audio target the listener asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ARTIFACTS
from ..models.train import AUDIO_FEATURE_COLS

# A battery covering each constraint type alone and in combination.
QUERIES: list[str] = [
    "12 chill songs for studying",
    "20 upbeat workout tracks",
    "8 acoustic songs for a rainy morning",
    "15 party bangers, nothing explicit",
    "10 sad songs for a late night drive",
    "25 songs of 90s hip hop",
    "workout playlist between 140 and 160 bpm, 12 tracks",
    "slow jazz for dinner, about 40 minutes",
    "one hour of instrumental focus music",
    "18 high energy dance tracks, no explicit lyrics",
    "6 mellow indie folk songs",
    "30 minutes of calm sleep music",
    "14 throwback 2000s pop songs, clean only",
    "songs like Bon Iver, 10 tracks",
    "romantic acoustic songs, 45 minutes, family friendly",
    "16 aggressive metal tracks over 130 bpm",
    "9 summer road trip singalongs",
    "20 songs that build up for a run, 150 to 175 bpm",
    "12 dreamy instrumental tracks under 100 bpm",
    "cozy coffee shop music, 35 minutes, nothing explicit",
]


@dataclass
class ConstraintReport:
    n_queries: int
    satisfaction: dict[str, dict[str, float]] = field(default_factory=dict)
    mood_error: dict[str, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    mean_long_tail_share: float = 0.0
    mean_intra_list_distance: float = 0.0
    failures: list[str] = field(default_factory=list)


def run(engine, queries: list[str] | None = None, verbose: bool = True) -> ConstraintReport:
    queries = queries or QUERIES
    hits: dict[str, list[bool]] = {}
    errors: dict[str, list[float]] = {}
    latencies: list[float] = []
    long_tail: list[float] = []
    ild: list[float] = []
    failures: list[str] = []

    for q in queries:
        playlist = engine.generate(q, write_copy=False)
        if not playlist.tracks:
            failures.append(f"{q!r}: returned no tracks")
            continue
        latencies.append(playlist.timings_ms.get("total", 0.0))
        if playlist.stats.long_tail_share is not None:
            long_tail.append(playlist.stats.long_tail_share)
        if playlist.stats.intra_list_distance is not None:
            ild.append(playlist.stats.intra_list_distance)

        for name, ok in playlist.constraint_report.items():
            hits.setdefault(name, []).append(bool(ok))
            if not ok:
                failures.append(f"{q!r}: {name} not satisfied")

        # Mood adherence: |delivered − requested| per stated audio dimension.
        active = playlist.intent.audio.active()
        idx = np.array([t.track.index for t in playlist.tracks], dtype=np.int64)
        for dim, target in active.items():
            if dim not in AUDIO_FEATURE_COLS:
                continue
            values = engine.catalog.col(dim)[idx]
            values = values[np.isfinite(values)]
            if values.size:
                errors.setdefault(dim, []).append(abs(float(values.mean()) - target))

    report = ConstraintReport(
        n_queries=len(queries),
        satisfaction={
            k: {"rate": float(np.mean(v)), "n": float(len(v))} for k, v in sorted(hits.items())
        },
        mood_error={k: round(float(np.mean(v)), 4) for k, v in sorted(errors.items())},
        mean_latency_ms=round(float(np.mean(latencies)), 1) if latencies else 0.0,
        mean_long_tail_share=round(float(np.mean(long_tail)), 4) if long_tail else 0.0,
        mean_intra_list_distance=round(float(np.mean(ild)), 4) if ild else 0.0,
        failures=failures,
    )
    if verbose:
        print(json.dumps(report.__dict__, indent=2))
    return report


def main(out: Path = ARTIFACTS / "constraint_report.json") -> ConstraintReport:
    from ..catalog import Catalog
    from ..engine import CadenceEngine
    from ..models.reranker import Reranker

    catalog = Catalog.load()
    reranker = Reranker.load() if (ARTIFACTS / "reranker.pkl").exists() else None
    engine = CadenceEngine(catalog, reranker=reranker)
    report = run(engine)
    out.write_text(json.dumps(report.__dict__, indent=2))
    return report
