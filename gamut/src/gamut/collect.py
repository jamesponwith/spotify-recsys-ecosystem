"""Run Cadence over a query battery and cache what it surfaced.

One retrieval pass funds the whole audit. `FusedCandidates` already carries the
per-channel rank of every candidate, so caching it once lets the per-channel
attribution *and* the re-ranking sweep run offline over the same evidence --
rather than re-querying the engine for each of the nine penalty strengths, which
would take a hundred times as long and introduce run-to-run drift between
conditions that are supposed to differ only in the penalty.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ARTIFACTS, CADENCE_PROCESSED, CHANNELS, AuditConfig


@dataclass
class Collected:
    """Cached retrieval output, one row block per query."""

    indices: np.ndarray  # (n_queries, depth) int32, -1 padded
    scores: np.ndarray  # (n_queries, depth) float32, fused RRF score
    channel_ranks: np.ndarray  # (n_channels, n_queries, depth) int32, -1 = absent
    truth: list[set[int]]
    titles: list[str]

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACTS / "collected.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            indices=self.indices,
            scores=self.scores,
            channel_ranks=self.channel_ranks,
            titles=np.array(self.titles, dtype=object),
            truth=np.array([json.dumps(sorted(t)) for t in self.truth], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> Collected:
        path = path or ARTIFACTS / "collected.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `gamut collect` first.")
        z = np.load(path, allow_pickle=True)
        return cls(
            indices=z["indices"],
            scores=z["scores"],
            channel_ranks=z["channel_ranks"],
            truth=[set(json.loads(s)) for s in z["truth"]],
            titles=list(z["titles"]),
        )


def collect(cfg: AuditConfig | None = None, verbose: bool = True) -> Collected:
    from cadence.catalog import Catalog
    from cadence.engine import CadenceEngine

    cfg = cfg or AuditConfig()
    t0 = time.perf_counter()
    challenges = json.loads((CADENCE_PROCESSED / "challenges.json").read_text())["0"]
    rng = np.random.default_rng(cfg.seed)
    picked = [challenges[i] for i in rng.permutation(len(challenges))[: cfg.n_queries]]

    catalog = Catalog.load()
    engine = CadenceEngine(catalog)
    d = cfg.depth

    indices = np.full((len(picked), d), -1, dtype=np.int32)
    scores = np.zeros((len(picked), d), dtype=np.float32)
    ranks = np.full((len(CHANNELS), len(picked), d), -1, dtype=np.int32)
    truth: list[set[int]] = []
    titles: list[str] = []

    for qi, ch in enumerate(picked):
        title = str(ch["title"])
        intent = engine.planner.plan(title, engine.known_tags).intent
        fused = engine.retrieve(intent).candidates
        n = min(d, len(fused))
        indices[qi, :n] = fused.indices[:n]
        scores[qi, :n] = fused.scores[:n]
        for ci, name in enumerate(CHANNELS):
            cr = fused.channel_ranks.get(name)
            if cr is not None:
                ranks[ci, qi, :n] = cr[:n]
        truth.append({int(t) for t in ch["held_out"]})
        titles.append(title)
        if verbose and (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(picked)} queries", flush=True)

    out = Collected(indices, scores, ranks, truth, titles)
    if verbose:
        print(f"collected {len(picked)} queries in {time.perf_counter() - t0:.0f}s")
    return out
