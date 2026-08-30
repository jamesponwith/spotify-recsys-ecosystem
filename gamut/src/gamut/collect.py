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

# The one value that means "this channel never returned this candidate" once
# the retrieval output is inside Gamut. Everything downstream filters on it.
ABSENT = -1


def strip_sentinel(ranks: np.ndarray) -> np.ndarray:
    """Map fusion's absent-candidate sentinel to ABSENT.

    Cadence's RRF marks a candidate a channel never returned with the rank
    ``channel_depth + 1`` -- deliberately, so its learned reranker can tell
    "ranked last" from "never seen". The audit needs the opposite contract:
    a rank is only meaningful if the channel actually produced it. Left in
    place, the sentinel passes any ``rank >= 0`` filter and every channel
    block silently becomes the whole pool re-sorted.

    Real ranks are unique within a query (each rank belongs to exactly one
    candidate) while every absent candidate shares the one sentinel value, so
    a duplicated maximum is the sentinel. A sentinel that appears exactly once
    is ``channel_depth + 1`` with all other pool candidates present, hence at
    least the pool size -- whereas a channel covering the whole pool has ranks
    forming a permutation whose maximum is the pool size minus one. The only
    misfire is a full-coverage channel whose deepest rank survived fused
    truncation; dropping that one candidate cannot reach any audited depth.
    """
    cleaned = np.asarray(ranks).astype(np.int32)
    if cleaned.size == 0:
        return cleaned
    top = cleaned.max()
    if top < 0:
        return cleaned
    hits = cleaned == top
    if hits.sum() > 1 or top >= cleaned.size:
        cleaned[hits] = ABSENT
    return cleaned


@dataclass
class Collected:
    """Cached retrieval output, one row block per query."""

    indices: np.ndarray  # (n_queries, depth) int32, ABSENT padded
    scores: np.ndarray  # (n_queries, depth) float32, fused RRF score
    channel_ranks: np.ndarray  # (n_channels, n_queries, depth) int32, ABSENT = absent
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
        if _contaminated(z["channel_ranks"]):
            raise ValueError(
                f"{path} predates the sentinel fix: its per-channel ranks contain "
                "duplicate values, which can only be fusion's absent-candidate "
                "sentinel. Re-run `gamut collect` before auditing."
            )
        return cls(
            indices=z["indices"],
            scores=z["scores"],
            channel_ranks=z["channel_ranks"],
            truth=[set(json.loads(s)) for s in z["truth"]],
            titles=list(z["titles"]),
        )


def _contaminated(channel_ranks: np.ndarray) -> bool:
    """True if any stored row still carries fusion's sentinel.

    Real ranks are unique per (channel, query); only the shared sentinel can
    repeat. This is what lets a pre-fix cache be refused instead of silently
    reproducing the published per-channel rows.
    """
    for channel in channel_ranks:
        for row in channel:
            real = row[row >= 0]
            if real.size != np.unique(real).size:
                return True
    return False


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

    indices = np.full((len(picked), d), ABSENT, dtype=np.int32)
    scores = np.zeros((len(picked), d), dtype=np.float32)
    ranks = np.full((len(CHANNELS), len(picked), d), ABSENT, dtype=np.int32)
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
                # Strip over the full pool: sentinel detection needs every row
                # entry, not just the slice that fits the audit depth.
                ranks[ci, qi, :n] = strip_sentinel(cr)[:n]
        truth.append({int(t) for t in ch["held_out"]})
        titles.append(title)
        if verbose and (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(picked)} queries", flush=True)

    out = Collected(indices, scores, ranks, truth, titles)
    if verbose:
        print(f"collected {len(picked)} queries in {time.perf_counter() - t0:.0f}s")
    return out
