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


def strip_sentinel(ranks: np.ndarray, depth: int) -> np.ndarray:
    """Map fusion's absent-candidate sentinel to ABSENT.

    Cadence's RRF marks a candidate a channel never returned with the rank
    ``depth + 1`` -- deliberately, so its learned reranker can tell "ranked
    last" from "never seen". The audit needs the opposite contract: left in
    place, the sentinel passes any ``rank >= 0`` filter and every channel
    block silently becomes the whole pool re-sorted. Real ranks are
    ``0 .. depth - 1``, so anything at or past the channel's depth is absent.
    """
    cleaned = np.asarray(ranks).astype(np.int32)
    cleaned[cleaned >= depth] = ABSENT
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
            # Format marker: ranks in this cache had fusion's absent sentinel
            # stripped at collection. load() refuses caches without it.
            sentinel_stripped=np.True_,
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None, min_depth: int | None = None) -> Collected:
        """Load the cache, refusing one too narrow for the depth being reported.

        ``min_depth`` is what the caller is about to *publish*. A narrower cache
        is not a smaller sample of the same window, it is a different window
        wearing the label -- so the width is checked rather than assumed.
        """
        path = path or ARTIFACTS / "collected.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `gamut collect` first.")
        z = np.load(path, allow_pickle=True)
        if "sentinel_stripped" not in z:
            raise ValueError(
                f"{path} predates the sentinel fix: its per-channel ranks treat "
                "fusion's absent-candidate sentinel as a real rank, so every "
                "channel row is the whole pool. Re-run `gamut collect`."
            )
        # All three blocks are written at one width by `save`, so a disagreement
        # means a hand-assembled cache. Checking only `indices` would let a
        # 1500-wide pool carry 100-wide channel ranks, and every channel row
        # would ship labelled with a depth it was never measured at.
        widths = {name: int(z[name].shape[-1]) for name in ("indices", "scores", "channel_ranks")}
        if len(set(widths.values())) != 1:
            raise ValueError(
                f"{path} has blocks of disagreeing width {widths}. Re-run `gamut collect`."
            )
        cached_depth = widths["indices"]
        if min_depth is not None and cached_depth < min_depth:
            raise ValueError(
                f"{path} was collected {cached_depth} deep but the audit reports "
                f"depth {min_depth}. Every figure derived from it would describe "
                f"the top {cached_depth} of the pool while being labelled as the "
                "whole of it. Re-run `gamut collect`."
            )
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
                ranks[ci, qi, :n] = strip_sentinel(cr, fused.channel_depths[name])[:n]
        truth.append({int(t) for t in ch["held_out"]})
        titles.append(title)
        if verbose and (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(picked)} queries", flush=True)

    out = Collected(indices, scores, ranks, truth, titles)
    if verbose:
        print(f"collected {len(picked)} queries in {time.perf_counter() - t0:.0f}s")
    return out
