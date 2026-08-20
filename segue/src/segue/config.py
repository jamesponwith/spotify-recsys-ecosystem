"""Paths, seed and hyperparameters.

Segue is the third application in the ecosystem and, like Timbre, a satellite of
Cadence: it consumes Cadence's catalog and its trained collaborative space rather
than rebuilding either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 20260815

SEGUE_ROOT = Path(__file__).resolve().parents[2]
CADENCE_ROOT = SEGUE_ROOT.parent / "cadence"
CADENCE_RAW = CADENCE_ROOT / "data" / "raw"
CADENCE_PROCESSED = CADENCE_ROOT / "data" / "processed"
CADENCE_ARTIFACTS = CADENCE_ROOT / "artifacts"
ARTIFACTS = SEGUE_ROOT / "artifacts"


@dataclass(frozen=True)
class SegueConfig:
    # How many of the most recent tracks the model sees, in order. Five is the
    # smallest window that spans the seed counts the RecSys 2018 task uses (1, 5,
    # 10, 25) without making the design matrix wider than the training set can
    # constrain.
    window: int = 5
    # Positions sampled to fit the transition operator. The full corpus has ~5.8M
    # usable positions; the normal equations converge long before that, and this
    # keeps the fit inside a couple of minutes.
    max_train_positions: int = 400_000
    # How far ahead the target looks. The obvious objective -- predict the very
    # next track -- is the wrong one: the RecSys task scores a system against
    # *every* withheld track, so a model trained on the immediate successor is
    # optimising a strictly narrower thing than it is judged on. The target is
    # the mean direction of the next `horizon` tracks. horizon=1 recovers the
    # naive objective and is kept as an ablation.
    horizon: int = 10
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10_000.0)
    # Candidate depth. The official Clicks metric is undefined past 500.
    top_k: int = 500
    seed_counts: tuple[int, ...] = (1, 5, 10, 25)
    n_eval: int = 2000
    seed: int = SEED
