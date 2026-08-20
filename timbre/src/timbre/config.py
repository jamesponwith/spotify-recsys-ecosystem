"""Paths, seeds and hyperparameters for Timbre.

Timbre is a satellite of Cadence: it consumes Cadence's built catalog and
trained spaces rather than rebuilding them. The path constants below are the
only place that coupling is expressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Same seed as Cadence, deliberately: Phase 0 reuses Cadence's holdout playlists,
# and two different seeds would make it ambiguous which split a number came from.
SEED = 20260815

TIMBRE_ROOT = Path(__file__).resolve().parents[2]
CADENCE_ROOT = TIMBRE_ROOT.parent / "cadence"
CADENCE_PROCESSED = CADENCE_ROOT / "data" / "processed"
CADENCE_ARTIFACTS = CADENCE_ROOT / "artifacts"
ARTIFACTS = TIMBRE_ROOT / "artifacts"
DOCS = TIMBRE_ROOT / "docs"

# The 11 descriptors Spotify exposes. Cadence's own AUDIO_FEATURE_COLS is a
# 7-column subset -- it drops key/mode as categorical and omits tempo/loudness
# from the standardised block. Phase 0 wants the full set, because the question
# is what acoustic description *in total* can predict.
AUDIO_CONTINUOUS_COLS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
]
AUDIO_CATEGORICAL_COLS = ["key", "mode"]
AUDIO_RAW_COLS = AUDIO_CONTINUOUS_COLS + AUDIO_CATEGORICAL_COLS


@dataclass(frozen=True)
class Phase0Config:
    """The falsification test. No audio, no torch -- see docs/EVALUATION.md."""

    test_fraction: float = 0.2
    recall_k: int = 100
    ridge_alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(10.0**e for e in range(-2, 5))
    )
    mlp_hidden: tuple[int, int] = (256, 256)
    # Measured, not guessed: with sklearn's default batch of 200 this is 637
    # minibatches per epoch over 127k rows, and an uncapped run passed 95 minutes
    # without triggering early stopping. That defeats the purpose of a gate whose
    # entire value is being cheap, so the epoch count is capped and the batch is
    # widened. The cap is a documented limitation of the MLP number, not a tuned
    # optimum -- ridge is the unbounded, fully-converged reference.
    mlp_max_iter: int = 60
    mlp_batch_size: int = 512
    mlp_early_stopping: bool = True
    # Ridge alone reaches a verdict in ~5 minutes end to end. For a gate whose
    # purpose is to be cheap, that is a first-class mode, not a debug shortcut.
    skip_mlp: bool = False
    # Query-side scoring is a dense matmul against the whole catalog; this caps
    # the peak score matrix at roughly 160k x 256 float32 (~160 MB).
    query_block: int = 256
    seed: int = SEED

    # Gate 0, from docs/EVALUATION.md. Stored here so the code that rules on the
    # gate cannot silently disagree with the document that defines it.
    gate_random_multiple: float = 3.0
    gate_oracle_fraction: float = 0.25
