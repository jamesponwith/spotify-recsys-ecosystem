"""Paths, seed and audit parameters.

Gamut is the fourth application in the ecosystem and the only one that measures
the *supply* side. Cadence, Timbre and Segue all ask whether the listener was
served well. None of them asks which artists were served at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cadence.config import RetrievalConfig

SEED = 20260815

# How deep Cadence's candidate pool actually is. Read from Cadence's own config
# rather than copied into this file, because a copy is exactly what went wrong:
# this module said `depth = 100` while `fused_candidates` was 1500, so the audit
# measured the top 100 of a 1500-deep pool and the report published that window
# as "reached by retrieval" -- a funnel stage measured 15x shallower than the
# stage it was named after. A derived constant cannot drift; a copied one did.
RETRIEVE_DEPTH = RetrievalConfig().fused_candidates

# The prefix of that pool `cadence.assemble.select.select` actually scores (its
# `pool_size` default). A candidate ranked past it cannot be shown however the
# pool is re-ordered, so this is a real funnel stage rather than an arbitrary
# window. It is a keyword default rather than a config field, so it cannot be
# derived the way RETRIEVE_DEPTH is -- `tests/test_funnel.py` pins it instead.
SELECT_POOL = 500

GAMUT_ROOT = Path(__file__).resolve().parents[2]
CADENCE_ROOT = GAMUT_ROOT.parent / "cadence"
CADENCE_PROCESSED = CADENCE_ROOT / "data" / "processed"
CADENCE_ARTIFACTS = CADENCE_ROOT / "artifacts"
ARTIFACTS = GAMUT_ROOT / "artifacts"

# Cadence's seven candidate-generation channels, audited individually so the
# report can name which ones concentrate exposure rather than blaming "the
# system" as a whole.
CHANNELS = (
    "collaborative",
    "cooccurrence",
    "tag",
    "tag_exact",
    "lexical",
    "audio",
    "popularity",
)


@dataclass(frozen=True)
class AuditConfig:
    # Queries drawn from Cadence's own held-out challenge set, title-only (k=0):
    # the cold natural-language case the system is really for, and the one where
    # channel choice decides everything.
    n_queries: int = 400
    # Depth at which the candidate pool is cached and counted. A track that never
    # enters the candidate set cannot be exposed by any downstream re-ranking, so
    # the audit measures the funnel at the point where the loss actually happens
    # -- which means this has to be the depth retrieval really returns, not a
    # shallower window that gets labelled as if it were.
    depth: int = RETRIEVE_DEPTH
    # Where exposure is actually counted. The candidate pool is `depth`; what the
    # listener sees is `cut`. Measuring exposure over the pool would be
    # invariant to re-ranking -- reordering a set does not change the set -- so
    # any intervention would appear to do nothing. Exposure is a property of what
    # gets shown, not of what got retrieved.
    cut: int = 20
    # The long tail, by playlist count. Cadence's own catalog is already filtered
    # at >= 4 playlists, so this is the tail *of the survivors*.
    tail_percentile: float = 50.0
    head_percentile: float = 90.0
    # Exposure-penalty strengths swept to build the accuracy/exposure frontier.
    penalties: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    seed: int = SEED
