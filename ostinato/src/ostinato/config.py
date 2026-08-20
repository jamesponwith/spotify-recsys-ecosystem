"""Simulation parameters.

Gamut measured exposure at one instant. A recommender is not a static object:
it is trained on interaction data that its own past output helped create. This
simulates closing that loop and asks whether concentration is a fixed property
of the system or a process that runs away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 20260815

OSTINATO_ROOT = Path(__file__).resolve().parents[2]
CADENCE_ROOT = OSTINATO_ROOT.parent / "cadence"
CADENCE_PROCESSED = CADENCE_ROOT / "data" / "processed"
CADENCE_ARTIFACTS = CADENCE_ROOT / "artifacts"
ARTIFACTS = OSTINATO_ROOT / "artifacts"

# The three worlds being compared. `organic` is the control that makes the
# others interpretable: it adds exactly as many new playlists per round, drawn
# from the catalog's own popularity distribution rather than from anything the
# recommender said. Without it, any drift could be blamed on simply having more
# data.
ARMS = ("organic", "closed_loop", "exposure_aware")


@dataclass(frozen=True)
class SimConfig:
    rounds: int = 6
    queries_per_round: int = 150
    playlist_len: int = 20
    # Position-based acceptance. A listener is far likelier to keep what is at
    # the top, which is the mechanism that turns a ranking bias into a data
    # bias -- and then into a training bias on the next round.
    accept_base: float = 0.55
    position_decay: float = 0.12
    # How much of new listening the recommender is assumed to drive. Each
    # accepted playlist is folded in with this weight.
    #
    # This exists because the first run was underpowered and the control proved
    # it: at dose 1, one round adds ~660 interactions to a corpus of 5.88M --
    # 0.011%, or 0.056% across five rounds. The organic arm's round-to-round
    # standard deviation in artist Gini is 0.0012 with a net five-round drift of
    # +0.0005, so any effect below ~0.003 is indistinguishable from the noise of
    # refitting a randomised SVD. Running that experiment to completion would
    # have produced a confident-looking null that was really a power failure.
    #
    # 25 puts a round at ~0.28% of the corpus and five rounds at ~1.4%, which is
    # a defensible stand-in for a service where the recommender drives a
    # meaningful minority of listening -- and is above the measured noise floor.
    dose: int = 25
    # Strength of the Gamut popularity penalty in the `exposure_aware` arm.
    # 0.3 was the cheap operating point on that project's frontier: +5.9 points
    # of long-tail share for a 15% accuracy cost.
    penalty: float = 0.3
    tail_percentile: float = 50.0
    head_percentile: float = 90.0
    seed: int = SEED
