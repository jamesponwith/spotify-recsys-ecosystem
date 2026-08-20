"""The joint demo: Cadence and Timbre in one loop.

The experiment is a *simulated release*. Take the tracks Cadence itself picks
for a natural-language query, then delete everything Cadence knows about them
that came from playlist history -- collaborative vectors, co-occurrence,
folksonomy embedding, popularity. That is precisely the state of a track
uploaded this morning. Re-run the same query and the tracks are gone.

Then hand Cadence one thing back: a folksonomy embedding predicted from audio
alone, and nothing else. Re-run again and count how many return.

What this establishes: that a content-predicted embedding is a working substitute
for the history-derived one *inside the real retrieval stack*, not in a notebook.
What it does not establish: that the recovered tracks are the objectively right
answer. The target set is "what Cadence chose when warm", which makes Cadence its
own ground truth. That is a demo, not an evaluation -- Rung 3 in
docs/EVALUATION.md is the experiment with a real held-out target.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from cadence.catalog import Catalog
from cadence.config import DEFAULT
from cadence.engine import CadenceEngine
from cadence.retrieval.ann import DenseIndex
from scipy import sparse

from .predictor import TagPredictor

# The channels that are derived from playlist history, and therefore exactly the
# ones a brand-new track has none of. Lexical and audio are content-based and
# survive a cold start unaided -- leaving them intact is what keeps the demo
# honest about how much of the recovery Timbre is actually responsible for.
HISTORY_CHANNELS = ("collaborative", "cooccurrence", "tag", "tag_exact")


@dataclass
class SystemResult:
    name: str
    ranks: dict[int, int]  # cold track index -> rank in fused candidates (1-based)
    in_top_n: int
    recovered_fraction: float
    median_rank: float | None


@dataclass
class DemoResult:
    query: str
    n_cold: int
    top_n: int
    cold_tracks: list[dict]
    systems: list[SystemResult]

    @property
    def by_name(self) -> dict[str, SystemResult]:
        return {s.name: s for s in self.systems}


def _zero_rows(matrix: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
    keep = np.ones(matrix.shape[0], dtype=np.float32)
    keep[rows] = 0.0
    return (sparse.diags(keep) @ matrix).tocsr()


def _zero_cols(matrix: sparse.csr_matrix, cols: np.ndarray) -> sparse.csr_matrix:
    keep = np.ones(matrix.shape[1], dtype=np.float32)
    keep[cols] = 0.0
    return (matrix @ sparse.diags(keep)).tocsr()


def freeze_out(catalog: Catalog, cold: np.ndarray) -> Catalog:
    """Return a catalog in which ``cold`` tracks have no playlist history at all.

    Zeroing a row is not a cosmetic edit: ``DenseIndex`` marks all-zero rows dead
    and scores them ``-inf``, so a frozen-out track is genuinely unreachable
    through that channel rather than merely down-weighted.
    """
    collab = catalog.collab.vectors.copy()
    collab[cold] = 0.0
    tag_tracks = catalog.tag_tracks.vectors.copy()
    tag_tracks[cold] = 0.0
    popularity = catalog.popularity.copy()
    popularity[cold] = 0.0

    frame = catalog.frame.copy()
    frame.loc[frame.index[cold], "n_playlists"] = 0

    return dataclasses.replace(
        catalog,
        frame=frame,
        collab=DenseIndex(collab),
        tag_tracks=DenseIndex(tag_tracks),
        popularity=popularity,
        tag_matrix=_zero_rows(catalog.tag_matrix, cold),
        interactions=_zero_cols(catalog.interactions, cold),
    )


def graft(catalog: Catalog, cold: np.ndarray, predicted: np.ndarray) -> Catalog:
    """Give the cold tracks a folksonomy embedding predicted from audio."""
    tag_tracks = catalog.tag_tracks.vectors.copy()
    tag_tracks[cold] = predicted
    return dataclasses.replace(catalog, tag_tracks=DenseIndex(tag_tracks))


def _ranks(engine: CadenceEngine, intent, cold: np.ndarray) -> dict[int, int]:
    trace = engine.retrieve(intent)
    order = trace.candidates.indices
    position = {int(t): i + 1 for i, t in enumerate(order)}
    return {int(c): position.get(int(c), 0) for c in cold}


def run_demo(
    query: str,
    *,
    n_cold: int = 20,
    top_n: int = 200,
    catalog: Catalog | None = None,
    predictor: TagPredictor | None = None,
) -> DemoResult:
    catalog = catalog or Catalog.load()
    predictor = predictor or TagPredictor.load()
    engine = CadenceEngine(catalog, cfg=DEFAULT)

    plan = engine.planner.plan(query, engine.known_tags)
    intent = plan.intent

    # 1. Warm: what Cadence picks today. These become the cold set.
    warm_trace = engine.retrieve(intent)
    cold = warm_trace.candidates.indices[:n_cold].astype(np.int64)
    warm_ranks = {int(c): i + 1 for i, c in enumerate(cold)}

    # 2. Cold: same query, no history for those tracks.
    frozen = freeze_out(catalog, cold)
    cold_ranks = _ranks(CadenceEngine(frozen, planner=engine.planner, cfg=DEFAULT), intent, cold)

    # 3. Timbre: hand back one audio-predicted embedding each.
    predicted = predictor.predict_frame(catalog.frame.iloc[cold])
    grafted = graft(frozen, cold, predicted)
    timbre_ranks = _ranks(CadenceEngine(grafted, planner=engine.planner, cfg=DEFAULT), intent, cold)

    names = catalog.col("name")
    artists = catalog.col("artist")
    cold_tracks = [
        {
            "index": int(c),
            "name": str(names[c]),
            "artist": str(artists[c]),
            "warm_rank": warm_ranks[int(c)],
            "cold_rank": cold_ranks[int(c)],
            "timbre_rank": timbre_ranks[int(c)],
        }
        for c in cold
    ]

    systems = []
    for name, ranks in (
        ("warm", warm_ranks),
        ("cold", cold_ranks),
        ("timbre", timbre_ranks),
    ):
        hits = [r for r in ranks.values() if 0 < r <= top_n]
        systems.append(
            SystemResult(
                name=name,
                ranks=ranks,
                in_top_n=len(hits),
                recovered_fraction=len(hits) / len(cold),
                median_rank=float(np.median(hits)) if hits else None,
            )
        )

    return DemoResult(
        query=query,
        n_cold=int(cold.size),
        top_n=top_n,
        cold_tracks=cold_tracks,
        systems=systems,
    )
