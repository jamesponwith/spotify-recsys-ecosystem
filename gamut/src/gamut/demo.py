"""Show the audit on a single query, where an aggregate cannot be argued with.

Gini and coverage are properties of a whole run; they are the right way to state
the problem and a poor way to *see* it. This renders one query: what Cadence
shows today, what the exposure-aware re-rank shows instead, and which tracks
changed places -- with the popularity of each one visible, so the trade is
legible rather than asserted.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .collect import Collected
from .config import ARTIFACTS, CADENCE_PROCESSED, AuditConfig
from .exposure import CatalogFacts
from .rerank import popularity_norm, rerank


def _rows(order: np.ndarray, frame_cols: dict, facts: CatalogFacts, n: int) -> list[dict]:
    out = []
    for rank, t in enumerate(order[:n], start=1):
        t = int(t)
        out.append(
            {
                "rank": rank,
                "index": t,
                "name": str(frame_cols["name"][t]),
                "artist": str(frame_cols["artist"][t]),
                "playlists": int(facts.play_counts[t]),
                "tail": bool(facts.tail_mask[t]),
            }
        )
    return out


def _summary(order: np.ndarray, facts: CatalogFacts, n: int) -> dict[str, Any]:
    top = order[:n].astype(np.int64)
    return {
        "tail_share": float(facts.tail_mask[top].mean()),
        "distinct_artists": int(np.unique(facts.artists[top]).size),
        "median_playlists": float(np.median(facts.play_counts[top])),
    }


def build_many(indices: tuple[int, ...], penalty: float = 0.3, n: int = 10) -> dict[str, Any]:
    """Several cases at once, for the report.

    The pairing is the point: a thematic query, where concentration is a defect
    the penalty can fix, next to an artist-name query, where concentration is the
    correct answer and the penalty is actively unhelpful. An audit that reports
    only the first would be recommending a change that breaks the second.
    """
    cases = [build(index=i, penalty=penalty, n=n, persist=False) for i in indices]
    out = {"cases": cases, "penalty": penalty, "n": n}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "demo.json").write_text(json.dumps(out, indent=2))
    return out


def build(
    index: int = 0,
    penalty: float = 0.3,
    n: int = 10,
    cfg: AuditConfig | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    cfg = cfg or AuditConfig()
    collected = Collected.load()
    frame = pd.read_parquet(CADENCE_PROCESSED / "tracks.parquet")
    facts = CatalogFacts.build(frame, cfg.tail_percentile, cfg.head_percentile)
    cols = {
        "name": frame["name"].to_numpy(dtype=object),
        "artist": frame["artist"].to_numpy(dtype=object),
    }
    pop = popularity_norm(facts.play_counts)

    i = index % len(collected.titles)
    row = collected.indices[i : i + 1]
    before = row[0][row[0] >= 0]
    after_block = rerank(row, collected.scores[i : i + 1], pop, penalty)
    after = after_block[0][after_block[0] >= 0]

    before_top = {int(x) for x in before[:n]}
    after_rows = _rows(after, cols, facts, n)
    for r in after_rows:
        r["new"] = r["index"] not in before_top

    out = {
        "query": collected.titles[i],
        "penalty": penalty,
        "n": n,
        "before": {"rows": _rows(before, cols, facts, n), **_summary(before, facts, n)},
        "after": {"rows": after_rows, **_summary(after, facts, n)},
        "held_out": len(collected.truth[i]),
    }
    if persist:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "demo.json").write_text(json.dumps(out, indent=2))
    return out
