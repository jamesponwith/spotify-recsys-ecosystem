"""Generate real playlists for the report page.

The report argues Cadence's case with metrics. Metrics are the right evidence and
a poor demonstration: nothing on the page showed what the system actually plays,
or the two things that distinguish it from a ranked list -- a grounded reason per
track, and a sequenced transition between them.

Queries are chosen to exercise different machinery: a mood, a constrained
request, and an era-plus-genre compound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadence.catalog import Catalog  # noqa: E402
from cadence.config import ARTIFACTS  # noqa: E402
from cadence.engine import CadenceEngine  # noqa: E402

QUERIES = [
    "chill acoustic songs for a rainy sunday",
    "high energy workout, nothing explicit, about 45 minutes",
    "90s alternative rock for a road trip",
]


def main() -> None:
    engine = CadenceEngine(Catalog.load())
    cases = []
    for q in QUERIES:
        p = engine.generate(q)
        cases.append(
            {
                "query": q,
                "title": p.title,
                "description": p.description,
                "n_tracks": p.stats.n_tracks,
                "duration_min": round(p.stats.total_duration_s / 60, 1),
                "mean_energy": p.stats.mean_energy,
                "mean_valence": p.stats.mean_valence,
                "constraint_report": p.constraint_report,
                "latency_ms": round(sum(p.timings_ms.values()), 1),
                "intent": {
                    "themes": list(p.intent.themes),
                    "genres": list(p.intent.genres),
                    "eras": list(p.intent.eras),
                },
                "tracks": [
                    {
                        "position": t.position,
                        "name": t.track.name,
                        "artist": t.track.artist,
                        "energy": t.track.energy,
                        "reasons": t.reasons[:2],
                        "transition_note": t.transition_note,
                    }
                    for t in p.tracks
                ],
            }
        )
        print(f"{q[:44]:46s} -> {p.stats.n_tracks:2d} tracks, {p.stats.total_duration_s / 60:.0f} min", flush=True)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "demo.json"
    out.write_text(json.dumps({"cases": cases}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
