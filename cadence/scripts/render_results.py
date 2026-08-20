#!/usr/bin/env python
"""Render artifacts/eval_report.json into the markdown tables used in the docs.

Generating the tables rather than transcribing them means the numbers in the
documentation cannot drift from the numbers the harness produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADLINE = ["r_precision", "r_precision_artist", "ndcg_100", "clicks", "recall_500"]
PRETTY = {
    "r_precision": "R-prec",
    "r_precision_artist": "R-prec (artist)",
    "ndcg_100": "NDCG@100",
    "clicks": "Clicks ↓",
    "recall_500": "Recall@500",
    "coverage_100": "Coverage",
    "latency_p50_ms": "p50 ms",
    "latency_p95_ms": "p95 ms",
}
SYSTEM_ORDER = [
    "full_reranked",
    "full_fusion",
    "item_knn",
    "popularity",
    "lexical_title",
]
ABLATION_ORDER = [
    "full_fusion",
    "no_cooccurrence",
    "no_collaborative",
    "no_tag",
    "no_lexical",
    "no_audio",
    "only_cooccurrence",
    "only_collaborative",
    "only_tag",
]
LABEL = {
    "full_reranked": "**Cadence (reranked)**",
    "full_fusion": "Cadence (fusion only)",
    "item_knn": "item-kNN baseline",
    "popularity": "popularity baseline",
    "lexical_title": "lexical-title baseline",
    "no_cooccurrence": "− exact co-occurrence",
    "no_collaborative": "− collaborative embedding",
    "no_tag": "− folksonomy tags",
    "no_lexical": "− lexical",
    "no_audio": "− audio",
    "only_cooccurrence": "only co-occurrence",
    "only_collaborative": "only collaborative",
    "only_tag": "only folksonomy tags",
}


def fmt(value: float, metric: str) -> str:
    return f"{value:.2f}" if metric == "clicks" else f"{value:.4f}"


def main() -> int:
    path = ROOT / "artifacts" / "eval_report.json"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    results = report["results"]
    ks = sorted(results, key=int)

    out: list[str] = []

    out.append("### Main results by seed count\n")
    out.append(
        "`k` = number of seed tracks revealed. **k=0 is the pure "
        "natural-language cold-start case**: title only, nothing for "
        "collaborative filtering to use.\n"
    )
    for k in ks:
        cell = results[k]
        out.append(f"\n**k = {k} seed tracks**\n")
        header = "| System | " + " | ".join(PRETTY[m] for m in HEADLINE) + " |"
        out.append(header)
        out.append("|" + "---|" * (len(HEADLINE) + 1))
        for name in SYSTEM_ORDER:
            if name not in cell:
                continue
            row = [LABEL.get(name, name)]
            row += [fmt(cell[name][m], m) for m in HEADLINE]
            out.append("| " + " | ".join(row) + " |")

    out.append("\n\n### Channel ablations\n")
    out.append("R-precision; each row removes or isolates one retrieval channel.\n")
    header = "| Configuration | " + " | ".join(f"k={k}" for k in ks) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(ks) + 1))
    for name in ABLATION_ORDER:
        if not any(name in results[k] for k in ks):
            continue
        row = [LABEL.get(name, name)]
        for k in ks:
            v = results[k].get(name)
            row.append(fmt(v["r_precision"], "r_precision") if v else "—")
        out.append("| " + " | ".join(row) + " |")

    out.append("\n\n### Beyond-accuracy and latency\n")
    out.append("| System | k | Coverage@100 | Gini@100 | p50 ms | p95 ms |")
    out.append("|---|---|---|---|---|---|")
    for k in ks:
        for name in ("full_reranked", "full_fusion", "popularity"):
            v = results[k].get(name)
            if not v:
                continue
            out.append(
                f"| {LABEL.get(name, name)} | {k} | {v.get('coverage_100', 0):.4f} | "
                f"{v.get('gini_100', 0):.3f} | {v.get('latency_p50_ms', 0):.0f} | "
                f"{v.get('latency_p95_ms', 0):.0f} |"
            )

    meta = report["meta"]
    out.append(
        f"\n\nEvaluated on {meta['limit_per_cell']} held-out playlists per cell, "
        f"retrieval depth {meta['depth']}, catalog {meta['n_tracks']:,} tracks."
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
