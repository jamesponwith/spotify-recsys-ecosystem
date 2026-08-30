#!/usr/bin/env python
"""Render artifacts/eval_report.json into the markdown tables used in the docs.

Generating the tables rather than transcribing them means the numbers in the
documentation cannot drift from the numbers the harness produced.

Every number is printed with its ±2×SE band. The harness has always written the
standard errors; this script used to read them and drop them, which is how the
docs came to quote four decimals with no indication that the last two were
noise. A band beside every figure makes the detection floor impossible to miss,
which is all this does — it does not make the harness one bit more sensitive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cadence.eval.metrics import BAND_Z, detection_floor, within_band

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
ABLATION_BASE = "full_fusion"
ABLATION_ORDER = [
    ABLATION_BASE,
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
# Marks an ablation cell whose difference from the fusion row is inside the band.
IN_BAND = "≈"


def fmt(cell: dict, metric: str) -> str:
    """`mean ± 2×SE`, at the precision the metric is conventionally read at."""
    digits = 2 if metric == "clicks" else 4
    value, se = cell[metric], cell.get(f"{metric}_se", 0.0)
    return f"{value:.{digits}f} ± {BAND_Z * se:.{digits}f}"


def in_band(cell: dict, base: dict, metric: str) -> bool:
    return within_band(cell[metric], cell[f"{metric}_se"], base[metric], base[f"{metric}_se"])


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
        f"collaborative filtering to use. Every cell is mean ± {BAND_Z:.0f}×SE.\n"
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
            row += [fmt(cell[name], m) for m in HEADLINE]
            out.append("| " + " | ".join(row) + " |")

    out.append("\n\n### Channel ablations\n")
    out.append(
        f"R-precision ± {BAND_Z:.0f}×SE; each row removes or isolates one retrieval "
        f"channel. `{IN_BAND}` marks a cell whose difference from the fusion row is "
        "inside the band of that difference — at this sample size the change is "
        "not distinguishable from noise.\n"
    )
    header = "| Configuration | " + " | ".join(f"k={k}" for k in ks) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(ks) + 1))
    n_removed = n_removed_in_band = 0
    for name in ABLATION_ORDER:
        if not any(name in results[k] for k in ks):
            continue
        row = [LABEL.get(name, name)]
        for k in ks:
            v = results[k].get(name)
            base = results[k].get(ABLATION_BASE)
            if not v:
                row.append("—")
                continue
            text = fmt(v, "r_precision")
            if name != ABLATION_BASE and base and in_band(v, base, "r_precision"):
                text += f" {IN_BAND}"
                n_removed_in_band += name.startswith("no_")
            n_removed += name.startswith("no_")
            row.append(text)
        out.append("| " + " | ".join(row) + " |")
    if n_removed:
        out.append(
            f"\n{n_removed_in_band} of {n_removed} `−` cells sit inside their own band: "
            "removing that channel cannot be told from noise in this report."
        )

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
    # Reports written before the harness stamped the floor still carry the SE
    # it is derived from, so derive it the same way rather than print nothing.
    floor = detection_floor(results)
    floor_value = meta.get("detection_floor", floor["value"])
    out.append(
        f"\n\nEvaluated on {meta['limit_per_cell']} held-out playlists per cell, "
        f"retrieval depth {meta['depth']}, catalog {meta['n_tracks']:,} tracks. "
        f"**Detection floor: {floor_value:.4f} R-precision** — {BAND_Z:.0f}×SE of the "
        f"k={floor['k']} `{floor['system']}` cell. A difference smaller than that "
        "cannot be distinguished from sampling noise anywhere in this report."
    )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
