"""Audit ``MOOD_LEXICON``'s audio targets against where humans file each word.

For every (word, dimension) pair where the lexicon word is also a tag in the
vocabulary, three numbers: the **target** the lexicon asserts, the plain mean of
that dimension over every track filed under the tag at least once
(**folksonomy_mean**), and the plain mean over the whole catalog
(**catalog_mean**) -- what you would aim at knowing nothing. A pair is *worse
than nothing* when ``|target - folksonomy| > |catalog - folksonomy|``.

The means are unweighted over tracks, not over playlist counts, so a track
filed under `chill` once counts the same as one filed there a thousand times;
that is the same population `sparse_tag_channel` retrieves from. Only tracks
with audio count, in both the per-tag and the catalog means, so the two are
over one population and the comparison is fair.

Reads only ``tracks.parquet``, ``tags.npz`` and ``tag_vocab.json`` -- no
engine, no trained spaces. What the result does and does not establish is in
``docs/FINDINGS.md`` §5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import ARTIFACTS, DATA_PROCESSED
from ..models.train import AUDIO_FEATURE_COLS
from ..planner.lexicon import MOOD_LEXICON, MoodEntry


def audit(
    frame: pd.DataFrame,
    tags: sparse.spmatrix,
    vocab: list[str],
    lexicon: dict[str, MoodEntry] = MOOD_LEXICON,
) -> dict[str, Any]:
    """Compare every lexicon (word, dimension) pair with the folksonomy.

    A word that is not a tag cannot be audited and is listed rather than
    silently dropped. Raises rather than writing NaN if the inputs disagree
    about their shape or a dimension has no audio at all.
    """
    if tags.shape != (len(frame), len(vocab)):
        raise ValueError(f"tags is {tags.shape}; expected ({len(frame)} tracks, {len(vocab)} tags)")
    tags = tags.tocsc()
    col_of = {t: i for i, t in enumerate(vocab)}
    has_audio = frame["has_audio"].to_numpy(dtype=bool)
    dims = [d for d in AUDIO_FEATURE_COLS if d in frame.columns]
    values = {d: frame[d].to_numpy(dtype=np.float64) for d in dims}
    valid = {d: has_audio & np.isfinite(values[d]) for d in dims}
    empty = [d for d in dims if not valid[d].any()]
    if empty:
        raise ValueError(f"no track has audio for {empty}; the audit would be over nothing")
    catalog_mean = {d: float(values[d][valid[d]].mean()) for d in dims}

    pairs: list[dict[str, Any]] = []
    excess: list[float] = []
    not_tags: list[str] = []
    for word, entry in lexicon.items():
        if not entry.audio:
            continue
        if word not in col_of:
            not_tags.append(word)
            continue
        filed = np.asarray(tags[:, col_of[word]].todense()).ravel() > 0
        for dim, target in entry.audio.items():
            if dim not in values:
                continue
            mask = filed & valid[dim]
            if not mask.any():
                continue
            folk = float(values[dim][mask].mean())
            cat = catalog_mean[dim]
            gap_target = abs(target - folk)
            gap_catalog = abs(cat - folk)
            pairs.append(
                {
                    "word": word,
                    "dimension": dim,
                    "target": float(target),
                    "folksonomy_mean": round(folk, 4),
                    "catalog_mean": round(cat, 4),
                    "n_tracks": int(mask.sum()),
                    "gap_target": round(gap_target, 4),
                    "gap_catalog": round(gap_catalog, 4),
                    "target_worse_than_catalog": bool(gap_target > gap_catalog),
                    # Does the target leave the catalog mean the same way the
                    # crowd does, and does it go past the crowd?
                    "direction_right": bool((target - cat) * (folk - cat) > 0),
                    "overshoots": bool((target - folk) * (folk - cat) > 0),
                }
            )
            excess.append(gap_target - gap_catalog)

    # Most damning first: how much further the target is than doing nothing.
    pairs = [p for _, p in sorted(zip(excess, pairs, strict=True), key=lambda t: -t[0])]
    by_dim: dict[str, dict[str, int]] = {}
    for p in pairs:
        row = by_dim.setdefault(p["dimension"], {"n_pairs": 0, "n_target_worse_than_catalog": 0})
        row["n_pairs"] += 1
        row["n_target_worse_than_catalog"] += p["target_worse_than_catalog"]
    return {
        "n_pairs": len(pairs),
        "n_target_worse_than_catalog": sum(p["target_worse_than_catalog"] for p in pairs),
        "n_direction_right": sum(p["direction_right"] for p in pairs),
        "n_overshoots": sum(p["overshoots"] for p in pairs),
        "median_gap_target": round(float(np.median([p["gap_target"] for p in pairs])), 4)
        if pairs
        else None,
        "median_gap_catalog": round(float(np.median([p["gap_catalog"] for p in pairs])), 4)
        if pairs
        else None,
        "n_words_audited": len({p["word"] for p in pairs}),
        "words_not_tags": sorted(not_tags),
        "by_dimension": by_dim,
        "catalog_mean": {d: round(m, 4) for d, m in catalog_mean.items()},
        "pairs": pairs,
    }


def main(
    processed_dir: Path = DATA_PROCESSED,
    out: Path = ARTIFACTS / "lexicon_calibration.json",
    verbose: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    frame = pd.read_parquet(processed_dir / "tracks.parquet")
    tags = sparse.load_npz(processed_dir / "tags.npz")
    vocab: list[str] = json.loads((processed_dir / "tag_vocab.json").read_text())
    report = audit(frame, tags, vocab)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if verbose:
        print(
            f"{report['n_target_worse_than_catalog']} of {report['n_pairs']} pairs assert a "
            f"target further from the folksonomy than the catalog mean is "
            f"({report['seconds']}s)"
        )
        print(f"{'word':<12} {'dimension':<17} {'target':>7} {'humans':>7} {'catalog':>8} {'n':>7}")
        for p in report["pairs"][:20]:
            print(
                f"{p['word']:<12} {p['dimension']:<17} {p['target']:>7.2f} "
                f"{p['folksonomy_mean']:>7.3f} {p['catalog_mean']:>8.3f} {p['n_tracks']:>7,}"
            )
    return report
