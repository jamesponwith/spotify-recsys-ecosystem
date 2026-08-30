"""Audit ``MOOD_LEXICON``'s audio targets against where humans file each word.

The lexicon asserts that "sleep" means energy 0.12. The folksonomy records
where people actually put music on playlists titled *sleep*. Those are two
definitions of the same word, and nothing in the repo had ever measured how far
apart they are -- which matters, because the audio-affinity term aims selection
at the lexicon's number, and a target far from human practice would explain
the term performing no better than a permutation of itself.

For every (word, dimension) pair where the word is also a tag in the vocabulary,
three numbers:

* **target** -- what the lexicon asserts;
* **folksonomy_mean** -- the plain mean of that dimension over every track
  filed under that tag at least once;
* **catalog_mean** -- the plain mean over the whole catalog, i.e. what you would
  aim at knowing nothing.

A pair is *worse than nothing* when ``|target - folksonomy| > |catalog -
folksonomy|``: the lexicon's number is further from human behaviour than the
uninformed prior is.

What this establishes is the distance between the two definitions. It does
**not** establish that the lexicon is wrong: someone asking for "chill" may well
want calmer music than the median playlist titled *chill* contains. The
folksonomy mean describes behaviour, not intent, and this audit says nothing
about the affinity weight itself.

Reads only ``tracks.parquet``, ``tags.npz`` and ``tag_vocab.json`` -- no engine,
no trained spaces -- so it runs in seconds on the built catalog.
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

    Only tracks with finite audio values count, in both the per-tag and the
    catalog means, so the two are over the same population and the comparison
    is fair. A word that is not a tag cannot be audited and is listed rather
    than silently dropped.
    """
    tags = tags.tocsc()
    col_of = {t: i for i, t in enumerate(vocab)}
    has_audio = (
        frame["has_audio"].to_numpy(dtype=bool)
        if "has_audio" in frame.columns
        else np.ones(len(frame), dtype=bool)
    )
    dims = [d for d in AUDIO_FEATURE_COLS if d in frame.columns]
    values = {d: frame[d].to_numpy(dtype=np.float64) for d in dims}
    valid = {d: has_audio & np.isfinite(values[d]) for d in dims}
    catalog_mean = {d: float(values[d][valid[d]].mean()) for d in dims}

    pairs: list[dict[str, Any]] = []
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
            gap_target = abs(target - folk)
            gap_catalog = abs(catalog_mean[dim] - folk)
            pairs.append(
                {
                    "word": word,
                    "dimension": dim,
                    "target": float(target),
                    "folksonomy_mean": round(folk, 4),
                    "catalog_mean": round(catalog_mean[dim], 4),
                    "n_tracks": int(mask.sum()),
                    "gap_target": round(gap_target, 4),
                    "gap_catalog": round(gap_catalog, 4),
                    "target_worse_than_catalog": bool(gap_target > gap_catalog),
                }
            )

    # Most damning first: how much further the target is than doing nothing.
    pairs.sort(key=lambda p: p["gap_target"] - p["gap_catalog"], reverse=True)
    return {
        "n_pairs": len(pairs),
        "n_target_worse_than_catalog": sum(p["target_worse_than_catalog"] for p in pairs),
        "n_words_audited": len({p["word"] for p in pairs}),
        "words_not_tags": sorted(not_tags),
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
