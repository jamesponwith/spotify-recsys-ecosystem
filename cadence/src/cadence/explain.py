"""Grounded explanations and the hallucination guard.

Two separate concerns:

* **Per-track reasons** are generated from data the engine actually used —
  folksonomy tags, co-occurrence, measured audio features. Every reason is a
  statement about a number that exists in the catalog, so none of them can be
  wrong in the way generated prose can be wrong.

* **Copy validation** checks LLM-written text against the real tracklist. Track
  hallucination is already impossible by construction (the model never picks
  tracks), but a model writing the *description* can still name an artist who
  is not on the playlist. This scans the copy for any catalog artist and flags
  ones that are not present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import Catalog
from .text import normalize
from .types import PlaylistIntent

# Audio-feature adjectives used when describing a track's measured position.
_FEATURE_WORDS: dict[str, tuple[str, str]] = {
    "energy": ("low-energy", "high-energy"),
    "valence": ("melancholy", "upbeat"),
    "danceability": ("loose-grooved", "danceable"),
    "acousticness": ("produced", "acoustic"),
    "instrumentalness": ("vocal-led", "instrumental"),
    "speechiness": ("sung", "speech-heavy"),
    "liveness": ("studio", "live"),
}


def track_reasons(
    catalog: Catalog,
    intent: PlaylistIntent,
    index: int,
    *,
    channel_ranks: dict[str, float] | None = None,
    intent_tag_cols: list[int] | None = None,
    max_reasons: int = 3,
) -> list[str]:
    """Evidence-backed reasons this track is on the playlist."""
    reasons: list[str] = []
    ranks = channel_ranks or {}

    # 1. Folksonomy overlap: the strongest human-grounded signal available.
    if intent_tag_cols:
        row = catalog.tag_matrix.getrow(index)
        if row.nnz:
            counts = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
            hits = [(catalog.tag_vocab[c], counts[c]) for c in intent_tag_cols if c in counts]
            hits.sort(key=lambda x: -x[1])
            # One playlist is not evidence. Below this threshold the claim
            # reads as authoritative while resting on a single data point.
            if hits and hits[0][1] >= 3:
                tag, n = hits[0]
                reasons.append(f"on {int(n):,} playlists tagged “{tag}”")

    # 2. Collaborative evidence.
    if ranks.get("collaborative") is not None and ranks["collaborative"] < 200:
        reasons.append(f"co-occurs with your seeds (CF rank {int(ranks['collaborative']) + 1})")

    # 3. Measured audio match against a stated target.
    active = intent.audio.active()
    if active:
        best: tuple[float, str] | None = None
        for name, target in active.items():
            value = catalog.col(name)[index]
            if not np.isfinite(value):
                continue
            gap = abs(float(value) - target)
            low, high = _FEATURE_WORDS.get(name, (name, name))
            word = high if target >= 0.5 else low
            if best is None or gap < best[0]:
                best = (gap, f"{word} ({name} {float(value):.2f} vs target {target:.2f})")
        if best and best[0] < 0.25:
            reasons.append(best[1])

    # 4. Tempo, when the listener asked for one.
    if intent.tempo.is_set():
        bpm = catalog.col("tempo")[index]
        if np.isfinite(bpm):
            reasons.append(f"{float(bpm):.0f} BPM")

    if not reasons:
        tags = catalog.top_tags(index, k=2)
        if tags:
            reasons.append("commonly tagged " + ", ".join(f"“{t}”" for t in tags))
        else:
            n = int(catalog.col("n_playlists")[index])
            reasons.append(f"appears on {n:,} playlists in the corpus")
    return reasons[:max_reasons]


@dataclass
class CopyValidation:
    ok: bool
    unsupported_artists: list[str]

    def as_warning(self) -> str | None:
        if self.ok:
            return None
        names = ", ".join(self.unsupported_artists[:5])
        return f"generated copy referenced artists not on the playlist ({names}); replaced with template copy"


def validate_copy(catalog: Catalog, text: str, playlist_indices: np.ndarray) -> CopyValidation:
    """Flag any catalog artist named in ``text`` who is not on the playlist.

    Scans word n-grams of the copy against the catalog's artist-name index. Only
    names of two or more words (or long single words) are considered, because
    short single tokens like "Air" or "Yes" are real artist names that collide
    constantly with ordinary English and would make the guard useless.
    """
    if not text:
        return CopyValidation(True, [])

    present = {
        normalize(str(a))
        for a in catalog.col("artist")[np.asarray(playlist_indices, dtype=np.int64)]
    }
    words = normalize(text).split()
    known = catalog._artist_name_index  # noqa: SLF001 - internal by design
    found: list[str] = []
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            gram = " ".join(words[i : i + size])
            if size == 1 and len(gram) < 8:
                continue
            if gram in known and gram not in present and gram not in found:
                found.append(gram)
    return CopyValidation(ok=not found, unsupported_artists=found)


def describe_arc(catalog: Catalog, indices: np.ndarray) -> str:
    """One-line summary of the energy arc actually achieved."""
    energy = catalog.col("energy")[np.asarray(indices, dtype=np.int64)]
    known = energy[np.isfinite(energy)]
    if known.size < 3:
        return "energy arc unavailable (audio features missing for most tracks)"
    head = float(known[: max(1, known.size // 3)].mean())
    tail = float(known[-max(1, known.size // 3) :].mean())
    delta = tail - head
    if delta > 0.08:
        shape = "builds"
    elif delta < -0.08:
        shape = "winds down"
    else:
        shape = "holds steady"
    return f"energy {shape} ({head:.2f} → {tail:.2f})"
