"""Order a selected set of tracks into a listenable arc.

Selection decides *what* is on the playlist; sequencing decides what it feels
like to listen to. Three signals drive it, all of which a DJ would recognise:

* **energy curve** - the playlist should follow a shape (build, wind down,
  steady, wave), not random-walk through intensity.
* **harmonic mixing** - adjacent tracks in compatible keys sound intentional.
  Compatibility is the standard Camelot wheel: same key, a fifth away, or the
  relative major/minor.
* **tempo continuity** - large BPM jumps read as jarring; half/double-time
  relationships do not, and are treated as near-free.

Ordering is a small asymmetric TSP, solved with beam search. With <= 100 tracks
the beam is exact enough and runs in microseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..catalog import Catalog
from ..types import EnergyCurve


def camelot(key: int | None, mode: int | None) -> tuple[int, int] | None:
    """(wheel position 1-12, 0=minor/A 1=major/B), or None if key is unknown.

    Spotify reports ``key`` as a pitch class 0-11 and ``mode`` as 1=major,
    0=minor. Camelot position walks the circle of fifths, hence the *7 mod 12.
    """
    if key is None or mode is None or not (0 <= key <= 11):
        return None
    # A minor key shares its wheel position with its relative major, a minor
    # third up: A minor and C major are both position 8.
    root = key if mode == 1 else (key + 3) % 12
    number = ((root * 7) % 12 + 7) % 12 + 1
    return number, int(mode)


def key_distance(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    """0.0 = harmonically compatible, 1.0 = clash. Unknown keys cost a neutral
    0.35 so that missing metadata neither rewards nor punishes a track."""
    if a is None or b is None:
        return 0.35
    (na, ma), (nb, mb) = a, b
    steps = min((na - nb) % 12, (nb - na) % 12)
    if na == nb and ma == mb:
        return 0.0
    if na == nb and ma != mb:  # relative major/minor
        return 0.15
    if steps == 1 and ma == mb:  # adjacent on the wheel (a fifth)
        return 0.2
    if steps == 2 and ma == mb:
        return 0.55
    return 1.0


def tempo_distance(a: float | None, b: float | None) -> float:
    """Normalised BPM gap, treating half- and double-time as near-equivalent."""
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return 0.3
    options = [abs(a - b), abs(a - 2 * b), abs(2 * a - b)]
    return float(min(min(options) / 40.0, 1.5))


def energy_curve_targets(curve: EnergyCurve, n: int, base: float, spread: float) -> np.ndarray:
    """Target energy at each position, in raw 0-1 feature units."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.array([0.5], dtype=np.float32)
    shape: np.ndarray
    if curve == "build":
        shape = t
    elif curve == "wind_down":
        shape = 1.0 - t
    elif curve == "wave":
        shape = (0.5 + 0.5 * np.sin(2 * np.pi * t - np.pi / 2)).astype(np.float32)
    elif curve == "peak_mid":
        shape = (1.0 - np.abs(2 * t - 1.0)).astype(np.float32)
    else:  # steady
        shape = np.full(n, 0.5, dtype=np.float32)
    centred = (shape - 0.5) * 2.0  # -1 .. 1
    return np.clip(base + centred * spread, 0.0, 1.0).astype(np.float32)


@dataclass
class SequencedPlaylist:
    order: np.ndarray  # positions into the input array
    total_cost: float
    transition_notes: list[str]


def sequence(
    catalog: Catalog,
    indices: np.ndarray,
    *,
    curve: EnergyCurve = "steady",
    beam_width: int = 24,
    w_tempo: float = 1.0,
    w_key: float = 0.8,
    w_energy: float = 1.4,
    w_artist: float = 2.0,
    energy_spread: float = 0.22,
) -> SequencedPlaylist:
    idx = np.asarray(indices, dtype=np.int64)
    n = idx.size
    if n <= 2:
        return SequencedPlaylist(np.arange(n), 0.0, [])

    tempo = catalog.col("tempo")[idx]
    energy = catalog.col("energy")[idx]
    keys = catalog.col("key")[idx]
    modes = catalog.col("mode")[idx]
    artists = catalog.artist_ids[idx]

    known_energy = energy[np.isfinite(energy)]
    base = float(known_energy.mean()) if known_energy.size else 0.5
    targets = energy_curve_targets(curve, n, base, energy_spread)

    cam = [
        camelot(int(k) if np.isfinite(k) else None, int(m) if np.isfinite(m) else None)
        for k, m in zip(keys, modes, strict=True)
    ]

    def position_cost(i: int, pos: int) -> float:
        e = energy[i]
        if not np.isfinite(e):
            return 0.15  # neutral: unknown energy is not evidence of a bad fit
        return float(abs(e - targets[pos]))

    def transition_cost(i: int, j: int) -> float:
        c = w_tempo * tempo_distance(
            float(tempo[i]) if np.isfinite(tempo[i]) else None,
            float(tempo[j]) if np.isfinite(tempo[j]) else None,
        )
        c += w_key * key_distance(cam[i], cam[j])
        if artists[i] == artists[j]:
            c += w_artist
        return c

    # Beam search over (used-bitmask, last) states.
    beams: list[tuple[float, int, int, list[int]]] = []
    start_order = np.argsort(np.abs(energy - targets[0]))
    for s in start_order[: max(beam_width, 4)]:
        s = int(s)
        beams.append((w_energy * position_cost(s, 0), 1 << s, s, [s]))

    for pos in range(1, n):
        nxt: list[tuple[float, int, int, list[int]]] = []
        for cost, used, last, path in beams:
            for j in range(n):
                if used >> j & 1:
                    continue
                c = cost + transition_cost(last, j) + w_energy * position_cost(j, pos)
                nxt.append((c, used | (1 << j), j, [*path, j]))
        nxt.sort(key=lambda x: x[0])
        # Deduplicate on (used, last): identical states differ only in tie-broken
        # history and keeping both wastes beam slots.
        seen: set[tuple[int, int]] = set()
        beams = []
        for item in nxt:
            key = (item[1], item[2])
            if key in seen:
                continue
            seen.add(key)
            beams.append(item)
            if len(beams) >= beam_width:
                break

    best_cost, _, _, best_path = min(beams, key=lambda x: x[0])
    order = np.asarray(best_path, dtype=np.int64)

    notes: list[str] = []
    for a, b in zip(order, order[1:], strict=False):
        bits: list[str] = []
        ta, tb = tempo[a], tempo[b]
        if np.isfinite(ta) and np.isfinite(tb):
            bits.append(f"{ta:.0f}->{tb:.0f} BPM")
        ka, kb = cam[a], cam[b]
        if ka and kb:
            letter = {0: "A", 1: "B"}
            ca, cb = f"{ka[0]}{letter[ka[1]]}", f"{kb[0]}{letter[kb[1]]}"
            kd = key_distance(ka, kb)
            quality = "harmonic" if kd <= 0.2 else ("near" if kd <= 0.55 else "contrast")
            bits.append(f"{ca}->{cb} ({quality})")
        notes.append(", ".join(bits) if bits else "")
    return SequencedPlaylist(order=order, total_cost=float(best_cost), transition_notes=notes)
