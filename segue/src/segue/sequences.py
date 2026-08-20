"""Rebuild true playlist order from the raw MPD slices.

**This module exists because the order is not recoverable from Cadence.**
`data/processed/interactions.npz` is a CSR matrix with `has_sorted_indices ==
True`, so each row's track ids come out in ascending numeric order, not in the
order a human arranged them. Every sequence claim in this project would be
measuring track-id ordering if it read from there.

Cadence is unaffected for its own purposes -- a bag of tracks is all its
collaborative channel ever consumed -- but see docs/FINDINGS.md for one place
where the distinction leaks into its evaluation harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS, CADENCE_PROCESSED, CADENCE_RAW


@dataclass
class Sequences:
    """Ragged array of ordered catalog indices, one row per playlist."""

    tracks: np.ndarray  # int32, concatenated
    offsets: np.ndarray  # int64, length n_playlists + 1
    pids: np.ndarray  # int64, MPD playlist id per row
    rows: np.ndarray  # int64, matching row in Cadence's playlists.parquet

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.tracks[self.offsets[i] : self.offsets[i + 1]]

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACTS / "sequences.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, tracks=self.tracks, offsets=self.offsets, pids=self.pids, rows=self.rows
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> Sequences:
        path = path or ARTIFACTS / "sequences.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `segue build` first.")
        z = np.load(path)
        return cls(tracks=z["tracks"], offsets=z["offsets"], pids=z["pids"], rows=z["rows"])


def build(raw_dir: Path = CADENCE_RAW, verbose: bool = True) -> Sequences:
    """Stream the MPD slices and emit ordered catalog indices per playlist.

    Tracks Cadence dropped (the `min_playlists >= 4` filter) are skipped rather
    than gap-filled: a sequence model cannot score what the catalog cannot serve,
    and silently substituting a neighbour would fabricate adjacencies that no
    human created.
    """
    from cadence.data.mpd import iter_playlists

    t0 = time.perf_counter()
    catalog = pd.read_parquet(CADENCE_PROCESSED / "tracks.parquet")
    uri_to_idx = {u: i for i, u in enumerate(catalog["track_uri"].to_numpy())}
    playlists = pd.read_parquet(CADENCE_PROCESSED / "playlists.parquet")
    pid_to_row = {int(p): int(r) for r, p in enumerate(playlists["pid"].to_numpy())}

    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    pids: list[int] = []
    rows: list[int] = []
    dropped = 0
    seen = 0

    for pl in iter_playlists(raw_dir):
        seen += 1
        row = pid_to_row.get(int(pl["pid"]))
        if row is None:
            continue  # playlist filtered out of the catalog build
        ordered = sorted(pl["tracks"], key=lambda t: t["pos"])
        idx = [uri_to_idx[t["track_uri"]] for t in ordered if t["track_uri"] in uri_to_idx]
        dropped += len(ordered) - len(idx)
        if len(idx) < 2:
            continue  # a sequence of one has no transition to learn from
        chunks.append(np.asarray(idx, dtype=np.int32))
        lengths.append(len(idx))
        pids.append(int(pl["pid"]))
        rows.append(row)
        if verbose and len(rows) % 20_000 == 0:
            print(f"  {len(rows):,} playlists ordered", flush=True)

    tracks = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    seq = Sequences(
        tracks=tracks,
        offsets=offsets,
        pids=np.asarray(pids, dtype=np.int64),
        rows=np.asarray(rows, dtype=np.int64),
    )
    if verbose:
        print(
            f"{len(seq):,} ordered playlists / {seen:,} seen · "
            f"{len(tracks):,} positions · {dropped:,} track slots dropped as out-of-catalog · "
            f"{time.perf_counter() - t0:.0f}s"
        )
    return seq
