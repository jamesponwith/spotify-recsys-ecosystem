"""Streaming reader for the Spotify Million Playlist Dataset (MPD).

The corpus ships as ~33 MB JSON slices of 1 000 playlists each. We stream slice
by slice so peak memory tracks the *compacted* representation rather than the
raw JSON.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

try:  # orjson is ~3x faster on these files, but the stdlib works fine.
    import orjson as _json

    def _loads(b: bytes):
        return _json.loads(b)

except ImportError:  # pragma: no cover - fallback path
    import json as _json  # type: ignore[no-redef]

    def _loads(b: bytes):
        return _json.loads(b.decode("utf-8"))


def slice_paths(raw_dir: Path, max_slices: int | None = None) -> list[Path]:
    """Deterministically ordered slice paths (sorted by numeric start offset)."""
    paths = sorted(
        raw_dir.glob("mpd.slice.*.json"),
        key=lambda p: int(p.name.split(".")[2].split("-")[0]),
    )
    return paths[:max_slices] if max_slices else paths


def iter_playlists(raw_dir: Path, max_slices: int | None = None) -> Iterator[dict]:
    """Yield raw MPD playlist dicts across every slice."""
    for path in slice_paths(raw_dir, max_slices):
        payload = _loads(path.read_bytes())
        yield from payload["playlists"]
