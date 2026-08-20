#!/usr/bin/env python
"""Download every raw input Cadence needs.

Three sources (see docs/DATA_CARD.md):
  1. MPD slices                 ~3 GB   playlists, titles, track metadata
  2. Audio-feature shards      ~12 GB   numeric features for 255 M track ids
  3. Genre/explicit table       ~20 MB  the only source of an `explicit` flag

The audio shards are streamed and filtered down to catalog tracks, so the 12 GB
is transient. Pass --skip-audio to build without them; the pipeline still runs,
it just loses mood targeting and sequencing quality.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

MPD_REPO = "jaxliu/Spotify_Million_Playlist_Dataset_Challenge"
AUDIO_REPO = "ozefe/spotify_audio_features"
GENRE_URL = (
    "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
)


def hf_files(repo: str, suffix: str) -> list[str]:
    import json
    import urllib.request

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}") as r:
        meta = json.load(r)
    return sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(suffix))


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    subprocess.run(
        ["curl", "-sSL", "--retry", "3", "--retry-delay", "2", url, "-o", str(tmp)], check=True
    )
    tmp.rename(dest)
    print(f"  ✓ {dest.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=0, help="0 = all available")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    print("1/3 MPD slices")
    names = hf_files(MPD_REPO, ".json")
    if args.slices:
        names = names[: args.slices]
    base = f"https://huggingface.co/datasets/{MPD_REPO}/resolve/main"
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda n: fetch(f"{base}/{n}", RAW / Path(n).name), names))

    print("2/3 genre + explicit table")
    fetch(GENRE_URL, RAW / "audio_features.csv")

    if args.skip_audio:
        print("3/3 skipped (--skip-audio)")
        return 0

    print("3/3 audio-feature shards (~12 GB, filtered to catalog on the fly)")
    shards = hf_files(AUDIO_REPO, ".parquet")
    abase = f"https://huggingface.co/datasets/{AUDIO_REPO}/resolve/main"
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda n: fetch(f"{abase}/{n}", RAW / "af" / Path(n).name), shards))

    print("\nNext: `make build` (run `cadence build` first if data/processed is empty),")
    print("then `python scripts/filter_audio_features.py` to join features to the catalog.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
