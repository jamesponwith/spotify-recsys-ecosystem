#!/usr/bin/env python
"""Reduce the 12 GB audio-feature dump to the catalog's tracks.

Streams row-group batches with only the needed columns, so peak memory stays
bounded no matter how large the shards are. Reading a shard whole into pandas
needs ~20 GB; this needs about 1 GB.

Run after an initial `cadence build`, then rebuild to pick up the features.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "id",
    "popularity",
    "duration_ms",
    "time_signature",
    "key",
    "mode",
    "tempo",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "null_response",
]


def main() -> int:
    catalog_path = ROOT / "data" / "processed" / "tracks.parquet"
    if not catalog_path.exists():
        print("run `cadence build` first so the catalog exists", file=sys.stderr)
        return 1

    catalog = pd.read_parquet(catalog_path, columns=["track_uri"])
    want = pa.array(sorted(set(catalog["track_uri"].str.rsplit(":", n=1).str[-1])))
    print(f"catalog ids: {len(want):,}")

    shards = sorted(glob.glob(str(ROOT / "data" / "raw" / "af" / "*.parquet")))
    if not shards:
        print("no shards in data/raw/af — run scripts/download_data.py", file=sys.stderr)
        return 1

    chunks: list[pa.Table] = []
    scanned = matched = 0
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=500_000, columns=COLUMNS):
            scanned += batch.num_rows
            table = pa.Table.from_batches([batch])
            hit = table.filter(pc.is_in(table["id"], value_set=want))
            if hit.num_rows:
                chunks.append(hit)
                matched += hit.num_rows
        print(f"  {Path(path).name}: scanned {scanned:,} matched {matched:,}")

    out = pa.concat_tables(chunks).to_pandas()
    if "null_response" in out.columns:
        out = out[out["null_response"] == 0]
    out = out.sort_values("popularity", ascending=False).drop_duplicates("id", keep="first")
    dest = ROOT / "data" / "raw" / "audio_features_full.parquet"
    out.to_parquet(dest, index=False)
    print(f"wrote {dest} — {len(out):,} tracks, coverage {len(out) / len(want):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
