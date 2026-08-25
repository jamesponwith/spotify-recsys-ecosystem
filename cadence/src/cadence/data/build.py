"""Build the Cadence catalog from raw MPD slices + the audio-feature table.

Outputs (under ``data/processed``)
----------------------------------
tracks.parquet      one row per catalog track: metadata + joined audio features
playlists.parquet   one row per retained playlist, aligned with the CSR rows
interactions.npz    CSR playlist x track binary matrix
tags.npz            CSR track x tag count matrix (folksonomy from playlist titles)
tag_vocab.json      tag strings, index-aligned with the tag matrix columns
build_meta.json     provenance: counts, filters, coverage

Design notes
------------
* A track's *tag profile* is the bag of title tokens of every playlist it
  appears in. This is the only place in the pipeline where free text is grounded
  in real human curation, and it is what makes cold-start natural-language
  retrieval work at all.
* Audio features come from a separate 114 k-track table and cover only part of
  the catalog. ``has_audio`` records that honestly rather than imputing values:
  an imputed 0.5 energy is indistinguishable from a measured one downstream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import DATA_PROCESSED, DATA_RAW, BuildConfig
from ..text import title_tokens
from .mpd import iter_playlists, slice_paths

AUDIO_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def _load_audio_features(raw_dir: Path) -> pd.DataFrame | None:
    """Load audio features, merging two sources.

    ``audio_features_full.parquet`` is the primary source: a filtered slice of a
    255 M-row Spotify feature dump, covering essentially the whole catalog. It
    carries the numeric features but no `explicit` flag and no genre.

    ``audio_features.csv`` is a smaller genre-balanced table that *does* carry
    both. It covers a minority of the catalog, so `explicit` is a partially
    observed field — recorded honestly rather than defaulted, since defaulting
    to False would silently claim every unlabelled track is clean.
    """
    full_path = raw_dir / "audio_features_full.parquet"
    small_path = raw_dir / "audio_features.csv"

    base: pd.DataFrame | None = None
    if full_path.exists():
        base = pd.read_parquet(full_path).set_index("id")
        base = base[[c for c in AUDIO_COLS if c in base.columns] + ["popularity"]]

    extra: pd.DataFrame | None = None
    if small_path.exists():
        df = pd.read_csv(small_path, index_col=0).dropna(subset=["track_id"])
        genres = (
            df.groupby("track_id")["track_genre"]
            .apply(lambda s: "|".join(sorted(set(s.dropna()))))
            .rename("genre")
        )
        df = df.sort_values("popularity", ascending=False).drop_duplicates("track_id", keep="first")
        extra = df.set_index("track_id").join(genres)

    if base is None and extra is None:
        return None
    if base is None:
        assert extra is not None
        out = extra[[*AUDIO_COLS, "explicit", "popularity", "genre"]].copy()
        out["explicit_known"] = True
        return out

    out = base.copy()
    if extra is not None:
        # Fill any feature the primary source lacks from the secondary one.
        for col in AUDIO_COLS:
            if col not in out.columns:
                out[col] = np.nan
            missing = out[col].isna()
            if missing.any() and col in extra.columns:
                out.loc[missing, col] = extra[col].reindex(out.index[missing]).to_numpy()
        out["explicit"] = extra["explicit"].reindex(out.index)
        out["genre"] = extra["genre"].reindex(out.index)
    else:
        out["explicit"] = pd.NA
        out["genre"] = None

    out["explicit_known"] = out["explicit"].notna()
    out["explicit"] = out["explicit"].fillna(False).astype(bool)
    return out[[*AUDIO_COLS, "explicit", "explicit_known", "popularity", "genre"]]


def _scan(raw_dir: Path, cfg: BuildConfig, verbose: bool):
    """Pass 1 — compact every slice into arrays of dense provisional ids."""
    uri_to_id: dict[str, int] = {}
    names: list[str] = []
    artists: list[str] = []
    artist_uris: list[str] = []
    albums: list[str] = []
    durations: list[int] = []

    titles: list[str] = []
    pids: list[int] = []
    rows: list[np.ndarray] = []

    n_seen = 0
    for pl in iter_playlists(raw_dir, cfg.max_slices):
        n_seen += 1
        tracks = pl["tracks"]
        if not (cfg.min_playlist_len <= len(tracks) <= cfg.max_playlist_len):
            continue
        row = np.empty(len(tracks), dtype=np.int32)
        for i, tr in enumerate(tracks):
            uri = tr["track_uri"]
            tid = uri_to_id.get(uri)
            if tid is None:
                tid = len(names)
                uri_to_id[uri] = tid
                names.append(tr["track_name"])
                artists.append(tr["artist_name"])
                artist_uris.append(tr["artist_uri"])
                albums.append(tr["album_name"])
                durations.append(int(tr["duration_ms"]))
            row[i] = tid
        # A track counts once per playlist even if the playlist repeats it.
        # Deduplicate by *first occurrence* rather than with np.unique, which
        # sorts as a side effect and so silently discards the running order. The
        # dedup was the intent; the sort was not, and it cost the corpus its
        # sequence long before the CSR matrix was built.
        _, first = np.unique(row, return_index=True)
        rows.append(row[np.sort(first)])
        titles.append(pl.get("name") or "")
        pids.append(int(pl["pid"]))
        if verbose and len(rows) % 25000 == 0:
            print(f"  scanned {len(rows):,} playlists / {len(names):,} distinct tracks")

    uris = np.empty(len(names), dtype=object)
    for uri, tid in uri_to_id.items():
        uris[tid] = uri
    meta = {
        "uri": uris,
        "name": names,
        "artist": artists,
        "artist_uri": artist_uris,
        "album": albums,
        "duration_ms": np.asarray(durations, dtype=np.int32),
    }
    return meta, titles, pids, rows, n_seen


def build(
    raw_dir: Path = DATA_RAW,
    out_dir: Path = DATA_PROCESSED,
    cfg: BuildConfig | None = None,
    verbose: bool = True,
) -> dict:
    cfg = cfg or BuildConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    paths = slice_paths(raw_dir, cfg.max_slices)
    if not paths:
        raise FileNotFoundError(f"no MPD slices found in {raw_dir}")

    meta, titles, pids, rows, n_seen = _scan(raw_dir, cfg, verbose)
    n_raw_tracks = len(meta["name"])
    raw_counts = np.bincount(np.concatenate(rows), minlength=n_raw_tracks).astype(np.int32)
    if verbose:
        print(
            f"pass 1: {n_seen:,} seen / {len(rows):,} retained playlists, "
            f"{n_raw_tracks:,} distinct tracks ({time.perf_counter() - t0:.1f}s)"
        )

    # ---- drop the extreme long tail -------------------------------------
    keep = raw_counts >= cfg.min_track_playlists
    n_tracks = int(keep.sum())
    if n_tracks == 0:
        raise ValueError("min_track_playlists filtered out every track")
    old_to_new = np.full(n_raw_tracks, -1, dtype=np.int32)
    old_to_new[keep] = np.arange(n_tracks, dtype=np.int32)

    # ---- pass 2: interaction matrix + retained playlist metadata --------
    indptr: list[int] = [0]
    indices: list[int] = []
    kept_titles: list[str] = []
    kept_pids: list[int] = []
    kept_tokens: list[list[str]] = []
    tag_df: dict[str, int] = {}

    for title, pid, row in zip(titles, pids, rows, strict=True):
        mapped = old_to_new[row]
        mapped = mapped[mapped >= 0]
        if mapped.size < cfg.min_playlist_len:
            continue
        indices.extend(mapped.tolist())
        indptr.append(len(indices))
        kept_titles.append(title)
        kept_pids.append(pid)
        toks = title_tokens(title)
        kept_tokens.append(toks)
        for t in set(toks):
            tag_df[t] = tag_df.get(t, 0) + 1

    n_playlists = len(kept_pids)
    # Capture playlist order *before* the CSR exists. `mapped` above is in the
    # order a human arranged the playlist, but SciPy keeps CSR column indices
    # sorted within a row, so once the matrix is built that order is gone --
    # reading a row back gives ascending track ids, not the sequence. Anything
    # that needs order has to take it here or rebuild it from the raw slices.
    order_tracks = np.asarray(indices, dtype=np.int32)
    order_offsets = np.asarray(indptr, dtype=np.int64)
    interactions = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n_playlists, n_tracks),
    )
    # Now that the rows arrive in playlist order, the constructed matrix has
    # unsorted column indices. Canonicalise it explicitly so the stored matrix is
    # byte-for-byte what it always was -- the ordering lives in order.npz, and
    # nothing downstream should change because it exists.
    interactions.sort_indices()
    del indices, indptr, rows

    # ---- folksonomy: track x title-token counts -------------------------
    vocab = sorted(t for t, df in tag_df.items() if df >= cfg.min_tag_playlists)
    tag_to_col = {t: i for i, t in enumerate(vocab)}
    tag_rows: list[np.ndarray] = []
    tag_cols: list[np.ndarray] = []
    for prow, toks in enumerate(kept_tokens):
        cols = [tag_to_col[t] for t in toks if t in tag_to_col]
        if not cols:
            continue
        trs = interactions.indices[interactions.indptr[prow] : interactions.indptr[prow + 1]]
        tag_rows.append(np.tile(trs, len(cols)))
        tag_cols.append(np.repeat(np.asarray(cols, dtype=np.int32), len(trs)))
    if tag_rows:
        tr = np.concatenate(tag_rows)
        tc = np.concatenate(tag_cols)
    else:  # pragma: no cover - only when every title is empty
        tr = np.zeros(0, dtype=np.int32)
        tc = np.zeros(0, dtype=np.int32)
    tags = sparse.coo_matrix(
        (np.ones(tr.size, dtype=np.float32), (tr, tc)), shape=(n_tracks, len(vocab))
    ).tocsr()
    tags.sum_duplicates()
    del tag_rows, tag_cols, tr, tc

    # ---- catalog frame ---------------------------------------------------
    idx = np.flatnonzero(keep)
    catalog = pd.DataFrame(
        {
            "index": np.arange(n_tracks, dtype=np.int32),
            "track_uri": meta["uri"][idx],
            "name": [meta["name"][i] for i in idx],
            "artist": [meta["artist"][i] for i in idx],
            "artist_uri": [meta["artist_uri"][i] for i in idx],
            "album": [meta["album"][i] for i in idx],
            "duration_ms": meta["duration_ms"][idx],
            "n_playlists": raw_counts[idx],
        }
    )

    audio = _load_audio_features(raw_dir)
    if audio is not None:
        spotify_ids = catalog["track_uri"].str.rsplit(":", n=1).str[-1]
        joined = audio.reindex(spotify_ids.to_numpy())
        for col in AUDIO_COLS:
            catalog[col] = joined[col].to_numpy(dtype=np.float32)
        catalog["explicit"] = joined["explicit"].fillna(False).to_numpy().astype(bool)
        catalog["explicit_known"] = joined["explicit_known"].fillna(False).to_numpy().astype(bool)
        catalog["popularity"] = joined["popularity"].to_numpy(dtype=np.float32)
        catalog["genre"] = joined["genre"].to_numpy()
        catalog["has_audio"] = joined["energy"].notna().to_numpy()
    else:
        for col in AUDIO_COLS:
            catalog[col] = np.float32("nan")
        catalog["explicit"] = False
        catalog["popularity"] = np.float32("nan")
        catalog["genre"] = None
        catalog["has_audio"] = False

    playlists = pd.DataFrame(
        {
            "row": np.arange(n_playlists, dtype=np.int32),
            "pid": np.asarray(kept_pids, dtype=np.int32),
            "name": kept_titles,
            "n_tracks": np.diff(interactions.indptr).astype(np.int32),
        }
    )

    catalog.to_parquet(out_dir / "tracks.parquet", index=False)
    playlists.to_parquet(out_dir / "playlists.parquet", index=False)
    sparse.save_npz(out_dir / "interactions.npz", interactions)
    # Ragged per-playlist track sequence, aligned with interactions' rows.
    np.savez_compressed(out_dir / "order.npz", tracks=order_tracks, offsets=order_offsets)
    sparse.save_npz(out_dir / "tags.npz", tags)
    (out_dir / "tag_vocab.json").write_text(json.dumps(vocab))

    pop = catalog["n_playlists"].to_numpy()
    has_audio = catalog["has_audio"].to_numpy()
    build_meta = {
        "n_slices": len(paths),
        "n_playlists_seen": n_seen,
        "n_playlists": n_playlists,
        "n_tracks_raw": n_raw_tracks,
        "n_tracks": n_tracks,
        "n_interactions": int(interactions.nnz),
        "n_tags": len(vocab),
        "tag_nnz": int(tags.nnz),
        "audio_coverage_tracks": round(float(has_audio.mean()), 4),
        "audio_coverage_impressions": round(float((has_audio * pop).sum() / pop.sum()), 4),
        "explicit_flag_coverage": round(float(catalog["explicit_known"].mean()), 4),
        "median_playlist_len": float(np.median(np.diff(interactions.indptr))),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "config": dict(cfg.__dict__),
    }
    (out_dir / "build_meta.json").write_text(json.dumps(build_meta, indent=2))
    if verbose:
        print(json.dumps(build_meta, indent=2))
    return build_meta
