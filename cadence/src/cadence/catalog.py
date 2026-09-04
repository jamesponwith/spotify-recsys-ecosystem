"""Serving-time view over the built catalog and trained spaces.

Everything the request path touches is preloaded into contiguous numpy arrays.
The pandas frame is kept only for humans and for building the lookup tables:
per-request pandas access would dominate the latency budget.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import ARTIFACTS, DATA_PROCESSED
from .models.train import AUDIO_FEATURE_COLS
from .retrieval.ann import DenseIndex
from .text import dedupe, normalize, tokenize
from .types import Track


@dataclass
class Catalog:
    """Immutable, preloaded catalog + learned spaces."""

    frame: pd.DataFrame
    collab: DenseIndex
    tag_tracks: DenseIndex
    tag_vectors: np.ndarray
    tag_vocab: list[str]
    lexical: sparse.csr_matrix
    lexical_vectorizer: Any
    audio_z: np.ndarray
    audio_mu: np.ndarray
    audio_sigma: np.ndarray
    audio_valid: np.ndarray
    popularity: np.ndarray
    tag_matrix: sparse.csr_matrix
    interactions: sparse.csr_matrix
    meta: dict

    # ---- construction ---------------------------------------------------
    @classmethod
    def load(cls, processed_dir: Path = DATA_PROCESSED, artifacts_dir: Path = ARTIFACTS) -> Catalog:
        frame = pd.read_parquet(processed_dir / "tracks.parquet")
        spaces = np.load(artifacts_dir / "spaces.npz")
        vocab: list[str] = json.loads((artifacts_dir / "tag_vocab.json").read_text())
        lexical = sparse.load_npz(artifacts_dir / "lexical.npz").tocsr()
        vectorizer = pickle.loads((artifacts_dir / "lexical_vectorizer.pkl").read_bytes())
        # Prefer the tag matrix the trainer actually fit on: processed tags.npz
        # counts *every* playlist, so a held-out challenge could credit a track
        # via the very playlist being scored. The fallback exists only for
        # artifact bundles from before tags_train.npz was persisted.
        tags_path = artifacts_dir / "tags_train.npz"
        tag_matrix = sparse.load_npz(
            tags_path if tags_path.exists() else processed_dir / "tags.npz"
        ).tocsr()
        if tag_matrix.shape != (len(frame), len(vocab)):
            raise ValueError(
                f"tag matrix {tag_matrix.shape} does not match the catalog "
                f"({len(frame)} tracks x {len(vocab)} tags) — stale artifacts? re-run `cadence train`"
            )
        train_path = artifacts_dir / "train_interactions.npz"
        interactions = sparse.load_npz(
            train_path if train_path.exists() else processed_dir / "interactions.npz"
        ).tocsr()
        meta = {
            "build": json.loads((processed_dir / "build_meta.json").read_text()),
            "train": json.loads((artifacts_dir / "train_meta.json").read_text()),
        }
        return cls(
            frame=frame,
            collab=DenseIndex(spaces["collab_vectors"]),
            tag_tracks=DenseIndex(spaces["tag_track_vectors"]),
            tag_vectors=spaces["tag_vectors"],
            tag_vocab=vocab,
            lexical=lexical,
            lexical_vectorizer=vectorizer,
            audio_z=spaces["audio_z"],
            audio_mu=spaces["audio_mu"],
            audio_sigma=spaces["audio_sigma"],
            audio_valid=spaces["audio_valid"],
            popularity=spaces["popularity"],
            tag_matrix=tag_matrix,
            interactions=interactions,
            meta=meta,
        )

    # ---- basic properties ------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame)

    @cached_property
    def _cols(self) -> dict[str, np.ndarray]:
        f = self.frame
        return {
            "name": f["name"].to_numpy(dtype=object),
            "artist": f["artist"].to_numpy(dtype=object),
            "artist_uri": f["artist_uri"].to_numpy(dtype=object),
            "album": f["album"].to_numpy(dtype=object),
            "track_uri": f["track_uri"].to_numpy(dtype=object),
            "genre": f["genre"].to_numpy(dtype=object),
            "duration_ms": f["duration_ms"].to_numpy(dtype=np.int64),
            "n_playlists": f["n_playlists"].to_numpy(dtype=np.int64),
            "explicit": f["explicit"].to_numpy(dtype=bool),
            "explicit_known": (
                f["explicit_known"].to_numpy(dtype=bool)
                if "explicit_known" in f.columns
                else np.zeros(len(f), dtype=bool)
            ),
            "has_audio": f["has_audio"].to_numpy(dtype=bool),
            "tempo": f["tempo"].to_numpy(dtype=np.float32),
            "energy": f["energy"].to_numpy(dtype=np.float32),
            "valence": f["valence"].to_numpy(dtype=np.float32),
            "danceability": f["danceability"].to_numpy(dtype=np.float32),
            "acousticness": f["acousticness"].to_numpy(dtype=np.float32),
            "instrumentalness": f["instrumentalness"].to_numpy(dtype=np.float32),
            "speechiness": f["speechiness"].to_numpy(dtype=np.float32),
            "liveness": f["liveness"].to_numpy(dtype=np.float32),
            "loudness": f["loudness"].to_numpy(dtype=np.float32),
            "key": f["key"].to_numpy(dtype=np.float32),
            "mode": f["mode"].to_numpy(dtype=np.float32),
            "popularity": f["popularity"].to_numpy(dtype=np.float32),
        }

    def col(self, name: str) -> np.ndarray:
        return self._cols[name]

    @cached_property
    def artist_ids(self) -> np.ndarray:
        """Dense artist id per track, for artist-level caps and diversity."""
        uris = self.col("artist_uri")
        lookup: dict[str, int] = {}
        out = np.empty(len(uris), dtype=np.int32)
        for i, u in enumerate(uris):
            aid = lookup.get(u)
            if aid is None:
                aid = len(lookup)
                lookup[u] = aid
            out[i] = aid
        return out

    @cached_property
    def _artist_name_index(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = defaultdict(list)
        for i, a in enumerate(self.col("artist")):
            idx[normalize(str(a))].append(i)
        return dict(idx)

    @cached_property
    def _artist_token_index(self) -> dict[str, set[str]]:
        """Token -> set of normalised artist names, for fuzzy fallback."""
        idx: dict[str, set[str]] = defaultdict(set)
        for name in self._artist_name_index:
            for tok in name.split():
                idx[tok].add(name)
        return dict(idx)

    @cached_property
    def _track_name_index(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = defaultdict(list)
        for i, n in enumerate(self.col("name")):
            idx[normalize(str(n))].append(i)
        return dict(idx)

    @cached_property
    def interactions_t(self):
        """track -> playlists, for neighbourhood lookups."""
        return self.interactions.T.tocsr()

    @cached_property
    def track_playlist_counts(self) -> np.ndarray:
        return np.asarray(self.interactions.sum(axis=0)).ravel().astype(np.float32)

    @cached_property
    def tag_matrix_csc(self):
        """Column-oriented view of the tag matrix.

        Slicing columns out of a CSR matrix costs O(nnz); the exact-tag channel
        does exactly that on every query, so it gets a CSC copy instead.
        """
        return self.tag_matrix.tocsc()

    @cached_property
    def tag_to_col(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.tag_vocab)}

    # ---- entity resolution -----------------------------------------------
    def resolve_artist(self, query: str, limit: int = 60) -> list[int]:
        """Track indices for an artist named in free text.

        Exact normalised match first; otherwise the artist whose name shares the
        most tokens with the query, requiring every query token to be present so
        that "Drake" never resolves to "Drake Bell" style false positives
        outranking the real match on popularity alone.
        """
        q = normalize(query)
        if not q:
            return []
        hits = self._artist_name_index.get(q)
        if hits is None:
            toks = [t for t in q.split() if t]
            if not toks:
                return []
            candidates: set[str] | None = None
            for tok in toks:
                names = self._artist_token_index.get(tok, set())
                candidates = names.copy() if candidates is None else (candidates & names)
                if not candidates:
                    break
            if not candidates:
                return []
            # Prefer the shortest matching artist name: it is the least likely
            # to be a different artist that merely contains the query.
            best = min(candidates, key=lambda n: (len(n.split()), len(n)))
            hits = self._artist_name_index[best]
        pop = self.col("n_playlists")[hits]
        order = np.argsort(-pop)[:limit]
        return [hits[i] for i in order]

    def resolve_track(self, query: str) -> list[int]:
        """Track indices for a 'Title - Artist' or bare-title mention."""
        raw = query
        artist_part = ""
        for sep in (" - ", " – ", " by ", " — "):
            if sep in raw:
                left, right = raw.split(sep, 1)
                raw, artist_part = left, right
                break
        q = normalize(raw)
        hits = list(self._track_name_index.get(q, []))
        if not hits:
            return []
        if artist_part:
            want = normalize(artist_part)
            artists = self.col("artist")
            refined = [i for i in hits if normalize(str(artists[i])) == want]
            if refined:
                hits = refined
        pop = self.col("n_playlists")[hits]
        return [hits[i] for i in np.argsort(-pop)]

    def resolve_tags(self, phrases: list[str]) -> list[int]:
        """Map free-text phrases to tag-vocabulary column ids.

        An exact phrase match wins outright: "hip hop" is one concept, and
        falling through to its unigrams would also pull in "hop" (as in
        hopeless, hopscotch) and double-count the concept in the centroid.
        """
        out: list[int] = []
        for phrase in phrases:
            exact = self.tag_to_col.get(normalize(phrase))
            if exact is not None:
                out.append(exact)
                continue
            hits = [c for tok in tokenize(phrase) if (c := self.tag_to_col.get(tok)) is not None]
            bigrams = [c for c in hits if " " in self.tag_vocab[c]]
            out.extend(bigrams or hits)
        return dedupe(out)

    # ---- projection ------------------------------------------------------
    def track(self, index: int) -> Track:
        c = self._cols

        def f(key: str) -> float | None:
            v = float(c[key][index])
            return None if not np.isfinite(v) else v

        return Track(
            index=int(index),
            track_uri=str(c["track_uri"][index]),
            name=str(c["name"][index]),
            artist=str(c["artist"][index]),
            artist_uri=str(c["artist_uri"][index]),
            album=str(c["album"][index]),
            duration_ms=int(c["duration_ms"][index]),
            n_playlists=int(c["n_playlists"][index]),
            has_audio=bool(c["has_audio"][index]),
            energy=f("energy"),
            valence=f("valence"),
            danceability=f("danceability"),
            acousticness=f("acousticness"),
            instrumentalness=f("instrumentalness"),
            speechiness=f("speechiness"),
            liveness=f("liveness"),
            loudness=f("loudness"),
            tempo=f("tempo"),
            key=int(c["key"][index]) if np.isfinite(c["key"][index]) else None,
            mode=int(c["mode"][index]) if np.isfinite(c["mode"][index]) else None,
            explicit=bool(c["explicit"][index]),
            genre=None if c["genre"][index] is None else str(c["genre"][index]),
            popularity=int(c["popularity"][index]) if np.isfinite(c["popularity"][index]) else None,
        )

    def top_tags(self, index: int, k: int = 6) -> list[str]:
        """Highest-count folksonomy tags for a track — used in explanations."""
        row = self.tag_matrix.getrow(index)
        if row.nnz == 0:
            return []
        order = np.argsort(-row.data)[:k]
        return [self.tag_vocab[row.indices[i]] for i in order]

    @property
    def audio_cols(self) -> list[str]:
        return list(AUDIO_FEATURE_COLS)
