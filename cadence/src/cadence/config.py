"""Central configuration. All tunable knobs live here so experiments are reproducible."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("CADENCE_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED = Path(os.environ.get("CADENCE_DATA_PROCESSED", PROJECT_ROOT / "data" / "processed"))
ARTIFACTS = Path(os.environ.get("CADENCE_ARTIFACTS", PROJECT_ROOT / "artifacts"))

SEED = 20260815


@dataclass(frozen=True)
class BuildConfig:
    """Catalog construction parameters."""

    # A track must appear in at least this many playlists to enter the catalog.
    # Filters the extreme long tail where co-occurrence statistics are pure noise.
    min_track_playlists: int = 4
    # Playlists outside this range are dropped (degenerate or pathological).
    min_playlist_len: int = 5
    max_playlist_len: int = 250
    # Title tokens must appear on at least this many playlists to become a tag.
    min_tag_playlists: int = 12
    max_slices: int | None = None  # None = use every downloaded slice


@dataclass(frozen=True)
class EmbeddingConfig:
    """Item-embedding parameters (SPPMI + truncated SVD)."""

    dim: int = 160
    # Negative-sample shift k for Shifted PPMI. log(k) is subtracted from PMI;
    # larger k => sparser, more selective association matrix.
    shift_k: float = 5.0
    # Down-weights very popular items when building co-occurrence counts.
    popularity_damping: float = 0.75
    # Playlists longer than this are subsampled when forming co-occurrence pairs,
    # bounding the O(n^2) pair explosion.
    max_pairs_per_playlist: int = 4000


@dataclass(frozen=True)
class TagConfig:
    """Folksonomy (playlist-title) tag-space parameters."""

    dim: int = 128
    shift_k: float = 2.0
    min_token_len: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    """Candidate generation parameters."""

    candidates_per_channel: int = 800
    fused_candidates: int = 1500
    rrf_k: float = 60.0
    # Relative trust of each channel inside reciprocal-rank fusion.
    channel_weights: dict[str, float] = field(
        default_factory=lambda: {
            "collaborative": 1.0,
            "cooccurrence": 1.3,
            "tag": 1.0,
            "tag_exact": 0.7,
            "audio": 0.6,
            "lexical": 0.5,
            "popularity": 0.25,
        }
    )


@dataclass(frozen=True)
class AssemblyConfig:
    """Selection + sequencing parameters."""

    default_length: int = 20
    max_tracks_per_artist: int = 2
    # Maximal Marginal Relevance tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = 0.72
    # How much an explicitly stated audio target (calm, upbeat, acoustic) pulls
    # selection toward it. Measured cost documented in docs/EVALUATION.md.
    audio_affinity_weight: float = 0.35
    beam_width: int = 24
    # Transition-cost weights used by the sequencer.
    w_tempo: float = 1.0
    w_key: float = 0.8
    w_energy_curve: float = 1.4
    w_artist_adjacent: float = 2.0


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings. Defaults to the offline provider so the system
    runs end-to-end with no API key and produces deterministic output."""

    provider: str = os.environ.get("CADENCE_LLM_PROVIDER", "offline")  # offline | anthropic
    model: str = os.environ.get("CADENCE_LLM_MODEL", "claude-opus-5")
    max_tokens: int = 4096
    effort: str = "medium"
    timeout_s: float = 60.0


@dataclass(frozen=True)
class Config:
    build: BuildConfig = field(default_factory=BuildConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tags: TagConfig = field(default_factory=TagConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    seed: int = SEED


DEFAULT = Config()
