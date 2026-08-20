"""Typed contracts shared across the pipeline.

`PlaylistIntent` doubles as the JSON schema handed to the LLM for structured
output, which is why every field carries a description: the descriptions *are*
the prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EnergyCurve = Literal["steady", "build", "wind_down", "wave", "peak_mid"]

AUDIO_DIMENSIONS = (
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
)


class AudioTargets(BaseModel):
    """Desired position in Spotify's audio-feature space. All values are 0..1.

    `None` means "no preference" — the dimension is ignored rather than
    defaulted, which matters because a defaulted 0.5 is an assertion, not a
    shrug.
    """

    energy: float | None = Field(None, ge=0, le=1, description="0 = calm/soft, 1 = intense/loud")
    valence: float | None = Field(None, ge=0, le=1, description="0 = sad/dark, 1 = happy/euphoric")
    danceability: float | None = Field(
        None, ge=0, le=1, description="rhythmic regularity and groove"
    )
    acousticness: float | None = Field(
        None, ge=0, le=1, description="1 = acoustic, 0 = electric/produced"
    )
    instrumentalness: float | None = Field(None, ge=0, le=1, description="1 = no vocals")
    speechiness: float | None = Field(None, ge=0, le=1, description="1 = spoken word / rap-heavy")
    liveness: float | None = Field(None, ge=0, le=1, description="1 = live recording with audience")

    def active(self) -> dict[str, float]:
        return {k: v for k in AUDIO_DIMENSIONS if (v := getattr(self, k)) is not None}


class TempoRange(BaseModel):
    min_bpm: float | None = Field(None, ge=30, le=250)
    max_bpm: float | None = Field(None, ge=30, le=250)

    @field_validator("max_bpm")
    @classmethod
    def _ordered(cls, v, info):
        lo = info.data.get("min_bpm")
        if v is not None and lo is not None and v < lo:
            raise ValueError("max_bpm must be >= min_bpm")
        return v

    def is_set(self) -> bool:
        return self.min_bpm is not None or self.max_bpm is not None


class Constraints(BaseModel):
    """Hard requirements. Anything here is enforced by construction, never by
    asking the model nicely."""

    exclude_explicit: bool = Field(False, description="drop tracks flagged explicit")
    track_count: int | None = Field(
        None, ge=1, le=100, description="exact number of tracks requested"
    )
    target_duration_minutes: float | None = Field(None, gt=0, le=600)
    max_per_artist: int | None = Field(None, ge=1, le=10)
    min_duration_s: float | None = Field(None, gt=0)
    max_duration_s: float | None = Field(None, gt=0)


class PlaylistIntent(BaseModel):
    """Structured reading of a free-text playlist request."""

    summary: str = Field("", description="one-line restatement of what the listener asked for")
    themes: list[str] = Field(
        default_factory=list,
        description="mood/activity/context words a human would put in a playlist title, "
        "e.g. 'rainy day', 'workout', 'study', 'road trip'",
    )
    genres: list[str] = Field(default_factory=list, description="genre names, e.g. 'indie folk'")
    seed_artists: list[str] = Field(
        default_factory=list, description="artists named as reference points"
    )
    seed_tracks: list[str] = Field(
        default_factory=list, description="specific tracks named, as 'Title - Artist'"
    )
    avoid_artists: list[str] = Field(default_factory=list)
    avoid_themes: list[str] = Field(default_factory=list)
    eras: list[str] = Field(
        default_factory=list,
        description="decade tags such as '1990s', '2000s' when the listener anchors to a period",
    )
    audio: AudioTargets = Field(default_factory=AudioTargets)
    tempo: TempoRange = Field(default_factory=TempoRange)
    constraints: Constraints = Field(default_factory=Constraints)
    energy_curve: EnergyCurve = Field(
        "steady", description="shape of the energy arc across the playlist"
    )
    notes: str = Field("", description="anything relevant that did not fit another field")

    def query_text(self) -> str:
        """Flat bag of words for the lexical / tag channels."""
        parts = [*self.themes, *self.genres, *self.eras, *self.seed_artists, self.summary]
        return " ".join(p for p in parts if p)


class Track(BaseModel):
    """A catalog entry. `index` is the dense row id used by every matrix."""

    index: int
    track_uri: str
    name: str
    artist: str
    artist_uri: str
    album: str
    duration_ms: int
    n_playlists: int
    # Audio features are present only for tracks joined against the audio-feature
    # table; `has_audio` tells the pipeline whether to trust them.
    has_audio: bool = False
    energy: float | None = None
    valence: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None
    liveness: float | None = None
    loudness: float | None = None
    tempo: float | None = None
    key: int | None = None
    mode: int | None = None
    explicit: bool = False
    genre: str | None = None
    popularity: int | None = None

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


class ScoredTrack(BaseModel):
    track: Track
    score: float
    # Per-channel contributions, kept for explanation and debugging.
    channel_scores: dict[str, float] = Field(default_factory=dict)
    channel_ranks: dict[str, int] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class PlaylistTrack(BaseModel):
    position: int
    track: Track
    score: float
    reasons: list[str] = Field(default_factory=list)
    transition_note: str | None = None


class PlaylistStats(BaseModel):
    n_tracks: int
    total_duration_s: float
    mean_energy: float | None = None
    mean_valence: float | None = None
    mean_tempo: float | None = None
    n_artists: int
    explicit_count: int
    intra_list_distance: float | None = None
    long_tail_share: float | None = None


class GeneratedPlaylist(BaseModel):
    title: str
    description: str
    query: str
    intent: PlaylistIntent
    tracks: list[PlaylistTrack]
    stats: PlaylistStats
    constraint_report: dict[str, bool] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
