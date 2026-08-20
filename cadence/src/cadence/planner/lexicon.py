"""Mood/activity lexicon mapping everyday words to audio-feature targets.

This table is deliberately a *visible artifact* rather than something buried in
a model. It is inspectable, testable and arguable — a reviewer can disagree that
"chill" means energy 0.30 and change one number, which is not true of an
embedding. It also gives the offline planner real competence, so the system has
no hard dependency on an LLM being reachable.

Values are raw Spotify audio-feature units (0-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MoodEntry:
    audio: dict[str, float] = field(default_factory=dict)
    tempo: tuple[float | None, float | None] | None = None
    curve: str | None = None
    themes: tuple[str, ...] = ()


# fmt: off
MOOD_LEXICON: dict[str, MoodEntry] = {
    "chill":       MoodEntry({"energy": 0.32, "valence": 0.45, "acousticness": 0.55}, themes=("chill",)),
    "mellow":      MoodEntry({"energy": 0.30, "valence": 0.45}),
    "calm":        MoodEntry({"energy": 0.22, "acousticness": 0.6}),
    "relaxing":    MoodEntry({"energy": 0.25, "acousticness": 0.6}, themes=("relax",)),
    "relax":       MoodEntry({"energy": 0.25, "acousticness": 0.6}, themes=("relax",)),
    "sleep":       MoodEntry({"energy": 0.12, "acousticness": 0.75, "instrumentalness": 0.5},
                             tempo=(None, 95), curve="wind_down", themes=("sleep",)),
    "bedtime":     MoodEntry({"energy": 0.15, "acousticness": 0.7}, curve="wind_down"),
    "study":       MoodEntry({"energy": 0.28, "instrumentalness": 0.62, "speechiness": 0.05},
                             themes=("study", "focus")),
    "focus":       MoodEntry({"energy": 0.30, "instrumentalness": 0.6, "speechiness": 0.05},
                             themes=("focus",)),
    "instrumental": MoodEntry({"instrumentalness": 0.8}),
    "acoustic":    MoodEntry({"acousticness": 0.85, "energy": 0.35}, themes=("acoustic",)),
    "unplugged":   MoodEntry({"acousticness": 0.85}),
    "upbeat":      MoodEntry({"energy": 0.80, "valence": 0.75, "danceability": 0.70}),
    "energetic":   MoodEntry({"energy": 0.85, "danceability": 0.65}),
    "hype":        MoodEntry({"energy": 0.90, "danceability": 0.72}, curve="build"),
    "pump":        MoodEntry({"energy": 0.90, "danceability": 0.70}, curve="build"),
    "party":       MoodEntry({"energy": 0.82, "danceability": 0.85, "valence": 0.72},
                             themes=("party",)),
    "dance":       MoodEntry({"danceability": 0.85, "energy": 0.78}, themes=("dance",)),
    "club":        MoodEntry({"danceability": 0.85, "energy": 0.85}, tempo=(118, 132)),
    "workout":     MoodEntry({"energy": 0.85, "danceability": 0.70, "valence": 0.62},
                             tempo=(120, 160), curve="build", themes=("workout", "gym")),
    "gym":         MoodEntry({"energy": 0.85, "danceability": 0.70}, tempo=(120, 160),
                             themes=("gym", "workout")),
    "running":     MoodEntry({"energy": 0.82, "danceability": 0.66}, tempo=(150, 180),
                             themes=("running",)),
    "run":         MoodEntry({"energy": 0.82}, tempo=(150, 180), themes=("running",)),
    "sad":         MoodEntry({"valence": 0.15, "energy": 0.32}, themes=("sad",)),
    "melancholy":  MoodEntry({"valence": 0.20, "energy": 0.32}),
    "heartbreak":  MoodEntry({"valence": 0.18, "energy": 0.35}, themes=("heartbreak",)),
    "crying":      MoodEntry({"valence": 0.15, "energy": 0.28}),
    "happy":       MoodEntry({"valence": 0.85, "energy": 0.68}, themes=("happy",)),
    "feel good":   MoodEntry({"valence": 0.82, "energy": 0.66}),
    "sunny":       MoodEntry({"valence": 0.82, "energy": 0.62}),
    "summer":      MoodEntry({"valence": 0.78, "energy": 0.70}, themes=("summer",)),
    "angry":       MoodEntry({"energy": 0.90, "valence": 0.20}),
    "aggressive":  MoodEntry({"energy": 0.92, "valence": 0.22}),
    "rage":        MoodEntry({"energy": 0.93, "valence": 0.18}),
    "romantic":    MoodEntry({"valence": 0.60, "energy": 0.38, "acousticness": 0.45},
                             themes=("love",)),
    "love":        MoodEntry({"valence": 0.62, "energy": 0.42}, themes=("love",)),
    "rainy":       MoodEntry({"energy": 0.28, "valence": 0.32, "acousticness": 0.6},
                             themes=("rainy day",)),
    "moody":       MoodEntry({"valence": 0.28, "energy": 0.40}),
    "dark":        MoodEntry({"valence": 0.20, "energy": 0.50}),
    "dreamy":      MoodEntry({"energy": 0.35, "acousticness": 0.45, "instrumentalness": 0.3}),
    "road trip":   MoodEntry({"energy": 0.68, "valence": 0.68}, themes=("road trip",)),
    "driving":     MoodEntry({"energy": 0.68, "valence": 0.62}, themes=("driving",)),
    "coffee":      MoodEntry({"energy": 0.30, "acousticness": 0.65}),
    "morning":     MoodEntry({"energy": 0.45, "valence": 0.62}, curve="build"),
    "late night":  MoodEntry({"energy": 0.35, "valence": 0.35}, curve="wind_down"),
    "cozy":        MoodEntry({"energy": 0.25, "acousticness": 0.7}),
    "background":  MoodEntry({"energy": 0.28, "instrumentalness": 0.55, "speechiness": 0.05}),
    "dinner":      MoodEntry({"energy": 0.32, "acousticness": 0.55}),
    "throwback":   MoodEntry({}, themes=("throwback",)),
    "high energy": MoodEntry({"energy": 0.88}),
    "low energy":  MoodEntry({"energy": 0.22}),
    "high-energy": MoodEntry({"energy": 0.88}),
    "banger":      MoodEntry({"energy": 0.88, "danceability": 0.78}),
    "bangers":     MoodEntry({"energy": 0.88, "danceability": 0.78}),
    "singalong":   MoodEntry({"valence": 0.75, "energy": 0.68}),
    "nostalgic":   MoodEntry({"valence": 0.45, "energy": 0.45}, themes=("throwback",)),
}
# fmt: on

# Curve hints that are stated directly rather than implied by a mood.
CURVE_PHRASES: dict[str, str] = {
    "build": "build",
    "builds": "build",
    "ramp up": "build",
    "warm up": "build",
    "wind down": "wind_down",
    "winds down": "wind_down",
    "cool down": "wind_down",
    "calm down": "wind_down",
    "peak in the middle": "peak_mid",
    "starts slow": "build",
    "gets more intense": "build",
    "ease into": "build",
}

# Words that flag a request to keep it clean.
CLEAN_PHRASES = (
    "no explicit",
    "nothing explicit",
    "no explicits",
    "not explicit",
    "non explicit",
    "nonexplicit",
    "clean version",
    "clean only",
    "keep it clean",
    "family friendly",
    "family-friendly",
    "kid friendly",
    "kid-friendly",
    "safe for work",
    "sfw",
    "no swearing",
    "no cursing",
    "no profanity",
)

# Rough qualitative tempo words -> BPM windows.
TEMPO_WORDS: dict[str, tuple[float | None, float | None]] = {
    "slow": (None, 95),
    "slower": (None, 95),
    "midtempo": (90, 120),
    "mid tempo": (90, 120),
    "fast": (125, None),
    "faster": (125, None),
    "uptempo": (120, None),
    "up tempo": (120, None),
    "high tempo": (130, None),
}
