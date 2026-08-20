"""Deterministic natural-language intent parser.

This is the default planner. It exists for three reasons:

1. The system must run end-to-end with no API key and no network.
2. Evaluation needs a fixed, reproducible planner — if the parser is
   nondeterministic, every retrieval metric inherits that variance and
   ablations stop meaning anything.
3. It is the fallback when the LLM is unavailable, malformed or slow, so a
   provider outage degrades quality instead of causing an outage.

It handles counts, durations, tempo, explicit filters, era anchors, seed
artists/tracks, energy curves and a 49-entry mood lexicon. It does *not* handle
open-ended compositional phrasing — that is where the LLM planner earns its
cost, and `docs/EVALUATION.md` quantifies the gap.
"""

from __future__ import annotations

import re

from ..text import STOPWORDS, decade_tags, normalize, tokenize
from ..types import AudioTargets, Constraints, PlaylistIntent, TempoRange
from .lexicon import CLEAN_PHRASES, CURVE_PHRASES, MOOD_LEXICON, TEMPO_WORDS

# Allows a few adjectives between the count and the noun:
# "20 upbeat workout tracks" must parse as well as "20 tracks".
_COUNT_RE = re.compile(
    r"\b(\d{1,3})\s*[- ]\s*(?:[a-z]+\s+){0,3}?(?:song|songs|track|tracks|tune|tunes)\b"
)
_HOURS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)\b")
_MINUTES_RE = re.compile(r"\b(\d{1,3})\s*(?:minute|minutes|min|mins)\b")
_BPM_RANGE_RE = re.compile(r"\b(\d{2,3})\s*(?:-|to|and)\s*(\d{2,3})\s*bpm\b")
_BPM_ONE_RE = re.compile(r"\b(?:around|about|near|at)?\s*(\d{2,3})\s*bpm\b")
_BPM_MIN_RE = re.compile(r"\b(?:over|above|faster than|at least)\s*(\d{2,3})\s*bpm\b")
_BPM_MAX_RE = re.compile(r"\b(?:under|below|slower than|at most)\s*(\d{2,3})\s*bpm\b")

# Units and quantities that describe the *request*, not the music.
_UNIT_WORDS = frozenset(
    [
        "minute",
        "minutes",
        "min",
        "mins",
        "hour",
        "hours",
        "hr",
        "hrs",
        "bpm",
        "tempo",
        "second",
        "seconds",
        "song",
        "songs",
        "track",
        "tracks",
        "tune",
        "tunes",
        "count",
        "length",
        "duration",
    ]
)

_SEED_PATTERNS = (
    re.compile(r"(?:sounds?|songs?|stuff|music|tracks?)\s+like\s+(.+?)(?=[,.;]|\band\b|$)"),
    re.compile(r"\bsimilar to\s+(.+?)(?=[,.;]|\band\b|$)"),
    re.compile(r"\bin the vein of\s+(.+?)(?=[,.;]|\band\b|$)"),
    re.compile(r"\bfans? of\s+(.+?)(?=[,.;]|\band\b|$)"),
    re.compile(r"\b(?:by|from)\s+(.+?)(?=[,.;]|\band\b|$)"),
)
_AVOID_RE = re.compile(r"\b(?:no|without|avoid|skip|nothing by|not)\s+(.+?)(?=[,.;]|\band\b|$)")
_QUOTED_RE = re.compile(r"[\"'“]([^\"'”]{2,60})[\"'”]")

# Words that follow "no ..." but describe a mood/constraint, not an artist.
_NON_ARTIST_AVOID = {
    "explicit",
    "swearing",
    "cursing",
    "profanity",
    "repeats",
    "duplicates",
    "vocals",
    "lyrics",
    "sad",
    "slow",
    "rap",
    "country",
    "metal",
}


def _extract_counts(text: str, c: Constraints) -> Constraints:
    data = c.model_dump()
    if m := _COUNT_RE.search(text):
        data["track_count"] = max(1, min(100, int(m.group(1))))
    minutes: float | None = None
    if m := _HOURS_RE.search(text):
        minutes = float(m.group(1)) * 60
    if m := _MINUTES_RE.search(text):
        minutes = (minutes or 0) + float(m.group(1))
    if minutes:
        data["target_duration_minutes"] = min(minutes, 600.0)
    if any(p in text for p in CLEAN_PHRASES):
        data["exclude_explicit"] = True
    return Constraints(**data)


def _extract_tempo(text: str) -> TempoRange:
    if m := _BPM_RANGE_RE.search(text):
        return TempoRange(min_bpm=float(m.group(1)), max_bpm=float(m.group(2)))
    if m := _BPM_MIN_RE.search(text):
        return TempoRange(min_bpm=float(m.group(1)))
    if m := _BPM_MAX_RE.search(text):
        return TempoRange(max_bpm=float(m.group(1)))
    if m := _BPM_ONE_RE.search(text):
        centre = float(m.group(1))
        return TempoRange(min_bpm=max(30.0, centre - 8), max_bpm=min(250.0, centre + 8))
    for word, (lo, hi) in TEMPO_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return TempoRange(min_bpm=lo, max_bpm=hi)
    return TempoRange()


def _clean_entity(raw: str) -> str:
    s = raw.strip(" ,.;:!?\"'")
    s = re.sub(
        r"^(?:the\s+band\s+|artists?\s+|some\s+|any\s+|more\s+|a\s+bit\s+of\s+)", "", s, flags=re.I
    )
    s = re.sub(r"\s+(?:but|though|however|except|for|when|while|with)\b.*$", "", s, flags=re.I)
    return s.strip()


def _extract_seeds(original: str) -> tuple[list[str], list[str]]:
    """Return (seed_artists, avoid_artists) as raw surface strings."""
    seeds: list[str] = []
    avoid: list[str] = []
    low = original.lower()

    for pat in _SEED_PATTERNS:
        for m in pat.finditer(low):
            ent = _clean_entity(m.group(1))
            if ent and len(ent) > 1:
                seeds.append(ent)

    for m in _AVOID_RE.finditer(low):
        ent = _clean_entity(m.group(1))
        head = ent.split()[0] if ent else ""
        if ent and head not in _NON_ARTIST_AVOID and len(ent) > 2:
            avoid.append(ent)

    # Quoted spans are almost always titles or artist names.
    for m in _QUOTED_RE.finditer(original):
        ent = _clean_entity(m.group(1))
        if ent:
            seeds.append(ent)

    def dedupe_by_normal_form(xs: list[str]) -> list[str]:
        """Dedupe on the normalised form so "Bon Iver" and "bon iver" collapse."""
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            key = normalize(x)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    return dedupe_by_normal_form(seeds)[:6], dedupe_by_normal_form(avoid)[:6]


def _extract_moods(text: str) -> tuple[dict[str, float], list[str], str | None, TempoRange | None]:
    """Average the audio targets of every mood phrase found."""
    acc: dict[str, list[float]] = {}
    themes: list[str] = []
    curve: str | None = None
    tempo: TempoRange | None = None

    # Longest phrases first so "late night" wins over "night".
    for phrase in sorted(MOOD_LEXICON, key=len, reverse=True):
        if not re.search(rf"\b{re.escape(phrase)}\b", text):
            continue
        entry = MOOD_LEXICON[phrase]
        for k, v in entry.audio.items():
            acc.setdefault(k, []).append(v)
        themes.extend(entry.themes or (phrase,))
        if entry.curve and curve is None:
            curve = entry.curve
        if entry.tempo and tempo is None:
            tempo = TempoRange(min_bpm=entry.tempo[0], max_bpm=entry.tempo[1])

    audio = {k: float(sum(v) / len(v)) for k, v in acc.items()}
    return audio, themes, curve, tempo


def _reconcile_curve(curve: str | None, audio: dict[str, float]) -> str | None:
    """Drop a mood-implied curve that contradicts the requested energy.

    "chill rainy morning" matches both `chill` (energy 0.32) and `morning`
    (curve "build"). Taking the curve at face value produces a playlist that
    ramps into high-energy tracks, which is the opposite of what was asked.
    An implied curve only survives if it agrees with the energy target.
    """
    energy = audio.get("energy")
    if curve is None or energy is None:
        return curve
    if curve == "build" and energy < 0.45:
        return "steady"
    if curve == "wind_down" and energy > 0.62:
        return "steady"
    return curve


def parse_intent(query: str, known_tags: set[str] | None = None) -> PlaylistIntent:
    """Parse a free-text playlist request into a structured intent."""
    original = query or ""
    text = normalize(original)

    constraints = _extract_counts(text, Constraints())
    tempo = _extract_tempo(text)
    audio_map, mood_themes, curve, mood_tempo = _extract_moods(text)
    if not tempo.is_set() and mood_tempo is not None:
        tempo = mood_tempo

    # An explicitly stated shape always wins over one merely implied by a mood.
    for phrase, c in CURVE_PHRASES.items():
        if phrase in text:
            curve = c
            break
    else:
        curve = _reconcile_curve(curve, audio_map)

    seeds, avoid = _extract_seeds(original)
    eras = decade_tags(original)

    # Remaining content tokens that exist in the tag vocabulary become themes.
    # Restricting to known tags keeps the query inside the space the index can
    # actually serve instead of inventing vocabulary.
    # Words already consumed by constraint parsing must not also become themes.
    # "nothing explicit" sets the clean filter; leaking `nothing` and `explicit`
    # into the tag centroid pulls the query toward unrelated music.
    consumed: set[str] = set()
    for phrase in CLEAN_PHRASES:
        if phrase in text:
            consumed.update(phrase.split())
    for word in TEMPO_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            consumed.update(word.split())
    consumed.update(_UNIT_WORDS)

    extra: list[str] = []
    if known_tags:
        seed_words = {w for s in seeds + avoid for w in normalize(s).split()}
        era_set = set(eras)
        for tok in tokenize(original):
            if tok not in known_tags or tok in mood_themes or tok in seed_words:
                continue
            # "90s" and "1990s" are the same request; keep the canonical one.
            if set(decade_tags(tok)) & era_set:
                continue
            # Bigrams built across a stopword ("party playlist", "and chill")
            # exist in the vocabulary but are noise as query themes.
            if " " in tok and any(w in STOPWORDS for w in tok.split()):
                continue
            if any(w in consumed for w in tok.split()):
                continue
            extra.append(tok)

    themes: list[str] = []
    for t in [*mood_themes, *extra]:
        if t not in themes:
            themes.append(t)
    # A bigram subsumes its parts: keeping "hip hop" alongside "hip" and "hop"
    # triples the weight of one concept and drags in unrelated tracks.
    covered = {w for t in themes if " " in t for w in t.split()}
    themes = [t for t in themes if " " in t or t not in covered]

    return PlaylistIntent(
        summary=original.strip()[:280],
        themes=themes[:12],
        genres=[],
        seed_artists=seeds,
        seed_tracks=[],
        avoid_artists=avoid,
        eras=eras,
        audio=AudioTargets(**audio_map),
        tempo=tempo,
        constraints=constraints,
        energy_curve=curve or "steady",
        notes="parsed by the deterministic offline planner",
    )
