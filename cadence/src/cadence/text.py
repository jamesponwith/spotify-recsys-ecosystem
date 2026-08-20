"""Text normalisation shared by the offline build and the online query path.

Playlist titles are the project's bridge from natural language to music, so the
same normaliser must run over MPD titles at build time and over user queries at
serve time. Any divergence silently destroys recall.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Hashable, Iterable
from typing import TypeVar

# Words that appear in thousands of playlist titles while carrying no musical
# signal. Dropping them keeps the tag vocabulary about *music*, not about the
# act of making a playlist.
# Kept as a wrapped block rather than a list literal: the autofixed one-line
# form is 90 words wide and unreviewable.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and the of for to in on my me you your our it its is are be with by at as
    playlist playlists list lists mix mixes music musica songs song track tracks
    tunes stuff things thing new old good great best fav favs favorite favorites
    favourite favourites vol volume pt part no na da de la el los las un une
    spotify shuffle random misc other others more all every some what that this
    now then just too very really much many one two three four five
    """.split()
)

T = TypeVar("T", bound=Hashable)


def dedupe(items: Iterable[T]) -> list[T]:
    """First-occurrence-wins dedupe that preserves order."""
    seen: set[T] = set()
    out: list[T] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# Decade surface forms -> canonical decade tag. MPD spans 2010-2017, so a bare
# two-digit decade below 30 resolves to the 2000s/2010s, not the 1900s.
_DECADE_RE = re.compile(r"\b(?:(19|20)?(\d0))(?:'?s)\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")


def _canonical_decade(century: str | None, decade: str) -> str | None:
    d = int(decade)
    if century:
        return f"{century}{decade}s"
    if d in (50, 60, 70, 80, 90):
        return f"19{decade}s"
    if d in (0, 10, 20):
        return f"20{decade}s"
    return None


def normalize(text: str) -> str:
    """Lowercase, strip accents and emoji, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9'&\s]+", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decade_tags(text: str) -> list[str]:
    """Extract canonical decade tags such as '1990s' from free text."""
    out: list[str] = []
    low = text.lower()
    for century, decade in _DECADE_RE.findall(low):
        tag = _canonical_decade(century or None, decade)
        if tag:
            out.append(tag)
    for year in _YEAR_RE.findall(low):
        out.append(f"{year[:3]}0s")
    return dedupe(out)


def tokenize(text: str, *, min_len: int = 2, bigrams: bool = True) -> list[str]:
    """Content unigrams + bigrams, plus canonical decade tags.

    Bigrams matter here: 'road trip', 'rainy day' and 'study music' are single
    concepts in playlist-title space, and unigram-only tokenisation loses them.
    """
    tags = decade_tags(text)
    norm = normalize(text)
    if not norm:
        return tags

    raw = [w for w in norm.split() if len(w) >= min_len]
    content = [w for w in raw if w not in STOPWORDS and not w.isdigit()]

    tokens: list[str] = list(tags)
    tokens.extend(content)
    if bigrams:
        # Bigrams are built over the *raw* sequence so that "day" in "rainy day"
        # survives even though "day" alone would be kept and "rainy day" adds the
        # compound meaning.
        for a, b in zip(raw, raw[1:], strict=False):
            if a in STOPWORDS and b in STOPWORDS:
                continue
            if a.isdigit() and b.isdigit():
                continue
            tokens.append(f"{a} {b}")

    return dedupe(tokens)


def title_tokens(title: str, *, max_tokens: int = 6, min_len: int = 2) -> list[str]:
    """Tokens for one playlist title, capped so a rambling title cannot dominate
    the tag co-occurrence matrix."""
    return tokenize(title, min_len=min_len)[:max_tokens]


def join_fields(*fields: Iterable[str] | str | None) -> str:
    parts: list[str] = []
    for f in fields:
        if f is None:
            continue
        if isinstance(f, str):
            parts.append(f)
        else:
            parts.extend(x for x in f if x)
    return " ".join(parts)
