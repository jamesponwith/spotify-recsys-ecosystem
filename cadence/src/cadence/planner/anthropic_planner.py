"""Claude-backed planner.

Two jobs, both with structured output so nothing has to be parsed out of prose:

1. ``plan``       free text -> ``PlaylistIntent``
2. ``write_copy`` intent + the *actual selected tracks* -> title + description

Grounding rules that make this safe to ship:

* The model is given the retrievable tag vocabulary and told to prefer it. An
  intent field the index cannot serve is a wasted field.
* The model never picks tracks. It expresses *intent*; retrieval picks tracks
  from the catalog. This is what makes track hallucination structurally
  impossible rather than merely unlikely.
* Copy is written only after selection, and is validated against the real
  tracklist by ``cadence.explain`` before it is shown.
* Every failure path returns the offline planner's answer with a warning
  attached, so a provider outage degrades quality instead of breaking.
"""

from __future__ import annotations

import os
import time
from typing import Any, cast

from pydantic import BaseModel, Field

from ..config import DEFAULT
from ..types import PlaylistIntent
from .base import Copy, PlanResult, _template_copy
from .offline import parse_intent

SYSTEM_PROMPT = """\
You translate a listener's free-text playlist request into a structured intent \
object for a music retrieval engine.

You are not choosing songs. A retrieval system selects tracks from a fixed \
catalog using the intent you produce, so your job is to express *what the \
listener wants* precisely enough for that system to act on.

Guidelines:
- Fill only what the request supports. Leave a field unset rather than guessing; \
an unset audio dimension is ignored, but a guessed one actively steers results.
- `themes` should read like words people put in real playlist titles \
("rainy day", "gym", "throwback", "study"), not like prose.
- Put named artists in `seed_artists` and named songs in `seed_tracks` \
("Title - Artist"). Do not invent either.
- `audio` values are Spotify audio features on a 0-1 scale. Set a dimension only \
when the request implies it: "upbeat" implies high energy and valence; \
"acoustic" implies high acousticness; "instrumental" implies instrumentalness.
- Use `energy_curve` when the request describes a shape over time \
(warming up, winding down, peaking in the middle).
- Hard requirements (an exact song count, a duration, "nothing explicit", a BPM \
window) belong in `constraints` and `tempo`.
"""

TAG_HINT = """\
The retrieval index is built from real playlist titles. These tags are \
available and score well; prefer them in `themes` when they fit the request:
{tags}
"""

COPY_SYSTEM = """\
You write the title and description for a playlist that has already been \
assembled.

Rules:
- The tracklist you are given is the complete, final playlist.
- Refer only to artists and songs that appear in that list. Never mention any \
other artist or song.
- Title: at most 6 words, no quotes, no emoji.
- Description: one or two sentences describing the feel and the arc. Do not \
list the tracks back; the listener can already see them.
"""


class _CopyModel(BaseModel):
    title: str = Field(..., max_length=80)
    description: str = Field(..., max_length=400)


class AnthropicPlanner:
    """Planner backed by the Claude API, degrading to rules on any failure."""

    name = "anthropic"

    def __init__(self, model: str | None = None, max_tags: int = 220) -> None:
        import anthropic  # imported here so the dependency stays optional

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # An unset key does not always mean "no credentials" (an `ant auth
            # login` profile also works), so we construct the client and let it
            # resolve; only a hard failure falls back.
            pass
        self._client = anthropic.Anthropic(timeout=DEFAULT.llm.timeout_s)
        self.model = model or DEFAULT.llm.model
        self.max_tags = max_tags

    # ---- intent ---------------------------------------------------------
    def plan(self, query: str, known_tags: set[str] | None = None) -> PlanResult:
        t0 = time.perf_counter()
        warnings: list[str] = []
        system = SYSTEM_PROMPT
        if known_tags:
            sample = sorted(known_tags)[: self.max_tags]
            system = system + "\n\n" + TAG_HINT.format(tags=", ".join(sample))
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=DEFAULT.llm.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config=cast(Any, {"effort": DEFAULT.llm.effort}),
                messages=[{"role": "user", "content": query}],
                output_format=PlaylistIntent,
            )
            intent = response.parsed_output
            if intent is None:
                raise ValueError("structured output returned no parsed intent")
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all fallback
            warnings.append(f"llm planner failed ({type(exc).__name__}); used offline parser")
            intent = parse_intent(query, known_tags)
            return PlanResult(
                intent=intent,
                provider="offline (fallback)",
                warnings=warnings,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # The LLM sets intent, but hard numeric constraints are re-derived from
        # the raw text: a regex that finds "45 minutes" is more reliable than a
        # model that has to remember to copy it into a field.
        rules = parse_intent(query, known_tags)
        intent = _merge_hard_constraints(intent, rules)
        return PlanResult(
            intent=intent,
            provider=self.name,
            warnings=warnings,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ---- copy -----------------------------------------------------------
    def write_copy(self, query: str, intent: PlaylistIntent, track_lines: list[str]) -> Copy:
        listing = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(track_lines))
        user = f"Listener asked for: {query}\n\nFinal tracklist:\n{listing}"
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=1024,
                system=COPY_SYSTEM,
                thinking={"type": "adaptive"},
                output_config=cast(Any, {"effort": "low"}),
                messages=[{"role": "user", "content": user}],
                output_format=_CopyModel,
            )
            parsed = response.parsed_output
            if parsed is None:
                raise ValueError("no parsed copy")
            return Copy(title=parsed.title.strip(), description=parsed.description.strip())
        except Exception:  # noqa: BLE001
            return _template_copy(intent, len(track_lines))


def _merge_hard_constraints(llm: PlaylistIntent, rules: PlaylistIntent) -> PlaylistIntent:
    """Prefer regex-extracted hard numbers; keep the LLM's semantic fields."""
    c = llm.constraints.model_copy()
    r = rules.constraints
    if r.track_count is not None:
        c.track_count = r.track_count
    if r.target_duration_minutes is not None:
        c.target_duration_minutes = r.target_duration_minutes
    if r.exclude_explicit:
        c.exclude_explicit = True

    tempo = llm.tempo
    if r.exclude_explicit or rules.tempo.is_set():
        tempo = rules.tempo if rules.tempo.is_set() else tempo

    merged = llm.model_copy(update={"constraints": c, "tempo": tempo})
    # Union the era tags: decade mentions are cheap to detect and easy to miss.
    eras = list(dict.fromkeys([*llm.eras, *rules.eras]))
    return merged.model_copy(update={"eras": eras})
