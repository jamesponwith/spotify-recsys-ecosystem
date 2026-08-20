"""Planner interface and provider selection.

Two implementations satisfy one protocol:

* ``OfflinePlanner``  - deterministic rules, no network, always available.
* ``AnthropicPlanner`` - Claude with structured output, falling back to the
  offline planner on any error.

Keeping the interface this narrow means the retrieval and assembly stages never
learn whether an LLM was involved, so every metric downstream is measured
against the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import PlaylistIntent
from .offline import parse_intent


@dataclass
class PlanResult:
    intent: PlaylistIntent
    provider: str
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class Copy:
    title: str
    description: str


@runtime_checkable
class Planner(Protocol):
    name: str

    def plan(self, query: str, known_tags: set[str] | None = None) -> PlanResult: ...

    def write_copy(self, query: str, intent: PlaylistIntent, track_lines: list[str]) -> Copy: ...


def _template_copy(intent: PlaylistIntent, n_tracks: int) -> Copy:
    """Deterministic copy used by the offline planner and as the LLM fallback."""
    # Era tags often also arrive as themes; showing "Party 1990s 1990s" is a
    # tell that nobody looked at the output.
    eras = {e.lower() for e in intent.eras}
    themes = [t for t in intent.themes if t.lower() not in eras]
    bits = [*themes[:2], *intent.genres[:1], *intent.eras[:1]]
    # Title-case each word except decade tags, where .title() would give "1990S".
    words = " ".join(b for b in bits if b).strip().split()
    label = " ".join(w if w[:1].isdigit() else w.title() for w in words)
    title = label if label else "Your Playlist"
    if intent.seed_artists:
        title = f"{title} — {intent.seed_artists[0].title()} Radio".strip(" —")

    parts = [f"{n_tracks} tracks"]
    if themes:
        parts.append("for " + ", ".join(themes[:3]))
    if intent.eras:
        parts.append("leaning " + ", ".join(intent.eras))
    if intent.tempo.is_set():
        lo = intent.tempo.min_bpm
        hi = intent.tempo.max_bpm
        if lo and hi:
            parts.append(f"{lo:.0f}-{hi:.0f} BPM")
        elif lo:
            parts.append(f"above {lo:.0f} BPM")
        elif hi:
            parts.append(f"under {hi:.0f} BPM")
    if intent.energy_curve != "steady":
        parts.append(intent.energy_curve.replace("_", " ") + " arc")
    return Copy(title=title[:80], description=", ".join(parts).capitalize() + ".")


class OfflinePlanner:
    """Rules only. The default, and the reference implementation for evaluation."""

    name = "offline"

    def plan(self, query: str, known_tags: set[str] | None = None) -> PlanResult:
        import time

        t0 = time.perf_counter()
        intent = parse_intent(query, known_tags)
        return PlanResult(
            intent=intent,
            provider=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def write_copy(self, query: str, intent: PlaylistIntent, track_lines: list[str]) -> Copy:
        return _template_copy(intent, len(track_lines))


def get_planner(provider: str | None = None, model: str | None = None) -> Planner:
    """Factory. Falls back to the offline planner whenever the LLM path is
    unavailable, so a missing key is a quality change, not an outage."""
    from ..config import DEFAULT

    provider = (provider or DEFAULT.llm.provider or "offline").lower()
    if provider in ("offline", "none", "rules"):
        return OfflinePlanner()
    if provider == "anthropic":
        try:
            from .anthropic_planner import AnthropicPlanner

            return AnthropicPlanner(model=model or DEFAULT.llm.model)
        except Exception:  # noqa: BLE001 - any import/config failure degrades gracefully
            return OfflinePlanner()
    raise ValueError(f"unknown planner provider: {provider!r}")
