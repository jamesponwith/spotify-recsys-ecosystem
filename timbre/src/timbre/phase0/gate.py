"""Rule on Gate 0.

A pure function over the measured recalls, so the verdict can be re-derived from
a stored report without repeating the experiment, and so the degenerate case
below is unit-testable.
"""

from __future__ import annotations

from typing import Any

from ..config import Phase0Config


def rule(results: dict[str, dict[str, float]], cfg: Phase0Config) -> dict[str, Any]:
    best = max(
        (k for k in results if k.startswith("content_")),
        key=lambda k: results[k]["recall_at_100"],
    )
    content = results[best]["recall_at_100"]
    floor = results["random"]["recall_at_100"]
    ceiling = results["oracle"]["recall_at_100"]

    # A floor of exactly zero does not make the ratio infinite -- it makes it
    # undefined. `float("inf")` also serialises to the literal `Infinity`, which
    # is not valid JSON and is rejected by strict parsers. And a criterion that
    # cannot be failed is not a criterion: it is recorded as null, flagged
    # vacuous, and the verdict rests on the oracle ratio alone.
    vacuous = floor <= 0.0
    multiple = None if vacuous else content / floor
    oracle_ratio = content / ceiling if ceiling > 0 else 0.0
    beats_floor = content > 0.0 if multiple is None else multiple >= cfg.gate_random_multiple

    return {
        "best_content_system": best,
        "content_recall": content,
        "random_recall": floor,
        "oracle_recall": ceiling,
        "random_multiple": multiple,
        "random_criterion_vacuous": vacuous,
        "beats_random_floor": bool(beats_floor),
        "oracle_recovery_ratio": oracle_ratio,
        "threshold_random_multiple": cfg.gate_random_multiple,
        "threshold_oracle_fraction": cfg.gate_oracle_fraction,
        "passed": bool(beats_floor and oracle_ratio >= cfg.gate_oracle_fraction),
    }
