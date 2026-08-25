"""Verify every hand-typed number in the root README against its source artifact.

The root README is the only document in this repo whose figures are typed rather
than generated, which is exactly why it is the only one that has ever drifted:
Ostinato's headline sat superseded on a public page until a second session caught
it.

The obvious fix is to generate the page. That is the wrong fix. The root README's
value is the argument it makes -- which project constrained which, what was wrong
and what that cost -- and generated prose is worse prose. What drifts is not the
narrative but the numbers inside it.

So this verifies instead of generating. Each claim below names a literal string in
the README and the computation that must reproduce it from an app's own JSON. The
prose stays hand-written; the arithmetic stays honest. Run it in `make lint`.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


@dataclass
class Claim:
    app: str
    literal: str  # the exact string the README must contain
    source: str  # artifact it comes from
    compute: Callable[[dict], str]


def _drift(d: dict, arm: str, key: str = "artist_gini") -> float:
    h = d["arms"][arm]
    return h[-1][key] - h[0][key]


def _band(d: dict) -> float:
    return 2 * min(st.stdev([h["artist_gini"] for h in d["arms"][a]]) for a in d["arms"])


def _pairs(d: dict) -> list[float]:
    a, c = d["arms"]["exposure_aware"], d["arms"]["closed_loop"]
    return [a[i]["tail_share"] - c[i]["tail_share"] for i in range(len(c))]


def _cheapest_arm(d: dict) -> float:
    """The lowest price multiple any arm reaches in any cell of the grid."""
    return min(min(v["price_multiple"] for v in c["metrics"].values()) for c in d["cells"])


def _claim(d: dict, key: str) -> dict:
    return next(c for c in d["claims"] if c["key"] == key)


OST = "ostinato/artifacts/sim_report.json"
GAM = "gamut/artifacts/audit_report.json"
SEG = "segue/artifacts/eval_report.json"
TIM = "timbre/artifacts/phase0_report.json"
CON_M = "concerto/artifacts/simulation.json"
CON_S = "concerto/artifacts/sensitivity.json"


def _arm(d: dict, key: str) -> dict:
    return next(a for a in d["arms"] if a["arm"] == key)


CLAIMS: list[Claim] = [
    # --- ostinato: the figures that drifted once already -------------------
    Claim("ostinato", "+0.00153", OST, lambda d: f"{_drift(d, 'organic'):+.5f}"),
    Claim(
        "ostinato", "−0.00003", OST, lambda d: f"{_drift(d, 'closed_loop'):+.5f}".replace("-", "−")
    ),
    Claim(
        "ostinato",
        "−0.00078",
        OST,
        lambda d: f"{_drift(d, 'exposure_aware'):+.5f}".replace("-", "−"),
    ),
    Claim("ostinato", "±0.00020", OST, lambda d: f"±{_band(d):.5f}"),
    Claim(
        "ostinato",
        "+4.76 points",
        OST,
        lambda d: f"+{100 * sum(_pairs(d)) / len(_pairs(d)):.2f} points",
    ),
    Claim(
        "ostinato",
        "6 of 6 rounds",
        OST,
        lambda d: f"{sum(1 for p in _pairs(d) if p > 0)} of {len(_pairs(d))} rounds",
    ),
    # --- gamut -------------------------------------------------------------
    Claim("gamut", "43.6%", GAM, lambda d: f"{d['baseline']['top1pct_artist_share']:.1%}"),
    Claim("gamut", "0.951", GAM, lambda d: f"{d['baseline']['artist_gini']:.3f}"),
    Claim("gamut", "2.99%", GAM, lambda d: f"{d['baseline']['track_coverage']:.2%}"),
    Claim("gamut", "10.7%", GAM, lambda d: f"{d['baseline']['pool']['track_coverage']:.1%}"),
    Claim(
        "gamut",
        "0.936",
        GAM,
        lambda d: f"{min(x['artist_gini'] for x in d['frontier'] + d['artist_caps']):.3f}",
    ),
    # --- segue -------------------------------------------------------------
    Claim(
        "segue",
        "13.7%",
        SEG,
        lambda d: "{:.1%}".format(
            -sum(
                d["seed_counts"][k]["systems"]["segue"]["clicks"]
                / d["seed_counts"][k]["systems"]["centroid"]["clicks"]
                - 1
                for k in d["seed_counts"]
            )
            / len(d["seed_counts"])
        ),
    ),
    Claim(
        "segue",
        "10.0%",
        SEG,
        lambda d: "{:.1%}".format(
            d["seed_counts"]["25"]["systems"]["segue"]["r_precision"]
            / d["seed_counts"]["25"]["systems"]["segue_shuffled"]["r_precision"]
            - 1
        ),
    ),
    # --- timbre ------------------------------------------------------------
    Claim("timbre", "13.2%", TIM, lambda d: f"{d['gate_0']['oracle_recovery_ratio']:.1%}"),
    Claim("timbre", "0.198", TIM, lambda d: f"{d['fits']['ridge']['mean_cosine']:.3f}"),
    Claim("timbre", "1,718", TIM, lambda d: f"{d['data']['n_queries']:,}"),
    # A threshold and a verdict, not measurements. Neither is computed from data,
    # so nothing else in the pipeline would notice them going stale: flip
    # gate_oracle_fraction in timbre's config and the README still reads "the bar
    # was 25%". Registering them turns a config edit into a failing check.
    Claim(
        "timbre",
        "the bar was 25%",
        TIM,
        lambda d: f"the bar was {d['gate_0']['threshold_oracle_fraction']:.0%}",
    ),
    Claim(
        "timbre",
        "Killed at its own gate",
        TIM,
        lambda d: (
            "Killed at its own gate"
            if not d["gate_0"]["passed"]
            else "PASSED ITS GATE -- the README still says it was killed"
        ),
    ),
    # --- concerto ----------------------------------------------------------
    # Concerto has no corpus, so its README figures are outputs of a simulation
    # rather than measurements of anything. That makes them *more* worth pinning,
    # not less: nothing external would ever contradict them.
    Claim("concerto", "180 parameter cells", CON_S, lambda d: f"{d['n_cells']} parameter cells"),
    Claim(
        "concerto",
        "180 of 180 cells",
        CON_S,
        lambda d: "{held} of {of} cells".format(**_claim(d, "margin_beats_identity")),
    ),
    # The threshold in "every policy leaves the average fan paying more than
    # 1.2x face", floored to the tenth. Verifies the sentence is still true *and*
    # still tight -- if the cheapest arm fell to 1.15x this computes "1.1x face"
    # and drifts, and if it rose to 1.45x it drifts the other way for
    # understating the result.
    Claim(
        "concerto",
        "1.2x face",
        CON_S,
        lambda d: f"{math.floor(_cheapest_arm(d) * 10) / 10:.1f}x face",
    ),
    Claim("concerto", "2.33x", CON_M, lambda d: f"{_arm(d, 'queue')['price_multiple']:.2f}x"),
    Claim("concerto", "3.06x", CON_M, lambda d: f"{_arm(d, 'clearing')['price_multiple']:.2f}x"),
]


def main() -> int:
    text = README.read_text()
    missing_src, absent, drifted = [], [], []

    for c in CLAIMS:
        path = ROOT / c.source
        if not path.exists():
            missing_src.append(f"{c.app}: {c.source} not built")
            continue
        actual = c.compute(load(c.source))
        if actual != c.literal:
            drifted.append(f"{c.app}: README says {c.literal!r}, artifact says {actual!r}")
        elif c.literal not in text:
            absent.append(f"{c.app}: {c.literal!r} no longer appears in README.md")

    for label, rows in (
        ("DRIFTED — the README contradicts its source", drifted),
        ("ABSENT — verified claim is no longer on the page", absent),
        ("SKIPPED — artifact not built", missing_src),
    ):
        if rows:
            print(f"\n{label}:")
            for r in rows:
                print(f"  {r}")

    if drifted or absent:
        print(f"\n{len(drifted) + len(absent)} problem(s). Regenerate the app, then fix README.md.")
        return 1
    checked = len(CLAIMS) - len(missing_src)
    print(f"root README: {checked}/{len(CLAIMS)} claims match their source artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
