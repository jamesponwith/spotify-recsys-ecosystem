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
import re
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


OST = "ostinato/artifacts/sim_report.json"
GAM = "gamut/artifacts/audit_report.json"
SEG = "segue/artifacts/eval_report.json"
TIM = "timbre/artifacts/phase0_report.json"


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
]


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def check_self_consistency(text: str) -> list[str]:
    """Catch prose arithmetic, which no artifact can adjudicate.

    Every claim above compares the page to a JSON file. This compares the page to
    *itself*, which is a different failure and the one that actually happened:
    Concerto leaving took a row out of the summary table and left the opening
    sentence saying "Six applications ... three of them answered it no" above five
    visible rows. Nothing sourced from an artifact could have noticed, because
    nothing was wrong with any artifact.

    The rule is simply that a number a reader can verify by counting must match
    what they would count.
    """
    problems: list[str] = []

    rows = re.findall(r"^\| \*\*\[([a-z]+)\]\(\1/\)\*\*", text, re.M)
    # The whitelist in .gitignore is the authority on what this repo contains --
    # not "every directory with a pyproject.toml", which also finds unrelated
    # projects sharing the parent directory.
    whitelist = re.findall(r"^!/([a-z-]+)/$", (ROOT / ".gitignore").read_text(), re.M)
    dirs = sorted(w for w in whitelist if (ROOT / w / "pyproject.toml").exists())

    if sorted(rows) != dirs:
        problems.append(f"summary table lists {sorted(rows)} but the repo contains {dirs}")

    m = re.search(r"^([A-Z][a-z]+) applications built on", text, re.M)
    if not m:
        problems.append("could not find the 'N applications built on' opening")
    else:
        stated = WORDS.get(m.group(1).lower())
        if stated != len(rows):
            problems.append(
                f"opening says {m.group(1).lower()!r} applications; the table has {len(rows)} rows"
            )

    m = re.search(r"not (\w+) demos of the same idea", text)
    if m and WORDS.get(m.group(1)) != len(rows):
        problems.append(
            f"'not {m.group(1)} demos' disagrees with the {len(rows)} rows in the table"
        )

    return problems


def main() -> int:
    text = README.read_text()
    missing_src, absent, drifted = [], [], []
    inconsistent = check_self_consistency(text)

    for c in CLAIMS:
        path = ROOT / c.source
        if not path.exists():
            # One line per artifact, not per claim: six copies of the same
            # missing file is noise, and the missing artifact — not each claim
            # riding on it — is the one problem the reader has to fix.
            line = f"{c.app}: {c.source} not built"
            if line not in missing_src:
                missing_src.append(line)
            continue
        actual = c.compute(load(c.source))
        if actual != c.literal:
            drifted.append(f"{c.app}: README says {c.literal!r}, artifact says {actual!r}")
        elif c.literal not in text:
            absent.append(f"{c.app}: {c.literal!r} no longer appears in README.md")

    # A missing artifact is a failure, not a skip. Every app's `make clean`
    # deletes artifacts/*.json, so treating "not built" as "nothing to check"
    # made `make clean && make lint-all` green while verifying nothing.
    for label, rows in (
        ("DRIFTED — the README contradicts its source", drifted),
        ("ABSENT — verified claim is no longer on the page", absent),
        ("MISSING — registered artifact not built, claims unverifiable", missing_src),
        ("INCONSISTENT — the page contradicts itself", inconsistent),
    ):
        if rows:
            print(f"\n{label}:")
            for r in rows:
                print(f"  {r}")

    if drifted or absent or missing_src or inconsistent:
        n = len(drifted) + len(absent) + len(missing_src) + len(inconsistent)
        print(
            f"\n{n} problem(s). Restore the artifact from git (or regenerate "
            "the app), then fix README.md."
        )
        return 1
    print(
        f"root README: all {len(CLAIMS)} claims match their source artifacts, "
        "and the page agrees with itself"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
