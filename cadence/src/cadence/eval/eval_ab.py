"""Paired A/B pricing of retrieval configuration.

``cadence evaluate`` reports each arm's mean with an *unpaired* standard error,
and at n=400 that band is ~0.0149 R-precision — wider than any single knob in
``RetrievalConfig`` is worth. Read that way every config change looks like "no
difference", which is a statement about the harness rather than about the knob.

Both arms here score the *same* challenges from the *same* planned intents, so
the comparison can be paired: differencing per challenge cancels the
between-challenge variance the two arms share, and the band on the difference
collapses by two to three orders of magnitude. So the command reports the
smallest difference it *could* have seen beside the difference it *did* see,
and prints the unpaired band alongside so the gap between the two is visible
rather than asserted.

Scope is deliberate. Only ``rrf_k`` and the seven channel weights can be set,
because those are the parameters this harness actually executes — it stops at
fusion and never runs selection or sequencing. An A/B on ``mmr_lambda`` here
would return a null that means nothing at all, so the parser refuses it by name
instead of letting it look measured.

A verdict of "not detectable" is a statement about resolution, never evidence
that the shipped value is optimal: it says the effect is smaller than the band,
and the band is printed next to it so a reader can see how small that is.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..catalog import Catalog
from ..config import ARTIFACTS, DATA_PROCESSED, DEFAULT, Config
from ..engine import CadenceEngine
from .metrics import BAND_Z, MetricAccumulator, evaluate_ranking, within_band
from .run_eval import DEPTH
from .splits import Challenge, load_splits

# The seven weights RRF actually consults, named from the shipped config so a
# channel added there cannot silently become un-priceable here.
CHANNEL_WEIGHT_KEYS = frozenset(DEFAULT.retrieval.channel_weights)
TUNABLE = frozenset({"rrf_k"}) | CHANNEL_WEIGHT_KEYS

# Named so the refusal can say *why* rather than "unknown knob". These are real
# parameters that simply live past this harness's last stage.
ASSEMBLY_KEYS = frozenset(
    {
        "default_length",
        "max_tracks_per_artist",
        "mmr_lambda",
        "audio_affinity_weight",
        "beam_width",
        "w_tempo",
        "w_key",
        "w_energy_curve",
        "w_artist_adjacent",
    }
)


def parse_overrides(items: list[str]) -> dict[str, float]:
    """Parse ``KEY=VALUE`` strings into a retrieval-config override map."""
    out: dict[str, float] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{key!r} needs a number, got {raw.strip()!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"{key!r} must be finite, got {value!r}")
        if key in ASSEMBLY_KEYS:
            raise ValueError(
                f"{key!r} is an assembly knob and this harness stops at fusion, so an "
                "A/B on it would return a null that means nothing. Use "
                "`cadence eval-constraints` or `cadence eval-affinity` instead."
            )
        if key not in TUNABLE:
            raise ValueError(f"unknown knob {key!r}; this command prices {sorted(TUNABLE)}")
        # rrf_k is the denominator's offset in w / (k + rank + 1); at k <= 0 some
        # rank divides by zero and the arm would be scored on infinities.
        if key == "rrf_k" and value <= 0:
            raise ValueError(f"rrf_k must be > 0, got {value!r}")
        if key in out and out[key] != value:
            raise ValueError(f"{key!r} given twice with different values")
        out[key] = value
    return out


def apply_overrides(cfg: Config, overrides: dict[str, float]) -> Config:
    """A copy of ``cfg`` with the retrieval overrides applied.

    Copies the weight dict rather than mutating it: ``DEFAULT`` is a module-level
    singleton, and an in-place edit would silently move the other arm too.
    """
    if not overrides:
        return cfg
    rcfg = cfg.retrieval
    weights = dict(rcfg.channel_weights)
    rrf_k = rcfg.rrf_k
    for key, value in overrides.items():
        if key == "rrf_k":
            rrf_k = value
        else:
            weights[key] = value
    return replace(cfg, retrieval=replace(rcfg, rrf_k=rrf_k, channel_weights=weights))


def label(overrides: dict[str, float]) -> str:
    return ",".join(f"{k}={v:g}" for k, v in sorted(overrides.items())) if overrides else "shipped"


def plan_intents(engine: CadenceEngine, items: list[Challenge]) -> list:
    """Plan every challenge once. Both arms then retrieve from the identical
    intent, so planner variation cannot leak into the delta."""
    tags = engine.known_tags
    return [engine.planner.plan(ch.title, tags).intent for ch in items]


def score_arm(
    engine: CadenceEngine,
    items: list[Challenge],
    intents: list,
    artist_ids: np.ndarray,
    *,
    use_reranker: bool = False,
) -> MetricAccumulator:
    """Score one arm, keeping the per-challenge vectors the pairing needs."""
    acc = MetricAccumulator()
    for ch, intent in zip(items, intents, strict=True):
        seeds = np.asarray(ch.seed_tracks, dtype=np.int64)
        trace = engine.retrieve(
            intent,
            extra_seed_indices=seeds,
            exclude=seeds,
            top_n=DEPTH,
        )
        preds = trace.candidates.indices
        if use_reranker and engine.reranker is not None and len(preds):
            scores = engine.reranker.score(engine.catalog, intent, trace)
            preds = preds[np.argsort(-scores, kind="stable")]
        acc.update(evaluate_ranking(preds, set(ch.held_out), artist_ids))
    return acc


def compare(arm_a: MetricAccumulator, arm_b: MetricAccumulator) -> dict[str, dict]:
    """Per-metric paired comparison of arm B against arm A.

    ``band`` is ±BAND_Z×SE of the *difference*; ``unpaired_band`` is the band
    ``cadence evaluate`` would have used, with the two arms' errors added in
    quadrature. Reporting both is the point: where they disagree, the shipped
    harness was calling a real difference noise.
    """
    a, b = arm_a.summary(), arm_b.summary()
    deltas = arm_b.paired_deltas(arm_a)
    n = int(a["n"])
    out: dict[str, dict] = {}
    for name in arm_a.values:
        if name not in arm_b.values:
            continue
        se_a, se_b = a[f"{name}_se"], b[f"{name}_se"]
        delta, delta_se = deltas[f"{name}_delta"], deltas[f"{name}_delta_se"]
        band = BAND_Z * delta_se
        # One challenge gives no spread to estimate a band from, and a band of
        # zero would make any nonzero delta look infinitely well resolved.
        detectable = bool(n > 1 and abs(delta) > band)
        out[name] = {
            "mean_a": a[name],
            "se_a": se_a,
            "mean_b": b[name],
            "se_b": se_b,
            "delta": delta,
            "delta_se": delta_se,
            "band": band,
            "n_changed": deltas[f"{name}_n_changed"],
            "detectable": detectable,
            "unpaired_band": BAND_Z * math.hypot(se_a, se_b),
            "unpaired_detectable": bool(n > 1 and not within_band(a[name], se_a, b[name], se_b)),
        }
    return out


def format_report(comparison: dict[str, dict], meta: dict) -> str:
    """The printed table: every published number beside the band it sits in."""
    lines = [
        f"k={meta['k']}  n={meta['n']}  depth={meta['depth']}  "
        f"reranker={'on' if meta['reranker'] else 'off'}",
        f"  arm A  {meta['arm_a']['label']}",
        f"  arm B  {meta['arm_b']['label']}",
        "",
        f"{'metric':<20}{'mean A':>10}{'mean B':>10}{'delta':>11}"
        f"{'+/-2SE':>10}  {'verdict':<15}{'unpaired +/-2SE':>17}",
    ]
    rescued = 0
    for name, m in comparison.items():
        if m["detectable"] and not m["unpaired_detectable"]:
            rescued += 1
        lines.append(
            f"{name:<20}{m['mean_a']:>10.5f}{m['mean_b']:>10.5f}{m['delta']:>+11.5f}"
            f"{m['band']:>10.5f}  {'DETECTABLE' if m['detectable'] else 'not detectable':<15}"
            f"{m['unpaired_band']:>17.5f}"
        )
    if rescued:
        lines.append(
            f"\n{rescued} of {len(comparison)} metrics are detectable paired and would read "
            "as 'no difference' at the unpaired band this harness used to publish."
        )
    return "\n".join(lines)


def run(
    *,
    k: int = 0,
    limit: int | None = 400,
    arm: dict[str, float] | None = None,
    base: dict[str, float] | None = None,
    use_reranker: bool = False,
    processed_dir: Path = DATA_PROCESSED,
    artifacts_dir: Path = ARTIFACTS,
    out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run both arms over one seed count's challenges and report the pairing."""
    arm = arm or {}
    base = base or {}
    if arm == base:
        raise ValueError("both arms have the same configuration; there is nothing to compare")

    catalog = Catalog.load(processed_dir, artifacts_dir)
    _, challenges = load_splits(processed_dir)
    if k not in challenges:
        raise ValueError(f"seed count {k} is not in the split; have {sorted(challenges)}")
    items = challenges[k][:limit] if limit else challenges[k]
    if not items:
        raise ValueError(f"seed count {k} has no challenges")

    reranker = None
    rr_path = artifacts_dir / "reranker.pkl"
    if use_reranker and rr_path.exists():
        from ..models.reranker import Reranker

        reranker = Reranker.load(rr_path)

    cfg_a = apply_overrides(DEFAULT, base)
    cfg_b = apply_overrides(DEFAULT, arm)
    engine_a = CadenceEngine(catalog, reranker=reranker, cfg=cfg_a)
    # One planner, planned once, shared: the arms differ only in fusion.
    engine_b = CadenceEngine(catalog, planner=engine_a.planner, reranker=reranker, cfg=cfg_b)

    intents = plan_intents(engine_a, items)
    artist_ids = catalog.artist_ids
    acc_a = score_arm(engine_a, items, intents, artist_ids, use_reranker=use_reranker)
    acc_b = score_arm(engine_b, items, intents, artist_ids, use_reranker=use_reranker)

    comparison = compare(acc_a, acc_b)
    meta = {
        "k": k,
        "limit": limit,
        "n": len(items),
        "depth": DEPTH,
        "reranker": reranker is not None,
        "band_z": BAND_Z,
        "arm_a": {
            "label": label(base),
            "overrides": base,
            "rrf_k": cfg_a.retrieval.rrf_k,
            "channel_weights": dict(cfg_a.retrieval.channel_weights),
        },
        "arm_b": {
            "label": label(arm),
            "overrides": arm,
            "rrf_k": cfg_b.retrieval.rrf_k,
            "channel_weights": dict(cfg_b.retrieval.channel_weights),
        },
        "build": catalog.meta["build"],
        "train": catalog.meta["train"],
    }
    report = {"meta": meta, "metrics": comparison}

    out_path = out_path or artifacts_dir / "eval_ab.json"
    out_path.write_text(json.dumps(report, indent=2))
    if verbose:
        print(format_report(comparison, meta))
        print(f"\nwrote {out_path}")
    return report
