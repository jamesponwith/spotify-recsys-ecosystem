"""The orchestrator: free text in, sequenced playlist out.

    query
      -> plan            (LLM or rules)      -> PlaylistIntent
      -> mask            (hard constraints)  -> eligible tracks
      -> channels        (CF / tag / lexical / audio / popularity)
      -> fusion          (reciprocal rank)   -> ~1500 candidates
      -> rerank          (learned, optional) -> reordered candidates
      -> select          (MMR + artist caps + duration)
      -> sequence        (energy arc + harmonic mixing)
      -> explain         (grounded reasons + validated copy)

``retrieve`` is deliberately public and separate from ``generate``: the
evaluation harness and the reranker's training loop both need candidates
without the assembly stages, and duplicating that path is how offline metrics
drift away from online behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .assemble.select import select
from .assemble.sequencer import sequence
from .catalog import Catalog
from .config import DEFAULT, Config
from .explain import describe_arc, track_reasons, validate_copy
from .models.train import AUDIO_FEATURE_COLS
from .planner.base import Planner, _template_copy, get_planner
from .retrieval.channels import (
    audio_channel,
    build_mask,
    collaborative_channel,
    cooccurrence_channel,
    lexical_channel,
    popularity_channel,
    seed_indices_from_intent,
    sparse_tag_channel,
    tag_channel,
)
from .retrieval.fusion import FusedCandidates, reciprocal_rank_fusion
from .types import (
    GeneratedPlaylist,
    PlaylistIntent,
    PlaylistStats,
    PlaylistTrack,
)


@dataclass
class RetrievalTrace:
    """Everything the retrieval stage learned, kept for eval and debugging."""

    candidates: FusedCandidates
    mask_applied: dict[str, int]
    seed_indices: np.ndarray
    seed_detail: dict
    tag_cols: list[int]
    timings_ms: dict[str, float]
    channel_sizes: dict[str, int]


class CadenceEngine:
    def __init__(
        self,
        catalog: Catalog,
        planner: Planner | None = None,
        reranker=None,
        cfg: Config = DEFAULT,
    ) -> None:
        self.catalog = catalog
        self.cfg = cfg
        self.planner = planner or get_planner()
        self.reranker = reranker

    # ---- derived catalog statistics --------------------------------------
    @cached_property
    def known_tags(self) -> set[str]:
        return set(self.catalog.tag_vocab)

    @cached_property
    def head_threshold(self) -> float:
        """Popularity cut separating the 'head' from the long tail (80th pct)."""
        return float(np.percentile(self.catalog.col("n_playlists"), 80))

    # ---- retrieval --------------------------------------------------------
    def retrieve(
        self,
        intent: PlaylistIntent,
        *,
        extra_seed_indices: np.ndarray | None = None,
        exclude: np.ndarray | None = None,
        top_n: int | None = None,
        channels: set[str] | None = None,
    ) -> RetrievalTrace:
        """``channels`` restricts which sources run, which is how the ablation
        study isolates each one's contribution."""
        cat = self.catalog
        rcfg = self.cfg.retrieval
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        mask, applied = build_mask(cat, intent)
        if exclude is not None and len(exclude):
            mask = mask.copy()
            mask[np.asarray(exclude, dtype=np.int64)] = False
        timings["mask"] = (time.perf_counter() - t0) * 1000

        t = time.perf_counter()
        seeds, weights, seed_detail = seed_indices_from_intent(cat, intent)
        if extra_seed_indices is not None and len(extra_seed_indices):
            extra = np.asarray(extra_seed_indices, dtype=np.int64)
            seeds = np.concatenate([seeds, extra])
            weights = np.concatenate([weights, np.ones(extra.size, dtype=np.float32)])
        tag_cols = cat.resolve_tags([*intent.themes, *intent.genres, *intent.eras])
        neg_cols = cat.resolve_tags(intent.avoid_themes)
        timings["resolve"] = (time.perf_counter() - t) * 1000

        k = rcfg.candidates_per_channel
        results = []
        sizes: dict[str, int] = {}

        t = time.perf_counter()
        results.append(collaborative_channel(cat, seeds, k, mask, weights))
        timings["ch_collaborative"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results.append(cooccurrence_channel(cat, seeds, k, mask))
        timings["ch_cooccurrence"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results.append(tag_channel(cat, tag_cols, k, mask, neg_cols))
        results.append(sparse_tag_channel(cat, tag_cols, k, mask))
        timings["ch_tag"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results.append(lexical_channel(cat, intent.query_text(), k, mask))
        timings["ch_lexical"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        results.append(audio_channel(cat, intent.audio, k, mask))
        timings["ch_audio"] = (time.perf_counter() - t) * 1000

        # Popularity only backstops a thin result set; it is never allowed to
        # dominate a query that other channels answered well.
        substantive = sum(len(r) for r in results)
        if substantive < rcfg.fused_candidates:
            results.append(popularity_channel(cat, k, mask))

        if channels is not None:
            results = [r for r in results if r.name in channels]
        for r in results:
            sizes[r.name] = len(r)

        t = time.perf_counter()
        fused = reciprocal_rank_fusion(
            results,
            rcfg.channel_weights,
            k=rcfg.rrf_k,
            top_n=top_n or rcfg.fused_candidates,
        )
        timings["fusion"] = (time.perf_counter() - t) * 1000
        timings["retrieve_total"] = (time.perf_counter() - t0) * 1000

        return RetrievalTrace(
            candidates=fused,
            mask_applied=applied,
            seed_indices=seeds,
            seed_detail=seed_detail,
            tag_cols=tag_cols,
            timings_ms=timings,
            channel_sizes=sizes,
        )

    # ---- full generation ---------------------------------------------------
    def generate(
        self,
        query: str,
        *,
        n_tracks: int | None = None,
        intent: PlaylistIntent | None = None,
        write_copy: bool = True,
    ) -> GeneratedPlaylist:
        cat = self.catalog
        acfg = self.cfg.assembly
        warnings: list[str] = []
        timings: dict[str, float] = {}

        t = time.perf_counter()
        if intent is None:
            plan = self.planner.plan(query, self.known_tags)
            intent = plan.intent
            warnings.extend(plan.warnings)
            timings["plan"] = plan.latency_ms
            provider = plan.provider
        else:
            timings["plan"] = 0.0
            provider = "supplied"

        trace = self.retrieve(intent)
        timings.update(trace.timings_ms)
        cand = trace.candidates

        if len(cand) == 0:
            warnings.append("no candidates matched the constraints")
            return GeneratedPlaylist(
                title="No results",
                description="No tracks in the catalog satisfied the request.",
                query=query,
                intent=intent,
                tracks=[],
                stats=PlaylistStats(
                    n_tracks=0, total_duration_s=0.0, n_artists=0, explicit_count=0
                ),
                constraint_report={},
                timings_ms=timings,
                warnings=warnings,
            )

        scores = cand.scores
        if self.reranker is not None:
            t = time.perf_counter()
            try:
                scores = self.reranker.score(cat, intent, trace)
                order = np.argsort(-scores, kind="stable")
                cand = FusedCandidates(
                    indices=cand.indices[order],
                    scores=scores[order],
                    channel_ranks={k: v[order] for k, v in cand.channel_ranks.items()},
                    channel_scores={k: v[order] for k, v in cand.channel_scores.items()},
                    channels_present=cand.channels_present,
                    channel_depths=cand.channel_depths,
                )
                scores = cand.scores
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"reranker failed ({type(exc).__name__}); used fusion order")
            timings["rerank"] = (time.perf_counter() - t) * 1000

        # ---- how many tracks, and how long -------------------------------
        target_n = (
            n_tracks
            if n_tracks is not None
            else (intent.constraints.track_count or acfg.default_length)
        )
        # "8 songs, about 30 minutes" states both. An exact count is the more
        # specific request, so it binds and the duration becomes a preference;
        # otherwise duration drives how many tracks are chosen.
        explicit_count = n_tracks is not None or intent.constraints.track_count is not None
        target_duration = (
            intent.constraints.target_duration_minutes * 60.0
            if intent.constraints.target_duration_minutes and not explicit_count
            else None
        )
        duration_is_binding = target_duration is not None
        max_per_artist = intent.constraints.max_per_artist or acfg.max_tracks_per_artist

        t = time.perf_counter()
        affinity = self._audio_affinity(intent, cand.indices)
        picked = select(
            cat,
            cand.indices,
            scores,
            n_tracks=target_n,
            max_per_artist=max_per_artist,
            mmr_lambda=acfg.mmr_lambda,
            target_duration_s=target_duration,
            affinity=affinity,
            affinity_weight=acfg.audio_affinity_weight if affinity is not None else 0.0,
        )
        timings["select"] = (time.perf_counter() - t) * 1000

        if len(picked.indices) == 0:
            warnings.append("constraints eliminated every candidate")

        t = time.perf_counter()
        seq = sequence(
            cat,
            picked.indices,
            curve=intent.energy_curve,
            beam_width=acfg.beam_width,
            w_tempo=acfg.w_tempo,
            w_key=acfg.w_key,
            w_energy=acfg.w_energy_curve,
            w_artist=acfg.w_artist_adjacent,
        )
        timings["sequence"] = (time.perf_counter() - t) * 1000
        ordered = picked.indices[seq.order]
        ordered_scores = picked.scores[seq.order]

        # ---- explanations --------------------------------------------------
        rank_lookup = {
            name: dict(zip(cand.indices.tolist(), vals.tolist(), strict=True))
            for name, vals in cand.channel_ranks.items()
        }
        tracks: list[PlaylistTrack] = []
        for pos, (idx, sc) in enumerate(zip(ordered, ordered_scores, strict=True)):
            ranks = {
                name: float(v)
                for name, lut in rank_lookup.items()
                if (v := lut.get(int(idx))) is not None
            }
            tracks.append(
                PlaylistTrack(
                    position=pos + 1,
                    track=cat.track(int(idx)),
                    score=float(sc),
                    reasons=track_reasons(
                        cat, intent, int(idx), channel_ranks=ranks, intent_tag_cols=trace.tag_cols
                    ),
                    transition_note=(
                        seq.transition_notes[pos - 1]
                        if 0 < pos <= len(seq.transition_notes)
                        else None
                    ),
                )
            )

        # ---- copy, validated against the real tracklist ---------------------
        t = time.perf_counter()
        title, description = self._copy(query, intent, tracks, ordered, warnings, write_copy)
        timings["copy"] = (time.perf_counter() - t) * 1000

        stats = self._stats(ordered)
        report = self._constraint_report(
            intent, ordered, stats, max_per_artist, duration_is_binding=duration_is_binding
        )
        timings["total"] = sum(
            v
            for k, v in timings.items()
            if k in ("plan", "retrieve_total", "select", "sequence", "copy")
        )
        if intent.constraints.exclude_explicit:
            pct = trace.mask_applied.get("explicit_flag_coverage_pct")
            if pct is not None and pct < 90:
                warnings.append(
                    f"explicit flags are known for only {pct}% of the catalog; "
                    "known-explicit tracks were removed, unlabelled ones could not be checked"
                )
        if provider != "supplied":
            warnings.append(f"planner: {provider}")

        return GeneratedPlaylist(
            title=title,
            description=description,
            query=query,
            intent=intent,
            tracks=tracks,
            stats=stats,
            constraint_report=report,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            warnings=warnings,
        )

    # ---- helpers ----------------------------------------------------------
    def _copy(self, query, intent, tracks, ordered, warnings, write_copy):
        fallback = _template_copy(intent, len(tracks))
        if not write_copy or not tracks:
            return fallback.title, fallback.description
        lines = [f"{t.track.name} — {t.track.artist}" for t in tracks]
        copy = self.planner.write_copy(query, intent, lines)
        check = validate_copy(self.catalog, f"{copy.title} {copy.description}", ordered)
        if not check.ok:
            warnings.append(check.as_warning() or "")
            return fallback.title, fallback.description
        return copy.title, copy.description

    def _audio_affinity(self, intent: PlaylistIntent, indices: np.ndarray) -> np.ndarray | None:
        """Closeness of each candidate to the listener's stated audio targets.

        Returns None when nothing was stated, so the term contributes nothing
        rather than nudging results toward the catalog mean.
        """
        active = intent.audio.active()
        if not active or len(indices) == 0:
            return None
        cat = self.catalog
        dims = [AUDIO_FEATURE_COLS.index(k) for k in active if k in AUDIO_FEATURE_COLS]
        if not dims:
            return None
        values = np.array([active[AUDIO_FEATURE_COLS[d]] for d in dims], dtype=np.float32)
        z_target = (values - cat.audio_mu[dims]) / cat.audio_sigma[dims]
        idx = np.asarray(indices, dtype=np.int64)
        dist = np.linalg.norm(cat.audio_z[idx][:, dims] - z_target[None, :], axis=1)
        dist /= np.sqrt(len(dims))
        aff = np.exp(-dist).astype(np.float32)
        # Tracks with no measurement get the median affinity: neither rewarded
        # nor punished for missing metadata.
        unknown = ~cat.audio_valid[idx]
        if unknown.any():
            aff[unknown] = float(np.median(aff[~unknown])) if (~unknown).any() else 0.5
        return aff

    def _stats(self, indices: np.ndarray) -> PlaylistStats:
        cat = self.catalog
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return PlaylistStats(n_tracks=0, total_duration_s=0.0, n_artists=0, explicit_count=0)

        def mean_known(col: str) -> float | None:
            v = cat.col(col)[idx]
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else None

        vectors = cat.collab.vectors[idx]
        if idx.size > 1:
            sims = vectors @ vectors.T
            iu = np.triu_indices(idx.size, k=1)
            ild = float(1.0 - sims[iu].mean())
        else:
            ild = None

        pop = cat.col("n_playlists")[idx]
        return PlaylistStats(
            n_tracks=int(idx.size),
            total_duration_s=float(cat.col("duration_ms")[idx].sum() / 1000.0),
            mean_energy=mean_known("energy"),
            mean_valence=mean_known("valence"),
            mean_tempo=mean_known("tempo"),
            n_artists=int(len(set(cat.artist_ids[idx].tolist()))),
            explicit_count=int(cat.col("explicit")[idx].sum()),
            intra_list_distance=ild,
            long_tail_share=float((pop < self.head_threshold).mean()),
        )

    def _constraint_report(
        self, intent, indices, stats, max_per_artist, *, duration_is_binding: bool = True
    ) -> dict[str, bool]:
        cat = self.catalog
        idx = np.asarray(indices, dtype=np.int64)
        report: dict[str, bool] = {}
        c = intent.constraints

        if c.exclude_explicit:
            # "no known-explicit track survived" is the claim we can actually back.
            report["no_known_explicit"] = bool(stats.explicit_count == 0)
        if c.track_count is not None:
            report["track_count"] = bool(stats.n_tracks == c.track_count)
        if c.target_duration_minutes is not None and duration_is_binding:
            target = c.target_duration_minutes * 60
            report["duration_within_10pct"] = bool(
                abs(stats.total_duration_s - target) <= 0.10 * target
            )
        if idx.size:
            counts = np.bincount(cat.artist_ids[idx])
            report["artist_cap"] = bool(counts.max() <= max_per_artist)
        if intent.tempo.is_set():
            bpm = cat.col("tempo")[idx]
            known = bpm[np.isfinite(bpm)]
            ok = np.ones(known.size, dtype=bool)
            if intent.tempo.min_bpm is not None:
                ok &= known >= intent.tempo.min_bpm
            if intent.tempo.max_bpm is not None:
                ok &= known <= intent.tempo.max_bpm
            report["tempo_range"] = bool(ok.all()) if known.size else True
        return report

    def arc_summary(self, playlist: GeneratedPlaylist) -> str:
        return describe_arc(self.catalog, np.array([t.track.index for t in playlist.tracks]))
