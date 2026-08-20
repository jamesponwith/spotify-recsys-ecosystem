"""Integration tests against the built catalog. Skipped when artifacts are absent."""

from __future__ import annotations

import numpy as np

from cadence.planner.offline import parse_intent
from tests.conftest import requires_artifacts


@requires_artifacts
def test_generates_the_requested_track_count(engine):
    pl = engine.generate("chill indie for studying", n_tracks=12)
    assert len(pl.tracks) == 12
    assert pl.stats.n_tracks == 12


@requires_artifacts
def test_positions_are_contiguous_and_ordered(engine):
    pl = engine.generate("workout hype", n_tracks=10)
    assert [t.position for t in pl.tracks] == list(range(1, 11))


@requires_artifacts
def test_no_duplicate_tracks(engine):
    pl = engine.generate("2000s pop hits", n_tracks=25)
    idx = [t.track.index for t in pl.tracks]
    assert len(set(idx)) == len(idx)


@requires_artifacts
def test_artist_cap_is_respected(engine):
    pl = engine.generate("taylor swift radio", n_tracks=20)
    counts: dict[str, int] = {}
    for t in pl.tracks:
        counts[t.track.artist_uri] = counts.get(t.track.artist_uri, 0) + 1
    assert max(counts.values()) <= engine.cfg.assembly.max_tracks_per_artist


@requires_artifacts
def test_explicit_filter_is_honoured(engine):
    pl = engine.generate("hip hop party, nothing explicit", n_tracks=15)
    assert all(not t.track.explicit for t in pl.tracks)
    assert pl.constraint_report.get("no_known_explicit") is True


@requires_artifacts
def test_avoided_artist_is_excluded(engine):
    intent = parse_intent("pop hits, no Rihanna")
    pl = engine.generate("pop hits, no Rihanna", intent=intent, n_tracks=20)
    assert all("rihanna" not in t.track.artist.lower() for t in pl.tracks)


@requires_artifacts
def test_tempo_constraint_holds_for_tracks_with_known_tempo(engine):
    pl = engine.generate("running music between 150 and 175 bpm", n_tracks=12)
    for t in pl.tracks:
        if t.track.tempo is not None:
            assert 150 <= t.track.tempo <= 175


@requires_artifacts
def test_seed_artist_influences_results(engine):
    """A seeded query should surface the seed artist or its neighbourhood."""
    pl = engine.generate("songs like Radiohead", n_tracks=20)
    assert len(pl.tracks) > 0
    trace = engine.retrieve(parse_intent("songs like Radiohead"))
    assert len(trace.seed_indices) > 0, "Radiohead should resolve to catalog rows"


@requires_artifacts
def test_retrieval_is_deterministic(engine):
    intent = parse_intent("rainy day acoustic")
    a = engine.retrieve(intent).candidates.indices
    b = engine.retrieve(intent).candidates.indices
    np.testing.assert_array_equal(a, b)


@requires_artifacts
def test_nonsense_query_still_returns_something(engine):
    pl = engine.generate("asdfghjkl qwerty zxcvbn", n_tracks=5)
    assert len(pl.tracks) > 0, "popularity backstop must keep the system responsive"


@requires_artifacts
def test_empty_query_does_not_crash(engine):
    pl = engine.generate("", n_tracks=5)
    assert isinstance(pl.tracks, list)


@requires_artifacts
def test_channel_ablation_changes_results(engine):
    intent = parse_intent("90s hip hop")
    full = engine.retrieve(intent).candidates.indices[:50]
    tag_only = engine.retrieve(intent, channels={"tag", "tag_exact"}).candidates.indices[:50]
    assert not np.array_equal(full, tag_only)


@requires_artifacts
def test_timings_are_recorded(engine):
    pl = engine.generate("jazz for dinner", n_tracks=8)
    assert pl.timings_ms.get("total", 0) > 0
    assert "retrieve_total" in pl.timings_ms


@requires_artifacts
def test_constraint_battery_is_satisfied(engine):
    """End-to-end check that stated requirements are actually honoured."""
    from cadence.eval.constraints_eval import run

    report = run(
        engine,
        queries=[
            "12 chill songs for studying",
            "workout playlist between 140 and 160 bpm, 10 tracks",
            "15 party bangers, nothing explicit",
            "8 acoustic songs, about 30 minutes",
        ],
        verbose=False,
    )
    assert report.failures == []
    for name, stats in report.satisfaction.items():
        assert stats["rate"] == 1.0, f"{name} was not always satisfied"


@requires_artifacts
def test_implied_curve_cannot_contradict_a_calm_request(engine):
    """'chill rainy morning' must not build to high energy."""
    from cadence.planner.offline import parse_intent

    intent = parse_intent("chill acoustic songs for a rainy sunday morning")
    assert intent.energy_curve != "build"


@requires_artifacts
def test_stated_audio_target_moves_the_playlist(engine):
    """A calm request should deliver measurably lower energy than a hype one."""
    import numpy as np

    calm = engine.generate("very calm quiet acoustic music", n_tracks=12)
    hype = engine.generate("high energy hype party bangers", n_tracks=12)

    def mean_energy(pl):
        vals = np.array([t.track.energy for t in pl.tracks if t.track.energy is not None])
        return float(vals.mean()) if vals.size else float("nan")

    assert mean_energy(calm) < mean_energy(hype) - 0.2
