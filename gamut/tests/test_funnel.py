"""Gamut's funnel is only honest if its depths are Cadence's depths.

The published error was not a bad measurement, it was a bad label: `depth` said
100, `fused_candidates` said 1500, and the audit reported the top 100 of a
1500-deep pool as "reached by retrieval". Nothing was going to notice, because
both numbers were internally consistent -- each file was right about itself.

So these pin the joins rather than the arithmetic: gamut's depths against the
Cadence values they are supposed to mean, and the report rows against the stage
names they are published under.
"""

import inspect
import json

import numpy as np
import pytest
from cadence.assemble.select import select
from cadence.config import RetrievalConfig

from gamut.audit import funnel_stages
from gamut.collect import ABSENT, Collected
from gamut.config import RETRIEVE_DEPTH, SELECT_POOL, AuditConfig
from gamut.exposure import CatalogFacts, measure


def test_audit_depth_is_cadences_fused_pool_depth():
    # The bug, stated as an assertion. A copy of this number is what drifted.
    assert AuditConfig().depth == RetrievalConfig().fused_candidates
    assert RetrievalConfig().fused_candidates == RETRIEVE_DEPTH


def test_select_pool_matches_the_prefix_select_actually_scores():
    # `pool_size` is a keyword default rather than a config field, so it cannot
    # be derived -- it has to be pinned, or the middle funnel stage silently
    # becomes fiction the day Cadence changes it.
    assert inspect.signature(select).parameters["pool_size"].default == SELECT_POOL


def test_funnel_stages_are_nested_prefixes_deepest_first():
    stages = funnel_stages(AuditConfig())
    assert [name for name, _ in stages] == ["retrieved", "read_by_select", "shown"]
    depths = [d for _, d in stages]
    assert depths == sorted(depths, reverse=True), "a later stage cannot read more"


def test_funnel_stage_depths_are_the_real_ones():
    stages = dict(funnel_stages(AuditConfig()))
    assert stages["retrieved"] == RetrievalConfig().fused_candidates
    assert stages["read_by_select"] == SELECT_POOL
    assert stages["shown"] == AuditConfig().cut


def test_funnel_never_claims_to_read_deeper_than_it_collected():
    # A shallow run (`gamut collect --depth 50`) must not report a 500-deep
    # select stage over a 50-deep cache; the stage clamps to what exists.
    stages = dict(funnel_stages(AuditConfig(depth=50)))
    assert stages["read_by_select"] == 50


def test_reach_is_monotone_down_the_funnel():
    # The stages are prefixes of one ranked list, so reach can only shrink. The
    # old report broke this by construction: its "pool" row was measured at 100
    # and published as the 1500-deep stage above the 20-deep one.
    rng = np.random.default_rng(0)
    n_tracks = 200
    facts = CatalogFacts(
        n_tracks=n_tracks,
        artists=np.arange(n_tracks, dtype=np.int32) // 4,
        n_artists=n_tracks // 4,
        play_counts=rng.integers(1, 50, n_tracks).astype(np.float64),
        tail_mask=np.zeros(n_tracks, dtype=bool),
        head_mask=np.zeros(n_tracks, dtype=bool),
        tail_share_of_catalog=0.5,
    )
    # Deeper than SELECT_POOL so all three stages are genuinely distinct widths.
    indices = rng.integers(0, n_tracks, (10, 600)).astype(np.int32)
    stages = funnel_stages(AuditConfig(depth=600, cut=20))
    assert len({d for _, d in stages}) == 3
    reach = [measure(indices, facts, d).track_coverage for _, d in stages]
    assert reach == sorted(reach, reverse=True)
    assert reach[0] > reach[-1], "a 600-deep pool must out-reach a 20-deep cut"


def _write_cache(path, depth: int) -> None:
    np.savez_compressed(
        path,
        indices=np.zeros((1, depth), dtype=np.int32),
        scores=np.zeros((1, depth), dtype=np.float32),
        channel_ranks=np.full((1, 1, depth), ABSENT, dtype=np.int32),
        titles=np.array(["q0"], dtype=object),
        truth=np.array([json.dumps([])], dtype=object),
        sentinel_stripped=np.True_,
    )


def test_load_refuses_a_cache_shallower_than_the_depth_being_reported(tmp_path):
    # Exactly the shipped condition: a 100-deep cache audited as a 1500-deep pool.
    path = tmp_path / "collected.npz"
    _write_cache(path, depth=100)
    with pytest.raises(ValueError, match="collected 100 deep"):
        Collected.load(path, min_depth=1500)


def test_load_accepts_a_cache_exactly_as_deep_as_reported(tmp_path):
    path = tmp_path / "collected.npz"
    _write_cache(path, depth=100)
    assert Collected.load(path, min_depth=100).indices.shape[1] == 100


def test_load_without_a_depth_claim_is_unchanged(tmp_path):
    # `demo` reads the cache without publishing a funnel, so it must not be
    # forced to re-collect 1500 deep to show one query.
    path = tmp_path / "collected.npz"
    _write_cache(path, depth=100)
    assert Collected.load(path).indices.shape[1] == 100
