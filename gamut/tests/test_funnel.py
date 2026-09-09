"""Gamut's funnel is only honest if its depths are Cadence's depths.

The published error was not a bad measurement, it was a bad label: `depth` said
100, `fused_candidates` said 1500, and the audit reported the top 100 of a
1500-deep pool as "reached by retrieval". Nothing was going to notice, because
both numbers were internally consistent -- each file was right about itself.

So these pin the joins rather than the arithmetic: gamut's depths against the
Cadence values they are supposed to mean.
"""

import inspect

from cadence.assemble.select import select
from cadence.config import RetrievalConfig

from gamut.audit import funnel_stages
from gamut.config import RETRIEVE_DEPTH, SELECT_POOL, AuditConfig


def test_audit_depth_is_cadences_fused_pool_depth():
    # The bug, stated as an assertion. A copy of this number is what drifted.
    assert AuditConfig().depth == RetrievalConfig().fused_candidates
    assert RetrievalConfig().fused_candidates == RETRIEVE_DEPTH


def test_select_pool_matches_the_prefix_select_actually_scores():
    # `pool_size` is a keyword default rather than a config field, so it cannot
    # be derived without importing Catalog into a deliberately cheap module --
    # it has to be pinned, or the middle funnel stage silently becomes fiction
    # the day Cadence changes it.
    assert inspect.signature(select).parameters["pool_size"].default == SELECT_POOL


def test_funnel_stages_are_the_real_depths_deepest_first():
    stages = funnel_stages(AuditConfig())
    assert [name for name, _ in stages] == ["retrieved", "read_by_select", "shown"]
    depths = [d for _, d in stages]
    assert depths == [RetrievalConfig().fused_candidates, SELECT_POOL, AuditConfig().cut]
    assert depths == sorted(depths, reverse=True), "a later stage cannot read more"


def test_no_stage_claims_to_read_deeper_than_was_collected():
    # A shallow run (`gamut collect --depth 50`) must not report a 500-deep
    # select stage over a 50-deep cache; every stage clamps to what exists.
    assert dict(funnel_stages(AuditConfig(depth=50)))["read_by_select"] == 50


def test_the_shown_stage_clamps_too():
    # `cut` is independent of `depth`, so an unclamped `shown` row would be
    # measured over fewer columns than the depth printed beside it -- and reach
    # would rise down the funnel, which is the shape of the original defect.
    depths = [d for _, d in funnel_stages(AuditConfig(depth=10, cut=20))]
    assert depths == [10, 10, 10]
    assert depths == sorted(depths, reverse=True)
