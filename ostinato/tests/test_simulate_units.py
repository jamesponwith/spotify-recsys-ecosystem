"""Hermetic tests for the simulation's mechanics."""

from __future__ import annotations

import numpy as np

from ostinato.config import SimConfig
from ostinato.simulate import _accept


def test_acceptance_is_position_biased():
    """The whole premise: what sits at the top is kept more often.

    If this were flat, a ranking bias could never become a data bias and the
    simulation would have nothing to show.
    """
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    ranked = np.arange(20)
    kept_pos = np.zeros(20)
    for _ in range(4000):
        for t in _accept(rng, ranked, cfg):
            kept_pos[t] += 1
    first_half, second_half = kept_pos[:10].sum(), kept_pos[10:].sum()
    assert first_half > second_half * 1.5


def test_acceptance_returns_a_subset_in_order():
    cfg = SimConfig()
    ranked = np.array([7, 3, 9, 1])
    kept = _accept(np.random.default_rng(1), ranked, cfg)
    assert set(kept.tolist()) <= set(ranked.tolist())
    order = {t: i for i, t in enumerate(ranked.tolist())}
    assert [order[t] for t in kept.tolist()] == sorted(order[t] for t in kept.tolist())


def test_zero_base_rate_accepts_nothing():
    cfg = SimConfig(accept_base=0.0)
    assert _accept(np.random.default_rng(2), np.arange(10), cfg).size == 0


def test_acceptance_is_reproducible_from_the_seed():
    cfg = SimConfig()
    a = _accept(np.random.default_rng(5), np.arange(20), cfg)
    b = _accept(np.random.default_rng(5), np.arange(20), cfg)
    assert a.tolist() == b.tolist()
