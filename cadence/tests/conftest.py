"""Shared fixtures.

Unit tests run on synthetic data so they are fast and hermetic. Integration
tests need the built artifacts and skip cleanly when they are absent, so a
fresh clone can still run the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadence.config import ARTIFACTS, DATA_PROCESSED  # noqa: E402

ARTIFACTS_READY = (ARTIFACTS / "spaces.npz").exists() and (
    DATA_PROCESSED / "tracks.parquet"
).exists()

requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS_READY, reason="built catalog/artifacts not present; run `cadence pipeline`"
)


@pytest.fixture(scope="session")
def synthetic_interactions() -> sparse.csr_matrix:
    """300 playlists over 90 tracks in 3 disjoint taste clusters."""
    rng = np.random.default_rng(0)
    rows, cols = [], []
    for p in range(300):
        c = p % 3
        items = rng.choice(np.arange(c * 30, (c + 1) * 30), size=8, replace=False)
        rows.extend([p] * len(items))
        cols.extend(items.tolist())
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(300, 90))


@pytest.fixture(scope="session")
def skewed_interactions() -> sparse.csr_matrix:
    """Playlists with a power-law popularity skew, so PMI genuinely varies.

    The clean cluster fixture has uniform PMI across every observed pair, which
    makes the SPPMI shift an all-or-nothing switch; this one exercises the
    graded behaviour the shift is actually for.
    """
    rng = np.random.default_rng(1)
    n_items = 200
    weights = 1.0 / np.arange(1, n_items + 1) ** 1.1
    weights /= weights.sum()
    rows, cols = [], []
    for p in range(600):
        size = int(rng.integers(5, 15))
        items = rng.choice(n_items, size=size, replace=False, p=weights)
        rows.extend([p] * size)
        cols.extend(items.tolist())
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(600, n_items)
    )


@pytest.fixture(scope="session")
def catalog():
    from cadence.catalog import Catalog

    if not ARTIFACTS_READY:
        pytest.skip("artifacts not built")
    return Catalog.load()


@pytest.fixture(scope="session")
def engine(catalog):
    from cadence.engine import CadenceEngine

    return CadenceEngine(catalog)
