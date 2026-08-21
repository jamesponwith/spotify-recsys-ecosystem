"""The exact-tag channel must reward breadth of concept match, not depth in one."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cadence.retrieval.channels import sparse_tag_channel


class _Cat:
    """Minimal stand-in: the channel only touches tag_matrix_csc."""

    def __init__(self, matrix):
        self.tag_matrix_csc = matrix.tocsc()


@pytest.fixture
def catalog():
    #            road_trip  1990s
    # track 0        500      0     <- one popular concept only
    # track 1         50     50     <- matches both
    # track 2          0      0
    return _Cat(sparse.csr_matrix(np.array([[500.0, 0.0], [50.0, 50.0], [0.0, 0.0]])))


def test_matching_both_concepts_beats_matching_one_popularly(catalog):
    """The '90s alternative rock' bug: summing raw counts made this an ANY-match,
    so a track riding one popular tag outranked one that satisfied the whole
    request."""
    idx = sparse_tag_channel(catalog, [0, 1], k=3).indices
    assert idx[0] == 1, "track matching both concepts must rank first"


def test_a_track_matching_nothing_is_excluded(catalog):
    result = sparse_tag_channel(catalog, [0, 1], k=3)
    assert 2 not in result.indices.tolist()


def test_single_concept_query_is_ranked_by_count(catalog):
    """With one concept there is no breadth to reward, so it must fall back to
    plain popularity within that tag."""
    result = sparse_tag_channel(catalog, [0], k=3)
    assert result.indices[0] == 0


def test_empty_tag_list_returns_empty(catalog):
    assert len(sparse_tag_channel(catalog, [], k=3)) == 0
