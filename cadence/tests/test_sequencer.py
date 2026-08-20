import numpy as np
import pytest

from cadence.assemble.sequencer import (
    camelot,
    energy_curve_targets,
    key_distance,
    tempo_distance,
)


@pytest.mark.parametrize(
    ("key", "mode", "expected"),
    [
        (0, 1, (8, 1)),  # C major  -> 8B
        (9, 0, (8, 0)),  # A minor  -> 8A
        (7, 1, (9, 1)),  # G major  -> 9B
        (4, 1, (12, 1)),  # E major  -> 12B
        (11, 1, (1, 1)),  # B major  -> 1B
    ],
)
def test_camelot_matches_the_standard_wheel(key, mode, expected):
    assert camelot(key, mode) == expected


def test_camelot_handles_missing_metadata():
    assert camelot(None, 1) is None
    assert camelot(0, None) is None
    assert camelot(99, 1) is None


def test_relative_major_minor_is_nearly_free():
    assert key_distance(camelot(0, 1), camelot(9, 0)) < 0.2


def test_adjacent_fifth_is_cheaper_than_a_tritone():
    fifth = key_distance(camelot(0, 1), camelot(7, 1))
    tritone = key_distance(camelot(0, 1), camelot(6, 1))
    assert fifth < tritone


def test_unknown_key_costs_a_neutral_amount():
    d = key_distance(None, camelot(0, 1))
    assert 0.0 < d < 1.0


def test_double_time_is_treated_as_close():
    assert tempo_distance(70.0, 140.0) < tempo_distance(70.0, 110.0)


def test_tempo_distance_handles_missing_values():
    assert 0.0 < tempo_distance(None, 120.0) < 1.0


def test_energy_curves_have_the_right_shape():
    build = energy_curve_targets("build", 5, 0.5, 0.25)
    assert build[0] < build[-1]
    down = energy_curve_targets("wind_down", 5, 0.5, 0.25)
    assert down[0] > down[-1]
    peak = energy_curve_targets("peak_mid", 5, 0.5, 0.25)
    assert peak[2] == peak.max()
    steady = energy_curve_targets("steady", 5, 0.5, 0.25)
    assert np.allclose(steady, steady[0])


def test_energy_curve_stays_in_unit_range():
    out = energy_curve_targets("build", 10, 0.95, 0.5)
    assert out.min() >= 0.0 and out.max() <= 1.0
