"""The rendered tables must carry the band the harness measured.

The failure this guards against is the one that shipped: `*_se` written by the
harness, read by the renderer, and dropped on the floor, so the docs quoted four
decimals of which the last two were noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cadence.eval.metrics import BAND_Z, detection_floor, within_band

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "eval_report.json"

sys.path.insert(0, str(ROOT / "scripts"))
import render_results  # noqa: E402


def _cell(value: float, se: float, **extra: float) -> dict:
    out = {"n": 4.0}
    for m in ("r_precision", "r_precision_artist", "ndcg_100", "clicks", "recall_500"):
        out[m], out[f"{m}_se"] = value, se
    out.update(extra)
    return out


def _synthetic_report() -> dict:
    # The headline cell reproduces the published k=0 number exactly, so the
    # string the docs must show is the string this test asserts on.
    return {
        "meta": {"n_tracks": 1000, "depth": 500, "limit_per_cell": 4},
        "results": {
            "0": {
                "full_reranked": _cell(0.14291403517010784, 0.007435214370452819),
                "full_fusion": _cell(0.0709, 0.0054),
                "no_audio": _cell(0.0716, 0.0054),  # inside the band
                "no_tag": _cell(0.0191, 0.0038),  # far outside it
                "popularity": _cell(0.0404, 0.0030),
            },
            "5": {
                "full_reranked": _cell(0.2416, 0.0095),
                "full_fusion": _cell(0.1849, 0.0077),
                "no_audio": _cell(0.1850, 0.0077),
                "no_tag": _cell(0.1200, 0.0071),
                "popularity": _cell(0.0378, 0.0029),
            },
        },
    }


@pytest.fixture
def rendered(tmp_path, monkeypatch, capsys) -> str:
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(_synthetic_report()))
    monkeypatch.setattr(sys, "argv", ["render_results.py", str(path)])
    assert render_results.main() == 0
    return capsys.readouterr().out


def test_every_headline_cell_carries_its_band(rendered):
    assert "0.1429 ± 0.0149" in rendered
    # Clicks keep their two-decimal convention, band included.
    assert "0.14 ± 0.01" in rendered
    # No bare four-decimal number survives in the main tables.
    main_tables = rendered.split("### Channel ablations")[0]
    for line in main_tables.splitlines():
        if line.startswith("| ") and "System" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")][1:]
            assert all("±" in c for c in cells), line


def test_ablation_table_marks_cells_inside_their_band(rendered):
    table = rendered.split("### Channel ablations")[1]
    rows = {
        line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("| ")
    }
    audio = rows["− audio"]
    tag = rows["− folksonomy tags"]
    assert audio.count("≈") == 2, audio  # inside at every k
    assert "≈" not in tag, tag
    assert "≈" not in rows["Cadence (fusion only)"]  # the base row is never marked
    assert "2 of 4 `−` cells sit inside their own band" in rendered


def test_footer_states_the_floor(rendered):
    assert "Detection floor: 0.0149 R-precision" in rendered
    assert "k=0 `full_reranked` cell" in rendered


def test_footer_prefers_the_stamped_floor_and_its_cell(tmp_path, monkeypatch, capsys):
    # A stamped value must be attributed to the stamped cell, never to a
    # recomputed one — otherwise the footer can quote one number and name another.
    report = _synthetic_report()
    report["meta"]["detection_floor"] = 0.0200
    report["meta"]["detection_floor_basis"] = {"k": 5, "system": "full_fusion"}
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["render_results.py", str(path)])
    render_results.main()
    out = capsys.readouterr().out
    assert "Detection floor: 0.0200 R-precision" in out
    assert "k=5 `full_fusion` cell" in out


def test_within_band_uses_the_error_of_the_difference():
    # |Δ| = 0.010 is outside 2×0.004 alone but inside 2×hypot(0.004, 0.004).
    assert within_band(0.100, 0.004, 0.110, 0.004)
    assert not within_band(0.100, 0.004, 0.115, 0.004)
    assert within_band(0.1, 0.0, 0.1, 0.0)


def test_detection_floor_is_the_headline_cells_band():
    results = _synthetic_report()["results"]
    floor = detection_floor(results)
    assert floor["value"] == 0.0149
    assert (floor["k"], floor["system"], floor["z"]) == (0, "full_reranked", BAND_Z)
    # Without a reranker the fusion row is the headline.
    del results["0"]["full_reranked"]
    assert detection_floor(results)["system"] == "full_fusion"


@pytest.mark.skipif(not REPORT.exists(), reason="artifacts/eval_report.json not built")
def test_published_report_carries_its_floor():
    """The committed artifact must agree with the arithmetic that stamps it.

    The value itself is not pinned: an honest re-run moves every number, and
    the docs are regenerated from the artifact rather than asserted against it.
    """
    report = json.loads(REPORT.read_text())
    floor = detection_floor(report["results"])
    assert report["meta"]["detection_floor"] == floor.pop("value")
    assert report["meta"]["detection_floor_basis"] == floor
