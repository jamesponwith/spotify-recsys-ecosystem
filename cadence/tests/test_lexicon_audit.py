"""Unit tests for the lexicon calibration audit, on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cadence.eval.lexicon_audit import audit, main
from cadence.planner.lexicon import MoodEntry


@pytest.fixture
def world():
    #  track:  energy  valence  has_audio   tags: sleep  party
    #    0      0.20    0.5     yes                1      0
    #    1      0.60    0.5     yes                1      0
    #    2      0.90    0.5     yes                0      1
    #    3      0.99    0.5     no                 1      1   <- excluded everywhere
    frame = pd.DataFrame(
        {
            "energy": [0.20, 0.60, 0.90, 0.99],
            "valence": [0.5, 0.5, 0.5, 0.5],
            "has_audio": [True, True, True, False],
        }
    )
    tags = sparse.csr_matrix(np.array([[1, 0], [3, 0], [0, 2], [1, 1]], dtype=np.float32))
    return frame, tags, ["sleep", "party"]


def test_means_are_over_filed_tracks_with_audio_only(world):
    frame, tags, vocab = world
    report = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.1})})
    (pair,) = report["pairs"]
    assert pair["n_tracks"] == 2  # track 3 is filed but has no audio
    assert pair["folksonomy_mean"] == pytest.approx(0.40)
    assert pair["catalog_mean"] == pytest.approx((0.20 + 0.60 + 0.90) / 3, abs=1e-4)


def test_a_target_further_than_the_catalog_mean_is_flagged(world):
    frame, tags, vocab = world
    # humans 0.40, catalog 0.567 (gap 0.167): 0.10 is worse, 0.30 is better.
    worse = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.10})})
    better = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.30})})
    assert worse["n_target_worse_than_catalog"] == 1
    assert better["n_target_worse_than_catalog"] == 0


def test_direction_and_overshoot_are_separate_questions(world):
    frame, tags, vocab = world
    # humans 0.40 sit below the catalog's 0.567. 0.10 goes the same way and
    # past them; 0.50 goes the same way but not far enough; 0.80 goes the
    # wrong way entirely.
    lexicon = {"sleep": MoodEntry({"energy": 0.10})}
    past = audit(frame, tags, vocab, lexicon)["pairs"][0]
    short = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.50})})["pairs"][0]
    wrong = audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.80})})["pairs"][0]
    assert (past["direction_right"], past["overshoots"]) == (True, True)
    assert (short["direction_right"], short["overshoots"]) == (True, False)
    assert (wrong["direction_right"], wrong["overshoots"]) == (False, False)


def test_a_word_that_is_not_a_tag_is_listed_not_dropped_silently(world):
    frame, tags, vocab = world
    lexicon = {"sleep": MoodEntry({"energy": 0.1}), "aggressive": MoodEntry({"energy": 0.9})}
    report = audit(frame, tags, vocab, lexicon)
    assert report["n_pairs"] == 1
    assert report["words_not_tags"] == ["aggressive"]


def test_every_asserted_dimension_is_a_pair_and_a_bare_theme_is_not(world):
    frame, tags, vocab = world
    lexicon = {
        "party": MoodEntry({"energy": 0.8, "valence": 0.7}),
        "sleep": MoodEntry({}, themes=("sleep",)),
    }
    report = audit(frame, tags, vocab, lexicon)
    assert report["n_pairs"] == 2
    assert report["n_words_audited"] == 1
    assert report["words_not_tags"] == []
    assert report["by_dimension"]["valence"] == {"n_pairs": 1, "n_target_worse_than_catalog": 1}


def test_pairs_are_ordered_most_damning_first(world):
    frame, tags, vocab = world
    lexicon = {"sleep": MoodEntry({"energy": 0.35, "valence": 0.0})}
    report = audit(frame, tags, vocab, lexicon)
    assert [p["dimension"] for p in report["pairs"]] == ["valence", "energy"]


def test_mismatched_inputs_raise_instead_of_misattributing(world):
    frame, tags, vocab = world
    with pytest.raises(ValueError):
        audit(frame, tags, vocab[:1])
    with pytest.raises(ValueError):
        audit(frame.iloc[:3], tags, vocab)


def test_a_dimension_with_no_audio_raises_instead_of_writing_nan(world):
    frame, tags, vocab = world
    frame = frame.assign(has_audio=False)
    with pytest.raises(ValueError):
        audit(frame, tags, vocab, {"sleep": MoodEntry({"energy": 0.1})})


def test_main_reads_the_three_files_and_writes_valid_json(world, tmp_path):
    frame, tags, vocab = world
    frame.to_parquet(tmp_path / "tracks.parquet", index=False)
    sparse.save_npz(tmp_path / "tags.npz", tags)
    (tmp_path / "tag_vocab.json").write_text(json.dumps(vocab))
    out = tmp_path / "artifacts" / "lexicon_calibration.json"
    report = main(processed_dir=tmp_path, out=out, verbose=False)
    on_disk = json.loads(out.read_text())
    assert on_disk == report
    assert on_disk["n_pairs"] > 0
    assert on_disk["seconds"] < 60
    assert {"target", "folksonomy_mean", "catalog_mean", "n_tracks"} <= on_disk["pairs"][0].keys()
