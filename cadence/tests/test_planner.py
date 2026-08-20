import pytest

from cadence.planner.base import OfflinePlanner, _template_copy, get_planner
from cadence.planner.offline import parse_intent


def test_track_count_survives_intervening_adjectives():
    assert parse_intent("20 upbeat workout tracks").constraints.track_count == 20
    assert parse_intent("a 15-track indie playlist").constraints.track_count == 15
    assert parse_intent("give me 8 songs").constraints.track_count == 8


def test_duration_in_minutes_and_hours():
    assert parse_intent("about 45 minutes of music").constraints.target_duration_minutes == 45
    assert parse_intent("a 2 hour set").constraints.target_duration_minutes == 120


@pytest.mark.parametrize(
    "phrase", ["nothing explicit", "no explicit", "keep it clean", "family friendly"]
)
def test_clean_requests_set_the_explicit_filter(phrase):
    assert parse_intent(f"party music, {phrase}").constraints.exclude_explicit


def test_bpm_range_and_bounds():
    t = parse_intent("tracks between 130 and 150 bpm").tempo
    assert (t.min_bpm, t.max_bpm) == (130, 150)
    assert parse_intent("something over 140 bpm").tempo.min_bpm == 140
    assert parse_intent("nothing under 90 bpm").tempo.max_bpm == 90


def test_single_bpm_becomes_a_window():
    t = parse_intent("around 120 bpm")
    assert t.tempo.min_bpm < 120 < t.tempo.max_bpm


def test_seed_and_avoid_artists():
    i = parse_intent("songs like Bon Iver, no Kanye West")
    assert "bon iver" in [s.lower() for s in i.seed_artists]
    assert "kanye west" in [a.lower() for a in i.avoid_artists]


def test_avoid_does_not_capture_mood_words_as_artists():
    i = parse_intent("upbeat pop, no explicit lyrics")
    assert not any("explicit" in a for a in i.avoid_artists)


def test_moods_map_to_audio_targets():
    audio = parse_intent("high energy party bangers").audio.active()
    assert audio["energy"] > 0.7


def test_calm_requests_get_low_energy():
    assert parse_intent("calm sleep music").audio.active()["energy"] < 0.35


def test_energy_curve_detection():
    assert parse_intent("workout mix that builds up").energy_curve == "build"
    assert parse_intent("music to wind down to").energy_curve == "wind_down"


def test_era_extraction():
    assert parse_intent("90s throwbacks").eras == ["1990s"]


def test_bigram_themes_suppress_their_own_unigrams():
    i = parse_intent("hip hop party", known_tags={"hip", "hop", "hip hop", "party"})
    assert "hip hop" in i.themes
    assert "hip" not in i.themes and "hop" not in i.themes


def test_decade_variants_are_not_duplicated_as_themes():
    i = parse_intent("90s jams", known_tags={"90s", "1990s", "jams"})
    assert "90s" not in i.themes


def test_parsing_is_deterministic():
    q = "chill acoustic rainy day, 12 songs, nothing explicit"
    assert parse_intent(q).model_dump() == parse_intent(q).model_dump()


def test_empty_query_does_not_crash():
    assert parse_intent("").themes == []


def test_offline_planner_roundtrip():
    p = OfflinePlanner()
    r = p.plan("chill 90s r&b")
    assert r.provider == "offline"
    copy = p.write_copy("q", r.intent, ["a", "b"])
    assert copy.title and copy.description


def test_template_copy_does_not_repeat_the_decade():
    intent = parse_intent("90s party", known_tags={"party", "90s", "1990s"})
    title = _template_copy(intent, 10).title
    assert title.lower().count("90s") <= 1


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        get_planner("gpt-9")
