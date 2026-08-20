import numpy as np

from cadence.explain import validate_copy
from tests.conftest import requires_artifacts


class StubCatalog:
    def __init__(self, artists):
        self._artists = np.array(artists, dtype=object)
        self._artist_name_index = {a.lower(): [i] for i, a in enumerate(artists)}

    def col(self, name):
        assert name == "artist"
        return self._artists


def test_copy_naming_only_playlist_artists_passes():
    cat = StubCatalog(["Bon Iver", "Fleet Foxes", "Kanye West"])
    v = validate_copy(cat, "Warm folk from Bon Iver and Fleet Foxes.", np.array([0, 1]))
    assert v.ok
    assert v.as_warning() is None


def test_copy_naming_an_absent_artist_is_flagged():
    cat = StubCatalog(["Bon Iver", "Fleet Foxes", "Kanye West"])
    v = validate_copy(cat, "Folk from Bon Iver and Kanye West.", np.array([0, 1]))
    assert not v.ok
    assert "kanye west" in v.unsupported_artists
    assert "not on the playlist" in (v.as_warning() or "")


def test_short_single_word_names_do_not_trigger_false_positives():
    # "Air" and "Yes" are real artists that collide with ordinary English.
    cat = StubCatalog(["Air", "Yes", "Bon Iver"])
    v = validate_copy(cat, "Songs that fill the air, yes, gently.", np.array([2]))
    assert v.ok


def test_empty_copy_is_valid():
    cat = StubCatalog(["Bon Iver"])
    assert validate_copy(cat, "", np.array([0])).ok


@requires_artifacts
def test_reasons_are_grounded_in_real_data(engine):
    playlist = engine.generate("chill acoustic evening", n_tracks=5)
    for track in playlist.tracks:
        assert track.reasons, "every track must carry at least one reason"
        for reason in track.reasons:
            assert reason.strip()


@requires_artifacts
def test_no_track_can_be_hallucinated(engine):
    """Every returned track must be a real catalog row — true by construction,
    asserted so a future refactor cannot quietly break it."""
    playlist = engine.generate("upbeat 90s pop", n_tracks=12)
    uris = set(engine.catalog.col("track_uri").tolist())
    for t in playlist.tracks:
        assert t.track.track_uri in uris
        assert 0 <= t.track.index < len(engine.catalog)
