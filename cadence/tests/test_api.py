import pytest
from fastapi.testclient import TestClient

from tests.conftest import ARTIFACTS_READY

pytestmark = pytest.mark.skipif(not ARTIFACTS_READY, reason="artifacts not built")


@pytest.fixture(scope="module")
def client():
    from cadence.service.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_tracks"] > 0


def test_generate(client):
    r = client.post("/generate", json={"query": "chill acoustic evening", "n_tracks": 8})
    assert r.status_code == 200
    body = r.json()
    assert len(body["tracks"]) == 8
    assert body["title"]
    assert all(t["track"]["track_uri"].startswith("spotify:track:") for t in body["tracks"])


def test_generate_validates_input(client):
    assert client.post("/generate", json={"query": ""}).status_code == 422
    assert client.post("/generate", json={"query": "x", "n_tracks": 0}).status_code == 422


def test_explain_returns_the_retrieval_trace(client):
    r = client.post("/explain", json={"query": "90s workout bangers"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"]["themes"]
    assert body["channel_sizes"]
    assert "retrieve_total" in body["timings_ms"]


def test_track_lookup(client):
    r = client.get("/tracks/0")
    assert r.status_code == 200
    assert "track" in r.json()


def test_unknown_track_is_404(client):
    assert client.get("/tracks/99999999").status_code == 404


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    assert "build" in r.json()
