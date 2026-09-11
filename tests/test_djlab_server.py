"""End-to-end Flask test-client tests: the real HTTP API, calling the real
engine (no mocking) against small synthetic audio files. These exercise
exactly the request flow the browser UI drives -- scan a folder, trigger
analysis, poll its status, then request a plan -- without needing a real
browser."""

from __future__ import annotations

import time

import soundfile as sf

from conftest import SR, click_track
from songanalysis.cache import AnalysisCache
from djlab.server import LabState, create_app


def _write_song(path, duration=3.0, bpm=120.0):
    sf.write(str(path), click_track(duration, bpm), SR)


def _make_client(tmp_path):
    state = LabState(cache=AnalysisCache(cache_dir=tmp_path / "cache"))
    app = create_app(state)
    app.testing = True
    return app.test_client(), state


def test_get_library_before_any_folder_is_loaded_returns_empty(tmp_path):
    client, _ = _make_client(tmp_path)
    resp = client.get("/api/library")
    assert resp.status_code == 200
    assert resp.get_json() == {"root": None, "songs": []}


def test_post_library_rejects_a_missing_directory(tmp_path):
    client, _ = _make_client(tmp_path)
    resp = client.post("/api/library", json={"path": str(tmp_path / "does-not-exist")})
    assert resp.status_code == 400


def test_post_library_scans_and_lists_songs(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_song(music_dir / "a.wav")
    _write_song(music_dir / "b.wav")

    client, _ = _make_client(tmp_path)
    resp = client.post("/api/library", json={"path": str(music_dir)})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["root"] == str(music_dir)
    assert {s["id"] for s in data["songs"]} == {"a.wav", "b.wav"}
    assert all(not s["analyzed"] for s in data["songs"])


def test_plan_requires_an_analyzed_starting_song(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_song(music_dir / "a.wav")

    client, _ = _make_client(tmp_path)
    client.post("/api/library", json={"path": str(music_dir)})

    resp = client.post("/api/plan", json={"starting_song_id": "a.wav", "direction": "none", "num_songs": 2})
    assert resp.status_code == 409


def test_full_workflow_scan_analyze_plan(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_song(music_dir / "a.wav", duration=4.0, bpm=120.0)
    _write_song(music_dir / "b.wav", duration=4.0, bpm=124.0)
    _write_song(music_dir / "c.wav", duration=4.0, bpm=118.0)

    client, _ = _make_client(tmp_path)

    scan = client.post("/api/library", json={"path": str(music_dir)}).get_json()
    assert len(scan["songs"]) == 3

    analyze = client.post("/api/analyze", json={}).get_json()
    assert analyze["started"] is True
    assert analyze["total"] == 3

    deadline = time.time() + 60
    status = None
    while time.time() < deadline:
        status = client.get("/api/analyze/status").get_json()
        if not status["running"]:
            break
        time.sleep(0.5)
    assert status is not None and not status["running"], "analysis job did not finish in time"
    assert status["completed"] == 3
    assert status["errors"] == []

    library = client.get("/api/library").get_json()
    assert all(s["analyzed"] for s in library["songs"])

    plan_resp = client.post(
        "/api/plan", json={"starting_song_id": "a.wav", "direction": "build", "num_songs": 2}
    )
    assert plan_resp.status_code == 200
    plan = plan_resp.get_json()
    assert "a.wav" not in [s["song_id"] for s in plan["steps"]]
    assert len(plan["transitions"]) == len(plan["steps"])
    assert plan["render_available"] is False


def test_analyze_rejects_a_second_job_while_one_is_running(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    for i in range(4):
        _write_song(music_dir / f"song{i}.wav", duration=3.0, bpm=110.0 + i)

    client, state = _make_client(tmp_path)
    client.post("/api/library", json={"path": str(music_dir)})

    first = client.post("/api/analyze", json={})
    assert first.get_json()["started"] is True

    second = client.post("/api/analyze", json={})
    assert second.status_code == 409

    deadline = time.time() + 60
    while time.time() < deadline and state.jobs.is_running():
        time.sleep(0.5)


def test_song_debug_endpoint_returns_structured_data_once_analyzed(tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    _write_song(music_dir / "a.wav", duration=4.0, bpm=120.0)

    client, _ = _make_client(tmp_path)
    client.post("/api/library", json={"path": str(music_dir)})

    resp = client.get("/api/songs/a.wav/debug")
    assert resp.status_code == 409  # not analyzed yet

    client.post("/api/analyze", json={})
    deadline = time.time() + 60
    while time.time() < deadline:
        if not client.get("/api/analyze/status").get_json()["running"]:
            break
        time.sleep(0.5)

    resp = client.get("/api/songs/a.wav/debug")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "tempo" in data and "harmonic" in data and "structure" in data
