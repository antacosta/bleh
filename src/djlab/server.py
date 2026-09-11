"""Flask app: the only place this UI talks HTTP. Every route is a thin
wrapper around ``djlab.library`` / ``djlab.jobs`` / ``djlab.planning`` /
``djlab.profile_view``, which themselves only call the existing
``songanalysis`` / ``songscoring`` / ``songplanner`` packages -- no
analysis, scoring, or planning logic lives in this file.

State is in-memory and single-user by design (one local person testing one
engine on one machine at a time) -- there is no database, no accounts, no
multi-session handling. The server binds to 127.0.0.1 only (see cli.py);
it is never meant to be reachable from another machine.
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from songanalysis.cache import AnalysisCache
from songscoring.song_profile import SongProfile, profile_from_analysis

from djlab.jobs import AnalysisJobRunner
from djlab.library import LibrarySong, scan_library
from djlab.planning import UnknownDirectionError, generate_plan, plan_response_dict
from djlab.profile_view import profile_debug_dict

STATIC_DIR = Path(__file__).parent / "static"


class LabState:
    """All of this server's mutable state, in one place, guarded by one
    lock -- simple and sufficient for a single local user."""

    def __init__(self, cache: AnalysisCache | None = None) -> None:
        self.lock = threading.Lock()
        self.cache = cache or AnalysisCache()
        self.library_root: Path | None = None
        self.songs: dict[str, LibrarySong] = {}
        self.jobs = AnalysisJobRunner(self.cache)

    def set_library(self, root: Path) -> list[LibrarySong]:
        songs = scan_library(root, self.cache)
        with self.lock:
            self.library_root = root
            self.songs = {s.id: s for s in songs}
        return songs

    def refresh_library(self) -> list[LibrarySong]:
        with self.lock:
            root = self.library_root
        if root is None:
            return []
        return self.set_library(root)

    def get_song(self, song_id: str) -> LibrarySong | None:
        with self.lock:
            return self.songs.get(song_id)

    def all_songs(self) -> list[LibrarySong]:
        with self.lock:
            return list(self.songs.values())

    def load_profile(self, song: LibrarySong) -> SongProfile | None:
        cached = self.cache.load(song.path)
        if cached is None:
            return None
        return profile_from_analysis(cached, id=song.id, source_path=song.path)


def create_app(state: LabState | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    state = state or LabState()
    app.config["LAB_STATE"] = state

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/library")
    def get_library():
        songs = state.refresh_library()
        root = state.library_root
        return jsonify(root=str(root) if root else None, songs=[s.to_dict() for s in songs])

    @app.post("/api/library")
    def set_library_route():
        payload = request.get_json(silent=True) or {}
        raw_path = (payload.get("path") or "").strip()
        if not raw_path:
            return jsonify(error="path is required"), 400
        root = Path(raw_path).expanduser()
        if not root.is_dir():
            return jsonify(error=f"not a directory: {root}"), 400
        songs = state.set_library(root)
        return jsonify(root=str(root), songs=[s.to_dict() for s in songs])

    @app.post("/api/analyze")
    def start_analyze():
        payload = request.get_json(silent=True) or {}
        song_ids = payload.get("song_ids")
        all_songs = state.all_songs()
        if song_ids:
            targets = [s for s in all_songs if s.id in set(song_ids)]
        else:
            targets = [s for s in all_songs if not s.analyzed]
        if not targets:
            return jsonify(started=False, total=0, reason="nothing to analyze")
        started = state.jobs.start(targets)
        if not started:
            return jsonify(started=False, reason="analysis already running"), 409
        return jsonify(started=True, total=len(targets))

    @app.get("/api/analyze/status")
    def analyze_status():
        return jsonify(state.jobs.status())

    @app.get("/api/songs/<path:song_id>/debug")
    def song_debug(song_id: str):
        song = state.get_song(song_id)
        if song is None:
            return jsonify(error="unknown song"), 404
        profile = state.load_profile(song)
        if profile is None:
            return jsonify(error="song not analyzed yet"), 409
        return jsonify(profile_debug_dict(profile))

    @app.post("/api/plan")
    def make_plan():
        payload = request.get_json(silent=True) or {}
        starting_id = payload.get("starting_song_id")
        direction = payload.get("direction", "none")
        num_songs = payload.get("num_songs")
        duration_minutes = payload.get("duration_minutes")

        if not starting_id:
            return jsonify(error="starting_song_id is required"), 400

        starting_song = state.get_song(starting_id)
        if starting_song is None:
            return jsonify(error="unknown starting_song_id"), 404

        starting_profile = state.load_profile(starting_song)
        if starting_profile is None:
            return jsonify(error="starting song has not been analyzed yet"), 409

        all_songs = state.all_songs()
        analyzed_others = [s for s in all_songs if s.analyzed and s.id != starting_id]
        excluded_unanalyzed = sum(1 for s in all_songs if not s.analyzed and s.id != starting_id)
        library_profiles = [p for p in (state.load_profile(s) for s in analyzed_others) if p is not None]

        try:
            plan = generate_plan(
                starting_profile,
                library_profiles,
                direction=direction,
                num_songs=num_songs,
                duration_minutes=duration_minutes,
            )
        except UnknownDirectionError as exc:
            return jsonify(error=str(exc)), 400

        return jsonify(plan_response_dict(plan, excluded_unanalyzed_count=excluded_unanalyzed))

    return app
