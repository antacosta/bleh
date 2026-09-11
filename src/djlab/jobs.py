"""Background analysis job: runs the real (slow) ``songanalysis`` pipeline
over a batch of songs on a worker thread, with a pollable status object --
so the UI's "Analyze" button doesn't block on a single long HTTP request.

This module only calls ``songanalysis.cache.analyze_song_cached``; it does
not reimplement any part of analysis.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from songanalysis.cache import AnalysisCache, analyze_song_cached
from songanalysis.errors import SongAnalysisError

from djlab.library import LibrarySong


@dataclass
class AnalysisJobStatus:
    running: bool = False
    total: int = 0
    completed: int = 0
    current: str | None = None
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "total": self.total,
            "completed": self.completed,
            "current": self.current,
            "errors": list(self.errors),
        }


class AnalysisJobRunner:
    """One job at a time, by design -- this is a single-user local tool, so
    there's no need for a job queue; starting a second job while one runs
    is simply rejected (see ``server.py``)."""

    def __init__(self, cache: AnalysisCache):
        self._cache = cache
        self._status = AnalysisJobStatus()
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            return self._status.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def start(self, songs: list[LibrarySong]) -> bool:
        """Returns False without starting anything if a job is already
        running."""
        with self._lock:
            if self._status.running:
                return False
            self._status = AnalysisJobStatus(running=True, total=len(songs), completed=0, current=None, errors=[])
        thread = threading.Thread(target=self._run, args=(songs,), daemon=True)
        thread.start()
        return True

    def _run(self, songs: list[LibrarySong]) -> None:
        for song in songs:
            with self._lock:
                self._status.current = song.filename
            try:
                analyze_song_cached(song.path, cache=self._cache)
            except SongAnalysisError as exc:
                with self._lock:
                    self._status.errors.append({"song_id": song.id, "filename": song.filename, "error": str(exc)})
            with self._lock:
                self._status.completed += 1
        with self._lock:
            self._status.running = False
            self._status.current = None
