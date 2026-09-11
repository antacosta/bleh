"""Library scanning: turns a local folder into a list of songs the UI can
show, distinguishing already-analyzed songs from ones that still need a
trip through the real ``songanalysis`` pipeline.

Listing a song never runs the (slow) DSP pipeline -- only the fast format
probe (``probe_audio_file``) and tag read (``extract_metadata``), both
already part of ``songanalysis``. Whether a song is "analyzed" is answered
by checking the *existing* on-disk analysis cache (``AnalysisCache``), not
by tracking state of our own -- this module adds no analysis state that
``songanalysis`` doesn't already own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from songanalysis.cache import AnalysisCache
from songanalysis.errors import SongAnalysisError
from songanalysis.io.loader import probe_audio_file
from songanalysis.io.metadata import extract_metadata

#: Containers ``songanalysis`` can read (libsndfile directly, or via the
#: ffmpeg fallback probe/decode path -- see songanalysis/io/loader.py).
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}


@dataclass(frozen=True)
class LibrarySong:
    """One song as the UI's library list shows it -- always populated from
    fast tag/format reads; the ``bpm``/``key``/``overall_energy`` preview
    fields are only ever filled in from an *existing* cached analysis, never
    computed here."""

    id: str  # path relative to the library root -- stable, JSON-safe
    path: str  # absolute path on disk
    filename: str
    artist: str | None
    title: str | None
    album: str | None
    genre: str | None
    duration_sec: float
    analyzed: bool
    bpm: float | None = None
    key: str | None = None
    overall_energy: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "genre": self.genre,
            "duration_sec": round(self.duration_sec, 1),
            "analyzed": self.analyzed,
            "bpm": round(self.bpm, 1) if self.bpm is not None else None,
            "key": self.key,
            "overall_energy": round(self.overall_energy, 3) if self.overall_energy is not None else None,
        }


def _song_from_path(path: Path, root: Path, cache: AnalysisCache) -> LibrarySong | None:
    try:
        raw = probe_audio_file(path)
        meta = extract_metadata(path, raw)
    except SongAnalysisError:
        return None  # not a readable audio file -- skip silently, a real folder has non-audio files in it

    cached = cache.load(path)
    glob = (cached or {}).get("global", {})

    return LibrarySong(
        id=str(path.relative_to(root)),
        path=str(path),
        filename=path.name,
        artist=meta.artist,
        title=meta.title,
        album=meta.album,
        genre=meta.genre,
        duration_sec=meta.duration_sec,
        analyzed=cached is not None,
        bpm=glob.get("bpm"),
        key=glob.get("key"),
        overall_energy=glob.get("overall_energy"),
    )


def scan_library(root: Path, cache: AnalysisCache) -> list[LibrarySong]:
    """List every readable audio file under ``root`` (recursively), each
    tagged with whether it already has a cached analysis.

    Re-checks the cache fresh on every call -- for a local test library
    (tens of songs) this is fast enough that no additional caching of our
    own is worth the complexity; see the module docstring.
    """
    songs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        song = _song_from_path(path, root, cache)
        if song is not None:
            songs.append(song)
    return songs
