"""Stage: serialization + disk cache.

Analysis results are cached by the content hash of the audio file (not its
path/mtime), so moving or renaming a file doesn't invalidate its cache
entry, but a byte-for-byte different file always gets its own entry.
Cache entries are tagged with the analysis schema's MAJOR version; a cache
entry whose MAJOR no longer matches the running code's is treated as a
miss and recomputed (see :mod:`songanalysis.version`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from songanalysis.errors import UnsupportedAudioFileError
from songanalysis.io.loader import DEFAULT_ANALYSIS_SR
from songanalysis.pipeline import analyze_song
from songanalysis.version import ANALYSIS_VERSION, is_compatible, parse_major

DEFAULT_CACHE_DIR = Path(os.environ.get("SONGANALYSIS_CACHE_DIR", str(Path.home() / ".cache" / "songanalysis")))


def content_hash(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise UnsupportedAudioFileError(f"Could not read file {path}: {exc}") from exc
    return digest.hexdigest()


class AnalysisCache:
    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}_v{parse_major(ANALYSIS_VERSION)}.json"

    def load(self, path: str | Path) -> dict | None:
        file_hash = content_hash(path)
        cache_path = self._cache_path(file_hash)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not is_compatible(str(data.get("analysis_version", ""))):
            return None
        return data

    def store(self, path: str | Path, result: dict) -> Path:
        file_hash = content_hash(path)
        cache_path = self._cache_path(file_hash)
        cache_path.write_text(json.dumps(result, indent=2))
        return cache_path

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


def analyze_song_cached(
    path: str | Path,
    *,
    cache: AnalysisCache | None = None,
    force: bool = False,
    analysis_sr: int = DEFAULT_ANALYSIS_SR,
) -> tuple[dict, bool]:
    """Returns ``(result_dict, was_cache_hit)``."""
    cache = cache or AnalysisCache()
    if not force:
        cached = cache.load(path)
        if cached is not None:
            return cached, True

    analysis = analyze_song(path, analysis_sr=analysis_sr)
    result = analysis.to_jsonable()
    cache.store(path, result)
    return result, False
