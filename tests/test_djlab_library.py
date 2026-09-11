from __future__ import annotations

from pathlib import Path

import soundfile as sf

from conftest import SR, click_track
from songanalysis.cache import AnalysisCache, analyze_song_cached
from djlab.library import SUPPORTED_EXTENSIONS, scan_library


def _write_song(path: Path, duration: float = 3.0, bpm: float = 120.0) -> None:
    sf.write(str(path), click_track(duration, bpm), SR)


def test_scan_finds_supported_audio_files_recursively(tmp_path):
    _write_song(tmp_path / "top.wav")
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_song(sub / "nested.wav")
    (tmp_path / "not_audio.txt").write_text("hello")

    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    songs = scan_library(tmp_path, cache)

    ids = {s.id for s in songs}
    assert ids == {"top.wav", str(Path("sub") / "nested.wav")}


def test_scan_skips_unreadable_files_silently(tmp_path):
    (tmp_path / "corrupt.wav").write_bytes(b"not actually audio")
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    songs = scan_library(tmp_path, cache)
    assert songs == []


def test_unanalyzed_song_is_reported_correctly(tmp_path):
    _write_song(tmp_path / "song.wav")
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    songs = scan_library(tmp_path, cache)

    assert len(songs) == 1
    song = songs[0]
    assert song.analyzed is False
    assert song.bpm is None
    assert song.key is None
    assert song.duration_sec > 0  # from the fast metadata probe, not full analysis


def test_analyzed_song_is_reported_with_preview_fields(tmp_path):
    path = tmp_path / "song.wav"
    _write_song(path, duration=6.0, bpm=128.0)
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    analyze_song_cached(path, cache=cache)  # pre-populate the cache, as if analyzed earlier

    songs = scan_library(tmp_path, cache)
    assert len(songs) == 1
    song = songs[0]
    assert song.analyzed is True
    assert song.bpm is not None


def test_supported_extensions_is_a_reasonable_set():
    assert ".wav" in SUPPORTED_EXTENSIONS
    assert ".mp3" in SUPPORTED_EXTENSIONS
    assert ".txt" not in SUPPORTED_EXTENSIONS
