from __future__ import annotations

import json

from songanalysis.cache import AnalysisCache, analyze_song_cached
from songanalysis.version import ANALYSIS_VERSION


def test_cache_miss_then_hit(tmp_path, click_track_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")

    result1, hit1 = analyze_song_cached(click_track_path, cache=cache)
    assert hit1 is False

    result2, hit2 = analyze_song_cached(click_track_path, cache=cache)
    assert hit2 is True
    assert result1 == result2

    cached_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cached_files) == 1


def test_force_recomputes_and_overwrites(tmp_path, click_track_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    result1, hit1 = analyze_song_cached(click_track_path, cache=cache)
    assert hit1 is False

    result2, hit2 = analyze_song_cached(click_track_path, cache=cache, force=True)
    assert hit2 is False
    assert result1 == result2  # deterministic recompute of the same file


def test_different_files_get_different_cache_entries(tmp_path, click_track_path, very_short_wav_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    analyze_song_cached(click_track_path, cache=cache)
    analyze_song_cached(very_short_wav_path, cache=cache)
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_incompatible_cached_version_is_treated_as_miss(tmp_path, click_track_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    result, hit = analyze_song_cached(click_track_path, cache=cache)
    assert hit is False

    # Simulate a cache entry left over from an incompatible schema MAJOR version.
    from songanalysis.cache import content_hash

    file_hash = content_hash(click_track_path)
    cache_path = cache._cache_path(file_hash)
    stale = json.loads(cache_path.read_text())
    stale["analysis_version"] = "999.0"
    cache_path.write_text(json.dumps(stale))

    _, hit_after_bump = analyze_song_cached(click_track_path, cache=cache)
    assert hit_after_bump is False  # recomputed, not served from the stale entry


def test_corrupted_cache_file_is_treated_as_miss(tmp_path, click_track_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    from songanalysis.cache import content_hash

    file_hash = content_hash(click_track_path)
    cache_path = cache._cache_path(file_hash)
    cache_path.write_text("{not valid json")

    result, hit = analyze_song_cached(click_track_path, cache=cache)
    assert hit is False
    assert result["analysis_version"] == ANALYSIS_VERSION


def test_clear_removes_all_entries(tmp_path, click_track_path, very_short_wav_path):
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    analyze_song_cached(click_track_path, cache=cache)
    analyze_song_cached(very_short_wav_path, cache=cache)
    removed = cache.clear()
    assert removed == 2
    assert list((tmp_path / "cache").glob("*.json")) == []
