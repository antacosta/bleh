from __future__ import annotations

import pytest

from songanalysis.errors import EmptyAudioError, UnsupportedAudioFileError
from songanalysis.io.loader import load_audio


def test_load_valid_file(click_track_path):
    buf = load_audio(click_track_path)
    assert buf.sr > 0
    assert buf.y.ndim == 1
    assert buf.y.size > 0
    assert buf.duration_sec == pytest.approx(20.0, abs=0.1)
    assert buf.raw.sample_rate > 0
    assert buf.raw.channels == 1


def test_load_very_short_file(very_short_wav_path):
    buf = load_audio(very_short_wav_path)
    assert buf.y.size > 0
    assert buf.duration_sec < 1.0


def test_silent_file_loads_and_is_flagged(silent_wav_path):
    buf = load_audio(silent_wav_path)
    assert buf.is_effectively_silent


def test_missing_file_raises(missing_file_path):
    with pytest.raises(UnsupportedAudioFileError):
        load_audio(missing_file_path)


def test_empty_file_raises(empty_file_path):
    with pytest.raises(UnsupportedAudioFileError):
        load_audio(empty_file_path)


def test_malformed_file_raises(malformed_file_path):
    with pytest.raises(UnsupportedAudioFileError):
        load_audio(malformed_file_path)


def test_analysis_sr_is_applied(click_track_path):
    buf = load_audio(click_track_path, analysis_sr=8000)
    assert buf.sr == 8000
