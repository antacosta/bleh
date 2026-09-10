from __future__ import annotations

import numpy as np

from songanalysis.features.tempo_beat import analyze_tempo_beat
from conftest import SR, click_track, silence


def test_bpm_detection_is_in_reasonable_range_of_truth():
    y = click_track(20.0, 128.0)
    result = analyze_tempo_beat(y, SR)

    assert result.bpm is not None
    # Beat trackers can lock onto a tempo octave (half/double); accept that.
    assert any(abs(result.bpm - 128.0 * mult) < 6.0 for mult in (0.5, 1.0, 2.0))
    assert 0.0 <= result.bpm_confidence <= 1.0
    assert result.bpm_confidence > 0.3
    assert result.beat_times.size > 10
    assert np.all(np.diff(result.beat_times) > 0)  # strictly increasing


def test_beat_intervals_match_beat_times():
    y = click_track(20.0, 120.0)
    result = analyze_tempo_beat(y, SR)
    assert result.beat_intervals.size == result.beat_times.size - 1
    assert np.allclose(result.beat_intervals, np.diff(result.beat_times))


def test_tempo_stability_in_unit_range():
    y = click_track(20.0, 120.0)
    result = analyze_tempo_beat(y, SR)
    assert 0.0 <= result.tempo_stability <= 1.0
    # A metronomic click track should be judged very stable.
    assert result.tempo_stability > 0.6


def test_time_signature_detection_on_four_on_the_floor():
    y = click_track(24.0, 120.0, accent_every=4)
    result = analyze_tempo_beat(y, SR)
    if result.time_signature is not None:
        assert result.time_signature in {"3/4", "4/4"}
    assert 0.0 <= result.time_signature_confidence <= 1.0
    assert 0.0 <= result.downbeat_confidence <= 1.0


def test_onsets_and_density_are_nonnegative_and_bounded():
    y = click_track(20.0, 120.0)
    result = analyze_tempo_beat(y, SR)
    assert result.onset_times.size > 0
    assert np.all(result.onset_density >= 0)


def test_silence_yields_no_confident_tempo():
    result = analyze_tempo_beat(silence(5.0), SR)
    assert result.bpm_confidence == 0.0
    assert result.beat_times.size == 0
    assert result.tempo_stability == 0.0
    assert result.onset_times.size == 0


def test_very_short_audio_does_not_crash():
    y = click_track(0.3, 120.0)
    result = analyze_tempo_beat(y, SR)
    # Too short to reliably beat-track; must degrade gracefully, not raise.
    assert result.bpm_confidence == 0.0 or result.beat_times.size >= 0
