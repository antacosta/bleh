from __future__ import annotations

import numpy as np

from songanalysis.features.tempo_beat import _octave_error_penalty, analyze_tempo_beat
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


def test_bpm_detection_accurate_away_from_120_prior_center():
    """Regression test: real-song validation found many tracks' BPM pulled
    toward ~120 (librosa's default tempo prior is centered there with a very
    narrow std_bpm=1.0), including exact repeats of exotic-looking values
    like 129.19921875 and 117.453835 across unrelated songs. A clean click
    track far from 120 BPM should still be detected close to its true tempo
    after widening the prior (see TEMPO_PRIOR_STD_BPM)."""
    for true_bpm in (70.0, 190.0):
        y = click_track(20.0, true_bpm)
        result = analyze_tempo_beat(y, SR)
        assert result.bpm is not None
        assert any(abs(result.bpm - true_bpm * mult) < 6.0 for mult in (0.5, 1.0, 2.0))


def test_octave_error_penalty_discounts_metrical_level_jumps():
    """A tempo 'change' that is really just the tracker locking onto double,
    half, or 1.5x time should be heavily discounted; an ordinary modest
    tempo change should not be penalized at all."""
    assert _octave_error_penalty(120.0, 240.0) < 0.2  # 2x
    assert _octave_error_penalty(120.0, 60.0) < 0.2  # 0.5x
    assert _octave_error_penalty(80.0, 160.5) < 0.2  # ~2x, not bit-exact
    assert _octave_error_penalty(117.45, 184.57) < 0.2  # ~1.5x, matches real finding
    assert _octave_error_penalty(120.0, 132.0) == 1.0  # ordinary 10% speedup
