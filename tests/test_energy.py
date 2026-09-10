from __future__ import annotations

import numpy as np

from songanalysis.features.energy import analyze_energy
from songanalysis.features.loudness import short_term_loudness_series
from songanalysis.features.tempo_beat import analyze_tempo_beat
from conftest import SR, click_track, silence


def _full_energy(y):
    tb = analyze_tempo_beat(y, SR)
    lt, lv = short_term_loudness_series(y, SR)
    return analyze_energy(
        y,
        SR,
        onset_strength_times=tb.onset_strength_times,
        onset_strength=tb.onset_strength,
        onset_density_times=tb.onset_density_times,
        onset_density=tb.onset_density,
        loudness_times=lt,
        loudness_values=lv,
    )


def test_energy_timeline_shapes_are_consistent():
    y = click_track(20.0, 120.0)
    result = _full_energy(y)
    n = result.times.size
    assert n > 0
    for arr in (result.rms, result.spectral_energy, result.bass_energy, result.spectral_flux, result.onset_density, result.loudness_lufs, result.composite_energy):
        assert arr.size == n


def test_composite_energy_bounded_0_1():
    y = click_track(20.0, 120.0)
    result = _full_energy(y)
    assert np.all(result.composite_energy >= -1e-9)
    assert np.all(result.composite_energy <= 1.0 + 1e-9)
    assert 0.0 <= result.overall_energy <= 1.0
    assert result.min_energy <= result.overall_energy <= result.max_energy


def test_louder_track_has_higher_raw_rms_than_quiet_one():
    # `composite_energy` is intentionally normalized *within* a single
    # song's own min/max (it's meant for finding peaks/valleys inside one
    # track), so it is not comparable across two different songs. The raw,
    # un-normalized component streams are, though -- that's the point of
    # preserving them.
    quiet = _full_energy(click_track(20.0, 120.0, amplitude=0.08))
    loud = _full_energy(click_track(20.0, 120.0, amplitude=0.8))
    assert np.mean(loud.rms) > np.mean(quiet.rms)


def test_extrema_and_changes_are_within_track_bounds():
    y = click_track(30.0, 120.0)
    result = _full_energy(y)
    duration = y.size / SR
    for peak in result.local_peaks + result.local_valleys:
        assert 0.0 <= peak.time <= duration
    for change in result.major_energy_changes:
        assert 0.0 <= change.time <= duration
        assert change.direction in {"increase", "decrease"}
        assert 0.0 <= change.confidence <= 1.0


def test_silence_gives_near_zero_energy():
    result = _full_energy(silence(10.0))
    assert result.overall_energy < 0.05
