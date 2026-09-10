from __future__ import annotations

import numpy as np

from songanalysis.features.vocals import analyze_vocals
from conftest import SR, am_vocal_tone, silence, sine


def test_am_modulated_tone_is_detected_as_mostly_vocal():
    y = am_vocal_tone(20.0, amplitude=0.3)
    result = analyze_vocals(y, SR)
    assert result.vocal_density > 0.5
    assert any(r.kind == "vocal" for r in result.regions)


def test_pure_steady_tone_is_mostly_instrumental():
    # A steady (unmodulated) tone has no syllabic-rate modulation energy.
    y = sine(220.0, 20.0, amplitude=0.3)
    result = analyze_vocals(y, SR)
    assert result.vocal_density < 0.3


def test_regions_partition_the_track_contiguously():
    y = am_vocal_tone(15.0, amplitude=0.3)
    result = analyze_vocals(y, SR)
    if len(result.regions) > 1:
        for a, b in zip(result.regions, result.regions[1:]):
            assert a.end == b.start
        assert result.regions[0].start == result.times[0]
        assert result.regions[-1].end == result.times[-1]


def test_probability_and_intensity_bounded():
    y = am_vocal_tone(15.0, amplitude=0.3)
    result = analyze_vocals(y, SR)
    assert np.all(result.vocal_probability >= 0.0)
    assert np.all(result.vocal_probability <= 1.0)
    assert np.all(result.vocal_intensity >= 0.0)
    assert np.all(result.vocal_intensity <= 1.0)
    assert 0.0 <= result.vocal_density <= 1.0


def test_confidence_is_explicitly_marked_heuristic():
    y = am_vocal_tone(15.0, amplitude=0.3)
    result = analyze_vocals(y, SR)
    assert result.confidence == "heuristic"
    assert result.method


def test_silence_and_too_short_degrade_gracefully():
    result_silent = analyze_vocals(silence(10.0), SR)
    assert result_silent.vocal_density == 0.0
    assert result_silent.regions == []

    result_short = analyze_vocals(sine(440.0, 0.2), SR)
    assert result_short.vocal_density == 0.0
