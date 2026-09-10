from __future__ import annotations

import numpy as np

from songanalysis.features.harmonic import analyze_harmonic
from conftest import SR, silence


def _chord(freqs, duration, sr=SR, amplitude=0.2):
    t = np.arange(int(duration * sr)) / sr
    y = sum(amplitude * np.sin(2 * np.pi * f * t) for f in freqs)
    return y.astype(np.float32)


def test_c_major_triad_is_detected_as_c_major():
    y = _chord([261.63, 329.63, 392.00], 12.0)  # C E G
    result = analyze_harmonic(y, SR)
    assert result.key == "C major"
    assert result.key_root == "C"
    assert result.key_mode == "major"
    assert 0.0 <= result.key_confidence <= 1.0
    assert result.chroma.shape[1] == 12
    assert result.chroma.shape[0] == result.chroma_times.size


def test_key_candidates_sorted_by_correlation_desc():
    y = _chord([261.63, 329.63, 392.00], 12.0)
    result = analyze_harmonic(y, SR)
    correlations = [c.correlation for c in result.key_candidates]
    assert correlations == sorted(correlations, reverse=True)
    assert len(result.key_candidates) <= 3


def test_silence_yields_no_key():
    result = analyze_harmonic(silence(5.0), SR)
    assert result.key is None
    assert result.key_confidence == 0.0
    assert result.chroma.shape[0] == 0


def test_confidence_never_fabricated_out_of_range():
    y = _chord([261.63, 329.63, 392.00], 12.0)
    result = analyze_harmonic(y, SR)
    for candidate in result.key_candidates:
        assert -1.0 <= candidate.correlation <= 1.0
    assert 0.0 <= result.key_confidence <= 1.0
