from __future__ import annotations

import numpy as np
import pytest

from songanalysis.features.loudness import analyze_loudness
from conftest import SR, sine, silence


def test_full_scale_sine_peak_and_rms():
    y = sine(440.0, 3.0, amplitude=1.0)
    result = analyze_loudness(y, SR)

    assert result.peak_linear == pytest.approx(1.0, abs=1e-3)
    assert result.peak_dbfs == pytest.approx(0.0, abs=0.1)
    # RMS of a full-scale sine is 1/sqrt(2) ~ -3.01 dBFS
    assert result.rms_dbfs == pytest.approx(-3.01, abs=0.5)
    assert result.crest_factor_db > 0
    assert result.integrated_lufs_confidence == "measured"
    assert result.integrated_lufs is not None


def test_quieter_signal_has_lower_rms_and_peak():
    loud = analyze_loudness(sine(440.0, 3.0, amplitude=0.9), SR)
    quiet = analyze_loudness(sine(440.0, 3.0, amplitude=0.1), SR)
    assert quiet.peak_dbfs < loud.peak_dbfs
    assert quiet.rms_dbfs < loud.rms_dbfs
    if quiet.integrated_lufs is not None and loud.integrated_lufs is not None:
        assert quiet.integrated_lufs < loud.integrated_lufs


def test_silence_reports_no_measurable_loudness():
    result = analyze_loudness(silence(3.0), SR)
    assert result.integrated_lufs is None
    assert result.integrated_lufs_confidence == "silent"
    assert result.peak_linear == 0.0


def test_too_short_for_lufs_is_explicit_about_it():
    y = sine(440.0, 0.1, amplitude=0.5)
    result = analyze_loudness(y, SR)
    assert result.integrated_lufs is None
    assert result.integrated_lufs_confidence == "insufficient_data"


def test_empty_array_does_not_crash():
    result = analyze_loudness(np.array([], dtype=np.float32), SR)
    assert result.integrated_lufs is None
    assert result.peak_linear == 0.0
    assert result.rms_linear == 0.0
