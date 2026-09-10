"""Loudness / level features: integrated LUFS, peak, RMS, dynamic range.

Uses ``pyloudnorm`` (ITU-R BS.1770 K-weighted loudness) for the integrated
loudness measurement -- no cloud APIs, no heavyweight ML, pure numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln

#: pyloudnorm/BS.1770 needs at least one 400ms gating block to produce a
#: meaningful integrated-loudness figure.
_MIN_DURATION_FOR_LUFS_SEC = 0.5
_SILENCE_FLOOR_DBFS = -120.0


@dataclass(frozen=True)
class LoudnessFeatures:
    integrated_lufs: float | None
    integrated_lufs_confidence: str  # "measured" | "insufficient_data" | "silent"
    peak_dbfs: float
    peak_linear: float
    rms_dbfs: float
    rms_linear: float
    crest_factor_db: float
    loudness_range_db: float | None
    loudness_range_confidence: str  # "measured" | "insufficient_data" | "silent"


def _linear_to_dbfs(value: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return _SILENCE_FLOOR_DBFS
    return max(20.0 * math.log10(value), _SILENCE_FLOOR_DBFS)


def safe_integrated_loudness(y: np.ndarray, sr: int) -> tuple[float | None, str]:
    """Integrated LUFS via BS.1770, or ``(None, reason)`` when it can't be
    reliably computed (too short, or effectively silent)."""
    if y.size == 0 or (y.size / sr) < _MIN_DURATION_FOR_LUFS_SEC:
        return None, "insufficient_data"
    if float(np.max(np.abs(y))) < 1e-6:
        return None, "silent"
    try:
        meter = pyln.Meter(sr)
        value = meter.integrated_loudness(y.astype(np.float64))
    except Exception:
        return None, "insufficient_data"
    if not math.isfinite(value) or value < -70.0:
        return None, "silent"
    return float(value), "measured"


def short_term_loudness_series(
    y: np.ndarray,
    sr: int,
    *,
    window_sec: float = 3.0,
    hop_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window LUFS estimate, used both for the loudness-range
    figure below and as one component of the energy timeline. Returns
    ``(times, lufs_values)`` where windows too quiet/short to measure are
    represented as ``-70.0`` (BS.1770's absolute silence gate) rather than
    dropped, so the series stays evenly sampled.
    """
    window = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if window <= 0 or hop <= 0 or y.size < window:
        return np.array([]), np.array([])

    try:
        meter = pyln.Meter(sr)
    except Exception:
        return np.array([]), np.array([])

    times = []
    values = []
    for start in range(0, y.size - window + 1, hop):
        chunk = y[start : start + window].astype(np.float64)
        try:
            loudness = meter.integrated_loudness(chunk)
        except Exception:
            loudness = -70.0
        if not math.isfinite(loudness):
            loudness = -70.0
        loudness = max(loudness, -70.0)
        times.append((start + window / 2) / sr)
        values.append(loudness)

    return np.array(times), np.array(values)


def analyze_loudness(y: np.ndarray, sr: int) -> LoudnessFeatures:
    integrated, integrated_conf = safe_integrated_loudness(y, sr)

    peak_linear = float(np.max(np.abs(y))) if y.size else 0.0
    peak_dbfs = _linear_to_dbfs(peak_linear)

    rms_linear = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    rms_dbfs = _linear_to_dbfs(rms_linear)

    crest_factor_db = peak_dbfs - rms_dbfs

    lra: float | None = None
    lra_conf = "insufficient_data"
    _, st_values = short_term_loudness_series(y, sr)
    voiced = st_values[st_values > -70.0] if st_values.size else st_values
    if voiced.size >= 4:
        lra = float(np.percentile(voiced, 95) - np.percentile(voiced, 10))
        lra_conf = "measured"
    elif st_values.size and voiced.size == 0:
        lra_conf = "silent"

    return LoudnessFeatures(
        integrated_lufs=integrated,
        integrated_lufs_confidence=integrated_conf,
        peak_dbfs=peak_dbfs,
        peak_linear=peak_linear,
        rms_dbfs=rms_dbfs,
        rms_linear=rms_linear,
        crest_factor_db=crest_factor_db,
        loudness_range_db=lra,
        loudness_range_confidence=lra_conf,
    )
