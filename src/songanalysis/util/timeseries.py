"""Generic helpers for analyzing 1-D time series derived from audio frames
(energy curves, onset-density curves, etc). Kept independent of librosa so
it can be unit tested on plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) * hop_length / sr


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float64, copy=True)
    window = min(window, x.size)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def normalize_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


@dataclass(frozen=True)
class LocalExtremum:
    time: float
    value: float
    index: int


def find_local_extrema(
    values: np.ndarray,
    times: np.ndarray,
    *,
    smoothing_window: int = 5,
    min_prominence_frac: float = 0.08,
    min_separation_sec: float = 2.0,
) -> tuple[list[LocalExtremum], list[LocalExtremum]]:
    """Find local peaks and valleys in a smoothed version of ``values``.

    ``min_prominence_frac`` is a fraction of the overall value range used as
    a minimum-prominence filter so that DC noise doesn't register as
    thousands of spurious extrema. Returns ``(peaks, valleys)``.
    """
    if values.size < 3:
        return [], []

    smoothed = moving_average(values, smoothing_window)
    value_range = float(np.max(smoothed) - np.min(smoothed))
    min_prominence = value_range * min_prominence_frac

    peaks: list[LocalExtremum] = []
    valleys: list[LocalExtremum] = []

    for i in range(1, len(smoothed) - 1):
        is_peak = smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
        is_valley = smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]
        if not (is_peak or is_valley):
            continue

        window_lo = max(0, i - 20)
        window_hi = min(len(smoothed), i + 20)
        local_window = smoothed[window_lo:window_hi]
        if is_peak:
            prominence = smoothed[i] - float(np.min(local_window))
        else:
            prominence = float(np.max(local_window)) - smoothed[i]
        if prominence < min_prominence:
            continue

        extremum = LocalExtremum(time=float(times[i]), value=float(values[i]), index=i)
        target = peaks if is_peak else valleys
        if target and (extremum.time - target[-1].time) < min_separation_sec:
            better = extremum.value > target[-1].value if is_peak else extremum.value < target[-1].value
            if better:
                target[-1] = extremum
            continue
        target.append(extremum)

    return peaks, valleys


def linear_trend(values: np.ndarray, times: np.ndarray) -> float:
    """Slope of a least-squares line fit through ``values`` vs ``times``,
    in value-units per second. Returns 0.0 for degenerate/near-constant
    input rather than raising."""
    if values.size < 2:
        return 0.0
    if float(np.ptp(times)) < 1e-9:
        return 0.0
    slope, _ = np.polyfit(times, values, 1)
    return float(slope)


def windowed_rate(
    event_times: np.ndarray,
    duration_sec: float,
    *,
    window_sec: float = 4.0,
    hop_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate (events/sec) of point events inside a sliding window, sampled
    every ``hop_sec``. Used for onset density, rhythmic density, etc."""
    if duration_sec <= 0:
        return np.array([]), np.array([])
    centers = []
    rates = []
    t = 0.0
    while t <= max(duration_sec - 1e-9, 0.0):
        w_start = t
        w_end = min(t + window_sec, duration_sec)
        eff_window = max(w_end - w_start, 1e-6)
        count = int(np.sum((event_times >= w_start) & (event_times < w_end)))
        centers.append((w_start + w_end) / 2)
        rates.append(count / eff_window)
        t += hop_sec
        if w_end >= duration_sec:
            break
    return np.array(centers), np.array(rates)


def change_points(
    values: np.ndarray,
    times: np.ndarray,
    *,
    smoothing_window: int = 5,
    z_threshold: float = 2.0,
    min_separation_sec: float = 3.0,
) -> list[LocalExtremum]:
    """Detect points of large, sustained change in a time series using a
    z-scored first derivative of a smoothed curve. Heuristic, not a
    statistically rigorous change-point detector."""
    if values.size < 4:
        return []
    smoothed = moving_average(values, smoothing_window)
    deriv = np.diff(smoothed)
    std = float(np.std(deriv))
    if std < 1e-12:
        return []
    z = np.abs(deriv) / std

    candidates: list[LocalExtremum] = []
    for i in np.where(z >= z_threshold)[0]:
        idx = int(i) + 1
        candidates.append(LocalExtremum(time=float(times[idx]), value=float(deriv[i]), index=idx))

    candidates.sort(key=lambda c: -abs(c.value))
    kept: list[LocalExtremum] = []
    for c in candidates:
        if all(abs(c.time - k.time) >= min_separation_sec for k in kept):
            kept.append(c)
    kept.sort(key=lambda c: c.time)
    return kept
