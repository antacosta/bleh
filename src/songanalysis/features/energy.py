"""Stage: feature normalization / time-varying energy analysis.

This module is where several independently-computed raw feature streams
(time-domain RMS, STFT-derived spectral/bass energy, onset density, spectral
flux, short-term loudness) get resampled onto one common, evenly-spaced time
grid and combined into a documented "composite energy" curve.

Every raw component is preserved alongside the composite on purpose: a
later scoring model may want "bass-heavy energy" or "purely rhythmic
density" instead of the blended composite, and should not have to
re-analyze the source audio to get it.

Note on ``composite_energy``: each component is min/max-normalized to the
*current song's own range* before blending, so the composite is only
meaningful for finding peaks/valleys/trends *within* one track (which is
what section/DJ-event detection needs). It is not a loudness measure and
must not be compared across two different songs -- for that, compare the
raw components (``rms``, ``spectral_energy``, ...) or the ``loudness``
stage's absolute LUFS/dBFS figures instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from songanalysis.util.timeseries import (
    LocalExtremum,
    change_points,
    find_local_extrema,
    linear_trend,
    normalize_01,
)

FRAME_LENGTH = 2048
HOP_LENGTH = 512
GRID_HOP_SEC = 0.5

_BASS_BAND_HZ = (20.0, 250.0)

# Weights used to blend the normalized (0..1) component streams into one
# composite energy curve. Documented here rather than hidden: consumers who
# want a different blend should combine the raw components themselves
# instead of treating this as ground truth.
_COMPOSITE_WEIGHTS = {
    "rms": 0.30,
    "spectral_energy": 0.20,
    "bass_energy": 0.15,
    "onset_density": 0.20,
    "spectral_flux": 0.15,
}


@dataclass(frozen=True)
class EnergyChangeEvent:
    time: float
    direction: str  # "increase" | "decrease"
    magnitude: float  # |delta| in composite-energy units (0..1 scale)
    confidence: float


@dataclass(frozen=True)
class EnergyTimeline:
    times: np.ndarray

    rms: np.ndarray
    spectral_energy: np.ndarray
    bass_energy: np.ndarray
    spectral_flux: np.ndarray
    onset_density: np.ndarray
    loudness_lufs: np.ndarray

    composite_energy: np.ndarray  # 0..1, documented blend of the above
    composite_weights: dict

    overall_energy: float
    min_energy: float
    max_energy: float
    energy_trend: float  # composite-energy units per second

    local_peaks: list[LocalExtremum] = field(default_factory=list)
    local_valleys: list[LocalExtremum] = field(default_factory=list)
    major_energy_changes: list[EnergyChangeEvent] = field(default_factory=list)


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if times.size < 2 or values.size < 2:
        return np.zeros_like(grid)
    return np.interp(grid, times, values, left=values[0], right=values[-1])


def analyze_energy(
    y: np.ndarray,
    sr: int,
    *,
    onset_strength_times: np.ndarray,
    onset_strength: np.ndarray,
    onset_density_times: np.ndarray,
    onset_density: np.ndarray,
    loudness_times: np.ndarray,
    loudness_values: np.ndarray,
    grid_hop_sec: float = GRID_HOP_SEC,
) -> EnergyTimeline:
    duration = y.size / sr if sr else 0.0
    if duration <= 0.0:
        empty = np.array([])
        return EnergyTimeline(
            times=empty,
            rms=empty,
            spectral_energy=empty,
            bass_energy=empty,
            spectral_flux=empty,
            onset_density=empty,
            loudness_lufs=empty,
            composite_energy=empty,
            composite_weights=_COMPOSITE_WEIGHTS,
            overall_energy=0.0,
            min_energy=0.0,
            max_energy=0.0,
            energy_trend=0.0,
        )

    rms_frames = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(np.arange(rms_frames.size), sr=sr, hop_length=HOP_LENGTH)

    stft = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    stft_times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    spectral_energy_frames = np.sqrt(np.mean(stft**2, axis=0))
    bass_mask = (freqs >= _BASS_BAND_HZ[0]) & (freqs <= _BASS_BAND_HZ[1])
    bass_energy_frames = (
        np.sqrt(np.mean(stft[bass_mask] ** 2, axis=0)) if np.any(bass_mask) else np.zeros(stft.shape[1])
    )

    grid = np.arange(0.0, duration, grid_hop_sec)
    if grid.size == 0:
        grid = np.array([0.0])

    rms_g = _resample(rms_times, rms_frames, grid)
    spectral_energy_g = _resample(stft_times, spectral_energy_frames, grid)
    bass_energy_g = _resample(stft_times, bass_energy_frames, grid)
    spectral_flux_g = _resample(onset_strength_times, onset_strength, grid)
    onset_density_g = _resample(onset_density_times, onset_density, grid)
    loudness_g = _resample(loudness_times, loudness_values, grid) if loudness_times.size else np.full(grid.shape, -70.0)

    norm_components = {
        "rms": normalize_01(rms_g),
        "spectral_energy": normalize_01(spectral_energy_g),
        "bass_energy": normalize_01(bass_energy_g),
        "onset_density": normalize_01(onset_density_g),
        "spectral_flux": normalize_01(spectral_flux_g),
    }
    composite = np.zeros_like(grid, dtype=np.float64)
    for name, weight in _COMPOSITE_WEIGHTS.items():
        composite += weight * norm_components[name]
    total_weight = sum(_COMPOSITE_WEIGHTS.values())
    if total_weight > 0:
        composite /= total_weight

    peaks, valleys = find_local_extrema(composite, grid, min_separation_sec=max(4.0, grid_hop_sec * 4))
    trend = linear_trend(composite, grid)

    raw_changes = change_points(composite, grid, z_threshold=2.0, min_separation_sec=4.0)
    major_changes = [
        EnergyChangeEvent(
            time=cp.time,
            direction="increase" if cp.value > 0 else "decrease",
            magnitude=float(abs(cp.value)),
            confidence=float(np.clip(abs(cp.value) / (np.std(np.diff(composite)) + 1e-9) / 3.0, 0.0, 1.0)),
        )
        for cp in raw_changes
    ]

    return EnergyTimeline(
        times=grid,
        rms=rms_g,
        spectral_energy=spectral_energy_g,
        bass_energy=bass_energy_g,
        spectral_flux=spectral_flux_g,
        onset_density=onset_density_g,
        loudness_lufs=loudness_g,
        composite_energy=composite,
        composite_weights=dict(_COMPOSITE_WEIGHTS),
        overall_energy=float(np.mean(composite)) if composite.size else 0.0,
        min_energy=float(np.min(composite)) if composite.size else 0.0,
        max_energy=float(np.max(composite)) if composite.size else 0.0,
        energy_trend=trend,
        local_peaks=peaks,
        local_valleys=valleys,
        major_energy_changes=major_changes,
    )
