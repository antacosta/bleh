"""Vocal/instrumental activity heuristic.

Deliberately *not* a source-separation model (no Demucs/Spleeter/etc): that
would pull in a heavyweight ML stack for a question this pipeline only
needs an approximate answer to ("is someone probably singing right now?").

Method (a light-weight, well-documented signal-processing heuristic, not a
trained classifier):

1. Harmonic/percussive source separation (``librosa.effects.hpss``, already
   a dependency) isolates the harmonic bed, where sung vocals mostly live,
   from drums/percussion.
2. Band-pass the harmonic signal to the vocal formant range (~150-4000 Hz).
3. Compute the amplitude envelope of that band and measure how much of its
   modulation energy falls in the ~2-6 Hz "syllabic rate" band -- a classic,
   simple speech/vocal-activity cue (cf. Scheirer & Slaney's 4 Hz modulation
   energy for speech/music discrimination).
4. Threshold (with hysteresis + minimum-duration merging) into vocal /
   instrumental regions.

Known limitations (why this stays a *heuristic* with confidence "low"):
sibilant/percussive vocal transients can leak into the percussive HPSS
component and get missed; sustained melodic lead instruments in the vocal
range (sax, guitar solos, synth leads) can produce similar modulation and
get misclassified as vocal; backing harmonies are not distinguished from
lead vocals; the technique estimates *any* vocal-like activity, not "a
human is definitely singing."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from songanalysis.util.timeseries import moving_average, normalize_01

_VOCAL_BAND_HZ = (150.0, 4000.0)
_MODULATION_BAND_HZ = (2.0, 6.0)
_MODULATION_CEILING_HZ = 12.0
_ENVELOPE_HOP = 256  # ~11.6ms at 22050 Hz -> envelope sample rate ~86 Hz
_ENVELOPE_FRAME = 1024
_WINDOW_SEC = 2.0
_HOP_SEC = 0.5
_VOCAL_ON_THRESHOLD = 0.55
_VOCAL_OFF_THRESHOLD = 0.45
_MIN_REGION_SEC = 1.5
_MIN_GAP_SEC = 1.0

METHOD_DESCRIPTION = (
    "hpss-harmonic + vocal-band (150-4000Hz) amplitude-envelope modulation energy "
    "in the 2-6Hz syllabic-rate band; heuristic, not a trained vocal/source-separation model"
)


@dataclass(frozen=True)
class VocalRegion:
    start: float
    end: float
    duration: float
    kind: str  # "vocal" | "instrumental"


@dataclass(frozen=True)
class VocalTimeline:
    times: np.ndarray
    vocal_probability: np.ndarray  # 0..1 heuristic score
    vocal_intensity: np.ndarray  # raw vocal-band energy (0..1 normalized), not gated by threshold
    regions: list[VocalRegion]
    vocal_density: float  # fraction of duration classified as vocal
    method: str = METHOD_DESCRIPTION
    confidence: str = "heuristic"


def _bandpass(y: np.ndarray, sr: int, band: tuple[float, float]) -> np.ndarray:
    nyquist = sr / 2.0
    low = max(band[0] / nyquist, 1e-4)
    high = min(band[1] / nyquist, 0.999)
    if low >= high:
        return np.zeros_like(y)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def _envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    env = librosa.feature.rms(y=y, frame_length=_ENVELOPE_FRAME, hop_length=_ENVELOPE_HOP)[0]
    times = librosa.frames_to_time(np.arange(env.size), sr=sr, hop_length=_ENVELOPE_HOP)
    return times, env


def _modulation_score(env_segment: np.ndarray, env_sr: float) -> float:
    if env_segment.size < 8:
        return 0.0
    centered = env_segment - np.mean(env_segment)
    if float(np.max(np.abs(centered))) < 1e-9:
        return 0.0
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / env_sr)
    total_mask = freqs <= _MODULATION_CEILING_HZ
    mod_mask = (freqs >= _MODULATION_BAND_HZ[0]) & (freqs <= _MODULATION_BAND_HZ[1])
    total_energy = float(np.sum(spectrum[total_mask] ** 2))
    if total_energy <= 1e-12:
        return 0.0
    mod_energy = float(np.sum(spectrum[mod_mask] ** 2))
    return float(np.clip(mod_energy / total_energy, 0.0, 1.0))


def _extract_regions(times: np.ndarray, probability: np.ndarray) -> list[VocalRegion]:
    if times.size == 0:
        return []
    is_vocal = False
    regions: list[VocalRegion] = []
    region_start = times[0]

    for i, (t, p) in enumerate(zip(times, probability)):
        if not is_vocal and p >= _VOCAL_ON_THRESHOLD:
            is_vocal = True
            region_start = t
        elif is_vocal and p <= _VOCAL_OFF_THRESHOLD:
            is_vocal = False
            regions.append((region_start, t))

    if is_vocal:
        regions.append((region_start, times[-1]))

    # Build the full vocal/instrumental partition (including gaps as instrumental).
    partition: list[VocalRegion] = []
    cursor = float(times[0])
    end_time = float(times[-1])
    for start, end in regions:
        if start > cursor + 1e-6:
            partition.append(VocalRegion(cursor, start, start - cursor, "instrumental"))
        partition.append(VocalRegion(start, end, end - start, "vocal"))
        cursor = end
    if cursor < end_time - 1e-6:
        partition.append(VocalRegion(cursor, end_time, end_time - cursor, "instrumental"))

    # Merge tiny regions (< MIN_REGION_SEC) into their neighbors to avoid flicker.
    merged: list[VocalRegion] = []
    for region in partition:
        if merged and region.duration < _MIN_REGION_SEC:
            prev = merged[-1]
            merged[-1] = VocalRegion(prev.start, region.end, region.end - prev.start, prev.kind)
        elif merged and merged[-1].kind == region.kind:
            prev = merged[-1]
            merged[-1] = VocalRegion(prev.start, region.end, region.end - prev.start, prev.kind)
        else:
            merged.append(region)
    return merged


def analyze_vocals(y: np.ndarray, sr: int) -> VocalTimeline:
    duration = y.size / sr if sr else 0.0
    if duration < _WINDOW_SEC or float(np.max(np.abs(y))) < 1e-6:
        return VocalTimeline(
            times=np.array([]),
            vocal_probability=np.array([]),
            vocal_intensity=np.array([]),
            regions=[],
            vocal_density=0.0,
        )

    harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))
    vocal_band = _bandpass(harmonic, sr, _VOCAL_BAND_HZ)

    env_times, env = _envelope(vocal_band, sr)
    env_sr = sr / _ENVELOPE_HOP

    window_frames = max(1, int(_WINDOW_SEC * env_sr))
    hop_frames = max(1, int(_HOP_SEC * env_sr))

    grid_times = []
    scores = []
    for start in range(0, max(env.size - window_frames, 0) + 1, hop_frames):
        segment = env[start : start + window_frames]
        score = _modulation_score(segment, env_sr)
        grid_times.append(float(env_times[min(start + window_frames // 2, env_times.size - 1)]))
        scores.append(score)

    if not grid_times:
        grid_times = [duration / 2]
        scores = [0.0]

    times = np.array(grid_times)
    probability = moving_average(np.array(scores), window=3)
    probability = np.clip(probability, 0.0, 1.0)

    intensity_raw = np.interp(times, env_times, env)
    vocal_intensity = normalize_01(intensity_raw)

    regions = _extract_regions(times, probability)
    vocal_total = sum(r.duration for r in regions if r.kind == "vocal")
    vocal_density = float(vocal_total / duration) if duration > 0 else 0.0

    return VocalTimeline(
        times=times,
        vocal_probability=probability,
        vocal_intensity=vocal_intensity,
        regions=regions,
        vocal_density=vocal_density,
    )
