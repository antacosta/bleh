"""Beat, tempo and onset/rhythm analysis.

Everything here preserves timestamps rather than collapsing to a single
number -- a future DJ planner needs to know exactly where beats, downbeats
and rhythmic events fall, not just "the song is 128 BPM".

Downbeat and time-signature estimation are heuristic (no dedicated downbeat
model is used, to avoid heavyweight/fragile dependencies like madmom): they
score candidate bar-groupings of the detected beat grid by accent strength
and report an explicit confidence, defaulting to "unknown" when the signal
is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import librosa.feature.rhythm as librosa_rhythm
import numpy as np

from songanalysis.util.timeseries import change_points, windowed_rate

HOP_LENGTH = 512
_MIN_DURATION_FOR_RHYTHM_SEC = 2.0
_MIN_BEATS_FOR_DOWNBEAT = 8
_CANDIDATE_BEATS_PER_BAR = (3, 4)


@dataclass(frozen=True)
class TempoChangeEvent:
    time: float
    from_bpm: float
    to_bpm: float
    confidence: float


@dataclass(frozen=True)
class RhythmicEvent:
    time: float
    strength: float  # normalized 0..1 relative to this song's onsets


@dataclass(frozen=True)
class TempoBeatFeatures:
    bpm: float | None
    bpm_confidence: float  # 0..1, relative strength of the dominant tempo periodicity
    beat_times: np.ndarray
    beat_intervals: np.ndarray
    tempo_stability: float  # 0..1, 1 = perfectly steady inter-beat spacing

    tempo_curve_times: np.ndarray
    tempo_curve_bpm: np.ndarray
    tempo_change_events: list[TempoChangeEvent]

    downbeat_times: np.ndarray
    downbeat_confidence: float
    beats_per_bar: int | None
    time_signature: str | None
    time_signature_confidence: float

    onset_times: np.ndarray
    onset_strength_times: np.ndarray
    onset_strength: np.ndarray
    onset_density_times: np.ndarray
    onset_density: np.ndarray  # onsets/sec, sliding window

    percussive_activity_times: np.ndarray
    percussive_activity: np.ndarray  # normalized 0..1 onset-strength of percussive component

    strong_rhythmic_events: list[RhythmicEvent] = field(default_factory=list)


def _empty_result() -> TempoBeatFeatures:
    empty = np.array([])
    return TempoBeatFeatures(
        bpm=None,
        bpm_confidence=0.0,
        beat_times=empty,
        beat_intervals=empty,
        tempo_stability=0.0,
        tempo_curve_times=empty,
        tempo_curve_bpm=empty,
        tempo_change_events=[],
        downbeat_times=empty,
        downbeat_confidence=0.0,
        beats_per_bar=None,
        time_signature=None,
        time_signature_confidence=0.0,
        onset_times=empty,
        onset_strength_times=empty,
        onset_strength=empty,
        onset_density_times=empty,
        onset_density=empty,
        percussive_activity_times=empty,
        percussive_activity=empty,
        strong_rhythmic_events=[],
    )


def _tempo_confidence(onset_env: np.ndarray, sr: int, bpm: float) -> float:
    if onset_env.size == 0 or bpm <= 0:
        return 0.0
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    profile = np.mean(tempogram, axis=1)
    freqs = librosa.tempo_frequencies(len(profile), hop_length=HOP_LENGTH, sr=sr)
    valid = freqs > 0
    if not np.any(valid):
        return 0.0
    profile, freqs = profile[valid], freqs[valid]
    idx = int(np.argmin(np.abs(freqs - bpm)))
    peak = profile[idx]
    baseline = float(np.median(profile))
    spread = float(np.max(profile) - baseline)
    if spread <= 1e-9:
        return 0.0
    return float(np.clip((peak - baseline) / spread, 0.0, 1.0))


def _tempo_stability(beat_intervals: np.ndarray) -> float:
    if beat_intervals.size < 2:
        return 0.0
    mean_interval = float(np.mean(beat_intervals))
    if mean_interval <= 1e-9:
        return 0.0
    cv = float(np.std(beat_intervals)) / mean_interval
    return float(np.clip(1.0 - cv / 0.15, 0.0, 1.0))


def _window_stability(segment: np.ndarray) -> float:
    """0..1: how internally consistent a short bpm-curve window is. Low for
    a window that is itself jittery/noisy (no reliable local periodicity),
    which should not be reported as a confident tempo change even if its
    mean differs a lot from a neighboring window's mean."""
    if segment.size < 2:
        return 0.0
    mean = float(np.mean(segment))
    if mean <= 1e-9:
        return 0.0
    cv = float(np.std(segment)) / mean
    return float(np.clip(1.0 - cv / 0.08, 0.0, 1.0))


def _tempo_change_events(
    times: np.ndarray, bpm_curve: np.ndarray, *, global_confidence: float
) -> list[TempoChangeEvent]:
    if bpm_curve.size < 8:
        return []
    points = change_points(bpm_curve, times, smoothing_window=8, z_threshold=2.5, min_separation_sec=5.0)
    events = []
    for cp in points:
        lo, hi = max(0, cp.index - 8), min(len(bpm_curve), cp.index + 8)
        before = bpm_curve[lo : cp.index] if cp.index > lo else bpm_curve[lo:hi]
        after = bpm_curve[cp.index : hi] if cp.index < hi else bpm_curve[lo:hi]
        from_bpm = float(np.mean(before)) if before.size else float(bpm_curve[cp.index])
        to_bpm = float(np.mean(after)) if after.size else float(bpm_curve[cp.index])
        if from_bpm <= 1e-6:
            continue
        rel_change = abs(to_bpm - from_bpm) / from_bpm
        if rel_change < 0.03:  # ignore sub-3% jitter
            continue
        # A jump between two noisy/unstable windows is more likely beat-tracker
        # confusion (e.g. sparse, non-percussive passages) than a real tempo
        # change, so local plateau-stability and the song's overall tempo
        # confidence both gate the final confidence, not just jump size.
        local_stability = min(_window_stability(before), _window_stability(after))
        confidence = float(
            np.clip(rel_change / 0.25, 0.0, 1.0) * local_stability * global_confidence
        )
        if confidence < 0.1:
            continue
        events.append(TempoChangeEvent(time=cp.time, from_bpm=from_bpm, to_bpm=to_bpm, confidence=confidence))
    return events


def _estimate_downbeats(
    beat_times: np.ndarray, onset_env: np.ndarray, onset_env_times: np.ndarray
) -> tuple[np.ndarray, float, int | None, str | None, float]:
    if beat_times.size < _MIN_BEATS_FOR_DOWNBEAT or onset_env.size == 0:
        return np.array([]), 0.0, None, None, 0.0

    beat_strength = np.interp(beat_times, onset_env_times, onset_env)
    overall_std = float(np.std(beat_strength)) + 1e-9

    scored: dict[int, tuple[int, float]] = {}
    for beats_per_bar in _CANDIDATE_BEATS_PER_BAR:
        best_phase, best_score = 0, -np.inf
        for phase in range(beats_per_bar):
            idx = np.arange(phase, beat_strength.size, beats_per_bar)
            if idx.size < 2:
                continue
            other_idx = np.setdiff1d(np.arange(beat_strength.size), idx)
            if other_idx.size < 2:
                continue
            score = float(beat_strength[idx].mean() - beat_strength[other_idx].mean())
            if score > best_score:
                best_phase, best_score = phase, score
        scored[beats_per_bar] = (best_phase, best_score)

    best_bpb = max(scored, key=lambda k: scored[k][1])
    best_phase, best_score = scored[best_bpb]
    other_bpb = [b for b in _CANDIDATE_BEATS_PER_BAR if b != best_bpb][0]
    other_score = scored[other_bpb][1]

    downbeat_confidence = float(np.clip(best_score / overall_std, 0.0, 1.0))
    ts_confidence = float(
        np.clip((best_score - other_score) / (abs(best_score) + abs(other_score) + 1e-9), 0.0, 1.0)
    )

    downbeat_idx = np.arange(best_phase, beat_times.size, best_bpb)
    downbeat_times = beat_times[downbeat_idx]

    time_signature = f"{best_bpb}/4" if ts_confidence >= 0.2 and downbeat_confidence >= 0.15 else None
    return downbeat_times, downbeat_confidence, best_bpb, time_signature, ts_confidence


def _strong_rhythmic_events(
    onset_times: np.ndarray, onset_env: np.ndarray, onset_env_times: np.ndarray
) -> list[RhythmicEvent]:
    if onset_times.size == 0:
        return []
    strengths = np.interp(onset_times, onset_env_times, onset_env)
    max_strength = float(np.max(strengths)) if strengths.size else 0.0
    if max_strength <= 1e-9:
        return []
    normalized = strengths / max_strength
    threshold = float(np.percentile(normalized, 75))
    return [
        RhythmicEvent(time=float(t), strength=float(s))
        for t, s in zip(onset_times, normalized)
        if s >= threshold
    ]


def analyze_tempo_beat(y: np.ndarray, sr: int) -> TempoBeatFeatures:
    duration = y.size / sr if sr else 0.0
    if duration < _MIN_DURATION_FOR_RHYTHM_SEC or float(np.max(np.abs(y))) < 1e-6:
        return _empty_result()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onset_env_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)

    if not np.any(onset_env > 1e-9):
        return _empty_result()

    tempo_raw, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units="frames"
    )
    bpm = float(np.atleast_1d(tempo_raw)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    beat_intervals = np.diff(beat_times)

    bpm_confidence = _tempo_confidence(onset_env, sr, bpm)
    tempo_stability = _tempo_stability(beat_intervals)

    local_tempo = librosa_rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, aggregate=None)
    tempo_curve_times = onset_env_times[: len(local_tempo)]
    tempo_change_events = _tempo_change_events(tempo_curve_times, local_tempo, global_confidence=bpm_confidence)

    downbeat_times, downbeat_confidence, beats_per_bar, time_signature, ts_confidence = _estimate_downbeats(
        beat_times, onset_env, onset_env_times
    )

    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units="time", backtrack=False
    )
    onset_density_times, onset_density = windowed_rate(onset_times, duration, window_sec=4.0, hop_sec=1.0)

    y_percussive = librosa.effects.percussive(y, margin=3.0)
    percussive_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=HOP_LENGTH)
    percussive_times = librosa.frames_to_time(np.arange(len(percussive_env)), sr=sr, hop_length=HOP_LENGTH)
    max_perc = float(np.max(percussive_env)) if percussive_env.size else 0.0
    percussive_norm = percussive_env / max_perc if max_perc > 1e-9 else percussive_env

    strong_events = _strong_rhythmic_events(onset_times, onset_env, onset_env_times)

    return TempoBeatFeatures(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beat_times=beat_times,
        beat_intervals=beat_intervals,
        tempo_stability=tempo_stability,
        tempo_curve_times=tempo_curve_times,
        tempo_curve_bpm=local_tempo,
        tempo_change_events=tempo_change_events,
        downbeat_times=downbeat_times,
        downbeat_confidence=downbeat_confidence,
        beats_per_bar=beats_per_bar,
        time_signature=time_signature,
        time_signature_confidence=ts_confidence,
        onset_times=onset_times,
        onset_strength_times=onset_env_times,
        onset_strength=onset_env,
        onset_density_times=onset_density_times,
        onset_density=onset_density,
        percussive_activity_times=percussive_times,
        percussive_activity=percussive_norm,
        strong_rhythmic_events=strong_events,
    )
