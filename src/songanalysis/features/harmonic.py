"""Harmonic analysis: chroma representation and musical key estimation.

Key detection uses the classic Krumhansl-Schmuckler key-profile correlation
against a chroma vector -- a well-understood, dependency-light technique
(just numpy) rather than a trained key-detection model. It is a heuristic:
confidence is the normalized margin between the best- and second-best
matching key, not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

HOP_LENGTH = 2048  # coarser hop than rhythm features; harmony changes slowly
_MIN_DURATION_FOR_KEY_SEC = 3.0

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (Krumhansl & Kessler, 1982).
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass(frozen=True)
class KeyCandidate:
    key: str  # e.g. "C major"
    root: str
    mode: str
    correlation: float


@dataclass(frozen=True)
class KeyChangeEvent:
    time: float
    from_key: str | None
    to_key: str | None
    confidence: float


@dataclass(frozen=True)
class HarmonicFeatures:
    chroma_times: np.ndarray
    chroma: np.ndarray  # shape (n_frames, 12)

    key: str | None  # e.g. "C major"
    key_root: str | None
    key_mode: str | None
    key_confidence: float
    key_candidates: list[KeyCandidate]

    key_curve_times: np.ndarray
    key_curve: list[str | None]
    key_change_events: list[KeyChangeEvent] = field(default_factory=list)


def _correlate_key(chroma_vector: np.ndarray) -> list[KeyCandidate]:
    if float(np.sum(chroma_vector)) <= 1e-9:
        return []
    candidates = []
    for shift in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, shift)
            corr = float(np.corrcoef(chroma_vector, rotated)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            root = _PITCH_CLASSES[shift]
            candidates.append(KeyCandidate(key=f"{root} {mode}", root=root, mode=mode, correlation=corr))
    candidates.sort(key=lambda c: -c.correlation)
    return candidates


def _key_confidence(candidates: list[KeyCandidate]) -> float:
    if len(candidates) < 2:
        return 0.0
    best, second = candidates[0].correlation, candidates[1].correlation
    worst = candidates[-1].correlation
    spread = best - worst
    if spread <= 1e-9:
        return 0.0
    return float(np.clip((best - second) / spread, 0.0, 1.0))


def _windowed_key_curve(
    chroma: np.ndarray, chroma_times: np.ndarray, sr: int, *, window_sec: float = 8.0, hop_sec: float = 4.0
) -> tuple[np.ndarray, list[str | None], list[float]]:
    if chroma.shape[0] == 0:
        return np.array([]), [], []
    frame_sec = HOP_LENGTH / sr
    window_frames = max(1, int(window_sec / frame_sec))
    hop_frames = max(1, int(hop_sec / frame_sec))

    times, keys, confidences = [], [], []
    for start in range(0, max(chroma.shape[0] - window_frames, 0) + 1, hop_frames):
        window = chroma[start : start + window_frames]
        if window.shape[0] == 0:
            continue
        mean_vec = window.mean(axis=0)
        candidates = _correlate_key(mean_vec)
        center_idx = min(start + window_frames // 2, chroma_times.size - 1)
        times.append(float(chroma_times[center_idx]))
        if candidates:
            keys.append(candidates[0].key)
            confidences.append(_key_confidence(candidates))
        else:
            keys.append(None)
            confidences.append(0.0)
    return np.array(times), keys, confidences


def _key_change_events(
    times: np.ndarray,
    keys: list[str | None],
    confidences: list[float],
    *,
    global_confidence: float,
    min_run: int = 2,
) -> list[KeyChangeEvent]:
    if len(keys) < min_run * 2:
        return []

    # Run-length encode the windowed key curve to ignore single-window flicker.
    runs: list[tuple[str | None, int, int]] = []  # (key, start_idx, end_idx_exclusive)
    i = 0
    while i < len(keys):
        j = i
        while j < len(keys) and keys[j] == keys[i]:
            j += 1
        runs.append((keys[i], i, j))
        i = j

    stable_runs = [r for r in runs if (r[2] - r[1]) >= min_run and r[0] is not None]
    events = []
    for prev_run, next_run in zip(stable_runs, stable_runs[1:]):
        if prev_run[0] == next_run[0]:
            continue
        change_idx = next_run[1]
        # A per-window key correlation margin only says "this window looks more
        # like key X than Y"; it says nothing about whether key detection is
        # trustworthy for this song at all. Scale by the song's overall key
        # confidence too, so a song the analyzer can't confidently key at all
        # doesn't still emit a string of nominally-plausible key-change events
        # (mirrors the same gating applied to tempo_change_events).
        window_conf = float(np.mean(confidences[next_run[1] : next_run[2]]))
        conf = float(np.clip(window_conf * global_confidence, 0.0, 1.0))
        if conf < 0.1:
            continue
        events.append(
            KeyChangeEvent(time=float(times[change_idx]), from_key=prev_run[0], to_key=next_run[0], confidence=conf)
        )
    return events


def analyze_harmonic(y: np.ndarray, sr: int) -> HarmonicFeatures:
    duration = y.size / sr if sr else 0.0
    empty_arr = np.array([])
    if duration < _MIN_DURATION_FOR_KEY_SEC or float(np.max(np.abs(y))) < 1e-6:
        return HarmonicFeatures(
            chroma_times=empty_arr,
            chroma=np.zeros((0, 12)),
            key=None,
            key_root=None,
            key_mode=None,
            key_confidence=0.0,
            key_candidates=[],
            key_curve_times=empty_arr,
            key_curve=[],
            key_change_events=[],
        )

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH).T  # (n_frames, 12)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[0]), sr=sr, hop_length=HOP_LENGTH)

    global_profile = chroma.mean(axis=0)
    candidates = _correlate_key(global_profile)
    confidence = _key_confidence(candidates)
    top = candidates[0] if candidates else None

    key_curve_times, key_curve, key_curve_conf = _windowed_key_curve(chroma, chroma_times, sr)
    key_change_events = _key_change_events(key_curve_times, key_curve, key_curve_conf, global_confidence=confidence)

    return HarmonicFeatures(
        chroma_times=chroma_times,
        chroma=chroma,
        key=top.key if top else None,
        key_root=top.root if top else None,
        key_mode=top.mode if top else None,
        key_confidence=confidence,
        key_candidates=candidates[:3],
        key_curve_times=key_curve_times,
        key_curve=key_curve,
        key_change_events=key_change_events,
    )
