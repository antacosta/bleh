"""Stage: song structure segmentation.

Boundary detection uses Foote's (2000) checkerboard-kernel novelty method
on a self-similarity matrix of timbre+harmony features -- a standard,
dependency-light (numpy/scipy/librosa only) technique for finding *where*
sections change, without requiring a trained model or claiming to know
*what* each section is.

Section labels are intentionally two-tier:
  - a neutral, always-present ``id`` ("section_1", "section_2", ...)
  - an optional ``heuristic_label`` (e.g. "chorus", "intro") that is only a
    best-effort guess from position/energy/vocal/repetition patterns, with
    an explicit, capped-low confidence. Consumers that only trust reliable
    boundaries should use ``id``; consumers willing to accept a heuristic
    guess can use ``heuristic_label`` alongside its confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from songanalysis.features.energy import EnergyTimeline
from songanalysis.features.vocals import VocalTimeline
from songanalysis.util.timeseries import find_local_extrema

GRID_HOP_SEC = 1.0
MIN_SECTION_SEC = 6.0
BEAT_SNAP_TOLERANCE_SEC = 1.0
REPETITION_SIMILARITY_THRESHOLD = 0.80
_MFCC_HOP = 2048


@dataclass(frozen=True)
class Section:
    id: str
    start: float
    end: float
    duration: float
    heuristic_label: str | None
    heuristic_label_confidence: float
    energy: float
    rhythmic_density: float
    vocal_presence: float
    dominant_timbre: str
    repetition_group: str | None
    boundary_confidence: float


@dataclass(frozen=True)
class RepetitionGroup:
    id: str
    section_ids: list[str]
    mean_similarity: float


@dataclass(frozen=True)
class StructuralTransition:
    time: float
    from_section: str
    to_section: str
    energy_delta: float
    magnitude: float


@dataclass(frozen=True)
class StructureAnalysis:
    sections: list[Section]
    repetitions: list[RepetitionGroup]
    major_transitions: list[StructuralTransition]
    boundary_times: np.ndarray
    novelty_curve_times: np.ndarray
    novelty_curve: np.ndarray
    method: str = "foote-checkerboard-novelty(chroma+mfcc)"


def _zscore_rows(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return (features - mean) / std


def _grid_features(
    y: np.ndarray, sr: int, grid: np.ndarray, *, chroma: np.ndarray | None = None, chroma_times: np.ndarray | None = None
) -> np.ndarray:
    # Reuse the harmonic-analysis chroma (same hop length) when available instead
    # of recomputing an identical CQT chroma from scratch.
    if chroma is None or chroma_times is None or chroma.shape[0] == 0:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_MFCC_HOP)
        chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=_MFCC_HOP)
        chroma = chroma.T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=_MFCC_HOP).T[:, 1:]  # drop c0 (energy)
    n = min(chroma.shape[0], mfcc.shape[0])
    chroma, mfcc = chroma[:n], mfcc[:n]
    frame_times = chroma_times[:n]

    bucket_idx = np.clip(np.searchsorted(grid, frame_times, side="right") - 1, 0, grid.size - 1)
    n_grid = grid.size
    chroma_g = np.zeros((n_grid, chroma.shape[1]))
    mfcc_g = np.zeros((n_grid, mfcc.shape[1]))
    counts = np.zeros(n_grid)
    for i, b in enumerate(bucket_idx):
        chroma_g[b] += chroma[i]
        mfcc_g[b] += mfcc[i]
        counts[b] += 1
    counts[counts == 0] = 1
    chroma_g /= counts[:, None]
    mfcc_g /= counts[:, None]

    return np.concatenate([_zscore_rows(chroma_g), _zscore_rows(mfcc_g)], axis=1)


def _self_similarity(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = features / norms
    return normalized @ normalized.T


def _checkerboard_kernel(half_width: int) -> np.ndarray:
    size = 2 * half_width + 1
    axis = np.arange(-half_width, half_width + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sign = np.ones((size, size))
    sign[: half_width + 1, half_width:] = -1
    sign[half_width:, : half_width + 1] = -1
    sign[: half_width + 1, : half_width + 1] = 1
    sign[half_width:, half_width:] = 1
    taper = np.exp(-(xx**2 + yy**2) / (2 * (half_width / 1.5) ** 2))
    return sign * taper


def _novelty_curve(similarity: np.ndarray, half_width: int) -> np.ndarray:
    n = similarity.shape[0]
    kernel = _checkerboard_kernel(half_width)
    novelty = np.zeros(n)
    for i in range(half_width, n - half_width):
        block = similarity[i - half_width : i + half_width + 1, i - half_width : i + half_width + 1]
        novelty[i] = float(np.sum(block * kernel))
    return novelty


def _snap_to_beat(time: float, beat_times: np.ndarray) -> float:
    if beat_times.size == 0:
        return time
    idx = int(np.argmin(np.abs(beat_times - time)))
    if abs(beat_times[idx] - time) <= BEAT_SNAP_TOLERANCE_SEC:
        return float(beat_times[idx])
    return time


def _interp_mean(times: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    if times.size == 0 or values.size == 0:
        return 0.0
    mask = (times >= start) & (times < end)
    if np.any(mask):
        return float(np.mean(values[mask]))
    return float(np.interp((start + end) / 2, times, values))


def _dominant_timbre(bass_z: float, spectral_z: float) -> str:
    if bass_z > 0.5 and bass_z >= spectral_z:
        return "bass-heavy"
    if spectral_z > 0.5 and spectral_z > bass_z:
        return "bright"
    return "balanced"


def _guess_label(
    *, is_first: bool, is_last: bool, energy: float, median_energy: float, vocal_presence: float, is_repeated: bool
) -> tuple[str | None, float]:
    low_vocal = vocal_presence < 0.15
    low_energy = energy < median_energy

    if is_first and (low_vocal or low_energy):
        return "intro", 0.5 if (low_vocal and low_energy) else 0.35
    if is_last and (low_vocal or low_energy):
        return "outro", 0.5 if (low_vocal and low_energy) else 0.35
    if low_vocal and low_energy:
        return "breakdown", 0.4
    if is_repeated and energy >= median_energy:
        return "chorus", 0.45
    if is_repeated and not low_vocal:
        return "verse", 0.35
    return None, 0.0


def analyze_structure(
    y: np.ndarray,
    sr: int,
    *,
    energy_timeline: EnergyTimeline,
    vocal_timeline: VocalTimeline,
    beat_times: np.ndarray,
    chroma: np.ndarray | None = None,
    chroma_times: np.ndarray | None = None,
) -> StructureAnalysis:
    duration = y.size / sr if sr else 0.0
    empty = np.array([])

    if duration < MIN_SECTION_SEC * 2 or float(np.max(np.abs(y))) < 1e-6:
        section = Section(
            id="section_1",
            start=0.0,
            end=duration,
            duration=duration,
            heuristic_label=None,
            heuristic_label_confidence=0.0,
            energy=energy_timeline.overall_energy,
            rhythmic_density=float(np.mean(energy_timeline.onset_density)) if energy_timeline.onset_density.size else 0.0,
            vocal_presence=vocal_timeline.vocal_density,
            dominant_timbre="balanced",
            repetition_group=None,
            boundary_confidence=0.0,
        )
        return StructureAnalysis(
            sections=[section],
            repetitions=[],
            major_transitions=[],
            boundary_times=np.array([0.0, duration]),
            novelty_curve_times=empty,
            novelty_curve=empty,
            method="too_short_for_segmentation",
        )

    grid = np.arange(0.0, duration, GRID_HOP_SEC)
    features = _grid_features(y, sr, grid, chroma=chroma, chroma_times=chroma_times)
    similarity = _self_similarity(features)

    half_width = int(np.clip(round(6.0 / GRID_HOP_SEC), 4, max(4, grid.size // 4)))
    novelty = _novelty_curve(similarity, half_width)

    peaks, _ = find_local_extrema(
        novelty, grid, min_prominence_frac=0.15, min_separation_sec=MIN_SECTION_SEC
    )
    boundary_times = sorted({0.0, duration, *(p.time for p in peaks)})

    merged = [boundary_times[0]]
    for t in boundary_times[1:]:
        if t - merged[-1] >= MIN_SECTION_SEC or t == duration:
            merged.append(t)
    if merged[-1] != duration:
        merged[-1] = duration
    boundary_times = merged

    if beat_times.size:
        boundary_times = [boundary_times[0]] + [
            _snap_to_beat(t, beat_times) for t in boundary_times[1:-1]
        ] + [boundary_times[-1]]
        boundary_times = sorted(set(boundary_times))

    novelty_max = float(max(np.max(novelty), 1e-9)) if novelty.size else 1e-9
    novelty_at = {round(t, 3): 0.0 for t in boundary_times}
    for t in boundary_times[1:-1]:
        idx = int(np.argmin(np.abs(grid - t)))
        novelty_at[round(t, 3)] = float(np.clip(novelty[idx] / novelty_max, 0.0, 1.0))

    energies = [
        _interp_mean(energy_timeline.times, energy_timeline.composite_energy, s, e)
        for s, e in zip(boundary_times[:-1], boundary_times[1:])
    ]
    median_energy = float(np.median(energies)) if energies else 0.0

    bass_series = energy_timeline.bass_energy
    spec_series = energy_timeline.spectral_energy
    bass_mean, bass_std = (float(np.mean(bass_series)), float(np.std(bass_series))) if bass_series.size else (0.0, 1.0)
    spec_mean, spec_std = (float(np.mean(spec_series)), float(np.std(spec_series))) if spec_series.size else (0.0, 1.0)
    bass_std = bass_std or 1.0
    spec_std = spec_std or 1.0

    raw_sections = []
    section_features = []
    for i, (start, end) in enumerate(zip(boundary_times[:-1], boundary_times[1:])):
        energy = energies[i]
        rhythmic_density = _interp_mean(energy_timeline.times, energy_timeline.onset_density, start, end)
        vocal_presence = _interp_mean(vocal_timeline.times, vocal_timeline.vocal_probability, start, end)
        bass_val = _interp_mean(energy_timeline.times, bass_series, start, end)
        spec_val = _interp_mean(energy_timeline.times, spec_series, start, end)
        timbre = _dominant_timbre((bass_val - bass_mean) / bass_std, (spec_val - spec_mean) / spec_std)

        grid_mask = (grid >= start) & (grid < end)
        section_features.append(features[grid_mask].mean(axis=0) if np.any(grid_mask) else np.zeros(features.shape[1]))

        raw_sections.append(
            dict(
                start=start,
                end=end,
                energy=energy,
                rhythmic_density=rhythmic_density,
                vocal_presence=vocal_presence,
                dominant_timbre=timbre,
                boundary_confidence=1.0 if i == 0 else float(np.clip(novelty_at.get(round(start, 3), 0.0), 0.0, 1.0)),
            )
        )

    section_feature_matrix = np.array(section_features)
    n_sections = len(raw_sections)
    repetition_group_of: list[str | None] = [None] * n_sections
    repetitions: list[RepetitionGroup] = []

    if n_sections >= 2:
        norms = np.linalg.norm(section_feature_matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        normed = section_feature_matrix / norms
        sim = normed @ normed.T

        visited = [False] * n_sections
        group_idx = 1
        for i in range(n_sections):
            if visited[i]:
                continue
            members = [i]
            for j in range(i + 1, n_sections):
                if not visited[j] and sim[i, j] >= REPETITION_SIMILARITY_THRESHOLD:
                    members.append(j)
            if len(members) >= 2:
                for m in members:
                    visited[m] = True
                    repetition_group_of[m] = f"rep_{group_idx}"
                pair_sims = [sim[a, b] for a in members for b in members if a < b]
                repetitions.append(
                    RepetitionGroup(
                        id=f"rep_{group_idx}",
                        section_ids=[f"section_{m + 1}" for m in members],
                        mean_similarity=float(np.mean(pair_sims)) if pair_sims else 0.0,
                    )
                )
                group_idx += 1

    sections: list[Section] = []
    for i, raw in enumerate(raw_sections):
        label, label_conf = _guess_label(
            is_first=(i == 0),
            is_last=(i == n_sections - 1),
            energy=raw["energy"],
            median_energy=median_energy,
            vocal_presence=raw["vocal_presence"],
            is_repeated=repetition_group_of[i] is not None,
        )
        sections.append(
            Section(
                id=f"section_{i + 1}",
                start=raw["start"],
                end=raw["end"],
                duration=raw["end"] - raw["start"],
                heuristic_label=label,
                heuristic_label_confidence=label_conf,
                energy=raw["energy"],
                rhythmic_density=raw["rhythmic_density"],
                vocal_presence=raw["vocal_presence"],
                dominant_timbre=raw["dominant_timbre"],
                repetition_group=repetition_group_of[i],
                boundary_confidence=raw["boundary_confidence"],
            )
        )

    transitions: list[StructuralTransition] = []
    for a, b in zip(sections, sections[1:]):
        energy_delta = b.energy - a.energy
        vocal_delta = b.vocal_presence - a.vocal_presence
        rhythm_delta = b.rhythmic_density - a.rhythmic_density
        rhythm_scale = max(1.0, float(np.mean(energy_timeline.onset_density)) if energy_timeline.onset_density.size else 1.0)
        magnitude = float(
            np.clip(0.5 * abs(energy_delta) + 0.3 * abs(vocal_delta) + 0.2 * abs(rhythm_delta) / rhythm_scale, 0.0, 1.0)
        )
        if magnitude >= 0.15:
            transitions.append(
                StructuralTransition(
                    time=b.start, from_section=a.id, to_section=b.id, energy_delta=energy_delta, magnitude=magnitude
                )
            )

    return StructureAnalysis(
        sections=sections,
        repetitions=repetitions,
        major_transitions=transitions,
        boundary_times=np.array(boundary_times),
        novelty_curve_times=grid,
        novelty_curve=novelty,
    )
