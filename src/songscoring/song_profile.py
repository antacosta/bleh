"""Turns a song-analysis JSON document into the typed view the scorer needs.

This is deliberately a thin extraction layer, not a second analysis pass:
every number here already exists in the analysis JSON produced by
``songanalysis``. The only "computation" done here is trivial aggregation
(e.g. averaging a chroma matrix into one vector, slicing a time series to
its first/last N seconds) -- nothing that reanalyzes audio or invents new
musical facts.

Missing/optional fields degrade to ``None`` or empty collections rather
than raising, since real analysis JSON varies (an old cache entry, a very
short or silent song, a file with no embedded tags).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _arr(values: Any) -> np.ndarray:
    if not values:
        return np.array([])
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class SectionSummary:
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


@dataclass(frozen=True)
class VocalRegion:
    start: float
    end: float
    kind: str  # "vocal" | "instrumental"


@dataclass(frozen=True)
class MixPointRef:
    time: float
    reasons: tuple[str, ...]
    confidence: float
    section_id: str | None


@dataclass(frozen=True)
class LoopRef:
    start: float
    end: float
    bars: int
    confidence: float


@dataclass(frozen=True)
class EventRef:
    time: float
    type: str
    confidence: float
    details: dict[str, Any]


@dataclass(frozen=True)
class SongProfile:
    """Everything the scoring layer is allowed to look at for one song."""

    id: str
    source_path: str | None

    # -- metadata --
    filename: str | None
    artist: str | None
    title: str | None
    album: str | None
    genre: str | None
    release_year: int | None
    duration_sec: float

    # -- global / rhythm --
    bpm: float | None
    bpm_confidence: float
    time_signature: str | None
    time_signature_confidence: float
    beats_per_bar: int | None
    tempo_stability: float
    downbeat_confidence: float
    downbeat_times: np.ndarray

    # -- harmonic --
    key: str | None
    key_root: str | None
    key_mode: str | None
    key_confidence: float
    chroma_mean: np.ndarray | None  # shape (12,) or None

    # -- loudness --
    integrated_lufs: float | None

    # -- energy timeline (per-song-relative composite_energy; see energy.py) --
    energy_times: np.ndarray
    composite_energy: np.ndarray
    overall_energy: float
    energy_trend: float

    # -- rhythm timeline --
    onset_density_times: np.ndarray
    onset_density: np.ndarray
    percussive_activity_times: np.ndarray
    percussive_activity: np.ndarray

    # -- vocals --
    vocal_regions: tuple[VocalRegion, ...]
    vocal_times: np.ndarray
    vocal_probability: np.ndarray
    vocal_density: float

    # -- structure --
    sections: tuple[SectionSummary, ...]

    # -- dj affordances --
    mix_in_points: tuple[MixPointRef, ...]
    mix_out_points: tuple[MixPointRef, ...]
    loop_candidates: tuple[LoopRef, ...]
    events: tuple[EventRef, ...]

    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    # ---- convenience windows, used by several components ----

    def energy_in_window(self, start: float, end: float) -> float | None:
        """Mean composite_energy in ``[start, end)``; ``None`` if no samples
        fall in that window (e.g. the song is shorter than the window)."""
        if self.energy_times.size == 0:
            return None
        mask = (self.energy_times >= start) & (self.energy_times < end)
        if not np.any(mask):
            return None
        return float(np.mean(self.composite_energy[mask]))

    def starting_energy(self, window_sec: float) -> float | None:
        return self.energy_in_window(0.0, window_sec)

    def ending_energy(self, window_sec: float, from_position: float | None = None) -> float | None:
        """Energy of the song's tail, i.e. where it's *about to end*. If
        ``from_position`` (current playback position) is given and is later
        than ``duration - window_sec``, the window starts there instead, so
        an already-in-progress song is judged by what's actually left of it."""
        default_start = max(0.0, self.duration_sec - window_sec)
        start = max(default_start, from_position) if from_position is not None else default_start
        return self.energy_in_window(start, self.duration_sec + 1e-6)

    def mean_onset_density(self) -> float | None:
        return float(np.mean(self.onset_density)) if self.onset_density.size else None

    def mean_percussive_activity(self) -> float | None:
        return float(np.mean(self.percussive_activity)) if self.percussive_activity.size else None

    def vocal_presence_in_window(self, start: float, end: float) -> float | None:
        if self.vocal_times.size == 0:
            return None
        mask = (self.vocal_times >= start) & (self.vocal_times < end)
        if not np.any(mask):
            return None
        return float(np.mean(self.vocal_probability[mask]))

    def timbre_distribution(self) -> dict[str, float]:
        """Fraction of total section duration in each dominant_timbre label."""
        totals: dict[str, float] = {}
        total_duration = 0.0
        for section in self.sections:
            totals[section.dominant_timbre] = totals.get(section.dominant_timbre, 0.0) + section.duration
            total_duration += section.duration
        if total_duration <= 0:
            return {}
        return {label: value / total_duration for label, value in totals.items()}

    def best_mix_out(self, after_position: float | None = None) -> MixPointRef | None:
        candidates = self.mix_out_points
        if after_position is not None:
            candidates = tuple(p for p in candidates if p.time >= after_position)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.confidence)

    def best_mix_in(self) -> MixPointRef | None:
        if not self.mix_in_points:
            return None
        return max(self.mix_in_points, key=lambda p: p.confidence)


def _section_from_dict(d: dict[str, Any]) -> SectionSummary:
    return SectionSummary(
        id=d.get("id", "section_1"),
        start=float(d.get("start", 0.0)),
        end=float(d.get("end", 0.0)),
        duration=float(d.get("duration", 0.0)),
        heuristic_label=d.get("heuristic_label"),
        heuristic_label_confidence=float(d.get("heuristic_label_confidence", 0.0)),
        energy=float(d.get("energy", 0.0)),
        rhythmic_density=float(d.get("rhythmic_density", 0.0)),
        vocal_presence=float(d.get("vocal_presence", 0.0)),
        dominant_timbre=d.get("dominant_timbre", "balanced"),
        repetition_group=d.get("repetition_group"),
    )


def _mix_point_from_dict(d: dict[str, Any]) -> MixPointRef:
    return MixPointRef(
        time=float(d.get("time", 0.0)),
        reasons=tuple(d.get("reasons") or ()),
        confidence=float(d.get("confidence", 0.0)),
        section_id=d.get("section_id"),
    )


def _loop_from_dict(d: dict[str, Any]) -> LoopRef:
    return LoopRef(
        start=float(d.get("start", 0.0)),
        end=float(d.get("end", 0.0)),
        bars=int(d.get("bars", 0)),
        confidence=float(d.get("confidence", 0.0)),
    )


def _event_from_dict(d: dict[str, Any]) -> EventRef:
    return EventRef(
        time=float(d.get("time", 0.0)),
        type=d.get("type", "unknown"),
        confidence=float(d.get("confidence", 0.0)),
        details=d.get("details") or {},
    )


def _vocal_region_from_dict(d: dict[str, Any]) -> VocalRegion:
    return VocalRegion(start=float(d.get("start", 0.0)), end=float(d.get("end", 0.0)), kind=d.get("kind", "instrumental"))


def profile_from_analysis(analysis: dict[str, Any], *, id: str | None = None, source_path: str | None = None) -> SongProfile:
    """Build a :class:`SongProfile` from a songanalysis JSON document
    (already-parsed dict, matching ``SongAnalysis.to_jsonable()``)."""
    metadata = analysis.get("metadata") or {}
    glob = analysis.get("global") or {}
    timeline = analysis.get("timeline") or {}
    energy = timeline.get("energy") or {}
    vocals = timeline.get("vocals") or {}
    structure = analysis.get("structure") or {}
    harmonic = analysis.get("harmonic") or {}
    rhythm = analysis.get("rhythm") or {}
    dj = analysis.get("dj_affordances") or {}

    chroma = harmonic.get("chroma") or []
    chroma_mean = np.mean(np.asarray(chroma, dtype=np.float64), axis=0) if chroma else None

    filename = metadata.get("filename")
    resolved_id = id or filename or source_path or "unknown"

    return SongProfile(
        id=resolved_id,
        source_path=source_path,
        filename=filename,
        artist=metadata.get("artist"),
        title=metadata.get("title"),
        album=metadata.get("album"),
        genre=metadata.get("genre"),
        release_year=metadata.get("release_year"),
        duration_sec=float(metadata.get("duration_sec") or glob.get("duration_sec") or 0.0),
        bpm=glob.get("bpm"),
        bpm_confidence=float(glob.get("bpm_confidence", 0.0)),
        time_signature=glob.get("time_signature"),
        time_signature_confidence=float(glob.get("time_signature_confidence", 0.0)),
        beats_per_bar=rhythm.get("beats_per_bar"),
        tempo_stability=float(rhythm.get("tempo_stability", 0.0)),
        downbeat_confidence=float(rhythm.get("downbeat_confidence", 0.0)),
        downbeat_times=_arr(rhythm.get("downbeat_times")),
        key=glob.get("key"),
        key_root=harmonic.get("key_root"),
        key_mode=harmonic.get("key_mode"),
        key_confidence=float(glob.get("key_confidence", 0.0)),
        chroma_mean=chroma_mean,
        integrated_lufs=glob.get("integrated_lufs"),
        energy_times=_arr(energy.get("times")),
        composite_energy=_arr(energy.get("composite_energy")),
        overall_energy=float(glob.get("overall_energy", energy.get("overall_energy", 0.0)) or 0.0),
        energy_trend=float(glob.get("energy_trend", energy.get("energy_trend", 0.0)) or 0.0),
        onset_density_times=_arr(rhythm.get("onset_density_times")),
        onset_density=_arr(rhythm.get("onset_density")),
        percussive_activity_times=_arr(rhythm.get("percussive_activity_times")),
        percussive_activity=_arr(rhythm.get("percussive_activity")),
        vocal_regions=tuple(_vocal_region_from_dict(r) for r in (vocals.get("regions") or [])),
        vocal_times=_arr(vocals.get("times")),
        vocal_probability=_arr(vocals.get("probability")),
        vocal_density=float(vocals.get("density", 0.0)),
        sections=tuple(_section_from_dict(s) for s in (structure.get("sections") or [])),
        mix_in_points=tuple(_mix_point_from_dict(p) for p in (dj.get("mix_in_points") or [])),
        mix_out_points=tuple(_mix_point_from_dict(p) for p in (dj.get("mix_out_points") or [])),
        loop_candidates=tuple(_loop_from_dict(l) for l in (dj.get("loop_candidates") or [])),
        events=tuple(_event_from_dict(e) for e in (dj.get("events") or [])),
        raw=analysis,
    )


def profile_from_json_file(path: str | Path, *, id: str | None = None) -> SongProfile:
    path = Path(path)
    with open(path) as f:
        analysis = json.load(f)
    return profile_from_analysis(analysis, id=id or path.stem, source_path=str(path))
