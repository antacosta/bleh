"""Stage: DJ-relevant mix-in/mix-out affordance detection.

Purely derived from the feature/structure stages already computed --
no new audio decoding happens here. Every candidate carries a heuristic
confidence and the reasons that produced it, so a future DJ-planning layer
can apply its own thresholds instead of trusting a single score blindly.

Explicitly out of scope here (belongs to a later layer): choosing *which*
points to actually use, scoring compatibility between two songs, or
performing any audio manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from songanalysis.features.energy import EnergyTimeline
from songanalysis.features.harmonic import HarmonicFeatures
from songanalysis.features.tempo_beat import TempoBeatFeatures
from songanalysis.features.vocals import VocalTimeline
from songanalysis.structure.segmentation import StructureAnalysis

LOW_VOCAL_DENSITY_THRESHOLD = 0.2
DOWNBEAT_SNAP_TOLERANCE_SEC = 1.0
DROP_MIN_MAGNITUDE = 0.08
DROP_LOOKBACK_SEC = 6.0
LOOP_BAR_LENGTHS = (4, 8, 16)
LOOP_MIN_CONFIDENCE = 0.55
LOOP_MAX_CANDIDATES = 20
EXTENDED_SECTION_MIN_SEC = 20.0
ABRUPT_ENDING_WINDOW_SEC = 0.3
ABRUPT_ENDING_REF_SEC = 1.5


@dataclass(frozen=True)
class MixPoint:
    time: float
    kind: str  # "mix_in" | "mix_out"
    reasons: list[str]
    confidence: float
    section_id: str | None


@dataclass(frozen=True)
class LoopCandidate:
    start: float
    end: float
    bars: int
    beats_per_bar: int
    confidence: float
    section_id: str | None


@dataclass(frozen=True)
class DJEvent:
    time: float
    type: str
    confidence: float
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DJAffordances:
    mix_in_points: list[MixPoint]
    mix_out_points: list[MixPoint]
    loop_candidates: list[LoopCandidate]
    events: list[DJEvent]


def _nearest_downbeat(time: float, downbeat_times: np.ndarray, beat_times: np.ndarray) -> float:
    grid = downbeat_times if downbeat_times.size else beat_times
    if grid.size == 0:
        return time
    idx = int(np.argmin(np.abs(grid - time)))
    return float(grid[idx]) if abs(grid[idx] - time) <= DOWNBEAT_SNAP_TOLERANCE_SEC else time


def _section_reasons_mix_in(section, is_first: bool) -> tuple[list[str], float]:
    reasons = []
    weight = 0.0
    if is_first and section.heuristic_label == "intro":
        reasons.append("instrumental_intro")
        weight += 0.4 * section.heuristic_label_confidence + 0.2
    if section.vocal_presence < LOW_VOCAL_DENSITY_THRESHOLD:
        reasons.append("low_vocal_density")
        weight += 0.3
    if section.heuristic_label == "breakdown":
        reasons.append("breakdown")
        weight += 0.3
    if section.repetition_group is not None:
        reasons.append("repeated_phrase")
        weight += 0.2
    return reasons, weight


def _section_reasons_mix_out(section, is_last: bool) -> tuple[list[str], float]:
    reasons = []
    weight = 0.0
    if is_last and section.heuristic_label == "outro":
        reasons.append("outro")
        weight += 0.4 * section.heuristic_label_confidence + 0.2
    if section.vocal_presence < LOW_VOCAL_DENSITY_THRESHOLD:
        reasons.append("low_vocal_density")
        weight += 0.3
    if section.heuristic_label == "breakdown":
        reasons.append("breakdown")
        weight += 0.3
    if section.repetition_group is not None:
        reasons.append("phrase_ending")
        weight += 0.2
    return reasons, weight


def _mix_points(
    structure: StructureAnalysis, tempo_beat: TempoBeatFeatures
) -> tuple[list[MixPoint], list[MixPoint]]:
    mix_in: list[MixPoint] = []
    mix_out: list[MixPoint] = []
    n = len(structure.sections)

    for i, section in enumerate(structure.sections):
        in_reasons, in_weight = _section_reasons_mix_in(section, is_first=(i == 0))
        if in_reasons:
            t = _nearest_downbeat(section.start, tempo_beat.downbeat_times, tempo_beat.beat_times)
            if t != section.start:
                in_reasons.append("clean_downbeat")
                in_weight += 0.1
            mix_in.append(
                MixPoint(
                    time=t,
                    kind="mix_in",
                    reasons=in_reasons,
                    confidence=float(np.clip(in_weight, 0.0, 1.0)),
                    section_id=section.id,
                )
            )

        out_reasons, out_weight = _section_reasons_mix_out(section, is_last=(i == n - 1))
        if out_reasons:
            t = _nearest_downbeat(section.end, tempo_beat.downbeat_times, tempo_beat.beat_times)
            if t != section.end:
                out_reasons.append("clean_downbeat")
                out_weight += 0.1
            mix_out.append(
                MixPoint(
                    time=t,
                    kind="mix_out",
                    reasons=out_reasons,
                    confidence=float(np.clip(out_weight, 0.0, 1.0)),
                    section_id=section.id,
                )
            )

    return mix_in, mix_out


def _section_at(structure: StructureAnalysis, time: float) -> str | None:
    for section in structure.sections:
        if section.start <= time < section.end:
            return section.id
    return None


def _chroma_half_similarity(chroma: np.ndarray, chroma_times: np.ndarray, start: float, end: float) -> float:
    if chroma.shape[0] == 0:
        return 0.0
    mask = (chroma_times >= start) & (chroma_times < end)
    idx = np.where(mask)[0]
    if idx.size < 4:
        return 0.0
    mid = idx[idx.size // 2]
    first_half = chroma[idx[0] : mid].mean(axis=0)
    second_half = chroma[mid : idx[-1] + 1].mean(axis=0)
    n1, n2 = np.linalg.norm(first_half), np.linalg.norm(second_half)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.clip(np.dot(first_half, second_half) / (n1 * n2), 0.0, 1.0))


def _loop_candidates(
    tempo_beat: TempoBeatFeatures,
    harmonic: HarmonicFeatures,
    energy_timeline: EnergyTimeline,
    structure: StructureAnalysis,
) -> list[LoopCandidate]:
    beat_times = tempo_beat.beat_times
    if beat_times.size < 8:
        return []

    beats_per_bar = tempo_beat.beats_per_bar or 4
    starts = tempo_beat.downbeat_times if tempo_beat.downbeat_times.size >= 2 else beat_times[::beats_per_bar]
    base_confidence_penalty = 0.0 if tempo_beat.downbeat_times.size >= 2 else 0.15

    candidates: list[LoopCandidate] = []
    for start_time in starts:
        start_idx = int(np.argmin(np.abs(beat_times - start_time)))
        for bars in LOOP_BAR_LENGTHS:
            loop_beats = bars * beats_per_bar
            end_idx = start_idx + loop_beats
            if end_idx >= beat_times.size:
                continue
            start_t, end_t = float(beat_times[start_idx]), float(beat_times[end_idx])

            interval_slice = tempo_beat.beat_intervals[start_idx:end_idx]
            if interval_slice.size == 0:
                continue
            mean_interval = float(np.mean(interval_slice))
            if mean_interval <= 1e-9:
                continue
            beat_consistency = float(np.clip(1.0 - (np.std(interval_slice) / mean_interval) / 0.1, 0.0, 1.0))

            energy_mask = (energy_timeline.times >= start_t) & (energy_timeline.times < end_t)
            energy_slice = energy_timeline.composite_energy[energy_mask]
            if energy_slice.size >= 2 and np.mean(energy_slice) > 1e-9:
                energy_stability = float(
                    np.clip(1.0 - (np.std(energy_slice) / (np.mean(energy_slice) + 1e-9)) / 0.6, 0.0, 1.0)
                )
            else:
                energy_stability = 0.5

            chroma_similarity = _chroma_half_similarity(harmonic.chroma, harmonic.chroma_times, start_t, end_t)

            confidence = float(
                np.clip(
                    (0.4 * beat_consistency + 0.3 * energy_stability + 0.3 * chroma_similarity)
                    - base_confidence_penalty,
                    0.0,
                    1.0,
                )
            )
            if confidence >= LOOP_MIN_CONFIDENCE:
                candidates.append(
                    LoopCandidate(
                        start=start_t,
                        end=end_t,
                        bars=bars,
                        beats_per_bar=beats_per_bar,
                        confidence=confidence,
                        section_id=_section_at(structure, start_t),
                    )
                )

    candidates.sort(key=lambda c: -c.confidence)
    return candidates[:LOOP_MAX_CANDIDATES]


def _abrupt_ending(y: np.ndarray, sr: int, duration: float) -> DJEvent | None:
    if duration < ABRUPT_ENDING_REF_SEC + ABRUPT_ENDING_WINDOW_SEC:
        return None
    tail = y[-int(ABRUPT_ENDING_WINDOW_SEC * sr) :]
    ref_start = -int((ABRUPT_ENDING_REF_SEC + ABRUPT_ENDING_WINDOW_SEC) * sr)
    ref_end = -int(ABRUPT_ENDING_WINDOW_SEC * sr)
    reference = y[ref_start:ref_end]
    tail_rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64)))) if tail.size else 0.0
    ref_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64)))) if reference.size else 0.0
    if ref_rms < 1e-4:
        return None
    if tail_rms > 0.5 * ref_rms:
        confidence = float(np.clip(tail_rms / ref_rms, 0.0, 1.0))
        return DJEvent(
            time=duration - ABRUPT_ENDING_WINDOW_SEC / 2,
            type="abrupt_ending",
            confidence=confidence,
            details={"tail_rms": tail_rms, "reference_rms": ref_rms},
        )
    return None


def _events(
    y: np.ndarray,
    sr: int,
    duration: float,
    tempo_beat: TempoBeatFeatures,
    harmonic: HarmonicFeatures,
    energy_timeline: EnergyTimeline,
    structure: StructureAnalysis,
) -> list[DJEvent]:
    events: list[DJEvent] = []

    for change in energy_timeline.major_energy_changes:
        events.append(
            DJEvent(
                time=change.time,
                type="major_energy_change",
                confidence=change.confidence,
                details={"direction": change.direction, "magnitude": change.magnitude},
            )
        )

    # A real DJ "drop" is a jump out of a genuine lull, not just any energy
    # increase that happens to sit near *a* locally-lower sample -- on tracks
    # with a fairly narrow, wobbly energy range (a lot of rap/rock/electronic
    # material), nearly every beat-to-beat swing has some technically-local
    # valley within a few seconds, which used to make nearly every
    # major_energy_change increase get relabeled "drop". Requiring the
    # preceding stretch to actually dip into the *song's own* bottom quartile
    # (not just be locally lower than its immediate neighbors) is a much
    # closer match for what "before the drop" actually sounds like.
    energy_low_threshold = (
        float(np.percentile(energy_timeline.composite_energy, 25)) if energy_timeline.composite_energy.size else 0.0
    )
    for change in energy_timeline.major_energy_changes:
        if change.direction != "increase":
            continue
        if change.magnitude < DROP_MIN_MAGNITUDE:
            continue
        lookback_mask = (energy_timeline.times >= change.time - DROP_LOOKBACK_SEC) & (
            energy_timeline.times <= change.time
        )
        preceding = energy_timeline.composite_energy[lookback_mask]
        if preceding.size == 0 or float(np.min(preceding)) > energy_low_threshold:
            continue
        confidence = float(np.clip(change.confidence * 1.1, 0.0, 1.0))
        events.append(
            DJEvent(
                time=change.time,
                type="drop",
                confidence=confidence,
                details={"magnitude": change.magnitude, "section_id": _section_at(structure, change.time)},
            )
            )

    for section in structure.sections:
        if section.heuristic_label == "breakdown":
            events.append(
                DJEvent(
                    time=section.start,
                    type="breakdown",
                    confidence=section.heuristic_label_confidence,
                    details={"section_id": section.id, "end": section.end},
                )
            )

    if tempo_beat.downbeat_times.size and tempo_beat.onset_strength.size:
        strengths = np.interp(tempo_beat.downbeat_times, tempo_beat.onset_strength_times, tempo_beat.onset_strength)
        max_strength = float(np.max(strengths)) if strengths.size else 0.0
        if max_strength > 1e-9:
            normalized = strengths / max_strength
            threshold = float(np.percentile(normalized, 85))
            for t, s in zip(tempo_beat.downbeat_times, normalized):
                if s >= threshold:
                    events.append(
                        DJEvent(
                            time=float(t),
                            type="strong_downbeat",
                            confidence=float(np.clip(s * tempo_beat.downbeat_confidence, 0.0, 1.0)),
                            details={},
                        )
                    )

    abrupt = _abrupt_ending(y, sr, duration)
    if abrupt is not None:
        events.append(abrupt)

    if structure.sections:
        first = structure.sections[0]
        if first.heuristic_label == "intro" and first.duration >= EXTENDED_SECTION_MIN_SEC:
            events.append(
                DJEvent(
                    time=0.0,
                    type="extended_intro",
                    confidence=first.heuristic_label_confidence,
                    details={"duration": first.duration},
                )
            )
        last = structure.sections[-1]
        if last.heuristic_label == "outro" and last.duration >= EXTENDED_SECTION_MIN_SEC:
            events.append(
                DJEvent(
                    time=last.start,
                    type="extended_outro",
                    confidence=last.heuristic_label_confidence,
                    details={"duration": last.duration},
                )
            )

    for change in tempo_beat.tempo_change_events:
        events.append(
            DJEvent(
                time=change.time,
                type="tempo_change",
                confidence=change.confidence,
                details={"from_bpm": change.from_bpm, "to_bpm": change.to_bpm},
            )
        )

    for change in harmonic.key_change_events:
        events.append(
            DJEvent(
                time=change.time,
                type="key_change",
                confidence=change.confidence,
                details={"from_key": change.from_key, "to_key": change.to_key},
            )
        )

    events.sort(key=lambda e: e.time)
    return events


def analyze_dj_affordances(
    y: np.ndarray,
    sr: int,
    *,
    tempo_beat: TempoBeatFeatures,
    harmonic: HarmonicFeatures,
    energy_timeline: EnergyTimeline,
    vocal_timeline: VocalTimeline,
    structure: StructureAnalysis,
) -> DJAffordances:
    duration = y.size / sr if sr else 0.0

    mix_in, mix_out = _mix_points(structure, tempo_beat)
    loops = _loop_candidates(tempo_beat, harmonic, energy_timeline, structure)
    events = _events(y, sr, duration, tempo_beat, harmonic, energy_timeline, structure)

    return DJAffordances(
        mix_in_points=sorted(mix_in, key=lambda p: p.time),
        mix_out_points=sorted(mix_out, key=lambda p: p.time),
        loop_candidates=loops,
        events=events,
    )
