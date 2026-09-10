"""Test-only factory for building SongProfile instances directly, without
round-tripping through JSON -- lets scoring tests set up precise, minimal
scenarios (a specific BPM, an absent key, a particular mix-in point) without
constructing a full analysis document each time."""

from __future__ import annotations

from typing import Any

import numpy as np

from songscoring.song_profile import LoopRef, MixPointRef, SectionSummary, SongProfile, VocalRegion


def make_profile(**overrides: Any) -> SongProfile:
    duration = overrides.pop("duration_sec", 200.0)

    energy_times = overrides.pop("energy_times", np.linspace(0, duration, 200))
    composite_energy = overrides.pop("composite_energy", np.full_like(energy_times, 0.5))

    vocal_times = overrides.pop("vocal_times", np.linspace(0, duration, 100))
    vocal_probability = overrides.pop("vocal_probability", np.full_like(vocal_times, 0.4))

    onset_density_times = overrides.pop("onset_density_times", np.linspace(0, duration, 100))
    onset_density = overrides.pop("onset_density", np.full_like(onset_density_times, 2.0))

    percussive_activity_times = overrides.pop("percussive_activity_times", np.linspace(0, duration, 100))
    percussive_activity = overrides.pop("percussive_activity", np.full_like(percussive_activity_times, 0.4))

    defaults: dict[str, Any] = dict(
        id="song",
        source_path=None,
        filename="song.mp3",
        artist="Artist",
        title="Title",
        album="Album",
        genre="Pop",
        release_year=2020,
        duration_sec=duration,
        bpm=120.0,
        bpm_confidence=0.8,
        time_signature="4/4",
        time_signature_confidence=0.8,
        beats_per_bar=4,
        tempo_stability=0.8,
        downbeat_confidence=0.6,
        downbeat_times=np.arange(0, duration, 2.0),
        key="C major",
        key_root="C",
        key_mode="major",
        key_confidence=0.7,
        chroma_mean=np.array([1.0, 0.2, 0.3, 0.1, 0.6, 0.3, 0.2, 0.8, 0.1, 0.2, 0.1, 0.4]),
        integrated_lufs=-14.0,
        energy_times=energy_times,
        composite_energy=composite_energy,
        overall_energy=float(np.mean(composite_energy)) if composite_energy.size else 0.0,
        energy_trend=0.0,
        onset_density_times=onset_density_times,
        onset_density=onset_density,
        percussive_activity_times=percussive_activity_times,
        percussive_activity=percussive_activity,
        vocal_regions=(VocalRegion(0.0, duration, "vocal"),),
        vocal_times=vocal_times,
        vocal_probability=vocal_probability,
        vocal_density=0.5,
        sections=(
            SectionSummary(
                id="section_1",
                start=0.0,
                end=duration,
                duration=duration,
                heuristic_label=None,
                heuristic_label_confidence=0.0,
                energy=0.5,
                rhythmic_density=2.0,
                vocal_presence=0.5,
                dominant_timbre="balanced",
                repetition_group=None,
            ),
        ),
        mix_in_points=(MixPointRef(time=0.0, reasons=("instrumental_intro",), confidence=0.6, section_id="section_1"),),
        mix_out_points=(
            MixPointRef(time=duration - 10.0, reasons=("outro",), confidence=0.6, section_id="section_1"),
        ),
        loop_candidates=(LoopRef(start=10.0, end=18.0, bars=4, confidence=0.8),),
        events=(),
        raw={},
    )
    defaults.update(overrides)
    return SongProfile(**defaults)
