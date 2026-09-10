"""Orchestration: wires the independent analysis stages together.

    audio loading -> feature extraction -> feature normalization
    -> structural analysis -> DJ-affordance detection -> serialization

Each stage is a plain function with explicit inputs/outputs living in its
own module; this file only sequences them and threads outputs that later
stages reuse (e.g. the onset envelope computed once in ``tempo_beat`` is
reused by ``energy`` and never recomputed).
"""

from __future__ import annotations

from pathlib import Path

from songanalysis.features.energy import analyze_energy
from songanalysis.features.harmonic import analyze_harmonic
from songanalysis.features.loudness import analyze_loudness, short_term_loudness_series
from songanalysis.features.tempo_beat import analyze_tempo_beat
from songanalysis.features.vocals import analyze_vocals
from songanalysis.dj.affordances import analyze_dj_affordances
from songanalysis.io.loader import DEFAULT_ANALYSIS_SR, load_audio
from songanalysis.io.metadata import extract_metadata
from songanalysis.schema import SongAnalysis
from songanalysis.structure.segmentation import analyze_structure
from songanalysis.version import ANALYSIS_VERSION


def analyze_song(path: str | Path, *, analysis_sr: int = DEFAULT_ANALYSIS_SR) -> SongAnalysis:
    """Run the full analysis pipeline on a local audio file.

    Raises:
        songanalysis.errors.UnsupportedAudioFileError: unreadable/undecodable file.
        songanalysis.errors.EmptyAudioError: file decodes to zero audio frames.
    """
    buffer = load_audio(path, analysis_sr=analysis_sr)
    metadata = extract_metadata(buffer.path, buffer.raw)

    loudness = analyze_loudness(buffer.y, buffer.sr)
    tempo_beat = analyze_tempo_beat(buffer.y, buffer.sr)
    harmonic = analyze_harmonic(buffer.y, buffer.sr)

    loudness_times, loudness_values = short_term_loudness_series(buffer.y, buffer.sr)
    energy_timeline = analyze_energy(
        buffer.y,
        buffer.sr,
        onset_strength_times=tempo_beat.onset_strength_times,
        onset_strength=tempo_beat.onset_strength,
        onset_density_times=tempo_beat.onset_density_times,
        onset_density=tempo_beat.onset_density,
        loudness_times=loudness_times,
        loudness_values=loudness_values,
    )

    vocal_timeline = analyze_vocals(buffer.y, buffer.sr)

    structure = analyze_structure(
        buffer.y,
        buffer.sr,
        energy_timeline=energy_timeline,
        vocal_timeline=vocal_timeline,
        beat_times=tempo_beat.beat_times,
        chroma=harmonic.chroma,
        chroma_times=harmonic.chroma_times,
    )

    dj_affordances = analyze_dj_affordances(
        buffer.y,
        buffer.sr,
        tempo_beat=tempo_beat,
        harmonic=harmonic,
        energy_timeline=energy_timeline,
        vocal_timeline=vocal_timeline,
        structure=structure,
    )

    return SongAnalysis(
        analysis_version=ANALYSIS_VERSION,
        metadata=metadata,
        loudness=loudness,
        tempo_beat=tempo_beat,
        harmonic=harmonic,
        energy_timeline=energy_timeline,
        vocal_timeline=vocal_timeline,
        structure=structure,
        dj_affordances=dj_affordances,
    )
