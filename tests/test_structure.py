from __future__ import annotations

from songanalysis.features.energy import analyze_energy
from songanalysis.features.harmonic import analyze_harmonic
from songanalysis.features.loudness import short_term_loudness_series
from songanalysis.features.tempo_beat import analyze_tempo_beat
from songanalysis.features.vocals import analyze_vocals
from songanalysis.structure.segmentation import analyze_structure
from conftest import SR, click_track, silence


def _analyze(y):
    tb = analyze_tempo_beat(y, SR)
    harm = analyze_harmonic(y, SR)
    lt, lv = short_term_loudness_series(y, SR)
    en = analyze_energy(
        y,
        SR,
        onset_strength_times=tb.onset_strength_times,
        onset_strength=tb.onset_strength,
        onset_density_times=tb.onset_density_times,
        onset_density=tb.onset_density,
        loudness_times=lt,
        loudness_values=lv,
    )
    voc = analyze_vocals(y, SR)
    return analyze_structure(
        y, SR, energy_timeline=en, vocal_timeline=voc, beat_times=tb.beat_times, chroma=harm.chroma, chroma_times=harm.chroma_times
    )


def test_structured_song_produces_multiple_contiguous_sections(structured_song_path):
    import soundfile as sf

    y, sr = sf.read(structured_song_path, dtype="float32")
    result = _analyze(y)

    assert len(result.sections) >= 1
    # sections must tile the song with no gaps/overlaps
    for a, b in zip(result.sections, result.sections[1:]):
        assert a.end == b.start
    assert result.sections[0].start == 0.0
    assert result.sections[-1].end == len(y) / sr

    for section in result.sections:
        assert section.id.startswith("section_")
        assert 0.0 <= section.energy <= 1.0
        assert 0.0 <= section.vocal_presence <= 1.0
        assert 0.0 <= section.heuristic_label_confidence <= 1.0
        assert 0.0 <= section.boundary_confidence <= 1.0
        assert section.dominant_timbre in {"bass-heavy", "bright", "balanced"}


def test_repetition_groups_reference_valid_sections(structured_song_path):
    import soundfile as sf

    y, sr = sf.read(structured_song_path, dtype="float32")
    result = _analyze(y)
    section_ids = {s.id for s in result.sections}
    for group in result.repetitions:
        assert len(group.section_ids) >= 2
        assert set(group.section_ids) <= section_ids
        assert 0.0 <= group.mean_similarity <= 1.0


def test_short_audio_falls_back_to_single_section():
    y = click_track(4.0, 120.0)
    result = _analyze(y)
    assert len(result.sections) == 1
    assert result.method == "too_short_for_segmentation"
    assert result.sections[0].start == 0.0


def test_silence_does_not_crash():
    result = _analyze(silence(10.0))
    assert len(result.sections) >= 1
