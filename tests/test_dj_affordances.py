from __future__ import annotations

import numpy as np
import soundfile as sf

from songanalysis.dj.affordances import analyze_dj_affordances
from songanalysis.features.energy import analyze_energy
from songanalysis.features.harmonic import analyze_harmonic
from songanalysis.features.loudness import short_term_loudness_series
from songanalysis.features.tempo_beat import analyze_tempo_beat
from songanalysis.features.vocals import analyze_vocals
from songanalysis.structure.segmentation import analyze_structure


def _full_analysis(y, sr):
    tb = analyze_tempo_beat(y, sr)
    harm = analyze_harmonic(y, sr)
    lt, lv = short_term_loudness_series(y, sr)
    en = analyze_energy(
        y,
        sr,
        onset_strength_times=tb.onset_strength_times,
        onset_strength=tb.onset_strength,
        onset_density_times=tb.onset_density_times,
        onset_density=tb.onset_density,
        loudness_times=lt,
        loudness_values=lv,
    )
    voc = analyze_vocals(y, sr)
    st = analyze_structure(y, sr, energy_timeline=en, vocal_timeline=voc, beat_times=tb.beat_times, chroma=harm.chroma, chroma_times=harm.chroma_times)
    dj = analyze_dj_affordances(y, sr, tempo_beat=tb, harmonic=harm, energy_timeline=en, vocal_timeline=voc, structure=st)
    return tb, harm, en, voc, st, dj


def test_mix_points_are_within_track_bounds_and_confident(structured_song_path):
    y, sr = sf.read(structured_song_path, dtype="float32")
    duration = len(y) / sr
    _, _, _, _, _, dj = _full_analysis(y, sr)

    assert len(dj.mix_in_points) > 0
    assert len(dj.mix_out_points) > 0
    for point in dj.mix_in_points + dj.mix_out_points:
        assert 0.0 <= point.time <= duration
        assert 0.0 <= point.confidence <= 1.0
        assert point.kind in {"mix_in", "mix_out"}
        assert len(point.reasons) > 0


def test_loop_candidates_are_beat_aligned_and_confident(structured_song_path):
    y, sr = sf.read(structured_song_path, dtype="float32")
    _, _, _, _, _, dj = _full_analysis(y, sr)

    for loop in dj.loop_candidates:
        assert loop.end > loop.start
        assert loop.bars in {4, 8, 16}
        assert 0.0 <= loop.confidence <= 1.0


def test_events_are_sorted_by_time_and_confident(structured_song_path):
    y, sr = sf.read(structured_song_path, dtype="float32")
    duration = len(y) / sr
    _, _, _, _, _, dj = _full_analysis(y, sr)

    times = [e.time for e in dj.events]
    assert times == sorted(times)
    for event in dj.events:
        assert 0.0 <= event.time <= duration
        assert 0.0 <= event.confidence <= 1.0
        assert event.type


def test_no_crash_on_short_silent_audio():
    y = np.zeros(int(22050 * 3), dtype=np.float32)
    tb, harm, en, voc, st, dj = _full_analysis(y, 22050)
    assert dj.mix_in_points == [] or all(0 <= p.confidence <= 1 for p in dj.mix_in_points)
    assert dj.loop_candidates == []


def test_drop_not_reported_for_narrow_energy_wobble():
    """Regression test: a steady beat whose energy never dips into a real
    lull (just ordinary beat-to-beat wobble) should not be riddled with
    "drop" events. Real-song validation found tracks with no breakdown at
    all (e.g. a continuous rap verse) getting 10-24 spurious "drop" events,
    one for nearly every minor energy wobble near *a* technically-local
    valley -- see dj/affordances.py's drop-detection comment."""
    from conftest import SR, click_track

    y = click_track(40.0, 120.0, amplitude=0.45)
    _, _, _, _, _, dj = _full_analysis(y, SR)
    drop_events = [e for e in dj.events if e.type == "drop"]
    assert len(drop_events) <= 1


def test_drop_reported_after_a_genuine_breakdown():
    """A real quiet-to-loud transition should still be detected as a drop."""
    from conftest import SR, click_track

    quiet = click_track(10.0, 120.0, amplitude=0.04)
    loud = click_track(15.0, 120.0, amplitude=0.7)
    y = np.concatenate([quiet, loud])
    _, _, _, _, _, dj = _full_analysis(y, SR)
    drop_events = [e for e in dj.events if e.type == "drop"]
    assert len(drop_events) >= 1
    assert any(8.0 <= e.time <= 13.0 for e in drop_events)
