from __future__ import annotations

import math

import pytest

from songanalysis.errors import UnsupportedAudioFileError
from songanalysis.pipeline import analyze_song
from songanalysis.version import ANALYSIS_VERSION


def _assert_confidence(value, name):
    assert isinstance(value, (int, float)), f"{name} should be numeric"
    assert 0.0 <= value <= 1.0, f"{name}={value} out of [0,1]"


def validate_schema(d: dict) -> None:
    assert d["analysis_version"] == ANALYSIS_VERSION

    meta = d["metadata"]
    for key in ("filename", "duration_sec", "sample_rate", "channels", "format"):
        assert key in meta
    assert meta["duration_sec"] >= 0
    assert meta["sample_rate"] > 0
    assert meta["channels"] >= 1

    glob = d["global"]
    _assert_confidence(glob["bpm_confidence"], "global.bpm_confidence")
    _assert_confidence(glob["key_confidence"], "global.key_confidence")
    _assert_confidence(glob["time_signature_confidence"], "global.time_signature_confidence")
    if glob["bpm"] is not None:
        assert glob["bpm"] > 0

    timeline = d["timeline"]
    assert isinstance(timeline["beats"], list)
    assert isinstance(timeline["downbeats"], list)
    assert isinstance(timeline["onsets"], list)
    energy = timeline["energy"]
    assert isinstance(energy["times"], list)
    assert len(energy["times"]) == len(energy["composite_energy"])
    for v in energy["composite_energy"]:
        assert -1e-6 <= v <= 1.0 + 1e-6

    vocals = timeline["vocals"]
    assert 0.0 <= vocals["density"] <= 1.0
    for region in vocals["regions"]:
        assert region["kind"] in {"vocal", "instrumental"}
        assert region["end"] >= region["start"]

    structure = d["structure"]
    assert isinstance(structure["sections"], list)
    for section in structure["sections"]:
        assert section["id"].startswith("section_")
        _assert_confidence(section["heuristic_label_confidence"], "section.heuristic_label_confidence")
        _assert_confidence(section["boundary_confidence"], "section.boundary_confidence")
    for group in structure["repetitions"]:
        assert len(group["section_ids"]) >= 2

    harmonic = d["harmonic"]
    assert harmonic["chroma"] == [] or len(harmonic["chroma"][0]) == 12

    rhythm = d["rhythm"]
    assert "bpm" in rhythm and "beat_times" in rhythm

    dj = d["dj_affordances"]
    for point in dj["mix_in_points"] + dj["mix_out_points"]:
        _assert_confidence(point["confidence"], "mix point confidence")
    for loop in dj["loop_candidates"]:
        assert loop["bars"] in (4, 8, 16)
        _assert_confidence(loop["confidence"], "loop confidence")
    for event in dj["events"]:
        _assert_confidence(event["confidence"], "event confidence")


def test_schema_validity_on_structured_song(structured_song_path):
    analysis = analyze_song(structured_song_path)
    d = analysis.to_jsonable()
    validate_schema(d)


def test_schema_validity_on_silent_file(silent_wav_path):
    analysis = analyze_song(silent_wav_path)
    d = analysis.to_jsonable()
    validate_schema(d)
    assert d["global"]["bpm"] is None or d["global"]["bpm_confidence"] == 0.0
    assert d["global"]["integrated_lufs"] is None
    assert d["global"]["integrated_lufs_confidence"] == "silent"


def test_schema_validity_on_very_short_file(very_short_wav_path):
    analysis = analyze_song(very_short_wav_path)
    d = analysis.to_jsonable()
    validate_schema(d)


def test_unusually_long_file_is_handled(tmp_path):
    import soundfile as sf

    from conftest import SR, click_track

    # ~4 minutes: exercises the full pipeline at a length closer to a real song.
    y = click_track(240.0, 120.0)
    path = tmp_path / "long.wav"
    sf.write(str(path), y, SR)

    analysis = analyze_song(str(path))
    d = analysis.to_jsonable()
    validate_schema(d)
    assert d["metadata"]["duration_sec"] == pytest.approx(240.0, abs=0.1)


def test_malformed_file_raises_clean_error(malformed_file_path):
    with pytest.raises(UnsupportedAudioFileError):
        analyze_song(malformed_file_path)


def test_determinism_same_file_same_result(click_track_path):
    d1 = analyze_song(click_track_path).to_jsonable()
    d2 = analyze_song(click_track_path).to_jsonable()
    assert d1 == d2


def test_no_nan_or_inf_leak_into_json(structured_song_path):
    import json

    analysis = analyze_song(structured_song_path)
    text = json.dumps(analysis.to_jsonable(), allow_nan=False)  # raises if NaN/Infinity present
    assert "NaN" not in text
    assert "Infinity" not in text
