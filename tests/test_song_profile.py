from __future__ import annotations

from songscoring.song_profile import profile_from_analysis


def _minimal_analysis(**overrides):
    base = {
        "metadata": {"filename": "x.mp3", "duration_sec": 120.0},
        "global": {"duration_sec": 120.0},
        "timeline": {"energy": {}, "vocals": {}},
        "structure": {},
        "harmonic": {},
        "rhythm": {},
        "dj_affordances": {},
    }
    base.update(overrides)
    return base


def test_loads_a_full_realistic_document():
    analysis = {
        "metadata": {
            "filename": "song.mp3",
            "artist": "Some Artist",
            "title": "Some Title",
            "album": "Some Album",
            "genre": "House",
            "release_year": 2019,
            "duration_sec": 240.0,
        },
        "global": {
            "bpm": 128.0,
            "bpm_confidence": 0.8,
            "time_signature": "4/4",
            "time_signature_confidence": 0.7,
            "key": "A minor",
            "key_confidence": 0.3,
            "integrated_lufs": -9.5,
            "overall_energy": 0.5,
            "energy_trend": 0.001,
        },
        "timeline": {
            "energy": {"times": [0.0, 1.0], "composite_energy": [0.2, 0.3]},
            "vocals": {"regions": [{"start": 0, "end": 10, "kind": "instrumental"}], "times": [0.0], "probability": [0.1], "density": 0.4},
        },
        "structure": {
            "sections": [
                {
                    "id": "section_1", "start": 0, "end": 240, "duration": 240,
                    "heuristic_label": "intro", "heuristic_label_confidence": 0.4,
                    "energy": 0.3, "rhythmic_density": 1.5, "vocal_presence": 0.2,
                    "dominant_timbre": "bright", "repetition_group": None,
                }
            ]
        },
        "harmonic": {"key_root": "A", "key_mode": "minor", "chroma": [[1.0] * 12, [0.5] * 12]},
        "rhythm": {
            "beats_per_bar": 4, "tempo_stability": 0.9, "downbeat_confidence": 0.6,
            "downbeat_times": [0.0, 2.0], "onset_density_times": [0.0], "onset_density": [2.0],
            "percussive_activity_times": [0.0], "percussive_activity": [0.5],
        },
        "dj_affordances": {
            "mix_in_points": [{"time": 0.0, "reasons": ["instrumental_intro"], "confidence": 0.7, "section_id": "section_1"}],
            "mix_out_points": [{"time": 220.0, "reasons": ["outro"], "confidence": 0.6, "section_id": "section_1"}],
            "loop_candidates": [{"start": 10.0, "end": 18.0, "bars": 4, "confidence": 0.8}],
            "events": [{"time": 30.0, "type": "drop", "confidence": 0.5, "details": {}}],
        },
    }
    profile = profile_from_analysis(analysis)
    assert profile.id == "song.mp3"
    assert profile.artist == "Some Artist"
    assert profile.bpm == 128.0
    assert profile.key_root == "A"
    assert profile.chroma_mean is not None and profile.chroma_mean.shape == (12,)
    assert len(profile.sections) == 1
    assert len(profile.mix_in_points) == 1
    assert profile.mix_in_points[0].reasons == ("instrumental_intro",)
    assert len(profile.loop_candidates) == 1
    assert len(profile.events) == 1
    assert profile.events[0].type == "drop"


def test_handles_completely_missing_optional_sections():
    analysis = _minimal_analysis()
    profile = profile_from_analysis(analysis)
    assert profile.bpm is None
    assert profile.key is None
    assert profile.chroma_mean is None
    assert profile.sections == ()
    assert profile.mix_in_points == ()
    assert profile.events == ()
    assert profile.duration_sec == 120.0


def test_id_defaults_to_filename_then_source_path():
    analysis = _minimal_analysis(metadata={"filename": "abc.mp3"})
    assert profile_from_analysis(analysis).id == "abc.mp3"

    analysis_no_filename = _minimal_analysis(metadata={})
    assert profile_from_analysis(analysis_no_filename, source_path="/x/y.json").id == "/x/y.json"

    assert profile_from_analysis(analysis_no_filename, id="explicit-id").id == "explicit-id"


def test_energy_window_helpers_handle_empty_arrays():
    profile = profile_from_analysis(_minimal_analysis())
    assert profile.starting_energy(20.0) is None
    assert profile.ending_energy(20.0) is None
    assert profile.mean_onset_density() is None
    assert profile.best_mix_in() is None
    assert profile.best_mix_out() is None
    assert profile.timbre_distribution() == {}
