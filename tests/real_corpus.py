"""Loads the real, validated 25-song corpus for tests that need genuine
song characteristics (not synthetic placeholders) -- notably the look-ahead
planner tests, which need real variety in tempo/key/genre/energy to
meaningfully exercise sequence-level scoring and beam search.

``tests/fixtures/real_corpus/*.json`` is a *distillation* of the full
songanalysis output for each of the 25 real tracks used in the validation
report (see README's "Scope"/validation history) -- every value is real,
measured data, just reduced to the fields SongProfile actually reads (the
full per-track analysis JSON also carries large per-frame arrays --
chroma matrix, tempo curve, raw onset strength -- that SongProfile never
loads at all, and one it does but only ever averages, so those are
downsampled by taking every Nth *real* sample, never synthesized).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from songscoring.song_profile import EventRef, LoopRef, MixPointRef, SectionSummary, SongProfile, VocalRegion

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "real_corpus"


def _arr(values: list[float] | None) -> np.ndarray:
    if not values:
        return np.array([])
    return np.asarray(values, dtype=np.float64)


def _profile_from_fixture(d: dict) -> SongProfile:
    return SongProfile(
        id=d["id"],
        source_path=None,
        filename=d["filename"],
        artist=d["artist"],
        title=d["title"],
        album=d["album"],
        genre=d["genre"],
        release_year=d["release_year"],
        duration_sec=d["duration_sec"],
        bpm=d["bpm"],
        bpm_confidence=d["bpm_confidence"],
        time_signature=d["time_signature"],
        time_signature_confidence=d["time_signature_confidence"],
        beats_per_bar=d["beats_per_bar"],
        tempo_stability=d["tempo_stability"],
        downbeat_confidence=d["downbeat_confidence"],
        downbeat_times=_arr(d["downbeat_times"]),
        key=d["key"],
        key_root=d["key_root"],
        key_mode=d["key_mode"],
        key_confidence=d["key_confidence"],
        chroma_mean=_arr(d["chroma_mean"]) if d["chroma_mean"] is not None else None,
        integrated_lufs=d["integrated_lufs"],
        energy_times=_arr(d["energy_times"]),
        composite_energy=_arr(d["composite_energy"]),
        overall_energy=d["overall_energy"],
        energy_trend=d["energy_trend"],
        onset_density_times=_arr(d["onset_density_times"]),
        onset_density=_arr(d["onset_density"]),
        percussive_activity_times=_arr(d["percussive_activity_times"]),
        percussive_activity=_arr(d["percussive_activity"]),
        vocal_regions=tuple(VocalRegion(r["start"], r["end"], r["kind"]) for r in d["vocal_regions"]),
        vocal_times=_arr(d["vocal_times"]),
        vocal_probability=_arr(d["vocal_probability"]),
        vocal_density=d["vocal_density"],
        sections=tuple(
            SectionSummary(
                id=s["id"], start=s["start"], end=s["end"], duration=s["duration"],
                heuristic_label=s["heuristic_label"], heuristic_label_confidence=s["heuristic_label_confidence"],
                energy=s["energy"], rhythmic_density=s["rhythmic_density"], vocal_presence=s["vocal_presence"],
                dominant_timbre=s["dominant_timbre"], repetition_group=s["repetition_group"],
            )
            for s in d["sections"]
        ),
        mix_in_points=tuple(
            MixPointRef(time=p["time"], reasons=tuple(p["reasons"]), confidence=p["confidence"], section_id=p["section_id"])
            for p in d["mix_in_points"]
        ),
        mix_out_points=tuple(
            MixPointRef(time=p["time"], reasons=tuple(p["reasons"]), confidence=p["confidence"], section_id=p["section_id"])
            for p in d["mix_out_points"]
        ),
        loop_candidates=tuple(
            LoopRef(start=l["start"], end=l["end"], bars=l["bars"], confidence=l["confidence"])
            for l in d["loop_candidates"]
        ),
        events=tuple(
            EventRef(time=e["time"], type=e["type"], confidence=e["confidence"], details=e["details"])
            for e in d["events"]
        ),
        raw={},
    )


def load_real_corpus() -> dict[str, SongProfile]:
    """All 25 real, validated songs as ``SongProfile``s, keyed by id."""
    profiles: dict[str, SongProfile] = {}
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        with open(path) as f:
            d = json.load(f)
        profiles[d["id"]] = _profile_from_fixture(d)
    return profiles
