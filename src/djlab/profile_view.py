"""Read-only JSON views of a ``SongProfile`` for the UI's debug panel.

Every field here already exists on ``SongProfile`` (``songscoring.
song_profile``) -- this module only picks out the scalar/summary fields
worth showing a human and rounds them for display. It computes nothing new
and makes no scoring or analysis decisions.
"""

from __future__ import annotations

from typing import Any

from songscoring.song_profile import SongProfile


def profile_summary_dict(profile: SongProfile) -> dict[str, Any]:
    """Compact fields for a song-list row (used alongside library.py's own
    lightweight listing once a song has actually been analyzed)."""
    return {
        "id": profile.id,
        "artist": profile.artist,
        "title": profile.title,
        "genre": profile.genre,
        "duration_sec": round(profile.duration_sec, 1),
        "bpm": round(profile.bpm, 1) if profile.bpm is not None else None,
        "key": profile.key,
        "overall_energy": round(profile.overall_energy, 3),
    }


def profile_debug_dict(profile: SongProfile) -> dict[str, Any]:
    """Full inspectable detail for the debug view: everything the scoring
    and planning layers actually read from this song."""
    return {
        "id": profile.id,
        "artist": profile.artist,
        "title": profile.title,
        "album": profile.album,
        "genre": profile.genre,
        "duration_sec": round(profile.duration_sec, 1),
        "tempo": {
            "bpm": round(profile.bpm, 2) if profile.bpm is not None else None,
            "bpm_confidence": round(profile.bpm_confidence, 3),
            "time_signature": profile.time_signature,
            "time_signature_confidence": round(profile.time_signature_confidence, 3),
            "tempo_stability": round(profile.tempo_stability, 3),
        },
        "harmonic": {
            "key": profile.key,
            "key_root": profile.key_root,
            "key_mode": profile.key_mode,
            "key_confidence": round(profile.key_confidence, 3),
        },
        "loudness": {
            "integrated_lufs": profile.integrated_lufs,
        },
        "energy": {
            "overall_energy": round(profile.overall_energy, 3),
            "energy_trend": round(profile.energy_trend, 5),
        },
        "rhythm": {
            "mean_onset_density": profile.mean_onset_density(),
            "mean_percussive_activity": profile.mean_percussive_activity(),
        },
        "vocals": {
            "vocal_density": round(profile.vocal_density, 3),
            "region_count": len(profile.vocal_regions),
        },
        "structure": {
            "sections": [
                {
                    "id": s.id,
                    "start": round(s.start, 1),
                    "end": round(s.end, 1),
                    "label": s.heuristic_label,
                    "label_confidence": round(s.heuristic_label_confidence, 3),
                    "energy": round(s.energy, 3),
                    "vocal_presence": round(s.vocal_presence, 3),
                    "dominant_timbre": s.dominant_timbre,
                }
                for s in profile.sections
            ],
        },
        "dj_affordances": {
            "mix_in_points": [
                {"time": round(p.time, 1), "confidence": round(p.confidence, 3), "reasons": list(p.reasons)}
                for p in profile.mix_in_points
            ],
            "mix_out_points": [
                {"time": round(p.time, 1), "confidence": round(p.confidence, 3), "reasons": list(p.reasons)}
                for p in profile.mix_out_points
            ],
            "loop_candidates": [
                {"start": round(l.start, 1), "end": round(l.end, 1), "bars": l.bars, "confidence": round(l.confidence, 3)}
                for l in profile.loop_candidates
            ],
        },
    }
