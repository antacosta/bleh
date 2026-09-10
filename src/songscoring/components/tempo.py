"""Tempo compatibility.

Not "is the BPM difference small" -- a future DJ engine can beatmatch a
9% tempo change, half-time, or double-time; it cannot manage an arbitrary
percentage change gracefully. So this scores *distance to the nearest
plausible manageable ratio* (1x, 0.5x, 2x by default), not distance to the
other song's raw BPM, and folds in both songs' tempo confidence and
stability: a BPM we're not sure about, or one that wanders, isn't a BPM
worth matching precisely against.
"""

from __future__ import annotations

import math

from songscoring.config import TempoConfig
from songscoring.song_profile import SongProfile
from songscoring.types import ComponentScore


def score_tempo(current: SongProfile | None, candidate: SongProfile, config: TempoConfig) -> ComponentScore:
    if current is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "no current song (start of set)"})
    if current.bpm is None or candidate.bpm is None or current.bpm <= 0 or candidate.bpm <= 0:
        return ComponentScore(
            value=50.0,
            confidence=0.0,
            explanation={"reason": "bpm unavailable for current and/or candidate song"},
        )

    ratio = candidate.bpm / current.bpm
    log_ratio = math.log2(ratio)

    best_anchor = min(config.manageable_ratios, key=lambda a: abs(log_ratio - math.log2(a)))
    log_distance = abs(log_ratio - math.log2(best_anchor))
    # Fractional adjustment a time-stretcher would need to apply to land the
    # candidate exactly on the chosen anchor ratio.
    fractional_adjustment = 2.0**log_distance - 1.0

    base = 100.0 * math.exp(-((fractional_adjustment / config.tolerance_fraction) ** 2))
    penalty = config.octave_ratio_penalty if best_anchor != 1.0 else 0.0
    value = max(0.0, min(100.0, base - penalty))

    stability_factor = (current.tempo_stability + candidate.tempo_stability) / 2.0
    # Both songs' BPM readings need to be trustworthy *and* the beat needs to
    # actually hold steady for "match this tempo" to mean anything.
    confidence = current.bpm_confidence * candidate.bpm_confidence * (0.4 + 0.6 * stability_factor)

    return ComponentScore(
        value=value,
        confidence=max(0.0, min(1.0, confidence)),
        explanation={
            "current_bpm": current.bpm,
            "candidate_bpm": candidate.bpm,
            "ratio": round(ratio, 4),
            "best_anchor_ratio": best_anchor,
            "fractional_adjustment_needed": round(fractional_adjustment, 4),
            "tempo_stability_avg": round(stability_factor, 3),
        },
    )
