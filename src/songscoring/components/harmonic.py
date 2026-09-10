"""Harmonic (key) compatibility, via circle-of-fifths distance.

Uses the standard DJ harmonic-mixing model: keys a fifth apart, or a
relative major/minor pair, mix easily; keys further around the circle of
fifths (a tritone being the worst case) don't. Relative-major normalization
means a minor key is compared at its relative major's position, so C major
and A minor land at distance 0 (very compatible, not identical).

Real-world key confidence tends to run low (see the analysis-layer
validation report: most real songs top out around 5-25% key confidence).
That is not papered over here -- this component's own confidence is the
product of both songs' key confidences, so it will often end up small,
correctly reducing how much this dimension can move the total score.
"""

from __future__ import annotations

import math

from songscoring.config import HarmonicConfig
from songscoring.song_profile import SongProfile
from songscoring.types import ComponentScore

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_RELATIVE_MAJOR_OFFSET_SEMITONES = 3  # A minor -> C major, etc.


def _fifths_position(root: str, mode: str) -> int | None:
    if root not in _PITCH_CLASSES:
        return None
    semitone = _PITCH_CLASSES.index(root)
    if mode == "minor":
        semitone = (semitone + _RELATIVE_MAJOR_OFFSET_SEMITONES) % 12
    return (semitone * 7) % 12


def _circular_distance(a: int, b: int, modulus: int = 12) -> int:
    diff = abs(a - b) % modulus
    return min(diff, modulus - diff)


def score_harmonic(current: SongProfile | None, candidate: SongProfile, config: HarmonicConfig) -> ComponentScore:
    if current is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "no current song (start of set)"})
    if not current.key_root or not candidate.key_root or not current.key_mode or not candidate.key_mode:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "key unavailable for current and/or candidate song"})

    pos_current = _fifths_position(current.key_root, current.key_mode)
    pos_candidate = _fifths_position(candidate.key_root, candidate.key_mode)
    if pos_current is None or pos_candidate is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "unrecognized key root"})

    distance = _circular_distance(pos_current, pos_candidate)
    same_key = current.key_root == candidate.key_root and current.key_mode == candidate.key_mode

    value = 100.0 * math.exp(-((distance / config.decay_scale) ** config.decay_power))
    if distance == 0 and not same_key:
        value -= config.relative_key_deduction
    value = max(0.0, min(100.0, value))

    confidence = max(0.0, min(1.0, current.key_confidence * candidate.key_confidence))

    relationship = "same key" if same_key else ("relative major/minor" if distance == 0 else f"{distance} steps on circle of fifths")

    return ComponentScore(
        value=value,
        confidence=confidence,
        explanation={
            "current_key": current.key,
            "candidate_key": candidate.key,
            "circle_of_fifths_distance": distance,
            "relationship": relationship,
            "current_key_confidence": round(current.key_confidence, 3),
            "candidate_key_confidence": round(candidate.key_confidence, 3),
        },
    )
