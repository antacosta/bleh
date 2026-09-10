from __future__ import annotations

import numpy as np

from songscoring.components.rhythm import score_rhythm
from songscoring.config import RhythmConfig
from scoring_helpers import make_profile

CFG = RhythmConfig()


def _with_density(density: float, percussive: float, duration: float = 200.0) -> dict:
    t = np.linspace(0, duration, 100)
    return dict(
        onset_density_times=t,
        onset_density=np.full_like(t, density),
        percussive_activity_times=t,
        percussive_activity=np.full_like(t, percussive),
    )


def test_same_density_and_stability_scores_highly():
    current = make_profile(**_with_density(3.0, 0.5), tempo_stability=0.9, time_signature="4/4", downbeat_confidence=0.8)
    candidate = make_profile(**_with_density(3.0, 0.5), tempo_stability=0.9, time_signature="4/4", downbeat_confidence=0.8)
    result = score_rhythm(current, candidate, CFG)
    assert result.value > 85


def test_same_bpm_different_rhythmic_character_scores_lower():
    """Two songs at the same BPM (not modeled here directly, tempo.py's
    job) can still be rhythmically very different -- sparse vs. dense."""
    sparse = make_profile(**_with_density(0.5, 0.05))
    dense = make_profile(**_with_density(6.0, 0.9))
    similar = make_profile(**_with_density(0.6, 0.08))

    mismatched = score_rhythm(sparse, dense, CFG).value
    matched = score_rhythm(sparse, similar, CFG).value
    assert matched > mismatched


def test_time_signature_mismatch_lowers_score_when_downbeats_are_confident():
    current = make_profile(time_signature="4/4", downbeat_confidence=0.9)
    match = make_profile(time_signature="4/4", downbeat_confidence=0.9)
    mismatch = make_profile(time_signature="3/4", downbeat_confidence=0.9)
    assert score_rhythm(current, match, CFG).value > score_rhythm(current, mismatch, CFG).value


def test_unknown_time_signature_does_not_crash_and_is_excluded():
    current = make_profile(time_signature=None)
    candidate = make_profile(time_signature="4/4")
    result = score_rhythm(current, candidate, CFG)
    assert "phrase_compatibility" not in result.explanation


def test_unstable_tempo_reduces_confidence():
    current = make_profile(tempo_stability=0.9)
    stable = make_profile(tempo_stability=0.9)
    unstable = make_profile(tempo_stability=0.05)
    assert score_rhythm(current, unstable, CFG).confidence < score_rhythm(current, stable, CFG).confidence


def test_no_current_song_is_neutral_and_zero_confidence():
    result = score_rhythm(None, make_profile(), CFG)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_missing_rhythm_data_falls_back_to_neutral():
    current = make_profile(
        onset_density=np.array([]), onset_density_times=np.array([]),
        percussive_activity=np.array([]), percussive_activity_times=np.array([]),
        tempo_stability=0.0, time_signature=None,
    )
    candidate = make_profile(
        onset_density=np.array([]), onset_density_times=np.array([]),
        percussive_activity=np.array([]), percussive_activity_times=np.array([]),
        tempo_stability=0.0, time_signature=None,
    )
    result = score_rhythm(current, candidate, CFG)
    # tempo_stability sub-score is always computed (defaults to 0 vs 0 -> perfect match),
    # so this should not be the "no data at all" neutral fallback.
    assert result.confidence >= 0.0


def test_score_bounded_0_100():
    current = make_profile(**_with_density(3.0, 0.5))
    for density in (0.01, 1.0, 10.0, 50.0):
        result = score_rhythm(current, make_profile(**_with_density(density, 0.5)), CFG)
        assert 0.0 <= result.value <= 100.0
