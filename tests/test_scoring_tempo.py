from __future__ import annotations

from songscoring.components.tempo import score_tempo
from songscoring.config import TempoConfig
from scoring_helpers import make_profile

CFG = TempoConfig()


def test_identical_bpm_scores_perfectly():
    current = make_profile(bpm=128.0, bpm_confidence=0.9, tempo_stability=0.9)
    candidate = make_profile(bpm=128.0, bpm_confidence=0.9, tempo_stability=0.9)
    result = score_tempo(current, candidate, CFG)
    assert result.value == 100.0
    assert result.confidence > 0.7


def test_octave_relationship_scores_well_but_below_native_match():
    current = make_profile(bpm=128.0)
    same = make_profile(bpm=128.0)
    double = make_profile(bpm=256.0)
    half = make_profile(bpm=64.0)

    native = score_tempo(current, same, CFG)
    doubled = score_tempo(current, double, CFG)
    halved = score_tempo(current, half, CFG)

    for r in (native, doubled, halved):
        assert r.value > 70  # all "manageable"
    assert native.value > doubled.value
    assert native.value > halved.value


def test_arbitrary_large_mismatch_scores_lower_than_octave_relationships():
    current = make_profile(bpm=128.0)
    double = make_profile(bpm=256.0)
    unrelated = make_profile(bpm=171.0)  # not near 1x/0.5x/2x of 128

    doubled_score = score_tempo(current, double, CFG).value
    unrelated_score = score_tempo(current, unrelated, CFG).value
    assert doubled_score > unrelated_score


def test_missing_bpm_is_neutral_and_zero_confidence():
    current = make_profile(bpm=128.0)
    candidate = make_profile(bpm=None)
    result = score_tempo(current, candidate, CFG)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_no_current_song_is_neutral_and_zero_confidence():
    candidate = make_profile(bpm=128.0)
    result = score_tempo(None, candidate, CFG)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_low_bpm_confidence_reduces_component_confidence_not_value():
    current = make_profile(bpm=128.0, bpm_confidence=0.9)
    confident_candidate = make_profile(bpm=128.0, bpm_confidence=0.9)
    uncertain_candidate = make_profile(bpm=128.0, bpm_confidence=0.05)

    confident = score_tempo(current, confident_candidate, CFG)
    uncertain = score_tempo(current, uncertain_candidate, CFG)

    assert confident.value == uncertain.value == 100.0  # the match itself is identical
    assert uncertain.confidence < confident.confidence  # but we trust it much less


def test_unstable_tempo_reduces_confidence():
    current = make_profile(bpm=128.0, tempo_stability=0.9)
    stable_candidate = make_profile(bpm=128.0, tempo_stability=0.9)
    unstable_candidate = make_profile(bpm=128.0, tempo_stability=0.1)

    stable = score_tempo(current, stable_candidate, CFG)
    unstable = score_tempo(current, unstable_candidate, CFG)
    assert unstable.confidence < stable.confidence


def test_score_bounded_0_100():
    current = make_profile(bpm=90.0)
    for bpm in (30.0, 60.0, 90.0, 180.0, 400.0):
        result = score_tempo(current, make_profile(bpm=bpm), CFG)
        assert 0.0 <= result.value <= 100.0
