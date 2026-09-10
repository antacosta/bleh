from __future__ import annotations

from songscoring.components.harmonic import score_harmonic
from songscoring.config import HarmonicConfig
from scoring_helpers import make_profile

CFG = HarmonicConfig()


def test_same_key_scores_perfectly():
    current = make_profile(key="C major", key_root="C", key_mode="major", key_confidence=0.8)
    candidate = make_profile(key="C major", key_root="C", key_mode="major", key_confidence=0.8)
    result = score_harmonic(current, candidate, CFG)
    assert result.value == 100.0


def test_relative_major_minor_scores_highly_but_below_identical():
    current = make_profile(key="C major", key_root="C", key_mode="major")
    relative = make_profile(key="A minor", key_root="A", key_mode="minor")
    identical = make_profile(key="C major", key_root="C", key_mode="major")

    relative_score = score_harmonic(current, relative, CFG).value
    identical_score = score_harmonic(current, identical, CFG).value
    assert 80 < relative_score < identical_score == 100.0


def test_perfect_fifth_scores_higher_than_distant_key():
    current = make_profile(key="C major", key_root="C", key_mode="major")
    fifth = make_profile(key="G major", key_root="G", key_mode="major")  # 1 step
    tritone = make_profile(key="F# major", key_root="F#", key_mode="major")  # 6 steps

    fifth_score = score_harmonic(current, fifth, CFG).value
    tritone_score = score_harmonic(current, tritone, CFG).value
    assert fifth_score > tritone_score


def test_missing_key_is_neutral_and_zero_confidence():
    current = make_profile(key="C major", key_root="C", key_mode="major")
    candidate = make_profile(key=None, key_root=None, key_mode=None)
    result = score_harmonic(current, candidate, CFG)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_no_current_song_is_neutral_and_zero_confidence():
    candidate = make_profile(key="C major", key_root="C", key_mode="major")
    result = score_harmonic(None, candidate, CFG)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_low_key_confidence_on_either_song_lowers_component_confidence():
    current = make_profile(key="C major", key_root="C", key_mode="major", key_confidence=0.05)
    candidate = make_profile(key="C major", key_root="C", key_mode="major", key_confidence=0.9)
    result = score_harmonic(current, candidate, CFG)
    assert result.value == 100.0  # the keys really do match
    assert result.confidence < 0.1  # but we shouldn't trust the current song's key much


def test_score_bounded_0_100():
    current = make_profile(key="C major", key_root="C", key_mode="major")
    for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        for mode in ("major", "minor"):
            result = score_harmonic(current, make_profile(key=f"{root} {mode}", key_root=root, key_mode=mode), CFG)
            assert 0.0 <= result.value <= 100.0
