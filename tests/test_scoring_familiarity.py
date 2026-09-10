from __future__ import annotations

from songscoring.components.familiarity import score_familiarity
from songscoring.state import DJState
from scoring_helpers import make_profile


def test_no_familiarity_data_is_neutral_and_zero_confidence():
    """Do not invent user taste data: with nothing supplied, this must be
    a true neutral, not a guess."""
    candidate = make_profile(id="song-1")
    state = DJState()
    result = score_familiarity(candidate, state)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_unknown_song_in_a_populated_map_is_still_neutral():
    candidate = make_profile(id="song-not-in-map")
    state = DJState(familiarity_by_song_id={"some-other-song": 0.9})
    result = score_familiarity(candidate, state)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_known_familiarity_is_used_and_confident():
    candidate = make_profile(id="song-1")
    state = DJState(familiarity_by_song_id={"song-1": 0.6})
    result = score_familiarity(candidate, state)
    assert result.confidence > 0.0
    assert result.explanation["familiarity"] == 0.6


def test_novelty_map_is_accepted_as_an_alternative_input():
    candidate = make_profile(id="song-1")
    state = DJState(novelty_by_song_id={"song-1": 0.2})  # familiarity = 0.8
    result = score_familiarity(candidate, state)
    assert result.explanation["familiarity"] == 0.8


def test_explicit_familiarity_takes_precedence_over_novelty():
    candidate = make_profile(id="song-1")
    state = DJState(familiarity_by_song_id={"song-1": 0.3}, novelty_by_song_id={"song-1": 0.9})
    result = score_familiarity(candidate, state)
    assert result.explanation["familiarity"] == 0.3


def test_score_bounded_0_100():
    candidate = make_profile(id="song-1")
    for familiarity in (0.0, 0.3, 0.65, 1.0):
        state = DJState(familiarity_by_song_id={"song-1": familiarity})
        result = score_familiarity(candidate, state)
        assert 0.0 <= result.value <= 100.0
