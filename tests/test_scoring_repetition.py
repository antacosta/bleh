from __future__ import annotations

from songscoring.components.repetition import score_repetition
from songscoring.config import RepetitionConfig
from songscoring.state import DJState
from scoring_helpers import make_profile

CFG = RepetitionConfig()


def test_no_recent_history_has_no_penalty():
    candidate = make_profile(id="song-1", artist="A", genre="Pop")
    state = DJState()
    result = score_repetition(candidate, state, CFG)
    assert result.value == 100.0
    assert result.confidence == 1.0


def test_just_played_song_is_heavily_penalized():
    candidate = make_profile(id="song-1")
    state = DJState(recent_songs=("song-1",))
    result = score_repetition(candidate, state, CFG)
    assert result.value < 20.0


def test_penalty_decays_with_recency():
    candidate = make_profile(id="song-1")
    just_played = score_repetition(candidate, DJState(recent_songs=("song-1", "x", "y")), CFG).value
    played_a_while_ago = score_repetition(candidate, DJState(recent_songs=("x", "y", "song-1")), CFG).value
    assert played_a_while_ago > just_played


def test_same_artist_recently_played_is_penalized_less_than_same_song():
    candidate = make_profile(id="song-2", artist="Artist A")
    same_song = score_repetition(candidate, DJState(recent_songs=("song-2",)), CFG).value
    same_artist = score_repetition(
        make_profile(id="song-2", artist="Artist A"), DJState(recent_artists=("Artist A",)), CFG
    ).value
    assert same_artist > same_song


def test_same_genre_streak_gives_a_small_penalty():
    candidate = make_profile(id="song-3", genre="Techno")
    result = score_repetition(candidate, DJState(recent_genres=("Techno", "Techno", "Techno")), CFG)
    assert result.value < 100.0
    assert result.value > 40.0  # much milder than repeating a song or artist


def test_lookback_window_is_respected():
    candidate = make_profile(id="song-1")
    far_back = ("x",) * CFG.lookback + ("song-1",)  # just outside the lookback window
    result = score_repetition(candidate, DJState(recent_songs=far_back), CFG)
    assert result.value == 100.0


def test_missing_artist_and_genre_on_candidate_does_not_crash():
    candidate = make_profile(id="song-1", artist=None, genre=None)
    state = DJState(recent_songs=("other",), recent_artists=("Someone",), recent_genres=("Pop",))
    result = score_repetition(candidate, state, CFG)
    assert result.value == 100.0


def test_score_bounded_0_100():
    candidate = make_profile(id="song-1", artist="A", genre="Pop")
    state = DJState(
        recent_songs=("song-1", "song-1", "song-1"),
        recent_artists=("A", "A", "A"),
        recent_genres=("Pop", "Pop", "Pop"),
    )
    result = score_repetition(candidate, state, CFG)
    assert 0.0 <= result.value <= 100.0
