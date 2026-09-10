from __future__ import annotations

from songscoring.state import DJState, EnergyDirection
from songplanner.planner_state import root_planner_state
from planner_helpers import make_candidate_score
from scoring_helpers import make_profile


def test_root_planner_state_wraps_the_real_state_unchanged():
    real_state = DJState(current_song=make_profile(id="current"), recent_songs=("a", "b"))
    root = root_planner_state(real_state)
    assert root.dj_state is real_state
    assert root.planned_sequence == ()
    assert root.step_scores == ()
    assert root.cumulative_transition_score == 0.0
    assert root.root_choice_id is None
    assert root.depth == 0


def test_advance_never_mutates_the_real_dj_state():
    real_state = DJState(current_song=make_profile(id="current"), recent_songs=("a",))
    root = root_planner_state(real_state)
    candidate = make_profile(id="next", artist="Someone", genre="Pop", overall_energy=0.7)
    score = make_candidate_score("next", total_score=80.0, confidence=0.9)

    advanced = root.advance(candidate, score, history_lookback=8)

    # The real, original DJState object is untouched.
    assert real_state.current_song.id == "current"
    assert real_state.recent_songs == ("a",)
    # The new state reflects the hypothetical move instead.
    assert advanced.dj_state.current_song.id == "next"
    assert advanced.dj_state.recent_songs == ("next", "a")
    assert advanced is not root


def test_advance_builds_history_most_recent_first_and_bounds_lookback():
    real_state = DJState(current_song=make_profile(id="current"))
    node = root_planner_state(real_state)
    for i in range(5):
        candidate = make_profile(id=f"song-{i}", artist=f"Artist-{i}", genre=f"Genre-{i}", overall_energy=0.1 * i)
        node = node.advance(candidate, make_candidate_score(f"song-{i}", 50.0), history_lookback=3)

    assert node.dj_state.recent_songs == ("song-4", "song-3", "song-2")
    assert node.dj_state.recent_artists == ("Artist-4", "Artist-3", "Artist-2")
    assert node.dj_state.recent_genres == ("Genre-4", "Genre-3", "Genre-2")
    assert len(node.dj_state.set_energy) == 3


def test_advance_preserves_desired_direction_and_intent_strength():
    real_state = DJState(
        current_song=make_profile(id="current"),
        desired_energy_direction=EnergyDirection.INCREASE,
    )
    node = root_planner_state(real_state)
    candidate = make_profile(id="next")
    node = node.advance(candidate, make_candidate_score("next", 70.0), history_lookback=8)
    assert node.dj_state.desired_energy_direction == EnergyDirection.INCREASE
    assert node.dj_state.energy_intent_strength == real_state.energy_intent_strength


def test_advance_sets_current_position_to_zero():
    real_state = DJState(current_song=make_profile(id="current"), current_position=180.0)
    node = root_planner_state(real_state).advance(make_profile(id="next"), make_candidate_score("next", 60.0), 8)
    assert node.dj_state.current_position == 0.0


def test_advance_accumulates_sequence_and_scores_and_cumulative_total():
    node = root_planner_state(DJState(current_song=make_profile(id="current")))
    a = make_profile(id="a")
    b = make_profile(id="b")
    node = node.advance(a, make_candidate_score("a", 60.0), 8)
    node = node.advance(b, make_candidate_score("b", 70.0), 8)

    assert [s.id for s in node.planned_sequence] == ["a", "b"]
    assert [s.total_score for s in node.step_scores] == [60.0, 70.0]
    assert node.cumulative_transition_score == 130.0
    assert node.depth == 2


def test_root_choice_id_is_set_once_and_then_preserved():
    node = root_planner_state(DJState(current_song=make_profile(id="current")))
    assert node.root_choice_id is None

    node = node.advance(make_profile(id="first"), make_candidate_score("first", 50.0), 8)
    assert node.root_choice_id == "first"

    node = node.advance(make_profile(id="second"), make_candidate_score("second", 50.0), 8)
    assert node.root_choice_id == "first"  # unchanged by later steps
