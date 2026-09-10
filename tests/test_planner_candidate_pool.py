from __future__ import annotations

from songscoring.config import DEFAULT_CONFIG
from songscoring.state import DJState
from songplanner.candidate_pool import build_pool
from songplanner.config import BeamSearchConfig, SequenceScoringConfig
from songplanner.planner_state import root_planner_state
from planner_helpers import make_candidate_score
from scoring_helpers import make_profile


def _library(n: int) -> list:
    return [make_profile(id=f"song-{i}", bpm=100.0 + i, artist=f"Artist-{i}") for i in range(n)]


def test_pool_excludes_the_real_current_song():
    library = _library(5)
    current = library[0]
    node = root_planner_state(DJState(current_song=current))
    by_id = {s.id: s for s in library}
    pool = build_pool(node, library, by_id, BeamSearchConfig(), SequenceScoringConfig(), DEFAULT_CONFIG)
    assert all(song.id != current.id for song, _ in pool)


def test_pool_is_truncated_to_max_pool_size():
    library = _library(20)
    node = root_planner_state(DJState(current_song=None))
    by_id = {s.id: s for s in library}
    config = BeamSearchConfig(max_pool_size=5)
    pool = build_pool(node, library, by_id, config, SequenceScoringConfig(), DEFAULT_CONFIG)
    assert len(pool) == 5


def test_pool_excludes_songs_already_used_in_this_hypothetical_path_when_forbidden():
    # A song used two steps back (not the immediately-preceding song, which
    # rank_candidates already structurally excludes as "the current song")
    # -- this specifically exercises forbid_repeat_song_within_horizon.
    library = _library(5)
    node = root_planner_state(DJState(current_song=None))
    used = library[1]
    node = node.advance(used, make_candidate_score(used.id, 90.0), history_lookback=8)
    node = node.advance(library[2], make_candidate_score(library[2].id, 90.0), history_lookback=8)
    by_id = {s.id: s for s in library}

    config = BeamSearchConfig()
    seq_config = SequenceScoringConfig(forbid_repeat_song_within_horizon=True)
    pool = build_pool(node, library, by_id, config, seq_config, DEFAULT_CONFIG)
    assert all(song.id != used.id for song, _ in pool)


def test_pool_allows_reusing_a_song_when_the_hard_constraint_is_disabled():
    library = _library(5)
    node = root_planner_state(DJState(current_song=None))
    used = library[1]
    node = node.advance(used, make_candidate_score(used.id, 90.0), history_lookback=8)
    node = node.advance(library[2], make_candidate_score(library[2].id, 90.0), history_lookback=8)
    by_id = {s.id: s for s in library}

    config = BeamSearchConfig()
    seq_config = SequenceScoringConfig(forbid_repeat_song_within_horizon=False)
    pool = build_pool(node, library, by_id, config, seq_config, DEFAULT_CONFIG)
    assert any(song.id == used.id for song, _ in pool)


def test_pool_entries_carry_their_full_one_step_candidate_score():
    library = _library(3)
    node = root_planner_state(DJState(current_song=None))
    by_id = {s.id: s for s in library}
    pool = build_pool(node, library, by_id, BeamSearchConfig(), SequenceScoringConfig(), DEFAULT_CONFIG)
    for song, score in pool:
        assert score.candidate_id == song.id
        assert 0.0 <= score.total_score <= 100.0


def test_pool_is_ranked_best_first():
    library = _library(6)
    node = root_planner_state(DJState(current_song=None))
    by_id = {s.id: s for s in library}
    pool = build_pool(node, library, by_id, BeamSearchConfig(), SequenceScoringConfig(), DEFAULT_CONFIG)
    scores = [score.total_score for _, score in pool]
    assert scores == sorted(scores, reverse=True)
