"""Unit tests for ``songplanner.planner`` itself: the beam search loop and
its ``local_vs_global`` explainability bookkeeping, as opposed to
``sequence_scoring`` (tested in ``test_planner_sequence_scoring.py``) or
end-to-end scenarios against the real corpus (``test_planner_scenarios.py``).
"""

from __future__ import annotations

from songscoring.state import DJState
from songplanner.candidate_pool import build_pool
from songplanner.config import BeamSearchConfig, PlannerConfig
from songplanner.planner import plan_sequence
from songplanner.planner_state import root_planner_state
from songplanner.sequence_scoring import score_sequence
from scoring_helpers import make_profile


def test_local_vs_global_reports_the_best_surviving_continuation_per_root():
    """A single opening move (``root_choice_id``) can still have more than
    one hypothetical continuation alive in the same beam-search round --
    beam search doesn't collapse to one lineage per first song. When that
    happens, the reported ``downstream_path_score`` for that root must be
    the *best* of its surviving continuations, not whichever one happened
    to be scored last while iterating the round -- otherwise the whole
    point of the field (showing how well an opening move could have done)
    is undermined by iteration-order noise.

    This is a real, previously-broken case: with the exact profiles below,
    ``X`` reaches a second beam round with two live continuations, ``Y1``
    (tempo-compatible, ranked first, a strong path) and ``W`` (a poor
    tempo match, ranked second, a much weaker path). The pool is always
    explored best-first, so "last scored" and "worst" coincide here -- an
    unconditional overwrite silently reports the worse one.
    """
    current = make_profile(id="current", bpm=120.0, artist="Cur", genre="Rock")
    w = make_profile(id="W", bpm=120.0, artist="ArtW", genre="Rock")
    x = make_profile(id="X", bpm=140.0, artist="ArtX", genre="Rock")
    y1 = make_profile(id="Y1", bpm=141.0, artist="ArtY1", genre="Rock")
    library = [w, x, y1]
    state = DJState(current_song=current)
    config = PlannerConfig(
        beam=BeamSearchConfig(horizon=2, beam_width=2, candidates_per_expansion=2, max_pool_size=10, min_pool_size=1)
    )

    # Independently (via the same public building blocks the planner
    # itself uses, not a reimplementation of its search) compute X's two
    # actual candidate continuations this round, so the expected value
    # below isn't a hand-typed magic number.
    root = root_planner_state(state)
    by_id = {song.id: song for song in library}
    root_pool = build_pool(root, library, by_id, config.beam, config.sequence, config.scoring)
    x_score = next(score for candidate, score in root_pool if candidate.id == "X")
    x_node = root.advance(x, x_score, config.scoring.repetition.lookback)
    x_pool = build_pool(x_node, library, by_id, config.beam, config.sequence, config.scoring)
    assert [candidate.id for candidate, _ in x_pool] == ["Y1", "W"]  # Y1 ranked strictly ahead of W

    continuation_path_scores = []
    for candidate, cand_score in x_pool:
        child = x_node.advance(candidate, cand_score, config.scoring.repetition.lookback)
        breakdown = score_sequence(
            state,
            child.planned_sequence,
            child.step_scores,
            config.sequence,
            config.scoring.energy_intent,
            config.scoring.energy.window_sec,
        )
        continuation_path_scores.append(breakdown.path_score)
    best_continuation_score = max(continuation_path_scores)
    assert continuation_path_scores[0] == best_continuation_score  # Y1 (ranked first) is also the better path here
    assert continuation_path_scores[1] < best_continuation_score  # W (ranked second) is a strictly worse path

    plan = plan_sequence(state, library, config)

    x_entry = next(c for c in plan.local_vs_global if c.song_id == "X")
    assert not x_entry.chosen  # X must lose overall for this check to be meaningful (see module docstring)
    assert x_entry.depth_reached == 2
    assert x_entry.downstream_path_score == best_continuation_score
