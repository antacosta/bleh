from __future__ import annotations

from songscoring.components.structure import score_structure
from songscoring.config import StructureConfig
from songscoring.song_profile import LoopRef, MixPointRef
from songscoring.state import DJState
from scoring_helpers import make_profile

CFG = StructureConfig()


def test_strong_exit_and_entry_scores_highly():
    current = make_profile(
        duration_sec=200.0,
        mix_out_points=(MixPointRef(time=180.0, reasons=("outro", "low_vocal_density"), confidence=0.9, section_id="s"),),
    )
    candidate = make_profile(
        mix_in_points=(MixPointRef(time=0.0, reasons=("instrumental_intro",), confidence=0.9, section_id="s"),)
    )
    state = DJState(current_song=current, current_position=0.0)
    result = score_structure(current, candidate, state, CFG)
    assert result.value > 80
    assert result.confidence > 0.7


def test_weak_or_absent_affordances_score_lower_but_not_zero():
    current = make_profile(duration_sec=200.0, mix_out_points=())
    candidate = make_profile(mix_in_points=())
    state = DJState(current_song=current, current_position=0.0)
    result = score_structure(current, candidate, state, CFG)
    assert 0.0 < result.value < 60.0
    assert result.confidence < CFG.has_affordance_confidence


def test_exit_points_before_current_position_are_ignored():
    current = make_profile(
        duration_sec=200.0,
        mix_out_points=(
            MixPointRef(time=10.0, reasons=(), confidence=0.95, section_id="s"),  # already passed
            MixPointRef(time=150.0, reasons=(), confidence=0.4, section_id="s"),
        ),
    )
    candidate = make_profile()
    state = DJState(current_song=current, current_position=100.0)
    result = score_structure(current, candidate, state, CFG)
    assert result.explanation["best_exit_time"] == 150.0


def test_shared_reason_gives_a_bonus():
    current = make_profile(
        duration_sec=200.0,
        mix_out_points=(MixPointRef(time=180.0, reasons=("clean_downbeat",), confidence=0.6, section_id="s"),),
    )
    candidate_shared = make_profile(
        mix_in_points=(MixPointRef(time=0.0, reasons=("clean_downbeat",), confidence=0.6, section_id="s"),)
    )
    candidate_unshared = make_profile(
        mix_in_points=(MixPointRef(time=0.0, reasons=("low_vocal_density",), confidence=0.6, section_id="s"),)
    )
    state = DJState(current_song=current, current_position=0.0)
    shared = score_structure(current, candidate_shared, state, CFG).value
    unshared = score_structure(current, candidate_unshared, state, CFG).value
    assert shared > unshared


def test_loop_candidates_give_a_bonus():
    current = make_profile(duration_sec=200.0)
    with_loops = make_profile(loop_candidates=(LoopRef(0, 8, 4, 0.9),))
    without_loops = make_profile(loop_candidates=())
    state = DJState(current_song=current, current_position=0.0)
    assert score_structure(current, with_loops, state, CFG).value > score_structure(current, without_loops, state, CFG).value


def test_no_current_song_only_considers_entry():
    candidate = make_profile(mix_in_points=(MixPointRef(time=0.0, reasons=(), confidence=0.9, section_id="s"),))
    state = DJState(current_song=None)
    result = score_structure(None, candidate, state, CFG)
    assert result.value == 90.0


def test_score_bounded_0_100():
    current = make_profile(duration_sec=200.0)
    state = DJState(current_song=current, current_position=0.0)
    for conf in (0.0, 0.5, 1.0):
        candidate = make_profile(mix_in_points=(MixPointRef(time=0.0, reasons=(), confidence=conf, section_id="s"),))
        result = score_structure(current, candidate, state, CFG)
        assert 0.0 <= result.value <= 100.0
