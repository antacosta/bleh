"""End-to-end look-ahead planner scenarios, all against the real, validated
25-song corpus (see tests/real_corpus.py) rather than synthetic profiles --
these are the scenarios the planning-layer brief asked for explicitly."""

from __future__ import annotations

import json
from dataclasses import replace

from songscoring.state import DJState, EnergyDirection
from songplanner.config import PlannerConfig
from songplanner.planner import plan_sequence
from real_corpus import load_real_corpus

CORPUS = load_real_corpus()
LIBRARY = list(CORPUS.values())


def _energy_levels(plan) -> list[float]:
    return list(plan.sequence_score.energy_levels)


def test_a_build_trends_upward_under_explicit_increase():
    current = CORPUS["15_rnb_kontraa_favor"]  # one of the lowest-energy tracks in the corpus
    state = DJState(current_song=current, current_position=current.duration_sec - 20.0, desired_energy_direction=EnergyDirection.INCREASE)

    plan = plan_sequence(state, LIBRARY)

    levels = _energy_levels(plan)
    assert levels[-1] > levels[0]
    aligned_steps = sum(1 for a, b in zip(levels, levels[1:]) if b >= a)
    assert aligned_steps >= len(levels) - 2  # allow one non-monotonic step -- not requiring perfect monotonicity
    assert plan.sequence_score.components["trajectory"].value > 60.0


def test_b_release_trends_downward_without_instant_collapse():
    current = CORPUS["09_ska_talco_tortuga"]  # one of the highest-energy tracks in the corpus
    state = DJState(current_song=current, current_position=current.duration_sec - 20.0, desired_energy_direction=EnergyDirection.DECREASE)

    plan = plan_sequence(state, LIBRARY)

    levels = _energy_levels(plan)
    assert levels[-1] < levels[0]
    assert not any("instant collapse" in r for r in plan.sequence_score.reasons)


def test_c_no_explicit_direction_produces_a_balanced_trajectory():
    current = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    state = DJState(current_song=current, current_position=current.duration_sec - 20.0)

    plan = plan_sequence(state, LIBRARY)

    assert plan.achieved_horizon == plan.requested_horizon
    assert any("no explicit direction" in r for r in plan.sequence_score.reasons)
    assert 0.0 <= plan.sequence_score.path_score <= 100.0


def test_d_local_vs_global_prefers_the_better_full_path_over_the_best_immediate_move():
    """The most important conceptual test: the planner must be willing to
    open with a song that scores lower one step ahead than an available
    alternative, because it leads to a better overall sequence -- this is a
    real, naturally-occurring instance in the validated corpus, not a
    contrived one."""
    current = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    state = DJState(current_song=current, current_position=current.duration_sec - 30.0, desired_energy_direction=EnergyDirection.INCREASE)

    plan = plan_sequence(state, LIBRARY)

    completed = [c for c in plan.local_vs_global if c.depth_reached == plan.requested_horizon]
    assert len(completed) >= 2  # need at least two comparable full-length paths for this to mean anything

    best_immediate = max(plan.local_vs_global, key=lambda c: c.immediate_transition_score)
    chosen = next(c for c in plan.local_vs_global if c.chosen)

    # The winner is the best among *completed* paths...
    assert chosen.downstream_path_score == max(c.downstream_path_score for c in completed)
    # ...but was not the move that looked best one step ahead.
    assert chosen.song_id != best_immediate.song_id
    assert chosen.immediate_transition_score < best_immediate.immediate_transition_score


def test_e_avoids_repeating_an_artist_within_the_plan_despite_high_one_step_scores():
    """Professor Kliq has three tracks in the corpus (11, 12, 13), all
    mutually tempo/style-compatible -- exactly the situation where a
    one-step-only planner would happily stack several of them back to
    back. The sequence-level variety penalty (plus the one-step repetition
    component, fed the extended plan history) must stop that."""
    current = CORPUS["11_electronic_kliq_bust_this_bust_that"]
    state = DJState(current_song=current, current_position=current.duration_sec - 20.0)

    # A small, adversarial library: the other two Professor Kliq tracks
    # (tempting, same-artist repeats) plus a handful of otherwise-compatible
    # alternatives from other artists.
    small_library = [
        CORPUS["12_electronic_kliq_plastic_flashing_lights"],
        CORPUS["13_electronic_kliq_surfs_up"],
        CORPUS["02_pop_bryyn_so_well"],
        CORPUS["06_altrock_bshake_insane"],
        CORPUS["23_jazz_lucidi_dehi"],
        CORPUS["01_pop_pornophonique_sad_robot"],
    ]
    config = PlannerConfig(beam=replace(PlannerConfig().beam, horizon=3))

    plan = plan_sequence(state, small_library, config)

    artists = [step.artist for step in plan.steps]
    assert len(artists) == len(set(artists)), f"artist repeated within plan: {artists}"


def test_f_variety_and_coherent_style_movement_no_genre_bounce():
    current = CORPUS["03_rock_antarhes_still_fighting"]  # genre: Rock
    state = DJState(current_song=current, current_position=current.duration_sec - 20.0)

    small_library = [
        CORPUS["10_ska_talco_combat_circus"],  # Ska
        CORPUS["05_rock_infadedglory_dead_can"],  # Rock
        CORPUS["09_ska_talco_tortuga"],  # Ska
        CORPUS["23_jazz_lucidi_dehi"],  # Jazz
        CORPUS["01_pop_pornophonique_sad_robot"],  # Pop
    ]
    config = PlannerConfig(beam=replace(PlannerConfig().beam, horizon=3))

    plan = plan_sequence(state, small_library, config)

    genres = [current.genre] + [step.genre for step in plan.steps]
    bounces = sum(
        1
        for i in range(2, len(genres))
        if genres[i] and genres[i - 1] and genres[i - 2] and genres[i] == genres[i - 2] and genres[i] != genres[i - 1]
    )
    assert bounces == 0
    assert plan.sequence_score.components["coherence"].value == 100.0


def test_g_low_confidence_analysis_does_not_produce_an_overconfident_plan():
    """A song with most optional analysis fields missing (the same
    real-world degradation the one-step scorer already handles gracefully,
    e.g. test_missing_optional_fields_across_the_board_does_not_crash) must
    not crash the planner, and the plan's reported confidence must
    genuinely reflect that uncertainty rather than defaulting to false
    confidence."""
    degraded = replace(
        CORPUS["12_electronic_kliq_plastic_flashing_lights"],
        bpm=None,
        key=None,
        key_root=None,
        key_mode=None,
        genre=None,
        chroma_mean=None,
        sections=(),
        mix_in_points=(),
        mix_out_points=(),
        loop_candidates=(),
        energy_times=CORPUS["12_electronic_kliq_plastic_flashing_lights"].energy_times[:0],
        composite_energy=CORPUS["12_electronic_kliq_plastic_flashing_lights"].composite_energy[:0],
    )
    state = DJState(current_song=degraded, desired_energy_direction=EnergyDirection.INCREASE)
    degraded_plan = plan_sequence(state, LIBRARY)

    assert 0.0 <= degraded_plan.sequence_score.path_score <= 100.0
    # The first transition -- the only one actually touching the degraded
    # song -- must visibly carry low confidence rather than a fabricated
    # "sure of it" reading.
    assert degraded_plan.steps[0].transition_score.confidence < 0.3

    # Compared to an otherwise-identical request against the same, real,
    # fully-analyzed song, the degraded plan's reported confidence must be
    # lower -- uncertainty in the input must actually show up in the output,
    # not be silently absorbed.
    full_state = DJState(
        current_song=CORPUS["12_electronic_kliq_plastic_flashing_lights"],
        desired_energy_direction=EnergyDirection.INCREASE,
    )
    full_plan = plan_sequence(full_state, LIBRARY)
    assert degraded_plan.sequence_score.path_confidence < full_plan.sequence_score.path_confidence


def test_plan_to_dict_is_json_serializable():
    current = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    state = DJState(current_song=current, desired_energy_direction=EnergyDirection.INCREASE)
    plan = plan_sequence(state, LIBRARY)
    json.dumps(plan.to_dict())  # raises if anything isn't serializable


def test_search_notes_flag_a_constrained_candidate_pool():
    current = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    state = DJState(current_song=current)
    tiny_library = [CORPUS["11_electronic_kliq_bust_this_bust_that"], CORPUS["13_electronic_kliq_surfs_up"]]
    config = PlannerConfig(beam=replace(PlannerConfig().beam, horizon=3, min_pool_size=5))

    plan = plan_sequence(state, tiny_library, config)

    assert any("min_pool_size" in note for note in plan.search_notes)


def test_plan_is_deterministic_across_repeated_calls_and_library_ordering():
    current = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    state = DJState(current_song=current, desired_energy_direction=EnergyDirection.INCREASE)

    first = plan_sequence(state, LIBRARY)
    second = plan_sequence(state, list(reversed(LIBRARY)))

    assert [s.song_id for s in first.steps] == [s.song_id for s in second.steps]
    assert first.sequence_score.path_score == second.sequence_score.path_score
