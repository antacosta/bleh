from __future__ import annotations

import numpy as np

from songscoring.config import EnergyIntentConfig
from songscoring.state import DJState, EnergyDirection, EnergyIntentStrength
from songplanner.config import SequenceScoringConfig
from songplanner.sequence_scoring import score_sequence
from planner_helpers import make_candidate_score
from scoring_helpers import make_profile

SEQ_CFG = SequenceScoringConfig()
INTENT_CFG = EnergyIntentConfig()


def _root(desired=None, strength=EnergyIntentStrength.EXPLICIT, current_energy=0.5, **overrides):
    # The energy anchor comes from ending_energy() over composite_energy,
    # not from overall_energy directly -- so both must be set consistently
    # for current_energy to actually control the trajectory's starting point.
    duration = 200.0
    times = np.linspace(0, duration, 200)
    current = make_profile(
        id="current",
        duration_sec=duration,
        energy_times=times,
        composite_energy=np.full_like(times, current_energy),
        overall_energy=current_energy,
        **overrides,
    )
    return DJState(current_song=current, desired_energy_direction=desired, energy_intent_strength=strength)


def test_build_rewards_a_rising_sequence_over_a_falling_one():
    root_rising = _root(desired=EnergyDirection.INCREASE, current_energy=0.3)
    rising_seq = (
        make_profile(id="a", overall_energy=0.45, artist="A1", genre="G1"),
        make_profile(id="b", overall_energy=0.6, artist="A2", genre="G2"),
        make_profile(id="c", overall_energy=0.75, artist="A3", genre="G3"),
    )
    falling_seq = (
        make_profile(id="d", overall_energy=0.2, artist="A4", genre="G4"),
        make_profile(id="e", overall_energy=0.1, artist="A5", genre="G5"),
        make_profile(id="f", overall_energy=0.05, artist="A6", genre="G6"),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "abc")
    rising = score_sequence(root_rising, rising_seq, step_scores, SEQ_CFG, INTENT_CFG)

    step_scores_f = tuple(make_candidate_score(s, 70.0) for s in "def")
    falling = score_sequence(root_rising, falling_seq, step_scores_f, SEQ_CFG, INTENT_CFG)

    assert rising.components["trajectory"].value > falling.components["trajectory"].value
    assert rising.path_score > falling.path_score


def test_release_rewards_a_falling_sequence_and_flags_instant_collapse():
    root_release = _root(desired=EnergyDirection.DECREASE, current_energy=0.8)
    gradual_seq = (
        make_profile(id="a", overall_energy=0.6, artist="A1", genre="G1"),
        make_profile(id="b", overall_energy=0.45, artist="A2", genre="G2"),
        make_profile(id="c", overall_energy=0.35, artist="A3", genre="G3"),
    )
    collapse_seq = (
        make_profile(id="d", overall_energy=0.1, artist="A4", genre="G4"),  # drop of 0.7 in one step
        make_profile(id="e", overall_energy=0.4, artist="A5", genre="G5"),
        make_profile(id="f", overall_energy=0.35, artist="A6", genre="G6"),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "abc")
    gradual = score_sequence(root_release, gradual_seq, step_scores, SEQ_CFG, INTENT_CFG)

    step_scores_c = tuple(make_candidate_score(s, 70.0) for s in "def")
    collapse = score_sequence(root_release, collapse_seq, step_scores_c, SEQ_CFG, INTENT_CFG)

    assert gradual.components["trajectory"].value > collapse.components["trajectory"].value
    assert any("instant collapse" in r for r in collapse.reasons)
    assert not any("instant collapse" in r for r in gradual.reasons)


def test_maintain_rewards_a_flat_sequence_over_a_volatile_one():
    root_maintain = _root(desired=EnergyDirection.MAINTAIN, current_energy=0.5)
    flat_seq = (
        make_profile(id="a", overall_energy=0.52, artist="A1", genre="G1"),
        make_profile(id="b", overall_energy=0.49, artist="A2", genre="G2"),
        make_profile(id="c", overall_energy=0.51, artist="A3", genre="G3"),
    )
    volatile_seq = (
        make_profile(id="d", overall_energy=0.95, artist="A4", genre="G4"),
        make_profile(id="e", overall_energy=0.1, artist="A5", genre="G5"),
        make_profile(id="f", overall_energy=0.9, artist="A6", genre="G6"),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "abc")
    flat = score_sequence(root_maintain, flat_seq, step_scores, SEQ_CFG, INTENT_CFG)
    step_scores_v = tuple(make_candidate_score(s, 70.0) for s in "def")
    volatile = score_sequence(root_maintain, volatile_seq, step_scores_v, SEQ_CFG, INTENT_CFG)

    assert flat.components["trajectory"].value > volatile.components["trajectory"].value


def test_no_explicit_direction_rewards_smooth_or_full_contrast_over_ambiguous_middle():
    root_none = _root(desired=None, current_energy=0.5)
    smooth_seq = (make_profile(id="a", overall_energy=0.52, artist="A1", genre="G1"),)
    ambiguous_seq = (make_profile(id="b", overall_energy=0.75, artist="A2", genre="G2"),)

    smooth = score_sequence(root_none, smooth_seq, (make_candidate_score("a", 70.0),), SEQ_CFG, INTENT_CFG)
    ambiguous = score_sequence(root_none, ambiguous_seq, (make_candidate_score("b", 70.0),), SEQ_CFG, INTENT_CFG)

    assert smooth.components["trajectory"].value > ambiguous.components["trajectory"].value


def test_variety_penalizes_repeated_song_artist_and_over_repeated_genre():
    root = _root(desired=None)
    sequence = (
        make_profile(id="a", artist="Same Artist", genre="Genre", overall_energy=0.5),
        make_profile(id="b", artist="Same Artist", genre="Genre", overall_energy=0.5),
        make_profile(id="c", artist="Other Artist", genre="Genre", overall_energy=0.5),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "abc")
    result = score_sequence(root, sequence, step_scores, SEQ_CFG, INTENT_CFG)

    assert result.components["variety"].value < 100.0
    assert any("artist repeated" in r for r in result.reasons)
    assert any("genre repeated" in r for r in result.reasons)


def test_variety_also_checks_against_real_recent_history_not_just_the_plan():
    current = make_profile(id="current")
    root = DJState(current_song=current, recent_artists=("Repeat Artist",), recent_songs=("old-song",))
    sequence = (make_profile(id="new-song", artist="Repeat Artist", genre="G", overall_energy=0.5),)
    result = score_sequence(root, sequence, (make_candidate_score("new-song", 70.0),), SEQ_CFG, INTENT_CFG)
    assert any("artist repeated" in r for r in result.reasons)


def test_variety_score_is_perfect_when_nothing_repeats():
    root = _root(desired=None)
    sequence = (
        make_profile(id="a", artist="A1", genre="G1", overall_energy=0.5),
        make_profile(id="b", artist="A2", genre="G2", overall_energy=0.5),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "ab")
    result = score_sequence(root, sequence, step_scores, SEQ_CFG, INTENT_CFG)
    assert result.components["variety"].value == 100.0


def test_coherence_penalizes_genre_bounce_back():
    current = make_profile(id="current", genre="Rock")
    root = DJState(current_song=current)
    sequence = (
        make_profile(id="a", artist="A1", genre="Ska", overall_energy=0.5),
        make_profile(id="b", artist="A2", genre="Rock", overall_energy=0.5),  # bounces back to Rock
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "ab")
    result = score_sequence(root, sequence, step_scores, SEQ_CFG, INTENT_CFG)
    assert result.components["coherence"].value < 100.0
    assert any("bounced back" in r for r in result.reasons)


def test_coherence_does_not_penalize_a_progression_or_a_sustained_run():
    current = make_profile(id="current", genre="Rock")
    root = DJState(current_song=current)
    progression = (
        make_profile(id="a", artist="A1", genre="Ska", overall_energy=0.5),
        make_profile(id="b", artist="A2", genre="Jazz", overall_energy=0.5),
    )
    sustained = (
        make_profile(id="c", artist="A3", genre="Rock", overall_energy=0.5),
        make_profile(id="d", artist="A4", genre="Rock", overall_energy=0.5),
    )
    step_scores = tuple(make_candidate_score(s, 70.0) for s in "ab")
    step_scores2 = tuple(make_candidate_score(s, 70.0) for s in "cd")
    prog_result = score_sequence(root, progression, step_scores, SEQ_CFG, INTENT_CFG)
    sustained_result = score_sequence(root, sustained, step_scores2, SEQ_CFG, INTENT_CFG)
    assert prog_result.components["coherence"].value == 100.0
    assert sustained_result.components["coherence"].value == 100.0


def test_low_energy_confidence_dampens_trajectorys_influence_without_crashing():
    """Uncertainty test: trajectory is fundamentally a claim about energy
    shape, so its confidence tracks the per-step *energy* component's
    confidence specifically (see score_sequence) -- not the whole blended
    one-step confidence, which would let unrelated tempo/harmonic/etc
    certainty paper over an actually-unreliable energy read. Identical
    trajectory *values* at low vs. high energy-confidence must not swing
    the path score just as hard toward that (unreliably-supported) claim."""
    root = _root(desired=EnergyDirection.INCREASE, current_energy=0.3)
    sequence = (make_profile(id="a", overall_energy=0.9, artist="A1", genre="G1"),)  # strong, ideal BUILD delta

    # Overall one-step confidence and total_score are identical in both
    # cases -- only the energy-specific confidence differs.
    confident = score_sequence(
        root, sequence, (make_candidate_score("a", 40.0, confidence=0.9, energy_confidence=0.9),), SEQ_CFG, INTENT_CFG
    )
    uncertain = score_sequence(
        root, sequence, (make_candidate_score("a", 40.0, confidence=0.9, energy_confidence=0.0),), SEQ_CFG, INTENT_CFG
    )

    assert uncertain.components["trajectory"].confidence < confident.components["trajectory"].confidence
    # Both see the identical strong upward trajectory value...
    assert confident.components["trajectory"].value == uncertain.components["trajectory"].value
    # ...but the low-energy-confidence path's score sits closer to the
    # (lower) transition value, since trajectory's effective weight was
    # dampened rather than blindly trusted.
    transition_value = confident.components["transition"].value
    trajectory_value = confident.components["trajectory"].value
    assert abs(uncertain.path_score - transition_value) < abs(confident.path_score - transition_value)
    assert trajectory_value > transition_value  # sanity: trajectory is the more optimistic term here
    # And nothing crashes or produces an out-of-range score at either extreme.
    assert 0.0 <= uncertain.path_score <= 100.0


def test_empty_sequence_scores_neutrally_without_crashing():
    root = _root(desired=EnergyDirection.INCREASE)
    result = score_sequence(root, (), (), SEQ_CFG, INTENT_CFG)
    assert result.path_score == 100.0
    assert result.path_confidence == 1.0
