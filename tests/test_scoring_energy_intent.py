"""Explicit energy intent must be able to dominate the ranking, not just
nudge it -- see EnergyIntentStrength / EnergyIntentConfig / _energy_weight_
multiplier in scorer.py. These are deliberately adversarial scenarios: every
non-energy component is stacked *against* the candidate that actually
delivers the requested direction, so a passing test means the mechanism is
strong enough to matter in practice, not just in an isolated unit test."""

from __future__ import annotations

import numpy as np

from songscoring.scorer import score_candidate
from songscoring.state import DJState, EnergyDirection, EnergyIntentStrength
from scoring_helpers import make_profile

DURATION = 200.0


def _flat(level: float, trend: float = 0.0) -> dict:
    times = np.linspace(0, DURATION, 200)
    return dict(
        composite_energy=np.full_like(times, level),
        energy_times=times,
        energy_trend=trend,
        overall_energy=level,
    )


def _current() -> "SongProfile":
    return make_profile(
        id="current", bpm=120.0, key="C major", key_root="C", key_mode="major", genre="Pop", **_flat(0.5)
    )


def _rising_candidate() -> "SongProfile":
    # Energy: rises sharply and keeps climbing. Everything else: a poor match.
    return make_profile(
        id="rising", bpm=171.0, key="F# major", key_root="F#", key_mode="major", genre="Metal", **_flat(0.95, trend=0.002)
    )


def _falling_candidate() -> "SongProfile":
    # Energy: drops sharply and keeps falling. Everything else: a poor match.
    return make_profile(
        id="falling", bpm=171.0, key="F# major", key_root="F#", key_mode="major", genre="Metal", **_flat(0.05, trend=-0.002)
    )


def _flat_continuation_candidate() -> "SongProfile":
    # Energy: stays right where the set already is. Everything else: a poor match.
    return make_profile(
        id="flat", bpm=171.0, key="F# major", key_root="F#", key_mode="major", genre="Metal", **_flat(0.5, trend=0.0)
    )


def _well_matched_but_wrong_direction_candidate(level: float, trend: float) -> "SongProfile":
    # Tempo/key/genre all match the current song closely -- everything a
    # one-step scorer would love -- but its energy does the *opposite* of
    # what's being asked for.
    return make_profile(
        id="well-matched", bpm=120.0, key="C major", key_root="C", key_mode="major", genre="Pop", **_flat(level, trend=trend)
    )


def test_explicit_build_favors_the_upward_candidate_over_a_better_matched_downward_one():
    current = _current()
    rising = _rising_candidate()
    well_matched_falling = _well_matched_but_wrong_direction_candidate(0.1, trend=-0.002)

    state = DJState(
        current_song=current,
        desired_energy_direction=EnergyDirection.INCREASE,
        energy_intent_strength=EnergyIntentStrength.EXPLICIT,
    )
    rising_score = score_candidate(state, rising)
    matched_score = score_candidate(state, well_matched_falling)
    assert rising_score.total_score > matched_score.total_score


def test_explicit_release_favors_the_downward_candidate_over_a_better_matched_upward_one():
    current = _current()
    falling = _falling_candidate()
    well_matched_rising = _well_matched_but_wrong_direction_candidate(0.95, trend=0.002)

    state = DJState(
        current_song=current,
        desired_energy_direction=EnergyDirection.DECREASE,
        energy_intent_strength=EnergyIntentStrength.EXPLICIT,
    )
    falling_score = score_candidate(state, falling)
    matched_score = score_candidate(state, well_matched_rising)
    assert falling_score.total_score > matched_score.total_score


def test_explicit_maintain_favors_a_flat_continuation_over_better_matched_but_directional_moves():
    current = _current()
    flat_continuation = _flat_continuation_candidate()
    well_matched_rising = _well_matched_but_wrong_direction_candidate(0.95, trend=0.002)
    well_matched_falling = _well_matched_but_wrong_direction_candidate(0.1, trend=-0.002)

    state = DJState(
        current_song=current,
        desired_energy_direction=EnergyDirection.MAINTAIN,
        energy_intent_strength=EnergyIntentStrength.EXPLICIT,
    )
    flat_score = score_candidate(state, flat_continuation)
    rising_score = score_candidate(state, well_matched_rising)
    falling_score = score_candidate(state, well_matched_falling)
    assert flat_score.total_score > rising_score.total_score
    assert flat_score.total_score > falling_score.total_score


def test_mild_intent_is_weaker_than_explicit_but_still_directional():
    """A mild preference should still lean toward the requested direction,
    just with less override power than an explicit one -- both are
    evaluated here on the same adversarial pair used for the BUILD test."""
    current = _current()
    rising = _rising_candidate()
    well_matched_falling = _well_matched_but_wrong_direction_candidate(0.1, trend=-0.002)

    mild_state = DJState(
        current_song=current,
        desired_energy_direction=EnergyDirection.INCREASE,
        energy_intent_strength=EnergyIntentStrength.MILD,
    )
    explicit_state = DJState(
        current_song=current,
        desired_energy_direction=EnergyDirection.INCREASE,
        energy_intent_strength=EnergyIntentStrength.EXPLICIT,
    )

    mild_gap = score_candidate(mild_state, rising).total_score - score_candidate(mild_state, well_matched_falling).total_score
    explicit_gap = (
        score_candidate(explicit_state, rising).total_score
        - score_candidate(explicit_state, well_matched_falling).total_score
    )
    # Mild already favors the rising candidate (a stated direction is still a
    # direction), but explicit should widen the margin further.
    assert mild_gap > 0
    assert explicit_gap > mild_gap


def test_without_explicit_direction_the_well_matched_candidate_still_wins():
    """The core regression guard: with no desired_energy_direction at all,
    the energy-intent multiplier must stay neutral and the original
    balanced-scoring behavior (tempo/harmonic/style compatibility wins over
    a raw energy jump) must be unchanged."""
    current = _current()
    rising = _rising_candidate()
    well_matched_falling = _well_matched_but_wrong_direction_candidate(0.1, trend=-0.002)

    state = DJState(current_song=current)  # no desired_energy_direction
    rising_score = score_candidate(state, rising)
    matched_score = score_candidate(state, well_matched_falling)
    assert matched_score.total_score > rising_score.total_score


def test_energy_intent_strength_is_irrelevant_without_a_desired_direction():
    """energy_intent_strength defaults to EXPLICIT on DJState so that just
    setting a direction reads as wanting it -- but with no direction set at
    all, that field must not matter: the multiplier is always neutral."""
    current = _current()
    rising = _rising_candidate()
    well_matched_falling = _well_matched_but_wrong_direction_candidate(0.1, trend=-0.002)

    default_state = DJState(current_song=current)
    explicit_but_no_direction = DJState(current_song=current, energy_intent_strength=EnergyIntentStrength.EXPLICIT)
    none_but_no_direction = DJState(current_song=current, energy_intent_strength=EnergyIntentStrength.NONE)

    for state in (default_state, explicit_but_no_direction, none_but_no_direction):
        rising_score = score_candidate(state, rising).total_score
        matched_score = score_candidate(state, well_matched_falling).total_score
        assert matched_score > rising_score
