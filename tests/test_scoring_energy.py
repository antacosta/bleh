from __future__ import annotations

import numpy as np

from songscoring.components.energy import score_energy
from songscoring.config import EnergyConfig, EnergyIntentConfig
from songscoring.state import DJState, EnergyDirection
from scoring_helpers import make_profile

CFG = EnergyConfig()
INTENT_CFG = EnergyIntentConfig()


def _score(current, candidate, state):
    return score_energy(current, candidate, state, CFG, INTENT_CFG)


def _flat_profile(level: float, duration: float = 200.0) -> dict:
    times = np.linspace(0, duration, 200)
    return dict(duration_sec=duration, energy_times=times, composite_energy=np.full_like(times, level))


def test_smooth_continuation_scores_well_with_no_desired_direction():
    current = make_profile(**_flat_profile(0.5))
    candidate = make_profile(**_flat_profile(0.5))
    state = DJState(current_song=current)
    result = _score(current, candidate, state)
    assert result.value > 85


def test_large_contrast_is_not_penalized_without_a_desired_direction():
    """Core principle: a candidate must not be penalized just for having a
    substantially different energy -- intentional contrast is supported."""
    current = make_profile(**_flat_profile(0.9))
    quiet_candidate = make_profile(**_flat_profile(0.1))
    similar_candidate = make_profile(**_flat_profile(0.85))
    state = DJState(current_song=current)

    contrast = _score(current, quiet_candidate, state).value
    smooth = _score(current, similar_candidate, state).value
    # Neither is "bad"; a full contrast should not score far below a smooth one.
    assert contrast > 65
    assert smooth > 65


def test_desired_increase_rewards_candidates_that_deliver_it():
    current = make_profile(**_flat_profile(0.3))
    rising_candidate = make_profile(**_flat_profile(0.75))
    flat_candidate = make_profile(**_flat_profile(0.3))
    state = DJState(current_song=current, desired_energy_direction=EnergyDirection.INCREASE)

    rising = _score(current, rising_candidate, state).value
    flat = _score(current, flat_candidate, state).value
    assert rising > flat


def test_desired_reset_rewards_a_quiet_opening_regardless_of_delta():
    current = make_profile(**_flat_profile(0.4))  # only a moderate ending energy
    quiet_candidate = make_profile(**_flat_profile(0.05))
    loud_candidate = make_profile(**_flat_profile(0.9))
    state = DJState(current_song=current, desired_energy_direction=EnergyDirection.RESET)

    quiet = _score(current, quiet_candidate, state).value
    loud = _score(current, loud_candidate, state).value
    assert quiet > loud


def test_current_position_moves_the_ending_energy_window():
    duration = 200.0
    times = np.linspace(0, duration, 400)
    # energy ramps from 0.1 to 0.9 across the song
    composite = np.linspace(0.1, 0.9, 400)
    current = make_profile(duration_sec=duration, energy_times=times, composite_energy=composite)
    candidate = make_profile(**_flat_profile(0.9))

    # Early in the song, "ending energy" falls back to the song's natural
    # tail window (duration - window_sec .. duration) regardless of position.
    early_state = DJState(current_song=current, current_position=0.0)
    # Deep into the tail (past where the natural window would start), the
    # window should start at "now" instead, catching only the very end of
    # the up-ramp -- higher than the wider natural-tail average.
    late_state = DJState(current_song=current, current_position=duration - 2.0)

    early = _score(current, candidate, early_state)
    late = _score(current, candidate, late_state)
    assert late.explanation["current_ending_energy"] > early.explanation["current_ending_energy"]


def test_no_current_song_does_not_penalize_any_opening():
    candidate = make_profile(**_flat_profile(0.9))
    state = DJState(current_song=None)
    result = _score(None, candidate, state)
    assert result.value == 100.0


def test_missing_energy_data_is_neutral_and_zero_confidence():
    current = make_profile(energy_times=np.array([]), composite_energy=np.array([]))
    candidate = make_profile(**_flat_profile(0.5))
    state = DJState(current_song=current)
    result = _score(current, candidate, state)
    assert result.value == 50.0
    assert result.confidence == 0.0


def test_score_bounded_0_100():
    current = make_profile(**_flat_profile(0.6))
    state = DJState(current_song=current)
    for level in (0.0, 0.01, 0.5, 0.99, 1.0):
        result = _score(current, make_profile(**_flat_profile(level)), state)
        assert 0.0 <= result.value <= 100.0
