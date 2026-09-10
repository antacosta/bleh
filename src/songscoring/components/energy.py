"""Energy progression: how the candidate's opening continues (or contrasts
with) the outgoing song's ending.

Two important, deliberate design choices:

1. ``composite_energy`` is normalized *per song* to that song's own
   min/max (see songanalysis.features.energy) -- it is not an absolute
   loudness scale, and comparing it across two songs is not "song A is
   objectively louder than song B". What it *does* tell us, correctly, is
   "where in this song's own dynamic arc are we right now" -- which is
   exactly the right question for "is this ending winding down or still
   at its peak" and "does this candidate open at its own quiet floor or
   its own full energy". That is the comparison made here.

2. Without an explicit desired direction, a big energy jump is not
   penalized. The brief is explicit that intentional contrast is a
   legitimate, sometimes-desirable move -- so the *no-preference* scoring
   only mildly rewards the two clearly-coherent shapes (a smooth
   continuation, or a full, clean contrast) over an ambiguous
   in-between wobble, rather than rewarding small deltas the way a naive
   "closer is better" model would.
"""

from __future__ import annotations

import math

from songscoring.config import EnergyConfig
from songscoring.song_profile import SongProfile
from songscoring.state import DJState, EnergyDirection
from songscoring.types import ComponentScore

_DIRECTION_IDEAL_DELTA = {
    EnergyDirection.INCREASE: 0.45,
    EnergyDirection.DECREASE: -0.45,
    EnergyDirection.MAINTAIN: 0.0,
}


def _classify(delta: float, candidate_start: float, config: EnergyConfig) -> str:
    if candidate_start <= config.reset_start_threshold and delta <= -config.reset_drop_threshold:
        return "reset"
    if delta > config.maintain_threshold:
        return "increase"
    if delta < -config.maintain_threshold:
        return "decrease"
    return "maintain"


def _neutral_shape_score(delta: float) -> float:
    """No explicit desired direction: reward a clean continuation (delta
    near 0) or a clean full contrast (|delta| near 1) about equally, and
    the ambiguous middle a bit less -- a smooth U-shape in |delta|, floor
    75 so no shape is ever treated as "bad" by default."""
    magnitude = abs(delta)
    dip = math.sin(magnitude * math.pi) ** 2  # 0 at magnitude=0 and 1, peaks mid-range
    return 100.0 - 25.0 * dip


def score_energy(
    current: SongProfile | None, candidate: SongProfile, state: DJState, config: EnergyConfig
) -> ComponentScore:
    if current is None:
        candidate_start = candidate.starting_energy(config.window_sec)
        conf = 0.3 if candidate_start is not None else 0.0
        return ComponentScore(
            value=100.0 if candidate_start is not None else 50.0,
            confidence=conf,
            explanation={"reason": "no current song (start of set) -- any opening energy is fine"},
        )

    current_ending = current.ending_energy(config.window_sec, from_position=state.current_position)
    candidate_start = candidate.starting_energy(config.window_sec)

    if current_ending is None or candidate_start is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "insufficient energy timeline data"})

    delta = candidate_start - current_ending
    shape = _classify(delta, candidate_start, config)

    desired = state.desired_energy_direction
    if desired is None:
        value = _neutral_shape_score(delta)
    elif desired == EnergyDirection.RESET:
        # Reward a low candidate opening regardless of exactly how big the
        # drop is, since "reset" means "bring it down", not "match a ratio".
        value = 100.0 * math.exp(-((candidate_start / max(config.reset_start_threshold, 1e-6)) ** 2))
    else:
        ideal = _DIRECTION_IDEAL_DELTA[desired]
        value = 100.0 * math.exp(-(((delta - ideal) / 0.35) ** 2))

    value = max(0.0, min(100.0, value))

    # Energy is a direct measurement, not an inferred category -- confidence
    # here reflects data availability, not musical ambiguity.
    confidence = 0.9

    return ComponentScore(
        value=value,
        confidence=confidence,
        explanation={
            "current_ending_energy": round(current_ending, 3),
            "candidate_starting_energy": round(candidate_start, 3),
            "delta": round(delta, 3),
            "transition_shape": shape,
            "desired_direction": desired.value if desired else None,
            "candidate_energy_trend": round(candidate.energy_trend, 5),
        },
    )
