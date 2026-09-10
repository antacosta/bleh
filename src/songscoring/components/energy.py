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

from songscoring.config import EnergyConfig, EnergyIntentConfig
from songscoring.song_profile import SongProfile
from songscoring.state import DJState, EnergyDirection, EnergyIntentStrength
from songscoring.types import ComponentScore

_DIRECTION_IDEAL_DELTA = {
    EnergyDirection.INCREASE: 0.45,
    EnergyDirection.DECREASE: -0.45,
    EnergyDirection.MAINTAIN: 0.0,
}

#: Sign of a "good" energy_trend (composite-energy units/sec, see
#: SongProfile.energy_trend) for each direction that isn't MAINTAIN, which
#: instead rewards a trend close to zero. RESET shares INCREASE's opposite --
#: bringing the room down should keep coming down, not immediately climb.
_DIRECTION_TREND_SIGN = {
    EnergyDirection.INCREASE: 1.0,
    EnergyDirection.DECREASE: -1.0,
    EnergyDirection.RESET: -1.0,
}

#: How much of the trend-alignment bonus applies at each intent strength --
#: a mild preference should nudge, not fully commit, while no explicit
#: direction means the bonus never applies at all (see score_energy).
_STRENGTH_SCALE = {
    EnergyIntentStrength.NONE: 0.0,
    EnergyIntentStrength.MILD: 0.4,
    EnergyIntentStrength.EXPLICIT: 1.0,
}

#: Trend magnitude (composite-energy units/sec) treated as "clearly moving"
#: for the purpose of the tanh soft-sign below. Real trends in the validated
#: corpus range roughly 0.00005..0.001, so this sits in the middle of that.
_TREND_SATURATION = 0.0005


def _trend_alignment_bonus(
    trend: float, desired: EnergyDirection | None, strength: EnergyIntentStrength, config: EnergyIntentConfig
) -> float:
    """Extra credit (or penalty) for the candidate's own internal energy
    trajectory agreeing with the requested direction -- distinct from
    ``delta`` above, which only looks at *where it starts*. Under BUILD, a
    candidate that keeps climbing once it's playing is a better choice than
    one that opens at the same level but immediately fades, even though
    both have an identical opening delta.
    """
    if desired is None:
        return 0.0
    strength_scale = _STRENGTH_SCALE.get(strength, 1.0)
    if strength_scale == 0.0:
        return 0.0

    normalized = math.tanh(trend / _TREND_SATURATION)  # soft sign, -1..1
    if desired == EnergyDirection.MAINTAIN:
        alignment = 1.0 - abs(normalized)
    else:
        alignment = normalized * _DIRECTION_TREND_SIGN[desired]

    return config.trend_alignment_bonus * strength_scale * alignment


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
    current: SongProfile | None,
    candidate: SongProfile,
    state: DJState,
    config: EnergyConfig,
    intent_config: EnergyIntentConfig,
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

    trend_bonus = 0.0
    if desired is not None:
        trend_bonus = _trend_alignment_bonus(candidate.energy_trend, desired, state.energy_intent_strength, intent_config)
        value += trend_bonus

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
            "energy_intent_strength": state.energy_intent_strength.value,
            "candidate_energy_trend": round(candidate.energy_trend, 5),
            "trend_alignment_bonus": round(trend_bonus, 3),
        },
    )
