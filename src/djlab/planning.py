"""Adapter between the UI's plan request and the real look-ahead planner.

Everything in this module is unit conversion and state construction -- a
direction string becomes an ``EnergyDirection``, a song count or target
duration becomes a beam-search ``horizon``, a starting song id becomes a
``DJState``. The actual planning decision happens entirely inside
``songplanner.plan_sequence``; nothing here re-implements, shortcuts, or
second-guesses it.

Transition Selection (the layer after the planner in the project
architecture) does not exist yet. ``transition_placeholder`` below reflects
that honestly -- a clearly-labeled "not implemented" marker, plus the raw
structural-affordance signal the one-step scorer already computes and that
future layer will eventually consume -- rather than inventing a stand-in
transition-choice heuristic in the UI layer.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Any

from songscoring.song_profile import SongProfile
from songscoring.state import DJState, EnergyDirection
from songscoring.types import CandidateScore
from songplanner.config import DEFAULT_PLANNER_CONFIG, PlannerConfig
from songplanner.planner import plan_sequence
from songplanner.types import Plan

#: UI-facing direction labels -> the engine's own EnergyDirection enum.
#: "none" intentionally maps to ``None`` -- see EnergyDirection's docstring
#: for why that's meaningfully different from "prefer no change".
DIRECTION_LABELS: dict[str, EnergyDirection | None] = {
    "none": None,
    "build": EnergyDirection.INCREASE,
    "maintain": EnergyDirection.MAINTAIN,
    "release": EnergyDirection.DECREASE,
}

MIN_HORIZON = 1
MAX_HORIZON = 8

#: How far into the starting song we assume playback has reached when
#: asking "what comes next" -- the same near-the-end convention already
#: used throughout songscoring/songplanner's own examples and evaluation
#: scripts, not a new modeling choice made for this UI.
ASSUMED_POSITION_FROM_END_SEC = 30.0


class UnknownDirectionError(ValueError):
    pass


def resolve_direction(direction: str) -> EnergyDirection | None:
    try:
        return DIRECTION_LABELS[direction]
    except KeyError:
        raise UnknownDirectionError(
            f"Unknown energy direction {direction!r}; expected one of {sorted(DIRECTION_LABELS)}"
        ) from None


def resolve_horizon(num_songs: int | None, duration_minutes: float | None, library: list[SongProfile]) -> int:
    """Turn "how much music" (a song count, or an approximate duration)
    into the planner's ``horizon``. A song count is used directly (clamped
    to a sane range); a duration is converted using the library's own
    average song length -- still just arithmetic, not a planning decision."""
    if num_songs is not None:
        return max(MIN_HORIZON, min(MAX_HORIZON, int(num_songs)))
    if duration_minutes is not None:
        durations = [s.duration_sec for s in library if s.duration_sec > 0]
        avg_duration = statistics.fmean(durations) if durations else 240.0
        estimated = round((duration_minutes * 60.0) / avg_duration)
        return max(MIN_HORIZON, min(MAX_HORIZON, estimated))
    return DEFAULT_PLANNER_CONFIG.beam.horizon


def build_dj_state(starting_song: SongProfile, direction: EnergyDirection | None) -> DJState:
    position = max(0.0, starting_song.duration_sec - ASSUMED_POSITION_FROM_END_SEC)
    return DJState(current_song=starting_song, current_position=position, desired_energy_direction=direction)


def build_planner_config(horizon: int) -> PlannerConfig:
    return replace(DEFAULT_PLANNER_CONFIG, beam=replace(DEFAULT_PLANNER_CONFIG.beam, horizon=horizon))


def generate_plan(
    starting_song: SongProfile,
    library: list[SongProfile],
    *,
    direction: str,
    num_songs: int | None = None,
    duration_minutes: float | None = None,
) -> Plan:
    """The one place the UI touches the planner.

    ``library`` should already be limited to *analyzed* songs -- an
    unanalyzed song has no ``SongProfile`` to score against, so it's the
    caller's job (see ``server.py``) to filter and report how many were
    excluded, not this function's.
    """
    energy_direction = resolve_direction(direction)
    state = build_dj_state(starting_song, energy_direction)
    horizon = resolve_horizon(num_songs, duration_minutes, library)
    config = build_planner_config(horizon)
    candidates = [song for song in library if song.id != starting_song.id]
    return plan_sequence(state, candidates, config)


def transition_placeholder(transition_score: CandidateScore) -> dict[str, Any]:
    """Transition Selection is not built yet -- this is an explicit,
    honest placeholder plus the one signal already computed today
    (structural mix-in/mix-out affordance compatibility) that a real
    transition-selection layer would consume. It is never presented as a
    chosen technique."""
    structure = transition_score.structure_score
    return {
        "status": "not_implemented",
        "note": "Transition Selection layer not yet built -- no technique has been chosen.",
        "available_signal": {
            "structure_affordance_score": round(structure.value, 2),
            "structure_affordance_confidence": round(structure.confidence, 3),
            "explanation": structure.explanation,
        },
    }


def plan_response_dict(plan: Plan, *, excluded_unanalyzed_count: int) -> dict[str, Any]:
    """Full JSON response for ``POST /api/plan`` -- the plan's own
    ``to_dict()`` unmodified, plus UI-facing context that isn't part of the
    planner's own output (transition placeholders, how many songs in the
    library couldn't be considered, and whether rendering/playback is
    available yet)."""
    data = plan.to_dict()
    data["transitions"] = [transition_placeholder(step.transition_score) for step in plan.steps]
    data["excluded_unanalyzed_count"] = excluded_unanalyzed_count
    data["render_available"] = False
    return data
