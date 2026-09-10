"""Structural transition opportunity.

Scores *whether a good transition point exists*, using the DJ affordances
the analysis layer already found (mix-in/out points, loop candidates) --
never recomputes or reinterprets the underlying audio, and never decides
*how* a transition would be performed (that's a future layer's job).

The exit side only considers mix-out points at or after the set's current
playback position -- a point earlier in the song than "now" isn't a usable
exit anymore.
"""

from __future__ import annotations

import math

from songscoring.config import StructureConfig
from songscoring.song_profile import SongProfile
from songscoring.state import DJState
from songscoring.types import ComponentScore


def score_structure(
    current: SongProfile | None, candidate: SongProfile, state: DJState, config: StructureConfig
) -> ComponentScore:
    entry_point = candidate.best_mix_in()
    entry_conf = entry_point.confidence if entry_point else config.no_affordance_baseline

    if current is None:
        value = 100.0 * entry_conf
        return ComponentScore(
            value=max(0.0, min(100.0, value)),
            confidence=config.has_affordance_confidence if entry_point else config.fallback_confidence,
            explanation={
                "reason": "no current song (start of set); only the candidate's entry point matters",
                "best_entry_time": entry_point.time if entry_point else None,
                "best_entry_confidence": round(entry_conf, 3),
            },
        )

    exit_point = current.best_mix_out(after_position=state.current_position)
    exit_conf = exit_point.confidence if exit_point else config.no_affordance_baseline

    value = 100.0 * math.sqrt(max(exit_conf, 0.0) * max(entry_conf, 0.0))

    if exit_point and entry_point:
        shared_reasons = set(exit_point.reasons) & set(entry_point.reasons)
        if shared_reasons:
            value += config.shared_reason_bonus

    if candidate.loop_candidates:
        best_loop_conf = max(loop.confidence for loop in candidate.loop_candidates)
        value += config.loop_availability_bonus * best_loop_conf

    value = max(0.0, min(100.0, value))
    confidence = config.has_affordance_confidence if (exit_point and entry_point) else config.fallback_confidence

    return ComponentScore(
        value=value,
        confidence=confidence,
        explanation={
            "best_exit_time": exit_point.time if exit_point else None,
            "best_exit_confidence": round(exit_conf, 3),
            "best_exit_reasons": list(exit_point.reasons) if exit_point else [],
            "best_entry_time": entry_point.time if entry_point else None,
            "best_entry_confidence": round(entry_conf, 3),
            "best_entry_reasons": list(entry_point.reasons) if entry_point else [],
            "candidate_loop_candidates": len(candidate.loop_candidates),
        },
    )
