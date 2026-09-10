"""Result types for a completed plan -- the planner's explainability output.

Every field here is either a number, a structured collection of numbers, or
a plain string built by formatting numbers already computed elsewhere in
this package (see ``sequence_scoring.py``). Nothing in this module -- or
anywhere in ``songplanner`` -- generates natural language via an LLM; the
"reasons" are deterministic, inspectable strings, not model-authored prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from songscoring.types import CandidateScore

from songplanner.sequence_scoring import SequenceScoreBreakdown


@dataclass(frozen=True)
class PlanStep:
    """One song in the plan and the full one-step breakdown of the
    transition into it (the existing scorer's own output, untouched)."""

    position: int  # 1-based
    song_id: str
    artist: str | None
    genre: str | None
    overall_energy: float
    transition_score: CandidateScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "song_id": self.song_id,
            "artist": self.artist,
            "genre": self.genre,
            "overall_energy": round(self.overall_energy, 3),
            "transition_score": self.transition_score.to_dict(),
        }


@dataclass(frozen=True)
class LocalVsGlobalChoice:
    """One candidate first move considered at the root, and the path score
    it had reached the last time beam search still carried it forward --
    lets the caller see, concretely, when the chosen opening was *not* the
    one with the best immediate one-step score (see planner.plan_sequence).

    ``downstream_path_score`` and ``depth_reached`` describe the *same*
    depth for a given entry, so comparisons across entries are only truly
    apples-to-apples when their ``depth_reached`` values match -- an
    opening pruned from the beam early (a smaller ``depth_reached``) may
    show a score that looks competitive or even higher than the eventual
    winner's, precisely because it reflects a shorter, incomplete path, not
    a worse full plan. That asymmetry is bounded beam search's known,
    accepted trade-off against exhaustive search (see planner.py), not an
    error in this figure -- it is kept visible rather than hidden.
    """

    song_id: str
    immediate_transition_score: float
    downstream_path_score: float
    depth_reached: int
    chosen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "song_id": self.song_id,
            "immediate_transition_score": round(self.immediate_transition_score, 2),
            "downstream_path_score": round(self.downstream_path_score, 2),
            "depth_reached": self.depth_reached,
            "chosen": self.chosen,
        }


@dataclass(frozen=True)
class Plan:
    """A complete look-ahead plan: the chosen sequence, its full scoring
    breakdown, and enough structured context to explain *why* it was chosen
    over the alternatives beam search also considered."""

    steps: tuple[PlanStep, ...]
    sequence_score: SequenceScoreBreakdown
    local_vs_global: tuple[LocalVsGlobalChoice, ...]
    requested_horizon: int
    achieved_horizon: int
    #: Notes about the *search itself* (not the chosen sequence's own
    #: scoring) -- currently just constrained-pool warnings, e.g. "depth 2:
    #: candidate pool size 2 fell below the configured min_pool_size=3".
    #: Empty in the common case; a small/unusual library is the only way to
    #: trigger this (see config.BeamSearchConfig.min_pool_size).
    search_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "sequence_score": self.sequence_score.to_dict(),
            "local_vs_global": [c.to_dict() for c in self.local_vs_global],
            "requested_horizon": self.requested_horizon,
            "achieved_horizon": self.achieved_horizon,
            "search_notes": list(self.search_notes),
            "major_reasons": list(self.sequence_score.reasons),
        }
