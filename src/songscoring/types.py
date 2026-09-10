"""Shared result types for the scoring layer.

A :class:`ComponentScore` always separates *what we think* (``value``) from
*how much we trust that assessment* (``confidence``). Keeping these apart is
what makes the system confidence-aware without silently distorting numbers:
a "keys are a perfect match, but we're not sure either key is right" case and
a "keys are a 50/50 match, and we're quite sure" case must not collapse into
the same number, or the whole system stops being inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentScore:
    """One scoring dimension's verdict on a single candidate.

    Attributes:
        value: 0..100, higher is always better for this dimension, on this
            same scale for every component (including ``repetition_penalty``,
            where 100 means "no repetition concern" -- see its module).
        confidence: 0..1, how much the *underlying analysis* backing this
            judgement should be trusted (not how good the match is). Low
            confidence means "we don't really know", not "this is bad".
        explanation: small, JSON-serializable dict of the concrete numbers
            that produced ``value`` -- this is what makes the score
            inspectable rather than a black box.
    """

    value: float
    confidence: float
    explanation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 100.0):
            raise ValueError(f"ComponentScore.value must be in [0, 100], got {self.value}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ComponentScore.confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class CandidateScore:
    """Full, inspectable scoring result for one candidate song."""

    candidate_id: str
    total_score: float
    confidence: float
    tempo_score: ComponentScore
    harmonic_score: ComponentScore
    energy_score: ComponentScore
    rhythm_score: ComponentScore
    structure_score: ComponentScore
    style_score: ComponentScore
    familiarity_score: ComponentScore
    repetition_penalty: ComponentScore

    def components(self) -> dict[str, ComponentScore]:
        """All eight components as a name -> ComponentScore mapping, in the
        same order the central weights are documented in (config.py)."""
        return {
            "tempo": self.tempo_score,
            "harmonic": self.harmonic_score,
            "energy": self.energy_score,
            "rhythm": self.rhythm_score,
            "structure": self.structure_score,
            "style": self.style_score,
            "familiarity": self.familiarity_score,
            "repetition": self.repetition_penalty,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "total_score": round(self.total_score, 2),
            "confidence": round(self.confidence, 3),
            **{
                f"{name}_score" if name != "repetition" else "repetition_penalty": {
                    "value": round(c.value, 2),
                    "confidence": round(c.confidence, 3),
                    "explanation": c.explanation,
                }
                for name, c in self.components().items()
            },
        }
