"""Test-only helpers for songplanner tests."""

from __future__ import annotations

from songscoring.types import CandidateScore, ComponentScore


def make_candidate_score(
    candidate_id: str, total_score: float, confidence: float = 0.8, energy_confidence: float | None = None
) -> CandidateScore:
    """A ``CandidateScore`` with every component set to the same
    ``(total_score, confidence)`` pair -- convenient for sequence-scoring
    unit tests that only care about the aggregate numbers, not which
    one-step component produced them.

    ``energy_confidence`` optionally overrides just the energy component's
    confidence (leaving its value and every other component at
    ``confidence``) -- for tests of the sequence-level trajectory term,
    whose confidence tracks the energy component specifically rather than
    the blended overall confidence (see sequence_scoring.score_sequence).
    """
    component = ComponentScore(value=total_score, confidence=confidence)
    energy_component = (
        component if energy_confidence is None else ComponentScore(value=total_score, confidence=energy_confidence)
    )
    return CandidateScore(
        candidate_id=candidate_id,
        total_score=total_score,
        confidence=confidence,
        tempo_score=component,
        harmonic_score=component,
        energy_score=energy_component,
        rhythm_score=component,
        structure_score=component,
        style_score=component,
        familiarity_score=component,
        repetition_penalty=component,
    )
