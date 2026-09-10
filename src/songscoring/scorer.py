"""Central scoring entry points: ``score_candidate`` and ``rank_candidates``.

This is the *only* place component scores get combined into a total, and
the *only* place weights get read from config -- no component module
imports ``ScoringWeights`` or does its own weighted combination. That keeps
retuning the model a one-file change.
"""

from __future__ import annotations

from songscoring.components.energy import score_energy
from songscoring.components.familiarity import score_familiarity
from songscoring.components.harmonic import score_harmonic
from songscoring.components.repetition import score_repetition
from songscoring.components.rhythm import score_rhythm
from songscoring.components.structure import score_structure
from songscoring.components.style import score_style
from songscoring.components.tempo import score_tempo
from songscoring.config import DEFAULT_CONFIG, ScoringConfig
from songscoring.song_profile import SongProfile
from songscoring.state import DJState
from songscoring.types import CandidateScore, ComponentScore


def compute_total_score(
    components: dict[str, ComponentScore], weights: dict[str, float], min_effective_weight: float
) -> tuple[float, float]:
    """Confidence-weighted combination of component scores into one total.

    Each component's *base* weight is scaled by its own confidence before
    being folded into the average. A low-confidence component still
    reports its best-guess ``value`` untouched (so it stays inspectable --
    "keys are a perfect match, we're just not sure either key is right"
    stays visible as value=100/confidence=low) but barely moves
    ``total_score``, rather than shrinking the value itself, which would
    make a "100% compatible, 10% sure" component indistinguishable from a
    "10% compatible, 100% sure" one. This is a simplified inverse-
    uncertainty weighting, not a formal Bayesian combination, chosen for
    the same reason the brief asked for it: it's easy to reason about and
    to retune once real feedback exists.

    Returns ``(total_score, overall_confidence)``. ``overall_confidence``
    is computed from *base* weights (not confidence-scaled), so it honestly
    reports how much of the judgement rests on solid ground rather than
    being self-reinforcing.
    """
    effective_weights = {name: max(weights[name] * c.confidence, min_effective_weight) for name, c in components.items()}
    total_effective = sum(effective_weights.values())
    total_score = sum(effective_weights[name] * c.value for name, c in components.items()) / total_effective

    total_base = sum(weights.values())
    overall_confidence = sum(weights[name] * c.confidence for name, c in components.items()) / total_base

    return max(0.0, min(100.0, total_score)), max(0.0, min(1.0, overall_confidence))


def score_candidate(state: DJState, candidate: SongProfile, config: ScoringConfig = DEFAULT_CONFIG) -> CandidateScore:
    """Score one candidate as "the next song" given the current DJ state.

    Never modifies audio, never picks a transition -- purely an evaluation
    of the pairing.
    """
    current = state.current_song

    tempo = score_tempo(current, candidate, config.tempo)
    harmonic = score_harmonic(current, candidate, config.harmonic)
    energy = score_energy(current, candidate, state, config.energy)
    rhythm = score_rhythm(current, candidate, config.rhythm)
    structure = score_structure(current, candidate, state, config.structure)
    style = score_style(current, candidate, config.style)
    familiarity = score_familiarity(candidate, state)
    repetition = score_repetition(candidate, state, config.repetition)

    components = {
        "tempo": tempo,
        "harmonic": harmonic,
        "energy": energy,
        "rhythm": rhythm,
        "structure": structure,
        "style": style,
        "familiarity": familiarity,
        "repetition": repetition,
    }
    total_score, overall_confidence = compute_total_score(components, config.weights.as_dict(), config.min_effective_weight)

    return CandidateScore(
        candidate_id=candidate.id,
        total_score=total_score,
        confidence=overall_confidence,
        tempo_score=tempo,
        harmonic_score=harmonic,
        energy_score=energy,
        rhythm_score=rhythm,
        structure_score=structure,
        style_score=style,
        familiarity_score=familiarity,
        repetition_penalty=repetition,
    )


def rank_candidates(
    state: DJState, library: list[SongProfile], config: ScoringConfig = DEFAULT_CONFIG
) -> list[CandidateScore]:
    """Score every song in ``library`` as a candidate "next song" and
    return them ranked best-first.

    The currently-playing song (if any) is excluded even if present in
    ``library`` -- "play the same song again immediately" isn't a
    meaningful recommendation regardless of how it would score, so this is
    filtered structurally rather than left to the repetition penalty to
    (possibly not) catch.

    Ties break on ``candidate_id`` so the ranking is fully deterministic:
    the same state and library always produce the same order, never
    dependent on input order, dict/hash iteration, or float noise.
    """
    current_id = state.current_song.id if state.current_song else None
    scores = [score_candidate(state, candidate, config) for candidate in library if candidate.id != current_id]
    scores.sort(key=lambda s: (-s.total_score, s.candidate_id))
    return scores
