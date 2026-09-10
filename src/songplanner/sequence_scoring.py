"""Sequence-level scoring: what makes a whole planned path good, beyond the
plain average of its per-transition one-step scores.

Combined via the *same* confidence-weighted combiner the one-step scorer
uses (``songscoring.scorer.compute_total_score``), applied here to four
"virtual" sequence-level components instead of the one-step scorer's eight:

- ``transition``: the mean of the path's own per-step one-step scores
  (already fully computed by the existing scorer -- reused, not redone).
- ``trajectory``: does the path's energy shape match what was asked for
  (BUILD/RELEASE/MAINTAIN/RESET), or read as a coherent shape at all with
  no explicit direction -- see ``_trajectory_score``.
- ``variety``: flat, non-decaying penalty for a song/artist/genre recurring
  within this short plan -- see ``_variety_score``.
- ``coherence``: penalty for erratic genre "bouncing" (A -> B -> A) -- see
  ``_coherence_score``.

Reusing ``compute_total_score`` (rather than inventing a second combination
formula) also means the ``trajectory`` component's *weight* gets scaled by
the same ``EnergyIntentConfig`` multiplier the one-step energy component
uses -- an EXPLICIT BUILD request should make trajectory-adherence matter
substantially more here too, exactly mirroring the one-step fix.

Nothing in this module is natural-language generation. The ``reasons``
lists are plain, deterministic strings built by formatting the numbers
already computed -- inspectable data, not model-authored prose.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass

from songscoring.config import EnergyIntentConfig
from songscoring.scorer import compute_total_score, energy_intent_multiplier
from songscoring.song_profile import SongProfile
from songscoring.state import DJState, EnergyDirection
from songscoring.types import CandidateScore, ComponentScore

from songplanner.config import SequenceScoringConfig


@dataclass(frozen=True)
class SequenceScoreBreakdown:
    """Fully inspectable result of scoring one (possibly partial) path."""

    path_score: float
    path_confidence: float
    components: dict[str, ComponentScore]
    energy_levels: tuple[float, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "path_score": round(self.path_score, 2),
            "path_confidence": round(self.path_confidence, 3),
            "components": {
                name: {"value": round(c.value, 2), "confidence": round(c.confidence, 3), "explanation": c.explanation}
                for name, c in self.components.items()
            },
            "energy_levels": [round(v, 3) for v in self.energy_levels],
            "reasons": list(self.reasons),
        }


def _anchor_energy_level(root_state: DJState) -> float:
    """Where the real set's energy currently stands, as the first point on
    the trajectory -- the outgoing song's ending energy if we have one,
    else its overall energy, else a neutral midpoint at the very start of
    a set."""
    current = root_state.current_song
    if current is None:
        return 0.5
    ending = current.ending_energy(20.0, from_position=root_state.current_position)
    return ending if ending is not None else current.overall_energy


def _trajectory_score(
    levels: list[float], desired: EnergyDirection | None, config: SequenceScoringConfig
) -> tuple[float, list[str]]:
    if len(levels) < 2:
        return 100.0, ["fewer than two energy points -- no trajectory to judge yet"]

    deltas = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
    net_delta = levels[-1] - levels[0]

    if desired is None:
        magnitude = min(1.0, statistics.fmean(abs(d) for d in deltas))
        dip = math.sin(magnitude * math.pi) ** 2
        value = 100.0 - (100.0 - config.neutral_shape_floor) * dip
        return value, [f"no explicit direction: mean |per-step delta|={magnitude:.3f}"]

    if desired == EnergyDirection.MAINTAIN:
        stdev = statistics.pstdev(levels)
        value = 100.0 * math.exp(-((stdev / max(config.maintain_tolerance, 1e-6)) ** 2))
        return value, [f"MAINTAIN: energy stdev across plan={stdev:.3f} (tolerance={config.maintain_tolerance})"]

    sign = 1.0 if desired == EnergyDirection.INCREASE else -1.0
    target = sign * config.build_release_target_net_delta
    net_component = math.exp(-(((net_delta - target) / max(config.net_delta_tolerance, 1e-6)) ** 2))
    aligned_steps = sum(1 for d in deltas if d * sign > 0)
    consistency = aligned_steps / len(deltas)
    value = 100.0 * ((1.0 - config.step_consistency_weight) * net_component + config.step_consistency_weight * consistency)
    reasons = [
        f"{desired.value.upper()}: net energy delta={net_delta:+.3f} (target {target:+.3f}), "
        f"{consistency * 100:.0f}% of steps moved the requested way"
    ]

    if desired == EnergyDirection.DECREASE and deltas[0] <= -config.instant_collapse_drop_threshold:
        value = max(0.0, value - config.instant_collapse_penalty)
        reasons.append(
            f"penalized: first step dropped {deltas[0]:+.3f}, an instant collapse rather than a gradual release"
        )

    return max(0.0, min(100.0, value)), reasons


def _variety_score(
    root_state: DJState, sequence: tuple[SongProfile, ...], config: SequenceScoringConfig
) -> tuple[float, list[str]]:
    """Flat, non-decaying penalty for a song/artist/genre recurring within
    the plan (or against the real recent history it extends) -- deliberately
    *not* recency-decayed like the one-step repetition component, since
    within one short plan a repeat is not meaningfully less repetitive for
    having happened a song or two further back (see config.py)."""
    penalty = 0.0
    reasons: list[str] = []

    seen_songs = set(root_state.recent_songs)
    seen_artists = Counter(a for a in root_state.recent_artists if a)
    seen_genres = Counter(g for g in root_state.recent_genres if g)

    for song in sequence:
        if song.id in seen_songs:
            penalty += config.same_song_in_plan_penalty
            reasons.append(f"song repeated within plan/history: {song.id}")
        seen_songs.add(song.id)

        if song.artist and seen_artists[song.artist] >= 1:
            penalty += config.same_artist_in_plan_penalty
            reasons.append(f"artist repeated within plan/history: {song.artist}")
        if song.artist:
            seen_artists[song.artist] += 1

        if song.genre and seen_genres[song.genre] >= config.same_genre_allowance:
            penalty += config.same_genre_in_plan_penalty
            reasons.append(f"genre repeated more than {config.same_genre_allowance} times: {song.genre}")
        if song.genre:
            seen_genres[song.genre] += 1

    value = max(0.0, 100.0 - penalty)
    if not reasons:
        reasons.append("no repeated song, artist, or over-repeated genre within the plan")
    return value, reasons


def _coherence_score(
    root_state: DJState, sequence: tuple[SongProfile, ...], config: SequenceScoringConfig
) -> tuple[float, list[str]]:
    """Penalize erratic genre "bouncing" (A -> B -> A) while allowing a
    sustained run (A -> A -> B) or a deliberate progression (A -> B -> C) --
    see config.genre_bounce_penalty."""
    genres = [root_state.current_song.genre if root_state.current_song else None] + [s.genre for s in sequence]
    penalty = 0.0
    reasons: list[str] = []
    for i in range(2, len(genres)):
        if genres[i] and genres[i - 1] and genres[i - 2] and genres[i] == genres[i - 2] and genres[i] != genres[i - 1]:
            penalty += config.genre_bounce_penalty
            reasons.append(f"genre bounced back to {genres[i]!r} at plan position {i - 1}")
    value = max(0.0, 100.0 - penalty)
    if not reasons:
        reasons.append("no erratic genre bounce-backs detected")
    return value, reasons


def score_sequence(
    root_state: DJState,
    sequence: tuple[SongProfile, ...],
    step_scores: tuple[CandidateScore, ...],
    config: SequenceScoringConfig,
    energy_intent_config: EnergyIntentConfig,
) -> SequenceScoreBreakdown:
    """Score one complete (or partial) planned path.

    ``root_state`` is the *real* current state (never a hypothetical one)
    -- it supplies the energy anchor point, the real recency history the
    variety check extends, and the originally requested direction/intent
    strength, all of which stay constant for every node in one planning
    call regardless of how deep the path has gone.
    """
    if not sequence:
        empty = ComponentScore(value=100.0, confidence=1.0, explanation={"reason": "empty sequence"})
        return SequenceScoreBreakdown(
            path_score=100.0,
            path_confidence=1.0,
            components={"transition": empty, "trajectory": empty, "variety": empty, "coherence": empty},
            energy_levels=(_anchor_energy_level(root_state),),
            reasons=("empty plan",),
        )

    mean_transition_value = statistics.fmean(s.total_score for s in step_scores)
    mean_transition_confidence = statistics.fmean(s.confidence for s in step_scores)
    transition_component = ComponentScore(
        value=mean_transition_value,
        confidence=mean_transition_confidence,
        explanation={"per_step_total_scores": [round(s.total_score, 2) for s in step_scores]},
    )

    # Trajectory is fundamentally a claim about *energy* shape, so its
    # confidence should track the per-step energy component's own
    # confidence specifically -- not the whole blended one-step confidence
    # (which also carries tempo/harmonic/etc uncertainty that has nothing
    # to do with whether the trajectory claim itself should be trusted).
    mean_energy_confidence = statistics.fmean(s.energy_score.confidence for s in step_scores)

    levels = [_anchor_energy_level(root_state)] + [s.overall_energy for s in sequence]
    trajectory_value, trajectory_reasons = _trajectory_score(levels, root_state.desired_energy_direction, config)
    trajectory_component = ComponentScore(
        value=trajectory_value,
        confidence=mean_energy_confidence,
        explanation={"levels": [round(v, 3) for v in levels], "desired_direction": (
            root_state.desired_energy_direction.value if root_state.desired_energy_direction else None
        )},
    )

    variety_value, variety_reasons = _variety_score(root_state, sequence, config)
    variety_component = ComponentScore(value=variety_value, confidence=1.0, explanation={"reasons": variety_reasons})

    coherence_value, coherence_reasons = _coherence_score(root_state, sequence, config)
    coherence_component = ComponentScore(value=coherence_value, confidence=1.0, explanation={"reasons": coherence_reasons})

    components = {
        "transition": transition_component,
        "trajectory": trajectory_component,
        "variety": variety_component,
        "coherence": coherence_component,
    }
    weights = {
        "transition": config.transition_weight,
        "trajectory": config.trajectory_weight,
        "variety": config.variety_weight,
        "coherence": config.coherence_weight,
    }
    # Same mechanism as the one-step energy-intent fix: an EXPLICIT request
    # should make trajectory-adherence matter substantially more here too,
    # via the *same* configured multiplier -- not a second, separately
    # tuned "how much do we care" number.
    trajectory_multiplier = energy_intent_multiplier(root_state, energy_intent_config)
    path_score, path_confidence = compute_total_score(
        components, weights, config.min_effective_weight, weight_multipliers={"trajectory": trajectory_multiplier}
    )

    reasons = tuple([f"transition: mean one-step score {mean_transition_value:.1f}"] + trajectory_reasons + variety_reasons + coherence_reasons)

    return SequenceScoreBreakdown(
        path_score=path_score,
        path_confidence=path_confidence,
        components=components,
        energy_levels=tuple(levels),
        reasons=reasons,
    )
