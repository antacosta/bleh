"""Central, tunable configuration for the look-ahead planner.

Same convention as ``songscoring.config``: nothing in ``songplanner`` hard-
codes a weight, threshold, or search-size limit -- every tunable number
lives here. ``PlannerConfig.scoring`` is a plain ``songscoring.config.
ScoringConfig`` (defaulting to the same ``DEFAULT_CONFIG`` the one-step
scorer uses) -- the planner does not fork or duplicate that configuration,
it just also carries its own sequence- and search-specific settings
alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from songscoring.config import DEFAULT_CONFIG, ScoringConfig


@dataclass(frozen=True)
class BeamSearchConfig:
    """Shape and size of the bounded beam search itself.

    This is deliberately a plain beam search -- generate, score, keep the
    best ``beam_width`` paths, expand, repeat -- not an exhaustive search
    (which would be exponential in ``horizon``) and not a general-purpose
    optimizer. Bounding every dimension below is what keeps it cheap and
    predictable regardless of library size.
    """

    #: How many future songs to plan (the brief's "~3-5"); default sits in
    #: the middle of that range.
    horizon: int = 4
    #: How many candidate paths survive after each depth's expansion. Must
    #: be > 1 for the planner's core "look-ahead" behavior to be possible at
    #: all: a beam width of 1 degenerates into pure greedy one-step
    #: selection, unable to ever prefer a locally-second-best move that
    #: leads somewhere better (see PlanExplanation.local_vs_global).
    beam_width: int = 6
    #: How many of the top-ranked candidates (from the pruned pool below)
    #: each surviving path actually branches into per depth. Bounds the
    #: branching factor independent of library size.
    candidates_per_expansion: int = 8
    #: The pruned candidate pool built via the existing one-step scorer, at
    #: each node, is truncated to at most this many candidates before
    #: ``candidates_per_expansion`` are drawn from it -- this is the actual
    #: "prune the library down" step the brief asks for, kept as its own
    #: knob so it can be widened independently of the branching factor.
    max_pool_size: int = 30
    #: If the pool (after excluding songs already used earlier in this same
    #: hypothetical path, and the real currently-playing song) shrinks
    #: below this, the planner does not error -- it just plans with
    #: whatever is available and the resulting plan may be shorter than
    #: ``horizon`` (see planner.plan_sequence). This only matters for very
    #: small libraries relative to the requested horizon.
    min_pool_size: int = 3


@dataclass(frozen=True)
class SequenceScoringConfig:
    """How a complete (or partial) planned sequence is scored, beyond the
    plain average of its per-transition one-step scores.

    Combined via the *same* confidence-weighted combiner used for one-step
    scoring (``songscoring.scorer.compute_total_score``), just applied to
    four sequence-level "virtual components" instead of the one-step
    scorer's eight -- see ``sequence_scoring.score_sequence``. That reuse is
    deliberate: no second combination formula, no special-casing.
    """

    #: Base weights for the four virtual components (renormalized
    #: regardless, like ScoringWeights -- see compute_total_score).
    transition_weight: float = 0.55
    trajectory_weight: float = 0.25
    variety_weight: float = 0.12
    coherence_weight: float = 0.08
    min_effective_weight: float = 1e-6

    #: Ideal *net* energy change (last song's overall_energy minus the
    #: anchor level) across the whole planned horizon for a committed
    #: BUILD/RELEASE/RESET request -- deliberately a shape over the whole
    #: path, not "every single step must move the same way".
    build_release_target_net_delta: float = 0.35
    #: Gaussian spread (same units) around that target.
    net_delta_tolerance: float = 0.30
    #: How much weight (0..1) "fraction of steps that individually moved
    #: the requested way" carries vs. the net-delta match above -- together
    #: these capture "generally builds" without demanding strict
    #: monotonicity, per the brief's explicit requirement.
    step_consistency_weight: float = 0.4

    #: RELEASE only: a first-step drop steeper than this (0..1, per-song-
    #: relative energy scale) reads as an "instant collapse" rather than an
    #: intentional, controlled release, and is penalized on top of the
    #: ordinary trajectory score. RESET is exempt -- a fast drop is exactly
    #: what RESET asks for.
    instant_collapse_drop_threshold: float = 0.55
    instant_collapse_penalty: float = 35.0

    #: MAINTAIN: the per-song-relative-energy standard deviation across the
    #: planned levels at which the trajectory score has fallen to ~37%
    #: (1/e) of its peak.
    maintain_tolerance: float = 0.15

    #: No explicit direction: mean-|step-delta| at which the "neutral
    #: shape" trajectory score dips to its floor (mirrors the one-step
    #: energy component's _neutral_shape_score, applied across the path).
    neutral_shape_floor: float = 75.0

    #: Sequence-scoped variety -- flat, *non*-decaying penalties for a
    #: song/artist/genre recurring anywhere within the planned horizon (or
    #: against the real recent history feeding into it). Deliberately not
    #: recency-decayed like the one-step repetition component: within one
    #: short 3-5 song plan, a repeat two songs back is not meaningfully
    #: less repetitive than one four songs back, so no decay "pass" is
    #: appropriate the way it is over a long real set history.
    same_song_in_plan_penalty: float = 100.0
    same_artist_in_plan_penalty: float = 30.0
    #: A genre may repeat up to this many times before being flagged -- a
    #: short cohesive run in one style is normal, not a variety failure.
    same_genre_allowance: int = 2
    same_genre_in_plan_penalty: float = 15.0
    #: "Bouncing" back to a genre used two songs ago (A -> B -> A) reads as
    #: erratic rather than an intentional, coherent style move; A -> B -> C
    #: (progression) and A -> A -> B (a sustained run) do not trigger this.
    genre_bounce_penalty: float = 12.0

    #: Hard constraint, checked before scoring: forbid the exact same song
    #: appearing twice within one planned horizon. Stricter than, and
    #: independent of, ``same_song_in_plan_penalty`` above (which would be
    #: the only guard if this were False) -- a DJ set does not replay a
    #: track a few songs later within one short plan.
    forbid_repeat_song_within_horizon: bool = True


@dataclass(frozen=True)
class PlannerConfig:
    beam: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    sequence: SequenceScoringConfig = field(default_factory=SequenceScoringConfig)
    #: The existing one-step scoring configuration, reused unmodified -- the
    #: planner does not fork ScoringWeights/EnergyIntentConfig/etc, it reads
    #: the same source of truth ``songscoring`` does.
    scoring: ScoringConfig = field(default_factory=lambda: DEFAULT_CONFIG)


DEFAULT_PLANNER_CONFIG = PlannerConfig()
