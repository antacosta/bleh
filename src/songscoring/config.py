"""Central, tunable configuration for candidate scoring.

Nothing in ``songscoring.components`` or ``songscoring.scorer`` hard-codes a
weight or a tolerance -- every tunable number lives here, so the whole
scoring model can be retuned from one place once we have real data to tune
it against (see the validation report's recommendation to re-run this kind
of check empirically rather than by hand-picked examples).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoringWeights:
    """Base weight for each component's contribution to ``total_score``.

    These are *base* weights: the scorer multiplies each one by that
    component's own confidence before combining (see
    ``scorer.compute_total_score``), so a component with weight 0.20 but
    confidence 0.1 ends up contributing far less than one with weight 0.10
    at confidence 0.9. Base weights should reflect how important a
    dimension is *when it is known*, not how often it happens to be known.

    Initial values are deliberately conservative and roughly-equal-ish
    rather than sharply peaked -- we have no empirical tuning data yet, so
    asserting strong opinions about relative importance beyond "structure
    and tempo probably matter a bit more than style" would be fabricated
    precision. They sum to 1.0 for interpretability when every component is
    fully confident, but the scorer renormalizes regardless, so this isn't
    load-bearing.
    """

    tempo: float = 0.20
    harmonic: float = 0.15
    energy: float = 0.15
    rhythm: float = 0.12
    structure: float = 0.15
    style: float = 0.08  # deliberately low -- the AI DJ should be able to surprise
    familiarity: float = 0.05  # near-inert until real user data exists (confidence ~0)
    repetition: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "tempo": self.tempo,
            "harmonic": self.harmonic,
            "energy": self.energy,
            "rhythm": self.rhythm,
            "structure": self.structure,
            "style": self.style,
            "familiarity": self.familiarity,
            "repetition": self.repetition,
        }


@dataclass(frozen=True)
class TempoConfig:
    #: Tempo ratios a future DJ engine could plausibly execute (native speed,
    #: half-time, double-time). Distance to the *nearest* of these -- not to
    #: the current song's raw BPM -- is what tempo compatibility measures.
    manageable_ratios: tuple[float, ...] = (1.0, 0.5, 2.0)
    #: Width (as a fraction, e.g. 0.08 = 8%) of the Gaussian falloff around a
    #: manageable ratio. At this fractional adjustment, score has dropped to
    #: ~37% (1/e) of its peak. Real time-stretching starts sounding strained
    #: well before 100% (an octave), so this is intentionally tight.
    tolerance_fraction: float = 0.08
    #: Flat deduction applied when the *best* matching ratio is 0.5x/2x
    #: rather than 1x: half/double-time mixing is a real, usable technique,
    #: but it is a more advanced move than matching tempo natively, so it
    #: should not score identically to a native match at the same precision.
    octave_ratio_penalty: float = 10.0


@dataclass(frozen=True)
class HarmonicConfig:
    #: Shape of the falloff (in circle-of-fifths steps, 0..6) from a perfect
    #: match. Tuned so distance 1 (a fifth apart) still scores fairly high
    #: (~75) -- a standard, easy DJ-mixing move -- while distance 6 (tritone,
    #: the least compatible relationship) is close to zero.
    decay_scale: float = 2.2
    decay_power: float = 1.6
    #: Small deduction when two keys share a circle-of-fifths position but
    #: are *not* literally the same key (relative major/minor, e.g. C major
    #: vs A minor) -- still very compatible, just not identical.
    relative_key_deduction: float = 8.0


@dataclass(frozen=True)
class EnergyConfig:
    window_sec: float = 20.0
    #: |delta| below this (on the 0..1 per-song-relative energy scale) counts
    #: as "maintain" rather than a directional move.
    maintain_threshold: float = 0.15
    #: A candidate whose own opening is below this (relative to itself) and
    #: represents a big drop from the outgoing song counts as a "reset".
    reset_start_threshold: float = 0.20
    reset_drop_threshold: float = 0.30


@dataclass(frozen=True)
class RhythmConfig:
    #: Width of the log-ratio Gaussian used to compare rhythmic density and
    #: percussive activity between songs (these are rate-like quantities,
    #: so ratio/log-space closeness, not raw difference, is the right metric).
    ratio_closeness_width: float = 0.6
    density_sub_weight: float = 1.0
    percussive_sub_weight: float = 1.0
    stability_sub_weight: float = 0.8
    phrase_sub_weight: float = 0.6
    #: Score for matching vs. mismatched time signature, before blending
    #: toward neutral (50) by how confident either song's downbeat read is.
    phrase_match_score: float = 85.0
    phrase_mismatch_score: float = 30.0


@dataclass(frozen=True)
class StructureConfig:
    #: Assumed quality of an exit/entry point when the analyzer found none
    #: at all -- a song can always technically be cut into/out of, just not
    #: at a point the analyzer flagged as good, so this is a mediocre
    #: baseline rather than zero.
    no_affordance_baseline: float = 0.25
    #: Bonus when the chosen exit and entry points share a reason tag (e.g.
    #: both "clean_downbeat") -- a hint the transition can be phrase-aligned.
    shared_reason_bonus: float = 8.0
    #: Bonus (scaled by the best candidate loop's own confidence) for the
    #: candidate having a usable loop near its start -- more flexibility for
    #: whatever plans the entry.
    loop_availability_bonus: float = 6.0
    has_affordance_confidence: float = 0.85
    fallback_confidence: float = 0.35


@dataclass(frozen=True)
class StyleConfig:
    """Sub-weights for the low-level style proxies. The *component's* own
    weight in ScoringWeights is what keeps style from dominating overall;
    these only control the mix of proxies used when it does contribute."""

    genre_match_sub_weight: float = 1.0
    instrumentation_sub_weight: float = 1.0
    vocal_character_sub_weight: float = 0.8
    harmonic_timbre_sub_weight: float = 1.0
    genre_match_score: float = 90.0
    genre_mismatch_score: float = 40.0
    vocal_density_closeness_width: float = 0.35


@dataclass(frozen=True)
class RepetitionConfig:
    #: Penalty (0..100 points) for a candidate that exactly matches a song
    #: in recent_songs, decaying with how long ago it played.
    same_song_base_penalty: float = 100.0
    same_artist_base_penalty: float = 45.0
    same_genre_base_penalty: float = 20.0
    #: Per-step decay applied per position further back in the recent-songs
    #: history (position 0 = just played). 0.7 means the second-most-recent
    #: item carries 70% of the penalty the most recent one would.
    recency_decay: float = 0.7
    #: How many recent items to look back through at all.
    lookback: int = 8


@dataclass(frozen=True)
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    tempo: TempoConfig = field(default_factory=TempoConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    #: Floor applied to (base_weight * confidence) during renormalization so
    #: a component at zero confidence still contributes an infinitesimal,
    #: numerically-safe amount rather than risking divide-by-zero if *every*
    #: component happened to be zero-confidence at once.
    min_effective_weight: float = 1e-6


DEFAULT_CONFIG = ScoringConfig()
