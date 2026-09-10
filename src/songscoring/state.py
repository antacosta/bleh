"""The DJ's current state: everything a scoring pass is evaluated against.

This is intentionally a plain, serializable snapshot -- not an object with
behavior -- so it can be constructed fresh for every ranking call, logged,
replayed, or eventually driven by a UI/session store without this layer
caring how it got built.

Only what's needed for one-step candidate scoring is implemented now. The
fields that don't do anything yet (``familiarity_by_song_id``, party/mood
knobs) are there so the *interface* doesn't need to change shape when those
features exist -- see EnergyDirection and the familiarity component for how
"not implemented yet" degrades to neutral rather than fabricated behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from songscoring.song_profile import SongProfile


class EnergyDirection(Enum):
    """Where the DJ (or a future planner/user nudge) wants the set's energy
    to go next. ``None`` on ``DJState.desired_energy_direction`` means no
    explicit intent -- see energy.py for why that is *not* the same as
    "prefer no change"."""

    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    RESET = "reset"


class EnergyIntentStrength(Enum):
    """How firmly ``desired_energy_direction`` should be honored.

    A direction can be stated three distinguishable ways:

    - ``NONE``: no direction at all (``desired_energy_direction is None``;
      this value is the field's meaning by convention, not something you'd
      set alongside a direction).
    - ``MILD``: a soft nudge -- "lean toward a build if a good option
      exists", without letting it override an otherwise clearly better
      candidate on every other dimension.
    - ``EXPLICIT``: a firm request -- "build the energy now". This should
      give the energy component substantially more influence than its base
      weight, per the validation finding that a stated direction was being
      outvoted by unrelated components.

    See ``songscoring.config.EnergyIntentConfig`` for how this is turned
    into an actual number, and ``scorer.compute_total_score`` for the
    (generic, not energy-specific) mechanism that applies it.
    """

    NONE = "none"
    MILD = "mild"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class DJState:
    """Snapshot of the set right before choosing the next song.

    Attributes:
        current_song: the song currently playing (or about to be mixed out
            of). ``None`` only at the very start of a set, before anything
            has played.
        current_position: seconds into ``current_song``'s playback. Used so
            "ending energy" reflects what's actually left to play, not
            necessarily the song's tail if we're mixing out earlier.
        recent_songs / recent_artists / recent_genres: most-recent-first
            history, used by the repetition component. Genres are whatever
            string the analysis metadata carried (often absent -- see the
            repetition component for how that degrades).
        set_energy: recent per-song ``overall_energy`` values (same
            per-song-relative scale as everywhere else), most-recent-first;
            a simple running trajectory, not a plan.
        desired_energy_direction: explicit intent for where energy should
            head next, if any. Left ``None`` by default (see EnergyDirection).
        energy_intent_strength: how firmly ``desired_energy_direction`` should
            be honored (see EnergyIntentStrength). Only meaningful when
            ``desired_energy_direction`` is not ``None``; defaults to
            ``EXPLICIT`` so that simply setting a direction -- without any
            further qualification -- reads as "I want this now", matching
            historical behavior for callers that only ever set a direction.
        familiarity_by_song_id: optional external lookup of a 0..1
            familiarity score per song id, meant to eventually be backed by
            real listening history. Never fabricated here -- absent ids (or
            an absent dict entirely) resolve to "unknown", not "unfamiliar".
    """

    current_song: SongProfile | None = None
    current_position: float = 0.0
    recent_songs: tuple[str, ...] = field(default_factory=tuple)
    recent_artists: tuple[str | None, ...] = field(default_factory=tuple)
    recent_genres: tuple[str | None, ...] = field(default_factory=tuple)
    set_energy: tuple[float, ...] = field(default_factory=tuple)
    desired_energy_direction: EnergyDirection | None = None
    energy_intent_strength: EnergyIntentStrength = EnergyIntentStrength.EXPLICIT

    # Forward-looking hooks -- deliberately inert until real data exists.
    familiarity_by_song_id: dict[str, float] | None = None
    novelty_by_song_id: dict[str, float] | None = None
