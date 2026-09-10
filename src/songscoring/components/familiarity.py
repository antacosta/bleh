"""Familiarity / novelty.

There is no user listening history yet, so this component's job right now
is purely to *exist with the right shape*: it reads an optional external
lookup off ``DJState`` (meant to eventually be backed by real play-history
data) and degrades to a neutral, zero-confidence result when that lookup is
absent or doesn't know a given song -- never a fabricated guess.

Familiarity and novelty are two names for the same axis here (1 - novelty
== familiarity): a caller can supply either one, and this takes whichever
is available, preferring an explicit familiarity value if both are given.
"""

from __future__ import annotations

from songscoring.song_profile import SongProfile
from songscoring.state import DJState
from songscoring.types import ComponentScore

#: A mild, deliberately soft preference curve over familiarity (0=novel,
#: 1=very familiar). Peaks a bit past the midpoint -- "recognizable but not
#: overplayed" -- rather than treating "totally novel" or "maximally
#: familiar" as ideal; this is a placeholder shape, not a tuned one.
_PREFERRED_FAMILIARITY = 0.65
_CURVE_WIDTH = 0.5


def score_familiarity(candidate: SongProfile, state: DJState) -> ComponentScore:
    familiarity = None
    source = None
    if state.familiarity_by_song_id is not None and candidate.id in state.familiarity_by_song_id:
        familiarity = state.familiarity_by_song_id[candidate.id]
        source = "familiarity_by_song_id"
    elif state.novelty_by_song_id is not None and candidate.id in state.novelty_by_song_id:
        familiarity = 1.0 - state.novelty_by_song_id[candidate.id]
        source = "novelty_by_song_id"

    if familiarity is None:
        return ComponentScore(
            value=50.0,
            confidence=0.0,
            explanation={"reason": "no familiarity/novelty data available for this song yet (neutral default)"},
        )

    familiarity = max(0.0, min(1.0, familiarity))
    distance = abs(familiarity - _PREFERRED_FAMILIARITY) / _CURVE_WIDTH
    value = 100.0 * max(0.0, 1.0 - distance**2)

    return ComponentScore(
        value=max(0.0, min(100.0, value)),
        confidence=0.6,  # supplied externally, so trustworthy, but not analysis-grade
        explanation={"familiarity": round(familiarity, 3), "source": source},
    )
