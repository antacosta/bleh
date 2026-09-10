"""Hypothetical, functional planning state.

``PlannerState`` extends the one-step scorer's ``DJState`` with the
bookkeeping a multi-song search needs (the sequence chosen so far, its
per-transition scores, a running transition-score total, and which first
song this hypothetical path branched from). It is built once from the real
``DJState`` at the start of a planning call and only ever advanced by
``advance()``, which returns a *new* ``PlannerState`` -- the real ``DJState``
passed in, and every intermediate state explored during search, is never
mutated. This is what lets beam search freely explore, discard, and compare
many hypothetical futures without any risk of corrupting the actual
playback/session state the caller owns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from songscoring.song_profile import SongProfile
from songscoring.state import DJState
from songscoring.types import CandidateScore


@dataclass(frozen=True)
class PlannerState:
    """One node in the search: the real current state plus everything
    hypothetically decided so far along one candidate path.

    Attributes:
        dj_state: a ``DJState`` *as it would be* after playing
            ``planned_sequence`` -- ``current_song`` is the last planned
            song (or the real current song, at the root), ``recent_songs``/
            ``recent_artists``/``recent_genres``/``set_energy`` are the real
            history with the planned songs prepended (most-recent-first,
            bounded the same way real history is -- see ``advance``).
            ``desired_energy_direction``/``energy_intent_strength`` are
            never touched by ``advance``, so the original request is
            carried through the whole plan unchanged.
        planned_sequence: the hypothetical future songs chosen so far,
            in play order (oldest first) -- *not* including the real
            current song.
        step_scores: one ``CandidateScore`` per entry in
            ``planned_sequence``, in the same order -- the exact,
            unmodified output of the existing one-step scorer for that
            transition, kept for explainability and for the sequence-level
            scorer to aggregate rather than recompute.
        cumulative_transition_score: running sum of ``step_scores[i].
            total_score`` -- a simple additive reference figure, not itself
            what beam search optimizes (see sequence_scoring.score_sequence
            for the actual path score).
        root_choice_id: the id of the very first song chosen on this path
            (set once, on the first ``advance()`` call, and left unchanged
            after). Used purely for explainability -- grouping surviving
            beam paths by their first move to show *why* a locally
            lower-scoring opening was preferred (see planner.plan_sequence).
    """

    dj_state: DJState
    planned_sequence: tuple[SongProfile, ...] = ()
    step_scores: tuple[CandidateScore, ...] = ()
    cumulative_transition_score: float = 0.0
    root_choice_id: str | None = None

    @property
    def depth(self) -> int:
        return len(self.planned_sequence)

    def advance(self, candidate: SongProfile, candidate_score: CandidateScore, history_lookback: int) -> PlannerState:
        """Return a new ``PlannerState`` with ``candidate`` appended.

        ``history_lookback`` should be ``config.scoring.repetition.
        lookback`` -- the same bound the one-step repetition component
        already uses -- so the extended history this produces is exactly
        as far back as anything downstream will ever look, no more.

        ``current_position`` on the new ``dj_state`` is set to 0.0, which
        is not "the song is at its start" but "no specific position is
        committed to" -- ``SongProfile.ending_energy`` already treats a
        position earlier than its own natural tail window as "use the
        song's natural tail", which is exactly the right assumption for a
        hypothetical full future play-through we have no more specific
        information about.
        """
        new_recent_songs = (candidate.id, *self.dj_state.recent_songs)[:history_lookback]
        new_recent_artists = (candidate.artist, *self.dj_state.recent_artists)[:history_lookback]
        new_recent_genres = (candidate.genre, *self.dj_state.recent_genres)[:history_lookback]
        new_set_energy = (candidate.overall_energy, *self.dj_state.set_energy)[:history_lookback]

        new_dj_state = replace(
            self.dj_state,
            current_song=candidate,
            current_position=0.0,
            recent_songs=new_recent_songs,
            recent_artists=new_recent_artists,
            recent_genres=new_recent_genres,
            set_energy=new_set_energy,
        )

        return PlannerState(
            dj_state=new_dj_state,
            planned_sequence=(*self.planned_sequence, candidate),
            step_scores=(*self.step_scores, candidate_score),
            cumulative_transition_score=self.cumulative_transition_score + candidate_score.total_score,
            root_choice_id=self.root_choice_id or candidate.id,
        )


def root_planner_state(state: DJState) -> PlannerState:
    """Build the initial search node from the real, live ``DJState``.

    This is the only place a real ``DJState`` is read into planning -- from
    here on, every state in the search tree is a ``PlannerState`` produced
    by ``advance()``, and ``state`` itself is never written to.
    """
    return PlannerState(dj_state=state)
