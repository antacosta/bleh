"""Candidate pruning: turns a full library into a small, ranked pool at one
search node, using the *existing* one-step scorer -- never a second
candidate-evaluation mechanism, and never an exhaustive per-depth scan of
the whole library beyond this one ranking pass per node.
"""

from __future__ import annotations

from songscoring.config import ScoringConfig
from songscoring.scorer import rank_candidates
from songscoring.song_profile import SongProfile
from songscoring.types import CandidateScore

from songplanner.config import BeamSearchConfig, SequenceScoringConfig
from songplanner.planner_state import PlannerState


def build_pool(
    node: PlannerState,
    library: list[SongProfile],
    by_id: dict[str, SongProfile],
    beam_config: BeamSearchConfig,
    sequence_config: SequenceScoringConfig,
    scoring_config: ScoringConfig,
) -> list[tuple[SongProfile, CandidateScore]]:
    """Rank the whole library against ``node``'s hypothetical state with the
    existing one-step scorer, then prune it down to a bounded pool.

    Excludes the real currently-playing song (handled by ``rank_candidates``
    itself) and, when ``forbid_repeat_song_within_horizon`` is set, any song
    already used earlier in this same hypothetical path -- a hard
    constraint, not merely a penalty (see config.py). Returns at most
    ``beam_config.max_pool_size`` entries, best-first, already carrying each
    candidate's full one-step ``CandidateScore`` so nothing is rescored.
    """
    ranked = rank_candidates(node.dj_state, library, scoring_config)

    if sequence_config.forbid_repeat_song_within_horizon:
        used_ids = {song.id for song in node.planned_sequence}
        ranked = [r for r in ranked if r.candidate_id not in used_ids]

    pool = ranked[: beam_config.max_pool_size]
    return [(by_id[r.candidate_id], r) for r in pool]
