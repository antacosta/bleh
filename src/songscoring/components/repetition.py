"""Repetition penalty: keep the set from repeating itself too soon.

Reported on the same 0..100, higher-is-better scale as every other
component (100 = no repetition concern at all) so the central weighted-sum
math in ``scorer.py`` stays uniform -- see ``types.ComponentScore`` and
``CandidateScore.repetition_penalty``. The raw penalty points (how much was
subtracted, and why) are always in ``explanation`` for inspection.

Pure bookkeeping against explicit recent-history inputs, not statistical
inference, so confidence is always 1.0 -- "no recent history was given" is
a known fact (no penalty applies), not an uncertain one.
"""

from __future__ import annotations

from songscoring.config import RepetitionConfig
from songscoring.song_profile import SongProfile
from songscoring.state import DJState
from songscoring.types import ComponentScore


def score_repetition(candidate: SongProfile, state: DJState, config: RepetitionConfig) -> ComponentScore:
    penalty = 0.0
    reasons: list[dict[str, object]] = []

    for i, song_id in enumerate(state.recent_songs[: config.lookback]):
        if song_id == candidate.id:
            p = config.same_song_base_penalty * (config.recency_decay**i)
            penalty += p
            reasons.append({"type": "same_song", "positions_ago": i, "penalty": round(p, 1)})

    if candidate.artist:
        for i, artist in enumerate(state.recent_artists[: config.lookback]):
            if artist and artist == candidate.artist:
                p = config.same_artist_base_penalty * (config.recency_decay**i)
                penalty += p
                reasons.append({"type": "same_artist", "positions_ago": i, "penalty": round(p, 1)})

    if candidate.genre:
        for i, genre in enumerate(state.recent_genres[: config.lookback]):
            if genre and genre == candidate.genre:
                p = config.same_genre_base_penalty * (config.recency_decay**i)
                penalty += p
                reasons.append({"type": "same_genre", "positions_ago": i, "penalty": round(p, 1)})

    penalty = min(100.0, penalty)
    value = 100.0 - penalty

    return ComponentScore(
        value=value,
        confidence=1.0,
        explanation={"penalty_points": round(penalty, 1), "reasons": reasons},
    )
