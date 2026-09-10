"""Musical/style relationship -- a soft, low-weight, evidence-based proxy.

Real embedded genre metadata is frequently absent (the validation corpus
had it on barely half the tracks), so this leans on content-derived
proxies -- section timbre mix, vocal character, and coarse harmonic/chroma
color -- rather than assuming genre tags exist. Every sub-signal that isn't
available is simply left out rather than penalized, and the *component's*
weight in ScoringWeights is kept low on purpose: this dimension should
nudge, not gate, so the AI DJ can still surprise the listener.
"""

from __future__ import annotations

import math

import numpy as np

from songscoring.config import StyleConfig
from songscoring.song_profile import SongProfile
from songscoring.types import ComponentScore


def _genre_score(current: SongProfile, candidate: SongProfile, config: StyleConfig) -> float | None:
    if not current.genre or not candidate.genre:
        return None
    match = current.genre.strip().lower() == candidate.genre.strip().lower()
    return config.genre_match_score if match else config.genre_mismatch_score


def _instrumentation_score(current: SongProfile, candidate: SongProfile) -> float | None:
    a, b = current.timbre_distribution(), candidate.timbre_distribution()
    if not a or not b:
        return None
    labels = set(a) | set(b)
    overlap = sum(min(a.get(label, 0.0), b.get(label, 0.0)) for label in labels)
    return 100.0 * overlap  # histogram intersection of two distributions that each sum to 1


def _vocal_character_score(current: SongProfile, candidate: SongProfile, config: StyleConfig) -> float:
    diff = abs(current.vocal_density - candidate.vocal_density)
    return 100.0 * math.exp(-((diff / config.vocal_density_closeness_width) ** 2))


def _harmonic_timbre_score(current: SongProfile, candidate: SongProfile) -> float | None:
    if current.chroma_mean is None or candidate.chroma_mean is None:
        return None
    a, b = current.chroma_mean, candidate.chroma_mean
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return None
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return 100.0 * (cosine + 1.0) / 2.0


def score_style(current: SongProfile | None, candidate: SongProfile, config: StyleConfig) -> ComponentScore:
    if current is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "no current song (start of set)"})

    sub_values: list[float] = []
    sub_weights: list[float] = []
    explanation: dict[str, object] = {}

    genre_score = _genre_score(current, candidate, config)
    if genre_score is not None:
        sub_values.append(genre_score)
        sub_weights.append(config.genre_match_sub_weight)
        explanation["genre_match"] = {"current": current.genre, "candidate": candidate.genre, "score": genre_score}

    instrumentation_score = _instrumentation_score(current, candidate)
    if instrumentation_score is not None:
        sub_values.append(instrumentation_score)
        sub_weights.append(config.instrumentation_sub_weight)
        explanation["instrumentation_similarity"] = round(instrumentation_score, 1)

    vocal_score = _vocal_character_score(current, candidate, config)
    sub_values.append(vocal_score)
    sub_weights.append(config.vocal_character_sub_weight)
    explanation["vocal_character_similarity"] = round(vocal_score, 1)

    harmonic_timbre_score = _harmonic_timbre_score(current, candidate)
    if harmonic_timbre_score is not None:
        sub_values.append(harmonic_timbre_score)
        sub_weights.append(config.harmonic_timbre_sub_weight)
        explanation["harmonic_timbre_similarity"] = round(harmonic_timbre_score, 1)

    total_weight = sum(sub_weights)
    value = sum(v * w for v, w in zip(sub_values, sub_weights)) / total_weight

    # Confidence reflects how much real evidence (vs. neutral filler) backs
    # this score -- genre and chroma are the strongest signals; vocal
    # character alone is a thin basis for a style judgement.
    n_strong_signals = sum(x is not None for x in (genre_score, instrumentation_score, harmonic_timbre_score))
    confidence = 0.2 + 0.25 * n_strong_signals

    return ComponentScore(value=max(0.0, min(100.0, value)), confidence=min(1.0, confidence), explanation=explanation)
