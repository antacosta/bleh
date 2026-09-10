"""Rhythmic compatibility -- deliberately independent of BPM.

Two songs at the same BPM can still feel rhythmically incompatible (a sparse
ballad-like pulse vs. a busy, percussion-dense groove), and two songs at
different-but-manageable tempos (see tempo.py) can be rhythmically very
similar. This compares rhythmic *density*, percussive activity, tempo
steadiness, and (when known) phrase/meter compatibility.
"""

from __future__ import annotations

import math

from songscoring.config import RhythmConfig
from songscoring.song_profile import SongProfile
from songscoring.types import ComponentScore


def _ratio_closeness(a: float | None, b: float | None, width: float) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    log_ratio = math.log(a / b)
    return 100.0 * math.exp(-((log_ratio / width) ** 2))


def score_rhythm(current: SongProfile | None, candidate: SongProfile, config: RhythmConfig) -> ComponentScore:
    if current is None:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "no current song (start of set)"})

    sub_values: list[float] = []
    sub_weights: list[float] = []
    explanation: dict[str, object] = {}

    density_score = _ratio_closeness(current.mean_onset_density(), candidate.mean_onset_density(), config.ratio_closeness_width)
    if density_score is not None:
        sub_values.append(density_score)
        sub_weights.append(config.density_sub_weight)
        explanation["rhythmic_density_similarity"] = round(density_score, 1)

    perc_score = _ratio_closeness(
        current.mean_percussive_activity(), candidate.mean_percussive_activity(), config.ratio_closeness_width
    )
    if perc_score is not None:
        sub_values.append(perc_score)
        sub_weights.append(config.percussive_sub_weight)
        explanation["percussive_activity_similarity"] = round(perc_score, 1)

    stability_diff = abs(current.tempo_stability - candidate.tempo_stability)
    both_stable_bonus = 0.7 + 0.3 * min(current.tempo_stability, candidate.tempo_stability)
    stability_score = max(0.0, min(100.0, 100.0 * (1.0 - stability_diff) * both_stable_bonus))
    sub_values.append(stability_score)
    sub_weights.append(config.stability_sub_weight)
    explanation["tempo_stability_compatibility"] = round(stability_score, 1)

    if current.time_signature and candidate.time_signature:
        match = current.time_signature == candidate.time_signature
        base = config.phrase_match_score if match else config.phrase_mismatch_score
        downbeat_conf = (current.downbeat_confidence + candidate.downbeat_confidence) / 2.0
        phrase_score = 50.0 + (base - 50.0) * downbeat_conf
        sub_values.append(phrase_score)
        sub_weights.append(config.phrase_sub_weight)
        explanation["phrase_compatibility"] = {
            "current_time_signature": current.time_signature,
            "candidate_time_signature": candidate.time_signature,
            "match": match,
            "downbeat_confidence_avg": round(downbeat_conf, 3),
            "score": round(phrase_score, 1),
        }

    if not sub_values:
        return ComponentScore(value=50.0, confidence=0.0, explanation={"reason": "no rhythmic data available"})

    total_weight = sum(sub_weights)
    value = sum(v * w for v, w in zip(sub_values, sub_weights)) / total_weight

    n_signals = len(sub_values)
    data_confidence = min(1.0, n_signals / 3.0)
    avg_stability = (current.tempo_stability + candidate.tempo_stability) / 2.0
    confidence = data_confidence * (0.5 + 0.5 * avg_stability)

    return ComponentScore(value=max(0.0, min(100.0, value)), confidence=max(0.0, min(1.0, confidence)), explanation=explanation)
