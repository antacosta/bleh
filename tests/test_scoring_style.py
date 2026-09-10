from __future__ import annotations

import numpy as np

from songscoring.components.style import score_style
from songscoring.config import StyleConfig
from songscoring.song_profile import SectionSummary
from scoring_helpers import make_profile

CFG = StyleConfig()


def test_same_genre_and_chroma_scores_highly():
    chroma = np.array([1.0, 0.2, 0.3, 0.1, 0.6, 0.3, 0.2, 0.8, 0.1, 0.2, 0.1, 0.4])
    current = make_profile(genre="House", chroma_mean=chroma, vocal_density=0.3)
    candidate = make_profile(genre="House", chroma_mean=chroma, vocal_density=0.3)
    result = score_style(current, candidate, CFG)
    assert result.value > 80


def test_different_genre_scores_lower_but_is_not_dominant():
    current = make_profile(genre="House", vocal_density=0.3)
    same_genre = make_profile(genre="House", vocal_density=0.3)
    different_genre = make_profile(genre="Death Metal", vocal_density=0.3)

    same_score = score_style(current, same_genre, CFG).value
    different_score = score_style(current, different_genre, CFG).value
    assert same_score > different_score
    # A genre mismatch alone (everything else held equal) should not crater
    # the score -- style is a soft nudge, not a gate.
    assert different_score > 60


def test_missing_genre_falls_back_to_content_proxies_gracefully():
    current = make_profile(genre=None)
    candidate = make_profile(genre=None)
    result = score_style(current, candidate, CFG)
    assert "genre_match" not in result.explanation
    assert 0.0 <= result.value <= 100.0


def test_instrumentation_similarity_uses_timbre_distribution():
    bright_section = SectionSummary("s", 0, 100, 100, None, 0.0, 0.5, 2.0, 0.5, "bright", None)
    bass_section = SectionSummary("s", 0, 100, 100, None, 0.0, 0.5, 2.0, 0.5, "bass-heavy", None)
    current = make_profile(genre=None, sections=(bright_section,))
    similar = make_profile(genre=None, sections=(bright_section,))
    different = make_profile(genre=None, sections=(bass_section,))

    assert score_style(current, similar, CFG).value > score_style(current, different, CFG).value


def test_confidence_reflects_amount_of_real_evidence():
    thin = make_profile(genre=None, chroma_mean=None, sections=())
    rich = make_profile(genre="House", chroma_mean=np.ones(12))
    assert score_style(thin, thin, CFG).confidence < score_style(rich, rich, CFG).confidence


def test_score_bounded_0_100():
    current = make_profile()
    for genre in (None, "Pop", "Jazz"):
        result = score_style(current, make_profile(genre=genre), CFG)
        assert 0.0 <= result.value <= 100.0
