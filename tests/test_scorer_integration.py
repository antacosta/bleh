from __future__ import annotations

from songscoring.config import DEFAULT_CONFIG
from songscoring.scorer import compute_total_score, rank_candidates, score_candidate
from songscoring.state import DJState
from songscoring.types import ComponentScore
from scoring_helpers import make_profile


def test_score_candidate_returns_all_components():
    current = make_profile(id="current")
    candidate = make_profile(id="candidate")
    result = score_candidate(DJState(current_song=current), candidate)
    components = result.components()
    assert set(components) == {
        "tempo", "harmonic", "energy", "rhythm", "structure", "style", "familiarity", "repetition",
    }
    for c in components.values():
        assert 0.0 <= c.value <= 100.0
        assert 0.0 <= c.confidence <= 1.0
    assert 0.0 <= result.total_score <= 100.0
    assert 0.0 <= result.confidence <= 1.0


def test_highly_compatible_candidate_outranks_a_contrasting_one():
    current = make_profile(id="current", bpm=128.0, key="A minor", key_root="A", key_mode="minor")
    compatible = make_profile(id="compatible", bpm=128.0, key="A minor", key_root="A", key_mode="minor")
    contrasting = make_profile(id="contrasting", bpm=171.0, key="F# major", key_root="F#", key_mode="major")

    state = DJState(current_song=current)
    ranking = rank_candidates(state, [compatible, contrasting])
    assert ranking[0].candidate_id == "compatible"


def test_ranking_is_deterministic_across_repeated_calls():
    current = make_profile(id="current")
    library = [make_profile(id=f"song-{i}", bpm=100.0 + i) for i in range(15)]
    state = DJState(current_song=current)

    first = [c.candidate_id for c in rank_candidates(state, library)]
    second = [c.candidate_id for c in rank_candidates(state, list(reversed(library)))]
    assert first == second


def test_current_song_is_excluded_from_its_own_ranking():
    current = make_profile(id="current")
    library = [current, make_profile(id="other")]
    ranking = rank_candidates(DJState(current_song=current), library)
    assert all(c.candidate_id != "current" for c in ranking)
    assert len(ranking) == 1


def test_tied_scores_break_on_candidate_id():
    current = make_profile(id="current")
    # Two candidates set up to be scored identically in every dimension.
    a = make_profile(id="b-song")
    b = make_profile(id="a-song")
    ranking = rank_candidates(DJState(current_song=current), [a, b])
    assert [c.candidate_id for c in ranking] == ["a-song", "b-song"]


def test_start_of_set_with_no_current_song_still_ranks():
    library = [make_profile(id=f"song-{i}") for i in range(5)]
    ranking = rank_candidates(DJState(current_song=None), library)
    assert len(ranking) == 5
    for c in ranking:
        assert 0.0 <= c.total_score <= 100.0


def test_recently_played_song_ranks_below_an_otherwise_similar_one():
    current = make_profile(id="current", bpm=128.0)
    recent = make_profile(id="recent", bpm=128.0, artist="X")
    fresh = make_profile(id="fresh", bpm=128.0, artist="X")
    state = DJState(current_song=current, recent_songs=("recent",))
    ranking = rank_candidates(state, [recent, fresh])
    assert ranking[0].candidate_id == "fresh"


def test_missing_optional_fields_across_the_board_does_not_crash():
    current = make_profile(
        id="current", bpm=None, key=None, key_root=None, key_mode=None, genre=None,
        chroma_mean=None, sections=(), mix_in_points=(), mix_out_points=(), loop_candidates=(),
    )
    candidate = make_profile(
        id="candidate", bpm=None, key=None, key_root=None, key_mode=None, genre=None,
        chroma_mean=None, sections=(), mix_in_points=(), mix_out_points=(), loop_candidates=(),
    )
    result = score_candidate(DJState(current_song=current), candidate)
    assert 0.0 <= result.total_score <= 100.0
    # With almost everything unknown, overall confidence should be low.
    assert result.confidence < 0.5


def test_to_dict_is_json_serializable():
    import json

    current = make_profile(id="current")
    candidate = make_profile(id="candidate")
    result = score_candidate(DJState(current_song=current), candidate)
    json.dumps(result.to_dict())  # raises if anything isn't serializable


def test_default_config_weights_sum_to_one():
    assert abs(sum(DEFAULT_CONFIG.weights.as_dict().values()) - 1.0) < 1e-9


def test_weight_multipliers_shift_the_total_toward_the_boosted_component():
    components = {
        "a": ComponentScore(value=100.0, confidence=1.0),
        "b": ComponentScore(value=0.0, confidence=1.0),
    }
    weights = {"a": 0.5, "b": 0.5}

    unboosted, _ = compute_total_score(components, weights, min_effective_weight=1e-6)
    boosted, _ = compute_total_score(components, weights, min_effective_weight=1e-6, weight_multipliers={"a": 5.0})
    assert unboosted == 50.0
    assert boosted > unboosted


def test_weight_multipliers_do_not_affect_overall_confidence():
    components = {
        "a": ComponentScore(value=100.0, confidence=0.4),
        "b": ComponentScore(value=0.0, confidence=0.9),
    }
    weights = {"a": 0.5, "b": 0.5}

    _, unboosted_confidence = compute_total_score(components, weights, min_effective_weight=1e-6)
    _, boosted_confidence = compute_total_score(
        components, weights, min_effective_weight=1e-6, weight_multipliers={"a": 5.0}
    )
    assert boosted_confidence == unboosted_confidence


def test_missing_weight_multiplier_entries_default_to_neutral():
    components = {
        "a": ComponentScore(value=80.0, confidence=1.0),
        "b": ComponentScore(value=20.0, confidence=1.0),
    }
    weights = {"a": 0.5, "b": 0.5}

    baseline, _ = compute_total_score(components, weights, min_effective_weight=1e-6)
    with_partial_multipliers, _ = compute_total_score(
        components, weights, min_effective_weight=1e-6, weight_multipliers={"a": 1.0}
    )
    assert baseline == with_partial_multipliers
