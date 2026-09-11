from __future__ import annotations

import pytest

from songscoring.state import EnergyDirection
from djlab.planning import (
    UnknownDirectionError,
    generate_plan,
    plan_response_dict,
    resolve_direction,
    resolve_horizon,
)
from real_corpus import load_real_corpus

CORPUS = load_real_corpus()
LIBRARY = list(CORPUS.values())


def test_resolve_direction_maps_ui_labels_to_energy_direction():
    assert resolve_direction("none") is None
    assert resolve_direction("build") == EnergyDirection.INCREASE
    assert resolve_direction("maintain") == EnergyDirection.MAINTAIN
    assert resolve_direction("release") == EnergyDirection.DECREASE


def test_resolve_direction_rejects_unknown_labels():
    with pytest.raises(UnknownDirectionError):
        resolve_direction("sideways")


def test_resolve_horizon_uses_num_songs_directly_when_given():
    assert resolve_horizon(3, None, LIBRARY) == 3


def test_resolve_horizon_clamps_num_songs_to_a_sane_range():
    assert resolve_horizon(100, None, LIBRARY) <= 8
    assert resolve_horizon(0, None, LIBRARY) >= 1


def test_resolve_horizon_converts_duration_to_a_song_count():
    # The real corpus's songs run a few minutes each -- a 20 minute request
    # should resolve to a small handful of songs, not 1 and not 8.
    horizon = resolve_horizon(None, 20.0, LIBRARY)
    assert 1 <= horizon <= 8


def test_resolve_horizon_falls_back_to_default_with_no_input():
    from songplanner.config import DEFAULT_PLANNER_CONFIG

    assert resolve_horizon(None, None, LIBRARY) == DEFAULT_PLANNER_CONFIG.beam.horizon


def test_generate_plan_runs_the_real_planner_end_to_end():
    starting = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    plan = generate_plan(starting, LIBRARY, direction="build", num_songs=3)
    assert plan.achieved_horizon <= 3
    assert all(step.song_id != starting.id for step in plan.steps)


def test_generate_plan_excludes_the_starting_song_from_candidates():
    starting = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    plan = generate_plan(starting, LIBRARY, direction="none", num_songs=4)
    assert starting.id not in [step.song_id for step in plan.steps]


def test_plan_response_dict_includes_honest_transition_placeholders():
    starting = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    plan = generate_plan(starting, LIBRARY, direction="build", num_songs=2)
    response = plan_response_dict(plan, excluded_unanalyzed_count=5)

    assert len(response["transitions"]) == len(plan.steps)
    for transition in response["transitions"]:
        assert transition["status"] == "not_implemented"
        assert "available_signal" in transition
    assert response["excluded_unanalyzed_count"] == 5
    assert response["render_available"] is False


def test_plan_response_dict_is_json_serializable():
    import json

    starting = CORPUS["12_electronic_kliq_plastic_flashing_lights"]
    plan = generate_plan(starting, LIBRARY, direction="release", num_songs=3)
    response = plan_response_dict(plan, excluded_unanalyzed_count=0)
    json.dumps(response)
