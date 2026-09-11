"""Bounded beam search over future song sequences.

This is the planner's one search algorithm, and it is deliberately simple:
generate candidates, score whole paths, keep the best ``beam_width``, expand,
repeat for ``horizon`` steps. Not exhaustive search (which would be
exponential in the branching factor), not a general-purpose optimizer --
just enough look-ahead to let a path with a better *eventual* trajectory
beat one that only looks better one step ahead (see ``plan_sequence``'s
``local_vs_global`` output for a concrete, per-plan account of exactly when
that happened).

Architecture reminder (see project README): this layer consumes
``songscoring`` -- it calls ``rank_candidates``/``score_candidate`` through
``candidate_pool.build_pool`` and reuses ``compute_total_score`` in
``sequence_scoring`` -- and produces nothing but a *sequence of songs* with
structured scoring/explanation attached. It does not choose or render a
transition, mix audio, or otherwise reach into the next layer.
"""

from __future__ import annotations

from songscoring.song_profile import SongProfile
from songscoring.state import DJState

from songplanner.candidate_pool import build_pool
from songplanner.config import DEFAULT_PLANNER_CONFIG, PlannerConfig
from songplanner.planner_state import PlannerState, root_planner_state
from songplanner.sequence_scoring import SequenceScoreBreakdown, score_sequence
from songplanner.types import LocalVsGlobalChoice, Plan, PlanStep


def _sequence_tie_break_key(node: PlannerState) -> tuple[str, ...]:
    """Deterministic tie-break: the plan's own song ids, in order -- so the
    same state and library always produce the same chosen plan, independent
    of dict/set iteration order or input ordering (same discipline as
    ``rank_candidates``'s ``candidate_id`` tie-break)."""
    return tuple(song.id for song in node.planned_sequence)


def _score_node(root_state: DJState, node: PlannerState, config: PlannerConfig) -> SequenceScoreBreakdown:
    return score_sequence(
        root_state,
        node.planned_sequence,
        node.step_scores,
        config.sequence,
        config.scoring.energy_intent,
        config.scoring.energy.window_sec,
    )


def _expand(
    node: PlannerState,
    library: list[SongProfile],
    by_id: dict[str, SongProfile],
    config: PlannerConfig,
    depth: int,
    search_notes: list[str],
) -> list[PlannerState]:
    pool = build_pool(node, library, by_id, config.beam, config.sequence, config.scoring)
    if not pool:
        return [node]  # nothing left to extend this path with -- stop growing it, don't error
    if 0 < len(pool) < config.beam.min_pool_size:
        search_notes.append(
            f"depth {depth}: candidate pool size {len(pool)} fell below the configured "
            f"min_pool_size={config.beam.min_pool_size} -- planning continued with what was available"
        )
    branch_candidates = pool[: config.beam.candidates_per_expansion]
    return [
        node.advance(candidate, candidate_score, config.scoring.repetition.lookback)
        for candidate, candidate_score in branch_candidates
    ]


def plan_sequence(state: DJState, library: list[SongProfile], config: PlannerConfig = DEFAULT_PLANNER_CONFIG) -> Plan:
    """Plan the next ``config.beam.horizon`` songs (a sequence, not a single
    next song) via bounded beam search.

    ``state`` is the real, current ``DJState`` -- read once to build the
    search root and never mutated (see ``planner_state.PlannerState``).
    ``library`` is scored through the existing one-step scorer at every
    node (see ``candidate_pool.build_pool``); this function does not
    reimplement or bypass that scoring.
    """
    exclude_id = state.current_song.id if state.current_song else None
    by_id = {song.id: song for song in library if song.id != exclude_id}
    pool_library = list(by_id.values())

    root = root_planner_state(state)
    beam: list[PlannerState] = [root]

    # Tracked across the whole search: for each distinct opening move, the
    # path score and depth the *last* time it was still part of the beam --
    # i.e. how far it got and how it scored there before either completing
    # the horizon or being pruned. See _build_local_vs_global for why this
    # is "latest seen", not "best score at any depth" (the two are not
    # comparable across different path lengths).
    depth0_immediate_by_root: dict[str, float] = {}
    latest_by_root: dict[str, tuple[int, float]] = {}
    search_notes: list[str] = []

    for depth in range(config.beam.horizon):
        expansions: list[PlannerState] = []
        for node in beam:
            expansions.extend(_expand(node, pool_library, by_id, config, depth, search_notes))
        if not expansions:
            break

        scored = [(n, _score_node(state, n, config)) for n in expansions]

        if depth == 0:
            for node, _ in scored:
                if node.root_choice_id is not None:
                    depth0_immediate_by_root.setdefault(node.root_choice_id, node.step_scores[0].total_score)

        for node, breakdown in scored:
            root_id = node.root_choice_id
            if root_id is None:
                continue
            # Multiple surviving paths can share the same opening move (a
            # root can still have >1 lineage in the beam at once), so more
            # than one entry for the same root_id can turn up in one round.
            # Those are directly comparable (same root, same depth), so keep
            # the best of them -- an unconditional overwrite here would make
            # the reported number depend on iteration order rather than on
            # which continuation was actually best.
            previous = latest_by_root.get(root_id)
            if previous is None or node.depth > previous[0] or (node.depth == previous[0] and breakdown.path_score > previous[1]):
                latest_by_root[root_id] = (node.depth, breakdown.path_score)

        scored.sort(key=lambda pair: (-pair[1].path_score, _sequence_tie_break_key(pair[0])))
        beam = [n for n, _ in scored[: config.beam.beam_width]]

    final_scored = [(n, _score_node(state, n, config)) for n in beam]
    final_scored.sort(key=lambda pair: (-pair[1].path_score, _sequence_tie_break_key(pair[0])))
    best_node, best_breakdown = final_scored[0]

    # The winning path's own number must exactly match what's shown for it
    # here, in case the final beam-trimming step dropped it from being the
    # "latest" update for its root (it wasn't, since final_scored is a
    # subset of the last depth's `scored`, but pinning it explicitly keeps
    # that invariant true even if this loop's shape changes later).
    if best_node.root_choice_id is not None:
        latest_by_root[best_node.root_choice_id] = (best_node.depth, best_breakdown.path_score)

    steps = tuple(
        PlanStep(
            position=i + 1,
            song_id=song.id,
            artist=song.artist,
            genre=song.genre,
            overall_energy=song.overall_energy,
            transition_score=score,
        )
        for i, (song, score) in enumerate(zip(best_node.planned_sequence, best_node.step_scores))
    )

    local_vs_global = _build_local_vs_global(depth0_immediate_by_root, latest_by_root, best_node.root_choice_id)

    return Plan(
        steps=steps,
        sequence_score=best_breakdown,
        local_vs_global=local_vs_global,
        requested_horizon=config.beam.horizon,
        achieved_horizon=best_node.depth,
        search_notes=tuple(search_notes),
    )


def _build_local_vs_global(
    depth0_immediate_by_root: dict[str, float],
    latest_by_root: dict[str, tuple[int, float]],
    chosen_root_id: str | None,
) -> tuple[LocalVsGlobalChoice, ...]:
    """For each distinct first move considered at the root: its own
    immediate one-step score, and how it scored the last time it was still
    part of the beam (which depth that was is reported alongside it -- see
    ``LocalVsGlobalChoice`` for why cross-entry comparison needs that).
    Sorted by immediate score (best first) so a reader immediately sees
    whether the plan chose the move that looked best one step ahead, or a
    different one that led somewhere better -- the concrete local-vs-global
    account the brief asks for.
    """
    return tuple(
        LocalVsGlobalChoice(
            song_id=root_id,
            immediate_transition_score=depth0_immediate_by_root[root_id],
            downstream_path_score=latest_by_root.get(root_id, (0, depth0_immediate_by_root[root_id]))[1],
            depth_reached=latest_by_root.get(root_id, (0, depth0_immediate_by_root[root_id]))[0],
            chosen=(root_id == chosen_root_id),
        )
        for root_id in sorted(depth0_immediate_by_root, key=lambda rid: -depth0_immediate_by_root[rid])
    )
