"""A* / weighted / beam search core for the shunting solver.

Implements a single configurable search loop. Mode selection (``exact`` /
``weighted`` / ``beam``) changes priority calculation and pruning, not the
loop shape itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from itertools import count
from time import perf_counter
from typing import Any

from fzed_shunting.domain.carry_order import is_carried_tail_block
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.exact_spot import (
    exact_spot_clearance_bonus,
    exact_spot_seeker_exposure_bonus,
)
from fzed_shunting.solver.heuristic import make_state_heuristic_real_hook
from fzed_shunting.solver.move_generator import (
    _collect_interfering_goal_targets_by_source,
    generate_real_hook_moves,
)
from fzed_shunting.solver.purity import compute_state_purity
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.route_blockage import (
    RouteBlockagePlan,
    compute_route_blockage_plan,
    route_blockage_release_score,
    route_release_continuation_bonus,
    route_release_focus_after_move,
)
from fzed_shunting.solver.state import (
    _apply_move,
    _canonical_random_depot_vehicle_nos,
    _is_goal,
    _state_key,
)
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

BEAM_SHALLOW_RESERVE = 1
BEAM_LOW_STRUCTURAL_DEBT_RESERVE = 1
BEAM_STRUCTURAL_DIVERSITY_FRONTIER_MULTIPLIER = 3
BEAM_STRUCTURAL_DIVERSITY_MIN_STAGING_TO_STAGING = 80
BEAM_STRUCTURAL_DIVERSITY_MIN_REPEATED_STAGING_TOUCHES = 80


@dataclass(order=True)
class QueueItem:
    priority: tuple
    seq: int
    state_key: tuple
    state: ReplayState
    plan: list[HookAction]
    cost: int = field(default=0, compare=False)
    structural_key: tuple[int, ...] = field(
        default=(-1, 0, 0, 0, 0, 0, 0),
        compare=False,
    )
    route_release_focus_tracks: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
    )
    route_release_focus_bonus: int = field(default=0, compare=False)
    route_release_focus_ttl: int = field(default=0, compare=False)


def _solve_search_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
    debug_stats: dict[str, Any] | None = None,
    budget: SearchBudget | None = None,
    enable_depot_late_scheduling: bool = False,
    enable_structural_diversity: bool = False,
) -> SolverResult:
    if budget is None:
        budget = SearchBudget()
    else:
        budget.reset()
    started_at = budget.started_at
    counter = count()
    queue: list[QueueItem] = []
    canonical_random_depot_vehicle_nos = _canonical_random_depot_vehicle_nos(plan_input)
    state_heuristic = make_state_heuristic_real_hook(plan_input)
    initial_key = _state_key(
        initial_state,
        canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
    )
    initial_heuristic = state_heuristic(initial_state)
    initial_purity = compute_state_purity(plan_input, initial_state)
    heappush(
        queue,
        QueueItem(
            priority=_priority(
                cost=0,
                heuristic=initial_heuristic,
                solver_mode=solver_mode,
                heuristic_weight=heuristic_weight,
                neg_depot_index_sum=0,
                carry_count=len(initial_state.loco_carry),
                carry_growth_penalty=0,
                purity_metrics=(
                    initial_purity.unfinished_count,
                    0,
                    initial_purity.preferred_violation_count,
                    initial_purity.staging_pollution_count,
                ),
            ),
            seq=next(counter),
            state_key=initial_key,
            state=initial_state,
            plan=[],
            structural_key=_approximate_beam_structural_key([], initial_state),
        ),
    )
    best_cost = {initial_key: 0}
    generated_nodes = 1
    expanded_nodes = 0
    closed_state_keys: set[tuple] = set()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in plan_input.vehicles}
    route_oracle = RouteOracle(master) if master is not None else None
    _initialize_debug_stats(debug_stats, generated_nodes=generated_nodes)
    best_goal_plan: list[HookAction] | None = None
    best_partial_plan: list[HookAction] = []
    best_partial_score: tuple[int, int, int] | None = None
    budget_exhausted = False
    while queue:
        if budget.exhausted():
            budget_exhausted = True
            break
        current = heappop(queue)
        current_cost = current.cost
        if best_cost.get(current.state_key) != current_cost:
            continue
        expanded_nodes += 1
        budget.tick_expand()
        closed_state_keys.add(current.state_key)
        if current.plan:
            partial_purity = compute_state_purity(plan_input, current.state)
            partial_structural = compute_structural_metrics(plan_input, current.state)
            partial_score = (
                state_heuristic(current.state),
                partial_purity.unfinished_count + partial_structural.target_sequence_defect_count,
                -len(current.plan),
            )
            if best_partial_score is None or partial_score < best_partial_score:
                best_partial_score = partial_score
                best_partial_plan = list(current.plan)
        if debug_stats is not None:
            debug_stats["expanded_states"] = expanded_nodes
            debug_stats["closed_states"] = len(closed_state_keys)
        if _is_goal(plan_input, current.state):
            if best_goal_plan is None or len(current.plan) < len(best_goal_plan):
                best_goal_plan = current.plan
            if solver_mode in ("exact", "real_hook"):
                return SolverResult(
                    plan=current.plan,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    closed_nodes=len(closed_state_keys),
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    is_complete=True,
                    is_proven_optimal=solver_mode == "exact",
                    fallback_stage=solver_mode,
                    debug_stats=debug_stats,
                )
            continue
        move_stats = {} if debug_stats is not None else None
        blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
            plan_input=plan_input,
            state=current.state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            route_oracle=route_oracle,
        )
        route_blockage_plan = (
            compute_route_blockage_plan(plan_input, current.state, route_oracle)
            if route_oracle is not None
            else None
        )
        moves = generate_real_hook_moves(
            plan_input,
            current.state,
            master=master,
            route_oracle=route_oracle,
            debug_stats=move_stats,
        )
        if debug_stats is not None:
            _accumulate_move_debug_stats(
                debug_stats=debug_stats,
                state=current.state,
                plan_len=len(current.plan),
                move_stats=move_stats or {},
            )
        for move in moves:
            next_state = _apply_move(
                state=current.state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            next_plan = current.plan + [move]
            cost = len(next_plan)
            state_key = _state_key(
                next_state,
                canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
            )
            if state_key in best_cost and best_cost[state_key] <= cost:
                continue
            best_cost[state_key] = cost
            heuristic = state_heuristic(next_state)
            next_purity = compute_state_purity(plan_input, next_state)
            blocker_bonus = _blocking_goal_target_bonus(
                state=current.state,
                move=move,
                blocking_goal_targets_by_source=blocking_goal_targets_by_source,
                route_blockage_plan=route_blockage_plan,
                route_release_focus_tracks=(
                    current.route_release_focus_tracks
                    if current.route_release_focus_ttl > 0
                    else frozenset()
                ),
                route_release_focus_bonus=current.route_release_focus_bonus,
            )
            blocker_bonus += exact_spot_clearance_bonus(
                plan_input=plan_input,
                state=current.state,
                move=move,
                next_state=next_state,
            )
            blocker_bonus += exact_spot_seeker_exposure_bonus(
                plan_input=plan_input,
                state=current.state,
                move=move,
                next_state=next_state,
            )
            blocker_bonus += _carry_exposure_bonus(
                move=move,
                state=current.state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            next_focus_tracks, next_focus_bonus, next_focus_ttl = route_release_focus_after_move(
                prior_focus_tracks=current.route_release_focus_tracks,
                prior_focus_bonus=current.route_release_focus_bonus,
                prior_focus_ttl=current.route_release_focus_ttl,
                move=move,
                route_blockage_plan=route_blockage_plan,
            )
            route_release_regression_penalty = _route_release_regression_penalty(
                state=next_state,
                route_oracle=route_oracle,
                focus_tracks=next_focus_tracks,
                focus_ttl=next_focus_ttl,
            )
            route_release_regression_penalty += _route_release_repark_penalty(
                move=move,
                state=current.state,
                next_state=next_state,
                plan_input=plan_input,
                route_oracle=route_oracle,
                route_blockage_plan=route_blockage_plan,
                focus_tracks=(
                    current.route_release_focus_tracks
                    if current.route_release_focus_ttl > 0
                    else frozenset()
                ),
                focus_ttl=current.route_release_focus_ttl,
            )
            neg_depot_index_sum = 0
            if enable_depot_late_scheduling and solver_mode == "exact":
                from fzed_shunting.solver.depot_late import weighted_depot_index_sum
                neg_depot_index_sum = -weighted_depot_index_sum(next_plan, vehicle_by_no)
            heappush(
                queue,
                QueueItem(
                    priority=_priority(
                        cost=cost,
                        heuristic=heuristic,
                        blocker_bonus=blocker_bonus,
                        route_release_regression_penalty=route_release_regression_penalty,
                        solver_mode=solver_mode,
                        heuristic_weight=heuristic_weight,
                        neg_depot_index_sum=neg_depot_index_sum,
                        purity_metrics=(
                            next_purity.unfinished_count,
                            0,
                            next_purity.preferred_violation_count,
                            next_purity.staging_pollution_count,
                        ),
                        carry_count=len(next_state.loco_carry),
                        carry_growth_penalty=_carry_growth_penalty_for_move(
                            move=move,
                            state=current.state,
                            vehicle_by_no=vehicle_by_no,
                        ),
                    ),
                    seq=next(counter),
                    state_key=state_key,
                    state=next_state,
                    plan=next_plan,
                    cost=cost,
                    structural_key=_approximate_beam_structural_key(
                        next_plan,
                        next_state,
                    ),
                    route_release_focus_tracks=next_focus_tracks,
                    route_release_focus_bonus=next_focus_bonus,
                    route_release_focus_ttl=next_focus_ttl,
                ),
            )
            generated_nodes += 1
            if debug_stats is not None:
                debug_stats["generated_nodes"] = generated_nodes
        if solver_mode == "beam" and beam_width is not None:
            _prune_queue(
                queue,
                best_cost,
                beam_width,
                plan_input=plan_input,
                enable_structural_diversity=enable_structural_diversity,
            )
    if best_goal_plan is None:
        if debug_stats is not None and best_partial_score is not None:
            debug_stats["search_best_partial_score"] = list(best_partial_score)
        if budget_exhausted:
            return SolverResult(
                plan=[],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=len(closed_state_keys),
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_complete=False,
                is_proven_optimal=False,
                fallback_stage=solver_mode,
                partial_plan=best_partial_plan,
                partial_fallback_stage=solver_mode if best_partial_plan else None,
                debug_stats=debug_stats,
            )
        raise ValueError("No solution found")
    proven_optimal = (not budget_exhausted) and solver_mode == "exact"
    return SolverResult(
        plan=best_goal_plan,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        closed_nodes=len(closed_state_keys),
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=proven_optimal,
        fallback_stage=solver_mode,
        debug_stats=debug_stats,
    )


def _priority(
    *,
    cost: int,
    heuristic: int,
    blocker_bonus: int = 0,
    route_release_regression_penalty: int = 0,
    solver_mode: str,
    heuristic_weight: float,
    neg_depot_index_sum: int = 0,
    purity_metrics: tuple[int, ...] = (0, 0, 0, 0),
    carry_count: int = 0,
    carry_growth_penalty: int = 0,
) -> tuple:
    if solver_mode == "beam":
        score = cost + heuristic
        return (
            route_release_regression_penalty,
            0 if blocker_bonus > 0 else 1,
            score + carry_growth_penalty,
            score,
            cost,
            heuristic,
            purity_metrics,
            carry_growth_penalty,
            carry_count,
            neg_depot_index_sum,
            -blocker_bonus,
            heuristic,
        )
    if solver_mode in ("weighted", "real_hook"):
        return (
            route_release_regression_penalty,
            0 if blocker_bonus > 0 else 1,
            cost + heuristic_weight * heuristic,
            cost,
            heuristic,
            purity_metrics,
            carry_growth_penalty,
            carry_count,
            neg_depot_index_sum,
            -blocker_bonus,
        )
    return (
        route_release_regression_penalty,
        cost + heuristic,
        cost,
        heuristic,
        purity_metrics,
        carry_growth_penalty,
        carry_count,
        neg_depot_index_sum,
        -blocker_bonus,
    )


def _carry_growth_penalty_for_move(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    if move.action_type != "ATTACH":
        return 0

    penalty = len(move.vehicle_nos) if state.loco_carry else 0
    projected_carry_count = len(state.loco_carry) + len(move.vehicle_nos)
    if projected_carry_count <= 10:
        return penalty

    if _is_soft_random_area_attach(move=move, vehicle_by_no=vehicle_by_no):
        penalty += (projected_carry_count - 10) * 2
    return penalty


def _is_soft_random_area_attach(
    *,
    move: HookAction,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if not move.vehicle_nos:
        return False
    for vehicle_no in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        goal = vehicle.goal
        if goal.target_mode not in {"AREA", "SNAPSHOT"}:
            return False
        if goal.target_area_code is None or ":RANDOM" not in goal.target_area_code:
            return False
    return True


def _route_release_regression_penalty(
    *,
    state: ReplayState,
    route_oracle: RouteOracle | None,
    focus_tracks: frozenset[str] | set[str],
    focus_ttl: int,
) -> int:
    if route_oracle is None or focus_ttl <= 0 or not focus_tracks:
        return 0
    penalty = 0
    for focus_track in sorted(focus_tracks):
        if focus_track == state.loco_track_name:
            continue
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=focus_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
        )
        if not access_result.is_valid:
            penalty += max(1, len(access_result.blocking_tracks))
    return penalty * max(1, focus_ttl)


def _route_release_repark_penalty(
    *,
    move: HookAction,
    state: ReplayState,
    next_state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
    route_blockage_plan: RouteBlockagePlan | None,
    focus_tracks: frozenset[str] | set[str],
    focus_ttl: int,
) -> int:
    if (
        route_oracle is None
        or route_blockage_plan is None
        or focus_ttl <= 0
        or not focus_tracks
        or move.action_type != "DETACH"
        or not state.loco_carry
    ):
        return 0
    if move.target_track not in route_blockage_plan.facts_by_blocking_track:
        return 0
    before_pressure = route_blockage_plan.total_blockage_pressure
    after_plan = compute_route_blockage_plan(plan_input, next_state, route_oracle)
    after_pressure = after_plan.total_blockage_pressure
    if after_pressure <= 0:
        return 0
    blocked_focus_tracks = _blocked_focus_tracks(
        state=next_state,
        route_oracle=route_oracle,
        focus_tracks=focus_tracks,
    )
    if not blocked_focus_tracks:
        return 0
    return max(1, after_pressure + max(0, after_pressure - before_pressure)) * max(1, focus_ttl)


def _blocked_focus_tracks(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    focus_tracks: frozenset[str] | set[str],
) -> set[str]:
    blocked: set[str] = set()
    for focus_track in sorted(focus_tracks):
        if focus_track == state.loco_track_name:
            continue
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=focus_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
        )
        if not access_result.is_valid:
            blocked.add(focus_track)
    return blocked


def _blocking_goal_target_bonus(
    *,
    state: ReplayState,
    move: HookAction,
    blocking_goal_targets_by_source: dict[str, set[str]],
    route_blockage_plan: RouteBlockagePlan | None = None,
    route_release_focus_tracks: frozenset[str] | set[str] | None = None,
    route_release_focus_bonus: int = 0,
) -> int:
    route_release_bonus = route_blockage_release_score(
        source_track=move.source_track,
        vehicle_nos=move.vehicle_nos,
        route_blockage_plan=route_blockage_plan,
    )
    continuation_bonus = route_release_continuation_bonus(
        state=state,
        move=move,
        focus_tracks=route_release_focus_tracks,
        focus_bonus=route_release_focus_bonus,
    )
    blocking_targets = blocking_goal_targets_by_source.get(move.source_track)
    if not blocking_targets:
        return route_release_bonus + continuation_bonus
    source_seq = state.track_sequences.get(move.source_track, [])
    if not move.vehicle_nos:
        return route_release_bonus + continuation_bonus
    if tuple(source_seq[: len(move.vehicle_nos)]) != tuple(move.vehicle_nos):
        return route_release_bonus + continuation_bonus
    if move.action_type == "ATTACH":
        return len(blocking_targets) + route_release_bonus + continuation_bonus
    if move.target_track not in blocking_targets:
        return route_release_bonus + continuation_bonus
    return len(blocking_targets) + route_release_bonus + continuation_bonus


def _carry_exposure_bonus(
    *,
    move: HookAction,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, Any],
) -> int:
    if move.action_type != "DETACH":
        return 0
    carry = list(state.loco_carry)
    tail_size = len(move.vehicle_nos)
    if not carry or not is_carried_tail_block(carry, move.vehicle_nos):
        return 0

    last_committed_index: int | None = None
    for index in range(len(carry) - 1, -1, -1):
        vehicle_no = carry[index]
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        if _is_critical_carry_vehicle(vehicle, state=state, plan_input=plan_input):
            last_committed_index = index
            break
    if last_committed_index is None:
        return 0
    tail_distance = len(carry) - 1 - last_committed_index
    if tail_distance == 0 or tail_size > tail_distance:
        return 0
    return 1


def _is_critical_carry_vehicle(
    vehicle: Any,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if getattr(vehicle, "need_weigh", False) and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return True
    if vehicle.goal.target_mode == "SPOT":
        return True
    if vehicle.goal.target_area_code not in {None, "大库:RANDOM", "大库外:RANDOM"}:
        return True
    return bool(vehicle.goal.preferred_target_tracks and not vehicle.goal.fallback_target_tracks)


def _prune_queue(
    queue: list[QueueItem],
    best_cost: dict[tuple, int],
    beam_width: int,
    *,
    plan_input: NormalizedPlanInput | None = None,
    enable_structural_diversity: bool = False,
) -> None:
    if len(queue) <= beam_width:
        return
    ranked = sorted(queue)
    if beam_width <= 1:
        kept = ranked[:beam_width]
    else:
        kept: list[QueueItem] = []
        kept_ids: set[int] = set()
        shallow_depth = min(len(item.plan) for item in ranked)
        shallow_candidates = [
            item
            for item in ranked
            if len(item.plan) == shallow_depth
        ]
        for item in shallow_candidates[:BEAM_SHALLOW_RESERVE]:
            kept.append(item)
            kept_ids.add(id(item))
        blocker_candidates = [
            item
            for item in ranked
            if item.priority[-2] < 0 and id(item) not in kept_ids
        ]
        if blocker_candidates:
            blocker_item = blocker_candidates[0]
            kept.append(blocker_item)
            kept_ids.add(id(blocker_item))
        if enable_structural_diversity and _has_structural_churn_pressure(ranked[:beam_width]):
            frontier_limit = max(
                beam_width,
                beam_width * BEAM_STRUCTURAL_DIVERSITY_FRONTIER_MULTIPLIER,
            )
            low_debt_candidates = [
                item
                for item in ranked[:frontier_limit]
                if id(item) not in kept_ids
            ]
            if plan_input is not None:
                for item in low_debt_candidates:
                    if item.structural_key[0] < 0:
                        item.structural_key = _full_beam_structural_key(
                            plan_input,
                            item.plan,
                            item.state,
                        )
            low_debt_candidates.sort(key=lambda item: (item.structural_key, item.priority))
            for item in low_debt_candidates[:BEAM_LOW_STRUCTURAL_DEBT_RESERVE]:
                if len(kept) >= beam_width:
                    break
                kept.append(item)
                kept_ids.add(id(item))
        for item in ranked:
            if len(kept) >= beam_width:
                break
            if id(item) in kept_ids:
                continue
            kept.append(item)
            kept_ids.add(id(item))
        kept.sort()
    kept_ids = {id(item) for item in kept}
    pruned = [item for item in ranked if id(item) not in kept_ids]
    queue[:] = kept
    heapify(queue)
    for item in pruned:
        if best_cost.get(item.state_key) == len(item.plan):
            del best_cost[item.state_key]


def _has_structural_churn_pressure(queue: list[QueueItem]) -> bool:
    return any(
        item.structural_key[2] >= BEAM_STRUCTURAL_DIVERSITY_MIN_STAGING_TO_STAGING
        or item.structural_key[3] >= BEAM_STRUCTURAL_DIVERSITY_MIN_REPEATED_STAGING_TOUCHES
        for item in queue
    )


def _beam_structural_key(
    plan: list[HookAction],
    state: ReplayState,
    *,
    target_sequence_defect_count: int = 0,
) -> tuple[int, ...]:
    staging_hooks = 0
    staging_to_staging_hooks = 0
    repeated_staging_touches = 0
    touched: dict[str, int] = {}
    for move in plan:
        source_is_staging = move.source_track in STAGING_TRACKS
        target_is_staging = move.target_track in STAGING_TRACKS
        if source_is_staging or target_is_staging:
            staging_hooks += 1
        if source_is_staging and target_is_staging:
            staging_to_staging_hooks += 1
        for vehicle_no in move.vehicle_nos:
            touched[vehicle_no] = touched.get(vehicle_no, 0) + 1
            if (source_is_staging or target_is_staging) and touched[vehicle_no] > 2:
                repeated_staging_touches += 1
    staging_debt = sum(
        len(seq)
        for track, seq in state.track_sequences.items()
        if track in STAGING_TRACKS
    )
    max_touch = max(touched.values(), default=0)
    return (
        target_sequence_defect_count,
        staging_debt,
        staging_to_staging_hooks,
        repeated_staging_touches,
        staging_hooks,
        max_touch,
        len(plan),
    )


def _approximate_beam_structural_key(plan: list[HookAction], state: ReplayState) -> tuple[int, ...]:
    structural_key = _beam_structural_key(plan, state)
    return (-1, *structural_key[1:])


def _full_beam_structural_key(
    plan_input: NormalizedPlanInput,
    plan: list[HookAction],
    state: ReplayState,
) -> tuple[int, ...]:
    structural = compute_structural_metrics(plan_input, state)
    return _beam_structural_key(
        plan,
        state,
        target_sequence_defect_count=structural.target_sequence_defect_count,
    )


def _initialize_debug_stats(
    debug_stats: dict[str, Any] | None,
    *,
    generated_nodes: int,
) -> None:
    if debug_stats is None:
        return
    debug_stats.clear()
    debug_stats.update(
        {
            "expanded_states": 0,
            "generated_nodes": generated_nodes,
            "closed_states": 0,
            "move_generation_calls": 0,
            "candidate_moves_total": 0,
            "candidate_direct_moves": 0,
            "candidate_staging_moves": 0,
            "max_candidate_moves_per_state": 0,
            "states_with_zero_moves": 0,
            "moves_by_target": {},
            "moves_by_source": {},
            "moves_by_block_size": {},
            "top_expansions": [],
        }
    )


def _accumulate_move_debug_stats(
    *,
    debug_stats: dict[str, Any],
    state: ReplayState,
    plan_len: int,
    move_stats: dict[str, Any],
) -> None:
    move_count = int(move_stats.get("total_moves", 0))
    debug_stats["move_generation_calls"] += 1
    debug_stats["candidate_moves_total"] += move_count
    debug_stats["candidate_direct_moves"] += int(move_stats.get("direct_moves", 0))
    debug_stats["candidate_staging_moves"] += int(move_stats.get("staging_moves", 0))
    debug_stats["max_candidate_moves_per_state"] = max(
        debug_stats["max_candidate_moves_per_state"],
        move_count,
    )
    if move_count == 0:
        debug_stats["states_with_zero_moves"] += 1
    _merge_counter_dict(debug_stats["moves_by_target"], move_stats.get("moves_by_target", {}))
    _merge_counter_dict(debug_stats["moves_by_source"], move_stats.get("moves_by_source", {}))
    _merge_counter_dict(debug_stats["moves_by_block_size"], move_stats.get("moves_by_block_size", {}))
    top_expansions = debug_stats["top_expansions"]
    top_expansions.append(
        {
            "plan_len": plan_len,
            "loco_track": state.loco_track_name,
            "move_count": move_count,
            "direct_moves": int(move_stats.get("direct_moves", 0)),
            "staging_moves": int(move_stats.get("staging_moves", 0)),
            "sources": move_stats.get("moves_by_source", {}),
            "targets": move_stats.get("moves_by_target", {}),
        }
    )
    top_expansions.sort(key=lambda item: item["move_count"], reverse=True)
    del top_expansions[10:]


def _merge_counter_dict(target: dict[str, int], incoming: dict[Any, int]) -> None:
    for key, value in incoming.items():
        target[str(key)] = target.get(str(key), 0) + int(value)
