"""A* / weighted / beam search core for the shunting solver.

Implements a single configurable search loop. Mode selection (``exact`` /
``weighted`` / ``beam``) changes priority calculation and pruning, not the
loop shape itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import count
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.heuristic import make_state_heuristic, make_state_heuristic_real_hook
from fzed_shunting.solver.move_generator import (
    _collect_interfering_goal_targets_by_source,
    generate_goal_moves,
    generate_real_hook_moves,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.state import (
    _apply_move,
    _canonical_random_depot_vehicle_nos,
    _is_goal,
    _state_key,
)
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

BEAM_SHALLOW_RESERVE = 1


@dataclass(order=True)
class QueueItem:
    priority: tuple
    seq: int
    state_key: tuple
    state: ReplayState
    plan: list[HookAction]


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
) -> SolverResult:
    if budget is None:
        budget = SearchBudget()
    else:
        budget.reset()
    started_at = budget.started_at
    counter = count()
    queue: list[QueueItem] = []
    canonical_random_depot_vehicle_nos = _canonical_random_depot_vehicle_nos(plan_input)
    state_heuristic = (
        make_state_heuristic_real_hook(plan_input)
        if solver_mode == "real_hook"
        else make_state_heuristic(plan_input)
    )
    initial_key = _state_key(
        initial_state,
        canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
    )
    initial_heuristic = state_heuristic(initial_state)
    heappush(
        queue,
        QueueItem(
            priority=_priority(
                cost=0,
                heuristic=initial_heuristic,
                solver_mode=solver_mode,
                heuristic_weight=heuristic_weight,
                neg_depot_index_sum=0,
            ),
            seq=next(counter),
            state_key=initial_key,
            state=initial_state,
            plan=[],
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
    budget_exhausted = False
    while queue:
        if budget.exhausted():
            budget_exhausted = True
            break
        current = heappop(queue)
        current_cost = len(current.plan)
        if best_cost.get(current.state_key) != current_cost:
            continue
        expanded_nodes += 1
        budget.tick_expand()
        closed_state_keys.add(current.state_key)
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
                    is_proven_optimal=solver_mode == "exact",
                    fallback_stage=solver_mode,
                    debug_stats=debug_stats,
                )
            continue
        move_stats = {} if debug_stats is not None else None
        blocking_goal_targets_by_source: dict[str, set[str]] = {}
        if solver_mode == "real_hook":
            moves = generate_real_hook_moves(
                plan_input,
                current.state,
                master=master,
                route_oracle=route_oracle,
            )
        else:
            blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
                plan_input=plan_input,
                state=current.state,
                goal_by_vehicle=goal_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                route_oracle=route_oracle,
            )
            moves = generate_goal_moves(
                plan_input,
                current.state,
                master=master,
                route_oracle=route_oracle,
                blocking_goal_targets_by_source=blocking_goal_targets_by_source,
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
            blocker_bonus = _blocking_goal_target_bonus(
                state=current.state,
                move=move,
                blocking_goal_targets_by_source=blocking_goal_targets_by_source,
            )
            neg_depot_index_sum = 0
            if enable_depot_late_scheduling and solver_mode == "exact":
                # Secondary key is only engaged in exact mode where the
                # admissible heuristic guarantees f-tie tiebreaking preserves
                # optimality. In heuristic modes (weighted/beam/greedy) an
                # injected secondary can redirect search into worse
                # primary-objective regions, so leave priority unchanged
                # there and rely on LNS and post-processing reorder for
                # depot-lateness.
                from fzed_shunting.solver.depot_late import weighted_depot_index_sum
                neg_depot_index_sum = -weighted_depot_index_sum(next_plan, vehicle_by_no)
            heappush(
                queue,
                QueueItem(
                    priority=_priority(
                        cost=cost,
                        heuristic=heuristic,
                        blocker_bonus=blocker_bonus,
                        solver_mode=solver_mode,
                        heuristic_weight=heuristic_weight,
                        neg_depot_index_sum=neg_depot_index_sum,
                    ),
                    seq=next(counter),
                    state_key=state_key,
                    state=next_state,
                    plan=next_plan,
                ),
            )
            generated_nodes += 1
            if debug_stats is not None:
                debug_stats["generated_nodes"] = generated_nodes
        if solver_mode == "beam" and beam_width is not None:
            _prune_queue(queue, best_cost, beam_width)
    if best_goal_plan is None:
        if budget_exhausted:
            return SolverResult(
                plan=[],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=len(closed_state_keys),
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_proven_optimal=False,
                fallback_stage=solver_mode,
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
        is_proven_optimal=proven_optimal,
        fallback_stage=solver_mode,
        debug_stats=debug_stats,
    )


def _priority(
    *,
    cost: int,
    heuristic: int,
    blocker_bonus: int = 0,
    solver_mode: str,
    heuristic_weight: float,
    neg_depot_index_sum: int = 0,
) -> tuple:
    if solver_mode == "beam":
        beam_heuristic_credit = 1 if blocker_bonus > 0 else 0
        adjusted_heuristic = heuristic - beam_heuristic_credit
        return (
            cost + adjusted_heuristic,
            cost,
            neg_depot_index_sum,
            adjusted_heuristic,
            -blocker_bonus,
            heuristic,
        )
    if solver_mode in ("weighted", "real_hook"):
        return (
            cost + heuristic_weight * heuristic,
            cost,
            neg_depot_index_sum,
            heuristic,
            -blocker_bonus,
        )
    return (
        cost + heuristic,
        cost,
        neg_depot_index_sum,
        heuristic,
        -blocker_bonus,
    )


def _blocking_goal_target_bonus(
    *,
    state: ReplayState,
    move: HookAction,
    blocking_goal_targets_by_source: dict[str, set[str]],
) -> int:
    blocking_targets = blocking_goal_targets_by_source.get(move.source_track)
    if not blocking_targets:
        return 0
    if move.target_track not in blocking_targets:
        return 0
    source_seq = state.track_sequences.get(move.source_track, [])
    if len(move.vehicle_nos) != len(source_seq):
        return 0
    if tuple(source_seq[: len(move.vehicle_nos)]) != tuple(move.vehicle_nos):
        return 0
    return len(blocking_targets)


def _prune_queue(
    queue: list[QueueItem],
    best_cost: dict[tuple, int],
    beam_width: int,
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
            if item.priority[4] < 0 and id(item) not in kept_ids
        ]
        if blocker_candidates:
            blocker_item = blocker_candidates[0]
            kept.append(blocker_item)
            kept_ids.add(id(blocker_item))
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
