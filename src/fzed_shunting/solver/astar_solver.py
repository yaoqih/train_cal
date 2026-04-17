from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import count
from time import perf_counter
from typing import Any

from fzed_shunting.domain.depot_spots import spot_candidates_for_vehicle
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.move_generator import generate_goal_moves
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState, replay_plan


@dataclass(order=True)
class QueueItem:
    priority: tuple[float, int, int]
    seq: int
    state_key: tuple
    state: ReplayState
    plan: list[HookAction]


@dataclass(frozen=True)
class SolverResult:
    plan: list[HookAction]
    expanded_nodes: int
    generated_nodes: int
    closed_nodes: int
    elapsed_ms: float
    debug_stats: dict[str, Any] | None = None


def solve_with_simple_astar(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    solver_mode: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    debug_stats: dict[str, Any] | None = None,
) -> list[HookAction]:
    return solve_with_simple_astar_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        debug_stats=debug_stats,
    ).plan


def solve_with_simple_astar_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    solver_mode: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    debug_stats: dict[str, Any] | None = None,
) -> SolverResult:
    _validate_solver_options(
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
    )
    if solver_mode == "lns":
        return _solve_with_lns_result(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            debug_stats=debug_stats,
        )
    return _solve_search_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        debug_stats=debug_stats,
    )


def _solve_search_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
    debug_stats: dict[str, Any] | None = None,
) -> SolverResult:
    started_at = perf_counter()
    counter = count()
    queue: list[QueueItem] = []
    initial_key = _state_key(initial_state)
    initial_heuristic = _heuristic(plan_input, initial_state)
    heappush(
        queue,
        QueueItem(
            priority=_priority(
                cost=0,
                heuristic=initial_heuristic,
                solver_mode=solver_mode,
                heuristic_weight=heuristic_weight,
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
    if debug_stats is not None:
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
    while queue:
        current = heappop(queue)
        current_cost = len(current.plan)
        if best_cost.get(current.state_key) != current_cost:
            continue
        expanded_nodes += 1
        closed_state_keys.add(current.state_key)
        if debug_stats is not None:
            debug_stats["expanded_states"] = expanded_nodes
            debug_stats["closed_states"] = len(closed_state_keys)
        if _is_goal(plan_input, current.state):
            return SolverResult(
                plan=current.plan,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=len(closed_state_keys),
                elapsed_ms=(perf_counter() - started_at) * 1000,
                debug_stats=debug_stats,
            )
        move_stats = {} if debug_stats is not None else None
        moves = generate_goal_moves(
            plan_input,
            current.state,
            master=master,
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
            next_state = replay_plan(
                current.state,
                [_to_hook_dict(move)],
                plan_input=plan_input,
            ).final_state
            next_plan = current.plan + [move]
            cost = len(next_plan)
            state_key = _state_key(next_state)
            if state_key in best_cost and best_cost[state_key] <= cost:
                continue
            best_cost[state_key] = cost
            heuristic = _heuristic(plan_input, next_state)
            heappush(
                queue,
                QueueItem(
                    priority=_priority(
                        cost=cost,
                        heuristic=heuristic,
                        solver_mode=solver_mode,
                        heuristic_weight=heuristic_weight,
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
    raise ValueError("No solution found")


def _is_goal(plan_input: NormalizedPlanInput, state: ReplayState) -> bool:
    for vehicle in plan_input.vehicles:
        current_track = _locate_vehicle(state, vehicle.vehicle_no)
        if current_track not in vehicle.goal.allowed_target_tracks:
            return False
        if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
            return False
        if vehicle.goal.target_mode == "SPOT":
            if state.spot_assignments.get(vehicle.vehicle_no) != vehicle.goal.target_spot_code:
                return False
        if vehicle.goal.target_area_code == "大库:RANDOM" and current_track in vehicle.goal.allowed_target_tracks:
            assigned_spot = state.spot_assignments.get(vehicle.vehicle_no)
            if assigned_spot is None:
                return False
            if assigned_spot not in spot_candidates_for_vehicle(vehicle, current_track, plan_input.yard_mode):
                return False
        if vehicle.goal.target_area_code in {"调棚:WORK", "调棚:PRE_REPAIR", "洗南:WORK", "油:WORK", "抛:WORK"}:
            assigned_spot = state.spot_assignments.get(vehicle.vehicle_no)
            if assigned_spot is None:
                return False
            if assigned_spot not in spot_candidates_for_vehicle(vehicle, current_track, plan_input.yard_mode):
                return False
        if vehicle.is_close_door and current_track == "存4北":
            final_seq = state.track_sequences.get("存4北", [])
            if final_seq.index(vehicle.vehicle_no) < 3:
                return False
    return True


def _heuristic(plan_input: NormalizedPlanInput, state: ReplayState) -> int:
    remaining = 0
    for vehicle in plan_input.vehicles:
        current_track = _locate_vehicle(state, vehicle.vehicle_no)
        if current_track not in vehicle.goal.allowed_target_tracks:
            remaining += 1
    return remaining


def _locate_vehicle(state: ReplayState, vehicle_no: str) -> str:
    for track, seq in state.track_sequences.items():
        if vehicle_no in seq:
            return track
    raise ValueError(f"Vehicle not found in state: {vehicle_no}")


def _state_key(state: ReplayState) -> tuple:
    return (
        state.loco_track_name,
        tuple((track, tuple(seq)) for track, seq in sorted(state.track_sequences.items())),
        tuple(sorted(state.weighed_vehicle_nos)),
        tuple(sorted(state.spot_assignments.items())),
    )


def _priority(
    *,
    cost: int,
    heuristic: int,
    solver_mode: str,
    heuristic_weight: float,
) -> tuple[float, int, int]:
    if solver_mode == "weighted":
        return (cost + heuristic_weight * heuristic, cost, heuristic)
    return (cost + heuristic, cost, heuristic)


def _validate_solver_options(
    *,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
) -> None:
    if solver_mode not in {"exact", "weighted", "beam", "lns"}:
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be >= 1.0")
    if beam_width is not None and beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    if solver_mode == "beam" and beam_width is None:
        raise ValueError("beam_width is required when solver_mode=beam")


def _prune_queue(
    queue: list[QueueItem],
    best_cost: dict[tuple, int],
    beam_width: int,
) -> None:
    if len(queue) <= beam_width:
        return
    ranked = sorted(queue)
    kept = ranked[:beam_width]
    pruned = ranked[beam_width:]
    queue[:] = kept
    heapify(queue)
    for item in pruned:
        if best_cost.get(item.state_key) == len(item.plan):
            del best_cost[item.state_key]


def _solve_with_lns_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    heuristic_weight: float,
    beam_width: int | None,
    repair_passes: int = 4,
    debug_stats: dict[str, Any] | None = None,
) -> SolverResult:
    started_at = perf_counter()
    seed_solver_mode = "beam" if beam_width is not None else "weighted"
    seed_debug_stats = debug_stats if debug_stats is not None else None
    incumbent = _solve_search_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        solver_mode=seed_solver_mode,
        heuristic_weight=max(heuristic_weight, 1.5),
        beam_width=beam_width,
        debug_stats=seed_debug_stats,
    )
    incumbent_plan = list(incumbent.plan)
    total_expanded = incumbent.expanded_nodes
    total_generated = incumbent.generated_nodes
    total_closed = incumbent.closed_nodes
    route_oracle = RouteOracle(master) if master is not None else None

    while True:
        snapshots = replay_plan(
            initial_state,
            [_to_hook_dict(move) for move in incumbent_plan],
            plan_input=plan_input,
        ).snapshots
        improved = False
        for cut_index in _candidate_repair_cut_points(incumbent_plan, repair_passes):
            prefix = incumbent_plan[:cut_index]
            start_state = snapshots[cut_index]
            repair_input = _build_repair_plan_input(plan_input, start_state)
            try:
                repaired = _solve_search_result(
                    plan_input=repair_input,
                    initial_state=start_state,
                    master=master,
                    solver_mode="exact" if beam_width is None else "beam",
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    debug_stats=None,
                )
            except ValueError:
                continue
            total_expanded += repaired.expanded_nodes
            total_generated += repaired.generated_nodes
            total_closed += repaired.closed_nodes
            candidate_plan = prefix + repaired.plan
            if _is_better_plan(candidate_plan, incumbent_plan, route_oracle):
                incumbent_plan = candidate_plan
                improved = True
                break
        if not improved:
            break

    return SolverResult(
        plan=incumbent_plan,
        expanded_nodes=total_expanded,
        generated_nodes=total_generated,
        closed_nodes=total_closed,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        debug_stats=debug_stats,
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


def _candidate_repair_cut_points(plan: list[HookAction], repair_passes: int) -> list[int]:
    if not plan:
        return []
    touch_count_by_track: dict[str, int] = {}
    for move in plan:
        touch_count_by_track[move.source_track] = touch_count_by_track.get(move.source_track, 0) + 1
        touch_count_by_track[move.target_track] = touch_count_by_track.get(move.target_track, 0) + 1

    scored = sorted(
        enumerate(plan),
        key=lambda item: _repair_cut_score(item[0], item[1], touch_count_by_track),
    )
    return [idx for idx, _ in scored[:repair_passes]]


def _repair_cut_score(
    index: int,
    move: HookAction,
    touch_count_by_track: dict[str, int],
) -> tuple[int, int, int, int]:
    hotspot_score = max(
        touch_count_by_track.get(move.source_track, 0),
        touch_count_by_track.get(move.target_track, 0),
    )
    staging_flag = int(
        move.source_track.startswith("临")
        or move.target_track.startswith("临")
        or move.source_track == "存4南"
        or move.target_track == "存4南"
    )
    repeated_touch_flag = int(hotspot_score > 1)
    return (
        -repeated_touch_flag,
        -staging_flag,
        -hotspot_score,
        index,
    )


def _is_better_plan(
    candidate_plan: list[HookAction],
    incumbent_plan: list[HookAction],
    route_oracle: RouteOracle | None,
) -> bool:
    candidate_metrics = _plan_quality(candidate_plan, route_oracle)
    incumbent_metrics = _plan_quality(incumbent_plan, route_oracle)
    return candidate_metrics < incumbent_metrics


def _plan_quality(
    plan: list[HookAction],
    route_oracle: RouteOracle | None,
) -> tuple[int, int, float]:
    total_length_m = 0.0
    total_branch_count = 0
    if route_oracle is None:
        total_length_m = float(sum(len(move.path_tracks) for move in plan))
        total_branch_count = int(sum(max(len(move.path_tracks) - 1, 0) for move in plan))
    else:
        for move in plan:
            route = route_oracle.resolve_route(move.source_track, move.target_track)
            if route is not None:
                total_length_m += route.total_length_m
                total_branch_count += len(route.branch_codes)
            else:
                total_branch_count += max(len(move.path_tracks) - 1, 0)
                total_length_m += float(len(move.path_tracks))
    return (len(plan), total_branch_count, total_length_m)


def _build_repair_plan_input(
    plan_input: NormalizedPlanInput,
    snapshot: ReplayState,
) -> NormalizedPlanInput:
    localized_vehicles: list[NormalizedVehicle] = []
    for vehicle in plan_input.vehicles:
        current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
        frozen_goal = vehicle.goal
        goal_satisfied = (
            current_track in vehicle.goal.allowed_target_tracks
            and (not vehicle.need_weigh or vehicle.vehicle_no in snapshot.weighed_vehicle_nos)
            and _spot_goal_satisfied(vehicle, snapshot)
        )
        if goal_satisfied:
            frozen_goal = GoalSpec(
                target_mode="TRACK",
                target_track=current_track,
                allowed_target_tracks=[current_track],
                target_area_code=vehicle.goal.target_area_code,
                target_spot_code=snapshot.spot_assignments.get(vehicle.vehicle_no),
            )
        localized_vehicles.append(
            vehicle.model_copy(
                update={
                    "current_track": current_track,
                    "goal": frozen_goal,
                }
            )
        )
    return plan_input.model_copy(
        update={
            "vehicles": localized_vehicles,
            "loco_track_name": snapshot.loco_track_name,
        }
    )


def _spot_goal_satisfied(vehicle: NormalizedVehicle, snapshot: ReplayState) -> bool:
    assigned_spot = snapshot.spot_assignments.get(vehicle.vehicle_no)
    if vehicle.goal.target_mode == "SPOT":
        return assigned_spot == vehicle.goal.target_spot_code
    if vehicle.goal.target_area_code == "大库:RANDOM":
        current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
        return assigned_spot in spot_candidates_for_vehicle(vehicle, current_track, "NORMAL")
    if vehicle.goal.target_area_code in {"调棚:WORK", "调棚:PRE_REPAIR", "洗南:WORK", "油:WORK", "抛:WORK"}:
        current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
        return assigned_spot in spot_candidates_for_vehicle(vehicle, current_track, "NORMAL")
    return True


def _to_hook_dict(move: HookAction) -> dict:
    return {
        "hookNo": 1,
        "actionType": move.action_type,
        "sourceTrack": move.source_track,
        "targetTrack": move.target_track,
        "vehicleNos": move.vehicle_nos,
        "pathTracks": move.path_tracks,
    }
