from __future__ import annotations

from typing import Any

from fzed_shunting.domain.depot_spots import allocate_spots_for_block
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

AREA_CAPACITY_LIMITS = {
    "调棚:WORK": 4,
    "洗南:WORK": 3,
    "油:WORK": 2,
    "抛:WORK": 2,
}
PRIMARY_STAGING_TRACK_TYPES = {"TEMPORARY"}
FALLBACK_STAGING_TRACK_TYPES = {"STORAGE"}
MAX_STAGING_TARGETS = 2


def generate_goal_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
    blocking_goal_targets_by_source: dict[str, set[str]] | None = None,
    debug_stats: dict[str, Any] | None = None,
) -> list[HookAction]:
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in plan_input.vehicles}
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    if route_oracle is None and master is not None:
        route_oracle = RouteOracle(master)
    if blocking_goal_targets_by_source is None:
        blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            route_oracle=route_oracle,
        )
    moves: list[HookAction] = []
    if debug_stats is not None:
        debug_stats.clear()
        debug_stats.update(
            {
                "total_moves": 0,
                "direct_moves": 0,
                "staging_moves": 0,
                "moves_by_target": {},
                "moves_by_source": {},
                "moves_by_block_size": {},
            }
        )
    for source_track, seq in state.track_sequences.items():
        if not seq:
            continue
        for prefix_size in range(len(seq), 0, -1):
            block = seq[:prefix_size]
            block_length = sum(length_by_vehicle[vehicle_no] for vehicle_no in block)
            block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if _same_goal(block, goal_by_vehicle):
                goal = goal_by_vehicle[block[0]]
                if _should_skip_pure_random_depot_rebalancing(
                    source_track=source_track,
                    seq=seq,
                    block=block,
                    state=state,
                    goal_by_vehicle=goal_by_vehicle,
                    vehicle_by_no=vehicle_by_no,
                    blocking_goal_targets_by_source=blocking_goal_targets_by_source,
                ):
                    continue
                candidate_targets = _candidate_targets(block, plan_input, state, vehicle_by_no)
                for target_track in candidate_targets:
                    move = _build_candidate_move(
                        source_track=source_track,
                        target_track=target_track,
                        block=block,
                        block_vehicles=block_vehicles,
                        block_length=block_length,
                        state=state,
                        capacity_by_track=capacity_by_track,
                        length_by_vehicle=length_by_vehicle,
                        vehicle_by_no=vehicle_by_no,
                        plan_input=plan_input,
                        route_oracle=route_oracle,
                    )
                    if move is not None:
                        moves.append(move)
                        _record_move_debug_stats(
                            debug_stats,
                            move=move,
                            is_staging=False,
                        )
        staging_requests = _collect_staging_requests_for_source(
            source_track=source_track,
            seq=seq,
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            blocking_goal_targets_by_source=blocking_goal_targets_by_source,
        )
        for prefix_sizes, goal_target_hints in staging_requests:
            for prefix_size in prefix_sizes:
                block = seq[:prefix_size]
                block_length = sum(length_by_vehicle[vehicle_no] for vehicle_no in block)
                block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
                if validate_hook_vehicle_group(block_vehicles):
                    continue
                if any(
                    vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos
                    for vehicle in block_vehicles
                ):
                    continue
                feasible_target_count = 0
                for target_track in _candidate_staging_targets(
                    source_track=source_track,
                    block=block,
                    state=state,
                    plan_input=plan_input,
                    master=master,
                    vehicle_by_no=vehicle_by_no,
                    goal_target_hints=goal_target_hints,
                    route_oracle=route_oracle,
                ):
                    if (
                        feasible_target_count > 0
                        and _is_fallback_staging_track(master, target_track)
                    ):
                        break
                    move = _build_candidate_move(
                        source_track=source_track,
                        target_track=target_track,
                        block=block,
                        block_vehicles=block_vehicles,
                        block_length=block_length,
                        state=state,
                        capacity_by_track=capacity_by_track,
                        length_by_vehicle=length_by_vehicle,
                        vehicle_by_no=vehicle_by_no,
                        plan_input=plan_input,
                        route_oracle=route_oracle,
                    )
                    if move is None:
                        continue
                    moves.append(move)
                    _record_move_debug_stats(
                        debug_stats,
                        move=move,
                        is_staging=True,
                    )
                    feasible_target_count += 1
                    if feasible_target_count >= MAX_STAGING_TARGETS:
                        break
                if feasible_target_count > 0:
                    break
    return moves


def _record_move_debug_stats(
    debug_stats: dict[str, Any] | None,
    *,
    move: HookAction,
    is_staging: bool,
) -> None:
    if debug_stats is None:
        return
    debug_stats["total_moves"] += 1
    if is_staging:
        debug_stats["staging_moves"] += 1
    else:
        debug_stats["direct_moves"] += 1
    debug_stats["moves_by_target"][move.target_track] = (
        debug_stats["moves_by_target"].get(move.target_track, 0) + 1
    )
    debug_stats["moves_by_source"][move.source_track] = (
        debug_stats["moves_by_source"].get(move.source_track, 0) + 1
    )
    block_size = len(move.vehicle_nos)
    debug_stats["moves_by_block_size"][block_size] = (
        debug_stats["moves_by_block_size"].get(block_size, 0) + 1
    )


def _same_goal(block: list[str], goal_by_vehicle: dict) -> bool:
    first = goal_by_vehicle[block[0]]
    key = (
        first.target_mode,
        first.target_track,
        tuple(first.allowed_target_tracks),
        first.target_area_code,
        first.target_spot_code,
    )
    for vehicle_no in block[1:]:
        goal = goal_by_vehicle[vehicle_no]
        other_key = (
            goal.target_mode,
            goal.target_track,
            tuple(goal.allowed_target_tracks),
            goal.target_area_code,
            goal.target_spot_code,
        )
        if other_key != key:
            return False
    return True


def _fits_capacity(
    target_track: str,
    block_length: float,
    state: ReplayState,
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
) -> bool:
    capacity = capacity_by_track.get(target_track)
    if capacity is None:
        return False
    current_length = sum(length_by_vehicle[vehicle_no] for vehicle_no in state.track_sequences.get(target_track, []))
    return current_length + block_length <= capacity + 1e-9


def _candidate_targets(
    block: list[str],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
) -> list[str]:
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    if any(
        vehicle_by_no[vehicle_no].need_weigh and vehicle_no not in state.weighed_vehicle_nos
        for vehicle_no in block
    ):
        return ["机库"]
    goal = vehicle_by_no[block[0]].goal
    targets = list(goal.allowed_target_tracks)
    if goal.target_area_code == "大库:RANDOM":
        targets.sort(
            key=lambda track_name: (
                -sum(
                    length_by_vehicle[vehicle_no]
                    for vehicle_no in state.track_sequences.get(track_name, [])
                ),
                track_name,
            )
        )
    return targets


def _candidate_staging_targets(
    source_track: str,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    master: MasterData | None,
    vehicle_by_no: dict,
    goal_target_hints: tuple[str, ...],
    route_oracle: RouteOracle | None,
) -> list[str]:
    if master is None:
        return []
    targets: list[tuple[tuple[float, float, str], str]] = []
    for info in plan_input.track_info:
        if info.track_name == source_track:
            continue
        type_priority = _staging_track_priority(master, info.track_name)
        if type_priority is None:
            continue
        if any(info.track_name in vehicle_by_no[vehicle_no].goal.allowed_target_tracks for vehicle_no in block):
            continue
        source_distance = _route_distance(route_oracle, source_track, info.track_name)
        if source_distance is None:
            continue
        combined_distance = source_distance
        if goal_target_hints:
            follow_distances = [
                distance
                for distance in (
                    _route_distance(route_oracle, info.track_name, goal_track)
                    for goal_track in goal_target_hints
                )
                if distance is not None
            ]
            if follow_distances:
                combined_distance = source_distance + min(follow_distances)
        targets.append(((type_priority, combined_distance, source_distance, info.track_name), info.track_name))
    targets.sort(key=lambda item: item[0])
    return [track_name for _, track_name in targets]


def _is_fallback_staging_track(
    master: MasterData | None,
    track_name: str,
) -> bool:
    type_priority = _staging_track_priority(master, track_name)
    return type_priority == 1


def _staging_track_priority(
    master: MasterData | None,
    track_name: str,
) -> int | None:
    if master is None:
        return None
    track = master.tracks.get(track_name)
    if track is None:
        return None
    if track.track_type in PRIMARY_STAGING_TRACK_TYPES:
        return 0
    if track.track_type in FALLBACK_STAGING_TRACK_TYPES:
        return 1
    return None


def _collect_staging_requests_for_source(
    source_track: str,
    seq: list[str],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    blocking_goal_targets_by_source: dict[str, set[str]],
) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:
    requests: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    front_prefix_size = _front_blocker_prefix_size(source_track, seq, goal_by_vehicle)
    if front_prefix_size is not None:
        next_vehicle_no = seq[front_prefix_size]
        target_hints = _candidate_targets(
            [next_vehicle_no],
            plan_input,
            state,
            vehicle_by_no,
        )
        requests.append(((front_prefix_size,), tuple(sorted(target_hints))))

    interfering_target_hints = blocking_goal_targets_by_source.get(source_track)
    if interfering_target_hints:
        prefix_sizes = _descending_valid_staging_prefix_sizes(seq, vehicle_by_no)
        if prefix_sizes:
            requests.append((prefix_sizes, tuple(sorted(interfering_target_hints))))

    return requests


def _should_skip_pure_random_depot_rebalancing(
    *,
    source_track: str,
    seq: list[str],
    block: list[str],
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    blocking_goal_targets_by_source: dict[str, set[str]],
) -> bool:
    goal = goal_by_vehicle[block[0]]
    if goal.target_area_code != "大库:RANDOM":
        return False
    if source_track not in goal.allowed_target_tracks:
        return False
    if any(
        goal_by_vehicle[vehicle_no].allowed_target_tracks != goal.allowed_target_tracks
        for vehicle_no in block
    ):
        return False
    if any(
        goal_by_vehicle[vehicle_no].target_area_code != goal.target_area_code
        for vehicle_no in block
    ):
        return False
    if any(
        vehicle_by_no[vehicle_no].need_weigh and vehicle_no not in state.weighed_vehicle_nos
        for vehicle_no in block
    ):
        return False
    if source_track in blocking_goal_targets_by_source:
        return False
    return _front_blocker_prefix_size(source_track, seq, goal_by_vehicle) != len(block)


def _front_blocker_prefix_size(
    source_track: str,
    seq: list[str],
    goal_by_vehicle: dict,
) -> int | None:
    if len(seq) <= 1:
        return None
    first_goal = goal_by_vehicle[seq[0]]
    if source_track not in first_goal.allowed_target_tracks:
        return None
    prefix_size = 1
    while prefix_size < len(seq) and _goal_key(goal_by_vehicle[seq[prefix_size]]) == _goal_key(first_goal):
        prefix_size += 1
    if prefix_size >= len(seq):
        return None
    return prefix_size


def _goal_key(goal: Any) -> tuple:
    return (
        goal.target_mode,
        goal.target_track,
        tuple(goal.allowed_target_tracks),
        goal.target_area_code,
        goal.target_spot_code,
    )


def _descending_valid_staging_prefix_sizes(
    seq: list[str],
    vehicle_by_no: dict,
) -> tuple[int, ...]:
    prefix_sizes: list[int] = []
    for prefix_size in range(len(seq), 0, -1):
        block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in seq[:prefix_size]]
        if not validate_hook_vehicle_group(block_vehicles):
            prefix_sizes.append(prefix_size)
    return tuple(prefix_sizes)


def _collect_interfering_goal_targets_by_source(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    route_oracle: RouteOracle | None,
) -> dict[str, set[str]]:
    if route_oracle is None:
        return {}
    blocking_goal_targets_by_source: dict[str, set[str]] = {}
    for source_track, seq in state.track_sequences.items():
        if not seq:
            continue
        for prefix_size in range(1, len(seq) + 1):
            block = seq[:prefix_size]
            if not _same_goal(block, goal_by_vehicle):
                continue
            candidate_targets = _candidate_targets(block, plan_input, state, vehicle_by_no)
            for target_track in candidate_targets:
                if target_track == source_track:
                    continue
                path_tracks = route_oracle.resolve_path_tracks(source_track, target_track)
                if path_tracks is None:
                    continue
                for track_code in path_tracks[1:-1]:
                    if not state.track_sequences.get(track_code):
                        continue
                    blocking_goal_targets_by_source.setdefault(track_code, set()).add(target_track)
    return blocking_goal_targets_by_source


def _route_distance(
    route_oracle: RouteOracle | None,
    source_track: str,
    target_track: str,
) -> float | None:
    if route_oracle is None:
        return 0.0 if source_track != target_track else None
    route = route_oracle.resolve_route(source_track, target_track)
    if route is None:
        return None
    return route.total_length_m


def _build_candidate_move(
    source_track: str,
    target_track: str,
    block: list[str],
    block_vehicles: list,
    block_length: float,
    state: ReplayState,
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
    vehicle_by_no: dict,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
) -> HookAction | None:
    if target_track == source_track:
        return None
    if _violates_close_door_hook_rule(block, target_track, vehicle_by_no):
        return None
    if not _fits_capacity(
        target_track,
        block_length,
        state,
        capacity_by_track,
        length_by_vehicle,
    ):
        return None
    if not _fits_area_capacity(block, target_track, state, vehicle_by_no):
        return None
    if allocate_spots_for_block(
        vehicles=block_vehicles,
        target_track=target_track,
        yard_mode=plan_input.yard_mode,
        occupied_spot_assignments=state.spot_assignments,
    ) is None:
        return None
    path_tracks = [source_track, target_track]
    if route_oracle is not None:
        resolved_path_tracks = route_oracle.resolve_path_tracks(source_track, target_track)
        if resolved_path_tracks is None:
            return None
        path_tracks = resolved_path_tracks
        resolved_route = route_oracle.resolve_route(source_track, target_track)
        if resolved_route is None:
            return None
        route_result = route_oracle.validate_path(
            source_track=source_track,
            target_track=target_track,
            path_tracks=path_tracks,
            train_length_m=block_length,
            occupied_track_sequences=state.track_sequences,
            expected_path_tracks=path_tracks,
            route=resolved_route,
        )
        if not route_result.is_valid:
            return None
    return HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(block),
        path_tracks=path_tracks,
    )


def _violates_close_door_hook_rule(
    block: list[str],
    target_track: str,
    vehicle_by_no: dict,
) -> bool:
    if target_track == "存4北":
        return False
    if len(block) <= 10:
        return False
    first_vehicle = vehicle_by_no[block[0]]
    return bool(first_vehicle.is_close_door)


def _fits_area_capacity(
    block: list[str],
    target_track: str,
    state: ReplayState,
    vehicle_by_no: dict,
) -> bool:
    vehicle = vehicle_by_no[block[0]]
    area_code = vehicle.goal.target_area_code
    if area_code not in AREA_CAPACITY_LIMITS:
        return True
    limit = AREA_CAPACITY_LIMITS[area_code]
    current_count = 0
    for vehicle_no in state.track_sequences.get(target_track, []):
        existing = vehicle_by_no.get(vehicle_no)
        if existing and existing.goal.target_area_code == area_code:
            current_count += 1
    return current_count + len(block) <= limit
