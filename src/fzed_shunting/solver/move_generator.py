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
    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0) + vehicle.vehicle_length
        )
    effective_capacity_by_track = {
        name: max(cap, initial_occupation_by_track.get(name, 0.0))
        for name, cap in capacity_by_track.items()
    }
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
        emitted_longest_single_target_direct_moves: set[str] = set()
        source_track_is_uniform_goal = _same_goal(seq, goal_by_vehicle)
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
                if _should_skip_shorter_single_target_direct_move(
                    candidate_targets=candidate_targets,
                    emitted_targets=emitted_longest_single_target_direct_moves,
                ):
                    continue
                generated_targets: list[str] = []
                for target_track in candidate_targets:
                    move = _build_candidate_move(
                        source_track=source_track,
                        target_track=target_track,
                        block=block,
                        block_vehicles=block_vehicles,
                        block_length=block_length,
                        state=state,
                        capacity_by_track=effective_capacity_by_track,
                        length_by_vehicle=length_by_vehicle,
                        vehicle_by_no=vehicle_by_no,
                        plan_input=plan_input,
                        route_oracle=route_oracle,
                    )
                    if move is not None:
                        moves.append(move)
                        generated_targets.append(target_track)
                        _record_move_debug_stats(
                            debug_stats,
                            move=move,
                            is_staging=False,
                        )
                if (
                    source_track_is_uniform_goal
                    and len(generated_targets) == 1
                    and not state.track_sequences.get(generated_targets[0])
                    and not _preserve_shorter_single_target_direct_moves(
                        master=master,
                        source_track=source_track,
                        source_seq=seq,
                    )
                ):
                    emitted_longest_single_target_direct_moves.add(generated_targets[0])
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
                        capacity_by_track=effective_capacity_by_track,
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
    moves.extend(
        _generate_capacity_eviction_moves(
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            effective_capacity_by_track=effective_capacity_by_track,
            master=master,
            route_oracle=route_oracle,
        )
    )
    moves.extend(
        _generate_spot_eviction_moves(
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            effective_capacity_by_track=effective_capacity_by_track,
            master=master,
            route_oracle=route_oracle,
        )
    )
    return _dedup_moves(moves)


def generate_real_hook_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
) -> list[HookAction]:
    """Generate ATTACH and DETACH moves for the real-hook cost model.

    Phase-separated: when loco_carry is empty, generate ATTACH moves (pick up
    vehicles from source tracks). When loco_carry is non-empty, generate DETACH
    moves (drop vehicles to goal/staging tracks) and allow additional ATTACHes
    only for vehicles going to the same set of destinations as the current carry
    (multi-pickup optimisation).

    Each action costs 1 hook.
    """
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in plan_input.vehicles}
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0) + vehicle.vehicle_length
        )
    effective_capacity_by_track = {
        name: max(cap, initial_occupation_by_track.get(name, 0.0))
        for name, cap in capacity_by_track.items()
    }
    if route_oracle is None and master is not None:
        route_oracle = RouteOracle(master)

    # Compute the set of goal tracks for currently-carried vehicles.
    # When loco_carry is non-empty, only additional ATTACHes that share
    # this destination set are generated (to limit branching).
    carry_goal_tracks: set[str] = set()
    for vno in state.loco_carry:
        v = vehicle_by_no.get(vno)
        if v is None:
            continue
        carry_goal_tracks.update(v.goal.allowed_target_tracks)

    moves: list[HookAction] = []

    # --- ATTACH moves ---
    # When loco_carry is non-empty, only allow picks whose goal overlaps with
    # carry_goal_tracks (multi-pickup for same destination).  This prunes the
    # branching factor from ~71 to a small set without losing optimal solutions
    # for the common "collect-and-deliver" pattern.
    for source_track, seq in state.track_sequences.items():
        if not seq:
            continue
        for prefix_size in range(len(seq), 0, -1):
            block = seq[:prefix_size]
            block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if all(
                source_track in vehicle_by_no[vno].goal.allowed_target_tracks
                for vno in block
            ):
                continue
            # When already carrying: only pick up if block shares a destination.
            if state.loco_carry:
                block_goals: set[str] = set()
                for vno in block:
                    v = vehicle_by_no.get(vno)
                    if v is not None:
                        block_goals.update(v.goal.allowed_target_tracks)
                if not (block_goals & carry_goal_tracks):
                    continue
            path_tracks = [source_track]
            moves.append(
                HookAction(
                    source_track=source_track,
                    target_track="LOCO",
                    vehicle_nos=list(block),
                    path_tracks=path_tracks,
                    action_type="ATTACH",
                )
            )

    # --- DETACH moves ---
    if state.loco_carry:
        carry = list(state.loco_carry)
        for prefix_size in range(1, len(carry) + 1):
            drop_block = carry[:prefix_size]
            drop_vehicles = [vehicle_by_no[vno] for vno in drop_block if vno in vehicle_by_no]
            if not drop_vehicles:
                continue
            if validate_hook_vehicle_group(drop_vehicles):
                continue
            block_length = sum(length_by_vehicle.get(vno, 0.0) for vno in drop_block)
            goal_targets: set[str] = set()
            for vno in drop_block:
                v = vehicle_by_no.get(vno)
                if v is None:
                    continue
                if v.need_weigh and vno not in state.weighed_vehicle_nos:
                    goal_targets.add("机库")
                else:
                    goal_targets.update(v.goal.allowed_target_tracks)
            detach_targets: list[str] = sorted(goal_targets)
            for target_track in detach_targets:
                move = _build_candidate_move(
                    source_track=state.loco_track_name,
                    target_track=target_track,
                    block=drop_block,
                    block_vehicles=drop_vehicles,
                    block_length=block_length,
                    state=state,
                    capacity_by_track=effective_capacity_by_track,
                    length_by_vehicle=length_by_vehicle,
                    vehicle_by_no=vehicle_by_no,
                    plan_input=plan_input,
                    route_oracle=route_oracle,
                )
                if move is None:
                    continue
                moves.append(
                    HookAction(
                        source_track="LOCO",
                        target_track=target_track,
                        vehicle_nos=list(drop_block),
                        path_tracks=move.path_tracks,
                        action_type="DETACH",
                    )
                )
            staging_targets = _candidate_staging_targets(
                source_track=state.loco_track_name,
                block=drop_block,
                state=state,
                plan_input=plan_input,
                master=master,
                vehicle_by_no=vehicle_by_no,
                goal_target_hints=tuple(sorted(goal_targets)),
                route_oracle=route_oracle,
            )
            staged_count = 0
            for target_track in staging_targets:
                if target_track in goal_targets:
                    continue
                if staged_count > 0 and _is_fallback_staging_track(master, target_track):
                    break
                move = _build_candidate_move(
                    source_track=state.loco_track_name,
                    target_track=target_track,
                    block=drop_block,
                    block_vehicles=drop_vehicles,
                    block_length=block_length,
                    state=state,
                    capacity_by_track=effective_capacity_by_track,
                    length_by_vehicle=length_by_vehicle,
                    vehicle_by_no=vehicle_by_no,
                    plan_input=plan_input,
                    route_oracle=route_oracle,
                )
                if move is None:
                    continue
                moves.append(
                    HookAction(
                        source_track="LOCO",
                        target_track=target_track,
                        vehicle_nos=list(drop_block),
                        path_tracks=move.path_tracks,
                        action_type="DETACH",
                    )
                )
                staged_count += 1
                if staged_count >= MAX_STAGING_TARGETS:
                    break

    return _dedup_moves(moves)


def _generate_capacity_eviction_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
) -> list[HookAction]:
    """Generate staging moves for identity-goal vehicles at capacity-full target tracks.

    When pending single-target arrivals at track T exceed available slack (after
    accounting for non-identity vehicles that will naturally depart), identity-goal
    vehicles at T must be temporarily evicted to make room. Generates those moves.
    """
    current_track_by_vehicle: dict[str, str] = {
        vno: track
        for track, seq in state.track_sequences.items()
        for vno in seq
    }
    pending_arrival_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        allowed = vehicle.goal.allowed_target_tracks
        if len(allowed) != 1:
            continue
        target = allowed[0]
        if current_track_by_vehicle.get(vehicle.vehicle_no) == target:
            continue
        pending_arrival_by_track[target] = (
            pending_arrival_by_track.get(target, 0.0)
            + length_by_vehicle.get(vehicle.vehicle_no, vehicle.vehicle_length)
        )

    eviction_moves: list[HookAction] = []
    for track, seq in state.track_sequences.items():
        if not seq:
            continue
        pending = pending_arrival_by_track.get(track, 0.0)
        if pending <= 1e-9:
            continue
        effective_cap = effective_capacity_by_track.get(track)
        if effective_cap is None:
            continue
        current_mass = sum(length_by_vehicle.get(vno, 0.0) for vno in seq)
        available_slack = effective_cap - current_mass
        # Non-identity vehicles will leave anyway: their departure is free slack
        non_identity_mass = sum(
            length_by_vehicle.get(vno, 0.0)
            for vno in seq
            if not (
                len(goal_by_vehicle[vno].allowed_target_tracks) == 1
                and goal_by_vehicle[vno].allowed_target_tracks[0] == track
            )
        )
        effective_slack = available_slack + non_identity_mass
        if effective_slack >= pending - 1e-9:
            continue
        # Collect consecutive identity-goal prefix sizes from the near end
        eviction_sizes: list[int] = []
        for prefix_size in range(1, len(seq) + 1):
            vno = seq[prefix_size - 1]
            allowed = goal_by_vehicle[vno].allowed_target_tracks
            if len(allowed) != 1 or allowed[0] != track:
                break
            block = seq[:prefix_size]
            block_vehicles = [vehicle_by_no[bvno] for bvno in block]
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if any(
                vehicle_by_no[bvno].need_weigh and bvno not in state.weighed_vehicle_nos
                for bvno in block
            ):
                continue
            eviction_sizes.append(prefix_size)
        if not eviction_sizes:
            continue
        # Use smallest valid eviction to minimise disruption
        prefix_size = eviction_sizes[0]
        block = seq[:prefix_size]
        block_length = sum(length_by_vehicle.get(vno, 0.0) for vno in block)
        block_vehicles = [vehicle_by_no[vno] for vno in block]
        feasible_count = 0
        for staging_target in _candidate_staging_targets(
            source_track=track,
            block=block,
            state=state,
            plan_input=plan_input,
            master=master,
            vehicle_by_no=vehicle_by_no,
            goal_target_hints=(track,),
            route_oracle=route_oracle,
        ):
            if feasible_count > 0 and _is_fallback_staging_track(master, staging_target):
                break
            move = _build_candidate_move(
                source_track=track,
                target_track=staging_target,
                block=block,
                block_vehicles=block_vehicles,
                block_length=block_length,
                state=state,
                capacity_by_track=effective_capacity_by_track,
                length_by_vehicle=length_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
            if move is None:
                continue
            eviction_moves.append(move)
            feasible_count += 1
            if feasible_count >= MAX_STAGING_TARGETS:
                break
    return eviction_moves


def _generate_spot_eviction_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
) -> list[HookAction]:
    """Generate staging moves to free spots blocked by vehicles that need to be evicted.

    Two cases handled:
    1. SPOT-mode: a vehicle needs a specific spot currently held by another vehicle.
    2. AREA-mode overflow: an AREA vehicle can't be allocated to ANY of its allowed
       target tracks because all their spots are full. Evict front vehicle from one target.
    """
    reverse_assignments: dict[str, str] = {
        spot: vno for vno, spot in state.spot_assignments.items()
    }
    current_track_by_vehicle: dict[str, str] = {
        vno: track
        for track, seq in state.track_sequences.items()
        for vno in seq
    }
    # evict_requests[track] = near-end prefix size needed to remove all blockers
    evict_requests: dict[str, int] = {}

    # Case 1: SPOT-mode conflicts
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT":
            continue
        target_spot = goal.target_spot_code
        if target_spot is None:
            continue
        if state.spot_assignments.get(vehicle.vehicle_no) == target_spot:
            continue
        occupant_vno = reverse_assignments.get(target_spot)
        if occupant_vno is None or occupant_vno == vehicle.vehicle_no:
            continue
        occupant_track = current_track_by_vehicle.get(occupant_vno)
        if occupant_track is None:
            continue
        seq = state.track_sequences.get(occupant_track, [])
        try:
            occ_pos = seq.index(occupant_vno)
        except ValueError:
            continue
        evict_requests[occupant_track] = max(evict_requests.get(occupant_track, 0), occ_pos + 1)

    # Case 2: AREA-mode overflow — vehicle can't be allocated to any allowed target
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track in vehicle.goal.allowed_target_tracks:
            continue
        allowed = vehicle.goal.allowed_target_tracks
        if len(allowed) == 0:
            continue
        any_feasible = False
        for target in allowed:
            result = allocate_spots_for_block(
                vehicles=[vehicle],
                target_track=target,
                yard_mode=plan_input.yard_mode,
                occupied_spot_assignments=state.spot_assignments,
            )
            if result is not None:
                any_feasible = True
                break
        if any_feasible:
            continue
        # All targets have full spot assignments: evict front of the first available target
        for target in allowed:
            seq = state.track_sequences.get(target, [])
            if not seq:
                continue
            front_vno = seq[0]
            front_goal = goal_by_vehicle.get(front_vno)
            if front_goal is None:
                continue
            # Don't evict identity-goal vehicles with single target — they have nowhere else to go
            if len(front_goal.allowed_target_tracks) == 1 and front_goal.allowed_target_tracks[0] == target:
                continue
            evict_requests[target] = max(evict_requests.get(target, 0), 1)
            break

    eviction_moves: list[HookAction] = []
    for track, prefix_size in evict_requests.items():
        seq = state.track_sequences.get(track, [])
        if not seq or prefix_size > len(seq):
            continue
        block = seq[:prefix_size]
        block_vehicles = [vehicle_by_no[vno] for vno in block]
        if validate_hook_vehicle_group(block_vehicles):
            continue
        if any(
            vehicle_by_no[vno].need_weigh and vno not in state.weighed_vehicle_nos
            for vno in block
        ):
            continue
        block_length = sum(length_by_vehicle.get(vno, 0.0) for vno in block)
        lead_goal = goal_by_vehicle.get(block[0])
        goal_hints = tuple(sorted(lead_goal.allowed_target_tracks)) if lead_goal else ()
        feasible_count = 0
        for staging_target in _candidate_staging_targets(
            source_track=track,
            block=block,
            state=state,
            plan_input=plan_input,
            master=master,
            vehicle_by_no=vehicle_by_no,
            goal_target_hints=goal_hints,
            route_oracle=route_oracle,
        ):
            if feasible_count > 0 and _is_fallback_staging_track(master, staging_target):
                break
            move = _build_candidate_move(
                source_track=track,
                target_track=staging_target,
                block=block,
                block_vehicles=block_vehicles,
                block_length=block_length,
                state=state,
                capacity_by_track=effective_capacity_by_track,
                length_by_vehicle=length_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
            if move is None:
                continue
            eviction_moves.append(move)
            feasible_count += 1
            if feasible_count >= MAX_STAGING_TARGETS:
                break
    return eviction_moves


def _dedup_moves(moves: list[HookAction]) -> list[HookAction]:
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    result: list[HookAction] = []
    for move in moves:
        key = (
            move.source_track,
            move.target_track,
            tuple(move.vehicle_nos),
            tuple(move.path_tracks),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(move)
    return result


def _should_skip_shorter_single_target_direct_move(
    *,
    candidate_targets: list[str],
    emitted_targets: set[str],
) -> bool:
    return len(candidate_targets) == 1 and candidate_targets[0] in emitted_targets


def _preserve_shorter_single_target_direct_moves(
    *,
    master: MasterData | None,
    source_track: str,
    source_seq: list[str],
) -> bool:
    if master is None:
        return False
    track = master.tracks.get(source_track)
    if track is None:
        return False
    return track.track_type in PRIMARY_STAGING_TRACK_TYPES and len(source_seq) == 2


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
    if _violates_close_door_hook_rule(block, target_track, vehicle_by_no, state):
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
    state: ReplayState,
) -> bool:
    if target_track == "存4北":
        existing_seq = state.track_sequences.get("存4北", [])
        projected_seq = list(existing_seq) + list(block)
        for position_index in range(min(3, len(projected_seq))):
            vehicle_no = projected_seq[position_index]
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            if not vehicle.is_close_door:
                continue
            goal = vehicle.goal
            if (
                goal.target_mode == "TRACK"
                and goal.target_track == "存4北"
                and list(goal.allowed_target_tracks) == ["存4北"]
            ):
                return True
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
