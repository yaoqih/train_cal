from __future__ import annotations

from typing import Any

from fzed_shunting.domain.depot_spots import (
    allocate_spots_for_block,
    exact_spot_reservations,
    is_depot_inner_track,
    realign_spots_for_track_order,
)
from fzed_shunting.domain.carry_order import iter_carried_tail_blocks
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.work_positions import (
    is_work_position_track,
    preview_work_positions_after_prepend,
    work_slot_violations_by_vehicle,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import (
    goal_effective_allowed_tracks,
    goal_is_preferred_satisfied,
    goal_is_satisfied,
    goal_track_preference_level,
)
from fzed_shunting.solver.route_blockage import RouteBlockagePlan, compute_route_blockage_plan
from fzed_shunting.solver.state import _apply_move, _is_goal, _state_key, _vehicle_track_lookup
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

PRIMARY_STAGING_TRACK_TYPES = {"TEMPORARY"}
FALLBACK_STAGING_TRACK_TYPES = {"STORAGE"}
MAX_STAGING_TARGETS = 2
MAX_PRIMARY_STAGING_TARGETS_BEFORE_STORAGE = 2
MAX_STORAGE_STAGING_TARGETS = 1
ROUTE_RELEASE_PRIMARY_STAGING_TARGETS = 2

RouteBlockageCache = dict[tuple, RouteBlockagePlan]


def _track_sequences_in_order(state: ReplayState):
    return state.track_sequences.items()


def _route_blockage_plan_for_state(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle | None,
    cache: RouteBlockageCache | None,
    blocked_source_tracks: set[str] | frozenset[str] | None = None,
) -> RouteBlockagePlan:
    if route_oracle is None:
        return RouteBlockagePlan(facts_by_blocking_track={})
    cache_key = (
        _state_key(state, plan_input),
        tuple(sorted(blocked_source_tracks or ())),
    )
    if cache is None:
        return compute_route_blockage_plan(
            plan_input,
            state,
            route_oracle,
            blocked_source_tracks=blocked_source_tracks,
        )
    cached = cache.get(cache_key)
    if cached is None:
        cached = compute_route_blockage_plan(
            plan_input,
            state,
            route_oracle,
            blocked_source_tracks=blocked_source_tracks,
        )
        cache[cache_key] = cached
    return cached

def _effective_target_tracks_for_detach(
    vehicle_no: str,
    *,
    vehicle_by_no: dict,
    weighed_vehicle_nos: set[str],
) -> set[str]:
    vehicle = vehicle_by_no.get(vehicle_no)
    if vehicle is None:
        return set()
    if vehicle.need_weigh and vehicle_no not in weighed_vehicle_nos:
        return {"机库"}
    return set(vehicle.goal.allowed_target_tracks)


def _min_detach_groups(
    vehicle_nos: tuple[str, ...] | list[str],
    *,
    vehicle_by_no: dict,
    weighed_vehicle_nos: set[str],
) -> int:
    if not vehicle_nos:
        return 0
    groups = 0
    shared_targets: set[str] = set()
    for vehicle_no in reversed(tuple(vehicle_nos)):
        effective_targets = _effective_target_tracks_for_detach(
            vehicle_no,
            vehicle_by_no=vehicle_by_no,
            weighed_vehicle_nos=weighed_vehicle_nos,
        )
        if not effective_targets:
            return len(vehicle_nos)
        if not shared_targets:
            groups += 1
            shared_targets = effective_targets.copy()
            continue
        next_shared_targets = shared_targets & effective_targets
        if next_shared_targets:
            shared_targets = next_shared_targets
            continue
        groups += 1
        shared_targets = effective_targets.copy()
    return groups


def _vehicle_goal_satisfied_on_track(
    vehicle: NormalizedVehicle,
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    return goal_is_satisfied(
        vehicle,
        track_name=track_name,
        state=state,
        plan_input=plan_input,
    )


def _collect_real_hook_identity_attach_requests(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    route_blockage_cache: RouteBlockageCache | None = None,
) -> dict[str, set[int]]:
    requests_by_source: dict[str, set[int]] = {}
    blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
        plan_input=plan_input,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )

    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        front_prefix_size = _front_blocker_prefix_size(
            source_track,
            seq,
            goal_by_vehicle,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if front_prefix_size is not None:
            requests_by_source.setdefault(source_track, set()).add(front_prefix_size)
        if source_track in blocking_goal_targets_by_source:
            clear_prefix_sizes = _access_blocker_clear_prefix_sizes(
                source_track=source_track,
                seq=seq,
                plan_input=plan_input,
                state=state,
                vehicle_by_no=vehicle_by_no,
                length_by_vehicle=length_by_vehicle,
                effective_capacity_by_track=effective_capacity_by_track,
                master=master,
                route_oracle=route_oracle,
                route_blockage_cache=route_blockage_cache,
                goal_target_hints=tuple(sorted(blocking_goal_targets_by_source[source_track])),
            )
            if clear_prefix_sizes:
                requests_by_source.setdefault(source_track, set()).update(clear_prefix_sizes)
        rebalance_prefix_sizes = _random_depot_preferred_rebalance_prefix_sizes(
            source_track=source_track,
            seq=seq,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if rebalance_prefix_sizes:
            requests_by_source.setdefault(source_track, set()).update(rebalance_prefix_sizes)

    pending_arrival_by_track: dict[str, float] = {}
    current_track_by_vehicle = _vehicle_track_lookup(state)
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

    for track, seq in _track_sequences_in_order(state):
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
        eviction_sizes: list[int] = []
        for prefix_size in range(1, len(seq) + 1):
            vno = seq[prefix_size - 1]
            allowed = goal_by_vehicle[vno].allowed_target_tracks
            if len(allowed) != 1 or allowed[0] != track:
                break
            block_vehicles = [vehicle_by_no[bvno] for bvno in seq[:prefix_size]]
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if any(
                vehicle_by_no[bvno].need_weigh and bvno not in state.weighed_vehicle_nos
                for bvno in seq[:prefix_size]
            ):
                continue
            eviction_sizes.append(prefix_size)
        if eviction_sizes:
            requests_by_source.setdefault(track, set()).add(eviction_sizes[0])

    reverse_assignments: dict[str, str] = {
        spot: vno for vno, spot in state.spot_assignments.items()
    }
    evict_requests: dict[str, int] = {}
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT":
            continue
        target_spot = goal.target_spot_code
        if target_spot is None or state.spot_assignments.get(vehicle.vehicle_no) == target_spot:
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

    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if (
            current_track is not None
            and goal_is_satisfied(
                vehicle,
                track_name=current_track,
                state=state,
                plan_input=plan_input,
            )
        ):
            continue
        allowed = vehicle.goal.allowed_target_tracks
        if not allowed:
            continue
        if vehicle.goal.target_mode != "WORK_POSITION":
            any_feasible = False
            for target in allowed:
                result = allocate_spots_for_block(
                    vehicles=[vehicle],
                    target_track=target,
                    yard_mode=plan_input.yard_mode,
                    occupied_spot_assignments=state.spot_assignments,
                    reserved_spot_codes=exact_spot_reservations(plan_input),
                )
                if result is not None:
                    any_feasible = True
                    break
            if any_feasible:
                continue
        for target in allowed:
            seq = state.track_sequences.get(target, [])
            if not seq:
                continue
            front_vno = seq[0]
            front_vehicle = vehicle_by_no.get(front_vno)
            if front_vehicle is None:
                continue
            if goal_is_satisfied(
                front_vehicle,
                track_name=target,
                state=state,
                plan_input=plan_input,
            ):
                continue
            evict_requests[target] = max(evict_requests.get(target, 0), 1)
            break

    for track, prefix_size in sorted(evict_requests.items()):
        seq = state.track_sequences.get(track, [])
        if seq and prefix_size <= len(seq):
            requests_by_source.setdefault(track, set()).add(prefix_size)

    return requests_by_source


def generate_goal_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
    blocking_goal_targets_by_source: dict[str, set[str]] | None = None,
    debug_stats: dict[str, Any] | None = None,
    route_blockage_cache: RouteBlockageCache | None = None,
) -> list[HookAction]:
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in plan_input.vehicles}
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    reserved_spot_codes = exact_spot_reservations(plan_input)
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
    for source_track, seq in _track_sequences_in_order(state):
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
                    plan_input=plan_input,
                    goal_by_vehicle=goal_by_vehicle,
                    vehicle_by_no=vehicle_by_no,
                    blocking_goal_targets_by_source=blocking_goal_targets_by_source,
                ):
                    continue
                candidate_targets = _candidate_targets(
                    block,
                    plan_input,
                    state,
                    vehicle_by_no,
                    route_oracle=route_oracle,
                )
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
                    route_blockage_cache=route_blockage_cache,
                    penalize_goal_corridor_reblock=False,
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
            route_blockage_cache=route_blockage_cache,
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
            route_blockage_cache=route_blockage_cache,
        )
    )
    return _dedup_moves(moves)


def generate_real_hook_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
    debug_stats: dict[str, Any] | None = None,
    require_empty_carry_followup: bool = True,
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
    reserved_spot_codes = exact_spot_reservations(plan_input)
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
    route_blockage_cache: RouteBlockageCache = {}

    current_carry_detach_groups = (
        _min_detach_groups(
            state.loco_carry,
            vehicle_by_no=vehicle_by_no,
            weighed_vehicle_nos=state.weighed_vehicle_nos,
        )
        if state.loco_carry
        else 0
    )
    identity_attach_requests = (
        _collect_real_hook_identity_attach_requests(
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            effective_capacity_by_track=effective_capacity_by_track,
            master=master,
            route_oracle=route_oracle,
            route_blockage_cache=route_blockage_cache,
        )
        if not state.loco_carry
        else {}
    )
    blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
        plan_input=plan_input,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )
    if not state.loco_carry:
        for track, prefix_sizes in _collect_real_hook_access_blocker_attach_requests(
            plan_input=plan_input,
            state=state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            effective_capacity_by_track=effective_capacity_by_track,
            master=master,
            route_oracle=route_oracle,
            route_blockage_cache=route_blockage_cache,
        ).items():
            identity_attach_requests.setdefault(track, set()).update(prefix_sizes)

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
    followup_attach_cache: dict[tuple, bool] = {}

    # --- ATTACH moves ---
    # When loco_carry is non-empty, only keep extra ATTACHes that can reduce
    # the remaining DETACH-group count versus solving the new block separately.
    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        requested_prefix_sizes = identity_attach_requests.get(source_track, set())
        frontier_attach_prefix_sizes = (
            _real_hook_attach_frontier_prefix_sizes(
                seq=seq,
                requested_prefix_sizes=requested_prefix_sizes,
                state=state,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
            )
            if not state.loco_carry
            else None
        )
        for prefix_size in range(len(seq), 0, -1):
            if frontier_attach_prefix_sizes is not None and prefix_size not in frontier_attach_prefix_sizes:
                continue
            block = seq[:prefix_size]
            block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if _violates_non_cun4bei_attach_close_door_rule(block, vehicle_by_no):
                continue
            block_goal_satisfied = all(
                _vehicle_goal_satisfied_on_track(
                    vehicle_by_no[vno],
                    track_name=source_track,
                    state=state,
                    plan_input=plan_input,
                )
                for vno in block
            )
            if block_goal_satisfied and prefix_size not in requested_prefix_sizes:
                continue
            if (
                not state.loco_carry
                and prefix_size not in requested_prefix_sizes
                and _same_goal(block, goal_by_vehicle)
                and (
                    _should_skip_pure_random_depot_rebalancing(
                        source_track=source_track,
                        seq=seq,
                        block=block,
                        state=state,
                        plan_input=plan_input,
                        goal_by_vehicle=goal_by_vehicle,
                        vehicle_by_no=vehicle_by_no,
                        blocking_goal_targets_by_source=blocking_goal_targets_by_source,
                    )
                )
            ):
                continue
            if route_oracle is not None:
                access_result = route_oracle.validate_loco_access(
                    loco_track=state.loco_track_name,
                    target_track=source_track,
                    occupied_track_sequences=state.track_sequences,
                    carried_train_length_m=sum(
                        length_by_vehicle.get(vehicle_no, 0.0)
                        for vehicle_no in state.loco_carry
                    ),
                    loco_node=state.loco_node,
                )
                if not access_result.is_valid:
                    continue
            if state.loco_carry:
                carried_length = sum(
                    length_by_vehicle.get(vehicle_no, 0.0)
                    for vehicle_no in state.loco_carry
                )
                source_effective_capacity = effective_capacity_by_track.get(source_track)
                if (
                    route_oracle is not None
                    and source_effective_capacity is not None
                    and carried_length + route_oracle.master.business_rules.loco_length_m
                    > source_effective_capacity + 1e-9
                ):
                    continue
                block_detach_groups = _min_detach_groups(
                    block,
                    vehicle_by_no=vehicle_by_no,
                    weighed_vehicle_nos=state.weighed_vehicle_nos,
                )
                combined_detach_groups = _min_detach_groups(
                    (*state.loco_carry, *block),
                    vehicle_by_no=vehicle_by_no,
                    weighed_vehicle_nos=state.weighed_vehicle_nos,
                )
                if combined_detach_groups >= current_carry_detach_groups + block_detach_groups:
                    continue
                # Weighing-last constraint: if carry already has an unweighed
                # need_weigh vehicle, only ATTACH more need_weigh vehicles so
                # the weigh group stays at the tail (DETACHed last to 机库).
                carry_has_unweighed_needweigh = any(
                    (cv := vehicle_by_no.get(cv_no)) is not None
                    and cv.need_weigh
                    and cv_no not in state.weighed_vehicle_nos
                    for cv_no in state.loco_carry
                )
                if carry_has_unweighed_needweigh:
                    block_all_needweigh = all(
                        (bv := vehicle_by_no.get(bv_no)) is not None
                        and bv.need_weigh
                        and bv_no not in state.weighed_vehicle_nos
                        for bv_no in block
                    )
                    if not block_all_needweigh:
                        continue
                if not _combined_carry_has_legal_detach_after_attach(
                    source_track=source_track,
                    block=block,
                    state=state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                    length_by_vehicle=length_by_vehicle,
                    effective_capacity_by_track=effective_capacity_by_track,
                    route_oracle=route_oracle,
                ):
                    continue
            path_tracks = [source_track]
            move = HookAction(
                source_track=source_track,
                target_track=source_track,
                vehicle_nos=list(block),
                path_tracks=path_tracks,
                action_type="ATTACH",
            )
            moves.append(move)
            _record_move_debug_stats(
                debug_stats,
                move=move,
                is_staging=block_goal_satisfied and prefix_size in requested_prefix_sizes,
            )

    # --- DETACH moves ---
    if state.loco_carry:
        for drop_block in iter_carried_tail_blocks(state.loco_carry):
            drop_vehicles = [vehicle_by_no[vno] for vno in drop_block if vno in vehicle_by_no]
            if not drop_vehicles:
                continue
            if validate_hook_vehicle_group(drop_vehicles):
                continue
            block_length = sum(length_by_vehicle.get(vno, 0.0) for vno in drop_block)
            carried_route_length = sum(
                length_by_vehicle.get(vno, 0.0)
                for vno in state.loco_carry
            )
            goal_targets: set[str] = set()
            for vno in drop_block:
                v = vehicle_by_no.get(vno)
                if v is None:
                    continue
                if v.need_weigh and vno not in state.weighed_vehicle_nos:
                    goal_targets.add("机库")
                else:
                    goal_targets.update(
                        goal_effective_allowed_tracks(
                            v,
                            state=state,
                            plan_input=plan_input,
                            route_oracle=route_oracle,
                        )
                    )
            if _tail_detach_exposes_carried_vehicle(
                carry=state.loco_carry,
                drop_block=drop_block,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
                state=state,
            ):
                goal_targets.add(state.loco_track_name)
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
                    allow_same_track=True,
                    route_train_length_m=carried_route_length,
                )
                if move is None:
                    continue
                if (
                    target_track == state.loco_track_name
                    and len(drop_block) == len(state.loco_carry)
                    and not _is_goal_after_detach(
                        state=state,
                        move=move,
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                    )
                ):
                    continue
                if (
                    target_track == state.loco_track_name
                    and len(drop_block) == len(state.loco_carry)
                    and _same_track_detach_buries_unfinished_vehicle(
                        state=state,
                        move=move,
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                    )
                ):
                    continue
                detach_move = HookAction(
                    source_track=state.loco_track_name,
                    target_track=target_track,
                    vehicle_nos=list(drop_block),
                    path_tracks=move.path_tracks,
                    action_type="DETACH",
                )
                if not _empty_carry_detach_has_followup_attach(
                    detach_move=detach_move,
                    drop_block=drop_block,
                    state=state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                    master=master,
                    route_oracle=route_oracle,
                    followup_attach_cache=followup_attach_cache,
                    required=require_empty_carry_followup,
                ):
                    continue
                moves.append(detach_move)
                _record_move_debug_stats(
                    debug_stats,
                    move=detach_move,
                    is_staging=False,
                )
            staging_targets = _candidate_staging_targets(
                source_track=state.loco_track_name,
                block=drop_block,
                state=state,
                plan_input=plan_input,
                master=master,
                vehicle_by_no=vehicle_by_no,
                goal_target_hints=tuple(
                    sorted(
                        goal_targets
                        | _route_blockage_target_hints_for_source(
                            source_track=state.loco_track_name,
                            plan_input=plan_input,
                            state=state,
                            route_oracle=route_oracle,
                            route_blockage_cache=route_blockage_cache,
                        )
                    )
                ),
                route_oracle=route_oracle,
                route_blockage_cache=route_blockage_cache,
            )
            primary_staging_limit = (
                ROUTE_RELEASE_PRIMARY_STAGING_TARGETS
                if _is_carrying_active_route_blocker(
                    source_track=state.loco_track_name,
                    block=drop_block,
                    state=state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                    route_oracle=route_oracle,
                    route_blockage_cache=route_blockage_cache,
                )
                else MAX_PRIMARY_STAGING_TARGETS_BEFORE_STORAGE + MAX_STORAGE_STAGING_TARGETS
            )
            if primary_staging_limit == ROUTE_RELEASE_PRIMARY_STAGING_TARGETS:
                staging_targets = _prefer_storage_staging_targets(
                    staging_targets,
                    master=master,
                    state=state,
                )
            primary_staged_count = 0
            storage_staged_count = 0
            for target_track in staging_targets:
                if target_track in goal_targets:
                    continue
                is_fallback_staging = _is_fallback_staging_track(master, target_track)
                if (
                    not is_fallback_staging
                    and primary_staged_count >= primary_staging_limit
                ):
                    continue
                if is_fallback_staging and storage_staged_count >= MAX_STORAGE_STAGING_TARGETS:
                    continue
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
                    allow_same_track=True,
                    route_train_length_m=carried_route_length,
                )
                if move is None:
                    continue
                if (
                    target_track == state.loco_track_name
                    and len(drop_block) == len(state.loco_carry)
                    and not _is_goal_after_detach(
                        state=state,
                        move=move,
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                    )
                ):
                    continue
                if (
                    target_track == state.loco_track_name
                    and len(drop_block) == len(state.loco_carry)
                    and _same_track_detach_buries_unfinished_vehicle(
                        state=state,
                        move=move,
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                    )
                ):
                    continue
                detach_move = HookAction(
                    source_track=state.loco_track_name,
                    target_track=target_track,
                    vehicle_nos=list(drop_block),
                    path_tracks=move.path_tracks,
                    action_type="DETACH",
                )
                if not _empty_carry_detach_has_followup_attach(
                    detach_move=detach_move,
                    drop_block=drop_block,
                    state=state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                    master=master,
                    route_oracle=route_oracle,
                    followup_attach_cache=followup_attach_cache,
                    required=require_empty_carry_followup,
                ):
                    continue
                moves.append(detach_move)
                _record_move_debug_stats(
                    debug_stats,
                    move=detach_move,
                    is_staging=True,
                )
                if is_fallback_staging:
                    storage_staged_count += 1
                else:
                    primary_staged_count += 1
                if (
                    primary_staged_count >= primary_staging_limit
                    and storage_staged_count >= MAX_STORAGE_STAGING_TARGETS
                ):
                    break

    return _sort_moves_for_stable_search(_dedup_moves(moves))


def _empty_carry_detach_has_followup_attach(
    *,
    detach_move: HookAction,
    drop_block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    followup_attach_cache: dict[tuple, bool],
    required: bool = True,
) -> bool:
    if not required:
        return True
    if len(drop_block) < len(state.loco_carry):
        return True
    try:
        next_state = _apply_move(
            state=state,
            move=detach_move,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return False
    if _is_goal(plan_input, next_state):
        return True
    prior_state_key = _state_key(state, plan_input)
    cache_key = (prior_state_key, _state_key(next_state, plan_input))
    if cache_key not in followup_attach_cache:
        has_productive_followup = _has_productive_followup_attach(
            state=next_state,
            prior_state_key=prior_state_key,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
            route_oracle=route_oracle,
        )
        followup_attach_cache[cache_key] = has_productive_followup
    return followup_attach_cache[cache_key]


def _has_productive_followup_attach(
    *,
    state: ReplayState,
    prior_state_key: tuple,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle | None,
) -> bool:
    if state.loco_carry:
        return False
    carried_train_length_m = 0.0
    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        if route_oracle is not None:
            access_result = route_oracle.validate_loco_access(
                loco_track=state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=state.track_sequences,
                carried_train_length_m=carried_train_length_m,
                loco_node=state.loco_node,
            )
            if not access_result.is_valid:
                continue
        for prefix_size in range(len(seq), 0, -1):
            block = seq[:prefix_size]
            block_vehicles = [
                vehicle_by_no[vehicle_no]
                for vehicle_no in block
                if vehicle_no in vehicle_by_no
            ]
            if len(block_vehicles) != len(block):
                continue
            if validate_hook_vehicle_group(block_vehicles):
                continue
            if _violates_non_cun4bei_attach_close_door_rule(block, vehicle_by_no):
                continue
            followup_state = _apply_move(
                state=state,
                move=HookAction(
                    source_track=source_track,
                    target_track=source_track,
                    vehicle_nos=list(block),
                    path_tracks=[source_track],
                    action_type="ATTACH",
                ),
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            if _state_key(followup_state, plan_input) != prior_state_key:
                return True
    return False


def _combined_carry_has_legal_detach_after_attach(
    *,
    source_track: str,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    route_oracle: RouteOracle | None,
) -> bool:
    projected_state = state.model_copy(
        update={
            "track_sequences": {
                **state.track_sequences,
                source_track: list(state.track_sequences.get(source_track, []))[len(block):],
            },
            "loco_track_name": source_track,
            "loco_node": (
                route_oracle.order_end_node(source_track)
                if route_oracle is not None
                else state.loco_node
            ),
            "loco_carry": tuple(state.loco_carry) + tuple(block),
        }
    )
    carried_route_length = sum(
        length_by_vehicle.get(vehicle_no, 0.0)
        for vehicle_no in projected_state.loco_carry
    )
    for drop_block in iter_carried_tail_blocks(projected_state.loco_carry):
        drop_vehicles = [
            vehicle_by_no[vehicle_no]
            for vehicle_no in drop_block
            if vehicle_no in vehicle_by_no
        ]
        if not drop_vehicles or validate_hook_vehicle_group(drop_vehicles):
            continue
        block_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in drop_block
        )
        goal_targets: set[str] = set()
        for vehicle_no in drop_block:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            if vehicle.need_weigh and vehicle_no not in projected_state.weighed_vehicle_nos:
                goal_targets.add("机库")
            else:
                goal_targets.update(
                    goal_effective_allowed_tracks(
                        vehicle,
                        state=projected_state,
                        plan_input=plan_input,
                        route_oracle=route_oracle,
                    )
                )
        if _tail_detach_exposes_carried_vehicle(
            carry=projected_state.loco_carry,
            drop_block=drop_block,
            vehicle_by_no=vehicle_by_no,
            plan_input=plan_input,
            state=projected_state,
        ):
            goal_targets.add(projected_state.loco_track_name)
        staging_targets = _candidate_staging_targets(
            source_track=projected_state.loco_track_name,
            block=drop_block,
            state=projected_state,
            plan_input=plan_input,
            master=route_oracle.master if route_oracle is not None else None,
            vehicle_by_no=vehicle_by_no,
            goal_target_hints=tuple(sorted(goal_targets)),
            route_oracle=route_oracle,
            route_pressure_sort=False,
        )
        for target_track in [*sorted(goal_targets), *staging_targets]:
            if target_track == projected_state.loco_track_name:
                continue
            move = _build_candidate_move(
                source_track=projected_state.loco_track_name,
                target_track=target_track,
                block=drop_block,
                block_vehicles=drop_vehicles,
                block_length=block_length,
                state=projected_state,
                capacity_by_track=effective_capacity_by_track,
                length_by_vehicle=length_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
                route_oracle=route_oracle,
                allow_same_track=True,
                route_train_length_m=carried_route_length,
            )
            if move is None:
                continue
            if _detach_block_satisfies_target(
                block=list(move.vehicle_nos),
                target_track=move.target_track,
                state=projected_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            ) and _remaining_carry_can_move_after_detach(
                state=projected_state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
                length_by_vehicle=length_by_vehicle,
                effective_capacity_by_track=effective_capacity_by_track,
                route_oracle=route_oracle,
            ):
                return True
    return False


def _detach_block_satisfies_target(
    *,
    block: list[str],
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if vehicle.need_weigh and vehicle_no not in state.weighed_vehicle_nos:
            if target_track != "机库":
                return False
            continue
        if target_track not in vehicle.goal.allowed_target_tracks:
            return False
    return True


def _remaining_carry_can_move_after_detach(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    route_oracle: RouteOracle | None,
) -> bool:
    try:
        next_state = _apply_move(
            state=state,
            move=HookAction(
                source_track=move.source_track,
                target_track=move.target_track,
                vehicle_nos=move.vehicle_nos,
                path_tracks=move.path_tracks,
                action_type="DETACH",
            ),
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return False
    if not next_state.loco_carry:
        return True
    remaining_length = sum(
        length_by_vehicle.get(vehicle_no, 0.0)
        for vehicle_no in next_state.loco_carry
    )
    if route_oracle is None:
        return True
    source_effective_capacity = effective_capacity_by_track.get(next_state.loco_track_name)
    if (
        source_effective_capacity is not None
        and remaining_length + route_oracle.master.business_rules.loco_length_m
        > source_effective_capacity + 1e-9
    ):
        return False
    next_carry = list(next_state.loco_carry)
    next_carry_vehicles = [
        vehicle_by_no[vehicle_no]
        for vehicle_no in next_carry
        if vehicle_no in vehicle_by_no
    ]
    if len(next_carry_vehicles) != len(next_carry) or validate_hook_vehicle_group(
        next_carry_vehicles
    ):
        return False
    remaining_block_length = sum(
        length_by_vehicle.get(vehicle_no, 0.0)
        for vehicle_no in next_carry
    )
    for target_track in _candidate_targets(
        next_carry,
        plan_input,
        next_state,
        vehicle_by_no,
        route_oracle=route_oracle,
    ):
        if target_track == "机库" and all(
            vehicle.need_weigh and vehicle.vehicle_no not in next_state.weighed_vehicle_nos
            for vehicle in next_carry_vehicles
        ):
            return True
        if target_track == next_state.loco_track_name:
            continue
        candidate_move = _build_candidate_move(
            source_track=next_state.loco_track_name,
            target_track=target_track,
            block=next_carry,
            block_vehicles=next_carry_vehicles,
            block_length=remaining_block_length,
            state=next_state,
            capacity_by_track=effective_capacity_by_track,
            length_by_vehicle=length_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            plan_input=plan_input,
            route_oracle=route_oracle,
            allow_same_track=True,
            route_train_length_m=remaining_length,
        )
        if candidate_move is not None:
            return True
    return False


def _remaining_carry_exit_targets(
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle,
) -> set[str]:
    targets: set[str] = set()
    for vehicle_no in state.loco_carry:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        targets.update(
            goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
        )
    return targets


def _is_goal_after_detach(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    try:
        next_state = _apply_move(
            state=state,
            move=HookAction(
                source_track=move.source_track,
                target_track=move.target_track,
                vehicle_nos=move.vehicle_nos,
                path_tracks=move.path_tracks,
                action_type="DETACH",
            ),
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return False
    return all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and (
            (
                vehicle.goal.work_position_kind is not None
                and move.target_track in vehicle.goal.allowed_target_tracks
            )
            or goal_is_satisfied(
                vehicle,
                track_name=move.target_track,
                state=next_state,
                plan_input=plan_input,
            )
        )
        for vehicle_no in move.vehicle_nos
    )


def _same_track_detach_buries_unfinished_vehicle(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if move.source_track != move.target_track:
        return False
    for vehicle_no in state.track_sequences.get(move.target_track, []):
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        if not goal_is_satisfied(
            vehicle,
            track_name=move.target_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _real_hook_attach_frontier_prefix_sizes(
    *,
    seq: list[str],
    requested_prefix_sizes: set[int],
    state: ReplayState,
    vehicle_by_no: dict,
    plan_input: NormalizedPlanInput,
) -> set[int]:
    if not seq:
        return set()
    frontier: dict[int, int] = {}
    for prefix_size in range(1, len(seq) + 1):
        block = seq[:prefix_size]
        block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
        if validate_hook_vehicle_group(block_vehicles):
            continue
        if _violates_non_cun4bei_attach_close_door_rule(block, vehicle_by_no):
            continue
        if prefix_size in requested_prefix_sizes:
            frontier[prefix_size] = prefix_size
            continue
        detach_groups = _min_detach_groups(
            block,
            vehicle_by_no=vehicle_by_no,
            weighed_vehicle_nos=state.weighed_vehicle_nos,
        )
        best_prefix_for_group = frontier.get(detach_groups)
        if best_prefix_for_group is None or prefix_size > best_prefix_for_group:
            frontier[detach_groups] = prefix_size
    return (
        set(frontier.values())
        | set(requested_prefix_sizes)
        | _large_random_goal_chunk_prefix_sizes(seq, vehicle_by_no)
    )


def _large_random_goal_chunk_prefix_sizes(
    seq: list[str],
    vehicle_by_no: dict,
) -> set[int]:
    if len(seq) <= 10:
        return set()
    first_vehicle = vehicle_by_no.get(seq[0])
    if first_vehicle is None:
        return set()
    goal = first_vehicle.goal
    if goal.target_area_code is None or ":RANDOM" not in goal.target_area_code:
        return set()
    if goal.target_mode not in {"AREA", "SNAPSHOT"}:
        return set()
    same_goal_prefix = 0
    first_key = _goal_key(goal)
    for vehicle_no in seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or _goal_key(vehicle.goal) != first_key:
            break
        same_goal_prefix += 1
    if same_goal_prefix <= 10:
        return set()
    return {10}


def _tail_detach_exposes_carried_vehicle(
    *,
    carry: tuple[str, ...],
    drop_block: list[str],
    vehicle_by_no: dict,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> bool:
    if len(drop_block) >= len(carry):
        return False
    exposed_vehicle_no = carry[-len(drop_block) - 1]
    exposed_vehicle = vehicle_by_no.get(exposed_vehicle_no)
    if exposed_vehicle is None:
        return False
    if goal_is_satisfied(
        exposed_vehicle,
        track_name=state.loco_track_name,
        state=state,
        plan_input=plan_input,
    ):
        return False
    if exposed_vehicle.need_weigh and exposed_vehicle_no not in state.weighed_vehicle_nos:
        return True
    if exposed_vehicle.goal.target_mode == "SPOT":
        return True
    if (
        exposed_vehicle.is_close_door
        and "存4北" in exposed_vehicle.goal.allowed_target_tracks
        and sum(
            1
            for vehicle_no in drop_block
            if (
                (vehicle := vehicle_by_no.get(vehicle_no)) is not None
                and not vehicle.is_close_door
                and "存4北" in vehicle.goal.allowed_target_tracks
            )
        )
        >= 3
    ):
        return True
    exposed_targets = _effective_target_tracks_for_detach(
        exposed_vehicle_no,
        vehicle_by_no=vehicle_by_no,
        weighed_vehicle_nos=state.weighed_vehicle_nos,
    )
    drop_targets: set[str] = set()
    for vehicle_no in drop_block:
        drop_targets.update(
            _effective_target_tracks_for_detach(
                vehicle_no,
                vehicle_by_no=vehicle_by_no,
                weighed_vehicle_nos=state.weighed_vehicle_nos,
            )
        )
    return bool(exposed_targets and not exposed_targets.issubset(drop_targets))


def _is_carrying_active_route_blocker(
    *,
    source_track: str,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
    route_oracle: RouteOracle | None,
    route_blockage_cache: RouteBlockageCache | None = None,
) -> bool:
    if route_oracle is None or not block:
        return False
    block_set = set(block)
    if not all(
        _vehicle_goal_satisfied_on_track(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in block
        if vehicle_no in vehicle_by_no
    ):
        return False
    restored_sequences = {
        track_name: list(seq)
        for track_name, seq in _track_sequences_in_order(state)
    }
    restored_sequences[source_track] = list(block) + list(restored_sequences.get(source_track, []))
    restored_state = state.model_copy(
        update={
            "track_sequences": restored_sequences,
            "loco_carry": tuple(
                vehicle_no
                for vehicle_no in state.loco_carry
                if vehicle_no not in block_set
            ),
        }
    )
    return source_track in _route_blockage_plan_for_state(
        plan_input=plan_input,
        state=restored_state,
        route_oracle=route_oracle,
        cache=route_blockage_cache,
    ).facts_by_blocking_track


def _route_blockage_target_hints_for_source(
    *,
    source_track: str,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle | None,
    route_blockage_cache: RouteBlockageCache | None,
) -> set[str]:
    if route_oracle is None:
        return set()
    route_blockage_plan = _route_blockage_plan_for_state(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        cache=route_blockage_cache,
    )
    target_hints: set[str] = set()
    for fact in route_blockage_plan.facts_by_blocking_track.values():
        if source_track in fact.source_tracks:
            target_hints.update(fact.target_tracks)
    return target_hints


def _generate_capacity_eviction_moves(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    route_blockage_cache: RouteBlockageCache | None = None,
) -> list[HookAction]:
    """Generate staging moves for identity-goal vehicles at capacity-full target tracks.

    When pending single-target arrivals at track T exceed available slack (after
    accounting for non-identity vehicles that will naturally depart), identity-goal
    vehicles at T must be temporarily evicted to make room. Generates those moves.
    """
    current_track_by_vehicle: dict[str, str] = {
        vno: track
        for track, seq in _track_sequences_in_order(state)
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
    for track, seq in _track_sequences_in_order(state):
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
            route_blockage_cache=route_blockage_cache,
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
    route_blockage_cache: RouteBlockageCache | None = None,
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
    reserved_spot_codes = exact_spot_reservations(plan_input)
    current_track_by_vehicle: dict[str, str] = {
        vno: track
        for track, seq in _track_sequences_in_order(state)
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
        if (
            current_track is not None
            and goal_is_satisfied(
                vehicle,
                track_name=current_track,
                state=state,
                plan_input=plan_input,
            )
        ):
            continue
        allowed = vehicle.goal.allowed_target_tracks
        if len(allowed) == 0:
            continue
        if vehicle.goal.target_mode != "WORK_POSITION":
            any_feasible = False
            for target in allowed:
                result = allocate_spots_for_block(
                    vehicles=[vehicle],
                    target_track=target,
                    yard_mode=plan_input.yard_mode,
                    occupied_spot_assignments=state.spot_assignments,
                    reserved_spot_codes=reserved_spot_codes,
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
            front_vehicle = vehicle_by_no.get(front_vno)
            if front_vehicle is None:
                continue
            if goal_is_satisfied(
                front_vehicle,
                track_name=target,
                state=state,
                plan_input=plan_input,
            ):
                continue
            evict_requests[target] = max(evict_requests.get(target, 0), 1)
            break

    eviction_moves: list[HookAction] = []
    for track, prefix_size in sorted(evict_requests.items()):
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
            route_blockage_cache=route_blockage_cache,
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


def _sort_moves_for_stable_search(moves: list[HookAction]) -> list[HookAction]:
    return list(moves)


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


def _is_final_goal_detach(
    *,
    block: list[str],
    target_track: str,
    state: ReplayState,
    vehicle_by_no: dict,
    plan_input: NormalizedPlanInput,
) -> bool:
    if not state.loco_carry:
        return False
    vehicles = [vehicle_by_no.get(vehicle_no) for vehicle_no in block]
    if any(vehicle is None for vehicle in vehicles):
        return False
    existing_target_seq = list(state.track_sequences.get(target_track, []))
    projected_target_seq = list(block) + existing_target_seq
    next_spot_assignments = dict(state.spot_assignments)
    for vehicle_no in block:
        next_spot_assignments.pop(vehicle_no, None)
    projected_spot_assignments = realign_spots_for_track_order(
        vehicle_nos_in_order=projected_target_seq,
        vehicle_by_no=vehicle_by_no,
        target_track=target_track,
        yard_mode=plan_input.yard_mode,
        current_spot_assignments=next_spot_assignments,
        reserved_spot_codes=exact_spot_reservations(plan_input),
    )
    if projected_spot_assignments is None:
        return False
    projected_state = state.model_copy(
        update={
            "track_sequences": {
                **state.track_sequences,
                target_track: projected_target_seq,
            },
            "spot_assignments": projected_spot_assignments,
        }
    )
    return all(
        vehicle is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=projected_state,
            plan_input=plan_input,
        )
        for vehicle in vehicles
    )


def _candidate_targets(
    block: list[str],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
    route_oracle: RouteOracle | None = None,
) -> list[str]:
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    if any(
        vehicle_by_no[vehicle_no].need_weigh and vehicle_no not in state.weighed_vehicle_nos
        for vehicle_no in block
    ):
        # 机库 weighing is 单钩 — only single-vehicle blocks may be sent there.
        return ["机库"] if len(block) == 1 else []
    goal = vehicle_by_no[block[0]].goal
    lead_vehicle = vehicle_by_no[block[0]]
    targets = goal_effective_allowed_tracks(
        lead_vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )
    if goal.target_area_code == "大库:RANDOM":
        targets.sort(
            key=lambda track_name: (
                goal_track_preference_level(
                    lead_vehicle,
                    track_name,
                    state=state,
                    plan_input=plan_input,
                ) or 0,
                sum(
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
    route_pressure_sort: bool = True,
    route_blockage_cache: RouteBlockageCache | None = None,
    penalize_goal_corridor_reblock: bool = True,
) -> list[str]:
    if master is None:
        return []
    candidates: list[tuple[tuple[float, float, float, str], str]] = []
    for info in plan_input.track_info:
        if info.track_name == source_track:
            continue
        type_priority = _staging_track_priority_for_block(
            master=master,
            track_name=info.track_name,
            block=block,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if type_priority is None:
            continue
        if any(
            _staging_target_is_hard_goal_for_vehicle(info.track_name, vehicle_by_no[vehicle_no])
            for vehicle_no in block
        ):
            continue
        source_distance = _route_distance(route_oracle, source_track, info.track_name)
        if source_distance is None:
            continue
        combined_distance = source_distance
        follow_distances: list[float] = []
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
        occupancy = len(state.track_sequences.get(info.track_name, []))
        corridor_penalty = (
            _staging_target_reblocks_goal_corridor(
                source_track=source_track,
                target_track=info.track_name,
                goal_target_hints=goal_target_hints,
                route_oracle=route_oracle,
            )
            if penalize_goal_corridor_reblock
            else 0
        )
        candidates.append(
            (
                (
                    type_priority,
                    corridor_penalty,
                    occupancy,
                    combined_distance,
                    source_distance,
                    info.track_name,
                ),
                info.track_name,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [track_name for _, track_name in candidates]


def _prefer_storage_staging_targets(
    targets: list[str],
    *,
    master: MasterData | None,
    state: ReplayState,
) -> list[str]:
    def storage_rank(track_name: str) -> tuple[int, int, int]:
        track = master.tracks.get(track_name) if master is not None else None
        if track is None or track.track_type not in FALLBACK_STAGING_TRACK_TYPES:
            return (0, 0, 0)
        has_existing_work = 1 if state.track_sequences.get(track_name) else 0
        is_yard_storage = 1 if track_name.startswith(("存", "调", "洗")) else 0
        return (0, -has_existing_work, -is_yard_storage)

    return sorted(targets, key=storage_rank)


def _staging_target_reblocks_goal_corridor(
    *,
    source_track: str,
    target_track: str,
    goal_target_hints: tuple[str, ...],
    route_oracle: RouteOracle | None,
) -> int:
    if route_oracle is None:
        return 0
    for goal_track in goal_target_hints:
        if goal_track == target_track:
            continue
        path_tracks = route_oracle.resolve_path_tracks(source_track, goal_track)
        if path_tracks is None:
            continue
        if target_track in path_tracks[1:-1]:
            return 1
    return 0


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


def _staging_track_priority_for_block(
    *,
    master: MasterData | None,
    track_name: str,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
) -> int | None:
    if (
        is_depot_inner_track(track_name)
        and _block_has_only_depot_terminal_goals(
            block=block,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    ):
        return -1
    return _staging_track_priority(master, track_name)


def _block_has_only_depot_terminal_goals(
    *,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
) -> bool:
    if not block:
        return False
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        effective_targets = set(
            goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
            )
        )
        if not effective_targets or not all(
            is_depot_inner_track(target) for target in effective_targets
        ):
            return False
    return True


def _staging_target_is_hard_goal_for_vehicle(
    track_name: str,
    vehicle: NormalizedVehicle,
) -> bool:
    goal = vehicle.goal
    if track_name in goal.preferred_target_tracks:
        return True
    if goal.target_mode == "SNAPSHOT" and track_name in goal.fallback_target_tracks:
        return False
    return track_name in goal.allowed_target_tracks


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
    front_prefix_size = _front_blocker_prefix_size(
        source_track,
        seq,
        goal_by_vehicle,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    if front_prefix_size is not None:
        next_vehicle_no = seq[front_prefix_size]
        target_hints = _candidate_targets(
            [next_vehicle_no],
            plan_input,
            state,
            vehicle_by_no,
            route_oracle=None,
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
    plan_input: NormalizedPlanInput,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    blocking_goal_targets_by_source: dict[str, set[str]],
) -> bool:
    goal = goal_by_vehicle[block[0]]
    if goal.target_area_code != "大库:RANDOM":
        return False
    if not all(
        goal_is_preferred_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in block
    ):
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
    if _random_depot_preferred_rebalance_is_available(
        source_track=source_track,
        block=block,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return False
    return (
        _front_blocker_prefix_size(
            source_track,
            seq,
            goal_by_vehicle,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        != len(block)
    )


def _random_depot_preferred_rebalance_is_available(
    *,
    source_track: str,
    block: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
) -> bool:
    if not block:
        return False
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if vehicle.goal.target_area_code != "大库:RANDOM":
            return False
        if source_track not in vehicle.goal.fallback_target_tracks:
            return False
        if not vehicle.goal.preferred_target_tracks:
            return False
        if not _random_depot_vehicle_has_preferred_rebalance_target(
            vehicle,
            state=state,
            plan_input=plan_input,
        ):
            return False
    return True


def _random_depot_preferred_rebalance_prefix_sizes(
    *,
    source_track: str,
    seq: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
) -> set[int]:
    prefix_sizes: set[int] = set()
    for prefix_size in range(1, len(seq) + 1):
        block = seq[:prefix_size]
        block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
        if validate_hook_vehicle_group(block_vehicles):
            continue
        if not _random_depot_preferred_rebalance_is_available(
            source_track=source_track,
            block=block,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        ):
            break
        prefix_sizes.add(prefix_size)
    return prefix_sizes


def _random_depot_vehicle_has_preferred_rebalance_target(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    reserved_spot_codes = exact_spot_reservations(plan_input)
    for target_track in vehicle.goal.preferred_target_tracks:
        if allocate_spots_for_block(
            vehicles=[vehicle],
            target_track=target_track,
            yard_mode=plan_input.yard_mode,
            occupied_spot_assignments=state.spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        ) is not None:
            return True
    return False


def _front_blocker_prefix_size(
    source_track: str,
    seq: list[str],
    goal_by_vehicle: dict,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict,
) -> int | None:
    if len(seq) <= 1:
        return None
    first_goal = goal_by_vehicle[seq[0]]
    first_vehicle = vehicle_by_no.get(seq[0])
    if first_vehicle is None:
        return None
    if not goal_is_satisfied(
        first_vehicle,
        track_name=source_track,
        state=state,
        plan_input=plan_input,
    ):
        return None
    prefix_size = 1
    while (
        prefix_size < len(seq)
        and _goal_key(goal_by_vehicle[seq[prefix_size]]) == _goal_key(first_goal)
        and (
            vehicle := vehicle_by_no.get(seq[prefix_size])
        ) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
    ):
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
    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        for prefix_size in range(1, len(seq) + 1):
            block = seq[:prefix_size]
            if not _same_goal(block, goal_by_vehicle):
                continue
            candidate_targets = _candidate_targets(
                block,
                plan_input,
                state,
                vehicle_by_no,
                route_oracle=route_oracle,
            )
            for target_track in candidate_targets:
                if target_track == source_track:
                    continue
                path_tracks = route_oracle.resolve_path_tracks(
                    source_track,
                    target_track,
                )
                if path_tracks is None:
                    continue
                for track_code in path_tracks[1:-1]:
                    if not state.track_sequences.get(track_code):
                        continue
                    blocking_goal_targets_by_source.setdefault(track_code, set()).add(target_track)
    return blocking_goal_targets_by_source


def _collect_real_hook_access_blocker_attach_requests(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    route_oracle: RouteOracle | None,
    length_by_vehicle: dict[str, float] | None = None,
    effective_capacity_by_track: dict[str, float] | None = None,
    master: MasterData | None = None,
    route_blockage_cache: RouteBlockageCache | None = None,
) -> dict[str, set[int]]:
    if route_oracle is None or state.loco_carry:
        return {}
    if master is None:
        master = route_oracle.master
    if length_by_vehicle is None:
        length_by_vehicle = {
            vehicle.vehicle_no: vehicle.vehicle_length
            for vehicle in plan_input.vehicles
        }
    if effective_capacity_by_track is None:
        capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
        initial_occupation_by_track: dict[str, float] = {}
        for vehicle in plan_input.vehicles:
            initial_occupation_by_track[vehicle.current_track] = (
                initial_occupation_by_track.get(vehicle.current_track, 0.0)
                + vehicle.vehicle_length
            )
        effective_capacity_by_track = {
            name: max(cap, initial_occupation_by_track.get(name, 0.0))
            for name, cap in capacity_by_track.items()
        }
    requests: dict[str, set[int]] = {}
    route_blockage_plan = _route_blockage_plan_for_state(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        cache=route_blockage_cache,
    )
    for blocking_track, fact in sorted(route_blockage_plan.facts_by_blocking_track.items()):
        if not _route_blockage_fact_warrants_clear_request(
            blocking_track=blocking_track,
            source_tracks=frozenset(sorted(fact.source_tracks)),
            blockage_count=fact.blockage_count,
            state=state,
            route_oracle=route_oracle,
        ):
            continue
        blocker_seq = state.track_sequences.get(blocking_track, [])
        if not blocker_seq:
            continue
        blocker_prefix_sizes = _access_blocker_clear_prefix_sizes(
            source_track=blocking_track,
            seq=blocker_seq,
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            effective_capacity_by_track=effective_capacity_by_track,
            master=master,
            route_oracle=route_oracle,
            route_blockage_cache=route_blockage_cache,
            goal_target_hints=tuple(sorted(fact.target_tracks)),
        )
        if blocker_prefix_sizes:
            requests.setdefault(blocking_track, set()).update(blocker_prefix_sizes)

    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        for prefix_size in _real_hook_attach_frontier_prefix_sizes(
            seq=seq,
            requested_prefix_sizes=set(),
            state=state,
            vehicle_by_no=vehicle_by_no,
            plan_input=plan_input,
        ):
            block = seq[:prefix_size]
            if all(
                _vehicle_goal_satisfied_on_track(
                    vehicle_by_no[vehicle_no],
                    track_name=source_track,
                    state=state,
                    plan_input=plan_input,
                )
                for vehicle_no in block
            ):
                continue
            access_result = route_oracle.validate_loco_access(
                loco_track=state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=state.track_sequences,
                loco_node=state.loco_node,
            )
            if access_result.is_valid:
                continue
            for blocking_track in access_result.blocking_tracks:
                blocker_seq = state.track_sequences.get(blocking_track, [])
                if not blocker_seq:
                    continue
                blocker_prefix_sizes = _access_blocker_clear_prefix_sizes(
                    source_track=blocking_track,
                    seq=blocker_seq,
                    plan_input=plan_input,
                    state=state,
                    vehicle_by_no=vehicle_by_no,
                    length_by_vehicle=length_by_vehicle,
                    effective_capacity_by_track=effective_capacity_by_track,
                    master=master,
                    route_oracle=route_oracle,
                    route_blockage_cache=route_blockage_cache,
                    goal_target_hints=(blocking_track,),
                )
                if not blocker_prefix_sizes:
                    continue
                requests.setdefault(blocking_track, set()).update(blocker_prefix_sizes)
    return requests


def _route_blockage_fact_warrants_clear_request(
    *,
    blocking_track: str,
    source_tracks: frozenset[str],
    blockage_count: int,
    state: ReplayState,
    route_oracle: RouteOracle,
) -> bool:
    for source_track in sorted(source_tracks):
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=source_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
        )
        if not access_result.is_valid and blocking_track in access_result.blocking_tracks:
            return True
    return blockage_count > 1


def _access_blocker_clear_prefix_sizes(
    *,
    source_track: str,
    seq: list[str],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
    length_by_vehicle: dict[str, float],
    effective_capacity_by_track: dict[str, float],
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    goal_target_hints: tuple[str, ...],
    route_blockage_cache: RouteBlockageCache | None = None,
) -> set[int]:
    """Pick the largest hookable prefix that can actually leave the blocker track.

    For front-goal blockers only the leading satisfied block needs to move.
    For route access blockers, any remaining vehicle on the intermediate
    track still blocks the locomotive. Prefer the largest prefix that has at
    least one legal non-source staging detach after ATTACH; if the whole block
    is too long or too large to stage, clear it in repeated feasible chunks.
    """
    prefix_sizes = _descending_valid_staging_prefix_sizes(seq, vehicle_by_no)
    if not prefix_sizes:
        return set()
    for prefix_size in prefix_sizes:
        block = seq[:prefix_size]
        block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in block]
        block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in block)
        next_track_sequences = dict(state.track_sequences)
        next_track_sequences[source_track] = list(seq[prefix_size:])
        next_spot_assignments = dict(state.spot_assignments)
        for vehicle_no in block:
            next_spot_assignments.pop(vehicle_no, None)
        after_attach = ReplayState(
            track_sequences=next_track_sequences,
            loco_track_name=source_track,
            loco_node=(
                route_oracle.order_end_node(source_track)
                if route_oracle is not None
                else state.loco_node
            ),
            weighed_vehicle_nos=set(state.weighed_vehicle_nos),
            spot_assignments=next_spot_assignments,
            loco_carry=state.loco_carry + tuple(block),
        )
        carried_route_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in after_attach.loco_carry
        )
        for staging_target in _candidate_staging_targets(
            source_track=source_track,
            block=block,
            state=after_attach,
            plan_input=plan_input,
            master=master,
            vehicle_by_no=vehicle_by_no,
            goal_target_hints=goal_target_hints,
            route_oracle=route_oracle,
            route_pressure_sort=False,
            route_blockage_cache=route_blockage_cache,
        ):
            move = _build_candidate_move(
                source_track=source_track,
                target_track=staging_target,
                block=block,
                block_vehicles=block_vehicles,
                block_length=block_length,
                state=after_attach,
                capacity_by_track=effective_capacity_by_track,
                length_by_vehicle=length_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                plan_input=plan_input,
                route_oracle=route_oracle,
                route_train_length_m=carried_route_length,
            )
            if move is not None:
                return {prefix_size}
    return {prefix_sizes[0]}


def _has_accessible_real_hook_attach_frontier(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    goal_by_vehicle: dict,
    vehicle_by_no: dict,
    route_oracle: RouteOracle,
) -> bool:
    """Return whether the empty locomotive has a normal legal ATTACH available.

    Access-blocker clearing is a dead-end recovery move. If ordinary reachable
    attach work exists, generating extra satisfied-blocker requests too early
    tends to disturb already-good blocks and creates staging churn.
    """
    for source_track, seq in _track_sequences_in_order(state):
        if not seq:
            continue
        requested_prefix_sizes: set[int] = set()
        front_prefix_size = _front_blocker_prefix_size(
            source_track,
            seq,
            goal_by_vehicle,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if front_prefix_size is not None:
            requested_prefix_sizes.add(front_prefix_size)
        for prefix_size in _real_hook_attach_frontier_prefix_sizes(
            seq=seq,
            requested_prefix_sizes=requested_prefix_sizes,
            state=state,
            vehicle_by_no=vehicle_by_no,
            plan_input=plan_input,
        ):
            block = seq[:prefix_size]
            block_goal_satisfied = all(
                _vehicle_goal_satisfied_on_track(
                    vehicle_by_no[vehicle_no],
                    track_name=source_track,
                    state=state,
                    plan_input=plan_input,
                )
                for vehicle_no in block
            )
            if block_goal_satisfied and prefix_size not in requested_prefix_sizes:
                continue
            access_result = route_oracle.validate_loco_access(
                loco_track=state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=state.track_sequences,
                loco_node=state.loco_node,
            )
            if access_result.is_valid:
                return True
    return False


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
    allow_same_track: bool = False,
    route_train_length_m: float | None = None,
) -> HookAction | None:
    if target_track == source_track and not allow_same_track:
        return None
    if _violates_close_door_hook_rule(block, target_track, vehicle_by_no, state):
        return None
    if not _is_final_goal_detach(
        block=block,
        target_track=target_track,
        state=state,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
    ) and not _fits_capacity(
        target_track,
        block_length,
        state,
        capacity_by_track,
        length_by_vehicle,
    ):
        return None
    if _violates_work_position_preview(block, target_track, state, vehicle_by_no):
        return None
    if allocate_spots_for_block(
        vehicles=block_vehicles,
        target_track=target_track,
        yard_mode=plan_input.yard_mode,
        occupied_spot_assignments=state.spot_assignments,
        reserved_spot_codes=exact_spot_reservations(plan_input),
    ) is None:
        return None
    path_tracks = [source_track, target_track]
    if route_oracle is not None:
        source_node = (
            state.loco_node
            if source_track == state.loco_track_name
            and (
                move_source_remainder_count := _remaining_source_vehicle_count_after_candidate(
                    source_track=source_track,
                    block=block,
                    state=state,
                )
            )
            > 0
            else None
        )
        target_node = (
            route_oracle.order_end_node(target_track)
            if target_track != source_track
            else None
        )
        resolved_path_tracks = route_oracle.resolve_clear_path_tracks(
            source_track,
            target_track,
            occupied_track_sequences=state.track_sequences,
            source_node=source_node,
            target_node=target_node,
        )
        if resolved_path_tracks is None:
            return None
        path_tracks = resolved_path_tracks
        resolved_route = route_oracle.resolve_route_for_path_tracks(
            path_tracks,
            source_node=source_node,
            target_node=target_node,
        )
        if resolved_route is None:
            return None
        route_result = route_oracle.validate_path(
            source_track=source_track,
            target_track=target_track,
            path_tracks=path_tracks,
            train_length_m=route_train_length_m if route_train_length_m is not None else block_length,
            occupied_track_sequences=state.track_sequences,
            expected_path_tracks=path_tracks,
            route=resolved_route,
            source_node=source_node,
            target_node=target_node,
        )
        if not route_result.is_valid:
            return None
    return HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(block),
        path_tracks=path_tracks,
        action_type="PUT",
    )


def _remaining_source_vehicle_count_after_candidate(
    *,
    source_track: str,
    block: list[str] | tuple[str, ...],
    state: ReplayState,
) -> int:
    source_seq = state.track_sequences.get(source_track, [])
    block_list = list(block)
    if block_list and source_seq[: len(block_list)] == block_list:
        return max(0, len(source_seq) - len(block_list))
    return len(source_seq)


def _violates_close_door_hook_rule(
    block: list[str],
    target_track: str,
    vehicle_by_no: dict,
    state: ReplayState,
) -> bool:
    if target_track == "存4北":
        # PREPEND model: a close-door vehicle in block lands at index 0 and only
        # reaches position ≥ 4 (required by goal) if ≥ 3 OTHER vehicles are
        # placed on 存4北 AFTER it.  "Pending" = not in this block, not currently
        # on 存4北, goal includes 存4北.
        cd_in_block = any(
            (v := vehicle_by_no.get(vno)) is not None and v.is_close_door
            for vno in block
        )
        if cd_in_block:
            block_set = set(block)
            current_4bei = set(state.track_sequences.get("存4北", []))
            pending = sum(
                1
                for vno, v in vehicle_by_no.items()
                if vno not in block_set
                and vno not in current_4bei
                and v is not None
                and "存4北" in getattr(v.goal, "allowed_target_tracks", [])
            )
            if pending < 3:
                return True  # not enough vehicles to push CD to position ≥ 4
        return False
    if len(block) <= 10:
        return False
    first_vehicle = vehicle_by_no[block[0]]
    return bool(first_vehicle.is_close_door)


def _violates_non_cun4bei_attach_close_door_rule(
    block: list[str],
    vehicle_by_no: dict,
) -> bool:
    if len(block) <= 10:
        return False
    first_vehicle = vehicle_by_no.get(block[0])
    return bool(first_vehicle is not None and first_vehicle.is_close_door)


def _violates_work_position_preview(
    block: list[str],
    target_track: str,
    state: ReplayState,
    vehicle_by_no: dict,
) -> bool:
    if not is_work_position_track(target_track):
        return False
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=list(block),
        existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return True
    preview_state = state.model_copy(
        update={
            "track_sequences": {
                **state.track_sequences,
                target_track: list(block)
                + list(state.track_sequences.get(target_track, [])),
            }
        }
    )
    return bool(
        work_slot_violations_by_vehicle(
            vehicles=[vehicle for vehicle in vehicle_by_no.values() if vehicle is not None],
            state=preview_state,
        )
    )
