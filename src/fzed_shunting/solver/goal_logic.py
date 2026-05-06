from __future__ import annotations

from functools import lru_cache

from fzed_shunting.domain.depot_spots import (
    allocate_spots_for_block,
    exact_spot_reservations,
    is_depot_inner_track,
    spot_candidates_for_vehicle,
)
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import (
    work_position_satisfied,
    work_slot_violations_by_vehicle,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.verify.replay import ReplayState

_PLAN_INPUT_REGISTRY: dict[int, NormalizedPlanInput] = {}
_ROUTE_ORACLE_REGISTRY: dict[int, RouteOracle] = {}
_STATE_GOAL_CACHE_KEYS: dict[int, tuple] = {}


def goal_can_use_fallback_now(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None = None,
) -> bool:
    cache_key = _goal_cache_key(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )
    if cache_key is not None:
        return _cached_goal_can_use_fallback_now(cache_key)
    return _goal_can_use_fallback_now_uncached(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )


def _goal_can_use_fallback_now_uncached(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None = None,
) -> bool:
    fallback_tracks = list(vehicle.goal.fallback_target_tracks)
    if not fallback_tracks:
        return False
    preferred_tracks = list(vehicle.goal.preferred_target_tracks)
    if not preferred_tracks:
        return True
    if _preferred_tracks_lack_group_spot_capacity(
        vehicle,
        preferred_tracks=preferred_tracks,
        state=state,
        plan_input=plan_input,
    ):
        return True
    occupied = dict(state.spot_assignments)
    for track in preferred_tracks:
        if not _preferred_track_is_currently_usable(
            vehicle,
            track_name=track,
            state=state,
            plan_input=plan_input,
            route_oracle=route_oracle,
        ):
            continue
        if allocate_spots_for_block(
            vehicles=[vehicle],
            target_track=track,
            yard_mode=plan_input.yard_mode,
            occupied_spot_assignments=occupied,
            reserved_spot_codes=exact_spot_reservations(plan_input),
        ) is not None:
            return False
    return True


@lru_cache(maxsize=200_000)
def _cached_goal_can_use_fallback_now(cache_key: tuple) -> bool:
    plan_input = _PLAN_INPUT_REGISTRY[cache_key[0]]
    route_oracle = _ROUTE_ORACLE_REGISTRY.get(cache_key[1])
    vehicle_no = cache_key[2]
    state = _state_from_goal_cache_key(cache_key[3])
    vehicle = next(item for item in plan_input.vehicles if item.vehicle_no == vehicle_no)
    return _goal_can_use_fallback_now_uncached(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )


def _goal_cache_key(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput | None,
    route_oracle: RouteOracle | None,
) -> tuple | None:
    if plan_input is None:
        return None
    plan_input_id = id(plan_input)
    _PLAN_INPUT_REGISTRY[plan_input_id] = plan_input
    route_oracle_id = id(route_oracle) if route_oracle is not None else 0
    if route_oracle is not None:
        _ROUTE_ORACLE_REGISTRY[route_oracle_id] = route_oracle
    return (
        plan_input_id,
        route_oracle_id,
        vehicle.vehicle_no,
        _state_goal_cache_key(state),
    )


def _state_goal_cache_key(state: ReplayState) -> tuple:
    state_id = id(state)
    cached = _STATE_GOAL_CACHE_KEYS.get(state_id)
    if cached is not None:
        return cached
    cache_key = (
        tuple(
            (track, tuple(seq))
            for track, seq in sorted(state.track_sequences.items())
            if seq
        ),
        state.loco_track_name,
        state.loco_node,
        tuple(sorted(state.weighed_vehicle_nos)),
        tuple(sorted(state.spot_assignments.items())),
        state.loco_carry,
    )
    _STATE_GOAL_CACHE_KEYS[state_id] = cache_key
    if len(_STATE_GOAL_CACHE_KEYS) > 500_000:
        _STATE_GOAL_CACHE_KEYS.clear()
    return cache_key


def _state_from_goal_cache_key(state_key: tuple) -> ReplayState:
    track_items, loco_track_name, loco_node, weighed_vehicle_nos, spot_items, loco_carry = state_key
    return ReplayState(
        track_sequences={track: list(seq) for track, seq in track_items},
        loco_track_name=loco_track_name,
        loco_node=loco_node,
        weighed_vehicle_nos=set(weighed_vehicle_nos),
        spot_assignments=dict(spot_items),
        loco_carry=tuple(loco_carry),
    )


def _preferred_tracks_lack_group_spot_capacity(
    vehicle: NormalizedVehicle,
    *,
    preferred_tracks: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if vehicle.goal.target_area_code != "大库:RANDOM":
        return False
    if not all(is_depot_inner_track(track) for track in preferred_tracks):
        return False
    reserved = exact_spot_reservations(plan_input)
    free_spots = _free_spots_on_tracks(
        preferred_tracks,
        state=state,
        plan_input=plan_input,
        reserved_spot_codes=reserved,
    )
    unfinished_preferred_demand = 0
    track_by_vehicle = _current_track_by_vehicle(state)
    for candidate in plan_input.vehicles:
        if candidate.goal.target_area_code != vehicle.goal.target_area_code:
            continue
        if list(candidate.goal.preferred_target_tracks) != preferred_tracks:
            continue
        current_track = track_by_vehicle.get(candidate.vehicle_no)
        if current_track is not None and _has_valid_preferred_depot_spot(
            candidate,
            current_track=current_track,
            preferred_tracks=preferred_tracks,
            state=state,
            plan_input=plan_input,
            reserved_spot_codes=reserved,
        ):
            continue
        unfinished_preferred_demand += 1
    return unfinished_preferred_demand > free_spots


def _has_valid_preferred_depot_spot(
    vehicle: NormalizedVehicle,
    *,
    current_track: str,
    preferred_tracks: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    reserved_spot_codes: set[str] | frozenset[str],
) -> bool:
    if current_track not in preferred_tracks:
        return False
    assigned_spot = state.spot_assignments.get(vehicle.vehicle_no)
    if assigned_spot is None:
        return False
    return assigned_spot in spot_candidates_for_vehicle(
        vehicle,
        current_track,
        plan_input.yard_mode,
        reserved_spot_codes=reserved_spot_codes,
    )


def _free_spots_on_tracks(
    tracks: list[str],
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    reserved_spot_codes: set[str] | frozenset[str],
) -> int:
    occupied_spots = set(state.spot_assignments.values())
    count = 0
    for track in tracks:
        for spot_code in _track_spots_for_random_depot(track, plan_input):
            if spot_code in occupied_spots or spot_code in reserved_spot_codes:
                continue
            count += 1
    return count


def _track_spots_for_random_depot(track: str, plan_input: NormalizedPlanInput) -> list[str]:
    probe = next(
        (
            vehicle
            for vehicle in plan_input.vehicles
            if vehicle.goal.target_area_code == "大库:RANDOM"
        ),
        None,
    )
    if probe is None:
        return []
    return spot_candidates_for_vehicle(
        probe,
        track,
        plan_input.yard_mode,
        reserved_spot_codes=frozenset(),
    )


def _preferred_track_is_currently_usable(
    vehicle: NormalizedVehicle,
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
) -> bool:
    if not _track_has_length_slack_for_vehicle(
        vehicle,
        track_name=track_name,
        state=state,
        plan_input=plan_input,
    ):
        return False
    if route_oracle is None:
        return True
    source_track = _current_track_for_vehicle(state, vehicle.vehicle_no)
    if source_track is None:
        source_track = state.loco_track_name if vehicle.vehicle_no in state.loco_carry else None
    if source_track is None or source_track == track_name:
        return True
    source_node = state.loco_node if source_track == state.loco_track_name else None
    target_node = route_oracle.order_end_node(track_name)
    return (
        route_oracle.resolve_clear_path_tracks(
            source_track,
            track_name,
            occupied_track_sequences=state.track_sequences,
            source_node=source_node,
            target_node=target_node,
        )
        is not None
    )


def _track_has_length_slack_for_vehicle(
    vehicle: NormalizedVehicle,
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    current_track_by_vehicle: dict[str, str] = {}
    for current_track, seq in state.track_sequences.items():
        for vehicle_no in seq:
            current_track_by_vehicle[vehicle_no] = current_track
    if current_track_by_vehicle.get(vehicle.vehicle_no) == track_name:
        return True
    capacity_by_track = {
        info.track_name: info.track_distance for info in plan_input.track_info
    }
    capacity = capacity_by_track.get(track_name)
    if capacity is None:
        return False
    initial_occupation = sum(
        item.vehicle_length
        for item in plan_input.vehicles
        if item.current_track == track_name
    )
    effective_capacity = max(capacity, initial_occupation)
    length_by_vehicle = {
        item.vehicle_no: item.vehicle_length for item in plan_input.vehicles
    }
    current_length = sum(
        length_by_vehicle.get(vehicle_no, 0.0)
        for vehicle_no in state.track_sequences.get(track_name, [])
    )
    return current_length + vehicle.vehicle_length <= effective_capacity + 1e-9


def _current_track_for_vehicle(state: ReplayState, vehicle_no: str) -> str | None:
    for track_name, seq in state.track_sequences.items():
        if vehicle_no in seq:
            return track_name
    return None


def _current_track_by_vehicle(state: ReplayState) -> dict[str, str]:
    return {
        vehicle_no: track_name
        for track_name, seq in state.track_sequences.items()
        for vehicle_no in seq
    }


def goal_effective_allowed_tracks(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None = None,
) -> list[str]:
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return ["机库"]
    preferred = list(vehicle.goal.preferred_target_tracks)
    fallback = list(vehicle.goal.fallback_target_tracks)
    if not preferred and not fallback:
        return list(vehicle.goal.allowed_target_tracks)
    if goal_can_use_fallback_now(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    ):
        return preferred + fallback
    return preferred


def goal_track_preference_level(
    vehicle: NormalizedVehicle,
    track_name: str,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput | None = None,
    route_oracle: RouteOracle | None = None,
) -> int | None:
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return 0 if track_name == "机库" else None
    if vehicle.goal.target_mode == "SNAPSHOT" and track_name in vehicle.goal.allowed_target_tracks:
        return 0 if track_name in vehicle.goal.preferred_target_tracks else 1
    if track_name in vehicle.goal.preferred_target_tracks:
        return 0
    if track_name in vehicle.goal.fallback_target_tracks:
        return 1
    if track_name in vehicle.goal.allowed_target_tracks and not vehicle.goal.preferred_target_tracks and not vehicle.goal.fallback_target_tracks:
        return 0
    return None


def goal_is_satisfied(
    vehicle: NormalizedVehicle,
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput | None = None,
    yard_mode: str | None = None,
    route_oracle: RouteOracle | None = None,
) -> bool:
    resolved_yard_mode = plan_input.yard_mode if plan_input is not None else (yard_mode or "NORMAL")
    if goal_track_preference_level(
        vehicle,
        track_name,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    ) is None:
        return False
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return False
    if vehicle.goal.work_position_kind is not None:
        if not work_position_satisfied(vehicle, track_name=track_name, state=state):
            return False
        if (
            plan_input is not None
            and vehicle.goal.work_position_kind == "EXACT_WORK_SLOT"
        ):
            return vehicle.vehicle_no not in work_slot_violations_by_vehicle(
                vehicles=plan_input.vehicles,
                state=state,
            )
        return True
    if vehicle.goal.target_mode == "SPOT":
        return state.spot_assignments.get(vehicle.vehicle_no) == vehicle.goal.target_spot_code
    if vehicle.goal.target_area_code == "大库:RANDOM":
        assigned_spot = state.spot_assignments.get(vehicle.vehicle_no)
        return assigned_spot in spot_candidates_for_vehicle(
            vehicle,
            track_name,
            resolved_yard_mode,
        )
    if vehicle.is_close_door and track_name == "存4北":
        final_seq = state.track_sequences.get("存4北", [])
        return vehicle.vehicle_no in final_seq and final_seq.index(vehicle.vehicle_no) >= 3
    return True


def goal_is_preferred_satisfied(
    vehicle: NormalizedVehicle,
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None = None,
) -> bool:
    if not goal_is_satisfied(
        vehicle,
        track_name=track_name,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    ):
        return False
    level = goal_track_preference_level(
        vehicle,
        track_name,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )
    return level == 0
