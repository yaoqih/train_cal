from __future__ import annotations

from fzed_shunting.domain.depot_spots import (
    allocate_spots_for_block,
    exact_spot_reservations,
    spot_candidates_for_vehicle,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.verify.replay import ReplayState


WORK_AREA_CODES = {"调棚:WORK", "调棚:PRE_REPAIR", "洗南:WORK", "油:WORK", "抛:WORK"}


def goal_can_use_fallback_now(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    fallback_tracks = list(vehicle.goal.fallback_target_tracks)
    if not fallback_tracks:
        return False
    preferred_tracks = list(vehicle.goal.preferred_target_tracks)
    if not preferred_tracks:
        return True
    occupied = dict(state.spot_assignments)
    for track in preferred_tracks:
        if allocate_spots_for_block(
            vehicles=[vehicle],
            target_track=track,
            yard_mode=plan_input.yard_mode,
            occupied_spot_assignments=occupied,
            reserved_spot_codes=exact_spot_reservations(plan_input),
        ) is not None:
            return False
    return True


def goal_effective_allowed_tracks(
    vehicle: NormalizedVehicle,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[str]:
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return ["机库"]
    preferred = list(vehicle.goal.preferred_target_tracks)
    fallback = list(vehicle.goal.fallback_target_tracks)
    if not preferred and not fallback:
        return list(vehicle.goal.allowed_target_tracks)
    if goal_can_use_fallback_now(vehicle, state=state, plan_input=plan_input):
        return preferred + fallback
    return preferred


def goal_track_preference_level(
    vehicle: NormalizedVehicle,
    track_name: str,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput | None = None,
) -> int | None:
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return 0 if track_name == "机库" else None
    if track_name in vehicle.goal.preferred_target_tracks:
        return 0
    if (
        track_name in vehicle.goal.fallback_target_tracks
        and (
            plan_input is None
            or goal_can_use_fallback_now(vehicle, state=state, plan_input=plan_input)
        )
    ):
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
) -> bool:
    resolved_yard_mode = plan_input.yard_mode if plan_input is not None else (yard_mode or "NORMAL")
    if goal_track_preference_level(vehicle, track_name, state=state, plan_input=plan_input) is None:
        return False
    if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return False
    if vehicle.goal.target_mode == "SPOT":
        return state.spot_assignments.get(vehicle.vehicle_no) == vehicle.goal.target_spot_code
    if vehicle.goal.target_area_code == "大库:RANDOM":
        assigned_spot = state.spot_assignments.get(vehicle.vehicle_no)
        return assigned_spot in spot_candidates_for_vehicle(
            vehicle,
            track_name,
            resolved_yard_mode,
        )
    if vehicle.goal.target_area_code in WORK_AREA_CODES:
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
) -> bool:
    if not goal_is_satisfied(vehicle, track_name=track_name, state=state, plan_input=plan_input):
        return False
    level = goal_track_preference_level(vehicle, track_name, state=state, plan_input=plan_input)
    return level == 0
