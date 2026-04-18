from __future__ import annotations

from pydantic import BaseModel, Field

from fzed_shunting.domain.depot_spots import spot_candidates_for_vehicle
from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.verify.replay import build_initial_state, replay_plan


class PlanVerificationReport(BaseModel):
    is_valid: bool
    global_errors: list[str] = Field(default_factory=list)
    hook_reports: list["HookVerificationReport"] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HookVerificationReport(BaseModel):
    hook_no: int
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    source_track: str
    target_track: str
    vehicle_nos: list[str] = Field(default_factory=list)
    path_tracks: list[str] = Field(default_factory=list)
    blocking_tracks: list[str] = Field(default_factory=list)
    route_length_m: float | None = None
    branch_codes: list[str] = Field(default_factory=list)
    reverse_branch_codes: list[str] = Field(default_factory=list)
    required_reverse_clearance_m: float | None = None


def verify_plan(
    master: MasterData,
    plan_input: NormalizedPlanInput,
    hook_plan: list[dict],
    initial_state_override=None,
) -> PlanVerificationReport:
    global_errors: list[str] = []
    hook_reports: list[HookVerificationReport] = []
    route_oracle = RouteOracle(master)
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    try:
        initial = initial_state_override or build_initial_state(plan_input)
        replay = replay_plan(initial, hook_plan, plan_input=plan_input)
    except Exception as exc:  # noqa: BLE001
        return PlanVerificationReport(
            is_valid=False,
            global_errors=[str(exc)],
            errors=[str(exc)],
        )

    final_state = replay.final_state
    for vehicle in plan_input.vehicles:
        final_track = _locate_vehicle(final_state.track_sequences, vehicle.vehicle_no)
        if vehicle.need_weigh and vehicle.vehicle_no not in final_state.weighed_vehicle_nos:
            global_errors.append(f"Vehicle {vehicle.vehicle_no} requires weigh before final track")
            continue
        if vehicle.goal.target_mode == "SPOT":
            assigned_spot = final_state.spot_assignments.get(vehicle.vehicle_no)
            if assigned_spot != vehicle.goal.target_spot_code:
                global_errors.append(
                    f"Vehicle {vehicle.vehicle_no} final spot {assigned_spot} "
                    f"does not match target spot {vehicle.goal.target_spot_code}"
                )
                continue
        if vehicle.goal.target_area_code == "大库:RANDOM" and final_track in vehicle.goal.allowed_target_tracks:
            assigned_spot = final_state.spot_assignments.get(vehicle.vehicle_no)
            if assigned_spot is None:
                global_errors.append(f"Vehicle {vehicle.vehicle_no} missing depot spot assignment")
                continue
            if assigned_spot not in spot_candidates_for_vehicle(vehicle, final_track, plan_input.yard_mode):
                global_errors.append(
                    f"Vehicle {vehicle.vehicle_no} final spot {assigned_spot} "
                    f"is not compatible with target track {final_track}"
                )
                continue
        if vehicle.goal.target_area_code in {"调棚:WORK", "调棚:PRE_REPAIR", "洗南:WORK", "油:WORK", "抛:WORK"}:
            assigned_spot = final_state.spot_assignments.get(vehicle.vehicle_no)
            if assigned_spot is None:
                global_errors.append(f"Vehicle {vehicle.vehicle_no} missing work-area spot assignment")
                continue
            if assigned_spot not in spot_candidates_for_vehicle(vehicle, final_track, plan_input.yard_mode):
                global_errors.append(
                    f"Vehicle {vehicle.vehicle_no} final spot {assigned_spot} "
                    f"is not compatible with work area {vehicle.goal.target_area_code}"
                )
                continue
        if vehicle.is_close_door and final_track == "存4北":
            final_seq = final_state.track_sequences.get("存4北", [])
            pos = final_seq.index(vehicle.vehicle_no) + 1
            if pos <= 3:
                global_errors.append(
                    f"Close-door vehicle {vehicle.vehicle_no} cannot be placed in top-3 positions of 存4北"
                )
                continue
        if final_track not in vehicle.goal.allowed_target_tracks:
            global_errors.append(
                f"Vehicle {vehicle.vehicle_no} final track {final_track} "
                f"not in allowed final tracks {vehicle.goal.allowed_target_tracks}"
            )
            continue
        if final_track in master.tracks:
            track = master.tracks[final_track]
            if not track.allow_parking:
                global_errors.append(f"Vehicle {vehicle.vehicle_no} parked on running track {final_track}")

    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0) + vehicle.vehicle_length
        )
    for track_code, seq in final_state.track_sequences.items():
        if track_code not in capacity_by_track:
            continue
        occupied = sum(length_by_vehicle[vehicle_no] for vehicle_no in seq)
        declared_cap = capacity_by_track[track_code]
        effective_cap = max(declared_cap, initial_occupation_by_track.get(track_code, 0.0))
        if occupied > effective_cap + 1e-9:
            global_errors.append(
                f"Capacity overflow on track {track_code}: occupied {occupied} > {declared_cap}"
            )

    for hook in hook_plan:
        hook_no = hook.get("hookNo")
        pre_state = replay.snapshots[hook_no - 1] if isinstance(hook_no, int) and hook_no > 0 else initial
        hook_errors: list[str] = []
        route_result = None
        path_tracks = hook.get("pathTracks", [])
        if not path_tracks:
            hook_errors.append("must include pathTracks")
        else:
            if path_tracks[0] != hook["sourceTrack"] or path_tracks[-1] != hook["targetTrack"]:
                hook_errors.append(
                    f"path must start at {hook['sourceTrack']} "
                    f"and end at {hook['targetTrack']}"
                )
            else:
                unknown_vehicle_nos = [
                    vehicle_no
                    for vehicle_no in hook["vehicleNos"]
                    if vehicle_no not in length_by_vehicle
                ]
                if unknown_vehicle_nos:
                    hook_errors.append(
                        f"contains unknown vehicles: {unknown_vehicle_nos}"
                    )
                else:
                    hook_vehicles = [
                        vehicle
                        for vehicle in plan_input.vehicles
                        if vehicle.vehicle_no in set(hook["vehicleNos"])
                    ]
                    hook_errors.extend(validate_hook_vehicle_group(hook_vehicles))
                    route_result = route_oracle.validate_path(
                        source_track=hook["sourceTrack"],
                        target_track=hook["targetTrack"],
                        path_tracks=path_tracks,
                        train_length_m=sum(
                            length_by_vehicle[vehicle_no] for vehicle_no in hook["vehicleNos"]
                        ),
                        occupied_track_sequences=pre_state.track_sequences,
                    )
                    hook_errors.extend(route_result.errors)
        if hook["targetTrack"] != "存4北" and len(hook["vehicleNos"]) > 10:
            first_vehicle = hook["vehicleNos"][0]
            vehicle = next(
                (item for item in plan_input.vehicles if item.vehicle_no == first_vehicle),
                None,
            )
            if vehicle and vehicle.is_close_door:
                hook_errors.append(
                    f"Close-door vehicle {first_vehicle} cannot be first position when hook size > 10"
                )
        hook_reports.append(
            HookVerificationReport(
                hook_no=hook_no if isinstance(hook_no, int) else 0,
                is_valid=not hook_errors,
                errors=hook_errors,
                source_track=hook["sourceTrack"],
                target_track=hook["targetTrack"],
                vehicle_nos=list(hook["vehicleNos"]),
                path_tracks=list(hook.get("pathTracks", [])),
                blocking_tracks=list(route_result.blocking_tracks) if route_result else [],
                route_length_m=route_result.total_length_m if route_result else None,
                branch_codes=list(route_result.branch_codes) if route_result else [],
                reverse_branch_codes=list(route_result.reverse_branch_codes) if route_result else [],
                required_reverse_clearance_m=(
                    route_result.required_reverse_clearance_m if route_result else None
                ),
            )
        )

    errors = list(global_errors)
    for hook_report in hook_reports:
        errors.extend(f"Hook {hook_report.hook_no} {error}" for error in hook_report.errors)

    return PlanVerificationReport(
        is_valid=not errors,
        global_errors=global_errors,
        hook_reports=hook_reports,
        errors=errors,
    )


def _locate_vehicle(track_sequences: dict[str, list[str]], vehicle_no: str) -> str:
    for track, seq in track_sequences.items():
        if vehicle_no in seq:
            return track
    raise ValueError(f"Vehicle not found in final state: {vehicle_no}")
