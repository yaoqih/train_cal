from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, Field

from fzed_shunting.domain.depot_spots import allocate_spots_for_block, build_initial_spot_assignments
from fzed_shunting.io.normalize_input import NormalizedPlanInput


class ReplayState(BaseModel):
    track_sequences: dict[str, list[str]] = Field(default_factory=dict)
    loco_track_name: str
    weighed_vehicle_nos: set[str] = Field(default_factory=set)
    spot_assignments: dict[str, str] = Field(default_factory=dict)
    loco_carry: tuple[str, ...] = ()


class ReplayResult(BaseModel):
    snapshots: list[ReplayState]
    final_state: ReplayState


def build_initial_state(plan_input: NormalizedPlanInput) -> ReplayState:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for vehicle in plan_input.vehicles:
        grouped.setdefault(vehicle.current_track, []).append((vehicle.order, vehicle.vehicle_no))
    ordered = {
        track: [vehicle_no for _, vehicle_no in sorted(entries, key=lambda item: item[0])]
        for track, entries in grouped.items()
    }
    return ReplayState(
        track_sequences=ordered,
        loco_track_name=plan_input.loco_track_name,
        weighed_vehicle_nos=set(),
        spot_assignments=build_initial_spot_assignments(plan_input),
    )


def replay_plan(
    initial_state: ReplayState,
    hook_plan: list[dict],
    plan_input: NormalizedPlanInput | None = None,
) -> ReplayResult:
    state = ReplayState.model_validate(initial_state.model_dump())
    snapshots = [ReplayState.model_validate(state.model_dump())]
    vehicle_by_no = (
        {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
        if plan_input is not None
        else {}
    )
    for hook in hook_plan:
        action_type = hook.get("actionType", "PUT")
        vehicle_nos = hook["vehicleNos"]
        if action_type == "PUT":
            source = hook["sourceTrack"]
            target = hook["targetTrack"]
            source_seq = state.track_sequences.setdefault(source, [])
            target_seq = state.track_sequences.setdefault(target, [])
            if source_seq[: len(vehicle_nos)] != vehicle_nos:
                raise ValueError("Vehicle block is not at the north-end prefix of source track")
            del source_seq[: len(vehicle_nos)]
            for vehicle_no in vehicle_nos:
                state.spot_assignments.pop(vehicle_no, None)
            target_seq.extend(vehicle_nos)
            if plan_input is not None:
                block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in vehicle_nos]
                new_spot_assignments = allocate_spots_for_block(
                    vehicles=block_vehicles,
                    target_track=target,
                    yard_mode=plan_input.yard_mode,
                    occupied_spot_assignments=state.spot_assignments,
                )
                if new_spot_assignments is None:
                    raise ValueError(
                        f"No available depot spot for hook to {target}: {vehicle_nos}"
                    )
                state.spot_assignments.update(new_spot_assignments)
            state.loco_track_name = target
            if target == "机库":
                state.weighed_vehicle_nos.update(vehicle_nos)
        elif action_type == "ATTACH":
            source = hook["sourceTrack"]
            source_seq = state.track_sequences.setdefault(source, [])
            if source_seq[: len(vehicle_nos)] != vehicle_nos:
                raise ValueError("Vehicle block is not at the north-end prefix of source track")
            del source_seq[: len(vehicle_nos)]
            state.loco_carry = state.loco_carry + tuple(vehicle_nos)
            state.loco_track_name = source
        elif action_type == "DETACH":
            target = hook["targetTrack"]
            carry_list = list(state.loco_carry)
            if carry_list[: len(vehicle_nos)] != vehicle_nos:
                raise ValueError("Vehicle block is not at the front of loco_carry")
            state.loco_carry = tuple(carry_list[len(vehicle_nos):])
            target_seq = state.track_sequences.setdefault(target, [])
            for vehicle_no in vehicle_nos:
                state.spot_assignments.pop(vehicle_no, None)
            target_seq.extend(vehicle_nos)
            if plan_input is not None:
                block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in vehicle_nos]
                new_spot_assignments = allocate_spots_for_block(
                    vehicles=block_vehicles,
                    target_track=target,
                    yard_mode=plan_input.yard_mode,
                    occupied_spot_assignments=state.spot_assignments,
                )
                if new_spot_assignments is None:
                    raise ValueError(
                        f"No available depot spot for detach to {target}: {vehicle_nos}"
                    )
                state.spot_assignments.update(new_spot_assignments)
            state.loco_track_name = target
            if target == "机库":
                state.weighed_vehicle_nos.update(vehicle_nos)
        else:
            raise ValueError(f"Unsupported actionType: {action_type}")
        snapshots.append(ReplayState.model_validate(deepcopy(state.model_dump())))
    return ReplayResult(snapshots=snapshots, final_state=state)
