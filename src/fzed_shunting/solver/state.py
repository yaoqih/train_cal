"""Pure state-space helpers shared by search, LNS, and constructive layers."""

from __future__ import annotations

from fzed_shunting.domain.depot_spots import allocate_spots_for_block, spot_candidates_for_vehicle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


def _is_goal(plan_input: NormalizedPlanInput, state: ReplayState) -> bool:
    if state.loco_carry:
        return False
    current_track_by_vehicle = _vehicle_track_lookup(state)
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle[vehicle.vehicle_no]
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


def _vehicle_track_lookup(state: ReplayState) -> dict[str, str]:
    return {
        vehicle_no: track_name
        for track_name, seq in state.track_sequences.items()
        for vehicle_no in seq
    }


def _locate_vehicle(state: ReplayState, vehicle_no: str) -> str:
    for track, seq in state.track_sequences.items():
        if vehicle_no in seq:
            return track
    raise ValueError(f"Vehicle not found in state: {vehicle_no}")


def _canonical_random_depot_vehicle_nos(plan_input: NormalizedPlanInput) -> frozenset[str]:
    if any(vehicle.goal.target_mode == "SPOT" for vehicle in plan_input.vehicles):
        return frozenset()
    return frozenset(
        vehicle.vehicle_no
        for vehicle in plan_input.vehicles
        if vehicle.goal.target_area_code == "大库:RANDOM"
    )


def _apply_move(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> ReplayState:
    if move.action_type == "ATTACH":
        return _apply_attach(state=state, move=move)
    if move.action_type == "DETACH":
        return _apply_detach(state=state, move=move, plan_input=plan_input, vehicle_by_no=vehicle_by_no)
    return _apply_put(state=state, move=move, plan_input=plan_input, vehicle_by_no=vehicle_by_no)


def _place_on_track(existing: list[str], new_vehicles: list[str]) -> list[str]:
    """All placement is at the north (accessible) end — new vehicles prepend."""
    return list(new_vehicles) + existing


def _apply_put(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> ReplayState:
    source_seq = state.track_sequences.get(move.source_track, [])
    if source_seq[: len(move.vehicle_nos)] != move.vehicle_nos:
        raise ValueError("Vehicle block is not at the north-end prefix of source track")

    next_track_sequences = dict(state.track_sequences)
    next_track_sequences[move.source_track] = list(source_seq[len(move.vehicle_nos):])
    next_target_seq = list(state.track_sequences.get(move.target_track, []))
    next_target_seq.extend(move.vehicle_nos)
    next_track_sequences[move.target_track] = next_target_seq

    next_spot_assignments = dict(state.spot_assignments)
    for vehicle_no in move.vehicle_nos:
        next_spot_assignments.pop(vehicle_no, None)
    block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in move.vehicle_nos]
    new_spot_assignments = allocate_spots_for_block(
        vehicles=block_vehicles,
        target_track=move.target_track,
        yard_mode=plan_input.yard_mode,
        occupied_spot_assignments=next_spot_assignments,
    )
    if new_spot_assignments is None:
        raise ValueError(
            f"No available depot spot for hook to {move.target_track}: {move.vehicle_nos}"
        )
    next_spot_assignments.update(new_spot_assignments)

    next_weighed_vehicle_nos = set(state.weighed_vehicle_nos)
    if move.target_track == "机库":
        next_weighed_vehicle_nos.update(move.vehicle_nos)

    return ReplayState(
        track_sequences=next_track_sequences,
        loco_track_name=move.target_track,
        weighed_vehicle_nos=next_weighed_vehicle_nos,
        spot_assignments=next_spot_assignments,
        loco_carry=state.loco_carry,
    )


def _apply_attach(
    *,
    state: ReplayState,
    move: HookAction,
) -> ReplayState:
    source_seq = state.track_sequences.get(move.source_track, [])
    if source_seq[: len(move.vehicle_nos)] != move.vehicle_nos:
        raise ValueError("Vehicle block is not at the north-end prefix of source track")
    next_track_sequences = dict(state.track_sequences)
    next_track_sequences[move.source_track] = list(source_seq[len(move.vehicle_nos):])
    return ReplayState(
        track_sequences=next_track_sequences,
        loco_track_name=move.source_track,
        weighed_vehicle_nos=set(state.weighed_vehicle_nos),
        spot_assignments=dict(state.spot_assignments),
        loco_carry=state.loco_carry + tuple(move.vehicle_nos),
    )


def _apply_detach(
    *,
    state: ReplayState,
    move: HookAction,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> ReplayState:
    carry_list = list(state.loco_carry)
    k = len(move.vehicle_nos)
    if carry_list[:k] != move.vehicle_nos:
        raise ValueError("Vehicle block is not at the front of loco_carry")
    next_carry = tuple(carry_list[k:])
    next_track_sequences = dict(state.track_sequences)
    existing_target_seq = list(state.track_sequences.get(move.target_track, []))
    next_target_seq = _place_on_track(existing_target_seq, move.vehicle_nos)
    next_track_sequences[move.target_track] = next_target_seq
    next_spot_assignments = dict(state.spot_assignments)
    for vehicle_no in move.vehicle_nos:
        next_spot_assignments.pop(vehicle_no, None)
    block_vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in move.vehicle_nos]
    new_spot_assignments = allocate_spots_for_block(
        vehicles=block_vehicles,
        target_track=move.target_track,
        yard_mode=plan_input.yard_mode,
        occupied_spot_assignments=next_spot_assignments,
    )
    if new_spot_assignments is None:
        raise ValueError(
            f"No available depot spot for detach to {move.target_track}: {move.vehicle_nos}"
        )
    next_spot_assignments.update(new_spot_assignments)
    next_weighed_vehicle_nos = set(state.weighed_vehicle_nos)
    if move.target_track == "机库":
        next_weighed_vehicle_nos.update(move.vehicle_nos)
    return ReplayState(
        track_sequences=next_track_sequences,
        loco_track_name=move.target_track,
        weighed_vehicle_nos=next_weighed_vehicle_nos,
        spot_assignments=next_spot_assignments,
        loco_carry=next_carry,
    )


def _state_key(
    state: ReplayState,
    plan_input: NormalizedPlanInput | None = None,
    *,
    canonical_random_depot_vehicle_nos: frozenset[str] | None = None,
) -> tuple:
    if canonical_random_depot_vehicle_nos is None:
        canonical_random_depot_vehicle_nos = (
            _canonical_random_depot_vehicle_nos(plan_input)
            if plan_input is not None
            else frozenset()
        )
    spot_items = tuple(
        (vehicle_no, spot_code)
        for vehicle_no, spot_code in sorted(state.spot_assignments.items())
        if not (
            vehicle_no in canonical_random_depot_vehicle_nos
            and spot_code.isdigit()
        )
    )
    return (
        tuple(
            (track, tuple(seq))
            for track, seq in sorted(state.track_sequences.items())
            if seq
        ),
        tuple(sorted(state.weighed_vehicle_nos)),
        spot_items,
        state.loco_carry,
    )
