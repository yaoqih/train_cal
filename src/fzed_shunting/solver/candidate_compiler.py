from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.state import _apply_move
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class CompiledHookSequence:
    steps: tuple[HookAction, ...]
    final_state: ReplayState
    transitions: tuple[tuple[ReplayState, HookAction, ReplayState], ...]


def replay_candidate_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    steps: list[HookAction] | tuple[HookAction, ...],
    route_oracle: RouteOracle | None = None,
    apply_move: Callable[..., ReplayState] = _apply_move,
) -> CompiledHookSequence | None:
    next_state = state
    applied: list[HookAction] = []
    transitions: list[tuple[ReplayState, HookAction, ReplayState]] = []
    for step in steps:
        try:
            if _step_violates_hook_vehicle_group(
                step=step,
                vehicle_by_no=vehicle_by_no,
            ):
                return None
            if _step_violates_close_door_front_rule(
                step=step,
                vehicle_by_no=vehicle_by_no,
            ):
                return None
            if route_oracle is not None and not _step_is_route_legal(
                step=step,
                state=next_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
            ):
                return None
            before_state = next_state
            next_state = apply_move(
                state=before_state,
                move=step,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
        applied.append(step)
        transitions.append((before_state, step, next_state))
    return CompiledHookSequence(
        steps=tuple(applied),
        final_state=next_state,
        transitions=tuple(transitions),
    )


def _step_violates_hook_vehicle_group(
    *,
    step: HookAction,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    vehicles: list[NormalizedVehicle] = []
    for vehicle_no in step.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return True
        vehicles.append(vehicle)
    return bool(validate_hook_vehicle_group(vehicles))


def _step_violates_close_door_front_rule(
    *,
    step: HookAction,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if step.target_track == "存4北" or len(step.vehicle_nos) <= 10:
        return False
    first_vehicle = vehicle_by_no.get(step.vehicle_nos[0])
    return bool(first_vehicle and first_vehicle.is_close_door)


def _step_is_route_legal(
    *,
    step: HookAction,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    length_by_vehicle = {
        vehicle_no: vehicle.vehicle_length for vehicle_no, vehicle in vehicle_by_no.items()
    }
    if step.action_type == "ATTACH":
        access = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=step.source_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
            carried_train_length_m=sum(
                length_by_vehicle.get(vehicle_no, 0.0)
                for vehicle_no in state.loco_carry
            ),
        )
        return access.is_valid
    if step.action_type == "DETACH":
        if step.source_track != state.loco_track_name:
            return False
        source_remaining = len(state.track_sequences.get(step.source_track, []))
        source_node = state.loco_node if source_remaining > 0 else None
        target_node = (
            route_oracle.order_end_node(step.target_track)
            if step.target_track != step.source_track
            else None
        )
        route = route_oracle.validate_path(
            source_track=step.source_track,
            target_track=step.target_track,
            path_tracks=list(step.path_tracks),
            train_length_m=sum(
                length_by_vehicle.get(vehicle_no, 0.0)
                for vehicle_no in (state.loco_carry or tuple(step.vehicle_nos))
            ),
            occupied_track_sequences=state.track_sequences,
            source_node=source_node,
            target_node=target_node,
        )
        return route.is_valid
    return False
