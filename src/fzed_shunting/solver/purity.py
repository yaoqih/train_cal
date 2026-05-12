from __future__ import annotations

from dataclasses import dataclass

from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import (
    goal_can_use_fallback_now,
    goal_is_preferred_satisfied,
    goal_is_satisfied,
)
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.verify.replay import ReplayState


STAGING_TRACKS = frozenset({"机北1", "机北2", "洗油北", "机南", "存4南"})


@dataclass(frozen=True)
class PurityMetrics:
    unfinished_count: int
    preferred_violation_count: int
    staging_pollution_count: int


def compute_state_purity(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> PurityMetrics:
    current_track_by_vehicle = _vehicle_track_lookup(state)
    unfinished_count = 0
    preferred_violation_count = 0
    staging_pollution_count = 0

    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None:
            unfinished_count += 1
            continue
        if _counts_as_unfinished(
            vehicle,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
            unfinished_count += 1
        elif _counts_as_preferred_violation(
            vehicle,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
            preferred_violation_count += 1
        if current_track in STAGING_TRACKS and current_track not in vehicle.goal.allowed_target_tracks:
            staging_pollution_count += 1

    if state.loco_carry:
        unfinished_count += len(state.loco_carry)

    return PurityMetrics(
        unfinished_count=unfinished_count,
        preferred_violation_count=preferred_violation_count,
        staging_pollution_count=staging_pollution_count,
    )


def _counts_as_unfinished(
    vehicle: NormalizedVehicle,
    *,
    current_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if not goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    ):
        return True
    return False


def _counts_as_preferred_violation(
    vehicle: NormalizedVehicle,
    *,
    current_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if not goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    ):
        return False
    if _is_fallback_while_preferred_still_feasible(
        vehicle,
        current_track=current_track,
        state=state,
        plan_input=plan_input,
    ):
        return True
    return not goal_is_preferred_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    )


def _is_fallback_while_preferred_still_feasible(
    vehicle: NormalizedVehicle,
    *,
    current_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if vehicle.goal.target_mode == "SNAPSHOT":
        return False
    if current_track not in vehicle.goal.fallback_target_tracks:
        return False
    if not vehicle.goal.preferred_target_tracks:
        return False
    return not goal_can_use_fallback_now(
        vehicle,
        state=state,
        plan_input=plan_input,
    )
