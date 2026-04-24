from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.purity import STAGING_TRACKS, compute_state_purity
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class StructuralMetrics:
    unfinished_count: int
    preferred_violation_count: int
    staging_debt_count: int
    staging_debt_by_track: dict[str, int]
    area_random_unfinished_count: int
    front_blocker_count: int
    front_blocker_by_track: dict[str, int]
    goal_track_blocker_count: int
    goal_track_blocker_by_track: dict[str, int]
    capacity_overflow_track_count: int
    capacity_debt_by_track: dict[str, float]
    loco_carry_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_structural_metrics(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> StructuralMetrics:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    track_by_vehicle = _vehicle_track_lookup(state)
    purity = compute_state_purity(plan_input, state)

    staging_debt_by_track: Counter[str] = Counter()
    area_random_unfinished_count = 0
    for vehicle in plan_input.vehicles:
        track = track_by_vehicle.get(vehicle.vehicle_no)
        if track is None:
            if vehicle.goal.target_area_code is not None:
                area_random_unfinished_count += 1
            continue
        is_finished = goal_is_satisfied(
            vehicle,
            track_name=track,
            state=state,
            plan_input=plan_input,
        )
        if not is_finished and vehicle.goal.target_area_code is not None:
            area_random_unfinished_count += 1
        if track in STAGING_TRACKS and not is_finished:
            staging_debt_by_track[track] += 1

    front_blocker_by_track = _front_blocker_pressure(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    goal_track_blocker_by_track = _goal_track_blockers(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    capacity_debt_by_track = _capacity_debt_by_track(plan_input, state)

    return StructuralMetrics(
        unfinished_count=purity.unfinished_count,
        preferred_violation_count=purity.preferred_violation_count,
        staging_debt_count=sum(staging_debt_by_track.values()),
        staging_debt_by_track=dict(sorted(staging_debt_by_track.items())),
        area_random_unfinished_count=area_random_unfinished_count,
        front_blocker_count=sum(front_blocker_by_track.values()),
        front_blocker_by_track=dict(sorted(front_blocker_by_track.items())),
        goal_track_blocker_count=sum(goal_track_blocker_by_track.values()),
        goal_track_blocker_by_track=dict(sorted(goal_track_blocker_by_track.items())),
        capacity_overflow_track_count=len(capacity_debt_by_track),
        capacity_debt_by_track=capacity_debt_by_track,
        loco_carry_count=len(state.loco_carry),
    )


def summarize_plan_shape(plan: list[HookAction]) -> dict[str, int]:
    touches: Counter[str] = Counter()
    staging_hook_count = 0
    staging_to_staging_hook_count = 0
    for move in plan:
        source_is_staging = move.source_track in STAGING_TRACKS
        target_is_staging = move.target_track in STAGING_TRACKS
        if source_is_staging or target_is_staging:
            staging_hook_count += 1
        if source_is_staging and target_is_staging:
            staging_to_staging_hook_count += 1
        touches.update(move.vehicle_nos)
    return {
        "staging_hook_count": staging_hook_count,
        "staging_to_staging_hook_count": staging_to_staging_hook_count,
        "rehandled_vehicle_count": sum(1 for count in touches.values() if count > 2),
        "max_vehicle_touch_count": max(touches.values(), default=0),
    }


def _front_blocker_pressure(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
) -> dict[str, int]:
    pressure: Counter[str] = Counter()
    for track, seq in state.track_sequences.items():
        seen_blocker = False
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            satisfied_here = goal_is_satisfied(
                vehicle,
                track_name=track,
                state=state,
                plan_input=plan_input,
            )
            if seen_blocker and not satisfied_here:
                pressure[track] += 1
                break
            if not satisfied_here:
                seen_blocker = True
    return dict(pressure)


def _goal_track_blockers(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
) -> dict[str, int]:
    needed_tracks: set[str] = set()
    current_track_by_vehicle = _vehicle_track_lookup(state)
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is not None and goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        needed_tracks.update(vehicle.goal.allowed_target_tracks)

    blockers: Counter[str] = Counter()
    for track in needed_tracks:
        for vehicle_no in state.track_sequences.get(track, []):
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            if not goal_is_satisfied(
                vehicle,
                track_name=track,
                state=state,
                plan_input=plan_input,
            ):
                blockers[track] += 1
    return dict(blockers)


def _capacity_debt_by_track(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> dict[str, float]:
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    required_by_track: Counter[str] = Counter()
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    for vehicle in plan_input.vehicles:
        if len(vehicle.goal.allowed_target_tracks) == 1:
            required_by_track[vehicle.goal.allowed_target_tracks[0]] += vehicle.vehicle_length
    for track, seq in state.track_sequences.items():
        for vehicle_no in seq:
            required_by_track[track] += length_by_vehicle.get(vehicle_no, 0.0)
    debt: dict[str, float] = {}
    for track, required in required_by_track.items():
        capacity = capacity_by_track.get(track)
        if capacity is not None and required > capacity + 1e-9:
            debt[track] = round(required - capacity, 3)
    return dict(sorted(debt.items()))
