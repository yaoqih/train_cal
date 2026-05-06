from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from fzed_shunting.domain.work_positions import (
    allowed_spotting_south_ranks,
    north_rank,
    preview_work_positions_after_prepend,
    south_rank,
)
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
    work_position_unfinished_count: int
    front_blocker_count: int
    front_blocker_by_track: dict[str, int]
    target_sequence_defect_count: int
    target_sequence_defect_by_track: dict[str, int]
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
    work_position_unfinished_count = 0
    for vehicle in plan_input.vehicles:
        track = track_by_vehicle.get(vehicle.vehicle_no)
        if track is None:
            if vehicle.goal.target_area_code is not None:
                area_random_unfinished_count += 1
            if vehicle.goal.work_position_kind is not None:
                work_position_unfinished_count += 1
            continue
        is_finished = goal_is_satisfied(
            vehicle,
            track_name=track,
            state=state,
            plan_input=plan_input,
        )
        if not is_finished and vehicle.goal.target_area_code is not None:
            area_random_unfinished_count += 1
        if not is_finished and vehicle.goal.work_position_kind is not None:
            work_position_unfinished_count += 1
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
    target_sequence_defect_by_track = _target_sequence_defects(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        current_track_by_vehicle=track_by_vehicle,
    )
    capacity_debt_by_track = _capacity_debt_by_track(plan_input, state)

    return StructuralMetrics(
        unfinished_count=purity.unfinished_count,
        preferred_violation_count=purity.preferred_violation_count,
        staging_debt_count=sum(staging_debt_by_track.values()),
        staging_debt_by_track=dict(sorted(staging_debt_by_track.items())),
        area_random_unfinished_count=area_random_unfinished_count,
        work_position_unfinished_count=work_position_unfinished_count,
        front_blocker_count=sum(front_blocker_by_track.values()),
        front_blocker_by_track=dict(sorted(front_blocker_by_track.items())),
        target_sequence_defect_count=sum(target_sequence_defect_by_track.values()),
        target_sequence_defect_by_track=dict(sorted(target_sequence_defect_by_track.items())),
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
        seen_front_vehicle = False
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                seen_front_vehicle = True
                continue
            satisfied_here = goal_is_satisfied(
                vehicle,
                track_name=track,
                state=state,
                plan_input=plan_input,
            )
            if seen_front_vehicle and not satisfied_here:
                pressure[track] += 1
                break
            seen_front_vehicle = True
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


def _target_sequence_defects(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict,
    current_track_by_vehicle: dict[str, str],
) -> dict[str, int]:
    defects: Counter[str] = Counter()
    unfinished_work_by_track: dict[str, list[tuple[Any, str | None]]] = {}
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind is None:
            continue
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is not None and goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        for target_track in vehicle.goal.allowed_target_tracks:
            unfinished_work_by_track.setdefault(target_track, []).append((vehicle, current_track))

    for target_track, vehicle_entries in unfinished_work_by_track.items():
        seq = state.track_sequences.get(target_track, [])
        defects[target_track] = max(
            _work_position_sequence_defect(
                vehicle,
                current_track=current_track,
                target_track=target_track,
                target_seq=seq,
                vehicle_by_no=vehicle_by_no,
            )
            for vehicle, current_track in vehicle_entries
        )
    return dict(defects)


def _work_position_sequence_defect(
    vehicle: Any,
    *,
    current_track: str | None,
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict,
) -> int:
    if current_track == target_track:
        return _work_position_rank_defect(vehicle, target_track=target_track, seq=target_seq)
    return _work_position_insertion_defect(
        vehicle,
        target_track=target_track,
        target_seq=target_seq,
        vehicle_by_no=vehicle_by_no,
    )


def _work_position_insertion_defect(
    vehicle: Any,
    *,
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict,
) -> int:
    kind = vehicle.goal.work_position_kind
    for clear_count in range(len(target_seq) + 1):
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=[vehicle.vehicle_no],
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        evaluation = preview.evaluations.get(vehicle.vehicle_no)
        if evaluation is None:
            continue
        if kind == "SPOTTING" and not evaluation.satisfied_now:
            continue
        return clear_count
    if kind == "SPOTTING":
        ranks = allowed_spotting_south_ranks(target_track)
        if not ranks:
            return 0
    return 0


def _work_position_rank_defect(vehicle: Any, *, target_track: str, seq: list[str]) -> int:
    kind = vehicle.goal.work_position_kind
    if kind == "SPOTTING":
        rank = south_rank(seq, vehicle.vehicle_no)
        ranks = allowed_spotting_south_ranks(target_track)
        if rank is None or not ranks or rank in ranks:
            return 0
        if rank > max(ranks):
            return rank - max(ranks)
        return min(ranks) - rank
    if kind in {"EXACT_NORTH_RANK", "EXACT_WORK_SLOT"}:
        rank = north_rank(seq, vehicle.vehicle_no)
        target_rank = vehicle.goal.target_rank
        if rank is None or target_rank is None:
            return 0
        if kind == "EXACT_WORK_SLOT":
            return max(0, rank - target_rank)
        return abs(rank - target_rank)
    return 0


def _capacity_debt_by_track(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> dict[str, float]:
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation_by_track: Counter[str] = Counter()
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] += vehicle.vehicle_length
    required_by_track: Counter[str] = Counter()
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    current_track_by_vehicle = _vehicle_track_lookup(state)
    for vehicle in plan_input.vehicles:
        if len(vehicle.goal.allowed_target_tracks) == 1:
            target_track = vehicle.goal.allowed_target_tracks[0]
            if current_track_by_vehicle.get(vehicle.vehicle_no) != target_track:
                required_by_track[target_track] += vehicle.vehicle_length
    for track, seq in state.track_sequences.items():
        for vehicle_no in seq:
            required_by_track[track] += length_by_vehicle.get(vehicle_no, 0.0)
    debt: dict[str, float] = {}
    for track, required in required_by_track.items():
        capacity = capacity_by_track.get(track)
        if capacity is None:
            continue
        effective_capacity = max(capacity, initial_occupation_by_track.get(track, 0.0))
        if required > effective_capacity + 1e-9:
            debt[track] = round(required - effective_capacity, 3)
    return dict(sorted(debt.items()))
