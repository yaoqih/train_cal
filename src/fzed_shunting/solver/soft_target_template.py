from __future__ import annotations

from dataclasses import asdict, dataclass

from fzed_shunting.domain.depot_spots import allocate_spots_for_block, exact_spot_reservations
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.capacity_release import (
    CapacityReleasePlan,
    compute_capacity_release_plan,
)
from fzed_shunting.solver.goal_logic import (
    goal_effective_allowed_tracks,
    goal_is_satisfied,
    goal_track_preference_level,
)
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class SoftTargetCandidate:
    track_name: str
    preference_level: int
    allocatable: bool
    current_length: float
    capacity_slack_length: float
    release_pressure_length: float
    current_vehicle_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VehicleSoftTargetTemplate:
    vehicle_no: str
    candidate_tracks: list[SoftTargetCandidate]
    preferred_track: str | None

    def to_dict(self) -> dict:
        return {
            "vehicle_no": self.vehicle_no,
            "preferred_track": self.preferred_track,
            "candidate_tracks": [
                candidate.to_dict() for candidate in self.candidate_tracks
            ],
        }


@dataclass(frozen=True)
class Cun4BeiTemplate:
    unresolved_close_door_vehicle_nos: list[str]
    required_plain_prefix_count: int
    available_plain_target_vehicle_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SoftTargetTemplate:
    vehicle_templates: dict[str, VehicleSoftTargetTemplate]
    cun4bei_template: Cun4BeiTemplate

    def to_dict(self) -> dict:
        return {
            "vehicle_templates": {
                vehicle_no: template.to_dict()
                for vehicle_no, template in sorted(self.vehicle_templates.items())
            },
            "cun4bei_template": self.cun4bei_template.to_dict(),
        }

    def to_summary_dict(self) -> dict:
        preferred_track_counts: dict[str, int] = {}
        random_vehicle_count = 0
        for template in self.vehicle_templates.values():
            if template.preferred_track is not None:
                preferred_track_counts[template.preferred_track] = (
                    preferred_track_counts.get(template.preferred_track, 0) + 1
                )
            if any(
                candidate.track_name.startswith("修")
                and candidate.track_name.endswith("库内")
                for candidate in template.candidate_tracks
            ):
                random_vehicle_count += 1
        return {
            "vehicle_template_count": len(self.vehicle_templates),
            "random_depot_template_count": random_vehicle_count,
            "preferred_track_counts": dict(sorted(preferred_track_counts.items())),
            "cun4bei_template": self.cun4bei_template.to_dict(),
        }


def compute_soft_target_template(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    capacity_plan: CapacityReleasePlan | None = None,
) -> SoftTargetTemplate:
    if capacity_plan is None:
        capacity_plan = compute_capacity_release_plan(plan_input, state)
    track_by_vehicle = _vehicle_track_lookup(state)
    vehicle_templates: dict[str, VehicleSoftTargetTemplate] = {}
    for vehicle in plan_input.vehicles:
        if _is_vehicle_finished(vehicle, track_by_vehicle, state, plan_input):
            continue
        candidates = _candidate_tracks_for_vehicle(
            vehicle=vehicle,
            plan_input=plan_input,
            state=state,
            capacity_plan=capacity_plan,
        )
        if not candidates:
            continue
        vehicle_templates[vehicle.vehicle_no] = VehicleSoftTargetTemplate(
            vehicle_no=vehicle.vehicle_no,
            candidate_tracks=candidates,
            preferred_track=candidates[0].track_name,
        )
    return SoftTargetTemplate(
        vehicle_templates=dict(sorted(vehicle_templates.items())),
        cun4bei_template=_compute_cun4bei_template(plan_input, state, track_by_vehicle),
    )


def _candidate_tracks_for_vehicle(
    *,
    vehicle: NormalizedVehicle,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    capacity_plan: CapacityReleasePlan,
) -> list[SoftTargetCandidate]:
    tracks = goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=plan_input,
    )
    candidates: list[SoftTargetCandidate] = []
    for track_name in tracks:
        fact = capacity_plan.facts_by_track.get(track_name)
        if fact is None:
            continue
        preference_level = goal_track_preference_level(
            vehicle,
            track_name,
            state=state,
            plan_input=plan_input,
        )
        if preference_level is None:
            continue
        slack = fact.capacity_length - fact.current_length
        if vehicle.goal.work_position_kind is not None:
            allocatable = slack + 1e-9 >= vehicle.vehicle_length
        else:
            allocatable = (
                allocate_spots_for_block(
                    vehicles=[vehicle],
                    target_track=track_name,
                    yard_mode=plan_input.yard_mode,
                    occupied_spot_assignments=state.spot_assignments,
                    reserved_spot_codes=exact_spot_reservations(plan_input),
                )
                is not None
            )
        candidates.append(
            SoftTargetCandidate(
                track_name=track_name,
                preference_level=preference_level,
                allocatable=allocatable,
                current_length=fact.current_length,
                capacity_slack_length=round(slack, 3),
                release_pressure_length=fact.release_pressure_length,
                current_vehicle_count=fact.current_vehicle_count,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.preference_level,
            0 if candidate.allocatable else 1,
            candidate.release_pressure_length,
            candidate.current_length,
            -candidate.capacity_slack_length,
            candidate.current_vehicle_count,
            candidate.track_name,
        )
    )
    return candidates


def _compute_cun4bei_template(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_by_vehicle: dict[str, str],
) -> Cun4BeiTemplate:
    unresolved_close_doors: list[str] = []
    available_plain_targets = 0
    for vehicle in sorted(plan_input.vehicles, key=lambda item: item.vehicle_no):
        if "存4北" not in vehicle.goal.allowed_target_tracks:
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        finished = (
            current_track is not None
            and goal_is_satisfied(
                vehicle,
                track_name=current_track,
                state=state,
                plan_input=plan_input,
            )
        )
        if finished:
            if not vehicle.is_close_door and current_track == "存4北":
                available_plain_targets += 1
            continue
        if vehicle.is_close_door:
            unresolved_close_doors.append(vehicle.vehicle_no)
        else:
            available_plain_targets += 1
    missing_prefix = max(0, 3 - _plain_prefix_count_at_cun4bei(plan_input, state))
    return Cun4BeiTemplate(
        unresolved_close_door_vehicle_nos=unresolved_close_doors,
        required_plain_prefix_count=missing_prefix if unresolved_close_doors else 0,
        available_plain_target_vehicle_count=available_plain_targets,
    )


def _plain_prefix_count_at_cun4bei(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> int:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    count = 0
    for vehicle_no in state.track_sequences.get("存4北", []):
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.is_close_door:
            break
        if "存4北" not in vehicle.goal.allowed_target_tracks:
            break
        count += 1
    return count


def _is_vehicle_finished(
    vehicle: NormalizedVehicle,
    track_by_vehicle: dict[str, str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    current_track = track_by_vehicle.get(vehicle.vehicle_no)
    if current_track is None:
        return False
    return goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    )
