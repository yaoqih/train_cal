from __future__ import annotations

from dataclasses import asdict, dataclass

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class TrackCapacityReleaseFact:
    track_name: str
    capacity_length: float
    current_length: float
    fixed_inbound_length: float
    keepable_current_length: float
    non_goal_current_length: float
    release_pressure_length: float
    front_release_vehicle_nos: list[str]
    front_release_length: float
    current_vehicle_count: int
    fixed_inbound_vehicle_count: int
    non_goal_current_vehicle_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CapacityReleasePlan:
    facts_by_track: dict[str, TrackCapacityReleaseFact]

    def to_dict(self) -> dict:
        return {
            "facts_by_track": {
                track: fact.to_dict()
                for track, fact in sorted(self.facts_by_track.items())
            }
        }


def compute_capacity_release_plan(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> CapacityReleasePlan:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    track_by_vehicle = _vehicle_track_lookup(state)
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0)
            + vehicle.vehicle_length
        )
    fixed_inbound_by_track: dict[str, list[str]] = {track: [] for track in capacity_by_track}
    for vehicle in plan_input.vehicles:
        allowed = vehicle.goal.allowed_target_tracks
        if len(allowed) != 1:
            continue
        target_track = allowed[0]
        if target_track not in capacity_by_track:
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track == target_track and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        fixed_inbound_by_track.setdefault(target_track, []).append(vehicle.vehicle_no)

    facts: dict[str, TrackCapacityReleaseFact] = {}
    for track_name, capacity in capacity_by_track.items():
        effective_capacity = max(capacity, initial_occupation_by_track.get(track_name, 0.0))
        seq = list(state.track_sequences.get(track_name, []))
        current_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in seq)
        keepable_current_length = 0.0
        non_goal_current_length = 0.0
        non_goal_current_count = 0
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is not None and goal_is_satisfied(
                vehicle,
                track_name=track_name,
                state=state,
                plan_input=plan_input,
            ):
                keepable_current_length += length_by_vehicle.get(vehicle_no, 0.0)
            else:
                non_goal_current_length += length_by_vehicle.get(vehicle_no, 0.0)
                non_goal_current_count += 1

        fixed_inbound = fixed_inbound_by_track.get(track_name, [])
        fixed_inbound_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in fixed_inbound)
        release_pressure = max(
            0.0,
            current_length + fixed_inbound_length - effective_capacity,
        )
        front_release_vehicle_nos = _front_release_vehicle_nos(
            seq=seq,
            release_pressure=release_pressure,
            length_by_vehicle=length_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        front_release_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in front_release_vehicle_nos
        )
        facts[track_name] = TrackCapacityReleaseFact(
            track_name=track_name,
            capacity_length=round(capacity, 3),
            current_length=round(current_length, 3),
            fixed_inbound_length=round(fixed_inbound_length, 3),
            keepable_current_length=round(keepable_current_length, 3),
            non_goal_current_length=round(non_goal_current_length, 3),
            release_pressure_length=round(release_pressure, 3),
            front_release_vehicle_nos=front_release_vehicle_nos,
            front_release_length=round(front_release_length, 3),
            current_vehicle_count=len(seq),
            fixed_inbound_vehicle_count=len(fixed_inbound),
            non_goal_current_vehicle_count=non_goal_current_count,
        )
    return CapacityReleasePlan(facts_by_track=dict(sorted(facts.items())))


def _front_release_vehicle_nos(
    *,
    seq: list[str],
    release_pressure: float,
    length_by_vehicle: dict[str, float],
    vehicle_by_no: dict,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[str]:
    if release_pressure <= 1e-9:
        return []
    released: list[str] = []
    released_length = 0.0
    for vehicle_no in seq:
        released.append(vehicle_no)
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is not None and not goal_is_satisfied(
            vehicle,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        ):
            released_length += length_by_vehicle.get(vehicle_no, 0.0)
        if released_length + 1e-9 >= release_pressure:
            break
    return released
