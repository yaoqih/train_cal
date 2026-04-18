from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fzed_shunting.domain.depot_spots import WORK_AREA_SPOTS
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class HeuristicBreakdown:
    h_misplaced: int
    h_distinct_transfer_pairs: int
    h_blocking: int
    h_weigh: int
    h_spot_evict: int

    @property
    def value(self) -> int:
        return max(
            self.h_distinct_transfer_pairs,
            self.h_blocking,
            self.h_weigh,
            self.h_spot_evict,
        )


def compute_admissible_heuristic(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> int:
    return compute_heuristic_breakdown(plan_input, state).value


def compute_heuristic_breakdown(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> HeuristicBreakdown:
    current_track_by_vehicle = _vehicle_track_lookup(state)
    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}

    return HeuristicBreakdown(
        h_misplaced=_h_misplaced_vehicles(
            plan_input=plan_input,
            current_track_by_vehicle=current_track_by_vehicle,
        ),
        h_distinct_transfer_pairs=_h_distinct_transfer_pairs(
            plan_input=plan_input,
            current_track_by_vehicle=current_track_by_vehicle,
        ),
        h_blocking=_h_blocking(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            current_track_by_vehicle=current_track_by_vehicle,
        ),
        h_weigh=_h_weigh_remaining(plan_input=plan_input, state=state),
        h_spot_evict=_h_spot_evict(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
        ),
    )


def make_state_heuristic(
    plan_input: NormalizedPlanInput,
) -> Callable[[ReplayState], int]:
    vehicles = tuple(plan_input.vehicles)
    vehicle_by_no = {v.vehicle_no: v for v in vehicles}
    weigh_vehicle_nos = frozenset(v.vehicle_no for v in vehicles if v.need_weigh)
    close_4north_vehicle_nos = frozenset(
        v.vehicle_no
        for v in vehicles
        if v.is_close_door and "存4北" in v.goal.allowed_target_tracks
    )

    def _heuristic(state: ReplayState) -> int:
        current_track_by_vehicle = _vehicle_track_lookup(state)
        h_pairs = _h_distinct_transfer_pairs(
            plan_input=plan_input,
            current_track_by_vehicle=current_track_by_vehicle,
        )
        h_block = _h_blocking(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            current_track_by_vehicle=current_track_by_vehicle,
        )
        h_weigh = _h_weigh_remaining_precomputed(
            weigh_vehicle_nos=weigh_vehicle_nos,
            weighed_vehicle_nos=state.weighed_vehicle_nos,
        )
        h_spot = _h_spot_evict(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
        )
        _ = close_4north_vehicle_nos
        return max(h_pairs, h_block, h_weigh, h_spot)

    return _heuristic


def _vehicle_track_lookup(state: ReplayState) -> dict[str, str]:
    return {
        vehicle_no: track_name
        for track_name, seq in state.track_sequences.items()
        for vehicle_no in seq
    }


def _h_misplaced_vehicles(
    *,
    plan_input: NormalizedPlanInput,
    current_track_by_vehicle: dict[str, str],
) -> int:
    count = 0
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None:
            count += 1
            continue
        if current_track not in vehicle.goal.allowed_target_tracks:
            count += 1
    return count


def _h_distinct_transfer_pairs(
    *,
    plan_input: NormalizedPlanInput,
    current_track_by_vehicle: dict[str, str],
) -> int:
    """Admissible lower bound on remaining hooks.

    Each hook transfers vehicles between exactly one (source_track, target_track)
    pair. The lower bound is built from two disjoint contributions:

    1. Forced pairs: misplaced vehicles whose goal allows exactly one target
       track. Each distinct (current, allowed[0]) pair requires one dedicated
       hook.

    2. Multi-target source groups: misplaced vehicles whose allowed target set
       has more than one element (e.g. 大库:RANDOM). They CAN share a hook
       with a forced pair if the forced pair's target is in their allowed
       set. Otherwise, they need at least one extra hook from their source.
       We count at most one extra hook per source track (conservatively
       collapsing multiple distinct allowed-sets at the same source into a
       single shared hook — this stays admissible because a single extra
       hook is always ≤ the true number of hooks needed).
    """
    forced_pairs: set[tuple[str, str]] = set()
    multi_target_buckets: dict[str, list[frozenset[str]]] = {}
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None:
            continue
        allowed = vehicle.goal.allowed_target_tracks
        if current_track in allowed:
            continue
        if len(allowed) == 0:
            continue
        if len(allowed) == 1:
            forced_pairs.add((current_track, allowed[0]))
            continue
        multi_target_buckets.setdefault(current_track, []).append(frozenset(allowed))

    extra_hooks = 0
    for source, allowed_sets in multi_target_buckets.items():
        forced_targets_at_source = {
            target for (src, target) in forced_pairs if src == source
        }
        uncovered = [
            allowed_set
            for allowed_set in allowed_sets
            if not (allowed_set & forced_targets_at_source)
        ]
        if uncovered:
            extra_hooks += 1
    return len(forced_pairs) + extra_hooks


def _h_blocking(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    current_track_by_vehicle: dict[str, str],
) -> int:
    goal_tracks_needed: set[str] = set()
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track in vehicle.goal.allowed_target_tracks:
            continue
        for candidate in vehicle.goal.allowed_target_tracks:
            goal_tracks_needed.add(candidate)

    blocker_count = 0
    for target_track in goal_tracks_needed:
        seq = state.track_sequences.get(target_track, [])
        if not seq:
            continue
        cluster_blockers = 0
        for vehicle_no in seq:
            blocker = vehicle_by_no.get(vehicle_no)
            if blocker is None:
                continue
            if target_track in blocker.goal.allowed_target_tracks:
                break
            cluster_blockers += 1
        if cluster_blockers > 0:
            blocker_count += 1
    return blocker_count


def _h_weigh_remaining(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> int:
    count = 0
    for vehicle in plan_input.vehicles:
        if not vehicle.need_weigh:
            continue
        if vehicle.vehicle_no in state.weighed_vehicle_nos:
            continue
        count += 1
    return count


def _h_weigh_remaining_precomputed(
    *,
    weigh_vehicle_nos: frozenset[str],
    weighed_vehicle_nos: set[str],
) -> int:
    return sum(1 for vehicle_no in weigh_vehicle_nos if vehicle_no not in weighed_vehicle_nos)


def _h_spot_evict(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    reverse_assignments = _reverse_spot_assignments(state.spot_assignments)
    evict_count = 0
    seen_wrong_occupants: set[str] = set()
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode == "SPOT":
            target_spot = goal.target_spot_code
            if target_spot is None:
                continue
            current_occupant = reverse_assignments.get(target_spot)
            if current_occupant is None or current_occupant == vehicle.vehicle_no:
                continue
            if current_occupant in seen_wrong_occupants:
                continue
            seen_wrong_occupants.add(current_occupant)
            evict_count += 1
        elif goal.target_area_code in WORK_AREA_SPOTS and goal.target_area_code != "大库:RANDOM":
            pass
    return evict_count


def _reverse_spot_assignments(spot_assignments: dict[str, str]) -> dict[str, str]:
    return {spot_code: vehicle_no for vehicle_no, spot_code in spot_assignments.items()}
