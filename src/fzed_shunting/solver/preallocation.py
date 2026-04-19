"""Pre-allocation of 大库:RANDOM depot targets to reduce search branching.

When a plan has many vehicles with ``target_area_code='大库:RANDOM'`` (i.e.
allowed to land in any of 修1/2/3/4库内), the search tree branches 4x per
vehicle at every decision point. For yards with 20+ random-depot vehicles,
this state-space explosion can dwarf the batching savings (external case
20260205W: 49 moves → 203 hooks because search couldn't batch paths).

This module narrows each random-depot vehicle's goal to a single specific
``修X库内`` before search starts via a load-balancing greedy:

1. Measure current occupancy (by vehicle length) of each 修X库内 in the
   initial state.
2. Respect per-track capacity (track_distance - already-occupied meters).
3. Place random-depot vehicles one at a time, each into the depot with
   the smallest post-assignment load (ties broken by path length).

Result: each random-depot vehicle has ``allowed_target_tracks=[chosen]``,
turning the 4-way choice into a single target. Spot allocation inside the
chosen depot remains dynamic (``target_area_code`` stays ``"大库:RANDOM"``).

## Trade-off — why this is NOT default-on

Empirically (external 109 benchmark, 180s budget):

- Extreme outlier (205W: 49 moves, 20 random-depot): max hooks
  dropped from 203 → ~25 (**-88%**). Huge win on this single case.
- Median random-depot cases (typically 5-10 random seekers): frequently
  become infeasible within budget. The narrowed path may force the
  search through awkward routing that breaks batchable sequences.

The heuristic narrowing is correct-by-construction (spot allocation still
succeeds) but the search's path/batching choices are significantly worse
on cases that had been exploiting random flexibility.

Callers with a known-large random-depot workload can opt in via
``enable_random_depot_preallocation=True``. Default is ``False`` to avoid
breaking the common case.

If the allocation would exceed any depot's capacity, the helper bails out
and returns the original input unmodified.
"""

from __future__ import annotations

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.verify.replay import ReplayState

DEPOT_INNER_TRACKS: tuple[str, ...] = ("修1库内", "修2库内", "修3库内", "修4库内")
# Threshold below which preallocation is skipped. When there are only a few
# random-depot vehicles, the search's branching factor is already manageable
# and the flexibility of multi-track goals often helps find batch moves.
# Above this threshold the 4^N branching dominates and narrowing helps.
MIN_RANDOM_DEPOT_VEHICLES_FOR_PREALLOCATION = 8


def preallocate_random_depot_targets(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
) -> NormalizedPlanInput:
    """Assign each ``大库:RANDOM`` vehicle to a specific 修X库内 track.

    Non-random-depot vehicles and inputs with no capacity pressure are
    returned unchanged. If the allocation is infeasible (over-capacity or
    missing depot tracks), the helper returns the original input so the
    downstream solver is free to use its full search.
    """
    candidates = [
        (idx, vehicle)
        for idx, vehicle in enumerate(plan_input.vehicles)
        if vehicle.goal.target_area_code == "大库:RANDOM"
        and len(vehicle.goal.allowed_target_tracks) > 1
    ]
    if len(candidates) < MIN_RANDOM_DEPOT_VEHICLES_FOR_PREALLOCATION:
        return plan_input

    length_by_vehicle = {v.vehicle_no: v.vehicle_length for v in plan_input.vehicles}
    capacity_by_track: dict[str, float] = {}
    for info in plan_input.track_info:
        capacity_by_track[info.track_name] = info.track_distance

    depot_tracks = [t for t in DEPOT_INNER_TRACKS if t in capacity_by_track]
    if not depot_tracks:
        return plan_input

    load_by_depot: dict[str, float] = {t: 0.0 for t in depot_tracks}
    for depot in depot_tracks:
        for vehicle_no in initial_state.track_sequences.get(depot, []):
            load_by_depot[depot] += length_by_vehicle.get(vehicle_no, 0.0)

    route_oracle = RouteOracle(master) if master is not None else None

    assignments: dict[int, str] = {}
    for idx, vehicle in candidates:
        depot_options = [t for t in depot_tracks if t in vehicle.goal.allowed_target_tracks]
        if not depot_options:
            return plan_input
        best_depot: str | None = None
        best_key: tuple[float, float] | None = None
        for depot in depot_options:
            projected_load = load_by_depot[depot] + vehicle.vehicle_length
            if projected_load > capacity_by_track[depot]:
                continue
            route_dist = _route_distance_hint(
                route_oracle=route_oracle,
                source_track=vehicle.current_track,
                target_track=depot,
            )
            key = (projected_load, route_dist)
            if best_key is None or key < best_key:
                best_depot = depot
                best_key = key
        if best_depot is None:
            # Over-capacity: bail out, let the regular solver handle it.
            return plan_input
        assignments[idx] = best_depot
        load_by_depot[best_depot] += vehicle.vehicle_length

    new_vehicles: list[NormalizedVehicle] = []
    for idx, vehicle in enumerate(plan_input.vehicles):
        if idx not in assignments:
            new_vehicles.append(vehicle)
            continue
        chosen = assignments[idx]
        new_vehicles.append(
            vehicle.model_copy(
                update={
                    "goal": vehicle.goal.model_copy(
                        update={
                            "target_track": chosen,
                            "allowed_target_tracks": [chosen],
                        }
                    ),
                }
            )
        )
    return plan_input.model_copy(update={"vehicles": new_vehicles})


def _route_distance_hint(
    *,
    route_oracle: RouteOracle | None,
    source_track: str,
    target_track: str,
) -> float:
    """Approximate path length used as a tie-breaker when loads are equal.

    Returns ``0.0`` when no oracle is available or the pair is unreachable —
    the function is purely a tie-breaker, not a feasibility check.
    """
    if route_oracle is None:
        return 0.0
    try:
        route = route_oracle.resolve_route(source_track, target_track)
    except Exception:  # noqa: BLE001
        return 0.0
    if route is None:
        return 0.0
    return route.total_length_m
