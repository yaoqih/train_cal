from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from fzed_shunting.domain.depot_spots import WORK_AREA_SPOTS
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import (
    goal_can_use_fallback_now,
    goal_effective_allowed_tracks,
    goal_is_satisfied,
)
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class HeuristicBreakdown:
    h_misplaced: int
    h_distinct_transfer_pairs: int
    h_blocking: int
    h_weigh: int
    h_spot_evict: int
    h_tight_capacity: int = 0

    @property
    def value(self) -> int:
        # h_tight_capacity is admissible ONLY when added to h_distinct_transfer_pairs
        # (evictions are a disjoint set of hooks from base transfers). The other
        # heuristics remain stand-alone candidates in the max() pool.
        return max(
            self.h_distinct_transfer_pairs + self.h_tight_capacity,
            self.h_blocking,
            self.h_weigh,
            self.h_spot_evict,
        )


def compute_admissible_heuristic(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> int:
    return compute_heuristic_breakdown(plan_input, state).value


def compute_admissible_heuristic_real_hook(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> int:
    """Admissible heuristic for the real-hook (ATTACH/DETACH) model.

    Each old PUT = 2 real hooks (1 ATTACH + 1 DETACH), so scale all existing
    lower bounds by 2.  Vehicles already in loco_carry only need DETACHes, so
    their contribution is cheaper: count the number of distinct goal target
    tracks among the carried vehicles that are not yet at their goal.
    """
    bd = compute_heuristic_breakdown(plan_input, state)
    h_on_tracks = max(
        2 * (bd.h_distinct_transfer_pairs + bd.h_tight_capacity),
        2 * bd.h_blocking,
        2 * bd.h_weigh,
        2 * bd.h_spot_evict,
    )
    h_carry = _h_carry_detach(plan_input, state)
    return h_on_tracks + h_carry


def _h_carry_detach(plan_input: NormalizedPlanInput, state: ReplayState) -> int:
    """Admissible lower bound on remaining DETACHes for vehicles in loco_carry.

    Greedy forward scan: maintain the intersection of allowed target tracks
    across the current DETACH prefix. When the intersection hits empty, a
    new DETACH group must start (increment count). Handles need_weigh vehicles
    by treating their effective goal as {"机库"} until weighed.

    Example: carry=[v1(T1), v2(T2), v3(T1)] → 3 groups (T1∩T2=∅, T2∩T1=∅).
    Previous distinct-targets count gave 2, which underestimated.
    """
    if not state.loco_carry:
        return 0
    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}
    groups = 0
    shared: set[str] = set()
    for vehicle_no in state.loco_carry:
        v = vehicle_by_no.get(vehicle_no)
        if v is None:
            groups += 1
            shared = set()
            continue
        if v.need_weigh and vehicle_no not in state.weighed_vehicle_nos:
            effective: set[str] = {"机库"}
        else:
            effective = set(v.goal.allowed_target_tracks)
        if not shared:
            groups += 1
            shared = effective.copy()
        else:
            new_shared = shared & effective
            if not new_shared:
                groups += 1
                shared = effective.copy()
            else:
                shared = new_shared
    return groups


def make_state_heuristic_real_hook(
    plan_input: NormalizedPlanInput,
) -> Callable[[ReplayState], int]:
    """Closure-style heuristic factory for the real-hook search mode.

    Atomic-hook cost model:
    - vehicles still on yard tracks need both ATTACH and DETACH work
    - vehicles already in loco_carry only need DETACH groups
    """
    vehicles = tuple(plan_input.vehicles)
    vehicle_by_no = {v.vehicle_no: v for v in vehicles}
    length_by_vno = {v.vehicle_no: v.vehicle_length for v in vehicles}
    weigh_vehicle_nos = frozenset(v.vehicle_no for v in vehicles if v.need_weigh)

    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation_by_track: dict[str, float] = {}
    for v in vehicles:
        initial_occupation_by_track[v.current_track] = (
            initial_occupation_by_track.get(v.current_track, 0.0) + v.vehicle_length
        )
    effective_cap_by_track = {
        name: max(cap, initial_occupation_by_track.get(name, 0.0))
        for name, cap in capacity_by_track.items()
    }

    def _heuristic(state: ReplayState) -> int:
        current_track_by_vehicle = _vehicle_track_lookup(state)
        h_pairs = _h_distinct_transfer_pairs(
            plan_input=plan_input,
            state=state,
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
        h_tight = _h_tight_capacity_eviction(
            plan_input=plan_input,
            state=state,
            length_by_vno=length_by_vno,
            current_track_by_vehicle=current_track_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            effective_cap_by_track=effective_cap_by_track,
        )
        h_on_tracks = max(
            2 * (h_pairs + h_tight),
            2 * h_block,
            2 * h_weigh,
            2 * h_spot,
        )
        h_carry = _h_carry_detach(plan_input, state)
        return h_on_tracks + h_carry

    return _heuristic


def compute_heuristic_breakdown(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> HeuristicBreakdown:
    current_track_by_vehicle = _vehicle_track_lookup(state)
    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}
    length_by_vno = {v.vehicle_no: v.vehicle_length for v in plan_input.vehicles}
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation_by_track: dict[str, float] = {}
    for v in plan_input.vehicles:
        initial_occupation_by_track[v.current_track] = (
            initial_occupation_by_track.get(v.current_track, 0.0) + v.vehicle_length
        )
    effective_cap_by_track = {
        name: max(cap, initial_occupation_by_track.get(name, 0.0))
        for name, cap in capacity_by_track.items()
    }

    return HeuristicBreakdown(
        h_misplaced=_h_misplaced_vehicles(
            plan_input=plan_input,
            state=state,
            current_track_by_vehicle=current_track_by_vehicle,
        ),
        h_distinct_transfer_pairs=_h_distinct_transfer_pairs(
            plan_input=plan_input,
            state=state,
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
        h_tight_capacity=_h_tight_capacity_eviction(
            plan_input=plan_input,
            state=state,
            length_by_vno=length_by_vno,
            current_track_by_vehicle=current_track_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            effective_cap_by_track=effective_cap_by_track,
        ),
    )


def make_state_heuristic(
    plan_input: NormalizedPlanInput,
) -> Callable[[ReplayState], int]:
    vehicles = tuple(plan_input.vehicles)
    vehicle_by_no = {v.vehicle_no: v for v in vehicles}
    length_by_vno = {v.vehicle_no: v.vehicle_length for v in vehicles}
    weigh_vehicle_nos = frozenset(v.vehicle_no for v in vehicles if v.need_weigh)
    close_4north_vehicle_nos = frozenset(
        v.vehicle_no
        for v in vehicles
        if v.is_close_door and "存4北" in v.goal.allowed_target_tracks
    )

    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation_by_track: dict[str, float] = {}
    for v in vehicles:
        initial_occupation_by_track[v.current_track] = (
            initial_occupation_by_track.get(v.current_track, 0.0) + v.vehicle_length
        )
    effective_cap_by_track = {
        name: max(cap, initial_occupation_by_track.get(name, 0.0))
        for name, cap in capacity_by_track.items()
    }

    def _heuristic(state: ReplayState) -> int:
        current_track_by_vehicle = _vehicle_track_lookup(state)
        h_pairs = _h_distinct_transfer_pairs(
            plan_input=plan_input,
            state=state,
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
        h_tight = _h_tight_capacity_eviction(
            plan_input=plan_input,
            state=state,
            length_by_vno=length_by_vno,
            current_track_by_vehicle=current_track_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            effective_cap_by_track=effective_cap_by_track,
        )
        _ = close_4north_vehicle_nos
        return max(h_pairs + h_tight, h_block, h_weigh, h_spot)

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
    state: ReplayState,
    current_track_by_vehicle: dict[str, str],
) -> int:
    count = 0
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None:
            count += 1
            continue
        if not goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            count += 1
            continue
        if _is_fallback_while_preferred_still_feasible(
            vehicle,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
            count += 1
    return count


def _h_distinct_transfer_pairs(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
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
        effective_allowed = goal_effective_allowed_tracks(
            vehicle,
            state=state,
            plan_input=plan_input,
        )
        allowed = effective_allowed or vehicle.goal.allowed_target_tracks
        if current_track in allowed and not _is_fallback_while_preferred_still_feasible(
            vehicle,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
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


def _is_fallback_while_preferred_still_feasible(
    vehicle: NormalizedVehicle,
    *,
    current_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if current_track not in vehicle.goal.fallback_target_tracks:
        return False
    if not vehicle.goal.preferred_target_tracks:
        return False
    return not goal_can_use_fallback_now(
        vehicle,
        state=state,
        plan_input=plan_input,
    )


def _h_blocking(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    current_track_by_vehicle: dict[str, str],
) -> int:
    """Admissible lower bound on hooks needed to clear blocked target tracks.

    Base term: 1 per target track whose near-end contains vehicles that must
    leave before the track can receive its intended vehicles.

    Cycle penalty: each strongly-connected component of size ≥ 2 in the
    blocker-destination graph requires exactly 1 extra temporary-parking hook
    (one vehicle must be staged elsewhere to break the mutual dependency).
    Adding one per SCC is admissible because no single hook can simultaneously
    resolve two independent cycles.
    """
    goal_tracks_needed: set[str] = set()
    for vehicle in plan_input.vehicles:
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None:
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ) and not _is_fallback_while_preferred_still_feasible(
            vehicle,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        for candidate in (
            goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
            )
            or vehicle.goal.allowed_target_tracks
        ):
            goal_tracks_needed.add(candidate)

    blocker_count = 0
    # blocker_dests[T] = goal tracks that near-end blockers at T want to reach
    blocker_dests: dict[str, set[str]] = {}
    for target_track in goal_tracks_needed:
        seq = state.track_sequences.get(target_track, [])
        if not seq:
            continue
        cluster_blockers = 0
        dests: set[str] = set()
        for vehicle_no in seq:
            blocker = vehicle_by_no.get(vehicle_no)
            if blocker is None:
                continue
            if goal_is_satisfied(
                blocker,
                track_name=target_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            cluster_blockers += 1
            dests.update(
                goal_effective_allowed_tracks(
                    blocker,
                    state=state,
                    plan_input=plan_input,
                )
                or blocker.goal.allowed_target_tracks
            )
        if cluster_blockers > 0:
            blocker_count += 1
            blocker_dests[target_track] = dests

    # Cycle penalty via SCC detection on the blocker-destination graph.
    # Edge T1→T2: T1 is blocked AND its blockers want to go to T2 (also blocked).
    blocked = set(blocker_dests.keys())
    if len(blocked) >= 2:
        graph: dict[str, set[str]] = {
            t: (blocker_dests[t] & blocked) - {t}
            for t in blocked
        }
        graph = {t: nbs for t, nbs in graph.items() if nbs}
        blocker_count += _count_nontrivial_sccs(graph)

    return blocker_count


def _count_nontrivial_sccs(graph: dict[str, set[str]]) -> int:
    """Count SCCs with ≥ 2 nodes using Tarjan's algorithm.

    Each such SCC is a set of mutually-blocking tracks that requires exactly
    one extra staging hook to resolve, regardless of cycle length.
    """
    nodes = list(graph.keys() | {v for vs in graph.values() for v in vs})
    if not nodes:
        return 0
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    counter = [0]
    result = [0]

    def strongconnect(v: str) -> None:
        index_of[v] = lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph.get(v, set()):
            if w not in index_of:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index_of[w])
        if lowlink[v] == index_of[v]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            if len(component) >= 2:
                result[0] += 1

    for v in nodes:
        if v not in index_of:
            strongconnect(v)
    return result[0]


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


def _h_tight_capacity_eviction(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    length_by_vno: dict[str, float],
    current_track_by_vehicle: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    effective_cap_by_track: dict[str, float],
) -> int:
    """Admissible lower bound on extra hooks forced by tight-capacity tracks.

    When pending arrivals at a track T exceed the available slack, one or more
    identity-goal vehicles (cars already at T AND whose allowed_target_tracks
    strictly equals {T}) must be temporarily evicted and later returned. Each
    evict-and-return costs 2 hooks.

    This term is admissible when ADDED to h_distinct_transfer_pairs: base
    transfer hooks bring pending vehicles in; eviction roundtrips are a
    DISJOINT set of hooks neither counted there nor in h_blocking, since they
    do not reach any goal target in the forced-pair sense (the evicted car
    eventually returns to the same track it started on).

    Strict equality on allowed_target_tracks avoids double-counting vehicles
    whose goal is multi-track (RANDOM depot, AREA, etc.) — they can't be
    "identity-goal" for a single track since they could dwell elsewhere.
    """
    # Group vehicles that are single-track targeted at exactly {T}
    target_demand_by_track: dict[str, float] = {}
    identity_goal_lens_by_track: dict[str, list[float]] = {}
    current_identity_goal_mass_by_track: dict[str, float] = {}

    for vehicle in plan_input.vehicles:
        allowed = tuple(vehicle.goal.allowed_target_tracks)
        if len(allowed) != 1:
            continue
        target = allowed[0]
        length = length_by_vno.get(vehicle.vehicle_no, vehicle.vehicle_length)
        target_demand_by_track[target] = target_demand_by_track.get(target, 0.0) + length
        current_track = current_track_by_vehicle.get(vehicle.vehicle_no)
        if current_track == target:
            current_identity_goal_mass_by_track[target] = (
                current_identity_goal_mass_by_track.get(target, 0.0) + length
            )
            identity_goal_lens_by_track.setdefault(target, []).append(length)

    # Current mass at each track from ReplayState, split into identity-goal
    # and non-identity-goal contributions. Non-identity-goal cars at T will
    # leave T at some point (their hook is already counted in
    # h_distinct_transfer_pairs or comes from multi-target routing). Their
    # departure creates slack for pending arrivals WITHOUT adding to the
    # eviction count. Identity-goal cars, by contrast, must end up back at T,
    # so if they must leave to make room, that's a +2-hook roundtrip we can
    # safely claim here.
    current_mass_by_track: dict[str, float] = {}
    current_nonidentity_mass_by_track: dict[str, float] = {}
    for track_name, seq in state.track_sequences.items():
        mass = 0.0
        nonid_mass = 0.0
        for vno in seq:
            v = vehicle_by_no.get(vno)
            if v is None:
                continue
            length = length_by_vno.get(vno, v.vehicle_length)
            mass += length
            allowed = tuple(v.goal.allowed_target_tracks)
            is_identity_at_this_track = len(allowed) == 1 and allowed[0] == track_name
            if not is_identity_at_this_track:
                nonid_mass += length
        current_mass_by_track[track_name] = mass
        current_nonidentity_mass_by_track[track_name] = nonid_mass

    total_eviction_hooks = 0
    eps = 1e-9
    for track, demand in target_demand_by_track.items():
        effective_cap = effective_cap_by_track.get(track)
        if effective_cap is None:
            # Unknown capacity (e.g. RANDOM depot track not in track_info) — skip.
            continue
        current_identity_mass = current_identity_goal_mass_by_track.get(track, 0.0)
        pending_arrivals_mass = demand - current_identity_mass
        if pending_arrivals_mass <= eps:
            continue
        current_mass = current_mass_by_track.get(track, 0.0)
        available_slack = effective_cap - current_mass
        # Non-identity cars at T will leave permanently, so their mass adds
        # to the slack budget without costing eviction hooks.
        free_slack = available_slack + current_nonidentity_mass_by_track.get(track, 0.0)
        overflow = pending_arrivals_mass - free_slack
        if overflow <= eps:
            continue
        ident_lens = identity_goal_lens_by_track.get(track, [])
        if not ident_lens:
            # Nothing available to evict — heuristic cannot claim a cost here
            # (the problem may be infeasible, but admissibility demands we
            # return 0 rather than an un-achievable lower bound).
            continue
        max_ident_len = max(ident_lens)
        if max_ident_len <= eps:
            continue
        # Ceil without float drift
        evictions = max(1, math.ceil((overflow - eps) / max_ident_len))
        # Can't evict more identity-goal cars than exist at this track
        evictions = min(evictions, len(ident_lens))
        total_eviction_hooks += evictions * 2
    return total_eviction_hooks
