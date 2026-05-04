from __future__ import annotations

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


def exact_spot_clearance_bonus(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    move: HookAction,
    next_state: ReplayState,
) -> int:
    released_conflicts = _exact_spot_conflicts(plan_input, state) - _exact_spot_conflicts(
        plan_input,
        next_state,
    )
    bonus = len(released_conflicts)
    if bonus:
        return bonus
    if move.action_type != "DETACH" or move.source_track == move.target_track:
        return 0
    if not state.loco_carry or not set(move.vehicle_nos).issubset(state.loco_carry):
        return 0

    before_spot_owner = _spot_owner_by_code(state)
    after_spot_owner = _spot_owner_by_code(next_state)
    continuation_bonus = 0
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT":
            continue
        if goal.target_track != move.source_track or goal.target_spot_code is None:
            continue
        if vehicle.vehicle_no in move.vehicle_nos:
            continue
        if state.spot_assignments.get(vehicle.vehicle_no) == goal.target_spot_code:
            continue
        if (
            before_spot_owner.get(goal.target_spot_code) is None
            and after_spot_owner.get(goal.target_spot_code) is None
        ):
            continuation_bonus += 1
    return continuation_bonus


def exact_spot_reblock_penalty(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    move: HookAction,
    next_state: ReplayState,
) -> int:
    """Count unsatisfied exact-spot goals newly reblocked by this detach.

    During a spot-clearance chain, a random-depot or area vehicle may already
    be "good enough" on the target track while still occupying the exact spot
    needed by another vehicle. Reparking that blocker on the same target track
    should not outrank the staging move that keeps the exact spot open.
    """
    if move.action_type != "DETACH":
        return 0
    before = _exact_spot_conflicts(plan_input, state)
    after = _exact_spot_conflicts(plan_input, next_state)
    newly_blocked = after - before
    if not newly_blocked:
        return 0
    moved = set(move.vehicle_nos)
    return sum(1 for _spot_code, _seeker, occupant in newly_blocked if occupant in moved)


def exact_spot_same_track_repark_penalty(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    move: HookAction,
    next_state: ReplayState,
) -> int:
    """Count pending exact-spot goals interrupted by same-track reparking."""
    if move.action_type != "DETACH" or move.source_track != move.target_track:
        return 0
    if not state.loco_carry:
        return 0
    moved = set(move.vehicle_nos)
    penalty = 0
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT" or goal.target_spot_code is None:
            continue
        if goal.target_track != move.target_track:
            continue
        if state.spot_assignments.get(vehicle.vehicle_no) == goal.target_spot_code:
            continue
        if next_state.spot_assignments.get(vehicle.vehicle_no) == goal.target_spot_code:
            continue
        if vehicle.vehicle_no in moved:
            continue
        penalty += 1
    return penalty


def exact_spot_seeker_exposure_bonus(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    move: HookAction,
    next_state: ReplayState,
) -> int:
    """Count exact-spot target vehicles newly exposed at their source north end."""
    if move.action_type != "ATTACH":
        return 0
    source_seq = state.track_sequences.get(move.source_track, [])
    if source_seq[: len(move.vehicle_nos)] != list(move.vehicle_nos):
        return 0
    next_source_seq = next_state.track_sequences.get(move.source_track, [])
    if not next_source_seq:
        return 0

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    spot_owner = _spot_owner_by_code(next_state)
    bonus = 0
    for vehicle_no in next_source_seq[:1]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        goal = vehicle.goal
        if goal.target_mode != "SPOT" or goal.target_spot_code is None:
            continue
        if state.spot_assignments.get(vehicle_no) == goal.target_spot_code:
            continue
        owner = spot_owner.get(goal.target_spot_code)
        if owner is not None and owner != vehicle_no:
            continue
        bonus += 1
    return bonus


def _exact_spot_conflicts(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> set[tuple[str, str, str]]:
    spot_owner = _spot_owner_by_code(state)
    conflicts: set[tuple[str, str, str]] = set()
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT" or goal.target_spot_code is None:
            continue
        if state.spot_assignments.get(vehicle.vehicle_no) == goal.target_spot_code:
            continue
        occupant = spot_owner.get(goal.target_spot_code)
        if occupant is None or occupant == vehicle.vehicle_no:
            continue
        conflicts.add((goal.target_spot_code, vehicle.vehicle_no, occupant))
    return conflicts


def _spot_owner_by_code(state: ReplayState) -> dict[str, str]:
    return {
        spot_code: vehicle_no
        for vehicle_no, spot_code in state.spot_assignments.items()
    }
