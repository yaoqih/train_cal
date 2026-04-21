"""Goal-by-goal rescue: targeted repair for vehicles that the main solver
left off-target. Runs after the main solver as an optional final pass.

When the full multi-vehicle multi-goal problem times out, the constructive
fallback may produce a plan with only a handful of vehicles off-target. Each
such vehicle is typically a small, local sub-problem that solves in 1-2
seconds in isolation. This module synthesises per-vehicle sub-problems where
only the misplaced vehicle has a real goal (all other vehicles' goals are
"stay where you are") and runs a short exact A* on each.
"""

from __future__ import annotations

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import (
    GoalSpec,
    NormalizedPlanInput,
    NormalizedVehicle,
)
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.state import _apply_move, _vehicle_track_lookup
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


def _misplaced_vehicles(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> list[str]:
    """Return vehicle_nos whose current track is NOT in their allowed targets."""
    current_by_vehicle = _vehicle_track_lookup(state)
    misplaced: list[str] = []
    for vehicle in plan_input.vehicles:
        current = current_by_vehicle.get(vehicle.vehicle_no)
        if current not in vehicle.goal.allowed_target_tracks:
            misplaced.append(vehicle.vehicle_no)
    return misplaced


def attempt_goal_rescue(
    *,
    plan_input: NormalizedPlanInput,
    current_plan: list[HookAction],
    terminal_state: ReplayState,
    initial_state: ReplayState,
    master: MasterData | None,
    per_vehicle_budget_ms: float = 2000.0,
) -> tuple[list[HookAction], ReplayState, list[str]]:
    """Attempt to rescue each misplaced vehicle via a focused sub-A*.

    For each misplaced vehicle V, construct a synthetic NormalizedPlanInput
    where only V has its real goal; all other vehicles' goals are frozen to
    "stay where you currently are" (the sub-solver treats them as achieved).
    Run an exact A* with a short per-vehicle budget. If a sub-plan is found
    that lands V at its real goal, append it to the main plan.

    Args:
        plan_input: The real (top-level) plan input with all vehicles' goals.
        current_plan: The main solver's plan so far (will be extended).
        terminal_state: The state after replaying ``current_plan`` from init.
        initial_state: The true initial state (unused by this call; kept for
            symmetry in case future variants re-replay).
        master: Master data required for move generation (RouteOracle).
        per_vehicle_budget_ms: Per-vehicle time budget for the sub-A*.

    Returns:
        Tuple of:
            - Extended plan (original + any appended rescue moves).
            - New terminal state after applying the rescue moves.
            - List of vehicle_nos that could NOT be rescued.
    """
    # Import here to avoid circular import (search imports from state, which
    # doesn't import rescue, but astar_solver imports search).
    from fzed_shunting.solver.search import _solve_search_result

    plan = list(current_plan)
    state = terminal_state

    misplaced = _misplaced_vehicles(plan_input, state)
    if not misplaced:
        return plan, state, []

    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}
    still_misplaced: list[str] = []

    for vno in misplaced:
        target_vehicle = vehicle_by_no.get(vno)
        if target_vehicle is None:
            still_misplaced.append(vno)
            continue

        # Build sub-problem: only THIS vehicle has a real goal; all others
        # have their goal frozen to their current track.
        current_by_vehicle = _vehicle_track_lookup(state)
        sub_vehicles: list[NormalizedVehicle] = []
        for v in plan_input.vehicles:
            current = current_by_vehicle.get(v.vehicle_no, v.current_track)
            if v.vehicle_no == vno:
                # Keep the real goal; just update current_track so the
                # sub-input reflects the post-main-plan position.
                sub_v = v.model_copy(update={"current_track": current})
            else:
                # Freeze goal to current track. Drop need_weigh if already
                # weighed in the current state so the sub-problem doesn't
                # re-weigh.
                frozen_goal = GoalSpec(
                    target_mode="TRACK",
                    target_track=current,
                    allowed_target_tracks=[current],
                    target_area_code=None,
                    target_spot_code=None,
                )
                sub_v = v.model_copy(update={
                    "current_track": current,
                    "goal": frozen_goal,
                    "need_weigh": (
                        v.need_weigh
                        and v.vehicle_no not in state.weighed_vehicle_nos
                    ),
                    "is_close_door": False,
                })
            sub_vehicles.append(sub_v)

        sub_plan_input = plan_input.model_copy(update={"vehicles": sub_vehicles})

        try:
            sub_budget = SearchBudget(time_budget_ms=per_vehicle_budget_ms)
            sub_result = _solve_search_result(
                plan_input=sub_plan_input,
                initial_state=state,
                master=master,
                solver_mode="exact",
                heuristic_weight=1.0,
                beam_width=None,
                debug_stats=None,
                budget=sub_budget,
                enable_depot_late_scheduling=False,
            )
        except (ValueError, KeyError):
            still_misplaced.append(vno)
            continue

        if not sub_result.plan:
            still_misplaced.append(vno)
            continue

        # Validate: applying sub_result.plan to the current state should land
        # `vno` at its real goal. Replay against the REAL plan_input so spot
        # allocations follow real goals (not the frozen sub goals).
        new_state = state
        replay_ok = True
        try:
            for move in sub_result.plan:
                new_state = _apply_move(
                    state=new_state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
        except (ValueError, KeyError):
            replay_ok = False

        if not replay_ok:
            still_misplaced.append(vno)
            continue

        # Confirm target vehicle is at its real goal track.
        new_current = _vehicle_track_lookup(new_state).get(vno)
        if new_current not in target_vehicle.goal.allowed_target_tracks:
            still_misplaced.append(vno)
            continue

        # Accept the rescue.
        plan.extend(sub_result.plan)
        state = new_state

    # Recompute final misplaced set against the true plan_input.
    final_still = _misplaced_vehicles(plan_input, state)
    return plan, state, final_still
