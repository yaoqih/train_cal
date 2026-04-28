from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.astar_solver import (
    RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
)
from fzed_shunting.solver.profile import (
    validation_recovery_should_continue_after_success,
    validation_retry_beam_widths,
    validation_retry_time_budget_ms,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.verify.replay import ReplayState


SolveResultFn = Callable[..., SolverResult]


def solve_with_validation_recovery_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    *,
    master: MasterData | None,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None,
    solve_result_fn: SolveResultFn,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult:
    """Run the validation-profile solve plus the same beam retry policy.

    The external validation runner retries no-solution beam cases with a larger
    time budget and wider beams. Demo/CLI paths must use the same policy;
    otherwise a valid production scenario can appear as a web-only failure.
    """
    initial = solve_result_fn(
        plan_input,
        initial_state,
        master=master,
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if solver_mode != "beam" or beam_width is None:
        return initial
    if initial.is_complete and not _should_continue_recovery_after_success(initial):
        return initial

    best_partial: SolverResult | None = None if initial.is_complete else initial
    best_complete: SolverResult | None = initial if initial.is_complete else None
    retry_budget_ms = validation_retry_time_budget_ms(time_budget_ms)
    for retry_beam_width in validation_retry_beam_widths(beam_width=beam_width):
        candidate = solve_result_fn(
            plan_input,
            initial_state,
            master=master,
            solver_mode=solver_mode,
            heuristic_weight=heuristic_weight,
            beam_width=retry_beam_width,
            time_budget_ms=retry_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            near_goal_partial_resume_max_final_heuristic=(
                RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
            ),
        )
        if candidate.is_complete:
            if best_complete is None or len(candidate.plan) < len(best_complete.plan):
                stats: dict[str, Any] = dict(candidate.debug_stats or {})
                stats["validation_recovery"] = {
                    "recovery_beam_width": retry_beam_width,
                    "recovery_time_budget_ms": retry_budget_ms,
                }
                best_complete = replace(candidate, debug_stats=stats)
            if not _should_continue_recovery_after_success(candidate):
                break
            continue
        if best_partial is None or len(candidate.partial_plan) > len(best_partial.partial_plan):
            best_partial = candidate
    if best_complete is not None:
        return best_complete
    if best_partial is not None:
        return best_partial
    return initial


def _should_continue_recovery_after_success(result: SolverResult) -> bool:
    shape = (result.debug_stats or {}).get("plan_shape_metrics") or {}
    return validation_recovery_should_continue_after_success(
        hook_count=len(result.plan),
        max_vehicle_touch_count=_as_int(shape.get("max_vehicle_touch_count")),
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
