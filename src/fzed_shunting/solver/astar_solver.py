from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.anytime import (
    _remaining_budget_ms,
    _remaining_budget_nodes,
    _run_anytime_fallback_chain as _anytime_run_fallback_chain,
)
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.lns import (
    _improve_incumbent_result,
    _solve_with_lns_result,
)
from fzed_shunting.solver.result import PlanVerificationError, SolverResult
from fzed_shunting.solver.search import (
    BEAM_SHALLOW_RESERVE,
    QueueItem,
    _accumulate_move_debug_stats,
    _blocking_goal_target_bonus,
    _initialize_debug_stats,
    _merge_counter_dict,
    _priority,
    _prune_queue,
    _solve_search_result,
)
from fzed_shunting.solver.state import (
    _apply_move,
    _canonical_random_depot_vehicle_nos,
    _is_goal,
    _locate_vehicle,
    _state_key,
    _vehicle_track_lookup,
)
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

BEAM_POST_REPAIR_PASSES = 1
BEAM_POST_REPAIR_MAX_ROUNDS = 1


def solve_with_simple_astar(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    solver_mode: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    debug_stats: dict[str, Any] | None = None,
    time_budget_ms: float | None = None,
    node_budget: int | None = None,
    verify: bool = True,
    enable_anytime_fallback: bool = True,
    enable_constructive_seed: bool = True,
) -> list[HookAction]:
    return solve_with_simple_astar_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        debug_stats=debug_stats,
        time_budget_ms=time_budget_ms,
        node_budget=node_budget,
        verify=verify,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_constructive_seed=enable_constructive_seed,
    ).plan


def solve_with_simple_astar_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    solver_mode: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    debug_stats: dict[str, Any] | None = None,
    time_budget_ms: float | None = None,
    node_budget: int | None = None,
    verify: bool = True,
    enable_anytime_fallback: bool = True,
    enable_constructive_seed: bool = True,
) -> SolverResult:
    _validate_solver_options(
        solver_mode=solver_mode,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
    )
    if verify and master is None:
        raise ValueError("verify=True requires master to be provided for plan_verifier")
    _validate_final_track_goal_capacities(plan_input)
    started_at = perf_counter()

    # Stage 0: constructive baseline (always returns a plan — SLA safety net).
    constructive_seed: SolverResult | None = None
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode == "exact"
    ):
        constructive_seed = _run_constructive_stage(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=time_budget_ms,
        )

    result: SolverResult
    if solver_mode == "lns":
        result = _solve_with_lns_result(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            debug_stats=debug_stats,
            repair_passes=4,
            solve_search_result=_solve_search_result,
        )
    else:
        exact_time_budget_ms: float | None = time_budget_ms
        if (
            solver_mode == "exact"
            and enable_anytime_fallback
            and time_budget_ms is not None
        ):
            exact_time_budget_ms = max(1.0, time_budget_ms * 0.4)
        try:
            result = _solve_search_result(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                solver_mode=solver_mode,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
                debug_stats=debug_stats,
                budget=SearchBudget(time_budget_ms=exact_time_budget_ms, node_budget=node_budget),
            )
        except ValueError:
            if constructive_seed is None:
                # No seed to fall back on; preserve the legacy "no solution" signal.
                raise
            # Constructive seed guarantees SLA — suppress raise, continue chain.
            result = SolverResult(
                plan=[],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_proven_optimal=False,
                fallback_stage=solver_mode,
                debug_stats=debug_stats,
            )
        if solver_mode == "exact" and enable_anytime_fallback and not result.is_proven_optimal:
            result = _anytime_run_fallback_chain(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                incumbent=result,
                started_at=started_at,
                time_budget_ms=time_budget_ms,
                node_budget=node_budget,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
                debug_stats=debug_stats,
                solve_search_result=_solve_search_result,
            )
        if solver_mode == "beam" and beam_width is not None:
            improved = result
            for _ in range(BEAM_POST_REPAIR_MAX_ROUNDS):
                candidate = _improve_incumbent_result(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    incumbent=improved,
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    repair_passes=BEAM_POST_REPAIR_PASSES,
                    max_rounds=1,
                    solve_search_result=_solve_search_result,
                )
                if len(candidate.plan) >= len(improved.plan):
                    break
                improved = candidate
            result = improved

    # If search produced nothing, fall back to constructive seed (which may be
    # a full-goal plan or a best-effort partial). Either way the SLA contract
    # is preserved: callers always get a non-empty plan unless the input is
    # intrinsically infeasible (pre-check already rejected those).
    if not result.plan and constructive_seed is not None and constructive_seed.plan:
        result = constructive_seed

    if verify:
        result = _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=initial_state,
        )
    return result


def _run_constructive_stage(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float | None,
) -> SolverResult | None:
    """Run the priority-rule dispatcher and wrap the result as a SolverResult."""
    from fzed_shunting.solver.constructive import solve_constructive

    constructive_budget = None
    if time_budget_ms is not None:
        constructive_budget = min(5_000.0, max(200.0, time_budget_ms * 0.05))
    try:
        ctr = solve_constructive(
            plan_input,
            initial_state,
            master=master,
            max_iterations=1500,
            time_budget_ms=constructive_budget,
        )
    except Exception:  # noqa: BLE001
        return None
    if not ctr.plan:
        return None
    return SolverResult(
        plan=list(ctr.plan),
        expanded_nodes=ctr.iterations,
        generated_nodes=ctr.iterations,
        closed_nodes=0,
        elapsed_ms=ctr.elapsed_ms,
        is_proven_optimal=False,
        fallback_stage="constructive" if ctr.reached_goal else "constructive_partial",
        debug_stats=ctr.debug_stats,
    )


def _attach_verification(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    master: MasterData | None,
    initial_state: ReplayState | None = None,
) -> SolverResult:
    if master is None:
        return result
    if not result.plan:
        return result
    from fzed_shunting.verify.plan_verifier import verify_plan

    hook_plan = [
        {
            "hookNo": index,
            "actionType": move.action_type,
            "sourceTrack": move.source_track,
            "targetTrack": move.target_track,
            "vehicleNos": list(move.vehicle_nos),
            "pathTracks": list(move.path_tracks),
        }
        for index, move in enumerate(result.plan, start=1)
    ]
    report = verify_plan(master, plan_input, hook_plan, initial_state_override=initial_state)
    best_effort = result.fallback_stage == "constructive_partial"
    if not report.is_valid and not best_effort:
        raise PlanVerificationError(report)
    return replace(result, verification_report=report)

def _heuristic(plan_input: NormalizedPlanInput, state: ReplayState) -> int:
    from fzed_shunting.solver.heuristic import compute_admissible_heuristic

    return compute_admissible_heuristic(plan_input, state)

def _validate_final_track_goal_capacities(plan_input: NormalizedPlanInput) -> None:
    capacity_by_track = {
        info.track_name: info.track_distance
        for info in plan_input.track_info
    }
    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0) + vehicle.vehicle_length
        )
    final_length_by_track: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        if vehicle.goal.target_mode != "TRACK":
            continue
        final_length_by_track[vehicle.goal.target_track] = (
            final_length_by_track.get(vehicle.goal.target_track, 0.0) + vehicle.vehicle_length
        )
    for track_name, total_length in final_length_by_track.items():
        capacity = capacity_by_track.get(track_name)
        if capacity is None:
            raise ValueError(f"Missing capacity for final arrangement track: {track_name}")
        effective_capacity = max(capacity, initial_occupation_by_track.get(track_name, 0.0))
        if total_length > effective_capacity + 1e-9:
            raise ValueError(
                f"final arrangement exceeds track capacity: {track_name} "
                f"requires {total_length:.1f}m but capacity is {capacity:.1f}m"
            )


def _validate_solver_options(
    *,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
) -> None:
    if solver_mode not in {"exact", "weighted", "beam", "lns"}:
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be >= 1.0")
    if beam_width is not None and beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    if solver_mode == "beam" and beam_width is None:
        raise ValueError("beam_width is required when solver_mode=beam")



# Backward-compat re-exports; prefer importing from the submodules directly.
from fzed_shunting.solver.lns import (  # noqa: E402,F401
    _build_repair_plan_input,
    _candidate_repair_cut_points,
    _cut_points_equidistant,
    _cut_points_hotspot,
    _cut_points_target_cluster,
    _cut_points_worst_cost,
    _is_better_plan,
    _plan_quality,
    _repair_cut_score,
    _spot_goal_satisfied,
    _to_hook_dict,
)
