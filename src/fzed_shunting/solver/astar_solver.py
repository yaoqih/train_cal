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
from fzed_shunting.solver.result import (
    PlanVerificationError,
    SolverResult,
    SolverTelemetry,
    emit_telemetry,
)
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
    enable_depot_late_scheduling: bool = True,
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
        enable_depot_late_scheduling=enable_depot_late_scheduling,
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
    enable_depot_late_scheduling: bool = True,
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
    phase_timings: dict[str, float] = {
        "constructive_ms": 0.0,
        "exact_ms": 0.0,
        "anytime_ms": 0.0,
        "lns_ms": 0.0,
        "verify_ms": 0.0,
    }

    # Stage 0: constructive baseline (always returns a plan — SLA safety net).
    constructive_seed: SolverResult | None = None
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode == "exact"
    ):
        _ts = perf_counter()
        constructive_seed = _run_constructive_stage(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        phase_timings["constructive_ms"] = (perf_counter() - _ts) * 1000

    # Stage 0.5: warm-start A* — if constructive stopped exactly 1 move from
    # goal (h=1), replay its plan and let a brief focused A* close the gap.
    # Limited to h=1 only: h≥2 states with evictions require expensive search
    # that wastes budget better spent in the anytime chain.
    if (
        enable_constructive_seed
        and constructive_seed is not None
        and constructive_seed.fallback_stage == "constructive_partial"
        and constructive_seed.plan
        and time_budget_ms is not None
    ):
        _ts = perf_counter()
        elapsed_so_far_ms = (perf_counter() - started_at) * 1000
        warm_budget = min(500.0, max(50.0, (time_budget_ms - elapsed_so_far_ms) * 0.05))
        warm_result = _try_warm_start_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            constructive_plan=constructive_seed.plan,
            master=master,
            time_budget_ms=warm_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            max_h=1,
        )
        if warm_result is not None:
            constructive_seed = warm_result
        phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    result: SolverResult
    if solver_mode == "lns":
        _ts = perf_counter()
        result = _solve_with_lns_result(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            debug_stats=debug_stats,
            repair_passes=4,
            solve_search_result=_solve_search_result,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000
    else:
        exact_time_budget_ms: float | None = time_budget_ms
        if (
            solver_mode == "exact"
            and enable_anytime_fallback
            and time_budget_ms is not None
        ):
            exact_time_budget_ms = max(1.0, time_budget_ms * 0.20)
        _ts = perf_counter()
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
                enable_depot_late_scheduling=enable_depot_late_scheduling,
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
        phase_timings["exact_ms"] = (perf_counter() - _ts) * 1000
        if solver_mode == "exact" and enable_anytime_fallback and not result.is_proven_optimal:
            _ts = perf_counter()
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
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            phase_timings["anytime_ms"] = (perf_counter() - _ts) * 1000
        if solver_mode == "beam" and beam_width is not None:
            _ts = perf_counter()
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
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
                if len(candidate.plan) >= len(improved.plan):
                    break
                improved = candidate
            result = improved
            phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000

    # If search produced nothing, fall back to constructive seed (which may be
    # a full-goal plan or a best-effort partial). Either way the SLA contract
    # is preserved: callers always get a non-empty plan unless the input is
    # intrinsically infeasible (pre-check already rejected those).
    if not result.plan and constructive_seed is not None and constructive_seed.plan:
        result = constructive_seed

    # Post-search LNS polish: on fallback-stage plans (non-exact), spend any
    # remaining budget compressing hook count via destroy/repair. Exact plans
    # are already proven optimal, so skip them. Worth doing when the plan is
    # long — the fixed per-pass cost amortises better on 50+ hook cases.
    # Skip ``solver_mode == "beam"`` because that path already has a dedicated
    # post-repair loop above with its own tuned budget.
    if (
        solver_mode != "beam"
        and result.plan
        and not result.is_proven_optimal
        and result.fallback_stage not in {"constructive_partial", "constructive"}
        and len(result.plan) > 40
    ):
        remaining_ms = None
        if time_budget_ms is not None:
            remaining_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        # Spend at most half of the remaining budget (or skip if < 2 sec left).
        if remaining_ms is None or remaining_ms > 2000:
            _ts = perf_counter()
            # Cap the polish pass at half the remaining budget to preserve
            # time for verify; on large plans this also prevents the inner
            # search calls from monopolising the wall clock.
            polish_budget_ms = None
            if remaining_ms is not None:
                polish_budget_ms = remaining_ms * 0.5
            polished = _improve_incumbent_result(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                incumbent=result,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width or 128,
                repair_passes=8,
                max_rounds=3,
                solve_search_result=_solve_search_result,
                time_budget_ms=polish_budget_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if len(polished.plan) < len(result.plan):
                result = polished
            phase_timings["lns_ms"] += (perf_counter() - _ts) * 1000

    # Goal-by-goal rescue was removed (W3-L): the root-cause fix in
    # _score_move prevents partial plans upstream, making post-hoc rescue
    # redundant.

    # Depot-late post-processing: only when the flag is on and we have a plan.
    # Reorders adjacent (depot, non-depot) hook pairs to push depot hooks
    # toward the tail, preserving replay-equivalent final state. On failure
    # (either an internal simulate rejection or a full-verifier rejection of
    # the reordered plan — e.g. intermediate route interference that
    # _apply_move doesn't detect) the plan is left unchanged.
    if enable_depot_late_scheduling and result.plan:
        from fzed_shunting.solver.depot_late import reorder_depot_late

        reordered_plan = reorder_depot_late(result.plan, initial_state, plan_input)
        if reordered_plan != result.plan:
            # Route-level verification guard: reorder_depot_late's internal
            # simulate only replays via _apply_move and compares terminal
            # state; it does not detect intermediate route interference that
            # the full plan_verifier catches. Run the verifier on the
            # candidate and fall back to the original plan on failure.
            candidate_is_safe = True
            if master is not None:
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
                    for index, move in enumerate(reordered_plan, start=1)
                ]
                try:
                    candidate_report = verify_plan(
                        master,
                        plan_input,
                        hook_plan,
                        initial_state_override=initial_state,
                    )
                    candidate_is_safe = bool(candidate_report.is_valid)
                except Exception:  # noqa: BLE001
                    candidate_is_safe = False
            if candidate_is_safe:
                result = replace(result, plan=reordered_plan)

    # Populate depot observability fields on every return path (flag on or off).
    from fzed_shunting.solver.depot_late import depot_earliness, is_depot_hook

    result = replace(
        result,
        depot_earliness=depot_earliness(result.plan),
        depot_hook_count=sum(1 for h in result.plan if is_depot_hook(h)),
    )

    if verify:
        _ts = perf_counter()
        result = _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=initial_state,
        )
        phase_timings["verify_ms"] = (perf_counter() - _ts) * 1000

    telemetry = _build_telemetry(
        plan_input=plan_input,
        result=result,
        phase_timings=phase_timings,
        total_ms=(perf_counter() - started_at) * 1000,
        time_budget_ms=time_budget_ms,
        node_budget=node_budget,
    )
    emit_telemetry(telemetry)
    return replace(result, telemetry=telemetry)


def _build_telemetry(
    *,
    plan_input: NormalizedPlanInput,
    result: SolverResult,
    phase_timings: dict[str, float],
    total_ms: float,
    time_budget_ms: float | None,
    node_budget: int | None,
) -> SolverTelemetry:
    weigh_count = sum(1 for v in plan_input.vehicles if v.need_weigh)
    close_door_count = sum(1 for v in plan_input.vehicles if v.is_close_door)
    spot_count = sum(1 for v in plan_input.vehicles if v.goal.target_mode == "SPOT")
    area_count = sum(1 for v in plan_input.vehicles if v.goal.target_area_code is not None)
    is_valid: bool | None = None
    if result.verification_report is not None:
        is_valid = bool(result.verification_report.is_valid)
    return SolverTelemetry(
        input_vehicle_count=len(plan_input.vehicles),
        input_track_count=len(plan_input.track_info),
        input_weigh_count=weigh_count,
        input_close_door_count=close_door_count,
        input_spot_count=spot_count,
        input_area_count=area_count,
        constructive_ms=phase_timings.get("constructive_ms", 0.0),
        exact_ms=phase_timings.get("exact_ms", 0.0),
        anytime_ms=phase_timings.get("anytime_ms", 0.0),
        lns_ms=phase_timings.get("lns_ms", 0.0),
        verify_ms=phase_timings.get("verify_ms", 0.0),
        total_ms=total_ms,
        plan_hook_count=len(result.plan),
        fallback_stage=result.fallback_stage,
        is_valid=is_valid,
        is_proven_optimal=result.is_proven_optimal,
        time_budget_ms=time_budget_ms,
        node_budget=node_budget,
    )


def _run_constructive_stage(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float | None,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult | None:
    """Run the priority-rule dispatcher and wrap the result as a SolverResult."""
    from fzed_shunting.solver.constructive import solve_constructive

    constructive_budget = None
    if time_budget_ms is not None:
        constructive_budget = min(8_000.0, max(500.0, time_budget_ms * 0.15))
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
    # Treat any plan derived from the constructive fallback (including
    # "+rescue"-annotated stages) as best-effort: it's already the last
    # resort and partial completeness is informative, not fatal.
    stage = result.fallback_stage or ""
    best_effort = stage.startswith("constructive_partial")
    if not report.is_valid and not best_effort:
        raise PlanVerificationError(report)
    return replace(result, verification_report=report)

def _heuristic(plan_input: NormalizedPlanInput, state: ReplayState) -> int:
    from fzed_shunting.solver.heuristic import compute_admissible_heuristic

    return compute_admissible_heuristic(plan_input, state)


def _try_warm_start_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    constructive_plan: list[HookAction],
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
    max_h: int = 4,
) -> SolverResult | None:
    """Replay constructive plan to get near-goal state, then run focused A* to finish."""
    from fzed_shunting.solver.heuristic import compute_admissible_heuristic
    from fzed_shunting.io.normalize_input import NormalizedVehicle

    vehicle_by_no: dict[str, NormalizedVehicle] = {v.vehicle_no: v for v in plan_input.vehicles}
    state = initial_state
    try:
        for move in constructive_plan:
            state = _apply_move(state=state, move=move, plan_input=plan_input, vehicle_by_no=vehicle_by_no)
    except Exception:  # noqa: BLE001
        return None

    h = compute_admissible_heuristic(plan_input, state)
    if h == 0 or h > max_h:
        return None

    try:
        completion = _solve_search_result(
            plan_input=plan_input,
            initial_state=state,
            master=master,
            solver_mode="exact",
            heuristic_weight=1.0,
            beam_width=None,
            budget=SearchBudget(time_budget_ms=time_budget_ms),
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
    except ValueError:
        return None
    if not completion.plan:
        return None

    return SolverResult(
        plan=list(constructive_plan) + list(completion.plan),
        expanded_nodes=completion.expanded_nodes,
        generated_nodes=completion.generated_nodes,
        closed_nodes=completion.closed_nodes,
        elapsed_ms=completion.elapsed_ms,
        is_proven_optimal=False,
        fallback_stage="constructive_warm_start",
        debug_stats=completion.debug_stats,
    )

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
