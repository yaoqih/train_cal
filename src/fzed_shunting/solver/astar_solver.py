from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.anytime import (
    _remaining_budget_ms,
    _remaining_budget_nodes,
    _run_anytime_fallback_chain as _anytime_run_fallback_chain,
)
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.lns import (
    _build_repair_plan_input,
    _improve_incumbent_result,
    _solve_with_lns_result,
)
from fzed_shunting.solver.purity import compute_state_purity
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
from fzed_shunting.solver.structural_metrics import (
    compute_structural_metrics,
    summarize_plan_shape,
)
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

BEAM_POST_REPAIR_PASSES = 1
BEAM_POST_REPAIR_MAX_ROUNDS = 1
LOCALIZED_RESUME_MAX_UNFINISHED = 16
PARTIAL_RESUME_MAX_CHECKPOINTS = 6
PARTIAL_ROUTE_RELEASE_MAX_CHECKPOINTS = 3
BEAM_COMPLETE_SEED_MIN_BUDGET_RATIO = 0.20
BEAM_COMPLETE_SEED_FULL_BUDGET_HOOKS = 100
BEAM_INCOMPLETE_SEED_PRIMARY_MIN_BUDGET_MS = 35_000.0
RELAXED_CONSTRUCTIVE_RETRY_BUDGET_MS = 8_000.0
ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS = 20_000.0
PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS = 30_000.0
EARLY_ROUTE_RELEASE_COMPLETION_BUDGET_MS = 12_000.0
EARLY_ROUTE_RELEASE_MIN_REMAINING_MS = 25_000.0
LOCALIZED_RESUME_BUDGET_RATIO = 0.50
LOCALIZED_RESUME_MIN_FULL_BEAM_MS = 500.0
DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 4
RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 10
RELAXED_RESCUE_MAX_FINAL_HEURISTIC = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
NEAR_GOAL_PARTIAL_RESUME_MAX_BUDGET_MS = 60_000.0
NEAR_GOAL_PARTIAL_RESUME_BUDGET_RATIO = 2.0 / 3.0
MIN_CHILD_STAGE_BUDGET_MS = 1.0
ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS = 12
ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS = 250.0


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
    near_goal_partial_resume_max_final_heuristic: int = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
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
        near_goal_partial_resume_max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
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
    near_goal_partial_resume_max_final_heuristic: int = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
) -> SolverResult:
    solver_mode = _normalize_solver_mode(solver_mode)
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
    optimize_depot_late_in_search = (
        enable_depot_late_scheduling
        and time_budget_ms is None
        and node_budget is None
    )
    reserve_primary_for_partial_rescue = (
        near_goal_partial_resume_max_final_heuristic
        <= DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )
    allow_deep_pre_primary_route_release = not reserve_primary_for_partial_rescue
    attempted_resume_partial_keys: set[tuple] = set()
    attempted_route_tail_partial_keys: set[tuple] = set()

    # Stage 0: constructive baseline (always returns a plan — SLA safety net).
    # Run for both "exact" and "beam" so beam search has a fallback when the
    # LIFO-aware state space makes beam exploration fail to find a solution.
    constructive_seed: SolverResult | None = None
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode in {"exact", "beam"}
    ):
        _ts = perf_counter()
        constructive_seed = _run_constructive_stage(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=optimize_depot_late_in_search,
        )
        phase_timings["constructive_ms"] = (perf_counter() - _ts) * 1000

    # Stage 0.5: warm-start A* — if constructive stopped exactly 1 move from
    # goal (h=1), replay its plan and let a brief focused A* close the gap.
    # Limited to h=1 only: h≥2 states with evictions require expensive search
    # that wastes budget better spent in the anytime chain.
    if (
        enable_constructive_seed
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
        if remaining_ms is not None and remaining_ms > MIN_CHILD_STAGE_BUDGET_MS:
            _ts = perf_counter()
            warm_budget = _cap_pre_primary_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                min(500.0, max(50.0, remaining_ms * 0.05)),
                solver_mode=solver_mode,
                reserve_primary=reserve_primary_for_partial_rescue,
            )
            if warm_budget is not None and warm_budget > 0:
                warm_result = _try_warm_start_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    constructive_plan=constructive_seed.partial_plan,
                    master=master,
                    time_budget_ms=warm_budget,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                    max_h=1,
                )
                if warm_result is not None:
                    constructive_seed = warm_result
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    # Stage 0.6: partial-resume search. If constructive reached a later
    # empty-carry checkpoint but still stopped short of goal, continue search
    # from that honest native state instead of always restarting from scratch.
    if (
        enable_constructive_seed
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
        if remaining_ms is not None and remaining_ms > MIN_CHILD_STAGE_BUDGET_MS:
            _ts = perf_counter()
            route_blockage_pressure = _partial_route_blockage_pressure(
                constructive_seed.partial_plan,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            if (
                route_blockage_pressure > 0
                and remaining_ms >= EARLY_ROUTE_RELEASE_MIN_REMAINING_MS
                and (_solver_result_final_heuristic(constructive_seed) or 0)
                <= near_goal_partial_resume_max_final_heuristic
                and reserve_primary_for_partial_rescue
            ):
                route_tail_signature = _partial_plan_signature(constructive_seed.partial_plan)
                attempted_route_tail_partial_keys.add(route_tail_signature)
                route_tail_budget = _cap_child_stage_budget_ms(
                    started_at,
                    time_budget_ms,
                    _route_release_tail_budget_ms(
                        remaining_budget_ms=remaining_ms,
                        solver_mode=solver_mode,
                        reserve_primary=reserve_primary_for_partial_rescue,
                    ),
                )
                if route_tail_budget is not None and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS:
                    route_tail_completion = _try_route_release_partial_completion(
                        plan_input=plan_input,
                        initial_state=initial_state,
                        partial_plan=constructive_seed.partial_plan,
                        master=master,
                        time_budget_ms=route_tail_budget,
                    )
                    if route_tail_completion is not None:
                        constructive_seed = _shorter_complete_result(
                            constructive_seed,
                            route_tail_completion,
                            plan_input=plan_input,
                            initial_state=initial_state,
                            master=master,
                        )
                remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
            resume_budget = _partial_resume_budget_ms(
                constructive_seed,
                remaining_budget_ms=remaining_ms or 0.0,
                max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
            )
            if (
                route_blockage_pressure > 0
                and remaining_ms is not None
                and remaining_ms > PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS + 1_000.0
            ):
                reserve_budget = min(
                    PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
                    max(5_000.0, remaining_ms * 0.25),
                )
                resume_budget = min(
                    resume_budget,
                    max(
                        MIN_CHILD_STAGE_BUDGET_MS,
                        remaining_ms - reserve_budget,
                    ),
                )
            resume_budget = _cap_pre_primary_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                resume_budget,
                solver_mode=solver_mode,
                reserve_primary=reserve_primary_for_partial_rescue,
            )
            if (
                not constructive_seed.is_complete
                and resume_budget is not None
                and resume_budget > 0
                and route_blockage_pressure <= 0
            ):
                attempted_resume_partial_keys.add(
                    _partial_plan_signature(constructive_seed.partial_plan)
                )
                resume_result = _try_resume_partial_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    constructive_plan=constructive_seed.partial_plan,
                    master=master,
                    time_budget_ms=resume_budget,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                )
                if resume_result is not None:
                    constructive_seed = resume_result
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode in {"exact", "beam"}
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        route_blockage_pressure = _partial_route_blockage_pressure(
            constructive_seed.partial_plan,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        if route_blockage_pressure > 0:
            route_tail_signature = _partial_plan_signature(constructive_seed.partial_plan)
            route_tail_budget = _cap_pre_primary_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
                solver_mode=solver_mode,
                reserve_primary=reserve_primary_for_partial_rescue,
            )
            if (
                allow_deep_pre_primary_route_release
                and route_tail_budget is not None
                and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS
            ):
                if route_tail_signature not in attempted_route_tail_partial_keys:
                    attempted_route_tail_partial_keys.add(route_tail_signature)
                    _ts = perf_counter()
                    route_tail_completion = _try_route_release_partial_completion(
                        plan_input=plan_input,
                        initial_state=initial_state,
                        partial_plan=constructive_seed.partial_plan,
                        master=master,
                        time_budget_ms=route_tail_budget,
                    )
                    if route_tail_completion is not None:
                        constructive_seed = _shorter_complete_result(
                            constructive_seed,
                            route_tail_completion,
                            plan_input=plan_input,
                            initial_state=initial_state,
                            master=master,
                        )
                    phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    # Stage 0.7: relaxed constructive rescue.  Strict anti-oscillation is a
    # better default for hook count, but some yard states need a late re-grab
    # from staging before a short tail search can close the plan.  Run this
    # before primary beam consumes the full budget; otherwise the rescue has no
    # time left precisely on hard partial cases.
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode in {"exact", "beam"}
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        relaxed_budget = _cap_pre_primary_child_stage_budget_ms(
            started_at,
            time_budget_ms,
            RELAXED_CONSTRUCTIVE_RETRY_BUDGET_MS,
            solver_mode=solver_mode,
            reserve_primary=reserve_primary_for_partial_rescue,
        )
        if relaxed_budget is not None and relaxed_budget > MIN_CHILD_STAGE_BUDGET_MS:
            _ts = perf_counter()
            relaxed_seed = _run_constructive_stage(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                time_budget_ms=relaxed_budget,
                enable_depot_late_scheduling=optimize_depot_late_in_search,
                strict_staging_regrab=False,
                budget_fraction=1.0,
            )
            if relaxed_seed is not None:
                relaxed_candidate = relaxed_seed
                relaxed_final_h = _solver_result_final_heuristic(relaxed_candidate)
                if (
                    not relaxed_candidate.is_complete
                    and relaxed_candidate.partial_plan
                    and (
                        relaxed_final_h is None
                        or relaxed_final_h <= max(
                            RELAXED_RESCUE_MAX_FINAL_HEURISTIC,
                            near_goal_partial_resume_max_final_heuristic,
                        )
                    )
                ):
                    relaxed_remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
                    if relaxed_remaining_ms is not None and relaxed_remaining_ms > 250:
                        relaxed_resume_budget = _partial_resume_budget_ms(
                            relaxed_candidate,
                            remaining_budget_ms=relaxed_remaining_ms,
                            max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
                        )
                        relaxed_resume_budget = _cap_pre_primary_child_stage_budget_ms(
                            started_at,
                            time_budget_ms,
                            relaxed_resume_budget,
                            solver_mode=solver_mode,
                            reserve_primary=reserve_primary_for_partial_rescue,
                        )
                        attempted_resume_partial_keys.add(
                            _partial_plan_signature(relaxed_candidate.partial_plan)
                        )
                        relaxed_resume = _try_resume_partial_completion(
                            plan_input=plan_input,
                            initial_state=initial_state,
                            constructive_plan=relaxed_candidate.partial_plan,
                            master=master,
                            time_budget_ms=relaxed_resume_budget,
                            enable_depot_late_scheduling=optimize_depot_late_in_search,
                        )
                        if relaxed_resume is not None:
                            relaxed_candidate = relaxed_resume
                constructive_seed = _shorter_complete_result(
                    constructive_seed,
                    relaxed_candidate,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
                if (
                    not constructive_seed.is_complete
                    and relaxed_candidate.partial_plan
                    and _partial_result_score(
                        relaxed_candidate,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                    < _partial_result_score(
                        constructive_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                ):
                    constructive_seed = replace(
                        constructive_seed,
                        partial_plan=list(relaxed_candidate.partial_plan),
                        partial_fallback_stage=relaxed_candidate.partial_fallback_stage,
                    )
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    # Stage 0.8: route-release constructive rescue. Some legal states are not
    # near-goal in the heuristic but are dominated by route blockers: staging
    # or already-satisfied tracks physically block access to target tracks.
    # Biasing constructive toward route-blockage release is a general recovery
    # strategy for those cases and is cheaper than spending the whole beam
    # budget in a broad state space.
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode in {"exact", "beam"}
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        route_release_budget = _cap_pre_primary_child_stage_budget_ms(
            started_at,
            time_budget_ms,
            ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS,
            solver_mode=solver_mode,
            reserve_primary=reserve_primary_for_partial_rescue,
        )
        if route_release_budget is not None and route_release_budget > MIN_CHILD_STAGE_BUDGET_MS:
            _ts = perf_counter()
            route_release_seed = _run_constructive_stage(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                time_budget_ms=route_release_budget,
                enable_depot_late_scheduling=optimize_depot_late_in_search,
                strict_staging_regrab=False,
                route_release_bias=True,
                budget_fraction=1.0,
            )
            if route_release_seed is not None:
                if route_release_seed.is_complete:
                    route_release_seed = replace(
                        route_release_seed,
                        fallback_stage="constructive_route_release",
                    )
                elif route_release_seed.partial_plan:
                    route_release_remaining_ms = _remaining_wall_budget_ms(
                        started_at,
                        time_budget_ms,
                    )
                    if route_release_remaining_ms is not None and route_release_remaining_ms > 250:
                        route_release_resume_budget = _partial_resume_budget_ms(
                            route_release_seed,
                            remaining_budget_ms=route_release_remaining_ms,
                            max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
                        )
                        route_release_resume_budget = _cap_pre_primary_child_stage_budget_ms(
                            started_at,
                            time_budget_ms,
                            route_release_resume_budget,
                            solver_mode=solver_mode,
                            reserve_primary=reserve_primary_for_partial_rescue,
                        )
                        attempted_resume_partial_keys.add(
                            _partial_plan_signature(route_release_seed.partial_plan)
                        )
                        route_release_resume = _try_resume_partial_completion(
                            plan_input=plan_input,
                            initial_state=initial_state,
                            constructive_plan=route_release_seed.partial_plan,
                            master=master,
                            time_budget_ms=route_release_resume_budget,
                            enable_depot_late_scheduling=optimize_depot_late_in_search,
                        )
                        if route_release_resume is not None:
                            route_release_seed = route_release_resume
                constructive_seed = _shorter_complete_result(
                    constructive_seed,
                    route_release_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
                if (
                    not constructive_seed.is_complete
                    and route_release_seed.partial_plan
                    and _partial_result_score(
                        route_release_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                    < _partial_result_score(
                        constructive_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                ):
                    constructive_seed = replace(
                        constructive_seed,
                        partial_plan=list(route_release_seed.partial_plan),
                        partial_fallback_stage=route_release_seed.partial_fallback_stage,
                    )
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    result: SolverResult
    if solver_mode == "lns":
        lns_budget_ms = _cap_child_stage_budget_ms(started_at, time_budget_ms, time_budget_ms)
        if lns_budget_ms is not None and lns_budget_ms <= 0 and constructive_seed is not None:
            result = constructive_seed
        else:
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
                enable_depot_late_scheduling=optimize_depot_late_in_search,
            )
            phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000
    else:
        effective_heuristic_weight = heuristic_weight
        if _should_skip_primary_after_complete_rescue(
            solver_mode=solver_mode,
            constructive_seed=constructive_seed,
        ):
            result = constructive_seed  # type: ignore[assignment]
        else:
            exact_time_budget_ms: float | None = time_budget_ms
            if (
                solver_mode == "exact"
                and enable_anytime_fallback
                and time_budget_ms is not None
            ):
                exact_time_budget_ms = max(1.0, time_budget_ms * 0.20)
            elif (
                solver_mode == "beam"
                and enable_anytime_fallback
                and time_budget_ms is not None
                and constructive_seed is not None
                and constructive_seed.is_complete
            ):
                # Beam is the native primary search, so partial constructive cases
                # keep the full budget.  Once constructive already has a valid
                # short plan, beam is only a cheap optimiser; for long seeds it is
                # the main hook-count compressor and must keep enough budget.
                exact_time_budget_ms = _beam_complete_seed_budget_ms(
                    time_budget_ms=time_budget_ms,
                    constructive_seed=constructive_seed,
                )
            exact_time_budget_ms = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                exact_time_budget_ms,
            )
            if exact_time_budget_ms is not None and exact_time_budget_ms <= 0:
                result = constructive_seed or _empty_budget_result(
                    solver_mode=solver_mode,
                    started_at=started_at,
                    debug_stats=debug_stats,
                )
            else:
                _ts = perf_counter()
                try:
                    search_result = _solve_search_result(
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                        solver_mode=solver_mode,
                        heuristic_weight=effective_heuristic_weight,
                        beam_width=beam_width,
                        debug_stats=debug_stats,
                        budget=SearchBudget(time_budget_ms=exact_time_budget_ms, node_budget=node_budget),
                        enable_depot_late_scheduling=optimize_depot_late_in_search,
                        enable_structural_diversity=solver_mode == "beam",
                    )
                    result = _shorter_complete_result(
                        constructive_seed,
                        search_result,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
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
                        is_complete=False,
                        is_proven_optimal=False,
                        fallback_stage=solver_mode,
                        debug_stats=debug_stats,
                    )
                phase_timings["exact_ms"] = (perf_counter() - _ts) * 1000
        if (
            solver_mode == "exact"
            and enable_anytime_fallback
            and not result.is_proven_optimal
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            _ts = perf_counter()
            chain_incumbent = result
            if (
                result.is_complete
                and not result.is_proven_optimal
                and constructive_seed is not None
                and time_budget_ms is not None
            ):
                chain_incumbent = SolverResult(
                    plan=[],
                    expanded_nodes=0,
                    generated_nodes=0,
                    closed_nodes=0,
                    elapsed_ms=0.0,
                    is_complete=False,
                    is_proven_optimal=False,
                    fallback_stage=solver_mode,
                    debug_stats=debug_stats,
                )
            chain_result = _anytime_run_fallback_chain(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                incumbent=chain_incumbent,
                started_at=started_at,
                time_budget_ms=time_budget_ms,
                node_budget=node_budget,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
                debug_stats=debug_stats,
                solve_search_result=_solve_search_result,
                enable_depot_late_scheduling=optimize_depot_late_in_search,
            )
            result = _shorter_complete_result(
                result,
                chain_result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            phase_timings["anytime_ms"] = (perf_counter() - _ts) * 1000
        if (
            solver_mode == "beam"
            and beam_width is not None
            and not _should_skip_primary_after_complete_rescue(
                solver_mode=solver_mode,
                constructive_seed=constructive_seed,
            )
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            _ts = perf_counter()
            improved = result
            _lns_remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
            for _ in range(BEAM_POST_REPAIR_MAX_ROUNDS):
                if _lns_remaining_ms is not None and _lns_remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                    break
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
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                    time_budget_ms=_lns_remaining_ms,
                )
                if (
                    not candidate.is_complete
                    or (
                        improved.is_complete
                        and len(candidate.plan) >= len(improved.plan)
                    )
                ):
                    break
                improved = candidate
                _lns_remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
            result = improved
            phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000
            # If beam search found no plan, run the anytime fallback chain.
            # When the constructive seed is partial, pass an empty incumbent so
            # the chain's early-exit guard (`if current.plan: break`) doesn't
            # short-circuit immediately.
            if (
                not result.is_complete
                and enable_anytime_fallback
                and _has_child_stage_budget(started_at, time_budget_ms)
            ):
                _ts = perf_counter()
                chain_incumbent = (
                    SolverResult(
                        plan=[],
                        expanded_nodes=0,
                        generated_nodes=0,
                        closed_nodes=0,
                        elapsed_ms=0.0,
                        is_complete=False,
                        is_proven_optimal=False,
                        fallback_stage="beam",
                        debug_stats=debug_stats,
                    )
                    if constructive_seed is not None
                    and not constructive_seed.is_complete
                    else (constructive_seed if constructive_seed is not None else result)
                )
                chain_result = _anytime_run_fallback_chain(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    incumbent=chain_incumbent,
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                    node_budget=node_budget,
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    debug_stats=debug_stats,
                    solve_search_result=_solve_search_result,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                )
                result = _shorter_complete_result(
                    result,
                    chain_result,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
                phase_timings["anytime_ms"] = (perf_counter() - _ts) * 1000

    if not result.is_complete and constructive_seed is not None:
        if constructive_seed.is_complete:
            result = constructive_seed
        elif (
            constructive_seed.partial_plan
            and _partial_result_score(
                constructive_seed,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            < _partial_result_score(
                result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
        ):
            result = replace(
                result,
                partial_plan=list(constructive_seed.partial_plan),
                partial_fallback_stage=constructive_seed.partial_fallback_stage,
                debug_stats=constructive_seed.debug_stats,
            )

    if (
        enable_anytime_fallback
        and not result.is_complete
        and result.partial_plan
        and _has_child_stage_budget(started_at, time_budget_ms)
    ):
        partial_has_route_pressure = (
            _partial_result_route_blockage_pressure(
                result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            > 0
        )
        if (
            not result.is_complete
            and partial_has_route_pressure
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            route_tail_budget = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
            )
            if route_tail_budget is not None and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS:
                _ts = perf_counter()
                route_tail_completion = _try_route_blockage_tail_clearance_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    partial_plan=result.partial_plan,
                    master=master,
                    time_budget_ms=route_tail_budget,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                )
                if route_tail_completion is None:
                    route_tail_completion = _try_route_release_partial_completion(
                        plan_input=plan_input,
                        initial_state=initial_state,
                        partial_plan=result.partial_plan,
                        master=master,
                        time_budget_ms=route_tail_budget,
                        enable_depot_late_scheduling=optimize_depot_late_in_search,
                    )
                if route_tail_completion is not None:
                    result = _shorter_complete_result(
                        result,
                        route_tail_completion,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000
        if (
            not result.is_complete
            and not partial_has_route_pressure
            and _partial_plan_signature(result.partial_plan)
            not in attempted_resume_partial_keys
            and _partial_result_is_near_goal(
                result,
                plan_input=plan_input,
                initial_state=initial_state,
            )
        ):
            near_goal_resume_budget = _partial_resume_budget_ms(
                result,
                remaining_budget_ms=_remaining_wall_budget_ms(started_at, time_budget_ms) or 0.0,
                max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
            )
            near_goal_resume_budget = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                near_goal_resume_budget,
            )
            if near_goal_resume_budget is None or near_goal_resume_budget <= MIN_CHILD_STAGE_BUDGET_MS:
                near_goal_resume_budget = None
        else:
            near_goal_resume_budget = None
        if near_goal_resume_budget is not None:
            _ts = perf_counter()
            attempted_resume_partial_keys.add(
                _partial_plan_signature(result.partial_plan)
            )
            resume_completion = _try_resume_partial_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                constructive_plan=result.partial_plan,
                master=master,
                time_budget_ms=near_goal_resume_budget,
                enable_depot_late_scheduling=optimize_depot_late_in_search,
            )
            if resume_completion is not None:
                result = _shorter_complete_result(
                    result,
                    resume_completion,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000
        if (
            not result.is_complete
            and not partial_has_route_pressure
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            route_tail_budget = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
            )
            if route_tail_budget is not None and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS:
                _ts = perf_counter()
                route_tail_completion = _try_route_release_partial_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    partial_plan=result.partial_plan,
                    master=master,
                    time_budget_ms=route_tail_budget,
                )
                if route_tail_completion is not None:
                    result = _shorter_complete_result(
                        result,
                        route_tail_completion,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    # Post-search LNS polish: on fallback-stage plans (non-exact), spend any
    # remaining budget compressing hook count via destroy/repair. Exact plans
    # are already proven optimal, so skip them. Worth doing when the plan is
    # long — the fixed per-pass cost amortises better on 50+ hook cases.
    # Skip ``solver_mode == "beam"`` because that path already has a dedicated
    # post-repair loop above with its own tuned budget.
    if (
        solver_mode != "beam"
        and result.is_complete
        and result.plan
        and not result.is_proven_optimal
        and result.fallback_stage not in {"constructive"}
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
                enable_depot_late_scheduling=optimize_depot_late_in_search,
            )
            if polished.is_complete and len(polished.plan) < len(result.plan):
                result = polished
            phase_timings["lns_ms"] += (perf_counter() - _ts) * 1000

    # Goal-by-goal rescue was removed (W3-L): the root-cause fix in
    # _score_move prevents partial plans upstream, making post-hoc rescue
    # redundant.

    result = _compress_complete_plan(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        debug_stats=debug_stats,
    )

    # Depot-late post-processing: only when the flag is on and we have a plan.
    # Reorders adjacent (depot, non-depot) hook pairs to push depot hooks
    # toward the tail, preserving replay-equivalent final state. On failure
    # (either an internal simulate rejection or a full-verifier rejection of
    # the reordered plan — e.g. intermediate route interference that
    # _apply_move doesn't detect) the plan is left unchanged.
    if enable_depot_late_scheduling and result.is_complete and result.plan:
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
        depot_earliness=depot_earliness(result.plan if result.is_complete else result.partial_plan),
        depot_hook_count=sum(
            1
            for h in (result.plan if result.is_complete else result.partial_plan)
            if is_depot_hook(h)
        ),
    )

    result = _attach_structural_debug_stats(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        debug_stats=debug_stats,
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

    total_elapsed_ms = (perf_counter() - started_at) * 1000
    telemetry = _build_telemetry(
        plan_input=plan_input,
        result=result,
        phase_timings=phase_timings,
        total_ms=total_elapsed_ms,
        time_budget_ms=time_budget_ms,
        node_budget=node_budget,
    )
    emit_telemetry(telemetry)
    return replace(result, elapsed_ms=total_elapsed_ms, telemetry=telemetry)


def _compress_complete_plan(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    debug_stats: dict[str, Any] | None,
) -> SolverResult:
    stats = debug_stats if debug_stats is not None else dict(result.debug_stats or {})
    if not result.is_complete or not result.plan or master is None:
        stats["plan_compression"] = {
            "accepted_rewrite_count": 0,
            "before_hook_count": len(result.plan),
            "after_hook_count": len(result.plan),
        }
        return replace(result, debug_stats=stats)

    from fzed_shunting.solver.plan_compressor import compress_plan

    compressed = compress_plan(
        plan_input,
        initial_state,
        result.plan,
        master=master,
    )
    stats["plan_compression"] = {
        "accepted_rewrite_count": compressed.accepted_rewrite_count,
        "before_hook_count": len(result.plan),
        "after_hook_count": len(compressed.compressed_plan),
    }
    if len(compressed.compressed_plan) < len(result.plan):
        return replace(result, plan=compressed.compressed_plan, debug_stats=stats)
    return replace(result, debug_stats=stats)


def _attach_structural_debug_stats(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    debug_stats: dict[str, Any] | None,
) -> SolverResult:
    stats = debug_stats if debug_stats is not None else dict(result.debug_stats or {})
    stats["initial_structural_metrics"] = compute_structural_metrics(
        plan_input,
        initial_state,
    ).to_dict()
    stats["initial_capacity_release_plan"] = compute_capacity_release_plan(
        plan_input,
        initial_state,
    ).to_dict()
    if master is not None:
        stats["initial_route_blockage_plan"] = compute_route_blockage_plan(
            plan_input,
            initial_state,
            RouteOracle(master),
        ).to_dict()
    active_plan = result.plan if result.is_complete else result.partial_plan
    stats["plan_shape_metrics"] = summarize_plan_shape(active_plan)
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=active_plan,
    )
    if final_state is not None:
        key = "final_structural_metrics" if result.is_complete else "partial_structural_metrics"
        stats[key] = compute_structural_metrics(plan_input, final_state).to_dict()
        if master is not None:
            route_key = (
                "final_route_blockage_plan"
                if result.is_complete
                else "partial_route_blockage_plan"
            )
            stats[route_key] = compute_route_blockage_plan(
                plan_input,
                final_state,
                RouteOracle(master),
            ).to_dict()
    return replace(result, debug_stats=stats)


def _replay_solver_moves(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    plan: list[HookAction],
) -> ReplayState | None:
    state = ReplayState.model_validate(initial_state.model_dump())
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    try:
        for move in plan:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
    except Exception:  # noqa: BLE001
        return None
    return state


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
    partial_is_valid: bool | None = None
    if result.partial_verification_report is not None:
        partial_is_valid = bool(result.partial_verification_report.is_valid)
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
        is_complete=result.is_complete,
        plan_hook_count=len(result.plan),
        fallback_stage=result.fallback_stage,
        is_valid=is_valid,
        partial_hook_count=len(result.partial_plan),
        partial_fallback_stage=result.partial_fallback_stage,
        partial_is_valid=partial_is_valid,
        is_proven_optimal=result.is_proven_optimal,
        time_budget_ms=time_budget_ms,
        node_budget=node_budget,
    )


def _beam_complete_seed_budget_ms(
    *,
    time_budget_ms: float,
    constructive_seed: SolverResult,
) -> float:
    seed_hook_count = len(constructive_seed.plan)
    ratio = max(
        BEAM_COMPLETE_SEED_MIN_BUDGET_RATIO,
        min(1.0, seed_hook_count / BEAM_COMPLETE_SEED_FULL_BUDGET_HOOKS),
    )
    return min(time_budget_ms, max(1.0, time_budget_ms * ratio))


def _should_skip_primary_after_complete_rescue(
    *,
    solver_mode: str,
    constructive_seed: SolverResult | None,
) -> bool:
    if solver_mode != "beam" or constructive_seed is None or not constructive_seed.is_complete:
        return False
    return constructive_seed.fallback_stage in {
        "constructive_partial_resume",
        "constructive_warm_start",
        "constructive_route_release",
        "constructive_route_release_tail",
    }


def _remaining_wall_budget_ms(started_at: float, time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return None
    return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000)


def _cap_child_stage_budget_ms(
    started_at: float,
    time_budget_ms: float | None,
    requested_budget_ms: float | None,
) -> float | None:
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is None:
        return requested_budget_ms
    if requested_budget_ms is None:
        return remaining_ms
    return min(max(0.0, requested_budget_ms), remaining_ms)


def _cap_pre_primary_child_stage_budget_ms(
    started_at: float,
    time_budget_ms: float | None,
    requested_budget_ms: float | None,
    *,
    solver_mode: str,
    reserve_primary: bool = True,
) -> float | None:
    capped_budget = _cap_child_stage_budget_ms(
        started_at,
        time_budget_ms,
        requested_budget_ms,
    )
    if (
        not reserve_primary
        or solver_mode != "beam"
        or time_budget_ms is None
        or capped_budget is None
    ):
        return capped_budget
    reserve_ms = min(time_budget_ms, BEAM_INCOMPLETE_SEED_PRIMARY_MIN_BUDGET_MS)
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is None:
        return capped_budget
    spendable_ms = max(0.0, remaining_ms - reserve_ms)
    return min(capped_budget, spendable_ms)


def _route_release_tail_budget_ms(
    *,
    remaining_budget_ms: float,
    solver_mode: str,
    reserve_primary: bool,
) -> float:
    remaining_budget_ms = max(0.0, remaining_budget_ms)
    spendable_ms = remaining_budget_ms
    if reserve_primary and solver_mode == "beam":
        spendable_ms = max(
            0.0,
            remaining_budget_ms - BEAM_INCOMPLETE_SEED_PRIMARY_MIN_BUDGET_MS,
        )
    return min(
        EARLY_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
        max(500.0, spendable_ms),
    )


def _partial_plan_signature(plan: list[HookAction]) -> tuple:
    return tuple(
        (
            move.action_type,
            move.source_track,
            move.target_track,
            tuple(move.vehicle_nos),
            tuple(move.path_tracks),
        )
        for move in plan
    )


def _has_child_stage_budget(started_at: float, time_budget_ms: float | None) -> bool:
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    return remaining_ms is None or remaining_ms > MIN_CHILD_STAGE_BUDGET_MS


def _empty_budget_result(
    *,
    solver_mode: str,
    started_at: float,
    debug_stats: dict[str, Any] | None,
) -> SolverResult:
    return SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=False,
        is_proven_optimal=False,
        fallback_stage=solver_mode,
        debug_stats=debug_stats,
    )


def _partial_resume_budget_ms(
    seed: SolverResult,
    *,
    remaining_budget_ms: float,
    max_final_heuristic: int = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
) -> float:
    remaining_budget_ms = max(0.0, remaining_budget_ms)
    final_h = _solver_result_final_heuristic(seed)
    if final_h is not None and final_h <= max_final_heuristic:
        return min(
            NEAR_GOAL_PARTIAL_RESUME_MAX_BUDGET_MS,
            max(500.0, remaining_budget_ms * NEAR_GOAL_PARTIAL_RESUME_BUDGET_RATIO),
        )
    return min(5_000.0, max(500.0, remaining_budget_ms * 0.20))


def _shorter_complete_result(
    incumbent: SolverResult | None,
    candidate: SolverResult | None,
    *,
    plan_input: NormalizedPlanInput | None = None,
    initial_state: ReplayState | None = None,
    master: MasterData | None = None,
) -> SolverResult:
    if candidate is None:
        if incumbent is None:
            raise ValueError("at least one solver result is required")
        return incumbent
    if incumbent is None:
        return candidate
    if not candidate.is_complete:
        if not incumbent.is_complete and _partial_result_score(
            candidate,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ) < _partial_result_score(
            incumbent,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ):
            return candidate
        return incumbent
    if not incumbent.is_complete:
        return candidate
    if len(candidate.plan) < len(incumbent.plan):
        return candidate
    if len(candidate.plan) == len(incumbent.plan) and (
        candidate.is_proven_optimal and not incumbent.is_proven_optimal
    ):
        return candidate
    return incumbent


def _partial_result_score(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput | None = None,
    initial_state: ReplayState | None = None,
    master: MasterData | None = None,
) -> tuple[int, int, int, int, int, int]:
    stats = result.debug_stats or {}
    structural = stats.get("partial_structural_metrics") or {}
    route_blockage = stats.get("partial_route_blockage_plan") or {}
    if (
        result.partial_plan
        and (not structural or not route_blockage)
        and plan_input is not None
        and initial_state is not None
    ):
        final_state = _replay_solver_moves(
            plan_input=plan_input,
            initial_state=initial_state,
            plan=result.partial_plan,
        )
        if final_state is not None:
            if not structural:
                structural = compute_structural_metrics(plan_input, final_state).to_dict()
            if not route_blockage and master is not None:
                route_blockage = compute_route_blockage_plan(
                    plan_input,
                    final_state,
                    RouteOracle(master),
                ).to_dict()
    unfinished = _optional_int(structural.get("unfinished_count"))
    blockage = _optional_int(route_blockage.get("total_blockage_pressure"))
    staging_debt = _optional_int(structural.get("staging_debt_count"))
    capacity_overflow = _optional_int(structural.get("capacity_overflow_track_count"))
    return (
        unfinished if unfinished is not None else 10**9,
        blockage if blockage is not None else 10**9,
        staging_debt if staging_debt is not None else 10**9,
        capacity_overflow if capacity_overflow is not None else 10**9,
        -len(result.partial_plan),
        _stage_rank(result.partial_fallback_stage or result.fallback_stage),
    )


def _stage_rank(stage: str | None) -> int:
    if stage is None:
        return 10
    if "beam" in stage:
        return 0
    if "constructive" in stage:
        return 1
    return 5


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _solver_result_final_heuristic(result: SolverResult) -> float | None:
    if result.debug_stats is None:
        return None
    value = result.debug_stats.get("final_heuristic")
    return float(value) if value is not None else None


def _partial_result_is_near_goal(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> bool:
    stats = result.debug_stats or {}
    structural = stats.get("partial_structural_metrics") or {}
    unfinished = _optional_int(structural.get("unfinished_count"))
    if unfinished is not None:
        return unfinished <= LOCALIZED_RESUME_MAX_UNFINISHED
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if final_state is None:
        return False
    purity = compute_state_purity(plan_input, final_state)
    return purity.unfinished_count <= LOCALIZED_RESUME_MAX_UNFINISHED


def _partial_result_route_blockage_pressure(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> int:
    route_blockage = (result.debug_stats or {}).get("partial_route_blockage_plan") or {}
    pressure = _optional_int(route_blockage.get("total_blockage_pressure"))
    if pressure is not None:
        return pressure
    if master is None:
        return 0
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if final_state is None:
        return 0
    return compute_route_blockage_plan(
        plan_input,
        final_state,
        RouteOracle(master),
    ).total_blockage_pressure


def _run_constructive_stage(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float | None,
    enable_depot_late_scheduling: bool = False,
    strict_staging_regrab: bool = True,
    route_release_bias: bool = False,
    budget_fraction: float = 0.15,
) -> SolverResult | None:
    """Run the priority-rule dispatcher and wrap the result as a SolverResult."""
    from fzed_shunting.solver.constructive import solve_constructive

    constructive_budget = None
    if time_budget_ms is not None:
        constructive_budget = min(8_000.0, max(500.0, time_budget_ms * budget_fraction))
    try:
        ctr = solve_constructive(
            plan_input,
            initial_state,
            master=master,
            max_iterations=1500,
            time_budget_ms=constructive_budget,
            strict_staging_regrab=strict_staging_regrab,
            route_release_bias=route_release_bias,
        )
    except Exception:  # noqa: BLE001
        return None
    if not ctr.plan and not ctr.reached_goal:
        return None
    debug_stats = dict(ctr.debug_stats or {})
    if ctr.final_heuristic is not None:
        debug_stats["final_heuristic"] = ctr.final_heuristic
    return SolverResult(
        plan=list(ctr.plan) if ctr.reached_goal else [],
        expanded_nodes=ctr.iterations,
        generated_nodes=ctr.iterations,
        closed_nodes=0,
        elapsed_ms=ctr.elapsed_ms,
        is_complete=ctr.reached_goal,
        is_proven_optimal=False,
        fallback_stage="constructive" if ctr.reached_goal else None,
        partial_plan=[] if ctr.reached_goal else list(ctr.plan),
        partial_fallback_stage=None if ctr.reached_goal else "constructive_partial",
        debug_stats=debug_stats,
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
    if not result.is_complete and not result.partial_plan:
        return result
    from fzed_shunting.verify.plan_verifier import verify_plan

    if result.is_complete:
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
        if not report.is_valid:
            raise PlanVerificationError(report)
        return replace(result, verification_report=report)

    if not result.partial_plan:
        return result

    partial_hook_plan = [
        {
            "hookNo": index,
            "actionType": move.action_type,
            "sourceTrack": move.source_track,
            "targetTrack": move.target_track,
            "vehicleNos": list(move.vehicle_nos),
            "pathTracks": list(move.path_tracks),
        }
        for index, move in enumerate(result.partial_plan, start=1)
    ]
    partial_report = verify_plan(
        master,
        plan_input,
        partial_hook_plan,
        initial_state_override=initial_state,
        require_complete_goals=False,
    )
    return replace(result, partial_verification_report=partial_report)

def _heuristic(plan_input: NormalizedPlanInput, state: ReplayState) -> int:
    from fzed_shunting.solver.heuristic import compute_admissible_heuristic_real_hook

    return compute_admissible_heuristic_real_hook(plan_input, state)


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
    from fzed_shunting.solver.heuristic import compute_admissible_heuristic_real_hook
    from fzed_shunting.io.normalize_input import NormalizedVehicle

    vehicle_by_no: dict[str, NormalizedVehicle] = {v.vehicle_no: v for v in plan_input.vehicles}
    state = initial_state
    try:
        for move in constructive_plan:
            state = _apply_move(state=state, move=move, plan_input=plan_input, vehicle_by_no=vehicle_by_no)
    except Exception:  # noqa: BLE001
        return None

    h = compute_admissible_heuristic_real_hook(plan_input, state)
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
    if not completion.is_complete:
        return None

    return SolverResult(
        plan=list(constructive_plan) + list(completion.plan),
        expanded_nodes=completion.expanded_nodes,
        generated_nodes=completion.generated_nodes,
        closed_nodes=completion.closed_nodes,
        elapsed_ms=completion.elapsed_ms,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="constructive_warm_start",
        debug_stats=completion.debug_stats,
    )


def _try_resume_partial_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    constructive_plan: list[HookAction],
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    from fzed_shunting.io.normalize_input import NormalizedVehicle

    if not constructive_plan:
        return None
    started_at = perf_counter()
    vehicle_by_no: dict[str, NormalizedVehicle] = {v.vehicle_no: v for v in plan_input.vehicles}
    state = initial_state
    resumed_prefix: list[HookAction] = []
    checkpoints: list[tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState]] = []
    try:
        for move in constructive_plan:
            state = _apply_move(state=state, move=move, plan_input=plan_input, vehicle_by_no=vehicle_by_no)
            resumed_prefix.append(move)
            if not state.loco_carry and not _is_goal(plan_input, state):
                purity = compute_state_purity(plan_input, state)
                checkpoints.append((
                    purity.unfinished_count,
                    -len(resumed_prefix),
                    list(resumed_prefix),
                    state,
                ))
    except Exception:  # noqa: BLE001
        return None

    if not checkpoints:
        return None

    ranked_checkpoints = [
        checkpoint
        for checkpoint in sorted(checkpoints)[:PARTIAL_RESUME_MAX_CHECKPOINTS]
    ]
    best_partial: SolverResult | None = None
    for index, (_unfinished_count, _negative_prefix_len, checkpoint_prefix, checkpoint_state) in enumerate(ranked_checkpoints):
        remaining_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if remaining_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        remaining_attempts = len(ranked_checkpoints) - index
        per_checkpoint_budget = remaining_budget_ms / max(1, remaining_attempts)
        checkpoint_result = _try_resume_from_checkpoint(
            plan_input=plan_input,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_state=checkpoint_state,
            original_initial_state=initial_state,
            master=master,
            time_budget_ms=per_checkpoint_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if checkpoint_result is None:
            continue
        if checkpoint_result.is_complete:
            return checkpoint_result
        route_tail_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if (
            checkpoint_result.partial_plan
            and master is not None
            and route_tail_budget_ms > ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
            and _partial_result_route_blockage_pressure(
                checkpoint_result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            > 0
        ):
            route_tail_completion = _try_route_blockage_tail_clearance_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                partial_plan=checkpoint_result.partial_plan,
                master=master,
                time_budget_ms=route_tail_budget_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if route_tail_completion is not None:
                return route_tail_completion
        if checkpoint_result.partial_plan and (
            best_partial is None
            or _partial_result_score(
                checkpoint_result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            < _partial_result_score(
                best_partial,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
        ):
            best_partial = checkpoint_result
    return best_partial


def _try_resume_from_checkpoint(
    *,
    plan_input: NormalizedPlanInput,
    checkpoint_prefix: list[HookAction],
    checkpoint_state: ReplayState,
    original_initial_state: ReplayState | None = None,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    started_at = perf_counter()
    localized_budget_ms = max(
        250.0,
        time_budget_ms * LOCALIZED_RESUME_BUDGET_RATIO,
    )
    localized_completion = _try_localized_resume_completion(
        plan_input=plan_input,
        initial_state=checkpoint_state,
        master=master,
        time_budget_ms=localized_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if localized_completion is None:
        remaining_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if remaining_budget_ms < LOCALIZED_RESUME_MIN_FULL_BEAM_MS:
            return None
        full_beam_budget_ms = remaining_budget_ms
        if master is not None:
            reserved_tail_budget_ms = min(
                PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
                max(1_000.0, remaining_budget_ms * 0.25),
            )
            if remaining_budget_ms > reserved_tail_budget_ms + LOCALIZED_RESUME_MIN_FULL_BEAM_MS:
                full_beam_budget_ms = remaining_budget_ms - reserved_tail_budget_ms
        try:
            localized_completion = _solve_search_result(
                plan_input=plan_input,
                initial_state=checkpoint_state,
                master=master,
                solver_mode="beam",
                heuristic_weight=1.0,
                beam_width=8,
                budget=SearchBudget(time_budget_ms=full_beam_budget_ms),
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                enable_structural_diversity=True,
            )
        except ValueError:
            localized_completion = None
    if localized_completion is None:
        return None
    if not localized_completion.is_complete:
        if not localized_completion.partial_plan:
            return None
        combined_partial_plan = checkpoint_prefix + list(localized_completion.partial_plan)
        remaining_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if (
            master is not None
            and original_initial_state is not None
            and remaining_budget_ms > ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
        ):
            partial_state = _replay_solver_moves(
                plan_input=plan_input,
                initial_state=checkpoint_state,
                plan=list(localized_completion.partial_plan),
            )
            if partial_state is not None and not partial_state.loco_carry:
                route_blockage_plan = compute_route_blockage_plan(
                    plan_input,
                    partial_state,
                    RouteOracle(master),
                )
            else:
                route_blockage_plan = None
            if (
                route_blockage_plan is not None
                and route_blockage_plan.total_blockage_pressure > 0
            ):
                tail_clearance = _try_route_blockage_tail_clearance_from_state(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=combined_partial_plan,
                    state=partial_state,
                    master=master,
                    time_budget_ms=remaining_budget_ms,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
                if tail_clearance is not None:
                    return tail_clearance
        return SolverResult(
            plan=[],
            expanded_nodes=localized_completion.expanded_nodes,
            generated_nodes=localized_completion.generated_nodes,
            closed_nodes=localized_completion.closed_nodes,
            elapsed_ms=localized_completion.elapsed_ms,
            is_complete=False,
            is_proven_optimal=False,
            fallback_stage="constructive_partial_resume",
            partial_plan=combined_partial_plan,
            partial_fallback_stage="constructive_partial_resume",
            debug_stats=localized_completion.debug_stats,
        )
    return SolverResult(
        plan=checkpoint_prefix + list(localized_completion.plan),
        expanded_nodes=localized_completion.expanded_nodes,
        generated_nodes=localized_completion.generated_nodes,
        closed_nodes=localized_completion.closed_nodes,
        elapsed_ms=localized_completion.elapsed_ms,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="constructive_partial_resume",
        debug_stats=localized_completion.debug_stats,
    )


def _try_route_release_partial_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    partial_plan: list[HookAction],
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult | None:
    if master is None or not partial_plan:
        return None
    from fzed_shunting.io.normalize_input import NormalizedVehicle
    from fzed_shunting.solver.constructive import solve_constructive

    vehicle_by_no: dict[str, NormalizedVehicle] = {v.vehicle_no: v for v in plan_input.vehicles}
    state = initial_state
    checkpoints: list[tuple[int, int, list[HookAction], ReplayState]] = []
    try:
        replayed_prefix: list[HookAction] = []
        for move in partial_plan:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            replayed_prefix.append(move)
            if not state.loco_carry and not _is_goal(plan_input, state):
                route_blockage_plan = compute_route_blockage_plan(
                    plan_input,
                    state,
                    RouteOracle(master),
                )
                if route_blockage_plan.total_blockage_pressure > 0:
                    structural = compute_structural_metrics(plan_input, state)
                    checkpoints.append(
                        (
                            (
                                route_blockage_plan.total_blockage_pressure,
                                structural.goal_track_blocker_count,
                                structural.staging_debt_count,
                                structural.unfinished_count,
                            ),
                            -len(replayed_prefix),
                            list(replayed_prefix),
                            state,
                        )
                    )
    except Exception:  # noqa: BLE001
        return None
    if not checkpoints:
        return None

    recent_window = sorted(checkpoints, key=lambda item: item[1])[:8]
    recent_min_pressure = min(checkpoint[0][0] for checkpoint in recent_window)
    low_pressure_recent = [
        checkpoint
        for checkpoint in recent_window
        if checkpoint[0][0] <= recent_min_pressure + 1
    ]
    recent_checkpoints = sorted(low_pressure_recent, key=lambda item: item[1])[:1]
    quality_checkpoints = sorted(
        checkpoints,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )[:2]
    ranked_checkpoints: list[tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState]] = []
    seen_prefix_lengths: set[int] = set()
    for checkpoint in [*recent_checkpoints, *quality_checkpoints]:
        prefix_len = -checkpoint[1]
        if prefix_len in seen_prefix_lengths:
            continue
        seen_prefix_lengths.add(prefix_len)
        ranked_checkpoints.append(checkpoint)
    started_at = perf_counter()
    best_partial: SolverResult | None = None
    for index, (_score, _negative_prefix_len, completion_prefix, completion_state) in enumerate(
        ranked_checkpoints
    ):
        remaining_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if remaining_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        per_checkpoint_budget = remaining_budget_ms
        tail_clearance = _try_route_blockage_tail_clearance_from_state(
            plan_input=plan_input,
            original_initial_state=initial_state,
            prefix_plan=completion_prefix,
            state=completion_state,
            master=master,
            time_budget_ms=per_checkpoint_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if tail_clearance is not None:
            return tail_clearance
        remaining_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if remaining_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        per_checkpoint_budget = remaining_budget_ms
        try:
            completion = solve_constructive(
                plan_input,
                completion_state,
                master=master,
                max_iterations=1500,
                time_budget_ms=per_checkpoint_budget,
                strict_staging_regrab=False,
                route_release_bias=True,
            )
        except Exception:  # noqa: BLE001
            continue
        if not completion.reached_goal:
            partial_candidate = SolverResult(
                plan=[],
                expanded_nodes=completion.iterations,
                generated_nodes=completion.iterations,
                closed_nodes=0,
                elapsed_ms=completion.elapsed_ms,
                is_complete=False,
                is_proven_optimal=False,
                fallback_stage="constructive_route_release_tail",
                partial_plan=list(completion_prefix) + list(completion.plan),
                partial_fallback_stage="constructive_route_release_tail",
                debug_stats=dict(completion.debug_stats or {}),
            )
            if (
                partial_candidate.partial_plan
                and (
                    best_partial is None
                    or _partial_result_score(
                        partial_candidate,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                    < _partial_result_score(
                        best_partial,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                )
            ):
                best_partial = partial_candidate
            continue
        combined_plan = list(completion_prefix) + list(completion.plan)
        result = SolverResult(
            plan=combined_plan,
            expanded_nodes=completion.iterations,
            generated_nodes=completion.iterations,
            closed_nodes=0,
            elapsed_ms=completion.elapsed_ms,
            is_complete=True,
            is_proven_optimal=False,
            fallback_stage="constructive_route_release_tail",
            debug_stats=dict(completion.debug_stats or {}),
        )
        try:
            return _attach_verification(
                result,
                plan_input=plan_input,
                master=master,
                initial_state=initial_state,
            )
        except Exception:  # noqa: BLE001
            continue
    return best_partial


def _try_route_blockage_tail_clearance_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    partial_plan: list[HookAction],
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult | None:
    if master is None:
        return None
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    state = ReplayState.model_validate(initial_state.model_dump())
    try:
        for move in partial_plan:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
    except Exception:  # noqa: BLE001
        return None
    return _try_route_blockage_tail_clearance_from_state(
        plan_input=plan_input,
        original_initial_state=initial_state,
        prefix_plan=list(partial_plan),
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _try_route_blockage_tail_clearance_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if state.loco_carry:
        return None
    route_oracle = RouteOracle(master)
    initial_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    if initial_blockage.total_blockage_pressure <= 0:
        return None

    started_at = perf_counter()
    frontier: list[tuple[ReplayState, list[HookAction]]] = [(state, [])]
    seen: set[tuple] = {_state_key(state, plan_input)}
    expanded = 0
    generated = 0

    while frontier:
        remaining_ms = time_budget_ms - (perf_counter() - started_at) * 1000
        if remaining_ms <= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS:
            break
        frontier.sort(
            key=lambda item: (
                compute_route_blockage_plan(plan_input, item[0], route_oracle).total_blockage_pressure,
                compute_structural_metrics(plan_input, item[0]).unfinished_count,
                len(item[1]),
            )
        )
        current_state, clearing_plan = frontier.pop(0)
        expanded += 1
        current_blockage = compute_route_blockage_plan(
            plan_input,
            current_state,
            route_oracle,
        )
        if current_blockage.total_blockage_pressure == 0 and not current_state.loco_carry:
            completion = _try_tail_clearance_resume_from_state(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=clearing_plan,
                state=current_state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=remaining_ms,
                expanded_nodes=expanded,
                generated_nodes=generated,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if completion is not None:
                return completion
        if len(clearing_plan) >= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS:
            continue

        for move, next_state in _route_blockage_tail_clearance_candidates(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            current_blockage=current_blockage,
        ):
            state_key = _state_key(next_state, plan_input)
            if state_key in seen:
                continue
            seen.add(state_key)
            generated += 1
            frontier.append((next_state, [*clearing_plan, move]))
            if len(frontier) > 32:
                frontier.sort(
                    key=lambda item: (
                        compute_route_blockage_plan(
                            plan_input,
                            item[0],
                            route_oracle,
                        ).total_blockage_pressure,
                        compute_structural_metrics(plan_input, item[0]).unfinished_count,
                        len(item[1]),
                    )
                )
                del frontier[32:]
    return None


def _route_blockage_tail_clearance_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    current_blockage: Any,
) -> list[tuple[HookAction, ReplayState]]:
    if current_blockage.total_blockage_pressure <= 0 and not state.loco_carry:
        return []
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    current_pressure = current_blockage.total_blockage_pressure
    facts_by_blocking_track = getattr(current_blockage, "facts_by_blocking_track", {})
    if not facts_by_blocking_track and not state.loco_carry:
        return []
    current_blocking_tracks = set(facts_by_blocking_track)
    moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )
    candidates: list[tuple[tuple[int, int, int, str, tuple[str, ...]], HookAction, ReplayState]] = []
    for move in moves:
        if state.loco_carry:
            if move.action_type != "DETACH":
                continue
            if move.target_track in current_blocking_tracks:
                continue
        else:
            if move.action_type != "ATTACH":
                continue
            fact = facts_by_blocking_track.get(move.source_track)
            if fact is None:
                continue
            if not (set(move.vehicle_nos) & set(fact.blocking_vehicle_nos)):
                continue
        try:
            next_state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        next_blockage = compute_route_blockage_plan(
            plan_input,
            next_state,
            route_oracle,
        )
        next_pressure = next_blockage.total_blockage_pressure
        if next_pressure > current_pressure:
            continue
        if state.loco_carry and move.target_track in next_blockage.facts_by_blocking_track:
            continue
        source_remainder = len(next_state.track_sequences.get(move.source_track, []))
        pressure_delta = current_pressure - next_pressure
        candidates.append(
            (
                (
                    -pressure_delta,
                    0 if move.action_type == "ATTACH" else 1,
                    source_remainder,
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
                next_state,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [(move, next_state) for _, move, next_state in candidates[:8]]


def _try_tail_clearance_resume_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    clearing_plan: list[HookAction],
    state: ReplayState,
    initial_blockage: Any,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if _is_goal(plan_input, state):
        result = SolverResult(
            plan=[*prefix_plan, *clearing_plan],
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            closed_nodes=expanded_nodes,
            elapsed_ms=0.0,
            is_complete=True,
            is_proven_optimal=False,
            fallback_stage="route_blockage_tail_clearance",
        )
        return _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=original_initial_state,
        )
    direct_completion = _try_direct_blocked_tail_completion_from_state(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=prefix_plan,
        clearing_plan=clearing_plan,
        state=state,
        initial_blockage=initial_blockage,
        master=master,
        time_budget_ms=time_budget_ms,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if direct_completion is not None:
        return direct_completion
    completion = _try_localized_resume_completion(
        plan_input=plan_input,
        initial_state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if completion is None or not completion.is_complete:
        return None
    combined_plan = [*prefix_plan, *clearing_plan, *completion.plan]
    result = SolverResult(
        plan=combined_plan,
        expanded_nodes=expanded_nodes + completion.expanded_nodes,
        generated_nodes=generated_nodes + completion.generated_nodes,
        closed_nodes=expanded_nodes + completion.closed_nodes,
        elapsed_ms=completion.elapsed_ms,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="route_blockage_tail_clearance",
        debug_stats=completion.debug_stats,
    )
    try:
        return _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=original_initial_state,
        )
    except Exception:  # noqa: BLE001
        return None


def _try_direct_blocked_tail_completion_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    clearing_plan: list[HookAction],
    state: ReplayState,
    initial_blockage: Any,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if state.loco_carry:
        return None
    facts_by_blocking_track = getattr(initial_blockage, "facts_by_blocking_track", {})
    if not facts_by_blocking_track:
        return None
    from fzed_shunting.solver.goal_logic import (
        goal_effective_allowed_tracks,
        goal_is_satisfied,
    )
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    direct_state = state
    direct_plan: list[HookAction] = []

    for fact in sorted(
        facts_by_blocking_track.values(),
        key=lambda item: (-getattr(item, "blockage_count", 0), getattr(item, "blocking_track", "")),
    ):
        for blocked_vehicle_no in getattr(fact, "blocked_vehicle_nos", []):
            if time_budget_ms - (perf_counter() - started_at) * 1000 <= 0:
                return None
            blocked_vehicle = vehicle_by_no.get(blocked_vehicle_no)
            if blocked_vehicle is None:
                continue
            track_by_vehicle = _vehicle_track_lookup(direct_state)
            source_track = track_by_vehicle.get(blocked_vehicle_no)
            if source_track is None:
                continue
            if goal_is_satisfied(
                blocked_vehicle,
                track_name=source_track,
                state=direct_state,
                plan_input=plan_input,
            ):
                continue
            source_seq = direct_state.track_sequences.get(source_track, [])
            try:
                source_index = source_seq.index(blocked_vehicle_no)
            except ValueError:
                continue
            source_prefix = source_seq[: source_index + 1]
            if source_prefix[-1] != blocked_vehicle_no:
                continue
            attach_move = _find_generated_move(
                generate_real_hook_moves(
                    plan_input,
                    direct_state,
                    master=master,
                    route_oracle=route_oracle,
                ),
                action_type="ATTACH",
                source_track=source_track,
                vehicle_nos=source_prefix,
            )
            if attach_move is None:
                continue
            next_state = _apply_move(
                state=direct_state,
                move=attach_move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            target_tracks = set(
                goal_effective_allowed_tracks(
                    blocked_vehicle,
                    state=next_state,
                    plan_input=plan_input,
                )
            )
            target_tracks.update(getattr(fact, "target_tracks", []))
            detach_blocked = _find_generated_move(
                generate_real_hook_moves(
                    plan_input,
                    next_state,
                    master=master,
                    route_oracle=route_oracle,
                ),
                action_type="DETACH",
                vehicle_nos=[blocked_vehicle_no],
                target_tracks=target_tracks,
            )
            if detach_blocked is None:
                continue
            next_state = _apply_move(
                state=next_state,
                move=detach_blocked,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            local_plan = [attach_move, detach_blocked]
            if next_state.loco_carry:
                restore_source = _find_generated_move(
                    generate_real_hook_moves(
                        plan_input,
                        next_state,
                        master=master,
                        route_oracle=route_oracle,
                    ),
                    action_type="DETACH",
                    target_track=source_track,
                    vehicle_nos=list(next_state.loco_carry),
                )
                if restore_source is None:
                    suffix_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
                    suffix_completion = _try_direct_tail_suffix_search(
                        plan_input=plan_input,
                        state=next_state,
                        master=master,
                        time_budget_ms=suffix_budget_ms,
                        enable_depot_late_scheduling=enable_depot_late_scheduling,
                    )
                    if suffix_completion is None:
                        continue
                    local_plan.extend(suffix_completion.plan)
                    expanded_nodes += suffix_completion.expanded_nodes
                    generated_nodes += suffix_completion.generated_nodes
                    complete_plan = [*prefix_plan, *clearing_plan, *direct_plan, *local_plan]
                    result = SolverResult(
                        plan=complete_plan,
                        expanded_nodes=expanded_nodes,
                        generated_nodes=generated_nodes,
                        closed_nodes=expanded_nodes,
                        elapsed_ms=(perf_counter() - started_at) * 1000,
                        is_complete=True,
                        is_proven_optimal=False,
                        fallback_stage="route_blockage_tail_clearance",
                        debug_stats=suffix_completion.debug_stats,
                    )
                    try:
                        return _attach_verification(
                            result,
                            plan_input=plan_input,
                            master=master,
                            initial_state=original_initial_state,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                else:
                    next_state = _apply_move(
                        state=next_state,
                        move=restore_source,
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                    )
                    local_plan.append(restore_source)
            if next_state.loco_carry:
                suffix_budget_ms = time_budget_ms - (perf_counter() - started_at) * 1000
                suffix_completion = _try_direct_tail_suffix_search(
                    plan_input=plan_input,
                    state=next_state,
                    master=master,
                    time_budget_ms=suffix_budget_ms,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
                if suffix_completion is None:
                    continue
                local_plan.extend(suffix_completion.plan)
                expanded_nodes += suffix_completion.expanded_nodes
                generated_nodes += suffix_completion.generated_nodes
                complete_plan = [*prefix_plan, *clearing_plan, *direct_plan, *local_plan]
                result = SolverResult(
                    plan=complete_plan,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    closed_nodes=expanded_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    is_complete=True,
                    is_proven_optimal=False,
                    fallback_stage="route_blockage_tail_clearance",
                    debug_stats=suffix_completion.debug_stats,
                )
                try:
                    return _attach_verification(
                        result,
                        plan_input=plan_input,
                        master=master,
                        initial_state=original_initial_state,
                    )
                except Exception:  # noqa: BLE001
                    continue
            direct_state = next_state
            direct_plan.extend(local_plan)

    if not direct_plan or direct_state.loco_carry:
        return None
    remaining_ms = time_budget_ms - (perf_counter() - started_at) * 1000
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    if _is_goal(plan_input, direct_state):
        completion_plan: list[HookAction] = []
    else:
        completion = _try_localized_resume_completion(
            plan_input=plan_input,
            initial_state=direct_state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if completion is None or not completion.is_complete:
            return None
        completion_plan = list(completion.plan)
        expanded_nodes += completion.expanded_nodes
        generated_nodes += completion.generated_nodes

    combined_plan = [*prefix_plan, *clearing_plan, *direct_plan, *completion_plan]
    result = SolverResult(
        plan=combined_plan,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        closed_nodes=expanded_nodes,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="route_blockage_tail_clearance",
    )
    try:
        return _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=original_initial_state,
        )
    except Exception:  # noqa: BLE001
        return None


def _try_direct_tail_suffix_search(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    try:
        suffix = _solve_search_result(
            plan_input=plan_input,
            initial_state=state,
            master=master,
            solver_mode="beam",
            heuristic_weight=1.0,
            beam_width=8,
            budget=SearchBudget(time_budget_ms=time_budget_ms),
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            enable_structural_diversity=True,
        )
    except ValueError:
        return None
    if not suffix.is_complete:
        return None
    return suffix


def _find_generated_move(
    moves: list[HookAction],
    *,
    action_type: str,
    vehicle_nos: list[str],
    source_track: str | None = None,
    target_track: str | None = None,
    target_tracks: set[str] | frozenset[str] | None = None,
) -> HookAction | None:
    for move in moves:
        if move.action_type != action_type:
            continue
        if source_track is not None and move.source_track != source_track:
            continue
        if target_track is not None and move.target_track != target_track:
            continue
        if target_tracks is not None and move.target_track not in target_tracks:
            continue
        if list(move.vehicle_nos) != list(vehicle_nos):
            continue
        return move
    return None


def _partial_route_blockage_pressure(
    partial_plan: list[HookAction],
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> int:
    if master is None or not partial_plan:
        return 0
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    state = initial_state
    try:
        for move in partial_plan:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
    except Exception:  # noqa: BLE001
        return 0
    if state.loco_carry:
        return 0
    return compute_route_blockage_plan(
        plan_input,
        state,
        RouteOracle(master),
    ).total_blockage_pressure


def _try_localized_resume_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if initial_state.loco_carry:
        return None
    purity = compute_state_purity(plan_input, initial_state)
    if purity.unfinished_count == 0:
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            is_proven_optimal=False,
            fallback_stage="localized_resume",
        )
    if purity.unfinished_count > LOCALIZED_RESUME_MAX_UNFINISHED:
        return None

    movable_vehicle_nos: set[str] = set()
    if master is not None:
        route_blockage_plan = compute_route_blockage_plan(
            plan_input,
            initial_state,
            RouteOracle(master),
        )
        movable_vehicle_nos.update(route_blockage_plan.blocking_vehicle_nos)
    localized_input = _build_repair_plan_input(
        plan_input,
        initial_state,
        movable_vehicle_nos=movable_vehicle_nos,
    )
    exact_completion: SolverResult | None = None
    if purity.unfinished_count <= 2:
        exact_budget_ms = min(5_000.0, max(500.0, time_budget_ms * 0.5))
        try:
            exact_completion = _solve_search_result(
                plan_input=localized_input,
                initial_state=initial_state,
                master=master,
                solver_mode="exact",
                heuristic_weight=1.0,
                beam_width=None,
                budget=SearchBudget(time_budget_ms=exact_budget_ms),
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
        except ValueError:
            exact_completion = None
        if exact_completion is not None and exact_completion.is_complete:
            return replace(exact_completion, fallback_stage="localized_resume_exact")

    beam_budget_ms = max(
        250.0,
        time_budget_ms - (exact_completion.elapsed_ms if exact_completion is not None else 0.0),
    )
    if beam_budget_ms <= 0:
        return None
    try:
        beam_completion = _solve_search_result(
            plan_input=localized_input,
            initial_state=initial_state,
            master=master,
            solver_mode="beam",
            heuristic_weight=1.0,
            beam_width=16,
            budget=SearchBudget(time_budget_ms=beam_budget_ms),
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            enable_structural_diversity=True,
        )
    except ValueError:
        return None
    if not beam_completion.is_complete:
        return None
    return replace(beam_completion, fallback_stage="localized_resume_beam")

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
    if solver_mode not in {"exact", "weighted", "beam", "lns", "real_hook"}:
        raise ValueError(f"Unsupported solver_mode: {solver_mode}")
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be >= 1.0")
    if beam_width is not None and beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    if solver_mode == "beam" and beam_width is None:
        raise ValueError("beam_width is required when solver_mode=beam")


def _normalize_solver_mode(solver_mode: str) -> str:
    if solver_mode == "real_hook":
        return "exact"
    return solver_mode



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
