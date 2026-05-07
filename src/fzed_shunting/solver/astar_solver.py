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
from fzed_shunting.solver.complete_selection import (
    complete_result_is_better,
    complete_result_quality_score,
)
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.lns import (
    _build_repair_plan_input,
    _improve_incumbent_result,
    _solve_with_lns_result,
)
from fzed_shunting.solver.move_candidates import generate_work_position_sequence_candidates
from fzed_shunting.solver.purity import STAGING_TRACKS, compute_state_purity
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
BEAM_COMPLETE_RESCUE_PRIMARY_BUDGET_MS = 3_000.0
BEAM_COMPLETE_SEED_MAX_PRIMARY_BUDGET_MS = 15_000.0
BEAM_COMPLETE_SEED_LONG_PLAN_MAX_PRIMARY_BUDGET_MS = 15_000.0
BEAM_COMPLETE_SEED_MIN_PRIMARY_BUDGET_MS = 8_000.0
BEAM_INCOMPLETE_SEED_PRIMARY_MIN_BUDGET_MS = 35_000.0
BEAM_POST_SEARCH_PRIMARY_MIN_BUDGET_MS = 20_000.0
BEAM_POST_SEARCH_PARTIAL_RESCUE_RESERVE_MS = 30_000.0
BEAM_ROUTE_TAIL_DETERMINISTIC_RESERVE_MS = 3_000.0
BEAM_POST_REPAIR_MIN_HOOKS = 40
BEAM_POST_REPAIR_MAX_BUDGET_MS = 3_000.0
PLAN_COMPRESSION_MAX_BUDGET_MS = 3_000.0
PLAN_COMPRESSION_MIN_BUDGET_MS = 250.0
PLAN_COMPRESSION_POSTPROCESS_RESERVE_MS = 500.0
COMPLETE_RESCUE_SKIP_PRIMARY_MAX_HOOKS = 80
COMPLETE_RESCUE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT = 20
PRE_PRIMARY_ROUTE_TAIL_MAX_COMPLETE_HOOK_FACTOR = 3.0
CONSTRUCTIVE_SKIP_PRIMARY_MAX_HOOKS = 100
CONSTRUCTIVE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT = 24
CONSTRUCTIVE_SKIP_PRIMARY_MAX_STAGING_TO_STAGING_HOOKS = 4
PRIMARY_PARTIAL_TAIL_COMPLETION_BUDGET_MS = 30_000.0
ROUTE_CLEAN_CONSTRUCTIVE_TAIL_MAX_BUDGET_MS = 8_000.0
CONSTRUCTIVE_STAGE_BUDGET_FRACTION = 0.25
CONSTRUCTIVE_STAGE_MAX_BUDGET_MS = 12_000.0
RELAXED_CONSTRUCTIVE_RETRY_BUDGET_MS = 8_000.0
ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS = 20_000.0
PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS = 30_000.0
ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD = 20
ROUTE_RELEASE_HIGH_PRESSURE_PRIMARY_MIN_BUDGET_MS = 2_000.0
ROUTE_BLOCKAGE_PRE_ROUTE_RELEASE_MAX_BUDGET_MS = 5_000.0
ROUTE_RELEASE_PARTIAL_MAX_EXTRA_HOOKS = 24
ROUTE_RELEASE_PARTIAL_MAX_HOOK_FACTOR = 3.0
ROUTE_RELEASE_PARTIAL_EXTRA_HOOKS_PER_PRESSURE_DROP = 1
EARLY_ROUTE_RELEASE_COMPLETION_BUDGET_MS = 12_000.0
ROUTE_RELEASE_TAIL_CHURNY_MAX_BUDGET_MS = EARLY_ROUTE_RELEASE_COMPLETION_BUDGET_MS
EARLY_ROUTE_RELEASE_MIN_REMAINING_MS = 25_000.0
PRE_PRIMARY_GOAL_FRONTIER_COMPLETION_BUDGET_MS = 15_000.0
PRE_PRIMARY_GOAL_FRONTIER_PRIMARY_RESERVE_MS = 5_000.0
LOCALIZED_RESUME_BUDGET_RATIO = 0.50
LOCALIZED_RESUME_MIN_FULL_BEAM_MS = 500.0
DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 4
RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 10
RELAXED_RESCUE_MAX_FINAL_HEURISTIC = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
NEAR_GOAL_PARTIAL_RESUME_MAX_BUDGET_MS = 60_000.0
NEAR_GOAL_PARTIAL_RESUME_BUDGET_RATIO = 2.0 / 3.0
BROAD_NEAR_GOAL_PRE_PRIMARY_RESUME_BUDGET_RATIO = 0.55
MIN_CHILD_STAGE_BUDGET_MS = 1.0
ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS = 12
ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS = 250.0
ROUTE_BLOCKAGE_TAIL_FOLLOWUP_RESERVE_MS = 5_000.0
ROUTE_RELEASE_CHECKPOINT_FOLLOWUP_RESERVE_MS = 2_000.0
COMPLETE_RESULT_FOLLOWUP_MIN_STAGING_TO_STAGING = 8
COMPLETE_RESULT_FOLLOWUP_MIN_VEHICLE_TOUCH_COUNT = 24


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
    if debug_stats is None:
        debug_stats = {}
    final_capacity_warnings = _final_track_goal_capacity_warnings(plan_input)
    if final_capacity_warnings and debug_stats is not None:
        debug_stats["final_capacity_warnings"] = final_capacity_warnings
    started_at = perf_counter()
    phase_timings: dict[str, float] = {
        "constructive_ms": 0.0,
        "exact_ms": 0.0,
        "anytime_ms": 0.0,
        "lns_ms": 0.0,
        "verify_ms": 0.0,
    }
    optimize_depot_late_in_search = enable_depot_late_scheduling
    reserve_primary_for_partial_rescue = (
        near_goal_partial_resume_max_final_heuristic
        <= DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )
    allow_deep_pre_primary_route_release = not reserve_primary_for_partial_rescue
    attempted_resume_partial_keys: set[tuple] = set()
    attempted_route_tail_partial_keys: set[tuple] = set()
    attempted_route_release_constructive = False

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
            budget_fraction=CONSTRUCTIVE_STAGE_BUDGET_FRACTION,
            max_budget_ms=CONSTRUCTIVE_STAGE_MAX_BUDGET_MS,
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
        and _effective_partial_route_blockage_pressure(
            constructive_seed,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        <= 0
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
            route_blockage_pressure = _effective_partial_route_blockage_pressure(
                constructive_seed,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            if (
                route_blockage_pressure > 0
                and remaining_ms >= EARLY_ROUTE_RELEASE_MIN_REMAINING_MS
                and _route_blocked_partial_should_try_tail_before_route_release(
                    constructive_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
                )
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
                        partial_plan=constructive_seed.partial_plan,
                    ),
                )
                if route_tail_budget is not None and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS:
                    route_tail_completion = _try_route_blocked_tail_completion(
                        constructive_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                        time_budget_ms=route_tail_budget,
                        enable_depot_late_scheduling=optimize_depot_late_in_search,
                    )
                    if route_tail_completion is not None and _accept_pre_primary_route_tail_completion(
                        result=route_tail_completion,
                        plan_input=plan_input,
                        reserve_primary=reserve_primary_for_partial_rescue,
                        baseline_partial_hook_count=len(constructive_seed.partial_plan),
                    ):
                        route_tail_completion = _with_pre_primary_tail_baseline(
                            route_tail_completion,
                            baseline_partial_hook_count=len(constructive_seed.partial_plan),
                        )
                        constructive_seed = _shorter_complete_result(
                            constructive_seed,
                            route_tail_completion,
                            plan_input=plan_input,
                            initial_state=initial_state,
                            master=master,
                        )
                remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
            if (
                route_blockage_pressure <= 0
                and not constructive_seed.is_complete
                and _partial_result_has_goal_frontier_pressure(
                    constructive_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                )
            ):
                goal_frontier_seed = _try_pre_primary_goal_frontier_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    constructive_seed=constructive_seed,
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                    solver_mode=solver_mode,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                )
                if goal_frontier_seed is not None:
                    constructive_seed = _shorter_complete_result(
                        constructive_seed,
                        goal_frontier_seed,
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
            resume_budget = _cap_broad_near_goal_pre_primary_resume_budget_ms(
                resume_budget,
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
                and not _partial_result_has_goal_frontier_pressure(
                    constructive_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                )
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
        and solver_mode == "beam"
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        route_blockage_pressure = _effective_partial_route_blockage_pressure(
            constructive_seed,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        if route_blockage_pressure > 0:
            should_try_route_tail_before_release = (
                route_blockage_pressure
                < ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD
                or
                _route_blocked_partial_should_try_tail_before_route_release(
                    constructive_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
                )
            )
            if (
                (
                    allow_deep_pre_primary_route_release
                    or route_blockage_pressure
                    >= ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD
                )
                and not _skip_route_release_constructive_for_near_goal_pressure(
                    constructive_seed,
                    route_blockage_pressure=route_blockage_pressure,
                    plan_input=plan_input,
                    initial_state=initial_state,
                )
            ):
                attempted_route_release_constructive = True
                _ts = perf_counter()
                route_release_seed = _try_pre_primary_route_release_constructive(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                    solver_mode=solver_mode,
                    reserve_primary=reserve_primary_for_partial_rescue,
                    near_goal_partial_resume_max_final_heuristic=(
                        near_goal_partial_resume_max_final_heuristic
                    ),
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                    attempted_resume_partial_keys=attempted_resume_partial_keys,
                    baseline_partial=constructive_seed,
                    route_blockage_pressure=route_blockage_pressure,
                )
                if route_release_seed is not None:
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
                        and _partial_replacement_candidate_improves(
                            route_release_seed,
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
            route_tail_signature = _partial_plan_signature(constructive_seed.partial_plan)
            route_tail_requested_budget = PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS
            if attempted_route_release_constructive:
                route_tail_requested_budget = _route_release_tail_budget_ms(
                    remaining_budget_ms=_remaining_wall_budget_ms(
                        started_at,
                        time_budget_ms,
                    )
                    or 0.0,
                    solver_mode=solver_mode,
                    reserve_primary=reserve_primary_for_partial_rescue,
                    partial_plan=constructive_seed.partial_plan,
                )
            route_tail_budget = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                route_tail_requested_budget,
            )
            if (
                not constructive_seed.is_complete
                and (
                    should_try_route_tail_before_release
                    or allow_deep_pre_primary_route_release
                    or attempted_route_release_constructive
                )
                and route_tail_budget is not None
                and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS
            ):
                if route_tail_signature not in attempted_route_tail_partial_keys:
                    attempted_route_tail_partial_keys.add(route_tail_signature)
                    _ts = perf_counter()
                    route_tail_completion = _try_route_blocked_tail_completion(
                        constructive_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                        time_budget_ms=route_tail_budget,
                        enable_depot_late_scheduling=optimize_depot_late_in_search,
                        allow_route_release_fallback=(
                            should_try_route_tail_before_release
                            or attempted_route_release_constructive
                        ),
                    )
                    if route_tail_completion is not None and _accept_pre_primary_route_tail_completion(
                        result=route_tail_completion,
                        plan_input=plan_input,
                        reserve_primary=reserve_primary_for_partial_rescue,
                        baseline_partial_hook_count=len(constructive_seed.partial_plan),
                    ):
                        route_tail_completion = _with_pre_primary_tail_baseline(
                            route_tail_completion,
                            baseline_partial_hook_count=len(constructive_seed.partial_plan),
                        )
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
                        relaxed_resume_budget = _cap_broad_near_goal_pre_primary_resume_budget_ms(
                            relaxed_resume_budget,
                            remaining_budget_ms=relaxed_remaining_ms,
                            max_final_heuristic=(
                                near_goal_partial_resume_max_final_heuristic
                            ),
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
                    and _partial_replacement_candidate_improves(
                        relaxed_candidate,
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
        and not attempted_route_release_constructive
    ):
        if not _skip_route_release_constructive_for_near_goal_pressure(
            constructive_seed,
            route_blockage_pressure=_effective_partial_route_blockage_pressure(
                constructive_seed,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            ),
            plan_input=plan_input,
            initial_state=initial_state,
        ):
            _ts = perf_counter()
            route_release_seed = _try_pre_primary_route_release_constructive(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                started_at=started_at,
                time_budget_ms=time_budget_ms,
                solver_mode=solver_mode,
                reserve_primary=reserve_primary_for_partial_rescue,
                near_goal_partial_resume_max_final_heuristic=(
                    near_goal_partial_resume_max_final_heuristic
                ),
                enable_depot_late_scheduling=optimize_depot_late_in_search,
                attempted_resume_partial_keys=attempted_resume_partial_keys,
                baseline_partial=constructive_seed,
                route_blockage_pressure=_effective_partial_route_blockage_pressure(
                    constructive_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                ),
            )
            if route_release_seed is not None:
                constructive_seed = _shorter_complete_result(
                    constructive_seed,
                    route_release_seed,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
            phase_timings["constructive_ms"] += (perf_counter() - _ts) * 1000

    # Stage 0.9: goal-frontier rescue. Once route pressure is gone, a common
    # hard tail is "already-satisfied north-end cars block unfinished cars
    # behind them". That is a local physical frontier problem, not a broad
    # global search problem, so try it before primary beam spends the SLA.
    if (
        enable_constructive_seed
        and enable_anytime_fallback
        and solver_mode == "beam"
        and constructive_seed is not None
        and not constructive_seed.is_complete
        and constructive_seed.partial_plan
        and time_budget_ms is not None
    ):
        _ts = perf_counter()
        goal_frontier_seed = _try_pre_primary_goal_frontier_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            constructive_seed=constructive_seed,
            started_at=started_at,
            time_budget_ms=time_budget_ms,
            solver_mode=solver_mode,
            enable_depot_late_scheduling=optimize_depot_late_in_search,
        )
        if goal_frontier_seed is not None:
            constructive_seed = _shorter_complete_result(
                constructive_seed,
                goal_frontier_seed,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
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
            started_at=started_at,
            time_budget_ms=time_budget_ms,
            has_final_capacity_warnings=bool(final_capacity_warnings),
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
                exact_time_budget_ms = _beam_complete_seed_budget_ms(
                    time_budget_ms=time_budget_ms,
                    constructive_seed=constructive_seed,
                )
            elif (
                solver_mode == "beam"
                and enable_anytime_fallback
                and time_budget_ms is not None
            ):
                exact_time_budget_ms = _beam_incomplete_seed_primary_budget_ms(
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                )
            exact_time_budget_ms = _cap_child_stage_budget_ms(
                started_at,
                time_budget_ms,
                exact_time_budget_ms,
            )
            exact_time_budget_ms = _reserve_post_primary_partial_tail_budget_ms(
                exact_time_budget_ms,
                result=constructive_seed,
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
                    if (
                        not result.is_complete
                        and result.partial_plan
                        and enable_anytime_fallback
                        and _has_child_stage_budget(started_at, time_budget_ms)
                    ):
                        _tail_ts = perf_counter()
                        tail_completion = _try_selected_partial_tail_completion(
                            result,
                            plan_input=plan_input,
                            initial_state=initial_state,
                            master=master,
                            started_at=started_at,
                            time_budget_ms=time_budget_ms,
                            requested_budget_ms=PRIMARY_PARTIAL_TAIL_COMPLETION_BUDGET_MS,
                            enable_depot_late_scheduling=optimize_depot_late_in_search,
                            include_near_goal_resume=solver_mode == "beam",
                            attempted_resume_partial_keys=attempted_resume_partial_keys,
                        )
                        if tail_completion is not None:
                            result = _shorter_complete_result(
                                result,
                                tail_completion,
                                plan_input=plan_input,
                                initial_state=initial_state,
                                master=master,
                            )
                        phase_timings["constructive_ms"] += (
                            perf_counter() - _tail_ts
                        ) * 1000
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
                        partial_plan=list(constructive_seed.partial_plan),
                        partial_fallback_stage=constructive_seed.partial_fallback_stage,
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
                started_at=started_at,
                time_budget_ms=time_budget_ms,
                has_final_capacity_warnings=bool(final_capacity_warnings),
            )
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            improved = result
            post_repair_budget_ms = _beam_post_repair_budget_ms(
                result,
                started_at=started_at,
                time_budget_ms=time_budget_ms,
            )
            if time_budget_ms is None or post_repair_budget_ms is not None:
                _ts = perf_counter()
                _lns_remaining_ms = post_repair_budget_ms
                for _ in range(BEAM_POST_REPAIR_MAX_ROUNDS):
                    if (
                        _lns_remaining_ms is not None
                        and _lns_remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS
                    ):
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
                    _lns_remaining_ms = _beam_post_repair_budget_ms(
                        improved,
                        started_at=started_at,
                        time_budget_ms=time_budget_ms,
                    )
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
                _tail_ts = perf_counter()
                tail_completion = _try_selected_partial_tail_completion(
                    result,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                    requested_budget_ms=PRIMARY_PARTIAL_TAIL_COMPLETION_BUDGET_MS,
                    enable_depot_late_scheduling=optimize_depot_late_in_search,
                    include_near_goal_resume=True,
                    attempted_resume_partial_keys=attempted_resume_partial_keys,
                )
                if tail_completion is not None:
                    result = _shorter_complete_result(
                        result,
                        tail_completion,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                phase_timings["constructive_ms"] += (perf_counter() - _tail_ts) * 1000
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
            and _partial_replacement_candidate_improves(
                constructive_seed,
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
        and solver_mode == "beam"
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
            and _has_child_stage_budget(started_at, time_budget_ms)
        ):
            _ts = perf_counter()
            tail_completion = _try_selected_partial_tail_completion(
                result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                started_at=started_at,
                time_budget_ms=time_budget_ms,
                requested_budget_ms=PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
                enable_depot_late_scheduling=optimize_depot_late_in_search,
                include_near_goal_resume=not partial_has_route_pressure,
                attempted_resume_partial_keys=attempted_resume_partial_keys,
            )
            if tail_completion is not None:
                result = _shorter_complete_result(
                    result,
                    tail_completion,
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
                force_near_goal=True,
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
            remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
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
        started_at=started_at,
        time_budget_ms=time_budget_ms,
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
    started_at: float | None = None,
    time_budget_ms: float | None = None,
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

    compression_budget_ms = _plan_compression_budget_ms(
        started_at=started_at,
        time_budget_ms=time_budget_ms,
    )
    if (
        compression_budget_ms is not None
        and (
            compression_budget_ms < PLAN_COMPRESSION_MIN_BUDGET_MS
            or (
                result.fallback_stage
                in {
                    "constructive_route_release_tail",
                    "route_blockage_tail_clearance",
                }
            )
        )
    ):
        stats["plan_compression"] = {
            "accepted_rewrite_count": 0,
            "before_hook_count": len(result.plan),
            "after_hook_count": len(result.plan),
        }
        return replace(result, debug_stats=stats)
    compressed = compress_plan(
        plan_input,
        initial_state,
        result.plan,
        master=master,
        time_budget_ms=compression_budget_ms,
    )
    stats["plan_compression"] = {
        "accepted_rewrite_count": compressed.accepted_rewrite_count,
        "before_hook_count": len(result.plan),
        "after_hook_count": len(compressed.compressed_plan),
    }
    if len(compressed.compressed_plan) < len(result.plan):
        return replace(result, plan=compressed.compressed_plan, debug_stats=stats)
    return replace(result, debug_stats=stats)


def _plan_compression_budget_ms(
    *,
    started_at: float | None,
    time_budget_ms: float | None,
) -> float | None:
    if started_at is None or time_budget_ms is None:
        return PLAN_COMPRESSION_MAX_BUDGET_MS
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is None:
        return PLAN_COMPRESSION_MAX_BUDGET_MS
    spendable_ms = max(0.0, remaining_ms - PLAN_COMPRESSION_POSTPROCESS_RESERVE_MS)
    return min(PLAN_COMPRESSION_MAX_BUDGET_MS, spendable_ms)


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
            stats[route_key] = _debug_dict(
                compute_route_blockage_plan(
                    plan_input,
                    final_state,
                    RouteOracle(master),
                )
            )
    return replace(result, debug_stats=stats)


def _debug_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(getattr(value, "__dict__", {}))


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
    work_position_count = sum(
        1 for v in plan_input.vehicles if v.goal.work_position_kind is not None
    )
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
        input_work_position_count=work_position_count,
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
    if constructive_seed.fallback_stage in {"constructive_route_release_tail"}:
        return min(time_budget_ms, BEAM_COMPLETE_RESCUE_PRIMARY_BUDGET_MS)
    if _complete_seed_primary_search_is_low_yield(constructive_seed):
        return min(time_budget_ms, BEAM_COMPLETE_RESCUE_PRIMARY_BUDGET_MS)
    ratio = max(
        BEAM_COMPLETE_SEED_MIN_BUDGET_RATIO,
        min(1.0, seed_hook_count / BEAM_COMPLETE_SEED_FULL_BUDGET_HOOKS),
    )
    return min(
        time_budget_ms,
        (
            BEAM_COMPLETE_SEED_LONG_PLAN_MAX_PRIMARY_BUDGET_MS
            if seed_hook_count >= BEAM_COMPLETE_SEED_FULL_BUDGET_HOOKS
            else BEAM_COMPLETE_SEED_MAX_PRIMARY_BUDGET_MS
        ),
        max(1.0, time_budget_ms * ratio),
    )


def _beam_post_repair_budget_ms(
    result: SolverResult,
    *,
    started_at: float,
    time_budget_ms: float | None,
) -> float | None:
    if not result.is_complete:
        return None
    if time_budget_ms is not None and len(result.plan) < BEAM_POST_REPAIR_MIN_HOOKS:
        return None
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is not None:
        remaining_ms = min(remaining_ms, BEAM_POST_REPAIR_MAX_BUDGET_MS)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
    return remaining_ms


def _beam_incomplete_seed_primary_budget_ms(
    *,
    started_at: float,
    time_budget_ms: float,
) -> float:
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms) or 0.0
    if remaining_ms <= BEAM_POST_SEARCH_PRIMARY_MIN_BUDGET_MS:
        return remaining_ms
    rescue_reserve_ms = min(
        BEAM_POST_SEARCH_PARTIAL_RESCUE_RESERVE_MS,
        remaining_ms - BEAM_POST_SEARCH_PRIMARY_MIN_BUDGET_MS,
    )
    return max(1.0, remaining_ms - rescue_reserve_ms)


def _reserve_post_primary_partial_tail_budget_ms(
    requested_budget_ms: float | None,
    *,
    result: SolverResult | None,
) -> float | None:
    return requested_budget_ms


def _accept_pre_primary_route_tail_completion(
    *,
    result: SolverResult,
    plan_input: NormalizedPlanInput,
    reserve_primary: bool,
    baseline_partial_hook_count: int,
) -> bool:
    if (
        reserve_primary
        and result.is_complete
        and result.fallback_stage
        in {"constructive_route_release_tail", "goal_frontier_tail_completion"}
        and len(result.plan)
        > max(
            baseline_partial_hook_count + 2,
            int(
                max(1, baseline_partial_hook_count)
                * PRE_PRIMARY_ROUTE_TAIL_MAX_COMPLETE_HOOK_FACTOR
            ),
        )
    ):
        return False
    return True


def _with_pre_primary_tail_baseline(
    result: SolverResult,
    *,
    baseline_partial_hook_count: int,
) -> SolverResult:
    stats = dict(result.debug_stats or {})
    stats["pre_primary_tail_baseline_hook_count"] = baseline_partial_hook_count
    return replace(result, debug_stats=stats)


def _should_skip_primary_after_complete_rescue(
    *,
    solver_mode: str,
    constructive_seed: SolverResult | None,
    started_at: float | None = None,
    time_budget_ms: float | None = None,
    has_final_capacity_warnings: bool = False,
) -> bool:
    if solver_mode != "beam" or constructive_seed is None or not constructive_seed.is_complete:
        return False
    if has_final_capacity_warnings:
        return False
    if constructive_seed.fallback_stage not in {
        "constructive",
        "constructive_partial_resume",
        "constructive_warm_start",
        "constructive_route_release",
        "route_blockage_tail_clearance",
        "goal_frontier_tail_completion",
    }:
        return False
    if constructive_seed.fallback_stage == "goal_frontier_tail_completion":
        baseline_hook_count = _optional_int(
            (constructive_seed.debug_stats or {}).get(
                "pre_primary_tail_baseline_hook_count"
            )
        )
        if (
            baseline_hook_count is not None
            and len(constructive_seed.plan)
            > max(
                baseline_hook_count + 2,
                int(
                    max(1, baseline_hook_count)
                    * PRE_PRIMARY_ROUTE_TAIL_MAX_COMPLETE_HOOK_FACTOR
                ),
            )
        ):
            return False
        return len(constructive_seed.plan) <= 3 or _complete_seed_primary_search_is_low_yield(
            constructive_seed,
            started_at=started_at,
            time_budget_ms=time_budget_ms,
        )
    if _complete_seed_primary_search_is_low_yield(
        constructive_seed,
        started_at=started_at,
        time_budget_ms=time_budget_ms,
    ):
        return True
    return _complete_rescue_is_light_enough_to_skip_primary(constructive_seed)


def _complete_seed_primary_search_is_low_yield(
    result: SolverResult,
    *,
    started_at: float | None = None,
    time_budget_ms: float | None = None,
) -> bool:
    if result.fallback_stage not in {
        "route_blockage_tail_clearance",
        "goal_frontier_tail_completion",
    }:
        return False
    if started_at is None or time_budget_ms is None:
        return False
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    return remaining_ms is not None and remaining_ms < BEAM_COMPLETE_SEED_MIN_PRIMARY_BUDGET_MS


def _complete_rescue_is_light_enough_to_skip_primary(result: SolverResult) -> bool:
    """Return True when a complete rescue is a clean finish, not a churn basin."""

    max_hooks = (
        CONSTRUCTIVE_SKIP_PRIMARY_MAX_HOOKS
        if result.fallback_stage == "constructive"
        else COMPLETE_RESCUE_SKIP_PRIMARY_MAX_HOOKS
    )
    max_touch_threshold = (
        CONSTRUCTIVE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT
        if result.fallback_stage == "constructive"
        else COMPLETE_RESCUE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT
    )
    if len(result.plan) > max_hooks:
        return False
    shape = ((result.debug_stats or {}).get("plan_shape_metrics") or summarize_plan_shape(result.plan))
    max_touch = _optional_int(shape.get("max_vehicle_touch_count"))
    if (
        max_touch is not None
        and max_touch > max_touch_threshold
    ):
        return False
    if result.fallback_stage == "constructive":
        staging_to_staging = _optional_int(shape.get("staging_to_staging_hook_count"))
        if (
            staging_to_staging is not None
            and staging_to_staging > CONSTRUCTIVE_SKIP_PRIMARY_MAX_STAGING_TO_STAGING_HOOKS
        ):
            return False
    return True


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
    partial_plan: list[HookAction] | None = None,
) -> float:
    remaining_budget_ms = max(0.0, remaining_budget_ms)
    _ = solver_mode, reserve_primary
    max_budget_ms = (
        ROUTE_RELEASE_TAIL_CHURNY_MAX_BUDGET_MS
        if _partial_plan_is_churny_for_route_tail(partial_plan)
        else PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS
    )
    return min(max_budget_ms, max(500.0, remaining_budget_ms))


def _partial_plan_is_churny_for_route_tail(plan: list[HookAction] | None) -> bool:
    if not plan:
        return False
    shape = summarize_plan_shape(plan)
    return (
        _optional_int(shape.get("max_vehicle_touch_count")) or 0
    ) > COMPLETE_RESCUE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT or (
        _optional_int(shape.get("staging_to_staging_hook_count")) or 0
    ) >= 8


def _route_release_checkpoint_budget_ms(
    *,
    remaining_budget_ms: float,
    remaining_attempts: int,
) -> float:
    if remaining_attempts <= 1:
        return min(
            remaining_budget_ms,
            PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
        )
    followup_reserve = min(
        remaining_budget_ms * 0.5,
        ROUTE_RELEASE_CHECKPOINT_FOLLOWUP_RESERVE_MS * (remaining_attempts - 1),
    )
    return min(
        max(MIN_CHILD_STAGE_BUDGET_MS, remaining_budget_ms - followup_reserve),
        PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS,
    )


def _try_pre_primary_route_release_constructive(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    started_at: float,
    time_budget_ms: float | None,
    solver_mode: str,
    reserve_primary: bool,
    near_goal_partial_resume_max_final_heuristic: int,
    enable_depot_late_scheduling: bool,
    attempted_resume_partial_keys: set[tuple],
    baseline_partial: SolverResult | None = None,
    route_blockage_pressure: int | None = None,
) -> SolverResult | None:
    route_release_budget = _route_release_constructive_budget_ms(
        started_at=started_at,
        time_budget_ms=time_budget_ms,
        solver_mode=solver_mode,
        reserve_primary=reserve_primary,
        route_blockage_pressure=route_blockage_pressure,
    )
    if route_release_budget is None or route_release_budget <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    route_release_seed = _run_constructive_stage(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        time_budget_ms=route_release_budget,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
        strict_staging_regrab=True,
        route_release_bias=True,
        budget_fraction=1.0,
        max_budget_ms=ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS,
    )
    if route_release_seed is None:
        return None
    if route_release_seed.is_complete:
        return replace(route_release_seed, fallback_stage="constructive_route_release")
    if not route_release_seed.partial_plan:
        return route_release_seed
    improves_baseline = (
        baseline_partial is None
        or _route_release_partial_improves_baseline(
            route_release_seed,
            baseline_partial=baseline_partial,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
    )
    if not _route_release_partial_is_bounded_improvement(
        route_release_seed,
        baseline_partial=baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return None
    route_release_pressure = _partial_result_route_blockage_pressure(
        route_release_seed,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    if route_release_pressure <= 0:
        goal_frontier_completion = _try_pre_primary_goal_frontier_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            constructive_seed=route_release_seed,
            started_at=started_at,
            time_budget_ms=time_budget_ms,
            solver_mode=solver_mode,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if goal_frontier_completion is not None:
            return goal_frontier_completion
    if route_release_pressure > 0:
        if improves_baseline and _route_release_partial_should_preserve_primary_budget(
            route_release_seed,
            baseline_partial=baseline_partial,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ):
            return route_release_seed
        if (route_release_seed.partial_fallback_stage or "") == "constructive_route_release":
            route_release_remaining_ms = _remaining_wall_budget_ms(
                started_at,
                time_budget_ms,
            )
            if (
                route_release_remaining_ms is not None
                and route_release_remaining_ms
                > ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
            ):
                route_tail_budget = _cap_child_stage_budget_ms(
                    started_at,
                    time_budget_ms,
                    _route_release_tail_budget_ms(
                        remaining_budget_ms=route_release_remaining_ms,
                        solver_mode=solver_mode,
                        reserve_primary=reserve_primary,
                        partial_plan=route_release_seed.partial_plan,
                    ),
                )
                if (
                    route_tail_budget is not None
                    and route_tail_budget > MIN_CHILD_STAGE_BUDGET_MS
                ):
                    attempted_resume_partial_keys.add(
                        _partial_plan_signature(route_release_seed.partial_plan)
                    )
                    route_tail_completion = _try_route_blocked_tail_completion(
                        route_release_seed,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                        time_budget_ms=route_tail_budget,
                        enable_depot_late_scheduling=enable_depot_late_scheduling,
                    )
                    if (
                        route_tail_completion is not None
                        and (
                            route_tail_completion.is_complete
                            or route_tail_completion.partial_plan
                        )
                    ):
                        return route_tail_completion
        return route_release_seed if improves_baseline else None
    if improves_baseline and _route_release_partial_should_preserve_primary_budget(
        route_release_seed,
        baseline_partial=baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return route_release_seed
    route_release_remaining_ms = _remaining_wall_budget_ms(
        started_at,
        time_budget_ms,
    )
    if (
        route_release_remaining_ms is not None
        and route_release_remaining_ms > ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
    ):
        route_release_tail_budget = _cap_child_stage_budget_ms(
            started_at,
            time_budget_ms,
            _route_release_tail_budget_ms(
                remaining_budget_ms=route_release_remaining_ms,
                solver_mode=solver_mode,
                reserve_primary=reserve_primary,
                partial_plan=route_release_seed.partial_plan,
            ),
        )
        if (
            route_release_tail_budget is not None
            and route_release_tail_budget > MIN_CHILD_STAGE_BUDGET_MS
        ):
            route_release_completion = _try_route_release_partial_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                partial_plan=route_release_seed.partial_plan,
                master=master,
                time_budget_ms=route_release_tail_budget,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if (
                route_release_completion is not None
                and (
                    route_release_completion.is_complete
                    or route_release_completion.partial_plan
                )
            ):
                if (
                    not route_release_completion.is_complete
                    and not _route_release_partial_is_bounded_improvement(
                        route_release_completion,
                        baseline_partial=baseline_partial,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                ):
                    return None
                return route_release_completion
    if route_release_pressure > 0:
        return route_release_seed if improves_baseline else None
    if _partial_result_loco_carry_count(
        route_release_seed,
        plan_input=plan_input,
        initial_state=initial_state,
    ) > 0:
        return route_release_seed if improves_baseline else None
    route_release_remaining_ms = _remaining_wall_budget_ms(
        started_at,
        time_budget_ms,
    )
    if route_release_remaining_ms is None or route_release_remaining_ms <= 250:
        return route_release_seed if improves_baseline else None
    route_release_resume_budget = _partial_resume_budget_ms(
        route_release_seed,
        remaining_budget_ms=route_release_remaining_ms,
        max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
    )
    route_release_resume_budget = _cap_broad_near_goal_pre_primary_resume_budget_ms(
        route_release_resume_budget,
        remaining_budget_ms=route_release_remaining_ms,
        max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
    )
    route_release_resume_budget = _cap_pre_primary_child_stage_budget_ms(
        started_at,
        time_budget_ms,
        route_release_resume_budget,
        solver_mode=solver_mode,
        reserve_primary=reserve_primary,
    )
    if (
        route_release_resume_budget is None
        or route_release_resume_budget <= MIN_CHILD_STAGE_BUDGET_MS
    ):
        return route_release_seed if improves_baseline else None
    attempted_resume_partial_keys.add(
        _partial_plan_signature(route_release_seed.partial_plan)
    )
    route_release_resume = _try_resume_partial_completion(
        plan_input=plan_input,
        initial_state=initial_state,
        constructive_plan=route_release_seed.partial_plan,
        master=master,
        time_budget_ms=route_release_resume_budget,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if route_release_resume is not None:
        return route_release_resume
    return route_release_seed if improves_baseline else None


def _route_release_constructive_budget_ms(
    *,
    started_at: float,
    time_budget_ms: float | None,
    solver_mode: str,
    reserve_primary: bool,
    route_blockage_pressure: int | None,
) -> float | None:
    if (
        not reserve_primary
        or solver_mode != "beam"
        or route_blockage_pressure is None
        or route_blockage_pressure < ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD
    ):
        return _cap_pre_primary_child_stage_budget_ms(
            started_at,
            time_budget_ms,
            ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS,
            solver_mode=solver_mode,
            reserve_primary=reserve_primary,
        )
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is None:
        return ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS
    spendable_ms = max(
        0.0,
        remaining_ms - min(
            remaining_ms,
            ROUTE_RELEASE_HIGH_PRESSURE_PRIMARY_MIN_BUDGET_MS,
        ),
    )
    return min(ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS, spendable_ms)


def _try_pre_primary_goal_frontier_completion(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    constructive_seed: SolverResult,
    started_at: float,
    time_budget_ms: float | None,
    solver_mode: str,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if master is None or time_budget_ms is None or not constructive_seed.partial_plan:
        return None
    if not _partial_result_has_goal_frontier_pressure(
        constructive_seed,
        plan_input=plan_input,
        initial_state=initial_state,
    ):
        return None
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    if remaining_ms is None or remaining_ms <= PRE_PRIMARY_GOAL_FRONTIER_PRIMARY_RESERVE_MS:
        return None
    requested_budget = min(
        PRE_PRIMARY_GOAL_FRONTIER_COMPLETION_BUDGET_MS,
        remaining_ms - PRE_PRIMARY_GOAL_FRONTIER_PRIMARY_RESERVE_MS,
    )
    goal_frontier_budget = _cap_child_stage_budget_ms(
        started_at,
        time_budget_ms,
        requested_budget,
    )
    if goal_frontier_budget is None or goal_frontier_budget <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    completion = _try_goal_frontier_tail_completion(
        plan_input=plan_input,
        initial_state=initial_state,
        partial_plan=constructive_seed.partial_plan,
        master=master,
        time_budget_ms=goal_frontier_budget,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if (
        completion is not None
        and completion.is_complete
        and len(completion.plan)
        > max(
            len(constructive_seed.partial_plan) + 2,
            int(
                max(1, len(constructive_seed.partial_plan))
                * PRE_PRIMARY_ROUTE_TAIL_MAX_COMPLETE_HOOK_FACTOR
            ),
        )
    ):
        return None
    return completion


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


def _route_release_partial_improves_baseline(
    candidate: SolverResult,
    *,
    baseline_partial: SolverResult,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    candidate_score = _partial_result_score(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    baseline_score = _partial_result_score(
        baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    return candidate_score < baseline_score


def _route_release_partial_is_bounded_improvement(
    candidate: SolverResult,
    *,
    baseline_partial: SolverResult | None,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    if candidate.is_complete:
        return True
    if not candidate.partial_plan:
        return False
    if baseline_partial is None or not baseline_partial.partial_plan:
        return True
    if not _route_release_partial_improves_baseline(
        candidate,
        baseline_partial=baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return False
    baseline_len = len(baseline_partial.partial_plan)
    candidate_len = len(candidate.partial_plan)
    baseline_pressure = _partial_route_pressure_for_bounded_improvement(
        baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    candidate_pressure = _partial_route_pressure_for_bounded_improvement(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    material_route_release_extra = 0
    if baseline_pressure is not None and candidate_pressure is not None:
        pressure_drop = max(0, baseline_pressure - candidate_pressure)
        if pressure_drop >= ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD:
            material_route_release_extra = (
                pressure_drop * ROUTE_RELEASE_PARTIAL_EXTRA_HOOKS_PER_PRESSURE_DROP
            )
    max_len = max(
        baseline_len + ROUTE_RELEASE_PARTIAL_MAX_EXTRA_HOOKS,
        int(baseline_len * ROUTE_RELEASE_PARTIAL_MAX_HOOK_FACTOR),
    ) + material_route_release_extra
    return candidate_len <= max_len


def _partial_route_pressure_from_debug(result: SolverResult) -> int | None:
    route_blockage = (result.debug_stats or {}).get("partial_route_blockage_plan") or {}
    return _optional_int(route_blockage.get("total_blockage_pressure"))


def _partial_route_pressure_for_bounded_improvement(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> int | None:
    pressure = _partial_route_pressure_from_debug(result)
    if pressure is not None:
        return pressure
    if master is None or not result.partial_plan:
        return None
    return _partial_route_blockage_pressure(
        result.partial_plan,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )


def _route_release_partial_should_preserve_primary_budget(
    candidate: SolverResult,
    *,
    baseline_partial: SolverResult | None,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    if candidate.is_complete or not candidate.partial_plan:
        return False
    if baseline_partial is None or not baseline_partial.partial_plan:
        return False
    baseline_pressure = _partial_route_pressure_for_bounded_improvement(
        baseline_partial,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    candidate_pressure = _partial_route_pressure_for_bounded_improvement(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    if baseline_pressure is None or candidate_pressure is None:
        return False
    pressure_drop = baseline_pressure - candidate_pressure
    if pressure_drop < ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD:
        return False
    return candidate_pressure <= max(4, baseline_pressure // 5)


def _skip_route_release_constructive_for_near_goal_pressure(
    result: SolverResult,
    *,
    route_blockage_pressure: int,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> bool:
    if route_blockage_pressure < ROUTE_RELEASE_CONSTRUCTIVE_HIGH_PRESSURE_SKIP_THRESHOLD:
        return False
    if result.is_complete or not result.partial_plan:
        return False
    final_h = _solver_result_final_heuristic(result)
    if (
        final_h is not None
        and final_h <= RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    ):
        return True
    return _partial_result_is_near_goal(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    )


def _route_blocked_partial_should_try_tail_before_route_release(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    max_final_heuristic: int,
) -> bool:
    final_h = _solver_result_final_heuristic(result)
    if final_h is not None and final_h <= max_final_heuristic:
        return True
    if _partial_result_has_goal_frontier_pressure(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    ):
        return True
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    staging_debt = _optional_int(structural.get("staging_debt_count")) or 0
    capacity_overflow = _optional_int(
        structural.get("capacity_overflow_track_count")
    ) or 0
    if staging_debt <= 0 and capacity_overflow <= 0:
        return False
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    unfinished = _optional_int(structural.get("unfinished_count"))
    if unfinished is None and result.partial_plan:
        final_state = _replay_solver_moves(
            plan_input=plan_input,
            initial_state=initial_state,
            plan=result.partial_plan,
        )
        if final_state is not None:
            unfinished = compute_structural_metrics(
                plan_input,
                final_state,
            ).unfinished_count
    return unfinished is not None and unfinished <= LOCALIZED_RESUME_MAX_UNFINISHED


def _effective_partial_route_blockage_pressure(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> int:
    debug_route_blockage = _optional_int(
        ((result.debug_stats or {}).get("partial_route_blockage_plan") or {}).get(
            "total_blockage_pressure"
        )
    )
    replayed_route_blockage = _partial_route_blockage_pressure(
        result.partial_plan,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    return max(replayed_route_blockage, debug_route_blockage or 0)


def _rank_route_release_checkpoints(
    checkpoints: list[tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState]],
    *,
    max_checkpoints: int = PARTIAL_ROUTE_RELEASE_MAX_CHECKPOINTS,
) -> list[tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState]]:
    """Rank route-tail checkpoints by global plan progress before route pressure.

    The tail completer is already invoked because route pressure exists. A
    checkpoint with slightly lower pressure but many unfinished cars tends to
    produce broad rehandling; prefer states that have already placed most cars,
    then use pressure as the tiebreaker.
    """

    ranked: list[tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState]] = []
    seen_prefix_lengths: set[int] = set()

    def progress_key(
        checkpoint: tuple[tuple[int, int, int, int], int, list[HookAction], ReplayState],
    ) -> tuple[int, int, int, int, int]:
        route_pressure, goal_blockers, staging_debt, unfinished = checkpoint[0]
        prefix_len = -checkpoint[1]
        route_clean_rank = 0 if route_pressure <= 0 else 1
        return (
            route_clean_rank,
            unfinished + max(0, route_pressure - 3),
            goal_blockers,
            staging_debt,
            route_pressure,
            -prefix_len,
        )

    for checkpoint in sorted(checkpoints, key=progress_key):
        prefix_len = -checkpoint[1]
        if prefix_len in seen_prefix_lengths:
            continue
        seen_prefix_lengths.add(prefix_len)
        ranked.append(checkpoint)
        if len(ranked) >= max_checkpoints:
            break
    return ranked


def _has_child_stage_budget(started_at: float, time_budget_ms: float | None) -> bool:
    remaining_ms = _remaining_wall_budget_ms(started_at, time_budget_ms)
    return remaining_ms is None or remaining_ms > MIN_CHILD_STAGE_BUDGET_MS


def _remaining_child_budget_ms(started_at: float, time_budget_ms: float) -> float:
    return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000)


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
    force_near_goal: bool = False,
) -> float:
    remaining_budget_ms = max(0.0, remaining_budget_ms)
    final_h = _solver_result_final_heuristic(seed)
    if force_near_goal or (final_h is not None and final_h <= max_final_heuristic):
        return min(
            NEAR_GOAL_PARTIAL_RESUME_MAX_BUDGET_MS,
            max(500.0, remaining_budget_ms * NEAR_GOAL_PARTIAL_RESUME_BUDGET_RATIO),
        )
    return min(5_000.0, max(500.0, remaining_budget_ms * 0.20))


def _cap_broad_near_goal_pre_primary_resume_budget_ms(
    requested_budget_ms: float,
    *,
    remaining_budget_ms: float,
    max_final_heuristic: int,
) -> float:
    if max_final_heuristic <= DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC:
        return requested_budget_ms
    remaining_budget_ms = max(0.0, remaining_budget_ms)
    return min(
        requested_budget_ms,
        max(
            500.0,
            remaining_budget_ms * BROAD_NEAR_GOAL_PRE_PRIMARY_RESUME_BUDGET_RATIO,
        ),
    )


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
        if (
            not incumbent.is_complete
            and plan_input is not None
            and initial_state is not None
            and _partial_tail_candidate_improves(
                candidate,
                incumbent,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
        ):
            return candidate
        if (
            not incumbent.is_complete
            and plan_input is None
            and initial_state is None
            and _partial_result_score(candidate, master=master)
            < _partial_result_score(incumbent, master=master)
        ):
            return candidate
        return incumbent
    if not incumbent.is_complete:
        return candidate
    if _complete_result_is_better(candidate, incumbent):
        return candidate
    return incumbent


def _complete_result_is_better(candidate: SolverResult, incumbent: SolverResult) -> bool:
    return complete_result_is_better(candidate, incumbent)


def _complete_result_needs_quality_followup(result: SolverResult) -> bool:
    staging_to_staging, max_touch_count = _complete_result_quality_score(result)
    return (
        staging_to_staging >= COMPLETE_RESULT_FOLLOWUP_MIN_STAGING_TO_STAGING
        or max_touch_count >= COMPLETE_RESULT_FOLLOWUP_MIN_VEHICLE_TOUCH_COUNT
    )


def _complete_result_quality_score(result: SolverResult) -> tuple[int, int]:
    return complete_result_quality_score(result.debug_stats, result.plan)


def _as_route_blockage_tail_completion(result: SolverResult) -> SolverResult:
    if not result.is_complete:
        return result
    if result.fallback_stage not in {None, "beam"}:
        return result
    return replace(result, fallback_stage="route_blockage_tail_clearance")


def _try_selected_partial_tail_completion(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    started_at: float,
    time_budget_ms: float | None,
    requested_budget_ms: float,
    enable_depot_late_scheduling: bool,
    include_near_goal_resume: bool = False,
    attempted_resume_partial_keys: set[tuple] | None = None,
) -> SolverResult | None:
    if result.is_complete or not result.partial_plan:
        return None
    tail_budget = _cap_child_stage_budget_ms(
        started_at,
        time_budget_ms,
        requested_budget_ms,
    )
    if tail_budget is None or tail_budget <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    route_pressure = _partial_result_route_blockage_pressure(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    if (
        include_near_goal_resume
        and route_pressure <= 0
        and not _partial_result_has_goal_frontier_pressure(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        )
        and _partial_result_is_near_goal(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        )
    ):
        signature = _partial_plan_signature(result.partial_plan)
        if attempted_resume_partial_keys is None or signature not in attempted_resume_partial_keys:
            if attempted_resume_partial_keys is not None:
                attempted_resume_partial_keys.add(signature)
            resume_completion = _try_resume_partial_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                constructive_plan=result.partial_plan,
                master=master,
                time_budget_ms=tail_budget,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if resume_completion is not None:
                return resume_completion
    if (
        route_pressure <= 0
        and _partial_result_has_goal_frontier_pressure(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        )
    ):
        goal_frontier_completion = _try_goal_frontier_tail_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            partial_plan=result.partial_plan,
            master=master,
            time_budget_ms=tail_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if goal_frontier_completion is not None:
            return goal_frontier_completion
    if _partial_result_loco_carry_count(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    ) > 0:
        route_release_completion = _try_route_blocked_tail_completion(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=tail_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if route_release_completion is not None and route_release_completion.is_complete:
            return route_release_completion
    return _try_partial_tail_fixed_point_completion(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        time_budget_ms=tail_budget,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _try_route_blocked_tail_completion(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
    allow_route_release_fallback: bool = True,
) -> SolverResult | None:
    if result.is_complete or not result.partial_plan:
        return None
    has_loco_carry = _partial_result_loco_carry_count(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    ) > 0
    if has_loco_carry and not _route_blocked_partial_prefers_blockage_clearance(result):
        release_completion = _try_route_release_partial_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            partial_plan=result.partial_plan,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if release_completion is not None:
            return release_completion
    route_tail_completion = _try_route_blockage_tail_clearance_completion(
        plan_input=plan_input,
        initial_state=initial_state,
        partial_plan=result.partial_plan,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if route_tail_completion is not None and (
        route_tail_completion.is_complete or route_tail_completion.partial_plan
    ):
        return route_tail_completion
    if has_loco_carry and not _route_blocked_partial_prefers_blockage_clearance(result):
        return None
    if not allow_route_release_fallback:
        return None
    return _try_route_release_partial_completion(
        plan_input=plan_input,
        initial_state=initial_state,
        partial_plan=result.partial_plan,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _route_blocked_partial_prefers_blockage_clearance(result: SolverResult) -> bool:
    final_h = _solver_result_final_heuristic(result)
    if (
        final_h is not None
        and final_h <= RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    ):
        return True
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    unfinished = _optional_int(structural.get("unfinished_count"))
    staging_debt = _optional_int(structural.get("staging_debt_count")) or 0
    capacity_overflow = _optional_int(
        structural.get("capacity_overflow_track_count")
    ) or 0
    return (
        unfinished is not None
        and unfinished <= LOCALIZED_RESUME_MAX_UNFINISHED
        and (staging_debt > 0 or capacity_overflow > 0)
    )


def _try_partial_tail_fixed_point_completion(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
    max_rounds: int = 4,
) -> SolverResult | None:
    current = result
    best: SolverResult | None = None
    started_at = perf_counter()
    seen_state_keys: set[tuple] = set()
    current_key = _partial_result_state_key(
        current,
        plan_input=plan_input,
        initial_state=initial_state,
    )
    if current_key is not None:
        seen_state_keys.add(current_key)
    for _ in range(max_rounds):
        if current.is_complete:
            return current
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        pass_budget_ms = _partial_tail_single_pass_budget_ms(
            current,
            remaining_budget_ms=remaining_ms,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        candidate = _try_partial_tail_single_pass(
            current,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=pass_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if candidate is None:
            break
        if candidate.is_complete:
            return candidate
        if not candidate.partial_plan:
            break
        candidate_key = _partial_result_state_key(
            candidate,
            plan_input=plan_input,
            initial_state=initial_state,
        )
        if candidate_key is not None and candidate_key in seen_state_keys:
            break
        if not _partial_tail_candidate_improves(
            candidate,
            current,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ):
            break
        if candidate_key is not None:
            seen_state_keys.add(candidate_key)
        candidate_structural, candidate_route = _partial_result_structural_route_stats(
            candidate,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        candidate_route_pressure = _optional_int(
            candidate_route.get("total_blockage_pressure")
        )
        candidate_carry_count = _optional_int(candidate_structural.get("loco_carry_count"))
        has_empty_carry_route_pressure = (
            candidate_route_pressure is not None
            and candidate_route_pressure > 0
            and candidate_carry_count == 0
        )
        save_candidate = (
            not has_empty_carry_route_pressure
            or _empty_carry_route_pressure_is_material_tail_progress(
                candidate,
                current,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
        )
        if _route_clean_carried_tail_regression(
            candidate,
            current,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ):
            save_candidate = False
        if save_candidate:
            best = candidate
        current = candidate
    return best


def _route_clean_carried_tail_regression(
    candidate: SolverResult,
    incumbent: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    candidate_structural, candidate_route = _partial_result_structural_route_stats(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_structural, incumbent_route = _partial_result_structural_route_stats(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    candidate_route_pressure = _optional_int(candidate_route.get("total_blockage_pressure"))
    candidate_carry_count = _optional_int(candidate_structural.get("loco_carry_count"))
    if (
        candidate_route_pressure != 0
        or candidate_carry_count is None
        or candidate_carry_count <= 0
    ):
        return False

    incumbent_route_pressure = _optional_int(incumbent_route.get("total_blockage_pressure"))
    if incumbent_route_pressure is not None and incumbent_route_pressure <= 0:
        return False

    regression_metrics = (
        "unfinished_count",
        "work_position_unfinished_count",
        "front_blocker_count",
        "goal_track_blocker_count",
        "staging_debt_count",
    )
    for metric in regression_metrics:
        candidate_value = _optional_int(candidate_structural.get(metric))
        incumbent_value = _optional_int(incumbent_structural.get(metric))
        if candidate_value is None or incumbent_value is None:
            continue
        if candidate_value > incumbent_value:
            return True
    return False


def _empty_carry_route_pressure_is_material_tail_progress(
    candidate: SolverResult,
    incumbent: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    candidate_structural, candidate_route = _partial_result_structural_route_stats(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_structural, incumbent_route = _partial_result_structural_route_stats(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    candidate_route_pressure = _optional_int(candidate_route.get("total_blockage_pressure"))
    candidate_carry_count = _optional_int(candidate_structural.get("loco_carry_count"))
    if (
        candidate_route_pressure is None
        or candidate_route_pressure <= 0
        or candidate_carry_count != 0
    ):
        return False
    incumbent_route_pressure = _optional_int(incumbent_route.get("total_blockage_pressure"))
    if incumbent_route_pressure is not None and candidate_route_pressure > max(
        3,
        incumbent_route_pressure + 1,
    ):
        return False

    improved_business_metrics = (
        "unfinished_count",
        "work_position_unfinished_count",
        "goal_track_blocker_count",
        "staging_debt_count",
        "capacity_overflow_track_count",
    )
    improved = False
    for metric in improved_business_metrics:
        candidate_value = _optional_int(candidate_structural.get(metric))
        incumbent_value = _optional_int(incumbent_structural.get(metric))
        if candidate_value is None or incumbent_value is None:
            continue
        if candidate_value < incumbent_value:
            improved = True
            break
    if not improved:
        return False

    candidate_unfinished = _optional_int(candidate_structural.get("unfinished_count"))
    incumbent_unfinished = _optional_int(incumbent_structural.get("unfinished_count"))
    if (
        candidate_unfinished is not None
        and incumbent_unfinished is not None
        and candidate_unfinished > incumbent_unfinished
    ):
        return False
    return True


def _partial_result_state_key(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> tuple | None:
    if not result.partial_plan:
        return None
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if final_state is None:
        return None
    return _state_key(final_state, plan_input)


def _partial_tail_single_pass_budget_ms(
    result: SolverResult,
    *,
    remaining_budget_ms: float,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> float:
    route_pressure = _partial_result_route_blockage_pressure(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    if route_pressure <= 0 and _partial_result_has_goal_frontier_pressure(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    ):
        return min(
            remaining_budget_ms,
            ROUTE_CLEAN_CONSTRUCTIVE_TAIL_MAX_BUDGET_MS,
        )
    if route_pressure <= 0:
        return remaining_budget_ms
    if remaining_budget_ms < PARTIAL_ROUTE_RELEASE_COMPLETION_BUDGET_MS:
        return remaining_budget_ms
    return max(
        MIN_CHILD_STAGE_BUDGET_MS,
        remaining_budget_ms - ROUTE_BLOCKAGE_TAIL_FOLLOWUP_RESERVE_MS,
    )


def _partial_tail_candidate_improves(
    candidate: SolverResult,
    incumbent: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    if _route_clean_carried_tail_regression(
        candidate,
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return False
    incumbent_structural, incumbent_route = _partial_result_structural_route_stats(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    candidate_structural, candidate_route = _partial_result_structural_route_stats(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_route_pressure = _optional_int(
        incumbent_route.get("total_blockage_pressure")
    )
    candidate_route_pressure = _optional_int(
        candidate_route.get("total_blockage_pressure")
    )
    if (
        incumbent_route_pressure == 0
        and candidate_route_pressure is not None
        and candidate_route_pressure > 0
    ):
        route_clean_stage = (
            incumbent.partial_fallback_stage or incumbent.fallback_stage or ""
        )
        if route_clean_stage in {
            "goal_frontier_tail_completion",
            "constructive_tail_rescue",
            "route_clean_random_area_tail_completion",
            "route_clean_structural_tail_cleanup",
        }:
            route_safety_metrics = (
                "staging_debt_count",
                "work_position_unfinished_count",
                "target_sequence_defect_count",
                "loco_carry_count",
            )
            for metric in route_safety_metrics:
                incumbent_value = _optional_int(incumbent_structural.get(metric)) or 0
                candidate_value = _optional_int(candidate_structural.get(metric)) or 0
                if candidate_value > incumbent_value:
                    return False
    candidate_progress = _partial_tail_progress_score(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_progress = _partial_tail_progress_score(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    if _route_clean_candidate_improves_business_frontier(
        candidate,
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return True
    if _empty_carry_route_pressure_is_material_tail_progress(
        candidate,
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ):
        return True
    if candidate_progress != incumbent_progress:
        return candidate_progress < incumbent_progress
    return _partial_result_score(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    ) < _partial_result_score(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )


def _partial_replacement_candidate_improves(
    candidate: SolverResult,
    incumbent: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    if candidate.is_complete or incumbent.is_complete:
        return False
    if not candidate.partial_plan:
        return False
    if not incumbent.partial_plan:
        return True
    return _partial_tail_candidate_improves(
        candidate,
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )


def _route_clean_candidate_improves_business_frontier(
    candidate: SolverResult,
    incumbent: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> bool:
    candidate_stage = candidate.partial_fallback_stage or candidate.fallback_stage or ""
    if not any(
        stage in candidate_stage
        for stage in ("goal_frontier_tail_completion", "constructive_tail_rescue")
    ):
        return False
    candidate_structural, candidate_route = _partial_result_structural_route_stats(
        candidate,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_structural, incumbent_route = _partial_result_structural_route_stats(
        incumbent,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    incumbent_blockage = _optional_int(incumbent_route.get("total_blockage_pressure"))
    candidate_blockage = _optional_int(candidate_route.get("total_blockage_pressure"))
    if incumbent_blockage is None or candidate_blockage is None:
        return False
    if incumbent_blockage > 0 or candidate_blockage <= incumbent_blockage:
        return False

    candidate_unfinished = _optional_int(candidate_structural.get("unfinished_count"))
    incumbent_unfinished = _optional_int(incumbent_structural.get("unfinished_count"))
    candidate_work = _optional_int(candidate_structural.get("work_position_unfinished_count"))
    incumbent_work = _optional_int(incumbent_structural.get("work_position_unfinished_count"))
    candidate_sequence = _optional_int(
        candidate_structural.get("target_sequence_defect_count")
    )
    incumbent_sequence = _optional_int(
        incumbent_structural.get("target_sequence_defect_count")
    )
    if candidate_sequence is None:
        candidate_sequence = 0
    if incumbent_sequence is None:
        incumbent_sequence = 0
    candidate_front = _optional_int(candidate_structural.get("front_blocker_count"))
    incumbent_front = _optional_int(incumbent_structural.get("front_blocker_count"))
    candidate_staging = _optional_int(candidate_structural.get("staging_debt_count"))
    incumbent_staging = _optional_int(incumbent_structural.get("staging_debt_count"))
    if any(
        value is None
        for value in (
            candidate_unfinished,
            incumbent_unfinished,
            candidate_work,
            incumbent_work,
            candidate_front,
            incumbent_front,
            candidate_staging,
            incumbent_staging,
        )
    ):
        return False
    if candidate_staging > incumbent_staging:
        return False
    return (
        candidate_unfinished < incumbent_unfinished
        or candidate_sequence < incumbent_sequence
        or candidate_work < incumbent_work
        or candidate_front < incumbent_front
    )


def _partial_tail_progress_score(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
) -> tuple[int, int, int, int, int, int, int]:
    structural, route_blockage = _partial_result_structural_route_stats(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    unfinished = _optional_int(structural.get("unfinished_count"))
    blockage = _optional_int(route_blockage.get("total_blockage_pressure"))
    staging_debt = _optional_int(structural.get("staging_debt_count"))
    work_position_unfinished = _optional_int(
        structural.get("work_position_unfinished_count")
    )
    target_sequence_defect = _optional_int(structural.get("target_sequence_defect_count")) or 0
    front_blocker = _optional_int(structural.get("front_blocker_count"))
    goal_track_blocker = _optional_int(structural.get("goal_track_blocker_count"))
    loco_carry = _optional_int(structural.get("loco_carry_count"))
    unknown = 10**9
    return (
        blockage if blockage is not None else unknown,
        unfinished if unfinished is not None else unknown,
        target_sequence_defect,
        staging_debt if staging_debt is not None else unknown,
        work_position_unfinished if work_position_unfinished is not None else unknown,
        front_blocker if front_blocker is not None else unknown,
        goal_track_blocker if goal_track_blocker is not None else unknown,
        loco_carry if loco_carry is not None else unknown,
    )


def _try_partial_tail_single_pass(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    started_at = perf_counter()

    def remaining_budget_ms() -> float | None:
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        return remaining_ms

    partial_has_route_pressure = (
        _partial_result_route_blockage_pressure(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        )
        > 0
    )
    if partial_has_route_pressure:
        route_budget_ms = remaining_budget_ms()
        if route_budget_ms is None:
            return None
        route_tail_completion = _try_route_blockage_tail_clearance_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            partial_plan=result.partial_plan,
            master=master,
            time_budget_ms=route_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if route_tail_completion is None:
            route_release_budget_ms = remaining_budget_ms()
            if route_release_budget_ms is None:
                return None
            route_tail_completion = _try_route_release_partial_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                partial_plan=result.partial_plan,
                master=master,
                time_budget_ms=route_release_budget_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
        if route_tail_completion is None or route_tail_completion.is_complete:
            return route_tail_completion
        if (
            route_tail_completion.partial_plan
            and _partial_result_route_blockage_pressure(
                route_tail_completion,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            > 0
        ):
            return route_tail_completion
        if (
            route_tail_completion.partial_plan
            and _partial_result_has_goal_frontier_pressure(
                route_tail_completion,
                plan_input=plan_input,
                initial_state=initial_state,
            )
        ):
            frontier_budget_ms = remaining_budget_ms()
            if frontier_budget_ms is None:
                return route_tail_completion
            goal_frontier_completion = _try_goal_frontier_tail_completion(
                plan_input=plan_input,
                initial_state=initial_state,
                partial_plan=route_tail_completion.partial_plan,
                master=master,
                time_budget_ms=frontier_budget_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if goal_frontier_completion is not None:
                if (
                    not goal_frontier_completion.is_complete
                    and not _partial_tail_candidate_improves(
                        goal_frontier_completion,
                        route_tail_completion,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    )
                ):
                    constructive_completion = _try_constructive_tail_rescue_completion(
                        route_tail_completion,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                        time_budget_ms=remaining_budget_ms(),
                        enable_depot_late_scheduling=enable_depot_late_scheduling,
                    )
                    if constructive_completion is not None:
                        return constructive_completion
                return _shorter_complete_result(
                    route_tail_completion,
                    goal_frontier_completion,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                )
        return route_tail_completion
    if (
        _partial_result_loco_carry_count(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        )
        > 0
    ):
        carry_tail_budget_ms = remaining_budget_ms()
        if carry_tail_budget_ms is None:
            return None
        carry_tail_completion = _try_route_blockage_tail_clearance_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            partial_plan=result.partial_plan,
            master=master,
            time_budget_ms=carry_tail_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if carry_tail_completion is not None:
            return carry_tail_completion
        if _partial_result_has_goal_frontier_pressure(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        ):
            frontier_budget_ms = remaining_budget_ms()
            if frontier_budget_ms is not None:
                return _try_goal_frontier_tail_completion(
                    plan_input=plan_input,
                    initial_state=initial_state,
                    partial_plan=result.partial_plan,
                    master=master,
                    time_budget_ms=frontier_budget_ms,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
        return None
    if master is not None:
        random_area_budget_ms = remaining_budget_ms()
        if random_area_budget_ms is None:
            return None
        route_clean_random_area_completion = (
            _try_route_clean_random_area_tail_completion(
                result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                time_budget_ms=random_area_budget_ms,
            )
        )
        if route_clean_random_area_completion is not None:
            return route_clean_random_area_completion
    if _partial_result_has_goal_frontier_pressure(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
    ):
        if _partial_result_route_blockage_pressure(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
        ) <= 0 and _partial_result_route_clean_constructive_first_tail(
            result,
            plan_input=plan_input,
            initial_state=initial_state,
        ):
            constructive_first_budget_ms = remaining_budget_ms()
            if constructive_first_budget_ms is not None:
                constructive_first_completion = _try_constructive_tail_rescue_completion(
                    result,
                    plan_input=plan_input,
                    initial_state=initial_state,
                    master=master,
                    time_budget_ms=min(
                        constructive_first_budget_ms,
                        ROUTE_CLEAN_CONSTRUCTIVE_TAIL_MAX_BUDGET_MS,
                    ),
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
                if constructive_first_completion is not None:
                    if constructive_first_completion.is_complete:
                        return constructive_first_completion
                    if _partial_tail_candidate_improves(
                        constructive_first_completion,
                        result,
                        plan_input=plan_input,
                        initial_state=initial_state,
                        master=master,
                    ):
                        result = constructive_first_completion
        frontier_budget_ms = remaining_budget_ms()
        if frontier_budget_ms is None:
            return None
        goal_frontier_completion = _try_goal_frontier_tail_completion(
            plan_input=plan_input,
            initial_state=initial_state,
            partial_plan=result.partial_plan,
            master=master,
            time_budget_ms=frontier_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if goal_frontier_completion is not None:
            if goal_frontier_completion.is_complete or _partial_tail_candidate_improves(
                goal_frontier_completion,
                result,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            ):
                return goal_frontier_completion
    return _try_constructive_tail_rescue_completion(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        time_budget_ms=remaining_budget_ms(),
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _try_constructive_tail_rescue_completion(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float | None,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if (
        master is None
        or time_budget_ms is None
        or time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS
        or not result.partial_plan
    ):
        return None
    state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if state is None or state.loco_carry or _is_goal(plan_input, state):
        return None
    structural = compute_structural_metrics(plan_input, state)
    route_clean_constructive_tail = (
        structural.unfinished_count <= LOCALIZED_RESUME_MAX_UNFINISHED
    )
    if structural.unfinished_count <= 0:
        return None
    return _try_constructive_tail_rescue_from_state(
        plan_input=plan_input,
        original_initial_state=initial_state,
        prefix_plan=list(result.partial_plan),
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _partial_result_score(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput | None = None,
    initial_state: ReplayState | None = None,
    master: MasterData | None = None,
) -> tuple[int, ...]:
    structural, route_blockage = _partial_result_structural_route_stats(
        result,
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
    )
    unfinished = _optional_int(structural.get("unfinished_count"))
    blockage = _optional_int(route_blockage.get("total_blockage_pressure"))
    staging_debt = _optional_int(structural.get("staging_debt_count"))
    target_sequence_defect = _optional_int(structural.get("target_sequence_defect_count")) or 0
    capacity_overflow = _optional_int(structural.get("capacity_overflow_track_count"))
    unfinished_score = unfinished if unfinished is not None else 10**9
    blockage_score = blockage if blockage is not None else 10**9
    shape = (result.debug_stats or {}).get("plan_shape_metrics") or summarize_plan_shape(
        result.partial_plan
    )
    churn_penalty = _partial_plan_churn_penalty(shape)
    return (
        unfinished_score + max(0, blockage_score - 3) + churn_penalty,
        blockage_score,
        target_sequence_defect,
        staging_debt if staging_debt is not None else 10**9,
        capacity_overflow if capacity_overflow is not None else 10**9,
        len(result.partial_plan),
        _stage_rank(result.partial_fallback_stage or result.fallback_stage),
        _optional_int(shape.get("rehandled_vehicle_count")) or 0,
    )


def _partial_result_structural_route_stats(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput | None = None,
    initial_state: ReplayState | None = None,
    master: MasterData | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return structural, route_blockage


def _partial_result_loco_carry_count(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> int:
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    count = _optional_int(structural.get("loco_carry_count"))
    if count is not None:
        return count
    if not result.partial_plan:
        return 0
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if final_state is None:
        return 0
    return len(final_state.loco_carry)


def _partial_plan_churn_penalty(shape: dict[str, Any]) -> int:
    max_touch = _optional_int(shape.get("max_vehicle_touch_count")) or 0
    staging_to_staging = _optional_int(shape.get("staging_to_staging_hook_count")) or 0
    rehandled = _optional_int(shape.get("rehandled_vehicle_count")) or 0
    return (
        max(0, max_touch - CONSTRUCTIVE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT)
        + max(0, staging_to_staging - CONSTRUCTIVE_SKIP_PRIMARY_MAX_STAGING_TO_STAGING_HOOKS) * 2
        + max(0, rehandled - CONSTRUCTIVE_SKIP_PRIMARY_MAX_VEHICLE_TOUCH_COUNT)
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


def _partial_result_has_goal_frontier_pressure(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> bool:
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    front_blocker = _optional_int(structural.get("front_blocker_count"))
    work_position_unfinished = _optional_int(structural.get("work_position_unfinished_count"))
    unfinished = _optional_int(structural.get("unfinished_count"))
    if unfinished is not None and (
        front_blocker is not None or work_position_unfinished is not None
    ):
        return unfinished > 0 and (
            (front_blocker or 0) > 0 or (work_position_unfinished or 0) > 0
        )
    final_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=initial_state,
        plan=result.partial_plan,
    )
    if final_state is None:
        return False
    metrics = compute_structural_metrics(plan_input, final_state)
    return metrics.unfinished_count > 0 and (
        metrics.front_blocker_count > 0
        or metrics.work_position_unfinished_count > 0
    )


def _partial_result_route_clean_constructive_first_tail(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
) -> bool:
    structural = (result.debug_stats or {}).get("partial_structural_metrics") or {}
    unfinished = _optional_int(structural.get("unfinished_count"))
    work_position_unfinished = _optional_int(
        structural.get("work_position_unfinished_count")
    )
    front_blocker = _optional_int(structural.get("front_blocker_count"))
    staging_debt = _optional_int(structural.get("staging_debt_count"))
    if unfinished is None or work_position_unfinished is None or front_blocker is None:
        if not result.partial_plan:
            return False
        final_state = _replay_solver_moves(
            plan_input=plan_input,
            initial_state=initial_state,
            plan=result.partial_plan,
        )
        if final_state is None:
            return False
        metrics = compute_structural_metrics(plan_input, final_state)
        unfinished = metrics.unfinished_count
        work_position_unfinished = metrics.work_position_unfinished_count
        front_blocker = metrics.front_blocker_count
        staging_debt = metrics.staging_debt_count
    return (
        0 < unfinished <= LOCALIZED_RESUME_MAX_UNFINISHED
        and (work_position_unfinished > 0 or front_blocker > 0)
        and (staging_debt or 0) == 0
    )


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


def _partial_result_has_known_route_pressure(result: SolverResult) -> bool:
    pressure = _partial_route_pressure_from_debug(result)
    return pressure is not None and pressure > 0


def _run_constructive_stage(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    time_budget_ms: float | None,
    enable_depot_late_scheduling: bool = False,
    strict_staging_regrab: bool = True,
    route_release_bias: bool = False,
    budget_fraction: float = 1.0,
    max_budget_ms: float | None = None,
) -> SolverResult | None:
    """Run the priority-rule dispatcher and wrap the result as a SolverResult."""
    from fzed_shunting.solver.constructive import solve_constructive

    constructive_budget = None
    if time_budget_ms is not None:
        constructive_budget = max(500.0, time_budget_ms * budget_fraction)
        if max_budget_ms is not None:
            constructive_budget = min(max_budget_ms, constructive_budget)
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
        remaining_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
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
        route_tail_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
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
        remaining_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
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
        combined_partial_plan = checkpoint_prefix + list(localized_completion.partial_plan)
        remaining_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            if not localized_completion.partial_plan:
                return None
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
        if (
            localized_completion.partial_plan
            and master is not None
            and original_initial_state is not None
        ):
            partial_state = _replay_solver_moves(
                plan_input=plan_input,
                initial_state=checkpoint_state,
                plan=list(localized_completion.partial_plan),
            )
        else:
            partial_state = checkpoint_state
        tail_rescue = _try_checkpoint_tail_rescue_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=combined_partial_plan,
            state=partial_state,
            master=master,
            time_budget_ms=remaining_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if tail_rescue is not None:
            return tail_rescue
        if not localized_completion.partial_plan:
            return None
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


def _try_checkpoint_tail_rescue_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState | None,
    prefix_plan: list[HookAction],
    state: ReplayState | None,
    master: MasterData | None,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if (
        master is None
        or original_initial_state is None
        or state is None
        or state.loco_carry
        or time_budget_ms <= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
    ):
        return None
    route_blockage_plan = compute_route_blockage_plan(
        plan_input,
        state,
        RouteOracle(master),
    )
    if route_blockage_plan.total_blockage_pressure > 0:
        return _try_route_blockage_tail_clearance_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            state=state,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
    structural = compute_structural_metrics(plan_input, state)
    if structural.unfinished_count <= 0:
        return None
    if structural.front_blocker_count > 0:
        frontier_completion = _try_goal_frontier_tail_completion_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            state=state,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if frontier_completion is not None:
            return frontier_completion
    if structural.capacity_overflow_track_count <= 0:
        return None
    suffix_completion = _try_direct_tail_suffix_search(
        plan_input=plan_input,
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if suffix_completion is None or not suffix_completion.is_complete:
        constructive_tail = _try_constructive_tail_rescue_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            state=state,
            master=master,
            time_budget_ms=time_budget_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if constructive_tail is not None:
            return constructive_tail
        return None
    result = SolverResult(
        plan=[*prefix_plan, *suffix_completion.plan],
        expanded_nodes=suffix_completion.expanded_nodes,
        generated_nodes=suffix_completion.generated_nodes,
        closed_nodes=suffix_completion.closed_nodes,
        elapsed_ms=suffix_completion.elapsed_ms,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="capacity_tail_suffix_search",
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
        return None


def _try_constructive_tail_rescue_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    from fzed_shunting.solver.constructive import solve_constructive

    if time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS or state.loco_carry:
        return None
    started_at = perf_counter()
    constructive_budget_ms = min(
        time_budget_ms,
        max(1_000.0, time_budget_ms * 0.5),
    )
    try:
        completion = solve_constructive(
            plan_input,
            state,
            master=master,
            max_iterations=1500,
            time_budget_ms=constructive_budget_ms,
            strict_staging_regrab=False,
            route_release_bias=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if not completion.plan:
        return None
    combined_plan = [*prefix_plan, *completion.plan]
    if completion.reached_goal:
        result = SolverResult(
            plan=combined_plan,
            expanded_nodes=completion.iterations,
            generated_nodes=completion.iterations,
            closed_nodes=completion.iterations,
            elapsed_ms=(perf_counter() - started_at) * 1000,
            is_complete=True,
            is_proven_optimal=False,
            fallback_stage="constructive_tail_rescue",
            debug_stats=dict(completion.debug_stats or {}),
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
    partial_state = _replay_solver_moves(
        plan_input=plan_input,
        initial_state=state,
        plan=list(completion.plan),
    )
    if partial_state is None or partial_state.loco_carry:
        return None
    route_blockage_plan = compute_route_blockage_plan(
        plan_input,
        partial_state,
        RouteOracle(master),
    )
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if (
        route_blockage_plan.total_blockage_pressure > 0
        and remaining_ms > ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS
    ):
        route_tail = _try_route_blockage_tail_clearance_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=combined_plan,
            state=partial_state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if route_tail is not None:
            return route_tail
    result = SolverResult(
        plan=[],
        expanded_nodes=completion.iterations,
        generated_nodes=completion.iterations,
        closed_nodes=completion.iterations,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=False,
        is_proven_optimal=False,
        fallback_stage="constructive_tail_rescue",
        partial_plan=combined_plan,
        partial_fallback_stage="constructive_tail_rescue",
        debug_stats=dict(completion.debug_stats or {}),
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
                structural = compute_structural_metrics(plan_input, state)
                if (
                    route_blockage_plan.total_blockage_pressure > 0
                    or structural.capacity_overflow_track_count > 0
                    or structural.front_blocker_count > 0
                ):
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
        structural = compute_structural_metrics(plan_input, state)
        route_blockage_plan = compute_route_blockage_plan(
            plan_input,
            state,
            RouteOracle(master),
        )
        if (
            not state.loco_carry
            and route_blockage_plan.total_blockage_pressure <= 0
            and (
                structural.capacity_overflow_track_count > 0
                or structural.front_blocker_count > 0
            )
        ):
            return _try_checkpoint_tail_rescue_from_state(
                plan_input=plan_input,
                original_initial_state=initial_state,
                prefix_plan=list(partial_plan),
                state=state,
                master=master,
                time_budget_ms=time_budget_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
        return None

    ranked_checkpoints = _rank_route_release_checkpoints(checkpoints)
    started_at = perf_counter()
    best_complete: SolverResult | None = None
    best_partial: SolverResult | None = None
    for index, (_score, _negative_prefix_len, completion_prefix, completion_state) in enumerate(
        ranked_checkpoints
    ):
        remaining_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        remaining_attempts = max(1, len(ranked_checkpoints) - index)
        per_checkpoint_budget = _route_release_checkpoint_budget_ms(
            remaining_budget_ms=remaining_budget_ms,
            remaining_attempts=remaining_attempts,
        )
        checkpoint_started_at = perf_counter()
        route_pressure = _score[0]
        if route_pressure > 0:
            tail_clearance = _try_route_blockage_tail_clearance_from_state(
                plan_input=plan_input,
                original_initial_state=initial_state,
                prefix_plan=completion_prefix,
                state=completion_state,
                master=master,
                time_budget_ms=per_checkpoint_budget,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
        else:
            tail_clearance = _try_checkpoint_tail_rescue_from_state(
                plan_input=plan_input,
                original_initial_state=initial_state,
                prefix_plan=completion_prefix,
                state=completion_state,
                master=master,
                time_budget_ms=per_checkpoint_budget,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
        if tail_clearance is not None:
            best_complete = _shorter_complete_result(
                best_complete,
                tail_clearance,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
            continue
        per_checkpoint_budget -= (perf_counter() - checkpoint_started_at) * 1000
        if per_checkpoint_budget <= MIN_CHILD_STAGE_BUDGET_MS:
            continue
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
            verified = _attach_verification(
                result,
                plan_input=plan_input,
                master=master,
                initial_state=initial_state,
            )
            best_complete = _shorter_complete_result(
                best_complete,
                verified,
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
            )
        except Exception:  # noqa: BLE001
            continue
    return best_complete or best_partial


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
    route_oracle = RouteOracle(master)
    initial_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    if state.loco_carry:
        return _try_tail_clearance_resume_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=[],
            state=state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=time_budget_ms,
            expanded_nodes=0,
            generated_nodes=0,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=frozenset({_state_key(state, plan_input)}),
        )
    if initial_blockage.total_blockage_pressure <= 0:
        return None

    started_at = perf_counter()
    frontier: list[tuple[ReplayState, list[HookAction]]] = [(state, [])]
    seen: set[tuple] = {_state_key(state, plan_input)}
    expanded = 0
    generated = 0
    best_partial: SolverResult | None = None

    def remember_outer_partial(result: SolverResult | None) -> None:
        nonlocal best_partial
        if result is None or result.is_complete:
            return
        best_partial = _shorter_complete_result(
            best_partial,
            result,
            plan_input=plan_input,
            initial_state=original_initial_state,
            master=master,
        )

    while frontier:
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MIN_COMPLETION_BUDGET_MS:
            break
        frontier.sort(
            key=lambda item: (
                compute_route_blockage_plan(plan_input, item[0], route_oracle).total_blockage_pressure,
                compute_structural_metrics(plan_input, item[0]).unfinished_count,
                len(item[1]),
                tuple(
                    (
                        move.action_type,
                        move.source_track,
                        move.target_track,
                        tuple(move.vehicle_nos),
                    )
                    for move in item[1]
                ),
            )
        )
        current_state, clearing_plan = frontier.pop(0)
        expanded += 1
        current_blockage = compute_route_blockage_plan(
            plan_input,
            current_state,
            route_oracle,
        )
        if current_blockage.total_blockage_pressure == 0:
            child_budget_ms = _route_blockage_tail_branch_budget_ms(remaining_ms)
            completion = _try_tail_clearance_resume_from_state(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=clearing_plan,
                state=current_state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=child_budget_ms,
                expanded_nodes=expanded,
                generated_nodes=generated,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                seen_state_keys=frozenset({_state_key(current_state, plan_input)}),
            )
            if completion is not None:
                if completion.is_complete:
                    return completion
                remember_outer_partial(completion)
        elif clearing_plan and (
            current_blockage.total_blockage_pressure
            < initial_blockage.total_blockage_pressure
        ):
            child_budget_ms = _route_blockage_tail_branch_budget_ms(remaining_ms)
            completion = _try_tail_clearance_resume_from_state(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=clearing_plan,
                state=current_state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=child_budget_ms,
                expanded_nodes=expanded,
                generated_nodes=generated,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                seen_state_keys=frozenset({_state_key(current_state, plan_input)}),
            )
            if completion is not None:
                if completion.is_complete:
                    return completion
                remember_outer_partial(completion)
        if len(clearing_plan) >= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS:
            continue

        for next_state, step_plan in _route_blockage_satisfied_blocker_staging_steps(
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
            generated += len(step_plan)
            frontier.append((next_state, [*clearing_plan, *step_plan]))
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
                        tuple(
                            (
                                move.action_type,
                                move.source_track,
                                move.target_track,
                                tuple(move.vehicle_nos),
                            )
                            for move in item[1]
                        ),
                    )
                )
                del frontier[32:]

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
                        tuple(
                            (
                                move.action_type,
                                move.source_track,
                                move.target_track,
                                tuple(move.vehicle_nos),
                            )
                            for move in item[1]
                        ),
                    )
                )
                del frontier[32:]
    return best_partial


def _route_blockage_satisfied_blocker_staging_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    current_blockage: Any,
) -> list[tuple[ReplayState, list[HookAction]]]:
    if state.loco_carry or current_blockage.total_blockage_pressure <= 0:
        return []
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    current_pressure = current_blockage.total_blockage_pressure
    candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], ReplayState, list[HookAction]]] = []
    attach_moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
        require_empty_carry_followup=False,
    )
    facts = sorted(
        getattr(current_blockage, "facts_by_blocking_track", {}).values(),
        key=lambda fact: (
            -getattr(fact, "blockage_count", 0),
            getattr(fact, "blocking_track", ""),
        ),
    )
    for fact in facts:
        source_track = getattr(fact, "blocking_track", "")
        blocking_vehicle_nos = set(getattr(fact, "blocking_vehicle_nos", []))
        source_seq = list(state.track_sequences.get(source_track, []))
        prefix_block: list[str] = []
        for vehicle_no in source_seq:
            if vehicle_no not in blocking_vehicle_nos:
                break
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None or not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            prefix_block.append(vehicle_no)
        if not prefix_block:
            continue
        attach_move = _find_generated_move(
            attach_moves,
            action_type="ATTACH",
            source_track=source_track,
            vehicle_nos=prefix_block,
        )
        if attach_move is None:
            attach_move = _build_goal_frontier_exact_attach(
                plan_input=plan_input,
                state=state,
                master=master,
                source_track=source_track,
                prefix_block=prefix_block,
            )
        if attach_move is None:
            continue
        try:
            attached_state = _apply_move(
                state=state,
                move=attach_move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        detach_moves = generate_real_hook_moves(
            plan_input,
            attached_state,
            master=master,
            route_oracle=route_oracle,
            require_empty_carry_followup=False,
        )
        for detach_move in detach_moves:
            if (
                detach_move.action_type != "DETACH"
                or detach_move.target_track == source_track
                or list(detach_move.vehicle_nos) != prefix_block
            ):
                continue
            try:
                next_state = _apply_move(
                    state=attached_state,
                    move=detach_move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                continue
            if next_state.loco_carry:
                continue
            next_blockage = compute_route_blockage_plan(
                plan_input,
                next_state,
                route_oracle,
            )
            if next_blockage.total_blockage_pressure >= current_pressure:
                continue
            candidates.append(
                (
                    (
                        next_blockage.total_blockage_pressure,
                        compute_structural_metrics(plan_input, next_state).unfinished_count,
                        len(next_state.track_sequences.get(detach_move.target_track, [])),
                        len(detach_move.path_tracks),
                        detach_move.target_track,
                        tuple(prefix_block),
                    ),
                    next_state,
                    [attach_move, detach_move],
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [(next_state, step_plan) for _score, next_state, step_plan in candidates[:8]]


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
    blocked_vehicle_nos_by_source_track: dict[str, set[str]] = {}
    for fact in facts_by_blocking_track.values():
        blocked_vehicle_nos = set(getattr(fact, "blocked_vehicle_nos", []))
        if not blocked_vehicle_nos:
            continue
        for source_track in getattr(fact, "source_tracks", []):
            blocked_vehicle_nos_by_source_track.setdefault(source_track, set()).update(
                blocked_vehicle_nos
            )
    current_source_blocked_vehicle_nos = blocked_vehicle_nos_by_source_track.get(
        state.loco_track_name,
        set(),
    )
    moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
        require_empty_carry_followup=not state.loco_carry,
    )
    cleared_route_carry = (
        bool(state.loco_carry)
        and current_pressure <= 0
        and not facts_by_blocking_track
    )
    candidates: list[tuple[tuple[Any, ...], HookAction, ReplayState]] = []
    fallback_pressure_increase_candidates: list[
        tuple[tuple[Any, ...], HookAction, ReplayState]
    ] = []
    fallback_goal_drop_candidates: list[
        tuple[tuple[Any, ...], HookAction, ReplayState]
    ] = []
    for move in moves:
        releases_carried_blocker = False
        if state.loco_carry:
            if move.action_type != "DETACH":
                continue
            carried_vehicle_nos = set(move.vehicle_nos)
            releases_carried_blocker = any(
                bool(carried_vehicle_nos & set(getattr(fact, "blocking_vehicle_nos", [])))
                for fact in facts_by_blocking_track.values()
            )
        else:
            if move.action_type != "ATTACH":
                continue
            moved_vehicle_nos = set(move.vehicle_nos)
            fact = facts_by_blocking_track.get(move.source_track)
            releases_blocking_track = (
                fact is not None
                and bool(moved_vehicle_nos & set(getattr(fact, "blocking_vehicle_nos", [])))
            )
            releases_blocked_source = bool(
                moved_vehicle_nos
                & blocked_vehicle_nos_by_source_track.get(move.source_track, set())
            )
            if not releases_blocking_track and not releases_blocked_source:
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
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        next_blockage = compute_route_blockage_plan(
            plan_input,
            next_state,
            route_oracle,
        )
        next_pressure = next_blockage.total_blockage_pressure
        goal_detach = _detach_places_all_vehicles_at_goal(
            move=move,
            next_state=next_state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        staged_carried_blocked_source = (
            state.loco_carry
            and move.action_type == "DETACH"
            and bool(set(move.vehicle_nos) & current_source_blocked_vehicle_nos)
        )
        reparks_to_blocking_track = (
            state.loco_carry
            and move.action_type == "DETACH"
            and move.target_track in next_blockage.facts_by_blocking_track
        )
        target_current_blockage_fact = facts_by_blocking_track.get(move.target_track)
        moves_active_target_blocker = (
            target_current_blockage_fact is not None
            and bool(
                set(move.vehicle_nos)
                & set(getattr(target_current_blockage_fact, "blocking_vehicle_nos", []))
            )
        )
        goal_detach_rank = 0 if goal_detach and not reparks_to_blocking_track else 1
        if cleared_route_carry and next_pressure > current_pressure:
            fallback_pressure_increase_candidates.append(
                (
                    (
                        next_pressure - current_pressure,
                        goal_detach_rank,
                        -len(move.vehicle_nos),
                        len(next_state.track_sequences.get(move.target_track, [])),
                        len(move.path_tracks),
                        move.target_track,
                        tuple(move.vehicle_nos),
                    ),
                    move,
                    next_state,
                )
            )
            continue
        if next_pressure > current_pressure and not (
            goal_detach or staged_carried_blocked_source
        ):
            if state.loco_carry and move.target_track in STAGING_TRACKS:
                fallback_pressure_increase_candidates.append(
                    (
                        (
                            next_pressure - current_pressure,
                            len(next_state.track_sequences.get(move.target_track, [])),
                            len(move.path_tracks),
                            move.target_track,
                            tuple(move.vehicle_nos),
                        ),
                        move,
                        next_state,
                    )
                )
                continue
            if state.loco_carry and not facts_by_blocking_track:
                fallback_pressure_increase_candidates.append(
                    (
                        (
                            goal_detach_rank,
                            next_pressure - current_pressure,
                            len(next_state.track_sequences.get(move.target_track, [])),
                            len(move.path_tracks),
                            move.target_track,
                            tuple(move.vehicle_nos),
                        ),
                        move,
                        next_state,
                    )
                )
                continue
            if not (state.loco_carry and move.target_track in STAGING_TRACKS):
                continue
            fallback_pressure_increase_candidates.append(
                (
                    (
                        goal_detach_rank,
                        0 if staged_carried_blocked_source else 1,
                        0 if releases_carried_blocker else 1,
                        next_pressure - current_pressure,
                        len(next_state.track_sequences.get(move.target_track, [])),
                        len(move.path_tracks),
                        move.target_track,
                        tuple(move.vehicle_nos),
                    ),
                    move,
                    next_state,
                )
            )
            continue
        if (
            state.loco_carry
            and goal_detach
            and next_pressure > current_pressure
            and not staged_carried_blocked_source
        ):
            fallback_goal_drop_candidates.append(
                (
                    (
                        0 if move.target_track == state.loco_track_name else 1,
                        next_pressure - current_pressure,
                        len(next_blockage.facts_by_blocking_track),
                        len(next_state.track_sequences.get(move.target_track, [])),
                        len(move.path_tracks),
                        move.target_track,
                        tuple(move.vehicle_nos),
                    ),
                    move,
                    next_state,
                )
            )
            continue
        if (
            state.loco_carry
            and reparks_to_blocking_track
            and not moves_active_target_blocker
            and not staged_carried_blocked_source
        ):
            continue
        source_remainder = len(next_state.track_sequences.get(move.source_track, []))
        pressure_delta = current_pressure - next_pressure
        candidates.append(
            (
                (
                    goal_detach_rank,
                    0 if staged_carried_blocked_source else 1,
                    1 if reparks_to_blocking_track else 0,
                    -pressure_delta,
                    0 if move.action_type == "ATTACH" else 1,
                    -len(move.vehicle_nos) if state.loco_carry else 0,
                    source_remainder,
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
                next_state,
            )
        )
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        fallback_candidates = [
            *fallback_pressure_increase_candidates,
            *fallback_goal_drop_candidates,
        ]
        if fallback_candidates:
            fallback_candidates.sort(
                key=lambda item: (
                    0
                    if item in fallback_goal_drop_candidates
                    and item[1].target_track == state.loco_track_name
                    else 1,
                    compute_route_blockage_plan(
                        plan_input,
                        item[2],
                        route_oracle,
                    ).total_blockage_pressure
                    - current_pressure,
                    0 if item in fallback_goal_drop_candidates else 1,
                    len(item[2].track_sequences.get(item[1].target_track, [])),
                    len(item[1].path_tracks),
                    item[1].target_track,
                    tuple(item[1].vehicle_nos),
                )
            )
            return [
                (move, next_state)
                for _, move, next_state in fallback_candidates[:4]
            ]
    return [(move, next_state) for _, move, next_state in candidates[:8]]


def _detach_places_all_vehicles_at_goal(
    *,
    move: HookAction,
    next_state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, Any],
) -> bool:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    if move.action_type != "DETACH":
        return False
    for vehicle_no in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or not goal_is_satisfied(
            vehicle,
            track_name=move.target_track,
            state=next_state,
            plan_input=plan_input,
        ):
            return False
    return True


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
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    started_at = perf_counter()
    if seen_state_keys is None:
        seen_state_keys = frozenset({_state_key(state, plan_input)})

    best_partial: SolverResult | None = None
    best_complete: SolverResult | None = None

    def current_clearing_partial() -> SolverResult | None:
        if not clearing_plan:
            return None
        return _build_route_blockage_tail_partial_result(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=clearing_plan,
            state=state,
            master=master,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
        )

    def remember_partial(result: SolverResult | None) -> None:
        nonlocal best_partial
        if result is None or result.is_complete:
            return
        best_partial = _shorter_complete_result(
            best_partial,
            result,
            plan_input=plan_input,
            initial_state=original_initial_state,
            master=master,
        )

    def remember_complete(result: SolverResult | None) -> bool:
        nonlocal best_complete
        if result is None or not result.is_complete:
            return False
        best_complete = _shorter_complete_result(
            best_complete,
            result,
            plan_input=plan_input,
            initial_state=original_initial_state,
            master=master,
        )
        return _complete_result_needs_quality_followup(best_complete)

    def best_or_current_partial() -> SolverResult | None:
        current = current_clearing_partial()
        if best_partial is None:
            return current
        if current is None:
            return best_partial
        return _shorter_complete_result(
            current,
            best_partial,
            plan_input=plan_input,
            initial_state=original_initial_state,
            master=master,
        )

    if state.loco_carry:
        route_oracle = RouteOracle(master)
        current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
        if current_blockage.total_blockage_pressure <= 0:
            carried_route_blocker_source_completion = (
                _try_carried_route_blocker_source_block_completion(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=clearing_plan,
                    state=state,
                    initial_blockage=initial_blockage,
                    master=master,
                    time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                    seen_state_keys=seen_state_keys,
                )
            )
            if carried_route_blocker_source_completion is not None:
                return _as_route_blockage_tail_completion(
                    carried_route_blocker_source_completion
                )
        if (
            current_blockage.total_blockage_pressure > 0
            and len(clearing_plan) < ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS
        ):
            for move, next_state in _route_blockage_tail_clearance_candidates(
                plan_input=plan_input,
                state=state,
                master=master,
                route_oracle=route_oracle,
                current_blockage=current_blockage,
            ):
                if len(next_state.loco_carry) >= len(state.loco_carry):
                    continue
                next_state_key = _state_key(next_state, plan_input)
                if next_state_key in seen_state_keys:
                    continue
                remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
                if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                    return best_or_current_partial()
                child_budget_ms = _route_blockage_tail_branch_budget_ms(remaining_ms)
                completion = _try_tail_clearance_resume_from_state(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=[*clearing_plan, move],
                    state=next_state,
                    initial_blockage=initial_blockage,
                    master=master,
                    time_budget_ms=child_budget_ms,
                    expanded_nodes=expanded_nodes + 1,
                    generated_nodes=generated_nodes + 1,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                    seen_state_keys=seen_state_keys | frozenset({next_state_key}),
                )
                if completion is not None:
                    if completion.is_complete:
                        return _as_route_blockage_tail_completion(completion)
                    remember_partial(completion)
            if best_partial is not None:
                return best_partial
            if clearing_plan:
                return _build_route_blockage_tail_partial_result(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=clearing_plan,
                    state=state,
                    master=master,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                )
        if current_blockage.total_blockage_pressure <= 0:
            for move, next_state in _route_blockage_tail_clearance_candidates(
                plan_input=plan_input,
                state=state,
                master=master,
                route_oracle=route_oracle,
                current_blockage=current_blockage,
            ):
                next_blockage = compute_route_blockage_plan(
                    plan_input,
                    next_state,
                    route_oracle,
                )
                next_state_key = _state_key(next_state, plan_input)
                if next_state_key in seen_state_keys:
                    continue
                remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
                if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                    return best_or_current_partial()
                child_budget_ms = _route_blockage_tail_branch_budget_ms(remaining_ms)
                completion = _try_tail_clearance_resume_from_state(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=[*clearing_plan, move],
                    state=next_state,
                    initial_blockage=initial_blockage,
                    master=master,
                    time_budget_ms=child_budget_ms,
                    expanded_nodes=expanded_nodes + 1,
                    generated_nodes=generated_nodes + 1,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                    seen_state_keys=seen_state_keys | frozenset({next_state_key}),
                )
                if completion is not None:
                    if completion.is_complete:
                        return _as_route_blockage_tail_completion(completion)
                    remember_partial(completion)
                remember_partial(
                    _build_route_blockage_tail_partial_result(
                        plan_input=plan_input,
                        original_initial_state=original_initial_state,
                        prefix_plan=prefix_plan,
                        clearing_plan=[*clearing_plan, move],
                        state=next_state,
                        master=master,
                        expanded_nodes=expanded_nodes + 1,
                        generated_nodes=generated_nodes + 1,
                        elapsed_ms=(perf_counter() - started_at) * 1000,
                    )
                )
        carried_work_position_completion = _try_carried_work_position_clearance_resume(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=clearing_plan,
            state=state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=seen_state_keys,
        )
        if carried_work_position_completion is not None:
            return _as_route_blockage_tail_completion(carried_work_position_completion)
        carried_goal_blocker_completion = _try_carried_goal_blocker_clearance_resume(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=clearing_plan,
            state=state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=seen_state_keys,
        )
        if carried_goal_blocker_completion is not None:
            return _as_route_blockage_tail_completion(carried_goal_blocker_completion)
        carried_route_blocker_source_completion = (
            _try_carried_route_blocker_source_block_completion(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=clearing_plan,
                state=state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                seen_state_keys=seen_state_keys,
            )
        )
        if carried_route_blocker_source_completion is not None:
            return _as_route_blockage_tail_completion(
                carried_route_blocker_source_completion
            )
        carried_random_area_completion = _try_carried_random_area_tail_resume(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=clearing_plan,
            state=state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=seen_state_keys,
        )
        if carried_random_area_completion is not None:
            return _as_route_blockage_tail_completion(carried_random_area_completion)
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return best_or_current_partial()
        suffix_completion = _try_direct_tail_suffix_search(
            plan_input=plan_input,
            state=state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if suffix_completion is None:
            if clearing_plan:
                return _build_route_blockage_tail_partial_result(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=clearing_plan,
                    state=state,
                    master=master,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                )
            return best_or_current_partial()
        combined_plan = [*prefix_plan, *clearing_plan, *suffix_completion.plan]
        result = SolverResult(
            plan=combined_plan,
            expanded_nodes=expanded_nodes + suffix_completion.expanded_nodes,
            generated_nodes=generated_nodes + suffix_completion.generated_nodes,
            closed_nodes=expanded_nodes + suffix_completion.closed_nodes,
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
            return None
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
    route_oracle = RouteOracle(master)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    if (
        current_blockage.total_blockage_pressure > 0
        and len(clearing_plan) >= ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS
    ):
        return _build_route_blockage_tail_partial_result(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=clearing_plan,
            state=state,
            master=master,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
        )
    if (
        current_blockage.total_blockage_pressure > 0
        and len(clearing_plan) < ROUTE_BLOCKAGE_TAIL_CLEARANCE_MAX_CLEARING_HOOKS
    ):
        if getattr(current_blockage, "facts_by_blocking_track", {}):
            parked_work_position_completion = (
                _try_parked_work_position_blocker_clearance_resume(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=clearing_plan,
                    state=state,
                    master=master,
                    time_budget_ms=_remaining_child_budget_ms(
                        started_at,
                        time_budget_ms,
                    ),
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                    seen_state_keys=seen_state_keys,
                )
            )
            if parked_work_position_completion is not None:
                return _as_route_blockage_tail_completion(parked_work_position_completion)
        for move, next_state in _route_blockage_tail_clearance_candidates(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            current_blockage=current_blockage,
        ):
            next_state_key = _state_key(next_state, plan_input)
            if next_state_key in seen_state_keys:
                continue
            remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
            if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                return best_or_current_partial()
            child_budget_ms = _route_blockage_tail_branch_budget_ms(remaining_ms)
            completion = _try_tail_clearance_resume_from_state(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=[*clearing_plan, move],
                state=next_state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=child_budget_ms,
                expanded_nodes=expanded_nodes + 1,
                generated_nodes=generated_nodes + 1,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                seen_state_keys=seen_state_keys | frozenset({next_state_key}),
            )
            if completion is not None:
                if completion.is_complete:
                    return _as_route_blockage_tail_completion(completion)
                remember_partial(completion)
    if current_blockage.total_blockage_pressure <= 0:
        random_area_completion = _try_route_clean_random_area_tail_completion_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=[*prefix_plan, *clearing_plan],
            state=state,
            master=master,
            time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
        )
        if random_area_completion is not None:
            if random_area_completion.is_complete:
                return random_area_completion
            remember_partial(random_area_completion)
    if current_blockage.total_blockage_pressure > 0:
        if getattr(current_blockage, "facts_by_blocking_track", {}):
            parked_work_position_completion = (
                _try_parked_work_position_blocker_clearance_resume(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    clearing_plan=clearing_plan,
                    state=state,
                    master=master,
                    time_budget_ms=_remaining_child_budget_ms(
                        started_at,
                        time_budget_ms,
                    ),
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                    seen_state_keys=seen_state_keys,
                )
            )
            if parked_work_position_completion is not None:
                return _as_route_blockage_tail_completion(parked_work_position_completion)
    direct_completion = _try_direct_blocked_tail_completion_from_state(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=prefix_plan,
        clearing_plan=clearing_plan,
        state=state,
        initial_blockage=initial_blockage,
        master=master,
        time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if direct_completion is not None:
        return _as_route_blockage_tail_completion(direct_completion)
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return best_or_current_partial()
    goal_frontier_completion = _try_goal_frontier_tail_completion_from_state(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=[*prefix_plan, *clearing_plan],
        state=state,
        master=master,
        time_budget_ms=remaining_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if goal_frontier_completion is not None:
        if goal_frontier_completion.is_complete:
            if not remember_complete(
                _as_route_blockage_tail_completion(goal_frontier_completion)
            ):
                return best_complete
        else:
            remember_partial(goal_frontier_completion)
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return best_complete or best_or_current_partial()
    structural = compute_structural_metrics(plan_input, state)
    structural_cleanup = _try_route_clean_structural_tail_cleanup_from_state(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=[*prefix_plan, *clearing_plan],
        state=state,
        master=master,
        time_budget_ms=remaining_ms,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if structural_cleanup is not None:
        if structural_cleanup.is_complete:
            if not remember_complete(_as_route_blockage_tail_completion(structural_cleanup)):
                return best_complete
        else:
            remember_partial(structural_cleanup)
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return best_complete or best_or_current_partial()
    route_clean_constructive_tail = (
        structural.unfinished_count <= LOCALIZED_RESUME_MAX_UNFINISHED
    )
    if route_clean_constructive_tail:
        constructive_tail = _try_constructive_tail_rescue_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=[*prefix_plan, *clearing_plan],
            state=state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if constructive_tail is not None:
            if constructive_tail.is_complete:
                if not remember_complete(_as_route_blockage_tail_completion(constructive_tail)):
                    return best_complete
            else:
                remember_partial(constructive_tail)
    if best_complete is not None:
        return best_complete
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return best_or_current_partial()
    completion: SolverResult | None = None
    if structural.unfinished_count <= LOCALIZED_RESUME_MAX_UNFINISHED:
        completion = _try_localized_resume_completion(
            plan_input=plan_input,
            initial_state=state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
    if completion is None or not completion.is_complete:
        if (
            getattr(initial_blockage, "total_blockage_pressure", 0) <= 0
            and current_blockage.total_blockage_pressure <= 0
            and structural.unfinished_count <= LOCALIZED_RESUME_MAX_UNFINISHED
            and best_partial is None
        ):
            return None
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return best_or_current_partial()
        broad_completion = _try_broad_tail_suffix_completion_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=[*prefix_plan, *clearing_plan],
            state=state,
            master=master,
            time_budget_ms=remaining_ms,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            fallback_stage="route_blockage_tail_clearance",
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if broad_completion is not None:
            return broad_completion
        return best_or_current_partial()
    if current_blockage.total_blockage_pressure > 0:
        return best_or_current_partial()
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


def _route_blockage_tail_branch_budget_ms(remaining_ms: float) -> float:
    if remaining_ms <= ROUTE_BLOCKAGE_TAIL_FOLLOWUP_RESERVE_MS * 2:
        return remaining_ms
    return max(
        MIN_CHILD_STAGE_BUDGET_MS,
        remaining_ms - ROUTE_BLOCKAGE_TAIL_FOLLOWUP_RESERVE_MS,
    )


def _try_broad_tail_suffix_completion_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    fallback_stage: str,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if state.loco_carry or time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    suffix_completion = _try_direct_tail_suffix_search(
        plan_input=plan_input,
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if suffix_completion is None or not suffix_completion.is_complete:
        return None
    result = SolverResult(
        plan=[*prefix_plan, *suffix_completion.plan],
        expanded_nodes=expanded_nodes + suffix_completion.expanded_nodes,
        generated_nodes=generated_nodes + suffix_completion.generated_nodes,
        closed_nodes=expanded_nodes + suffix_completion.closed_nodes,
        elapsed_ms=suffix_completion.elapsed_ms,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage=fallback_stage,
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
        return None


def _build_route_blockage_tail_partial_result(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    clearing_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    expanded_nodes: int,
    generated_nodes: int,
    elapsed_ms: float,
) -> SolverResult | None:
    if not clearing_plan:
        return None
    route_oracle = RouteOracle(master)
    debug_stats = {
        "partial_structural_metrics": _debug_dict(
            compute_structural_metrics(
                plan_input,
                state,
            )
        ),
        "partial_route_blockage_plan": _debug_dict(
            compute_route_blockage_plan(
                plan_input,
                state,
                route_oracle,
            )
        ),
        "plan_shape_metrics": summarize_plan_shape([*prefix_plan, *clearing_plan]),
    }
    result = SolverResult(
        plan=[],
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        closed_nodes=expanded_nodes,
        elapsed_ms=elapsed_ms,
        is_complete=False,
        is_proven_optimal=False,
        fallback_stage="route_blockage_tail_clearance",
        partial_plan=[*prefix_plan, *clearing_plan],
        partial_fallback_stage="route_blockage_tail_clearance",
        debug_stats=debug_stats,
    )
    try:
        return _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=original_initial_state,
        )
    except Exception:  # noqa: BLE001
        return result


def _try_carried_goal_blocker_clearance_resume(
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
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    if not state.loco_carry:
        return None
    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    current_structural = compute_structural_metrics(plan_input, state)
    blocking_vehicle_nos = set(getattr(initial_blockage, "blocking_vehicle_nos", []))
    blocking_vehicle_nos.update(getattr(current_blockage, "blocking_vehicle_nos", []))
    if not (set(state.loco_carry) & blocking_vehicle_nos):
        return None

    from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], HookAction, ReplayState]] = []
    for move in generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    ):
        if move.action_type != "DETACH":
            continue
        moved = set(move.vehicle_nos)
        if not (moved & blocking_vehicle_nos):
            continue
        if not all(vehicle_no in state.loco_carry for vehicle_no in move.vehicle_nos):
            continue
        if not all(
            move.target_track
            in goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
            )
            for vehicle_no in move.vehicle_nos
            for vehicle in [vehicle_by_no.get(vehicle_no)]
            if vehicle is not None
        ):
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
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        next_blockage = compute_route_blockage_plan(plan_input, next_state, route_oracle)
        next_structural = compute_structural_metrics(plan_input, next_state)
        placed_goal_count = sum(
            1
            for vehicle_no in move.vehicle_nos
            for vehicle in [vehicle_by_no.get(vehicle_no)]
            if vehicle is not None
            and goal_is_satisfied(
                vehicle,
                track_name=move.target_track,
                state=next_state,
                plan_input=plan_input,
            )
        )
        if (
            placed_goal_count <= 0
            and next_blockage.total_blockage_pressure >= current_blockage.total_blockage_pressure
            and next_structural.unfinished_count >= current_structural.unfinished_count
        ):
            continue
        candidates.append(
            (
                (
                    next_blockage.total_blockage_pressure,
                    next_structural.unfinished_count,
                    -placed_goal_count,
                    len(move.path_tracks),
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
                next_state,
            )
        )
    candidates.sort(key=lambda item: item[0])

    for _score, move, next_state in candidates:
        next_state_key = _state_key(next_state, plan_input)
        if seen_state_keys is not None and next_state_key in seen_state_keys:
            continue
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        completion = _try_tail_clearance_resume_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=[*clearing_plan, move],
            state=next_state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=remaining_ms,
            expanded_nodes=expanded_nodes + 1,
            generated_nodes=generated_nodes + 1,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=(
                (seen_state_keys or frozenset({_state_key(state, plan_input)}))
                | frozenset({next_state_key})
            ),
        )
        if completion is not None:
            return completion
        if _is_goal(plan_input, next_state):
            result = SolverResult(
                plan=[*prefix_plan, *clearing_plan, move],
                expanded_nodes=expanded_nodes + 1,
                generated_nodes=generated_nodes + 1,
                closed_nodes=expanded_nodes + 1,
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
                continue
    return None


def _try_carried_route_blocker_source_block_completion(
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
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    if not state.loco_carry or time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    current_structural = compute_structural_metrics(plan_input, state)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    candidates: list[
        tuple[
            tuple[int, int, int, int, str, tuple[str, ...]],
            ReplayState,
            list[HookAction],
        ]
    ] = []
    for attach_move in _carried_route_blocker_source_attach_candidates(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
    ):
        try:
            attached_state = _apply_move(
                state=state,
                move=attach_move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if _state_key(attached_state, plan_input) == _state_key(state, plan_input):
            continue
        detach_plan = _carried_route_blocker_ordered_detach_plan(
            plan_input=plan_input,
            state=attached_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            original_carry=list(state.loco_carry),
        )
        if detach_plan is None:
            next_state = attached_state
            detach_moves: list[HookAction] = []
        else:
            next_state, detach_moves = detach_plan
            if next_state.loco_carry:
                continue
        next_structural = compute_structural_metrics(plan_input, next_state)
        next_blockage = compute_route_blockage_plan(plan_input, next_state, route_oracle)
        if (
            not next_state.loco_carry
            and next_structural.unfinished_count >= current_structural.unfinished_count
            and next_structural.work_position_unfinished_count
            >= current_structural.work_position_unfinished_count
            and next_blockage.total_blockage_pressure
            >= current_blockage.total_blockage_pressure
        ):
            continue
        step_plan = [attach_move, *detach_moves]
        candidates.append(
            (
                (
                    next_blockage.total_blockage_pressure,
                    next_structural.unfinished_count,
                    next_structural.work_position_unfinished_count,
                    len(step_plan),
                    attach_move.source_track,
                    tuple(attach_move.vehicle_nos),
                ),
                next_state,
                step_plan,
            )
        )
    candidates.sort(key=lambda item: item[0])

    base_seen = seen_state_keys or frozenset({_state_key(state, plan_input)})
    for _score, next_state, step_plan in candidates[:4]:
        next_state_key = _state_key(next_state, plan_input)
        if next_state_key in base_seen:
            continue
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        completion = _try_tail_clearance_resume_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=[*clearing_plan, *step_plan],
            state=next_state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=remaining_ms,
            expanded_nodes=expanded_nodes + 1,
            generated_nodes=generated_nodes + len(step_plan),
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=base_seen | frozenset({next_state_key}),
        )
        if completion is not None:
            return completion
    return None


def _carried_route_blocker_source_attach_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
) -> list[HookAction]:
    if not state.loco_carry:
        return []
    from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length
        for vehicle in plan_input.vehicles
    }
    current_tracks = _vehicle_track_lookup(state)
    candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], HookAction]] = []
    for source_track, seq in sorted(state.track_sequences.items()):
        if not seq or source_track == state.loco_track_name:
            continue
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=source_track,
            occupied_track_sequences=state.track_sequences,
            carried_train_length_m=sum(
                length_by_vehicle.get(vehicle_no, 0.0)
                for vehicle_no in state.loco_carry
            ),
            loco_node=state.loco_node,
        )
        if not access_result.is_valid:
            continue
        prefix: list[str] = []
        seen_unfinished = False
        unfinished_count = 0
        last_unfinished_prefix_size = 0
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                break
            prefix.append(vehicle_no)
            block_vehicles = [
                vehicle_by_no.get(block_vehicle_no)
                for block_vehicle_no in prefix
            ]
            if any(block_vehicle is None for block_vehicle in block_vehicles):
                break
            if validate_hook_vehicle_group(
                [block_vehicle for block_vehicle in block_vehicles if block_vehicle is not None]
            ):
                break
            if not goal_is_satisfied(
                vehicle,
                track_name=current_tracks.get(vehicle_no, source_track),
                state=state,
                plan_input=plan_input,
            ):
                seen_unfinished = True
                unfinished_count += 1
                last_unfinished_prefix_size = len(prefix)
            if not seen_unfinished:
                continue
            attached_carry = [*state.loco_carry, *prefix]
            if not _carried_route_blocker_detach_groups(attached_carry, vehicle_by_no):
                continue
            incomplete_prefix_penalty = (
                0 if len(prefix) >= last_unfinished_prefix_size else 1
            )
            candidates.append(
                (
                    (
                        incomplete_prefix_penalty,
                        _carried_route_blocker_goal_group_count(
                            attached_carry,
                            vehicle_by_no,
                        ),
                        -unfinished_count,
                        -len(prefix),
                        len(access_result.branch_codes or ()),
                        source_track,
                        tuple(prefix),
                    ),
                    HookAction(
                        source_track=source_track,
                        target_track=source_track,
                        vehicle_nos=list(prefix),
                        path_tracks=[source_track],
                        action_type="ATTACH",
                    ),
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [move for _score, move in candidates[:16]]


def _carried_route_blocker_ordered_detach_plan(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    original_carry: list[str],
) -> tuple[ReplayState, list[HookAction]] | None:
    current_state = state
    moves: list[HookAction] = []
    while current_state.loco_carry:
        groups = _carried_route_blocker_detach_groups(
            list(current_state.loco_carry),
            vehicle_by_no,
        )
        if not groups:
            return None
        vehicle_nos, target_track = groups[0]
        detach = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=vehicle_nos,
            target_track=target_track,
        )
        if detach is None:
            return None
        try:
            next_state = _apply_move(
                state=current_state,
                move=detach,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
        if _state_key(next_state, plan_input) == _state_key(current_state, plan_input):
            return None
        current_state = next_state
        moves.append(detach)
    if not moves:
        return None
    if original_carry and not any(
        list(move.vehicle_nos) == list(original_carry)
        for move in moves
    ):
        return None
    return current_state, moves


def _carried_route_blocker_detach_groups(
    carry: list[str] | tuple[str, ...],
    vehicle_by_no: dict[str, Any],
) -> list[tuple[list[str], str]]:
    groups: list[tuple[list[str], str]] = []
    index = len(carry)
    while index > 0:
        vehicle = vehicle_by_no.get(carry[index - 1])
        if vehicle is None or not vehicle.goal.allowed_target_tracks:
            return []
        target_track = vehicle.goal.allowed_target_tracks[0]
        start = index - 1
        while start > 0:
            prev_vehicle = vehicle_by_no.get(carry[start - 1])
            if prev_vehicle is None or prev_vehicle.goal.allowed_target_tracks != [target_track]:
                break
            start -= 1
        groups.append((list(carry[start:index]), target_track))
        index = start
    return groups


def _carried_route_blocker_goal_group_count(
    carry: list[str] | tuple[str, ...],
    vehicle_by_no: dict[str, Any],
) -> int:
    groups = _carried_route_blocker_detach_groups(carry, vehicle_by_no)
    return len(groups) if groups else 10**9


def _try_carried_random_area_tail_resume(
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
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    if not state.loco_carry:
        return None
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    if not any(
        _is_soft_random_area_vehicle(vehicle_by_no.get(vehicle_no))
        for vehicle_no in state.loco_carry
    ):
        return None

    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    current_structural = compute_structural_metrics(plan_input, state)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)

    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], HookAction, ReplayState]] = []
    for move in generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
        require_empty_carry_followup=False,
    ):
        if move.action_type != "DETACH":
            continue
        if len(move.vehicle_nos) >= len(state.loco_carry):
            continue
        if not all(vehicle_no in state.loco_carry for vehicle_no in move.vehicle_nos):
            continue
        moved_soft_random = [
            vehicle_by_no.get(vehicle_no)
            for vehicle_no in move.vehicle_nos
            if _is_soft_random_area_vehicle(vehicle_by_no.get(vehicle_no))
        ]
        if not moved_soft_random:
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
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        placed_goal_count = sum(
            1
            for vehicle_no in move.vehicle_nos
            for vehicle in [vehicle_by_no.get(vehicle_no)]
            if vehicle is not None
            and goal_is_satisfied(
                vehicle,
                track_name=move.target_track,
                state=next_state,
                plan_input=plan_input,
            )
        )
        if placed_goal_count <= 0:
            continue
        next_structural = compute_structural_metrics(plan_input, next_state)
        next_blockage = compute_route_blockage_plan(plan_input, next_state, route_oracle)
        if (
            next_structural.unfinished_count >= current_structural.unfinished_count
            and next_structural.area_random_unfinished_count
            >= current_structural.area_random_unfinished_count
            and next_blockage.total_blockage_pressure
            >= current_blockage.total_blockage_pressure
        ):
            continue
        candidates.append(
            (
                (
                    len(next_state.loco_carry),
                    next_structural.area_random_unfinished_count,
                    next_structural.unfinished_count,
                    len(move.path_tracks),
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
                next_state,
            )
        )
    candidates.sort(key=lambda item: item[0])

    for _score, move, next_state in candidates:
        next_state_key = _state_key(next_state, plan_input)
        if seen_state_keys is not None and next_state_key in seen_state_keys:
            continue
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        completion = _try_tail_clearance_resume_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=[*clearing_plan, move],
            state=next_state,
            initial_blockage=initial_blockage,
            master=master,
            time_budget_ms=remaining_ms,
            expanded_nodes=expanded_nodes + 1,
            generated_nodes=generated_nodes + 1,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=(
                (seen_state_keys or frozenset({_state_key(state, plan_input)}))
                | frozenset({next_state_key})
            ),
        )
        if completion is not None:
            return completion
        if _is_goal(plan_input, next_state):
            result = SolverResult(
                plan=[*prefix_plan, *clearing_plan, move],
                expanded_nodes=expanded_nodes + 1,
                generated_nodes=generated_nodes + 1,
                closed_nodes=expanded_nodes + 1,
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
                continue
    return None


def _is_soft_random_area_vehicle(vehicle: Any) -> bool:
    if vehicle is None:
        return False
    goal = vehicle.goal
    return (
        goal.target_mode in {"AREA", "SNAPSHOT"}
        and goal.target_area_code is not None
        and ":RANDOM" in goal.target_area_code
    )


def _try_parked_work_position_blocker_clearance_resume(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    clearing_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    enable_depot_late_scheduling: bool,
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    if state.loco_carry:
        return None
    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    facts_by_blocking_track = getattr(current_blockage, "facts_by_blocking_track", {})
    if current_blockage.total_blockage_pressure <= 0 or not facts_by_blocking_track:
        return None
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    current_structural = compute_structural_metrics(plan_input, state)
    blocking_vehicle_nos: set[str] = set()
    for fact in facts_by_blocking_track.values():
        blocking_vehicle_nos.update(getattr(fact, "blocking_vehicle_nos", []))

    candidates: list[tuple[tuple[int, int, int, int, str], ReplayState, list[HookAction]]] = []
    for vehicle_no in sorted(blocking_vehicle_nos):
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.goal.work_position_kind is None:
            continue
        step = _build_work_position_tail_step(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle=vehicle,
        )
        if step is None:
            continue
        next_state, step_plan = step
        if not step_plan or _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        next_blockage = compute_route_blockage_plan(plan_input, next_state, route_oracle)
        next_structural = compute_structural_metrics(plan_input, next_state)
        if (
            next_blockage.total_blockage_pressure >= current_blockage.total_blockage_pressure
            and next_structural.work_position_unfinished_count
            >= current_structural.work_position_unfinished_count
            and next_structural.unfinished_count >= current_structural.unfinished_count
        ):
            continue
        candidates.append(
            (
                (
                    next_blockage.total_blockage_pressure,
                    next_structural.unfinished_count,
                    next_structural.work_position_unfinished_count,
                    len(step_plan),
                    vehicle.vehicle_no,
                ),
                next_state,
                step_plan,
            )
        )
    candidates.sort(key=lambda item: item[0])

    for _score, next_state, step_plan in candidates:
        next_state_key = _state_key(next_state, plan_input)
        if seen_state_keys is not None and next_state_key in seen_state_keys:
            continue
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        completion = _try_tail_clearance_resume_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            clearing_plan=[*clearing_plan, *step_plan],
            state=next_state,
            initial_blockage=current_blockage,
            master=master,
            time_budget_ms=remaining_ms,
            expanded_nodes=expanded_nodes + 1,
            generated_nodes=generated_nodes + len(step_plan),
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            seen_state_keys=(
                (seen_state_keys or frozenset({_state_key(state, plan_input)}))
                | frozenset({next_state_key})
            ),
        )
        if completion is not None:
            return completion
        if _is_goal(plan_input, next_state):
            result = SolverResult(
                plan=[*prefix_plan, *clearing_plan, *step_plan],
                expanded_nodes=expanded_nodes + 1,
                generated_nodes=generated_nodes + len(step_plan),
                closed_nodes=expanded_nodes + 1,
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
                continue
    return None


def _try_carried_work_position_clearance_resume(
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
    seen_state_keys: frozenset[tuple] | None = None,
) -> SolverResult | None:
    if not state.loco_carry:
        return None
    started_at = perf_counter()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    carried_work_position_vehicles = [
        vehicle_by_no[vehicle_no]
        for vehicle_no in state.loco_carry
        if vehicle_no in vehicle_by_no
        and vehicle_by_no[vehicle_no].goal.work_position_kind is not None
    ]
    if not carried_work_position_vehicles:
        return None

    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    route_oracle = RouteOracle(master)
    current_blockage = compute_route_blockage_plan(plan_input, state, route_oracle)
    current_structural = compute_structural_metrics(plan_input, state)
    candidates = _route_blockage_tail_clearance_candidates(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        current_blockage=current_blockage,
    )
    candidates.extend(
        _carried_work_position_staging_detach_candidates(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            initial_blockage=initial_blockage,
        )
    )
    staged_candidates: list[tuple[tuple[int, int, int, str, tuple[str, ...]], HookAction, ReplayState]] = []
    for move, staged_state in candidates:
        if move.action_type != "DETACH":
            continue
        moved_work_position = [
            vehicle
            for vehicle in carried_work_position_vehicles
            if vehicle.vehicle_no in move.vehicle_nos
        ]
        if not moved_work_position or staged_state.loco_carry:
            continue
        if _state_key(staged_state, plan_input) == _state_key(state, plan_input):
            continue
        staged_blockage = compute_route_blockage_plan(
            plan_input,
            staged_state,
            route_oracle,
        )
        staged_candidates.append(
            (
                (
                    staged_blockage.total_blockage_pressure,
                    len(staged_state.track_sequences.get(move.target_track, [])),
                    len(move.path_tracks),
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
                staged_state,
            )
        )
    staged_candidates.sort(key=lambda item: item[0])

    for _score, staging_move, staged_state in staged_candidates:
        for vehicle in sorted(
            carried_work_position_vehicles,
            key=lambda item: item.vehicle_no,
        ):
            if vehicle.vehicle_no not in staging_move.vehicle_nos:
                continue
            track = _vehicle_track_lookup(staged_state).get(vehicle.vehicle_no)
            if track is None:
                continue
            local_state = staged_state
            local_plan: list[HookAction] = [staging_move]
            if not goal_is_satisfied(
                vehicle,
                track_name=track,
                state=local_state,
                plan_input=plan_input,
            ):
                step = _build_work_position_tail_step(
                    plan_input=plan_input,
                    state=staged_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    vehicle=vehicle,
                )
                if step is None:
                    continue
                local_state, step_plan = step
                if _state_key(local_state, plan_input) == _state_key(staged_state, plan_input):
                    continue
                local_plan.extend(step_plan)
            next_structural = compute_structural_metrics(plan_input, local_state)
            if (
                next_structural.work_position_unfinished_count
                >= current_structural.work_position_unfinished_count
                and next_structural.unfinished_count >= current_structural.unfinished_count
                and not _is_goal(plan_input, local_state)
            ):
                continue
            local_state_key = _state_key(local_state, plan_input)
            if seen_state_keys is not None and local_state_key in seen_state_keys:
                continue
            remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
            if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                return None
            completion = _try_tail_clearance_resume_from_state(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                clearing_plan=[*clearing_plan, *local_plan],
                state=local_state,
                initial_blockage=initial_blockage,
                master=master,
                time_budget_ms=remaining_ms,
                expanded_nodes=expanded_nodes + 1,
                generated_nodes=generated_nodes + len(local_plan),
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                seen_state_keys=(
                    (seen_state_keys or frozenset({_state_key(state, plan_input)}))
                    | frozenset({local_state_key})
                ),
            )
            if completion is not None:
                return completion
            if _is_goal(plan_input, local_state):
                result = SolverResult(
                    plan=[*prefix_plan, *clearing_plan, *local_plan],
                    expanded_nodes=expanded_nodes + 1,
                    generated_nodes=generated_nodes + len(local_plan),
                    closed_nodes=expanded_nodes + 1,
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
                    continue
    return None


def _carried_work_position_staging_detach_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    initial_blockage: Any,
) -> list[tuple[HookAction, ReplayState]]:
    if not state.loco_carry:
        return []
    from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length
        for vehicle in plan_input.vehicles
    }
    block = list(state.loco_carry)
    if not any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind is not None
        for vehicle_no in block
    ):
        return []
    block_vehicles = [vehicle_by_no.get(vehicle_no) for vehicle_no in block]
    if any(vehicle is None for vehicle in block_vehicles):
        return []
    if validate_hook_vehicle_group([vehicle for vehicle in block_vehicles if vehicle is not None]):
        return []
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in block)
    current_pressure = compute_route_blockage_plan(
        plan_input,
        state,
        route_oracle,
    ).total_blockage_pressure
    max_pressure = max(
        current_pressure,
        _optional_int(getattr(initial_blockage, "total_blockage_pressure", None)) or 0,
    )
    candidates: list[tuple[tuple[int, int, int, str], HookAction, ReplayState]] = []
    for target_track in sorted(STAGING_TRACKS):
        if target_track == state.loco_track_name:
            continue
        track = master.tracks.get(target_track)
        if track is None or not track.allow_parking:
            continue
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(target_track, [])
        )
        if current_length + block_length > track.effective_length_m + 1e-9:
            continue
        source_node = state.loco_node if state.track_sequences.get(state.loco_track_name) else None
        target_node = route_oracle.order_end_node(target_track)
        path_tracks = route_oracle.resolve_clear_path_tracks(
            state.loco_track_name,
            target_track,
            occupied_track_sequences=state.track_sequences,
            source_node=source_node,
            target_node=target_node,
        )
        if path_tracks is None:
            continue
        route = route_oracle.resolve_route_for_path_tracks(
            path_tracks,
            source_node=source_node,
            target_node=target_node,
        )
        if route is None:
            continue
        route_result = route_oracle.validate_path(
            source_track=state.loco_track_name,
            target_track=target_track,
            path_tracks=path_tracks,
            train_length_m=sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in state.loco_carry),
            occupied_track_sequences=state.track_sequences,
            expected_path_tracks=path_tracks,
            route=route,
            source_node=source_node,
            target_node=target_node,
        )
        if not route_result.is_valid:
            continue
        move = HookAction(
            source_track=state.loco_track_name,
            target_track=target_track,
            vehicle_nos=list(block),
            path_tracks=path_tracks,
            action_type="DETACH",
        )
        try:
            next_state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        next_pressure = compute_route_blockage_plan(
            plan_input,
            next_state,
            route_oracle,
        ).total_blockage_pressure
        if next_pressure > max_pressure:
            continue
        candidates.append(
            (
                (
                    next_pressure,
                    len(next_state.track_sequences.get(target_track, [])),
                    len(path_tracks),
                    target_track,
                ),
                move,
                next_state,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [(move, next_state) for _score, move, next_state in candidates[:4]]


def _try_goal_frontier_tail_completion(
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
    return _try_goal_frontier_tail_completion_from_state(
        plan_input=plan_input,
        original_initial_state=initial_state,
        prefix_plan=list(partial_plan),
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )


def _try_route_clean_random_area_tail_completion(
    result: SolverResult,
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
) -> SolverResult | None:
    if not result.partial_plan or result.is_complete:
        return None
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    state = ReplayState.model_validate(initial_state.model_dump())
    try:
        for move in result.partial_plan:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
    except Exception:  # noqa: BLE001
        return None
    return _try_route_clean_random_area_tail_completion_from_state(
        plan_input=plan_input,
        original_initial_state=initial_state,
        prefix_plan=list(result.partial_plan),
        state=state,
        master=master,
        time_budget_ms=time_budget_ms,
        expanded_nodes=result.expanded_nodes,
        generated_nodes=result.generated_nodes,
    )


def _try_route_clean_random_area_tail_completion_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
) -> SolverResult | None:
    if state.loco_carry or time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    route_oracle = RouteOracle(master)
    if compute_route_blockage_plan(
        plan_input,
        state,
        route_oracle,
    ).total_blockage_pressure > 0:
        return None
    structural = compute_structural_metrics(plan_input, state)
    area_random_unfinished = getattr(
        structural,
        "area_random_unfinished_count",
        0,
    )
    if (
        structural.unfinished_count <= 0
        or area_random_unfinished <= 0
        or structural.work_position_unfinished_count > 0
    ):
        return None

    started_at = perf_counter()
    current_state = state
    current_plan: list[HookAction] = []
    seen_keys: set[tuple] = {_state_key(current_state, plan_input)}
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    current_structural = structural

    while True:
        if _is_goal(plan_input, current_state):
            if not current_plan:
                return None
            result = SolverResult(
                plan=[*prefix_plan, *current_plan],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes + len(current_plan),
                closed_nodes=expanded_nodes,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_complete=True,
                is_proven_optimal=False,
                fallback_stage="route_clean_random_area_tail_completion",
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

        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            break
        step = _route_clean_random_area_tail_step(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            baseline_structural=current_structural,
        )
        if step is None:
            break
        next_state, step_plan = step
        next_key = _state_key(next_state, plan_input)
        if next_key in seen_keys:
            break
        next_structural = compute_structural_metrics(plan_input, next_state)
        if not _random_area_tail_step_improves(
            before=current_structural,
            after=next_structural,
        ):
            break
        current_state = next_state
        current_plan.extend(step_plan)
        current_structural = next_structural
        seen_keys.add(next_key)

    if not current_plan:
        return None
    result = SolverResult(
        plan=[],
        partial_plan=[*prefix_plan, *current_plan],
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes + len(current_plan),
        closed_nodes=expanded_nodes,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=False,
        is_proven_optimal=False,
        fallback_stage="route_clean_random_area_tail_completion",
        partial_fallback_stage="route_clean_random_area_tail_completion",
        debug_stats={
            "partial_structural_metrics": current_structural.to_dict(),
            "partial_route_blockage_plan": compute_route_blockage_plan(
                plan_input,
                current_state,
                route_oracle,
            ),
            "plan_shape_metrics": summarize_plan_shape([*prefix_plan, *current_plan]),
        },
    )
    result = replace(
        result,
        debug_stats={
            **(result.debug_stats or {}),
            "partial_route_blockage_plan": _debug_dict(
                (result.debug_stats or {})["partial_route_blockage_plan"]
            ),
        },
    )
    try:
        return _attach_verification(
            result,
            plan_input=plan_input,
            master=master,
            initial_state=original_initial_state,
        )
    except Exception:  # noqa: BLE001
        return result


def _random_area_tail_step_improves(*, before: Any, after: Any) -> bool:
    return (
        after.unfinished_count < before.unfinished_count
        or getattr(after, "area_random_unfinished_count", 0)
        < getattr(before, "area_random_unfinished_count", 0)
        or after.front_blocker_count < before.front_blocker_count
        or after.goal_track_blocker_count < before.goal_track_blocker_count
    )


def _route_clean_random_area_tail_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    baseline_structural: Any,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    candidates: list[
        tuple[
            tuple[int, int, int, int, int, int, str, tuple[str, ...]],
            ReplayState,
            list[HookAction],
        ]
    ] = []
    track_by_vehicle = _vehicle_track_lookup(state)
    for source_track, seq in sorted(state.track_sequences.items()):
        if not seq:
            continue
        preserved_prefix: list[str] = []
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                break
            if not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            preserved_prefix.append(vehicle_no)
        start = len(preserved_prefix)
        if start >= len(seq):
            continue
        target_block: list[str] = []
        for vehicle_no in seq[start:]:
            vehicle = vehicle_by_no.get(vehicle_no)
            if (
                vehicle is None
                or not _is_soft_random_area_vehicle(vehicle)
                or goal_is_satisfied(
                    vehicle,
                    track_name=track_by_vehicle.get(vehicle_no, source_track),
                    state=state,
                    plan_input=plan_input,
                )
            ):
                break
            target_block.append(vehicle_no)
        for block_size in range(len(target_block), 0, -1):
            block = target_block[:block_size]
            for target_track in _shared_random_area_target_tracks(
                block,
                source_track=source_track,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
            ):
                step = _build_random_area_tail_reorder_step(
                    plan_input=plan_input,
                    state=state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    preserved_prefix=preserved_prefix,
                    target_block=block,
                    target_track=target_track,
                )
                if step is None:
                    continue
                next_state, step_plan = step
                if next_state.loco_carry:
                    continue
                next_structural = compute_structural_metrics(plan_input, next_state)
                if not _random_area_tail_step_improves(
                    before=baseline_structural,
                    after=next_structural,
                ):
                    continue
                try:
                    route_pressure = compute_route_blockage_plan(
                        plan_input,
                        next_state,
                        route_oracle,
                    ).total_blockage_pressure
                except Exception:  # noqa: BLE001
                    route_pressure = 10**9
                candidates.append(
                    (
                        (
                            next_structural.unfinished_count,
                            next_structural.area_random_unfinished_count,
                            next_structural.front_blocker_count,
                            next_structural.goal_track_blocker_count,
                            route_pressure,
                            len(step_plan),
                            target_track,
                            tuple(block),
                        ),
                        next_state,
                        step_plan,
                    )
                )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def _shared_random_area_target_tracks(
    vehicle_nos: list[str],
    *,
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
) -> list[str]:
    from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks

    shared: list[str] | None = None
    for vehicle_no in vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        targets = [
            target
            for target in goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
            if target != source_track
        ]
        if not targets:
            return []
        if shared is None:
            shared = targets
            continue
        target_set = set(targets)
        shared = [target for target in shared if target in target_set]
        if not shared:
            return []
    return shared or []


def _build_random_area_tail_reorder_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    source_track: str,
    preserved_prefix: list[str],
    target_block: list[str],
    target_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    attach_block = [*preserved_prefix, *target_block]
    if not attach_block or source_track == target_track:
        return None
    attach = _build_goal_frontier_exact_attach(
        plan_input=plan_input,
        state=state,
        master=master,
        source_track=source_track,
        prefix_block=attach_block,
    )
    if attach is None:
        return None
    try:
        carried_state = _apply_move(
            state=state,
            move=attach,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    detach_target = _build_work_position_exact_detach(
        plan_input=plan_input,
        state=carried_state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        vehicle_nos=target_block,
        target_track=target_track,
    )
    if detach_target is None:
        return None
    try:
        target_state = _apply_move(
            state=carried_state,
            move=detach_target,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=target_state,
            plan_input=plan_input,
        )
        for vehicle_no in target_block
    ):
        return None
    step_plan = [attach, detach_target]
    next_state = target_state
    if preserved_prefix:
        restore = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=next_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=preserved_prefix,
            target_track=source_track,
        )
        if restore is None:
            return None
        try:
            next_state = _apply_move(
                state=next_state,
                move=restore,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
        step_plan.append(restore)
        if not _vehicles_satisfied_on_track(
            preserved_prefix,
            track_name=source_track,
            state=next_state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        ):
            return None
    return next_state, step_plan


def _try_route_clean_structural_tail_cleanup_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    if state.loco_carry:
        return None
    started_at = perf_counter()
    current_state = state
    current_plan: list[HookAction] = []
    current_structural = compute_structural_metrics(plan_input, current_state)
    seen_keys: set[tuple] = {_state_key(current_state, plan_input)}
    route_oracle = RouteOracle(master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}

    def structural_key(metrics: Any) -> tuple[int, int, int, int]:
        return (
            metrics.unfinished_count,
            metrics.front_blocker_count,
            metrics.goal_track_blocker_count,
            metrics.work_position_unfinished_count,
        )

    while True:
        if _is_goal(plan_input, current_state):
            if not current_plan:
                return None
            result = SolverResult(
                plan=[*prefix_plan, *current_plan],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=expanded_nodes,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_complete=True,
                is_proven_optimal=False,
                fallback_stage="route_clean_structural_tail_cleanup",
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
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if (
            remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS
            or current_structural.unfinished_count <= 0
        ):
            break
        moved = False
        for vehicle in _unfinished_work_position_vehicles(
            plan_input,
            current_state,
            vehicle_by_no,
        ):
            step = _build_work_position_tail_step(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle=vehicle,
            )
            if step is None:
                continue
            next_state, step_plan = step
            if next_state.loco_carry:
                continue
            next_key = _state_key(next_state, plan_input)
            if next_key in seen_keys:
                continue
            next_structural = compute_structural_metrics(plan_input, next_state)
            if structural_key(next_structural) >= structural_key(current_structural):
                continue
            current_state = next_state
            current_plan.extend(step_plan)
            current_structural = next_structural
            seen_keys.add(next_key)
            generated_nodes += len(step_plan)
            moved = True
            break
        if moved:
            continue
        for source_track, prefix_block, target_vehicle_no in _goal_frontier_blockers(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
        ):
            step = _goal_frontier_deep_block_step(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                prefix_block=prefix_block,
                target_vehicle_no=target_vehicle_no,
            )
            if step is None:
                continue
            next_state, step_plan = step
            if next_state.loco_carry:
                continue
            next_key = _state_key(next_state, plan_input)
            if next_key in seen_keys:
                continue
            next_structural = compute_structural_metrics(plan_input, next_state)
            if structural_key(next_structural) >= structural_key(current_structural):
                continue
            current_state = next_state
            current_plan.extend(step_plan)
            current_structural = next_structural
            seen_keys.add(next_key)
            generated_nodes += len(step_plan)
            moved = True
            break
        if not moved:
            break

    if not current_plan:
        return None
    if _is_goal(plan_input, current_state):
        result = SolverResult(
            plan=[*prefix_plan, *current_plan],
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            closed_nodes=expanded_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
            is_complete=True,
            is_proven_optimal=False,
            fallback_stage="route_clean_structural_tail_cleanup",
        )
    else:
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms > MIN_CHILD_STAGE_BUDGET_MS:
            suffix_completion = _try_direct_tail_suffix_search(
                plan_input=plan_input,
                state=current_state,
                master=master,
                time_budget_ms=remaining_ms,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
            )
            if suffix_completion is not None and suffix_completion.is_complete:
                result = SolverResult(
                    plan=[*prefix_plan, *current_plan, *suffix_completion.plan],
                    expanded_nodes=expanded_nodes + suffix_completion.expanded_nodes,
                    generated_nodes=generated_nodes + suffix_completion.generated_nodes,
                    closed_nodes=expanded_nodes + suffix_completion.closed_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    is_complete=True,
                    is_proven_optimal=False,
                    fallback_stage="route_clean_structural_tail_cleanup",
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
                    return None
        result = SolverResult(
            plan=[],
            partial_plan=[*prefix_plan, *current_plan],
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            closed_nodes=expanded_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
            is_complete=False,
            is_proven_optimal=False,
            fallback_stage="route_clean_structural_tail_cleanup",
            partial_fallback_stage="route_clean_structural_tail_cleanup",
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


def _try_goal_frontier_tail_completion_from_state(
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
    structural = compute_structural_metrics(plan_input, state)
    has_work_position_debt = structural.work_position_unfinished_count > 0
    if structural.unfinished_count <= 0 or (
        structural.front_blocker_count <= 0 and not has_work_position_debt
    ):
        return None

    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    direct_state = state
    direct_plan: list[HookAction] = []
    expanded_nodes = 0
    generated_nodes = 0

    while True:
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return _build_goal_frontier_budget_partial_result(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                direct_plan=direct_plan,
                direct_state=direct_state,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                master=master,
            )
        if _is_goal(plan_input, direct_state):
            if not direct_plan:
                return None
            result = SolverResult(
                plan=[*prefix_plan, *direct_plan],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=expanded_nodes,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_complete=True,
                is_proven_optimal=False,
                fallback_stage="goal_frontier_tail_completion",
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

        frontiers = _goal_frontier_blockers(
            plan_input=plan_input,
            state=direct_state,
            vehicle_by_no=vehicle_by_no,
        )
        if not frontiers:
            break
        made_progress = False
        for source_track, prefix_block, target_vehicle_no in frontiers:
            expanded_nodes += 1
            attach_prefix = _find_generated_move(
                generate_real_hook_moves(
                    plan_input,
                    direct_state,
                    master=master,
                    route_oracle=route_oracle,
                ),
                action_type="ATTACH",
                source_track=source_track,
                vehicle_nos=prefix_block,
            )
            if attach_prefix is None:
                attach_prefix = _build_goal_frontier_exact_attach(
                    plan_input=plan_input,
                    state=direct_state,
                    master=master,
                    source_track=source_track,
                    prefix_block=prefix_block,
                )
            if attach_prefix is None:
                continue
            generated_nodes += 1
            next_state = _apply_move(
                state=direct_state,
                move=attach_prefix,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            if _state_key(next_state, plan_input) == _state_key(direct_state, plan_input):
                continue
            staging_detach = _best_goal_frontier_staging_detach(
                plan_input=plan_input,
                state=next_state,
                master=master,
                route_oracle=route_oracle,
                prefix_block=prefix_block,
                source_track=source_track,
                target_vehicle_no=target_vehicle_no,
            )
            if staging_detach is None:
                deep_step = _goal_frontier_deep_block_step(
                    plan_input=plan_input,
                    state=direct_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    prefix_block=prefix_block,
                    target_vehicle_no=target_vehicle_no,
                )
                if deep_step is None:
                    continue
                direct_state, step_plan = deep_step
                generated_nodes += len(step_plan)
                direct_plan.extend(step_plan)
                made_progress = True
                break
            generated_nodes += 1
            direct_state = _apply_move(
                state=next_state,
                move=staging_detach,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            if _state_key(direct_state, plan_input) == _state_key(next_state, plan_input):
                continue
            direct_plan.extend([attach_prefix, staging_detach])
            made_progress = True

            remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
            if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                return _build_goal_frontier_budget_partial_result(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    direct_plan=direct_plan,
                    direct_state=direct_state,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    master=master,
                )
            target_vehicle = vehicle_by_no.get(target_vehicle_no)
            if target_vehicle is None:
                break
            target_track = _vehicle_track_lookup(direct_state).get(target_vehicle_no)
            if target_track is None or goal_is_satisfied(
                target_vehicle,
                track_name=target_track,
                state=direct_state,
                plan_input=plan_input,
            ):
                break
            target_seq = direct_state.track_sequences.get(target_track, [])
            try:
                target_index = target_seq.index(target_vehicle_no)
            except ValueError:
                break
            exact_target_block = list(target_seq[: target_index + 1])
            target_moves = generate_real_hook_moves(
                plan_input,
                direct_state,
                master=master,
                route_oracle=route_oracle,
            )
            attach_target = _find_best_goal_frontier_target_attach(
                target_moves,
                plan_input=plan_input,
                state=direct_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=target_track,
                target_vehicle_no=target_vehicle_no,
            )
            if attach_target is None:
                attach_target = _find_goal_frontier_attach_move(
                    target_moves,
                    source_track=target_track,
                    target_vehicle_no=target_vehicle_no,
                )
            if attach_target is None:
                attach_target = _build_goal_frontier_exact_attach(
                    plan_input=plan_input,
                    state=direct_state,
                    master=master,
                    source_track=target_track,
                    prefix_block=exact_target_block,
                )
            if attach_target is None:
                break
            target_block = list(attach_target.vehicle_nos)
            if target_block[: len(exact_target_block)] != exact_target_block:
                break
            generated_nodes += 1
            next_state = _apply_move(
                state=direct_state,
                move=attach_target,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            if _state_key(next_state, plan_input) == _state_key(direct_state, plan_input):
                break
            detach_target = _best_goal_frontier_target_detach(
                plan_input=plan_input,
                state=next_state,
                baseline_state=direct_state,
                master=master,
                route_oracle=route_oracle,
                target_block=target_block,
                target_vehicle_no=target_vehicle_no,
            )
            if detach_target is None:
                break
            generated_nodes += 1
            direct_state = _apply_move(
                state=next_state,
                move=detach_target,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            if _state_key(direct_state, plan_input) == _state_key(next_state, plan_input):
                break
            direct_plan.extend([attach_target, detach_target])
            if not _is_goal(plan_input, direct_state):
                restored = _restore_goal_frontier_prefix(
                    plan_input=plan_input,
                    state=direct_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    prefix_block=prefix_block,
                    staging_track=staging_detach.target_track,
                    source_track=source_track,
                )
                if restored is not None:
                    direct_state, restore_plan = restored
                    generated_nodes += len(restore_plan)
                    direct_plan.extend(restore_plan)
            break
        if not made_progress:
            break

    direct_prefix_completion = _try_route_clean_direct_prefix_tail_completion_from_state(
        plan_input=plan_input,
        state=direct_state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
    )
    if direct_prefix_completion is not None:
        direct_state, prefix_steps = direct_prefix_completion
        direct_plan.extend(prefix_steps)
        generated_nodes += len(prefix_steps)

    if direct_state.loco_carry:
        return None
    if not direct_plan and not has_work_position_debt:
        return None
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return _build_goal_frontier_budget_partial_result(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            direct_plan=direct_plan,
            direct_state=direct_state,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
            master=master,
        )
    work_position_completion = _try_work_position_rank_padding_completion_from_state(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=prefix_plan,
        direct_plan=direct_plan,
        state=direct_state,
        master=master,
        time_budget_ms=remaining_ms,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if work_position_completion is not None:
        return work_position_completion
    completion = _try_localized_resume_completion(
        plan_input=plan_input,
        initial_state=direct_state,
        master=master,
        time_budget_ms=remaining_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if completion is None or not completion.is_complete:
        return _build_goal_frontier_partial_result(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=prefix_plan,
            direct_plan=direct_plan,
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            elapsed_ms=(perf_counter() - started_at) * 1000,
            master=master,
        )
    result = SolverResult(
        plan=[*prefix_plan, *direct_plan, *completion.plan],
        expanded_nodes=expanded_nodes + completion.expanded_nodes,
        generated_nodes=generated_nodes + completion.generated_nodes,
        closed_nodes=expanded_nodes + completion.closed_nodes,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="goal_frontier_tail_completion",
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


def _try_work_position_rank_padding_completion_from_state(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    direct_plan: list[HookAction],
    state: ReplayState,
    master: MasterData,
    time_budget_ms: float,
    expanded_nodes: int,
    generated_nodes: int,
    enable_depot_late_scheduling: bool,
) -> SolverResult | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    started_at = perf_counter()
    route_oracle = RouteOracle(master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    current_state = state
    local_plan: list[HookAction] = []
    while True:
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            return None
        sequence_completion = _best_work_position_sequence_completion_step(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        )
        if sequence_completion is not None:
            next_state, step_plan = sequence_completion
            if _state_key(next_state, plan_input) == _state_key(current_state, plan_input):
                return None
            current_state = next_state
            local_plan.extend(step_plan)
            generated_nodes += len(step_plan)
            continue
        unfinished = _unfinished_work_position_vehicles(plan_input, current_state, vehicle_by_no)
        if not unfinished:
            break
        candidate_steps: list[tuple[tuple[int, int, str, tuple[str, ...]], ReplayState, list[HookAction]]] = []
        for vehicle in unfinished:
            step = _build_work_position_tail_step(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle=vehicle,
            )
            if step is None:
                continue
            next_state, step_plan = step
            next_structural = compute_structural_metrics(plan_input, next_state)
            candidate_steps.append(
                (
                    (
                        next_structural.unfinished_count,
                        next_structural.work_position_unfinished_count,
                        vehicle.vehicle_no,
                        tuple(
                            (move.action_type, move.source_track, move.target_track, tuple(move.vehicle_nos))
                            for move in step_plan
                        ),
                    ),
                    next_state,
                    step_plan,
                )
            )
        if not candidate_steps:
            if local_plan:
                continuation = _try_goal_frontier_tail_completion_from_state(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=[*prefix_plan, *direct_plan, *local_plan],
                    state=current_state,
                    master=master,
                    time_budget_ms=_remaining_child_budget_ms(started_at, time_budget_ms),
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
                if continuation is not None:
                    return continuation
                return _build_goal_frontier_partial_result(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    direct_plan=[*direct_plan, *local_plan],
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    master=master,
                )
            return None
        _score, next_state, step_plan = min(candidate_steps, key=lambda item: item[0])
        if _state_key(next_state, plan_input) == _state_key(current_state, plan_input):
            return None
        current_state = next_state
        local_plan.extend(step_plan)
        generated_nodes += len(step_plan)

    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    completion_plan: list[HookAction] = []
    completion_debug = None
    completion_expanded = 0
    completion_generated = 0
    if not _is_goal(plan_input, current_state):
        continuation = _try_goal_frontier_tail_completion_from_state(
            plan_input=plan_input,
            original_initial_state=original_initial_state,
            prefix_plan=[*prefix_plan, *direct_plan, *local_plan],
            state=current_state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if continuation is not None and continuation.is_complete:
            return continuation
        remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
        if remaining_ms <= MIN_CHILD_STAGE_BUDGET_MS:
            if continuation is not None:
                return continuation
            return _build_goal_frontier_partial_result(
                plan_input=plan_input,
                original_initial_state=original_initial_state,
                prefix_plan=prefix_plan,
                direct_plan=[*direct_plan, *local_plan],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                elapsed_ms=(perf_counter() - started_at) * 1000,
                master=master,
            )
        completion = _try_localized_resume_completion(
            plan_input=plan_input,
            initial_state=current_state,
            master=master,
            time_budget_ms=remaining_ms,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if completion is None or not completion.is_complete:
            if continuation is not None:
                return continuation
            if local_plan:
                return _build_goal_frontier_partial_result(
                    plan_input=plan_input,
                    original_initial_state=original_initial_state,
                    prefix_plan=prefix_plan,
                    direct_plan=[*direct_plan, *local_plan],
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    master=master,
                )
            return None
        completion_plan = list(completion.plan)
        completion_debug = completion.debug_stats
        completion_expanded = completion.expanded_nodes
        completion_generated = completion.generated_nodes
    result = SolverResult(
        plan=[*prefix_plan, *direct_plan, *local_plan, *completion_plan],
        expanded_nodes=expanded_nodes + completion_expanded,
        generated_nodes=generated_nodes + completion_generated,
        closed_nodes=expanded_nodes + completion_expanded,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=False,
        fallback_stage="goal_frontier_tail_completion",
        debug_stats=completion_debug,
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


def _best_work_position_sequence_completion_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
) -> tuple[ReplayState, list[HookAction]] | None:
    before = compute_structural_metrics(plan_input, state)
    if before.work_position_unfinished_count <= 0 and before.target_sequence_defect_count <= 0:
        return None
    candidates = generate_work_position_sequence_candidates(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )
    ranked: list[tuple[tuple[int, int, int, int, tuple], ReplayState, list[HookAction]]] = []
    for candidate in candidates:
        next_state = state
        for move in candidate.steps:
            try:
                next_state = _apply_move(
                    state=next_state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                next_state = None
                break
        if next_state is None:
            continue
        after = compute_structural_metrics(plan_input, next_state)
        if (
            after.target_sequence_defect_count > before.target_sequence_defect_count
            or after.work_position_unfinished_count >= before.work_position_unfinished_count
        ):
            continue
        ranked.append(
            (
                (
                    after.target_sequence_defect_count,
                    after.work_position_unfinished_count,
                    after.unfinished_count,
                    len(candidate.steps),
                    tuple(
                        (
                            move.action_type,
                            move.source_track,
                            move.target_track,
                            tuple(move.vehicle_nos),
                        )
                        for move in candidate.steps
                    ),
                ),
                next_state,
                list(candidate.steps),
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1], ranked[0][2]


def _try_route_clean_direct_prefix_tail_completion_from_state(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    time_budget_ms: float,
) -> tuple[ReplayState, list[HookAction]] | None:
    if state.loco_carry or time_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
        return None
    started_at = perf_counter()
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    current_state = state
    direct_plan: list[HookAction] = []
    seen_keys: set[tuple] = set()
    while _remaining_child_budget_ms(started_at, time_budget_ms) > MIN_CHILD_STAGE_BUDGET_MS:
        state_key = _state_key(current_state, plan_input)
        if state_key in seen_keys:
            break
        seen_keys.add(state_key)
        candidates: list[
            tuple[tuple[int, int, int, int, str, tuple[str, ...]], ReplayState, list[HookAction]]
        ] = []
        attach_moves = generate_real_hook_moves(
            plan_input,
            current_state,
            master=master,
            route_oracle=route_oracle,
        )
        for source_track, prefix_block, target_track in _route_clean_direct_prefix_blocks(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        ):
            attach_move = _find_generated_move(
                attach_moves,
                action_type="ATTACH",
                source_track=source_track,
                vehicle_nos=prefix_block,
            )
            if attach_move is None:
                attach_move = _build_goal_frontier_exact_attach(
                    plan_input=plan_input,
                    state=current_state,
                    master=master,
                    source_track=source_track,
                    prefix_block=prefix_block,
                )
            if attach_move is None:
                continue
            try:
                attached_state = _apply_move(
                    state=current_state,
                    move=attach_move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                continue
            detach_moves = generate_real_hook_moves(
                plan_input,
                attached_state,
                master=master,
                route_oracle=route_oracle,
            )
            candidate_target_tracks = _route_clean_common_goal_targets(
                prefix_block,
                state=attached_state,
                plan_input=plan_input,
                route_oracle=route_oracle,
                detach_moves=detach_moves,
                vehicle_by_no=vehicle_by_no,
            )
            if target_track is not None:
                candidate_target_tracks = [
                    target for target in candidate_target_tracks if target == target_track
                ]
            detach_move = _find_generated_move(
                detach_moves,
                action_type="DETACH",
                target_tracks=set(candidate_target_tracks),
                vehicle_nos=prefix_block,
            )
            if detach_move is None and target_track in _work_position_target_tracks(plan_input):
                detach_move = _build_work_position_exact_detach(
                    plan_input=plan_input,
                    state=attached_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    vehicle_nos=prefix_block,
                    target_track=target_track,
                )
            if detach_move is None:
                if target_track is None:
                    continue
                split_step = _route_clean_source_remainder_split_step(
                    plan_input=plan_input,
                    state=current_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    prefix_block=prefix_block,
                    target_track=target_track,
                )
                if split_step is None:
                    continue
                next_state, step_plan = split_step
                if next_state.loco_carry or _state_key(next_state, plan_input) == state_key:
                    continue
                if not _route_clean_direct_prefix_improves(
                    plan_input=plan_input,
                    before=current_state,
                    after=next_state,
                    moved_vehicle_nos=prefix_block,
                    target_track=target_track,
                    vehicle_by_no=vehicle_by_no,
                ):
                    continue
                next_structural = compute_structural_metrics(plan_input, next_state)
                try:
                    route_pressure = compute_route_blockage_plan(
                        plan_input,
                        next_state,
                        route_oracle,
                    ).total_blockage_pressure
                except Exception:  # noqa: BLE001
                    route_pressure = 10**9
                candidates.append(
                    (
                        (
                            next_structural.unfinished_count,
                            next_structural.work_position_unfinished_count,
                            route_pressure,
                            -len(prefix_block),
                            target_track or "",
                            tuple(prefix_block),
                        ),
                        next_state,
                        step_plan,
                    )
                )
                continue
            try:
                next_state = _apply_move(
                    state=attached_state,
                    move=detach_move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                continue
            if next_state.loco_carry or _state_key(next_state, plan_input) == state_key:
                continue
            if not _route_clean_direct_prefix_improves(
                plan_input=plan_input,
                before=current_state,
                after=next_state,
                moved_vehicle_nos=prefix_block,
                target_track=detach_move.target_track,
                vehicle_by_no=vehicle_by_no,
            ):
                continue
            next_structural = compute_structural_metrics(plan_input, next_state)
            try:
                route_pressure = compute_route_blockage_plan(
                    plan_input,
                    next_state,
                    route_oracle,
                ).total_blockage_pressure
            except Exception:  # noqa: BLE001
                route_pressure = 10**9
            candidates.append(
                (
                    (
                        next_structural.unfinished_count,
                        next_structural.work_position_unfinished_count,
                        route_pressure,
                        -len(prefix_block),
                        detach_move.target_track,
                        tuple(prefix_block),
                    ),
                    next_state,
                    [attach_move, detach_move],
                )
            )
        for (
            source_track,
            preserved_prefix,
            target_block,
            target_track,
        ) in _route_clean_buried_target_blocks(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        ):
            split_step = _route_clean_buried_target_block_step(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                preserved_prefix=preserved_prefix,
                target_block=target_block,
                target_track=target_track,
            )
            if split_step is None:
                continue
            next_state, step_plan = split_step
            if next_state.loco_carry or _state_key(next_state, plan_input) == state_key:
                continue
            if not _route_clean_direct_prefix_improves(
                plan_input=plan_input,
                before=current_state,
                after=next_state,
                moved_vehicle_nos=target_block,
                target_track=target_track,
                vehicle_by_no=vehicle_by_no,
            ):
                continue
            next_structural = compute_structural_metrics(plan_input, next_state)
            try:
                route_pressure = compute_route_blockage_plan(
                    plan_input,
                    next_state,
                    route_oracle,
                ).total_blockage_pressure
            except Exception:  # noqa: BLE001
                route_pressure = 10**9
            candidates.append(
                (
                    (
                        next_structural.unfinished_count,
                        next_structural.work_position_unfinished_count,
                        route_pressure,
                        -len(target_block),
                        target_track,
                        tuple(target_block),
                    ),
                    next_state,
                    step_plan,
                )
            )
        if not candidates:
            break
        _score, next_state, step_plan = min(candidates, key=lambda item: item[0])
        current_state = next_state
        direct_plan.extend(step_plan)
        if _is_goal(plan_input, current_state):
            break
    if not direct_plan:
        return None
    return current_state, direct_plan


def _route_clean_source_remainder_split_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    source_track: str,
    prefix_block: list[str],
    target_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(prefix_block)] != list(prefix_block):
        return None
    remainder = source_seq[len(prefix_block):]
    if not remainder:
        return None
    full_block = list(prefix_block) + remainder
    attach_full = _build_goal_frontier_exact_attach(
        plan_input=plan_input,
        state=state,
        master=master,
        source_track=source_track,
        prefix_block=full_block,
    )
    if attach_full is None:
        return None
    try:
        full_carry_state = _apply_move(
            state=state,
            move=attach_full,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    remainder_staging_options = _route_clean_remainder_staging_options(
        plan_input=plan_input,
        state=full_carry_state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        remainder=remainder,
        source_track=source_track,
        target_track=target_track,
    )
    candidates: list[tuple[tuple[int, int, int, str], ReplayState, list[HookAction]]] = []
    for stage_remainder in remainder_staging_options:
        try:
            staged_state = _apply_move(
                state=full_carry_state,
                move=stage_remainder,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        detach_prefix = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=staged_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=prefix_block,
            target_track=target_track,
        )
        if detach_prefix is None:
            continue
        try:
            placed_state = _apply_move(
                state=staged_state,
                move=detach_prefix,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        restore_remainder = _build_goal_frontier_exact_attach(
            plan_input=plan_input,
            state=placed_state,
            master=master,
            source_track=stage_remainder.target_track,
            prefix_block=remainder,
        )
        if restore_remainder is None:
            continue
        try:
            carried_remainder_state = _apply_move(
                state=placed_state,
                move=restore_remainder,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        restore_source = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=carried_remainder_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=remainder,
            target_track=source_track,
        )
        if restore_source is None:
            continue
        try:
            next_state = _apply_move(
                state=carried_remainder_state,
                move=restore_source,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if next_state.loco_carry:
            continue
        metrics = compute_structural_metrics(plan_input, next_state)
        candidates.append(
            (
                (
                    metrics.unfinished_count,
                    metrics.work_position_unfinished_count,
                    len(stage_remainder.path_tracks),
                    stage_remainder.target_track,
                ),
                next_state,
                [
                    attach_full,
                    stage_remainder,
                    detach_prefix,
                    restore_remainder,
                    restore_source,
                ],
            )
        )
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return None
    return candidates[0][1], candidates[0][2]


def _route_clean_buried_target_block_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    source_track: str,
    preserved_prefix: list[str],
    target_block: list[str],
    target_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    if not preserved_prefix or not target_block:
        return None
    attach_block = [*preserved_prefix, *target_block]
    attach_full = _build_goal_frontier_exact_attach(
        plan_input=plan_input,
        state=state,
        master=master,
        source_track=source_track,
        prefix_block=attach_block,
    )
    if attach_full is None:
        return None
    try:
        carried_state = _apply_move(
            state=state,
            move=attach_full,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None

    detach_target = _find_generated_move(
        _generate_tail_moves(
            plan_input=plan_input,
            state=carried_state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="DETACH",
        vehicle_nos=target_block,
        target_track=target_track,
    )
    if detach_target is None:
        detach_target = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=carried_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=target_block,
            target_track=target_track,
        )
    if detach_target is None:
        return None
    try:
        target_placed_state = _apply_move(
            state=carried_state,
            move=detach_target,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None

    restore_prefix = _find_generated_move(
        _generate_tail_moves(
            plan_input=plan_input,
            state=target_placed_state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="DETACH",
        vehicle_nos=preserved_prefix,
        target_track=source_track,
    )
    if restore_prefix is None:
        restore_prefix = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=target_placed_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=preserved_prefix,
            target_track=source_track,
        )
    if restore_prefix is None:
        return None
    try:
        next_state = _apply_move(
            state=target_placed_state,
            move=restore_prefix,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    if next_state.loco_carry:
        return None
    if not _vehicles_satisfied_on_track(
        preserved_prefix,
        track_name=source_track,
        state=next_state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    return next_state, [attach_full, detach_target, restore_prefix]


def _generate_tail_moves(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
) -> list[HookAction]:
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    return generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )


def _route_clean_remainder_staging_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    remainder: list[str],
    source_track: str,
    target_track: str,
) -> list[HookAction]:
    _ = target_track
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length
        for vehicle in plan_input.vehicles
    }
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in remainder)
    candidates: list[tuple[tuple[int, int, str], HookAction]] = []
    for staging_track in sorted(STAGING_TRACKS):
        if staging_track == source_track:
            continue
        track = master.tracks.get(staging_track)
        if track is None or not track.allow_parking:
            continue
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(staging_track, [])
        )
        if current_length + block_length > track.effective_length_m + 1e-9:
            continue
        detach = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=remainder,
            target_track=staging_track,
        )
        if detach is None:
            continue
        candidates.append(
            (
                (
                    len(state.track_sequences.get(staging_track, [])),
                    len(detach.path_tracks),
                    staging_track,
                ),
                detach,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [move for _score, move in candidates[:4]]


def _route_clean_direct_prefix_blocks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
) -> list[tuple[str, list[str], str | None]]:
    from fzed_shunting.solver.goal_logic import (
        goal_effective_allowed_tracks,
        goal_is_satisfied,
    )

    blocks: list[tuple[str, list[str], str | None]] = []
    for source_track, seq in state.track_sequences.items():
        prefix: list[str] = []
        shared_targets: set[str] | None = None
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                break
            if goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            targets = [
                target
                for target in goal_effective_allowed_tracks(
                    vehicle,
                    state=state,
                    plan_input=plan_input,
                    route_oracle=route_oracle,
                )
                if target != source_track
            ]
            if not targets:
                break
            next_shared_targets = set(targets)
            if shared_targets is None:
                shared_targets = next_shared_targets
            else:
                intersected_targets = shared_targets & next_shared_targets
                if not intersected_targets:
                    break
                shared_targets = intersected_targets
            prefix.append(vehicle_no)
        if prefix and shared_targets:
            target_track = next(iter(shared_targets)) if len(shared_targets) == 1 else None
            blocks.append((source_track, prefix, target_track))
    blocks.sort(key=lambda item: (-len(item[1]), item[0], item[2] or "", tuple(item[1])))
    return blocks


def _route_clean_common_goal_targets(
    vehicle_nos: list[str],
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    detach_moves: list[HookAction],
    vehicle_by_no: dict[str, Any],
) -> list[str]:
    from fzed_shunting.solver.goal_logic import (
        goal_effective_allowed_tracks,
        goal_is_satisfied,
        goal_track_preference_level,
    )

    if not vehicle_nos:
        return []
    shared_targets: set[str] | None = None
    for vehicle_no in vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        targets = set(
            goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
        )
        shared_targets = targets if shared_targets is None else shared_targets & targets
    if not shared_targets:
        return []

    candidates: list[tuple[tuple[int, float, float, str], str]] = []
    for target_track in shared_targets:
        move = _find_generated_move(
            detach_moves,
            action_type="DETACH",
            vehicle_nos=vehicle_nos,
            target_track=target_track,
        )
        if move is None:
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
        if not all(
            (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and goal_is_satisfied(
                vehicle,
                track_name=target_track,
                state=next_state,
                plan_input=plan_input,
            )
            for vehicle_no in vehicle_nos
        ):
            continue
        preference_levels: list[int] = []
        for vehicle_no in vehicle_nos:
            level = goal_track_preference_level(
                vehicle_by_no[vehicle_no],
                target_track,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
            preference_levels.append(level if level is not None else 10**9)
        preference = min(preference_levels)
        target_length = sum(
            vehicle_by_no[vehicle_no].vehicle_length
            for vehicle_no in state.track_sequences.get(target_track, [])
            if vehicle_no in vehicle_by_no
        )
        candidates.append(
            (
                (
                    preference,
                    target_length,
                    float(len(move.path_tracks)),
                    target_track,
                ),
                target_track,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [target_track for _score, target_track in candidates]


def _route_clean_buried_target_blocks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
) -> list[tuple[str, list[str], list[str], str]]:
    from fzed_shunting.solver.goal_logic import (
        goal_effective_allowed_tracks,
        goal_is_satisfied,
    )

    blocks: list[tuple[str, list[str], list[str], str]] = []
    for source_track, seq in state.track_sequences.items():
        preserved_prefix: list[str] = []
        scan_index = 0
        while scan_index < len(seq):
            vehicle = vehicle_by_no.get(seq[scan_index])
            if vehicle is None or not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            preserved_prefix.append(seq[scan_index])
            scan_index += 1
        if not preserved_prefix or scan_index >= len(seq):
            continue

        target_track: str | None = None
        target_block: list[str] = []
        for vehicle_no in seq[scan_index:]:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                break
            if goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            ):
                break
            targets = [
                target
                for target in goal_effective_allowed_tracks(
                    vehicle,
                    state=state,
                    plan_input=plan_input,
                    route_oracle=route_oracle,
                )
                if target != source_track
            ]
            if len(targets) != 1:
                break
            if target_track is None:
                target_track = targets[0]
            if targets[0] != target_track:
                break
            target_block.append(vehicle_no)

        if target_block and target_track is not None:
            blocks.append((source_track, preserved_prefix, target_block, target_track))

    blocks.sort(
        key=lambda item: (
            -len(item[2]),
            len(item[1]),
            item[0],
            item[3],
            tuple(item[2]),
        )
    )
    return blocks


def _route_clean_direct_prefix_improves(
    *,
    plan_input: NormalizedPlanInput,
    before: ReplayState,
    after: ReplayState,
    moved_vehicle_nos: list[str],
    target_track: str,
    vehicle_by_no: dict[str, Any],
) -> bool:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    if any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is None
        or not goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=after,
            plan_input=plan_input,
        )
        for vehicle_no in moved_vehicle_nos
    ):
        return False
    before_metrics = compute_structural_metrics(plan_input, before)
    after_metrics = compute_structural_metrics(plan_input, after)
    return (
        after_metrics.unfinished_count < before_metrics.unfinished_count
        or after_metrics.work_position_unfinished_count
        < before_metrics.work_position_unfinished_count
        or after_metrics.front_blocker_count < before_metrics.front_blocker_count
        or after_metrics.goal_track_blocker_count < before_metrics.goal_track_blocker_count
    )


def _vehicles_satisfied_on_track(
    vehicle_nos: list[str],
    *,
    track_name: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, Any],
) -> bool:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    track_seq = set(state.track_sequences.get(track_name, []))
    return all(
        vehicle_no in track_seq
        and (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in vehicle_nos
    )


def _work_position_target_tracks(plan_input: NormalizedPlanInput) -> set[str]:
    return {
        target
        for vehicle in plan_input.vehicles
        if vehicle.goal.work_position_kind is not None
        for target in vehicle.goal.allowed_target_tracks
    }


def _build_work_position_tail_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    vehicle: Any,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    if not vehicle.goal.allowed_target_tracks:
        return None
    target_track = vehicle.goal.allowed_target_tracks[0]
    source_track = _vehicle_track_lookup(state).get(vehicle.vehicle_no)
    if source_track is None:
        return None
    if vehicle.goal.work_position_kind == "SPOTTING":
        sequence_step = _best_work_position_sequence_completion_step(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        )
        if sequence_step is not None:
            return sequence_step
        return _build_work_position_spotting_insert_step(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle=vehicle,
            source_track=source_track,
            target_track=target_track,
        )
    if vehicle.goal.work_position_kind == "EXACT_NORTH_RANK":
        exact_rank_step = _build_work_position_exact_rank_reorder_step(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle=vehicle,
            source_track=source_track,
            target_track=target_track,
        )
        if exact_rank_step is not None:
            return exact_rank_step
    attach_target = _find_goal_frontier_attach_move(
        generate_real_hook_moves(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
        ),
        source_track=source_track,
        target_vehicle_no=vehicle.vehicle_no,
    )
    if attach_target is None:
        return None
    next_state = _apply_move(
        state=state,
        move=attach_target,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    target_detach_moves = generate_real_hook_moves(
        plan_input,
        next_state,
        master=master,
        route_oracle=route_oracle,
    )
    detach_target = _find_work_position_target_detach_move(
        moves=target_detach_moves,
        plan_input=plan_input,
        state=next_state,
        vehicle_by_no=vehicle_by_no,
        vehicle=vehicle,
        target_track=target_track,
    )
    if detach_target is None:
        detach_target = _find_generated_move(
            target_detach_moves,
            action_type="DETACH",
            vehicle_nos=[vehicle.vehicle_no],
            target_track=target_track,
        )
    if detach_target is None:
        temporary_drop = _build_work_position_temporary_drop_completion(
            plan_input=plan_input,
            state=state,
            attached_state=next_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle=vehicle,
            source_track=source_track,
            target_track=target_track,
            attach_target=attach_target,
        )
        if temporary_drop is not None:
            return temporary_drop
    if detach_target is None:
        return None
    next_state = _apply_move(
        state=next_state,
        move=detach_target,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    step_plan = [attach_target, detach_target]
    if goal_is_satisfied(
        vehicle,
        track_name=target_track,
        state=next_state,
        plan_input=plan_input,
    ) and not next_state.loco_carry:
        return next_state, step_plan
    if next_state.loco_carry:
        restored = _finish_goal_frontier_deep_block_carry(
            plan_input=plan_input,
            state=next_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
        )
        if restored is not None:
            restored_state, restore_plan = restored
            restored_step_plan = [*step_plan, *restore_plan]
            if goal_is_satisfied(
                vehicle,
                track_name=target_track,
                state=restored_state,
                plan_input=plan_input,
            ):
                return restored_state, restored_step_plan
    padding_plan = _build_work_position_padding_plan(
        plan_input=plan_input,
        state=next_state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        target_track=target_track,
        target_vehicle=vehicle,
    )
    if padding_plan is None:
        return None
    next_state, padding_moves = padding_plan
    step_plan.extend(padding_moves)
    if goal_is_satisfied(
        vehicle,
        track_name=target_track,
        state=next_state,
        plan_input=plan_input,
    ):
        return next_state, step_plan
    return None


def _build_work_position_temporary_drop_completion(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    attached_state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    vehicle: Any,
    source_track: str,
    target_track: str,
    attach_target: HookAction,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    candidate_staging_moves = [
        move
        for move in generate_real_hook_moves(
            plan_input,
            attached_state,
            master=master,
            route_oracle=route_oracle,
        )
        if move.action_type == "DETACH"
        and vehicle.vehicle_no in move.vehicle_nos
        and move.target_track not in {source_track, target_track}
    ]
    if not candidate_staging_moves:
        return None

    def staging_score(move: HookAction) -> tuple[int, int, int, int, str, tuple[str, ...]]:
        try:
            staged_state = _apply_move(
                state=attached_state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            route_pressure = compute_route_blockage_plan(
                plan_input,
                staged_state,
                route_oracle,
            ).total_blockage_pressure
        except Exception:  # noqa: BLE001
            route_pressure = 10**9
        return (
            route_pressure,
            0 if move.target_track in STAGING_TRACKS else 1,
            len(move.path_tracks),
            len(move.vehicle_nos),
            move.target_track,
            tuple(move.vehicle_nos),
        )

    candidate_staging_moves.sort(key=staging_score)
    for staging_move in candidate_staging_moves:
        try:
            staged_state = _apply_move(
                state=attached_state,
                move=staging_move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if _state_key(staged_state, plan_input) == _state_key(attached_state, plan_input):
            continue
        step_plan = [attach_target, staging_move]
        restored_state = staged_state
        if staged_state.loco_carry:
            restored = _finish_goal_frontier_deep_block_carry(
                plan_input=plan_input,
                state=staged_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
            )
            if restored is None:
                continue
            restored_state, restore_plan = restored
            step_plan.extend(restore_plan)
        if restored_state.loco_carry:
            continue
        staging_attach = _find_goal_frontier_attach_move(
            generate_real_hook_moves(
                plan_input,
                restored_state,
                master=master,
                route_oracle=route_oracle,
            ),
            source_track=staging_move.target_track,
            target_vehicle_no=vehicle.vehicle_no,
        )
        if staging_attach is None or list(staging_attach.vehicle_nos) != list(staging_move.vehicle_nos):
            continue
        try:
            staged_attach_state = _apply_move(
                state=restored_state,
                move=staging_attach,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if _state_key(staged_attach_state, plan_input) == _state_key(restored_state, plan_input):
            continue
        followup_moves = generate_real_hook_moves(
            plan_input,
            staged_attach_state,
            master=master,
            route_oracle=route_oracle,
        )
        final_detach = _find_work_position_target_detach_move(
            moves=followup_moves,
            plan_input=plan_input,
            state=staged_attach_state,
            vehicle_by_no=vehicle_by_no,
            vehicle=vehicle,
            target_track=target_track,
        )
        if final_detach is None:
            final_detach = _find_generated_move(
                followup_moves,
                action_type="DETACH",
                vehicle_nos=list(staging_attach.vehicle_nos),
                target_track=target_track,
            )
        if final_detach is None:
            continue
        try:
            final_state = _apply_move(
                state=staged_attach_state,
                move=final_detach,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if _state_key(final_state, plan_input) == _state_key(staged_attach_state, plan_input):
            continue
        step_plan.extend([staging_attach, final_detach])
        if goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=final_state,
            plan_input=plan_input,
        ) and not final_state.loco_carry:
            return final_state, step_plan
    return None


def _find_work_position_target_detach_move(
    *,
    moves: list[HookAction],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, Any],
    vehicle: Any,
    target_track: str,
) -> HookAction | None:
    from fzed_shunting.domain.work_positions import north_rank
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    candidates: list[tuple[tuple[int, int, int, tuple[str, ...]], HookAction]] = []
    for move in moves:
        if (
            move.action_type != "DETACH"
            or move.target_track != target_track
            or vehicle.vehicle_no not in move.vehicle_nos
        ):
            continue
        if any(
            not _can_use_vehicle_as_work_position_padding(
                vehicle=moved_vehicle,
                target_track=target_track,
                current_track=state.loco_track_name,
                state=state,
                plan_input=plan_input,
            )
            for moved_no in move.vehicle_nos
            if moved_no != vehicle.vehicle_no
            for moved_vehicle in [vehicle_by_no.get(moved_no)]
            if moved_vehicle is not None
        ):
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
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        target_seq = list(next_state.track_sequences.get(target_track, []))
        if vehicle.vehicle_no not in target_seq:
            continue
        if vehicle.goal.work_position_kind == "EXACT_NORTH_RANK":
            rank = north_rank(target_seq, vehicle.vehicle_no)
            target_rank = vehicle.goal.target_rank
            if rank is None or target_rank is None or rank > target_rank:
                continue
            rank_gap = target_rank - rank
        else:
            rank_gap = (
                0
                if goal_is_satisfied(
                    vehicle,
                    track_name=target_track,
                    state=next_state,
                    plan_input=plan_input,
                )
                else 1
            )
        candidates.append(
            (
                (
                    rank_gap,
                    len(next_state.loco_carry),
                    -len(move.vehicle_nos),
                    tuple(move.vehicle_nos),
                ),
                move,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _build_work_position_exact_rank_reorder_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    vehicle: Any,
    source_track: str,
    target_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    if source_track != target_track:
        return None
    target_rank = vehicle.goal.target_rank
    if target_rank is None or target_rank <= 1:
        return None
    target_seq = list(state.track_sequences.get(target_track, []))
    try:
        target_index = target_seq.index(vehicle.vehicle_no)
    except ValueError:
        return None
    current_rank = target_index + 1
    if current_rank >= target_rank:
        return None
    needed_prefix = target_rank - current_rank
    south_block = target_seq[target_index + 1 : target_index + 1 + needed_prefix]
    if len(south_block) < needed_prefix:
        return None
    if any(
        (pad_vehicle := vehicle_by_no.get(vehicle_no)) is None
        or target_track not in pad_vehicle.goal.allowed_target_tracks
        or pad_vehicle.goal.work_position_kind == "EXACT_NORTH_RANK"
        for vehicle_no in south_block
    ):
        return None

    attach_block = target_seq[: target_index + 1 + needed_prefix]
    attach_clear = _build_goal_frontier_exact_attach(
        plan_input=plan_input,
        state=state,
        master=master,
        source_track=target_track,
        prefix_block=attach_block,
    )
    if attach_clear is None:
        return None
    after_attach = _apply_move(
        state=state,
        move=attach_clear,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    target_block = [vehicle.vehicle_no, *south_block]
    detach_staging = _work_position_staging_detach_candidates(
        plan_input=plan_input,
        state=after_attach,
        master=master,
        route_oracle=route_oracle,
        block=south_block,
        source_track=target_track,
    )
    if not detach_staging:
        return None
    staged_state = _apply_move(
        state=after_attach,
        move=detach_staging[0],
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    step_plan = [attach_clear, detach_staging[0]]

    detach_target = _find_generated_move(
        generate_real_hook_moves(
            plan_input,
            staged_state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="DETACH",
        target_track=target_track,
        vehicle_nos=[vehicle.vehicle_no],
    )
    if detach_target is None:
        detach_target = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=staged_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=[vehicle.vehicle_no],
            target_track=target_track,
        )
    if detach_target is None:
        return None
    next_state = _apply_move(
        state=staged_state,
        move=detach_target,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    step_plan.append(detach_target)

    attach_padding = _find_generated_move(
        generate_real_hook_moves(
            plan_input,
            next_state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="ATTACH",
        source_track=detach_staging[0].target_track,
        vehicle_nos=south_block,
    )
    if attach_padding is None:
        return None
    after_attach_padding = _apply_move(
        state=next_state,
        move=attach_padding,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    detach_padding = _find_generated_move(
        generate_real_hook_moves(
            plan_input,
            after_attach_padding,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="DETACH",
        target_track=target_track,
        vehicle_nos=south_block,
    )
    if detach_padding is None:
        detach_padding = _build_work_position_exact_detach(
            plan_input=plan_input,
            state=after_attach_padding,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=south_block,
            target_track=target_track,
        )
    if detach_padding is None:
        return None
    next_state = _apply_move(
        state=after_attach_padding,
        move=detach_padding,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    step_plan.extend([attach_padding, detach_padding])

    if next_state.loco_carry:
        restore_block = list(next_state.loco_carry)
        restore_source = _find_generated_move(
            generate_real_hook_moves(
                plan_input,
                next_state,
                master=master,
                route_oracle=route_oracle,
            ),
            action_type="DETACH",
            vehicle_nos=restore_block,
            target_track=target_track,
        )
        if restore_source is None:
            restore_source = _build_work_position_exact_detach(
                plan_input=plan_input,
                state=next_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle_nos=restore_block,
                target_track=target_track,
            )
        if restore_source is None:
            return None
        next_state = _apply_move(
            state=next_state,
            move=restore_source,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        step_plan.append(restore_source)
    if not goal_is_satisfied(
        vehicle,
        track_name=target_track,
        state=next_state,
        plan_input=plan_input,
    ):
        return None
    return next_state, step_plan


def _build_work_position_spotting_insert_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    vehicle: Any,
    source_track: str,
    target_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.domain.work_positions import (
        preview_work_positions_after_prepend,
    )
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    if source_track == target_track:
        return None
    target_seq = list(state.track_sequences.get(target_track, []))
    candidate_clear_counts: list[int] = []
    for clear_count in range(len(target_seq) + 1):
        clear_block = target_seq[:clear_count]
        final_preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=[*clear_block, vehicle.vehicle_no],
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        evaluation = final_preview.evaluations.get(vehicle.vehicle_no)
        if final_preview.valid and evaluation is not None and evaluation.satisfied_now:
            candidate_clear_counts.append(clear_count)
    for clear_count in sorted(candidate_clear_counts):
        next_state = state
        step_plan: list[HookAction] = []
        clear_block = target_seq[:clear_count]
        attach_clear: HookAction | None = None
        staged_clear_options: list[tuple[HookAction, ReplayState, HookAction, ReplayState]] = []
        if not clear_block:
            staged_clear_options.append(
                (
                    HookAction(
                        source_track=target_track,
                        target_track=target_track,
                        vehicle_nos=[],
                        path_tracks=[],
                        action_type="ATTACH",
                    ),
                    next_state,
                    HookAction(
                        source_track=target_track,
                        target_track=target_track,
                        vehicle_nos=[],
                        path_tracks=[],
                        action_type="DETACH",
                    ),
                    next_state,
                )
            )
        else:
            attach_clear = _build_goal_frontier_exact_attach(
                plan_input=plan_input,
                state=next_state,
                master=master,
                source_track=target_track,
                prefix_block=clear_block,
            )
            if attach_clear is None:
                continue
            after_attach_clear = _apply_move(
                state=next_state,
                move=attach_clear,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            for detach_clear in _work_position_staging_detach_candidates(
                plan_input=plan_input,
                state=after_attach_clear,
                master=master,
                route_oracle=route_oracle,
                block=clear_block,
                source_track=target_track,
            ):
                staged_state = _apply_move(
                    state=after_attach_clear,
                    move=detach_clear,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
                staged_clear_options.append(
                    (attach_clear, after_attach_clear, detach_clear, staged_state)
                )
        for attach_clear_move, _after_attach_clear, detach_clear_move, staged_state in staged_clear_options:
            next_state = staged_state
            step_plan = []
            if clear_block:
                step_plan.extend([attach_clear_move, detach_clear_move])

            source_seq = list(next_state.track_sequences.get(source_track, []))
            if vehicle.vehicle_no not in source_seq:
                continue
            source_prefix = source_seq[: source_seq.index(vehicle.vehicle_no) + 1]
            attach_target = _find_goal_frontier_attach_move(
                generate_real_hook_moves(
                    plan_input,
                    next_state,
                    master=master,
                    route_oracle=route_oracle,
                ),
                source_track=source_track,
                target_vehicle_no=vehicle.vehicle_no,
            )
            if attach_target is None or list(attach_target.vehicle_nos) != source_prefix:
                attach_target = _build_goal_frontier_exact_attach(
                    plan_input=plan_input,
                    state=next_state,
                    master=master,
                    source_track=source_track,
                    prefix_block=source_prefix,
                )
            if attach_target is None:
                continue
            after_attach_target = _apply_move(
                state=next_state,
                move=attach_target,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            detach_target = _find_generated_move(
                generate_real_hook_moves(
                    plan_input,
                    after_attach_target,
                    master=master,
                    route_oracle=route_oracle,
                ),
                action_type="DETACH",
                vehicle_nos=[vehicle.vehicle_no],
                target_track=target_track,
            )
            if detach_target is None:
                continue
            next_state = _apply_move(
                state=after_attach_target,
                move=detach_target,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            step_plan.extend([attach_target, detach_target])

            if next_state.loco_carry:
                restore_source = _find_generated_move(
                    generate_real_hook_moves(
                        plan_input,
                        next_state,
                        master=master,
                        route_oracle=route_oracle,
                    ),
                    action_type="DETACH",
                    vehicle_nos=list(next_state.loco_carry),
                    target_track=source_track,
                )
                if restore_source is None:
                    continue
                next_state = _apply_move(
                    state=next_state,
                    move=restore_source,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
                step_plan.append(restore_source)

            if clear_block:
                attach_restore = _find_generated_move(
                    generate_real_hook_moves(
                        plan_input,
                        next_state,
                        master=master,
                        route_oracle=route_oracle,
                    ),
                    action_type="ATTACH",
                    source_track=detach_clear_move.target_track,
                    vehicle_nos=clear_block,
                )
                if attach_restore is None:
                    continue
                after_attach_restore = _apply_move(
                    state=next_state,
                    move=attach_restore,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
                detach_restore = _find_generated_move(
                    generate_real_hook_moves(
                        plan_input,
                        after_attach_restore,
                        master=master,
                        route_oracle=route_oracle,
                    ),
                    action_type="DETACH",
                    vehicle_nos=clear_block,
                    target_track=target_track,
                )
                if detach_restore is None:
                    detach_restore = _build_work_position_exact_detach(
                        plan_input=plan_input,
                        state=after_attach_restore,
                        master=master,
                        route_oracle=route_oracle,
                        vehicle_by_no=vehicle_by_no,
                        vehicle_nos=clear_block,
                        target_track=target_track,
                    )
                if detach_restore is None:
                    continue
                next_state = _apply_move(
                    state=after_attach_restore,
                    move=detach_restore,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
                step_plan.extend([attach_restore, detach_restore])

            if goal_is_satisfied(
                vehicle,
                track_name=target_track,
                state=next_state,
                plan_input=plan_input,
            ):
                return next_state, step_plan
    return None


def _build_work_position_exact_detach(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    vehicle_nos: list[str],
    target_track: str,
) -> HookAction | None:
    if not vehicle_nos:
        return None
    from fzed_shunting.domain.carry_order import is_carried_tail_block
    from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
    from fzed_shunting.domain.work_positions import preview_work_positions_after_prepend

    if not is_carried_tail_block(state.loco_carry, vehicle_nos):
        return None
    vehicles = [vehicle_by_no.get(vehicle_no) for vehicle_no in vehicle_nos]
    if any(vehicle is None for vehicle in vehicles):
        return None
    if validate_hook_vehicle_group([vehicle for vehicle in vehicles if vehicle is not None]):
        return None
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=list(vehicle_nos),
        existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return None
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length
        for vehicle in plan_input.vehicles
    }
    source_track = state.loco_track_name
    source_node = (
        state.loco_node
        if _remaining_source_vehicle_count_after_detach(
            state=state,
            source_track=source_track,
            vehicle_nos=vehicle_nos,
        )
        > 0
        else None
    )
    target_node = route_oracle.order_end_node(target_track)
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=target_node,
    )
    if path_tracks is None:
        return None
    route = route_oracle.resolve_route_for_path_tracks(
        path_tracks,
        source_node=source_node,
        target_node=target_node,
    )
    if route is None:
        return None
    route_result = route_oracle.validate_path(
        source_track=source_track,
        target_track=target_track,
        path_tracks=path_tracks,
        train_length_m=sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in state.loco_carry),
        occupied_track_sequences=state.track_sequences,
        expected_path_tracks=path_tracks,
        route=route,
        source_node=source_node,
        target_node=target_node,
    )
    if not route_result.is_valid:
        return None
    return HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=path_tracks,
        action_type="DETACH",
    )


def _remaining_source_vehicle_count_after_detach(
    *,
    state: ReplayState,
    source_track: str,
    vehicle_nos: list[str] | tuple[str, ...],
) -> int:
    source_seq = list(state.track_sequences.get(source_track, []))
    vehicle_list = list(vehicle_nos)
    if vehicle_list and source_seq[: len(vehicle_list)] == vehicle_list:
        return max(0, len(source_seq) - len(vehicle_list))
    return len(source_seq)


def _work_position_staging_detach_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    block: list[str],
    source_track: str,
) -> HookAction | None:
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    candidates = [
        move
        for move in generate_real_hook_moves(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
        )
        if move.action_type == "DETACH"
        and list(move.vehicle_nos) == list(block)
        and move.target_track != source_track
    ]
    candidates.sort(
        key=lambda move: (
            0 if move.target_track in STAGING_TRACKS else 1,
            len(state.track_sequences.get(move.target_track, [])),
            len(move.path_tracks),
            move.target_track,
        )
    )
    return candidates


def _unfinished_work_position_vehicles(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, Any],
) -> list[Any]:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    track_by_vehicle = _vehicle_track_lookup(state)
    unfinished = []
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind is None:
            continue
        track = track_by_vehicle.get(vehicle.vehicle_no)
        if track is None or not goal_is_satisfied(
            vehicle,
            track_name=track,
            state=state,
            plan_input=plan_input,
        ):
            unfinished.append(vehicle)
    return unfinished


def _build_work_position_padding_move_pair(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    target_track: str,
    target_vehicle_no: str,
) -> tuple[HookAction, HookAction] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    track_by_vehicle = _vehicle_track_lookup(state)
    candidate_vehicle_nos: list[str] = []
    for vehicle in plan_input.vehicles:
        if vehicle.vehicle_no == target_vehicle_no:
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is None or current_track == target_track:
            continue
        if not _can_use_vehicle_as_work_position_padding(
            vehicle=vehicle,
            target_track=target_track,
            current_track=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        candidate_vehicle_nos.append(vehicle.vehicle_no)
    for vehicle_no in sorted(candidate_vehicle_nos):
        source_track = track_by_vehicle.get(vehicle_no)
        if source_track is None:
            continue
        attach_padding = _find_goal_frontier_attach_move(
            generate_real_hook_moves(
                plan_input,
                state,
                master=master,
                route_oracle=route_oracle,
            ),
            source_track=source_track,
            target_vehicle_no=vehicle_no,
        )
        if attach_padding is None or list(attach_padding.vehicle_nos) != [vehicle_no]:
            continue
        next_state = _apply_move(
            state=state,
            move=attach_padding,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        detach_padding = _find_generated_move(
            generate_real_hook_moves(
                plan_input,
                next_state,
                master=master,
                route_oracle=route_oracle,
            ),
            action_type="DETACH",
            vehicle_nos=[vehicle_no],
            target_track=target_track,
        )
        if detach_padding is not None:
            return attach_padding, detach_padding
    return None


def _build_work_position_padding_plan(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    target_track: str,
    target_vehicle: Any,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    target_rank = target_vehicle.goal.target_rank
    max_padding_iterations = (target_rank - 1) if target_rank is not None else 4
    current_state = state
    moves: list[HookAction] = []
    for _ in range(max(0, max_padding_iterations)):
        if goal_is_satisfied(
            target_vehicle,
            track_name=target_track,
            state=current_state,
            plan_input=plan_input,
        ):
            return current_state, moves
        padding_pair = _build_work_position_padding_move_pair(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=target_track,
            target_vehicle_no=target_vehicle.vehicle_no,
        )
        if padding_pair is None:
            return None
        attach_padding, detach_padding = padding_pair
        next_state = _apply_move(
            state=current_state,
            move=attach_padding,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        next_state = _apply_move(
            state=next_state,
            move=detach_padding,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if _state_key(next_state, plan_input) == _state_key(current_state, plan_input):
            return None
        current_state = next_state
        moves.extend([attach_padding, detach_padding])
    if goal_is_satisfied(
        target_vehicle,
        track_name=target_track,
        state=current_state,
        plan_input=plan_input,
    ):
        return current_state, moves
    return None


def _can_use_vehicle_as_work_position_padding(
    *,
    vehicle: Any,
    target_track: str,
    current_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    if target_track not in vehicle.goal.allowed_target_tracks:
        return False
    if goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    ):
        return True
    return (
        vehicle.goal.target_track == target_track
        and vehicle.goal.work_position_kind in {"FREE", "SPOTTING"}
    )


def _build_goal_frontier_budget_partial_result(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    direct_plan: list[HookAction],
    direct_state: ReplayState,
    expanded_nodes: int,
    generated_nodes: int,
    elapsed_ms: float,
    master: MasterData,
) -> SolverResult | None:
    if not direct_plan or direct_state.loco_carry:
        return None
    return _build_goal_frontier_partial_result(
        plan_input=plan_input,
        original_initial_state=original_initial_state,
        prefix_plan=prefix_plan,
        direct_plan=direct_plan,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        elapsed_ms=elapsed_ms,
        master=master,
    )


def _build_goal_frontier_partial_result(
    *,
    plan_input: NormalizedPlanInput,
    original_initial_state: ReplayState,
    prefix_plan: list[HookAction],
    direct_plan: list[HookAction],
    expanded_nodes: int,
    generated_nodes: int,
    elapsed_ms: float,
    master: MasterData,
) -> SolverResult | None:
    partial_plan = [*prefix_plan, *direct_plan]
    if not partial_plan:
        return None
    result = SolverResult(
        plan=[],
        partial_plan=partial_plan,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        closed_nodes=expanded_nodes,
        elapsed_ms=elapsed_ms,
        is_complete=False,
        is_proven_optimal=False,
        fallback_stage=None,
        partial_fallback_stage="goal_frontier_tail_completion",
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


def _restore_goal_frontier_prefix(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    prefix_block: list[str],
    staging_track: str,
    source_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    if state.loco_carry:
        return None
    if state.track_sequences.get(staging_track, [])[: len(prefix_block)] != prefix_block:
        return None
    attach_prefix = _find_generated_move(
        generate_real_hook_moves(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="ATTACH",
        source_track=staging_track,
        vehicle_nos=prefix_block,
    )
    if attach_prefix is None:
        return None
    next_state = _apply_move(
        state=state,
        move=attach_prefix,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    detach_prefix = _find_generated_move(
        generate_real_hook_moves(
            plan_input,
            next_state,
            master=master,
            route_oracle=route_oracle,
        ),
        action_type="DETACH",
        target_track=source_track,
        vehicle_nos=prefix_block,
    )
    if detach_prefix is None:
        return None
    next_state = _apply_move(
        state=next_state,
        move=detach_prefix,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    return next_state, [attach_prefix, detach_prefix]


def _first_goal_frontier_blocker(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, Any],
) -> tuple[str, list[str], str] | None:
    frontiers = _goal_frontier_blockers(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    return frontiers[0] if frontiers else None


def _goal_frontier_blockers(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, Any],
) -> list[tuple[str, list[str], str]]:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    frontiers: list[tuple[str, list[str], str]] = []
    for track, seq in state.track_sequences.items():
        prefix: list[str] = []
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                break
            if goal_is_satisfied(
                vehicle,
                track_name=track,
                state=state,
                plan_input=plan_input,
            ):
                prefix.append(vehicle_no)
                continue
            if prefix:
                frontiers.append((track, prefix, vehicle_no))
            break
    return frontiers


def _best_goal_frontier_staging_detach(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    prefix_block: list[str],
    source_track: str,
    target_vehicle_no: str,
) -> HookAction | None:
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )
    candidates = [
        move
        for move in moves
        if move.action_type == "DETACH"
        and list(move.vehicle_nos) == list(prefix_block)
        and move.target_track != source_track
    ]

    reachable_candidates = [
        move
        for move in candidates
        if _goal_frontier_target_reachable_after_staging(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            staging_move=move,
            source_track=source_track,
            target_vehicle_no=target_vehicle_no,
        )
    ]
    if reachable_candidates:
        candidates = reachable_candidates
    else:
        return None
    def staging_score(move: HookAction) -> tuple[int, int, int, int, str]:
        try:
            staged_state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            route_pressure = compute_route_blockage_plan(
                plan_input,
                staged_state,
                route_oracle,
            ).total_blockage_pressure
        except Exception:  # noqa: BLE001
            route_pressure = 10**9
        return (
            route_pressure,
            0 if move.target_track in STAGING_TRACKS else 1,
            len(state.track_sequences.get(move.target_track, [])),
            len(move.path_tracks),
            move.target_track,
        )

    candidates.sort(key=staging_score)
    return candidates[0] if candidates else None


def _goal_frontier_deep_block_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    source_track: str,
    prefix_block: list[str],
    target_vehicle_no: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    attach_target = _find_goal_frontier_attach_move(
        generate_real_hook_moves(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
        ),
        source_track=source_track,
        target_vehicle_no=target_vehicle_no,
    )
    if attach_target is None or len(attach_target.vehicle_nos) <= len(prefix_block):
        return None
    next_state = _apply_move(
        state=state,
        move=attach_target,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    detach_target = _best_goal_frontier_target_detach(
        plan_input=plan_input,
        state=next_state,
        baseline_state=state,
        master=master,
        route_oracle=route_oracle,
        target_block=list(attach_target.vehicle_nos),
        target_vehicle_no=target_vehicle_no,
    )
    if detach_target is None or target_vehicle_no not in detach_target.vehicle_nos:
        return None
    step_plan = [attach_target, detach_target]
    next_state = _apply_move(
        state=next_state,
        move=detach_target,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    if next_state.loco_carry:
        restored = _finish_goal_frontier_deep_block_carry(
            plan_input=plan_input,
            state=next_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
        )
        if restored is None:
            return None
        next_state, restore_plan = restored
        step_plan.extend(restore_plan)
    return next_state, step_plan


def _finish_goal_frontier_deep_block_carry(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    source_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    next_state = state
    plan: list[HookAction] = []
    seen_keys: set[tuple] = set()
    while next_state.loco_carry:
        state_key = _state_key(next_state, plan_input)
        if state_key in seen_keys:
            return None
        seen_keys.add(state_key)
        candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], HookAction, ReplayState]] = []
        for move in generate_real_hook_moves(
            plan_input,
            next_state,
            master=master,
            route_oracle=route_oracle,
        ):
            if move.action_type != "DETACH":
                continue
            try:
                candidate_state = _apply_move(
                    state=next_state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                continue
            if not all(
                (
                    vehicle := vehicle_by_no.get(vehicle_no)
                ) is not None
                and goal_is_satisfied(
                    vehicle,
                    track_name=move.target_track,
                    state=candidate_state,
                    plan_input=plan_input,
                )
                for vehicle_no in move.vehicle_nos
            ):
                continue
            candidates.append(
                (
                    (
                        0 if move.target_track == source_track else 1,
                        0 if list(move.vehicle_nos) == list(next_state.loco_carry) else 1,
                        len(candidate_state.loco_carry),
                        len(move.path_tracks),
                        move.target_track,
                        tuple(move.vehicle_nos),
                    ),
                    move,
                    candidate_state,
                )
            )
        if not candidates:
            return None
        _score, selected_move, next_state = min(candidates, key=lambda item: item[0])
        plan.append(selected_move)
    return next_state, plan


def _goal_frontier_target_reachable_after_staging(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, Any],
    staging_move: HookAction,
    source_track: str,
    target_vehicle_no: str,
) -> bool:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    _ = master
    try:
        staged_state = _apply_move(
            state=state,
            move=staging_move,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return False
    generated_attach = _find_best_goal_frontier_target_attach(
        generate_real_hook_moves(
            plan_input,
            staged_state,
            master=master,
            route_oracle=route_oracle,
        ),
        plan_input=plan_input,
        state=staged_state,
        master=master,
        route_oracle=route_oracle,
        source_track=source_track,
        target_vehicle_no=target_vehicle_no,
        vehicle_by_no=vehicle_by_no,
    )
    return generated_attach is not None


def _best_goal_frontier_target_detach(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    baseline_state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    target_block: list[str],
    target_vehicle_no: str,
) -> HookAction | None:
    from fzed_shunting.solver.goal_logic import (
        goal_effective_allowed_tracks,
        goal_is_satisfied,
    )
    from fzed_shunting.solver.move_generator import generate_real_hook_moves

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    target_vehicle = vehicle_by_no.get(target_vehicle_no)
    if target_vehicle is None:
        return None
    generated_moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )
    allowed_targets = set(
        goal_effective_allowed_tracks(
            target_vehicle,
            state=state,
            plan_input=plan_input,
        )
    )
    baseline_structural = compute_structural_metrics(plan_input, baseline_state)
    candidates: list[tuple[tuple[int, int, int, int, int, str, tuple[str, ...]], HookAction]] = []
    for move in generated_moves:
        if (
            move.action_type != "DETACH"
            or target_vehicle_no not in move.vehicle_nos
            or list(move.vehicle_nos) != list(target_block[-len(move.vehicle_nos):])
        ):
            continue
        if move.target_track not in allowed_targets:
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
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        if _state_key(next_state, plan_input) == _state_key(baseline_state, plan_input):
            continue
        target_satisfied = goal_is_satisfied(
            target_vehicle,
            track_name=move.target_track,
            state=next_state,
            plan_input=plan_input,
        )
        if (
            not target_satisfied
            and target_vehicle.goal.work_position_kind is None
        ):
            continue
        placed_all_at_goal = 0
        for vehicle_no in move.vehicle_nos:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None or not goal_is_satisfied(
                vehicle,
                track_name=move.target_track,
                state=next_state,
                plan_input=plan_input,
            ):
                placed_all_at_goal = 1
                break
        next_structural = compute_structural_metrics(plan_input, next_state)
        improves_structural = (
            next_structural.unfinished_count < baseline_structural.unfinished_count
            or next_structural.front_blocker_count
            < baseline_structural.front_blocker_count
            or next_structural.goal_track_blocker_count
            < baseline_structural.goal_track_blocker_count
            or next_structural.work_position_unfinished_count
            < baseline_structural.work_position_unfinished_count
        )
        candidates.append(
            (
                (
                    0 if target_satisfied else 1,
                    len(next_state.loco_carry),
                    0 if improves_structural else 1,
                    placed_all_at_goal,
                    -len(move.vehicle_nos),
                    move.target_track,
                    tuple(move.vehicle_nos),
                ),
                move,
            )
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


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
            if _remaining_child_budget_ms(started_at, time_budget_ms) <= MIN_CHILD_STAGE_BUDGET_MS:
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
                    suffix_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
                    if suffix_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                        continue
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
                suffix_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
                if suffix_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
                    continue
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
    remaining_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
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


def _find_goal_frontier_attach_move(
    moves: list[HookAction],
    *,
    source_track: str,
    target_vehicle_no: str,
) -> HookAction | None:
    candidates = [
        move
        for move in moves
        if move.action_type == "ATTACH"
        and move.source_track == source_track
        and target_vehicle_no in move.vehicle_nos
    ]
    candidates.sort(
        key=lambda move: (
            list(move.vehicle_nos).index(target_vehicle_no),
            len(move.vehicle_nos),
            len(move.path_tracks),
            tuple(move.vehicle_nos),
        )
    )
    return candidates[0] if candidates else None


def _find_best_goal_frontier_target_attach(
    moves: list[HookAction],
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    source_track: str,
    target_vehicle_no: str,
    vehicle_by_no: dict[str, Any],
) -> HookAction | None:
    from fzed_shunting.solver.goal_logic import goal_is_satisfied

    target_vehicle = vehicle_by_no.get(target_vehicle_no)
    if target_vehicle is None:
        return None
    candidates: list[tuple[tuple[int, int, int, int, str, tuple[str, ...]], HookAction]] = []
    for move in moves:
        if (
            move.action_type != "ATTACH"
            or move.source_track != source_track
            or target_vehicle_no not in move.vehicle_nos
        ):
            continue
        try:
            attached_state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        detach = _best_goal_frontier_target_detach(
            plan_input=plan_input,
            state=attached_state,
            baseline_state=state,
            master=master,
            route_oracle=route_oracle,
            target_block=list(move.vehicle_nos),
            target_vehicle_no=target_vehicle_no,
        )
        target_satisfied = 1
        all_satisfied = 1
        if detach is not None:
            try:
                detached_state = _apply_move(
                    state=attached_state,
                    move=detach,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except Exception:  # noqa: BLE001
                detached_state = None
            if detached_state is not None and goal_is_satisfied(
                target_vehicle,
                track_name=detach.target_track,
                state=detached_state,
                plan_input=plan_input,
            ):
                target_satisfied = 0
                all_satisfied = 0
                for vehicle_no in detach.vehicle_nos:
                    vehicle = vehicle_by_no.get(vehicle_no)
                    if vehicle is None or not goal_is_satisfied(
                        vehicle,
                        track_name=detach.target_track,
                        state=detached_state,
                        plan_input=plan_input,
                    ):
                        all_satisfied = 1
                        break
        if detach is not None:
            candidates.append(
                (
                    (
                        target_satisfied,
                        all_satisfied,
                        -len(move.vehicle_nos),
                        len(move.path_tracks),
                        move.target_track,
                        tuple(move.vehicle_nos),
                    ),
                    move,
                )
            )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else None


def _build_goal_frontier_exact_attach(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    source_track: str,
    prefix_block: list[str],
) -> HookAction | None:
    if state.loco_carry:
        return None
    if not prefix_block:
        return None
    if state.track_sequences.get(source_track, [])[: len(prefix_block)] != list(prefix_block):
        return None
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    vehicles = [vehicle_by_no.get(vehicle_no) for vehicle_no in prefix_block]
    if any(vehicle is None for vehicle in vehicles):
        return None
    from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group

    if validate_hook_vehicle_group([vehicle for vehicle in vehicles if vehicle is not None]):
        return None
    route_oracle = RouteOracle(master)
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length
        for vehicle in plan_input.vehicles
    }
    access_result = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=state.track_sequences,
        carried_train_length_m=sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.loco_carry
        ),
        loco_node=state.loco_node,
    )
    if not access_result.is_valid:
        return None
    return HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(prefix_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )


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
    started_at = perf_counter()
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

    beam_budget_ms = _remaining_child_budget_ms(started_at, time_budget_ms)
    if beam_budget_ms <= MIN_CHILD_STAGE_BUDGET_MS:
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

def _final_track_goal_capacity_warnings(plan_input: NormalizedPlanInput) -> list[dict[str, float | str]]:
    warnings: list[dict[str, float | str]] = []
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
            warnings.append(
                {
                    "track": track_name,
                    "required_length": round(total_length, 1),
                    "capacity": round(capacity, 1),
                    "effective_capacity": round(effective_capacity, 1),
                }
            )
    return warnings


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
