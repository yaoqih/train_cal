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
BEAM_COMPLETE_SEED_MIN_BUDGET_RATIO = 0.20
BEAM_COMPLETE_SEED_FULL_BUDGET_HOOKS = 100
RELAXED_CONSTRUCTIVE_RETRY_BUDGET_MS = 8_000.0
ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS = 20_000.0
LOCALIZED_RESUME_BUDGET_RATIO = 0.50
LOCALIZED_RESUME_MIN_FULL_BEAM_MS = 500.0
DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 4
RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 10
RELAXED_RESCUE_MAX_FINAL_HEURISTIC = DEFAULT_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
NEAR_GOAL_PARTIAL_RESUME_MAX_BUDGET_MS = 60_000.0
NEAR_GOAL_PARTIAL_RESUME_BUDGET_RATIO = 2.0 / 3.0


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
        _ts = perf_counter()
        elapsed_so_far_ms = (perf_counter() - started_at) * 1000
        warm_budget = min(500.0, max(50.0, (time_budget_ms - elapsed_so_far_ms) * 0.05))
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
        _ts = perf_counter()
        elapsed_so_far_ms = (perf_counter() - started_at) * 1000
        resume_budget = _partial_resume_budget_ms(
            constructive_seed,
            remaining_budget_ms=time_budget_ms - elapsed_so_far_ms,
            max_final_heuristic=near_goal_partial_resume_max_final_heuristic,
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
        _ts = perf_counter()
        relaxed_seed = _run_constructive_stage(
            plan_input=plan_input,
            initial_state=initial_state,
            master=master,
            time_budget_ms=min(time_budget_ms, RELAXED_CONSTRUCTIVE_RETRY_BUDGET_MS),
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
                    or relaxed_final_h <= RELAXED_RESCUE_MAX_FINAL_HEURISTIC
                )
            ):
                relaxed_remaining_ms = time_budget_ms - (perf_counter() - started_at) * 1000
                if relaxed_remaining_ms > 250:
                    relaxed_resume = _try_resume_partial_completion(
                        plan_input=plan_input,
                        initial_state=initial_state,
                        constructive_plan=relaxed_candidate.partial_plan,
                        master=master,
                        time_budget_ms=relaxed_remaining_ms,
                        enable_depot_late_scheduling=optimize_depot_late_in_search,
                    )
                    if relaxed_resume is not None:
                        relaxed_candidate = relaxed_resume
            constructive_seed = _shorter_complete_result(constructive_seed, relaxed_candidate)
            if (
                not constructive_seed.is_complete
                and relaxed_candidate.partial_plan
                and len(relaxed_candidate.partial_plan) > len(constructive_seed.partial_plan)
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
        _ts = perf_counter()
        route_release_budget = min(
            time_budget_ms,
            ROUTE_RELEASE_CONSTRUCTIVE_RETRY_BUDGET_MS,
        )
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
            constructive_seed = _shorter_complete_result(
                constructive_seed,
                route_release_seed,
            )
            if (
                not constructive_seed.is_complete
                and route_release_seed.partial_plan
                and len(route_release_seed.partial_plan) > len(constructive_seed.partial_plan)
            ):
                constructive_seed = replace(
                    constructive_seed,
                    partial_plan=list(route_release_seed.partial_plan),
                    partial_fallback_stage=route_release_seed.partial_fallback_stage,
                )
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
            enable_depot_late_scheduling=optimize_depot_late_in_search,
        )
        phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000
    else:
        effective_heuristic_weight = heuristic_weight
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
            )
            result = _shorter_complete_result(constructive_seed, search_result)
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
        if solver_mode == "exact" and enable_anytime_fallback and not result.is_proven_optimal:
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
            result = _shorter_complete_result(result, chain_result)
            phase_timings["anytime_ms"] = (perf_counter() - _ts) * 1000
        if solver_mode == "beam" and beam_width is not None:
            _ts = perf_counter()
            improved = result
            _lns_remaining_ms = (
                max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000)
                if time_budget_ms is not None else None
            )
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
            result = improved
            phase_timings["lns_ms"] = (perf_counter() - _ts) * 1000
            # If beam search found no plan, run the anytime fallback chain.
            # When the constructive seed is partial, pass an empty incumbent so
            # the chain's early-exit guard (`if current.plan: break`) doesn't
            # short-circuit immediately.
            if not result.is_complete and enable_anytime_fallback:
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
                result = _anytime_run_fallback_chain(
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
                phase_timings["anytime_ms"] = (perf_counter() - _ts) * 1000

    if not result.is_complete and constructive_seed is not None:
        if constructive_seed.is_complete:
            result = constructive_seed
        elif constructive_seed.partial_plan:
            result = replace(
                result,
                partial_plan=list(constructive_seed.partial_plan),
                partial_fallback_stage=constructive_seed.partial_fallback_stage,
            )

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
) -> SolverResult:
    if candidate is None:
        if incumbent is None:
            raise ValueError("at least one solver result is required")
        return incumbent
    if incumbent is None:
        return candidate
    if not candidate.is_complete:
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


def _solver_result_final_heuristic(result: SolverResult) -> float | None:
    if result.debug_stats is None:
        return None
    value = result.debug_stats.get("final_heuristic")
    return float(value) if value is not None else None


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
    vehicle_by_no: dict[str, NormalizedVehicle] = {v.vehicle_no: v for v in plan_input.vehicles}
    state = initial_state
    resumed_prefix: list[HookAction] = []
    checkpoints: list[tuple[int, int, list[HookAction], ReplayState]] = []
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

    last_checkpoint = checkpoints[-1]
    last_result = _try_resume_from_checkpoint(
        plan_input=plan_input,
        checkpoint_prefix=last_checkpoint[2],
        checkpoint_state=last_checkpoint[3],
        master=master,
        time_budget_ms=time_budget_ms,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if last_result is not None:
        return last_result

    ranked_checkpoints = [
        checkpoint
        for checkpoint in sorted(checkpoints)[:PARTIAL_RESUME_MAX_CHECKPOINTS]
        if checkpoint is not last_checkpoint
    ]
    per_checkpoint_budget = max(250.0, time_budget_ms / max(1, len(ranked_checkpoints)))

    for _unfinished_count, _negative_prefix_len, checkpoint_prefix, checkpoint_state in ranked_checkpoints:
        checkpoint_result = _try_resume_from_checkpoint(
            plan_input=plan_input,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_state=checkpoint_state,
            master=master,
            time_budget_ms=per_checkpoint_budget,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if checkpoint_result is not None:
            return checkpoint_result
    return None


def _try_resume_from_checkpoint(
    *,
    plan_input: NormalizedPlanInput,
    checkpoint_prefix: list[HookAction],
    checkpoint_state: ReplayState,
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
        try:
            localized_completion = _solve_search_result(
                plan_input=plan_input,
                initial_state=checkpoint_state,
                master=master,
                solver_mode="beam",
                heuristic_weight=1.0,
                beam_width=8,
                budget=SearchBudget(time_budget_ms=remaining_budget_ms),
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                enable_structural_diversity=True,
            )
        except ValueError:
            localized_completion = None
    if localized_completion is None or not localized_completion.is_complete:
        return None
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

    localized_input = _build_repair_plan_input(plan_input, initial_state)
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
