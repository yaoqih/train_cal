"""A* / weighted / beam search core for the shunting solver.

Implements a single configurable search loop. Mode selection (``exact`` /
``weighted`` / ``beam``) changes priority calculation and pruning, not the
loop shape itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from itertools import count
from time import perf_counter
from typing import Any

from fzed_shunting.domain.carry_order import is_carried_tail_block
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.exact_spot import (
    exact_spot_clearance_bonus,
    exact_spot_seeker_exposure_bonus,
)
from fzed_shunting.solver.heuristic import make_state_heuristic_real_hook
from fzed_shunting.solver.move_generator import (
    _collect_interfering_goal_targets_by_source,
)
from fzed_shunting.solver.candidate_compiler import replay_candidate_steps
from fzed_shunting.solver.debt_graph import build_debt_graph
from fzed_shunting.solver.move_candidates import MoveCandidate, generate_move_candidates
from fzed_shunting.solver.purity import compute_state_purity
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.route_blockage import (
    RouteBlockagePlan,
    compute_route_blockage_plan,
    route_blockage_release_score,
    route_release_continuation_bonus,
    route_release_focus_after_move,
)
from fzed_shunting.solver.state import (
    _apply_move,
    _canonical_random_depot_vehicle_nos,
    _is_goal,
    _state_key,
)
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

BEAM_SHALLOW_RESERVE = 1
BEAM_LOW_STRUCTURAL_DEBT_RESERVE = 1
BEAM_STRUCTURAL_DIVERSITY_FRONTIER_MULTIPLIER = 3
BEAM_STRUCTURAL_DIVERSITY_MIN_STAGING_TO_STAGING = 80
BEAM_STRUCTURAL_DIVERSITY_MIN_REPEATED_STAGING_TOUCHES = 80


@dataclass(order=True)
class QueueItem:
    priority: tuple
    seq: int
    state_key: tuple
    state: ReplayState
    plan: list[HookAction]
    cost: int = field(default=0, compare=False)
    structural_key: tuple[int, ...] = field(
        default=(-1, 0, 0, 0, 0, 0, 0),
        compare=False,
    )
    route_release_focus_tracks: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
    )
    route_release_focus_bonus: int = field(default=0, compare=False)
    route_release_focus_ttl: int = field(default=0, compare=False)
    structural_reserve_tracks: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
    )
    blocker_reserve: bool = field(default=False, compare=False)
    last_candidate_origin: str = field(default="", compare=False)
    active_intent_type: str = field(default="", compare=False)
    active_intent_anchor: str = field(default="", compare=False)
    active_intent_target: str = field(default="", compare=False)
    active_intent_stage: str = field(default="", compare=False)
    active_intent_ttl: int = field(default=0, compare=False)
    problem_commitment_type: str = field(default="", compare=False)
    problem_commitment_track: str = field(default="", compare=False)
    problem_commitment_ttl: int = field(default=0, compare=False)
    structural_commitment_type: str = field(default="", compare=False)
    structural_commitment_anchor: str = field(default="", compare=False)
    structural_commitment_target: str = field(default="", compare=False)
    structural_commitment_ttl: int = field(default=0, compare=False)


@dataclass(frozen=True)
class CandidateScoring:
    blocker_bonus: int
    structural_progress_bonus: int
    exact_spot_priority: int
    route_release_regression_penalty: int
    staging_churn_penalty: int
    carry_fragmentation_penalty: int
    route_release_focus_tracks: frozenset[str]
    route_release_focus_bonus: int
    route_release_focus_ttl: int
    carry_growth_penalty: int
    intent_continuation_bonus: int = 0
    problem_commitment_bonus: int = 0
    intent_switch_penalty: int = 0
    structural_commitment_bonus: int = 0
    structural_stability_penalty: int = 0
    soft_penalty: int = 0


@dataclass(frozen=True)
class AppliedCandidate:
    final_state: ReplayState
    steps: list[HookAction]
    transitions: list[tuple[ReplayState, HookAction, ReplayState]]


@dataclass(frozen=True)
class CandidateBranchRecord:
    candidate: MoveCandidate
    applied_candidate: AppliedCandidate
    scoring: CandidateScoring
    heuristic: int


def _solve_search_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    solver_mode: str,
    heuristic_weight: float,
    beam_width: int | None,
    debug_stats: dict[str, Any] | None = None,
    budget: SearchBudget | None = None,
    enable_depot_late_scheduling: bool = False,
    enable_structural_diversity: bool = False,
) -> SolverResult:
    if budget is None:
        budget = SearchBudget()
    else:
        budget.reset()
    started_at = budget.started_at
    counter = count()
    queue: list[QueueItem] = []
    canonical_random_depot_vehicle_nos = _canonical_random_depot_vehicle_nos(plan_input)
    state_heuristic = make_state_heuristic_real_hook(plan_input)
    initial_key = _state_key(
        initial_state,
        canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
    )
    initial_heuristic = state_heuristic(initial_state)
    initial_purity = compute_state_purity(plan_input, initial_state)
    heappush(
        queue,
        QueueItem(
            priority=_priority(
                cost=0,
                heuristic=initial_heuristic,
                soft_penalty=0,
                solver_mode=solver_mode,
                heuristic_weight=heuristic_weight,
                neg_depot_index_sum=0,
                carry_count=len(initial_state.loco_carry),
                carry_fragmentation_penalty=0,
                carry_growth_penalty=0,
                purity_metrics=(
                    initial_purity.unfinished_count,
                    0,
                    initial_purity.preferred_violation_count,
                    initial_purity.staging_pollution_count,
                ),
            ),
            seq=next(counter),
            state_key=initial_key,
            state=initial_state,
            plan=[],
            structural_key=_approximate_beam_structural_key([], initial_state),
            active_intent_type="",
            active_intent_anchor="",
            active_intent_target="",
            active_intent_stage="",
            active_intent_ttl=0,
            problem_commitment_type="",
            problem_commitment_track="",
            problem_commitment_ttl=0,
            structural_commitment_type="",
            structural_commitment_anchor="",
            structural_commitment_target="",
            structural_commitment_ttl=0,
        ),
    )
    best_cost = {initial_key: 0}
    generated_nodes = 1
    expanded_nodes = 0
    closed_state_keys: set[tuple] = set()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in plan_input.vehicles}
    route_oracle = RouteOracle(master) if master is not None else None
    _initialize_debug_stats(debug_stats, generated_nodes=generated_nodes)
    best_goal_plan: list[HookAction] | None = None
    best_partial_plan: list[HookAction] = []
    best_partial_score: tuple[int, ...] | None = None
    budget_exhausted = False
    while queue:
        if budget.exhausted():
            budget_exhausted = True
            break
        current = heappop(queue)
        current_cost = current.cost
        if best_cost.get(current.state_key) != current_cost:
            continue
        expanded_nodes += 1
        budget.tick_expand()
        closed_state_keys.add(current.state_key)
        if current.plan:
            partial_purity = compute_state_purity(plan_input, current.state)
            partial_structural = compute_structural_metrics(plan_input, current.state)
            partial_route_pressure = (
                compute_route_blockage_plan(plan_input, current.state, route_oracle).total_blockage_pressure
                if route_oracle is not None
                else 0
            )
            partial_score = _search_partial_score(
                heuristic=state_heuristic(current.state),
                purity=partial_purity,
                structural=partial_structural,
                route_pressure=partial_route_pressure,
                plan_len=len(current.plan),
            )
            if best_partial_score is None or partial_score < best_partial_score:
                best_partial_score = partial_score
                best_partial_plan = list(current.plan)
        if debug_stats is not None:
            debug_stats["expanded_states"] = expanded_nodes
            debug_stats["closed_states"] = len(closed_state_keys)
        if _is_goal(plan_input, current.state):
            if best_goal_plan is None or len(current.plan) < len(best_goal_plan):
                best_goal_plan = current.plan
            if solver_mode in ("exact", "real_hook"):
                return SolverResult(
                    plan=current.plan,
                    expanded_nodes=expanded_nodes,
                    generated_nodes=generated_nodes,
                    closed_nodes=len(closed_state_keys),
                    elapsed_ms=(perf_counter() - started_at) * 1000,
                    is_complete=True,
                    is_proven_optimal=solver_mode == "exact",
                    fallback_stage=solver_mode,
                    debug_stats=debug_stats,
                )
            continue
        move_stats = {} if debug_stats is not None else None
        blocking_goal_targets_by_source = _collect_interfering_goal_targets_by_source(
            plan_input=plan_input,
            state=current.state,
            goal_by_vehicle=goal_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            route_oracle=route_oracle,
        )
        route_blockage_plan = (
            compute_route_blockage_plan(plan_input, current.state, route_oracle)
            if route_oracle is not None
            else None
        )
        candidates = _search_generate_move_candidates(
            plan_input=plan_input,
            state=current.state,
            master=master,
            route_oracle=route_oracle,
            debug_stats=move_stats,
            active_problem_type=(
                current.problem_commitment_type or current.active_intent_type
            ),
            active_problem_track=(
                current.problem_commitment_track or current.active_intent_anchor
            ),
        )
        if current.active_intent_type and move_stats is not None:
            move_stats["prior_active_intent"] = {
                "intent_type": current.active_intent_type,
                "intent_anchor": current.active_intent_anchor,
                "intent_target": current.active_intent_target,
                "intent_stage": current.active_intent_stage,
                "intent_ttl": current.active_intent_ttl,
            }
        if current.problem_commitment_track and move_stats is not None:
            move_stats["prior_problem_commitment"] = {
                "problem_type": current.problem_commitment_type,
                "problem_track": current.problem_commitment_track,
                "ttl": current.problem_commitment_ttl,
            }
        if debug_stats is not None:
            _accumulate_move_debug_stats(
                debug_stats=debug_stats,
                state=current.state,
                plan_len=len(current.plan),
                move_stats=move_stats or {},
        )
        candidate_branches = _collect_candidate_branches(
            candidates=candidates,
            plan_input=plan_input,
            base_state=current.state,
            vehicle_by_no=vehicle_by_no,
            goal_by_vehicle=goal_by_vehicle,
            route_oracle=route_oracle,
            state_heuristic=state_heuristic,
            blocking_goal_targets_by_source=blocking_goal_targets_by_source,
            route_blockage_plan=route_blockage_plan,
            prior_focus_tracks=(
                current.route_release_focus_tracks
                if current.route_release_focus_ttl > 0
                else frozenset()
            ),
            prior_focus_bonus=current.route_release_focus_bonus,
            prior_focus_ttl=current.route_release_focus_ttl,
            active_intent_type=current.active_intent_type,
            active_intent_anchor=current.active_intent_anchor,
            active_intent_target=current.active_intent_target,
            active_intent_stage=current.active_intent_stage,
            active_intent_ttl=current.active_intent_ttl,
            problem_commitment_type=current.problem_commitment_type,
            problem_commitment_track=current.problem_commitment_track,
            problem_commitment_ttl=current.problem_commitment_ttl,
            structural_commitment_type=current.structural_commitment_type,
            structural_commitment_anchor=current.structural_commitment_anchor,
            structural_commitment_target=current.structural_commitment_target,
            structural_commitment_ttl=current.structural_commitment_ttl,
        )
        prioritized_branch_records = _prioritize_candidate_branches(candidate_branches)
        if debug_stats is not None:
            _record_expansion_candidate_trace(
                debug_stats,
                plan_input=plan_input,
                current_plan_len=len(current.plan),
                current_state=current.state,
                branch_records=prioritized_branch_records,
                route_oracle=route_oracle,
            )
        for branch_record in prioritized_branch_records:
            candidate = branch_record.candidate
            branch = branch_record.applied_candidate
            candidate_scoring = branch_record.scoring
            next_state = branch.final_state
            candidate_steps = branch.steps
            next_plan = current.plan + candidate_steps
            cost = len(next_plan)
            state_key = _state_key(
                next_state,
                canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
            )
            if solver_mode == "beam" and _is_goal(plan_input, next_state):
                if best_goal_plan is None or len(next_plan) < len(best_goal_plan):
                    best_goal_plan = next_plan
                generated_nodes += 1
                if debug_stats is not None:
                    debug_stats["generated_nodes"] = generated_nodes
                    _record_selected_candidate_debug_stats(
                        debug_stats,
                        candidate=candidate,
                        step_count=len(candidate_steps),
                    )
                continue
            if state_key in best_cost and best_cost[state_key] <= cost:
                continue
            best_cost[state_key] = cost
            next_purity = compute_state_purity(plan_input, next_state)
            blocker_bonus = candidate_scoring.blocker_bonus
            structural_progress_bonus = candidate_scoring.structural_progress_bonus
            next_focus_tracks = candidate_scoring.route_release_focus_tracks
            next_focus_bonus = candidate_scoring.route_release_focus_bonus
            next_focus_ttl = candidate_scoring.route_release_focus_ttl
            route_release_regression_penalty = (
                candidate_scoring.route_release_regression_penalty
            )
            staging_churn_penalty = candidate_scoring.staging_churn_penalty
            carry_growth_penalty = candidate_scoring.carry_growth_penalty
            carry_fragmentation_penalty = candidate_scoring.carry_fragmentation_penalty
            route_release_regression_penalty += _route_release_regression_penalty(
                state=next_state,
                route_oracle=route_oracle,
                focus_tracks=next_focus_tracks,
                focus_ttl=next_focus_ttl,
            )
            if not _structural_candidate_owns_staging(candidate):
                staging_churn_penalty += _staging_churn_penalty(
                    transitions=branch.transitions,
                    state=current.state,
                    next_state=next_state,
                )
            else:
                carry_growth_penalty = 0
            neg_depot_index_sum = 0
            if enable_depot_late_scheduling and solver_mode == "exact":
                from fzed_shunting.solver.depot_late import weighted_depot_index_sum
                neg_depot_index_sum = -weighted_depot_index_sum(next_plan, vehicle_by_no)
            heappush(
                queue,
                QueueItem(
                    priority=_priority(
                        cost=cost,
                        heuristic=branch_record.heuristic,
                        soft_penalty=candidate_scoring.soft_penalty,
                        blocker_bonus=blocker_bonus,
                        structural_progress_bonus=structural_progress_bonus,
                        problem_continuation_bonus=(
                            candidate_scoring.intent_continuation_bonus
                            + candidate_scoring.problem_commitment_bonus
                        ),
                        structural_commitment_bonus=(
                            candidate_scoring.structural_commitment_bonus
                        ),
                        exact_spot_priority=candidate_scoring.exact_spot_priority,
                        route_release_regression_penalty=route_release_regression_penalty,
                        staging_churn_penalty=staging_churn_penalty,
                        carry_fragmentation_penalty=carry_fragmentation_penalty,
                        solver_mode=solver_mode,
                        heuristic_weight=heuristic_weight,
                        neg_depot_index_sum=neg_depot_index_sum,
                        purity_metrics=(
                            next_purity.unfinished_count,
                            0,
                            next_purity.preferred_violation_count,
                            next_purity.staging_pollution_count,
                        ),
                        carry_count=len(next_state.loco_carry),
                        carry_growth_penalty=carry_growth_penalty,
                    ),
                    seq=next(counter),
                    state_key=state_key,
                    state=next_state,
                    plan=next_plan,
                    cost=cost,
                    structural_key=_approximate_beam_structural_key(
                        next_plan,
                        next_state,
                    ),
                    route_release_focus_tracks=next_focus_tracks,
                    route_release_focus_bonus=next_focus_bonus,
                    route_release_focus_ttl=next_focus_ttl,
                    structural_reserve_tracks=(
                        frozenset(candidate.focus_tracks)
                        if candidate.structural_reserve
                        else frozenset()
                    ),
                    blocker_reserve=blocker_bonus > 0,
                    last_candidate_origin=(
                        candidate.origin or candidate.reason or candidate.kind
                    ),
                    active_intent_type=(
                        candidate.intent_type or current.active_intent_type
                    ),
                    active_intent_anchor=(
                        candidate.intent_anchor or current.active_intent_anchor
                    ),
                    active_intent_target=(
                        candidate.intent_target or current.active_intent_target
                    ),
                    active_intent_stage=(
                        candidate.problem_stage or current.active_intent_stage
                    ),
                    active_intent_ttl=_next_active_intent_ttl(
                        current_ttl=current.active_intent_ttl,
                        current_intent_type=current.active_intent_type,
                        current_intent_anchor=current.active_intent_anchor,
                        candidate=candidate,
                        scoring=candidate_scoring,
                    ),
                    problem_commitment_type=(
                        candidate.problem_type or current.problem_commitment_type
                    ),
                    problem_commitment_track=(
                        candidate.problem_track or current.problem_commitment_track
                    ),
                    problem_commitment_ttl=_next_problem_commitment_ttl(
                        current_ttl=current.problem_commitment_ttl,
                        current_problem_type=current.problem_commitment_type,
                        current_problem_track=current.problem_commitment_track,
                        candidate=candidate,
                        scoring=candidate_scoring,
                    ),
                    structural_commitment_type=(
                        candidate.intent_type
                        if candidate.intent_type and candidate_scoring.structural_commitment_bonus > 0
                        else current.structural_commitment_type
                    ),
                    structural_commitment_anchor=(
                        candidate.intent_anchor
                        if candidate.intent_anchor and candidate_scoring.structural_commitment_bonus > 0
                        else current.structural_commitment_anchor
                    ),
                    structural_commitment_target=(
                        candidate.intent_target
                        if candidate.intent_target and candidate_scoring.structural_commitment_bonus > 0
                        else current.structural_commitment_target
                    ),
                    structural_commitment_ttl=_next_structural_commitment_ttl(
                        current_ttl=current.structural_commitment_ttl,
                        current_commitment_type=current.structural_commitment_type,
                        current_commitment_anchor=current.structural_commitment_anchor,
                        candidate=candidate,
                        scoring=candidate_scoring,
                    ),
                ),
            )
            generated_nodes += 1
            if debug_stats is not None:
                debug_stats["generated_nodes"] = generated_nodes
                origin = candidate.origin or candidate.reason or candidate.kind
                selected_counter = debug_stats.setdefault(
                    "candidate_selected_origin_counts_pre_prune",
                    {},
                )
                selected_counter[origin] = selected_counter.get(origin, 0) + 1
                _record_selected_candidate_debug_stats(
                    debug_stats,
                    candidate=candidate,
                    step_count=len(candidate_steps),
                )
        if solver_mode == "beam" and beam_width is not None:
            _prune_queue(
                queue,
                best_cost,
                beam_width,
                plan_input=plan_input,
                enable_structural_diversity=enable_structural_diversity,
                debug_stats=debug_stats,
            )
    if best_goal_plan is None:
        if debug_stats is not None and best_partial_score is not None:
            debug_stats["search_best_partial_score"] = list(best_partial_score)
        if budget_exhausted:
            return SolverResult(
                plan=[],
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                closed_nodes=len(closed_state_keys),
                elapsed_ms=(perf_counter() - started_at) * 1000,
                is_complete=False,
                is_proven_optimal=False,
                fallback_stage=solver_mode,
                partial_plan=best_partial_plan,
                partial_fallback_stage=solver_mode if best_partial_plan else None,
                debug_stats=debug_stats,
            )
        raise ValueError("No solution found")
    proven_optimal = (not budget_exhausted) and solver_mode == "exact"
    return SolverResult(
        plan=best_goal_plan,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        closed_nodes=len(closed_state_keys),
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=proven_optimal,
        fallback_stage=solver_mode,
        debug_stats=debug_stats,
    )


def _apply_candidate_steps(
    *,
    candidate: MoveCandidate,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle | None = None,
) -> AppliedCandidate | None:
    steps = list(candidate.steps)
    if not steps:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=route_oracle,
        apply_move=_apply_move,
    )
    if compiled is None:
        return None
    return AppliedCandidate(
        final_state=compiled.final_state,
        steps=steps,
        transitions=list(compiled.transitions),
    )


def _candidate_queue_branches(
    candidate: MoveCandidate,
    applied_candidate: AppliedCandidate,
    *,
    plan_input: NormalizedPlanInput,
    base_state: ReplayState,
    route_oracle: RouteOracle | None,
) -> list[AppliedCandidate]:
    branches = [applied_candidate]
    if not candidate.structural_reserve or len(applied_candidate.transitions) <= 1:
        return branches
    best_progress_key = _candidate_branch_progress_key(
        plan_input=plan_input,
        state=base_state,
        route_oracle=route_oracle,
    )
    improving_prefixes: list[AppliedCandidate] = []
    for step_index in range(1, len(applied_candidate.transitions)):
        prefix_transitions = applied_candidate.transitions[:step_index]
        prefix_candidate = AppliedCandidate(
            final_state=prefix_transitions[-1][2],
            steps=applied_candidate.steps[:step_index],
            transitions=prefix_transitions,
        )
        prefix_progress_key = _candidate_branch_progress_key(
            plan_input=plan_input,
            state=prefix_candidate.final_state,
            route_oracle=route_oracle,
        )
        if prefix_progress_key >= best_progress_key:
            continue
        improving_prefixes.append(prefix_candidate)
        best_progress_key = prefix_progress_key
    if improving_prefixes:
        branches.append(improving_prefixes[0])
    return branches


def _candidate_branch_progress_key(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle | None,
) -> tuple[int | float, ...]:
    structural = compute_structural_metrics(plan_input, state)
    debt_graph = build_debt_graph(plan_input, state, route_oracle=route_oracle)
    route_pressure = (
        compute_route_blockage_plan(plan_input, state, route_oracle).total_blockage_pressure
        if route_oracle is not None
        else 0
    )
    return (
        structural.target_sequence_defect_count,
        structural.work_position_unfinished_count,
        debt_graph.max_multi_track_pressure,
        debt_graph.multi_track_component_count,
        structural.front_blocker_count,
        structural.capacity_overflow_track_count,
        _capacity_debt_total(structural),
        route_pressure,
        structural.goal_track_blocker_count,
        structural.unfinished_count,
        len(state.loco_carry),
    )


def _collect_candidate_branches(
    *,
    candidates: list[MoveCandidate],
    plan_input: NormalizedPlanInput,
    base_state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_by_vehicle: dict[str, Any],
    route_oracle: RouteOracle | None,
    state_heuristic: Any,
    blocking_goal_targets_by_source: dict[str, set[str]],
    route_blockage_plan: RouteBlockagePlan | None,
    prior_focus_tracks: frozenset[str] | set[str],
    prior_focus_bonus: int,
    prior_focus_ttl: int,
    active_intent_type: str = "",
    active_intent_anchor: str = "",
    active_intent_target: str = "",
    active_intent_stage: str = "",
    active_intent_ttl: int = 0,
    problem_commitment_type: str = "",
    problem_commitment_track: str = "",
    problem_commitment_ttl: int = 0,
    structural_commitment_type: str = "",
    structural_commitment_anchor: str = "",
    structural_commitment_target: str = "",
    structural_commitment_ttl: int = 0,
) -> list[CandidateBranchRecord]:
    branch_records: list[CandidateBranchRecord] = []
    for candidate in candidates:
        applied_candidate = _apply_candidate_steps(
            candidate=candidate,
            state=base_state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
            route_oracle=route_oracle,
        )
        if applied_candidate is None:
            continue
        for branch in _candidate_queue_branches(
            candidate,
            applied_candidate,
            plan_input=plan_input,
            base_state=base_state,
            route_oracle=route_oracle,
        ):
            final_state = branch.final_state
            branch_records.append(
                CandidateBranchRecord(
                    candidate=candidate,
                    applied_candidate=branch,
                    scoring=_evaluate_candidate_steps(
                        plan_input=plan_input,
                        state=base_state,
                        final_state=final_state,
                        transitions=branch.transitions,
                        candidate=candidate,
                        vehicle_by_no=vehicle_by_no,
                        goal_by_vehicle=goal_by_vehicle,
                        route_oracle=route_oracle,
                        blocking_goal_targets_by_source=blocking_goal_targets_by_source,
                        route_blockage_plan=route_blockage_plan,
                        prior_focus_tracks=prior_focus_tracks,
                        prior_focus_bonus=prior_focus_bonus,
                        prior_focus_ttl=prior_focus_ttl,
                        active_intent_type=active_intent_type,
                        active_intent_anchor=active_intent_anchor,
                        active_intent_target=active_intent_target,
                        active_intent_stage=active_intent_stage,
                        active_intent_ttl=active_intent_ttl,
                        problem_commitment_type=problem_commitment_type,
                        problem_commitment_track=problem_commitment_track,
                        problem_commitment_ttl=problem_commitment_ttl,
                        structural_commitment_type=structural_commitment_type,
                        structural_commitment_anchor=structural_commitment_anchor,
                        structural_commitment_target=structural_commitment_target,
                        structural_commitment_ttl=structural_commitment_ttl,
                    ),
                    heuristic=state_heuristic(final_state),
                )
            )
    return branch_records


def _prioritize_candidate_branches(
    branch_records: list[CandidateBranchRecord],
) -> list[CandidateBranchRecord]:
    if not branch_records:
        return []
    strong_keys = {_candidate_entry_gate_key(record.candidate) for record in branch_records}
    return sorted(
        branch_records,
        key=lambda record: _candidate_entry_gate_sort_key(
            record,
            strong_keys=strong_keys,
        ),
    )


def _search_generate_move_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    debug_stats: dict[str, Any] | None,
    active_problem_type: str,
    active_problem_track: str,
) -> list[MoveCandidate]:
    try:
        return generate_move_candidates(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
            debug_stats=debug_stats,
            active_problem_type=active_problem_type,
            active_problem_track=active_problem_track,
        )
    except TypeError as exc:
        if "active_problem_type" not in str(exc):
            raise
        return generate_move_candidates(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
            debug_stats=debug_stats,
        )


def _candidate_entry_gate_sort_key(
    record: CandidateBranchRecord,
    *,
    strong_keys: set[tuple[str, str]],
) -> tuple[Any, ...]:
    candidate = record.candidate
    origin = candidate.origin or candidate.reason or candidate.kind
    source_track = _candidate_primary_source_track(candidate)
    lane_rank = 0
    if origin in {"resource_release_dispatch", "resource_release_stage_target"}:
        lane_rank = 1
    if source_track and (source_track, "frontier") in strong_keys:
        if origin == "resource_release_dispatch":
            lane_rank = max(lane_rank, 2)
        elif origin == "resource_release_stage_target":
            lane_rank = max(lane_rank, 3)
    if source_track and (source_track, "direct") in strong_keys:
        if origin == "resource_release_dispatch":
            lane_rank = max(lane_rank, 2)
        elif origin == "resource_release_stage_target":
            lane_rank = max(lane_rank, 2)
        elif origin == "goal_frontier_source_opening":
            lane_rank = max(lane_rank, 1)
    scoring = record.scoring
    return (
        lane_rank,
        0 if candidate.kind == "structural" else 1,
        0 if scoring.exact_spot_priority > 0 else 1,
        {
            "ADVANCE_WORK_SLOT": 0,
            "ADVANCE_ORDER": 1,
            "DELIVER": 2,
            "OPEN_GOAL_ACCESS": 3,
            "OPEN_SOURCE_ACCESS": 4,
            "RELIEVE_CAPACITY": 5,
            "BREAK_CHAIN": 6,
            "BORROW_BUFFER": 7,
            "RESTORE": 8,
        }.get(candidate.intent_type, 9),
        0 if candidate.problem_stage.endswith("primary") else 1,
        0 if scoring.structural_progress_bonus > 0 else 1,
        0 if scoring.blocker_bonus > 0 else 1,
        -scoring.structural_progress_bonus,
        -scoring.blocker_bonus,
        scoring.intent_switch_penalty,
        scoring.soft_penalty,
        -scoring.intent_continuation_bonus,
        len(record.applied_candidate.steps),
        record.heuristic,
        origin,
    )


def _candidate_entry_gate_key(candidate: MoveCandidate) -> tuple[str, str] | None:
    source_track = _candidate_primary_source_track(candidate)
    if source_track is None:
        return None
    origin = candidate.origin or candidate.reason or candidate.kind
    if origin == "goal_frontier_source_opening":
        return (source_track, "frontier")
    if origin == "resource_release_direct_target":
        return (source_track, "direct")
    return None


def _candidate_primary_source_track(candidate: MoveCandidate) -> str | None:
    for step in candidate.steps:
        if step.source_track:
            return step.source_track
    return None


def _evaluate_candidate_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    final_state: ReplayState,
    transitions: list[tuple[ReplayState, HookAction, ReplayState]],
    candidate: MoveCandidate,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_by_vehicle: dict[str, Any],
    route_oracle: RouteOracle | None,
    blocking_goal_targets_by_source: dict[str, set[str]],
    route_blockage_plan: RouteBlockagePlan | None,
    prior_focus_tracks: frozenset[str] | set[str],
    prior_focus_bonus: int,
    prior_focus_ttl: int,
    active_intent_type: str = "",
    active_intent_anchor: str = "",
    active_intent_target: str = "",
    active_intent_stage: str = "",
    active_intent_ttl: int = 0,
    problem_commitment_type: str = "",
    problem_commitment_track: str = "",
    problem_commitment_ttl: int = 0,
    structural_commitment_type: str = "",
    structural_commitment_anchor: str = "",
    structural_commitment_target: str = "",
    structural_commitment_ttl: int = 0,
) -> CandidateScoring:
    blocker_bonus = 0
    soft_penalty = getattr(candidate, "soft_penalty", 0)
    exact_spot_priority = 0
    route_release_regression_penalty = 0
    staging_churn_penalty = 0
    carry_fragmentation_penalty = 0
    carry_growth_penalty = 0
    carry_structural_bonus = 0
    intent_continuation_bonus = 0
    problem_commitment_bonus = 0
    intent_switch_penalty = 0
    structural_commitment_bonus = 0
    structural_stability_penalty = 0
    focus_tracks = frozenset(prior_focus_tracks)
    focus_bonus = prior_focus_bonus
    focus_ttl = prior_focus_ttl
    before_structural = compute_structural_metrics(plan_input, state)
    after_structural = compute_structural_metrics(plan_input, final_state)
    before_route_pressure = (
        route_blockage_plan.total_blockage_pressure
        if route_blockage_plan is not None
        else 0
    )
    after_route_pressure = before_route_pressure
    if (
        route_oracle is not None
        and route_blockage_plan is not None
        and before_route_pressure > 0
    ):
        after_route_pressure = compute_route_blockage_plan(
            plan_input,
            final_state,
            route_oracle,
        ).total_blockage_pressure
    structural_progress_bonus = _structural_metric_progress_bonus(
        before_structural,
        after_structural,
        step_count=len(transitions),
        before_route_pressure=before_route_pressure,
        after_route_pressure=after_route_pressure,
    )
    current_blocking_goal_targets = blocking_goal_targets_by_source
    current_route_blockage_plan = route_blockage_plan
    if candidate.intent_type:
        same_intent = (
            active_intent_ttl > 0
            and candidate.intent_type == active_intent_type
            and (not active_intent_anchor or candidate.intent_anchor == active_intent_anchor)
        )
        if same_intent:
            intent_continuation_bonus += 18
            if active_intent_stage and candidate.problem_stage == active_intent_stage:
                intent_continuation_bonus += 6
            if active_intent_target and candidate.intent_target == active_intent_target:
                intent_continuation_bonus += 4
        elif active_intent_ttl > 0 and active_intent_type:
            intent_switch_penalty += 10
    same_problem_commitment_track = (
        problem_commitment_ttl > 0
        and candidate.problem_track
        and candidate.problem_track == problem_commitment_track
    )
    same_problem_commitment_type = (
        same_problem_commitment_track
        and problem_commitment_type
        and candidate.problem_type == problem_commitment_type
    )
    if same_problem_commitment_track:
        problem_commitment_bonus += 16
        if same_problem_commitment_type:
            problem_commitment_bonus += 8
        if structural_progress_bonus > 0 or blocker_bonus > 0:
            problem_commitment_bonus += 10
    elif (
        problem_commitment_ttl > 0
        and problem_commitment_track
        and candidate.problem_track
        and candidate.problem_track != problem_commitment_track
        and structural_progress_bonus < 12
    ):
        intent_switch_penalty += 8
    same_structural_commitment = (
        structural_commitment_ttl > 0
        and candidate.intent_type
        and candidate.intent_type == structural_commitment_type
        and (
            not structural_commitment_anchor
            or candidate.intent_anchor == structural_commitment_anchor
        )
    )
    if candidate.kind == "structural" and candidate.intent_type:
        if structural_progress_bonus >= 24:
            structural_commitment_bonus += 16
        elif structural_progress_bonus >= 12:
            structural_commitment_bonus += 8
        if same_structural_commitment and structural_progress_bonus > 0:
            structural_commitment_bonus += 24
            if (
                structural_commitment_target
                and candidate.intent_target == structural_commitment_target
            ):
                structural_commitment_bonus += 8
        elif structural_commitment_ttl > 0 and structural_progress_bonus < 12:
            intent_switch_penalty += 12
        if (
            active_intent_ttl > 0
            and candidate.intent_type == active_intent_type
            and (not active_intent_anchor or candidate.intent_anchor == active_intent_anchor)
            and structural_progress_bonus > 0
        ):
            structural_commitment_bonus += 12
    elif (
        candidate.kind == "primitive"
        and (
            (active_intent_ttl > 0 and active_intent_type)
            or (structural_commitment_ttl > 0 and structural_commitment_type)
        )
    ):
        intent_switch_penalty += 14
    for step_index, (current_state, step, next_state) in enumerate(transitions):
        if step_index > 0:
            current_blocking_goal_targets = _collect_interfering_goal_targets_by_source(
                plan_input=plan_input,
                state=current_state,
                goal_by_vehicle=goal_by_vehicle,
                vehicle_by_no=vehicle_by_no,
                route_oracle=route_oracle,
            )
            current_route_blockage_plan = (
                compute_route_blockage_plan(plan_input, current_state, route_oracle)
                if route_oracle is not None
                else None
            )

        active_focus_tracks = focus_tracks if focus_ttl > 0 else frozenset()
        blocker_bonus += _blocking_goal_target_bonus(
            state=current_state,
            move=step,
            blocking_goal_targets_by_source=current_blocking_goal_targets,
            route_blockage_plan=current_route_blockage_plan,
            route_release_focus_tracks=active_focus_tracks,
            route_release_focus_bonus=focus_bonus,
        )
        exact_clearance_bonus = exact_spot_clearance_bonus(
            plan_input=plan_input,
            state=current_state,
            move=step,
            next_state=next_state,
        )
        exact_exposure_bonus = exact_spot_seeker_exposure_bonus(
            plan_input=plan_input,
            state=current_state,
            move=step,
            next_state=next_state,
        )
        exact_spot_priority += exact_clearance_bonus + exact_exposure_bonus
        blocker_bonus += exact_clearance_bonus + exact_exposure_bonus
        blocker_bonus += _carry_exposure_bonus(
            move=step,
            state=current_state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        route_release_regression_penalty += _route_release_repark_penalty(
            move=step,
            state=current_state,
            next_state=next_state,
            plan_input=plan_input,
            route_oracle=route_oracle,
            route_blockage_plan=current_route_blockage_plan,
            focus_tracks=active_focus_tracks,
            focus_ttl=focus_ttl,
        )
        if not _structural_candidate_owns_staging(candidate):
            staging_churn_penalty += _staging_churn_penalty(
                transitions=[(current_state, step, next_state)],
                state=current_state,
                next_state=next_state,
            )
        carry_growth_penalty = max(
            carry_growth_penalty,
            _carry_growth_penalty_for_move(
                move=step,
                state=current_state,
                vehicle_by_no=vehicle_by_no,
            ),
        )
        focus_tracks, focus_bonus, focus_ttl = route_release_focus_after_move(
            prior_focus_tracks=focus_tracks,
            prior_focus_bonus=focus_bonus,
            prior_focus_ttl=focus_ttl,
            move=step,
            route_blockage_plan=current_route_blockage_plan,
        )

    if state.loco_carry and candidate.kind == "structural":
        if candidate.reason.startswith(
            (
                "route_release_",
                "goal_frontier_",
                "work_position_",
                "resource_release",
                "chain_macro_",
            )
        ):
            carry_structural_bonus = min(12, 4 + len(transitions) * 2)
            blocker_bonus += carry_structural_bonus
    carry_fragmentation_penalty = _carry_commitment_penalty(
        candidate=candidate,
        state=state,
        final_state=final_state,
        before_structural=before_structural,
        structural_progress_bonus=structural_progress_bonus,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    )
    soft_penalty += _pseudo_progress_penalty(
        candidate=candidate,
        before=before_structural,
        after=after_structural,
        blocker_bonus=blocker_bonus,
        structural_progress_bonus=structural_progress_bonus,
        before_route_pressure=before_route_pressure,
        after_route_pressure=after_route_pressure,
        step_count=len(transitions),
    )
    structural_stability_penalty = _structural_instability_penalty(
        candidate=candidate,
        before=before_structural,
        after=after_structural,
        before_route_pressure=before_route_pressure,
        after_route_pressure=after_route_pressure,
    )
    soft_penalty += structural_stability_penalty
    phase1_structure_bonus, phase1_structure_penalty = _phase1_structure_bonus_penalty(
        plan_input=plan_input,
        state=state,
        final_state=final_state,
    )
    structural_progress_bonus += phase1_structure_bonus
    soft_penalty += phase1_structure_penalty
    if (
        candidate.kind == "primitive"
        and structural_commitment_ttl > 0
        and structural_commitment_type
        and structural_progress_bonus < 12
        and blocker_bonus <= 0
    ):
        soft_penalty += 22
    return CandidateScoring(
        blocker_bonus=blocker_bonus,
        structural_progress_bonus=structural_progress_bonus,
        exact_spot_priority=exact_spot_priority,
        route_release_regression_penalty=route_release_regression_penalty,
        staging_churn_penalty=staging_churn_penalty,
        carry_fragmentation_penalty=carry_fragmentation_penalty,
        route_release_focus_tracks=frozenset(focus_tracks),
        route_release_focus_bonus=focus_bonus,
        route_release_focus_ttl=focus_ttl,
        carry_growth_penalty=carry_growth_penalty,
        intent_continuation_bonus=intent_continuation_bonus,
        problem_commitment_bonus=problem_commitment_bonus,
        intent_switch_penalty=intent_switch_penalty,
        structural_commitment_bonus=structural_commitment_bonus,
        structural_stability_penalty=structural_stability_penalty,
        soft_penalty=soft_penalty + intent_switch_penalty,
    )


def _phase1_structure_bonus_penalty(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    final_state: ReplayState,
) -> tuple[int, int]:
    stage_policy = plan_input.stage_policy or {}
    if str(stage_policy.get("stageMode") or "") != "PHASE1_PRE_REPAIR_BUFFERING":
        return 0, 0
    raw_assignments = (
        stage_policy.get("packageAssignments")
        or stage_policy.get("layoutAssignments")
        or stage_policy.get("backboneAssignments")
        or {}
    )
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        return 0, 0
    assignments = {
        str(vehicle_no): str(track_name)
        for vehicle_no, track_name in raw_assignments.items()
    }
    buffer_tracks = frozenset(str(item) for item in stage_policy.get("bufferTracks") or ())
    deferred_vehicle_nos = frozenset(
        str(item) for item in stage_policy.get("deferredVehicleNos") or ()
    )
    diagnostics = stage_policy.get("phase1Diagnostics") or {}
    hidden_vehicle_nos = frozenset(
        str(item)
        for item in (
            diagnostics.get("hiddenVehicleNos")
            if isinstance(diagnostics, dict)
            else ()
        )
        or ()
    )
    raw_anchor_ranks = (
        stage_policy.get("packageTargetRanks")
        or stage_policy.get("layoutTargetRanks")
        or stage_policy.get("backboneTargetRanks")
        or {}
    )
    anchor_ranks = {
        str(vehicle_no): int(rank)
        for vehicle_no, rank in raw_anchor_ranks.items()
        if str(vehicle_no) in assignments
    }
    before_track_by_vehicle, before_rank_by_vehicle = _state_vehicle_track_layout(state)
    after_track_by_vehicle, after_rank_by_vehicle = _state_vehicle_track_layout(final_state)
    bonus = 0
    penalty = 0

    for vehicle_no, assigned_track in assignments.items():
        before_track = before_track_by_vehicle.get(vehicle_no)
        after_track = after_track_by_vehicle.get(vehicle_no)
        if before_track != assigned_track and after_track == assigned_track:
            bonus += 10
        elif before_track == assigned_track and after_track != assigned_track:
            penalty += 18
        if after_track in buffer_tracks and after_track != assigned_track:
            penalty += 8
        if before_track in buffer_tracks and before_track != assigned_track and after_track == assigned_track:
            bonus += 4

    for vehicle_no in deferred_vehicle_nos:
        before_track = before_track_by_vehicle.get(vehicle_no)
        after_track = after_track_by_vehicle.get(vehicle_no)
        if before_track != after_track and after_track in buffer_tracks:
            penalty += 14

    for vehicle_no in hidden_vehicle_nos:
        before_track = before_track_by_vehicle.get(vehicle_no)
        after_track = after_track_by_vehicle.get(vehicle_no)
        if before_track != after_track:
            penalty += 18

    before_inversions = _phase1_anchor_inversion_count(
        assignments=assignments,
        anchor_ranks=anchor_ranks,
        track_by_vehicle=before_track_by_vehicle,
        rank_by_vehicle=before_rank_by_vehicle,
    )
    after_inversions = _phase1_anchor_inversion_count(
        assignments=assignments,
        anchor_ranks=anchor_ranks,
        track_by_vehicle=after_track_by_vehicle,
        rank_by_vehicle=after_rank_by_vehicle,
    )
    if after_inversions < before_inversions:
        bonus += (before_inversions - after_inversions) * 8
    elif after_inversions > before_inversions:
        penalty += (after_inversions - before_inversions) * 10
    return bonus, penalty


def _state_vehicle_track_layout(
    state: ReplayState,
) -> tuple[dict[str, str], dict[str, int]]:
    track_by_vehicle: dict[str, str] = {}
    rank_by_vehicle: dict[str, int] = {}
    for track_name, sequence in state.track_sequences.items():
        for rank, vehicle_no in enumerate(sequence, start=1):
            track_by_vehicle[str(vehicle_no)] = str(track_name)
            rank_by_vehicle[str(vehicle_no)] = rank
    return track_by_vehicle, rank_by_vehicle


def _phase1_anchor_inversion_count(
    *,
    assignments: dict[str, str],
    anchor_ranks: dict[str, int],
    track_by_vehicle: dict[str, str],
    rank_by_vehicle: dict[str, int],
) -> int:
    anchors_by_track: dict[str, list[str]] = {}
    for vehicle_no, target_track in assignments.items():
        if vehicle_no not in anchor_ranks:
            continue
        anchors_by_track.setdefault(target_track, []).append(vehicle_no)
    inversion_count = 0
    for target_track, anchor_vehicle_nos in anchors_by_track.items():
        ordered_vehicle_nos = sorted(
            anchor_vehicle_nos,
            key=lambda vehicle_no: (anchor_ranks[vehicle_no], vehicle_no),
        )
        present_positions = [
            rank_by_vehicle[vehicle_no]
            for vehicle_no in ordered_vehicle_nos
            if track_by_vehicle.get(vehicle_no) == target_track
            and vehicle_no in rank_by_vehicle
        ]
        for left_index, left_pos in enumerate(present_positions):
            for right_pos in present_positions[left_index + 1:]:
                if left_pos > right_pos:
                    inversion_count += 1
    return inversion_count


def _priority(
    *,
    cost: int,
    heuristic: int,
    blocker_bonus: int = 0,
    soft_penalty: int = 0,
    structural_progress_bonus: int = 0,
    problem_continuation_bonus: int = 0,
    exact_spot_priority: int = 0,
    route_release_regression_penalty: int = 0,
    staging_churn_penalty: int = 0,
    carry_fragmentation_penalty: int = 0,
    solver_mode: str,
    heuristic_weight: float,
    neg_depot_index_sum: int = 0,
    purity_metrics: tuple[int, ...] = (0, 0, 0, 0),
    carry_count: int = 0,
    carry_growth_penalty: int = 0,
    structural_commitment_bonus: int = 0,
) -> tuple:
    combined_bonus = (
        max(blocker_bonus, 0)
        + max(structural_progress_bonus, 0)
        + max(problem_continuation_bonus, 0)
        + max(structural_commitment_bonus, 0)
    )
    score = cost + heuristic
    progress_bonus_cap = max(score, 1) if solver_mode == "beam" else max(score, 1) // 2
    progress_bonus = min(combined_bonus, progress_bonus_cap)
    if solver_mode == "beam":
        adjusted_score = score + soft_penalty + carry_fragmentation_penalty + carry_growth_penalty - progress_bonus
        return (
            route_release_regression_penalty,
            staging_churn_penalty,
            carry_fragmentation_penalty,
            0 if exact_spot_priority > 0 else 1,
            0 if structural_commitment_bonus > 0 else 1,
            -structural_commitment_bonus,
            0 if structural_progress_bonus > 0 else 1,
            -structural_progress_bonus,
            0 if blocker_bonus > 0 else 1,
            adjusted_score,
            score,
            cost,
            heuristic,
            purity_metrics,
            carry_growth_penalty,
            carry_count,
            neg_depot_index_sum,
            -blocker_bonus,
            -exact_spot_priority,
            heuristic,
            soft_penalty,
        )
    if solver_mode in ("weighted", "real_hook"):
        score = cost + heuristic_weight * heuristic
        adjusted_score = score + soft_penalty + carry_fragmentation_penalty + carry_growth_penalty - progress_bonus
        return (
            route_release_regression_penalty,
            staging_churn_penalty,
            carry_fragmentation_penalty,
            0 if exact_spot_priority > 0 else 1,
            0 if structural_progress_bonus > 0 else 1,
            -structural_progress_bonus,
            0 if blocker_bonus > 0 else 1,
            adjusted_score,
            cost,
            heuristic,
            purity_metrics,
            carry_growth_penalty,
            carry_count,
            neg_depot_index_sum,
            -blocker_bonus,
            -exact_spot_priority,
            soft_penalty,
        )
    return (
        route_release_regression_penalty,
        staging_churn_penalty,
        carry_fragmentation_penalty,
        cost + heuristic,
        cost,
        heuristic,
        purity_metrics,
        carry_growth_penalty,
        carry_count,
        neg_depot_index_sum,
        -blocker_bonus,
        -exact_spot_priority,
        -structural_progress_bonus,
        soft_penalty,
    )


def _next_active_intent_ttl(
    *,
    current_ttl: int,
    current_intent_type: str,
    current_intent_anchor: str,
    candidate: MoveCandidate,
    scoring: CandidateScoring,
) -> int:
    if not candidate.intent_type:
        return max(current_ttl - 1, 0)
    same_intent = (
        candidate.intent_type == current_intent_type
        and (not current_intent_anchor or candidate.intent_anchor == current_intent_anchor)
    )
    if same_intent and scoring.structural_progress_bonus > 0:
        return min(max(current_ttl, 2) + 1, 4)
    if same_intent:
        return max(current_ttl, 2)
    if scoring.structural_progress_bonus > 0 or scoring.blocker_bonus > 0:
        return 2
    return 1


def _next_structural_commitment_ttl(
    *,
    current_ttl: int,
    current_commitment_type: str,
    current_commitment_anchor: str,
    candidate: MoveCandidate,
    scoring: CandidateScoring,
) -> int:
    if (
        candidate.kind == "structural"
        and candidate.intent_type
        and scoring.structural_commitment_bonus > 0
    ):
        same_commitment = (
            candidate.intent_type == current_commitment_type
            and (
                not current_commitment_anchor
                or candidate.intent_anchor == current_commitment_anchor
            )
        )
        if same_commitment:
            return min(max(current_ttl, 2) + 1, 5)
        return 3
    return max(current_ttl - 1, 0)


def _next_problem_commitment_ttl(
    *,
    current_ttl: int,
    current_problem_type: str,
    current_problem_track: str,
    candidate: MoveCandidate,
    scoring: CandidateScoring,
) -> int:
    if not candidate.problem_track:
        return max(current_ttl - 1, 0)
    same_track = (
        current_problem_track
        and candidate.problem_track == current_problem_track
    )
    same_type = (
        same_track
        and current_problem_type
        and candidate.problem_type == current_problem_type
    )
    if scoring.problem_commitment_bonus > 0:
        if same_type:
            return min(max(current_ttl, 2) + 1, 5)
        if same_track:
            return min(max(current_ttl, 2) + 1, 4)
        return 2
    if scoring.structural_progress_bonus > 0 or scoring.blocker_bonus > 0:
        return 2
    return 1


def _search_partial_score(
    *,
    heuristic: int,
    purity: Any,
    structural: Any,
    route_pressure: int,
    plan_len: int,
) -> tuple[int, ...]:
    return (
        structural.target_sequence_defect_count,
        route_pressure,
        structural.staging_debt_count,
        structural.capacity_overflow_track_count,
        structural.work_position_unfinished_count,
        structural.goal_track_blocker_count,
        purity.preferred_violation_count,
        purity.unfinished_count,
        heuristic,
        -plan_len,
    )


def _carry_commitment_penalty(
    *,
    candidate: MoveCandidate,
    state: ReplayState,
    final_state: ReplayState,
    before_structural: Any,
    structural_progress_bonus: int,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    if candidate.kind != "primitive" or not state.loco_carry:
        return 0
    move = candidate.steps[0]
    if (
        final_state.loco_carry
        and len(final_state.loco_carry) < len(state.loco_carry)
        and structural_progress_bonus <= 0
    ):
        return 8 + len(final_state.loco_carry)
    if (
        not final_state.loco_carry
        and len(candidate.steps) == 1
        and move.action_type == "DETACH"
        and move.vehicle_nos
        and _primitive_detach_commits_carried_goal_block(
            final_state=final_state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
            move=move,
        )
        and (
            before_structural.work_position_unfinished_count > 0
            or before_structural.front_blocker_count > 0
            or before_structural.goal_track_blocker_count > 0
        )
        and structural_progress_bonus <= 6
    ):
        return 10
    return 0


def _primitive_detach_commits_carried_goal_block(
    *,
    move: HookAction,
    final_state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    return all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=move.target_track,
            state=final_state,
            plan_input=plan_input,
        )
        for vehicle_no in move.vehicle_nos
    )


def _structural_progress_bonus(
    *,
    plan_input: NormalizedPlanInput,
    before_state: ReplayState,
    after_state: ReplayState,
    step_count: int,
    route_oracle: RouteOracle | None,
    route_blockage_plan: RouteBlockagePlan | None,
) -> int:
    if step_count <= 0:
        return 0
    before = compute_structural_metrics(plan_input, before_state)
    after = compute_structural_metrics(plan_input, after_state)
    before_route_pressure = (
        route_blockage_plan.total_blockage_pressure
        if route_blockage_plan is not None
        else 0
    )
    after_route_pressure = before_route_pressure
    if (
        route_oracle is not None
        and route_blockage_plan is not None
        and before_route_pressure > 0
    ):
        after_route_pressure = compute_route_blockage_plan(
            plan_input,
            after_state,
            route_oracle,
        ).total_blockage_pressure
    return _structural_metric_progress_bonus(
        before,
        after,
        step_count=step_count,
        before_route_pressure=before_route_pressure,
        after_route_pressure=after_route_pressure,
    )


def _structural_metric_progress_bonus(
    before: Any,
    after: Any,
    *,
    step_count: int,
    before_route_pressure: int = 0,
    after_route_pressure: int = 0,
) -> int:
    progress = 0
    progress += max(0, before.target_sequence_defect_count - after.target_sequence_defect_count) * 18
    progress += max(0, before.work_position_unfinished_count - after.work_position_unfinished_count) * 12
    progress += max(0, before.capacity_overflow_track_count - after.capacity_overflow_track_count) * 16
    progress += max(0, before.front_blocker_count - after.front_blocker_count) * 8
    progress += max(0, before.goal_track_blocker_count - after.goal_track_blocker_count) * 3
    progress += max(0.0, _capacity_debt_total(before) - _capacity_debt_total(after))
    progress += max(0, before_route_pressure - after_route_pressure) * 4
    progress += max(0, before.unfinished_count - after.unfinished_count)
    staging_regression = max(0, after.staging_debt_count - before.staging_debt_count)
    progress -= staging_regression
    if progress <= 0:
        return 0
    return min(progress, max(step_count, 1) * 16)


def _pseudo_progress_penalty(
    *,
    candidate: MoveCandidate,
    before: Any,
    after: Any,
    blocker_bonus: int,
    structural_progress_bonus: int,
    before_route_pressure: int,
    after_route_pressure: int,
    step_count: int,
) -> int:
    if step_count <= 0:
        return 0
    improved_core = (
        after.target_sequence_defect_count < before.target_sequence_defect_count
        or after.work_position_unfinished_count < before.work_position_unfinished_count
        or after.capacity_overflow_track_count < before.capacity_overflow_track_count
        or after.front_blocker_count < before.front_blocker_count
        or after.goal_track_blocker_count < before.goal_track_blocker_count
        or _capacity_debt_total(after) + 1e-9 < _capacity_debt_total(before)
        or after_route_pressure < before_route_pressure
        or after.unfinished_count < before.unfinished_count
    )
    penalty = 0
    if candidate.kind == "structural" and not improved_core and blocker_bonus <= 0:
        penalty += 18 + min(step_count, 3) * 4
    elif candidate.kind == "primitive" and not improved_core and blocker_bonus <= 0:
        penalty += 6

    if (
        after.staging_debt_count > before.staging_debt_count
        and structural_progress_bonus <= 0
    ):
        penalty += (after.staging_debt_count - before.staging_debt_count) * 6
    if (
        after.front_blocker_count > before.front_blocker_count
        and structural_progress_bonus <= 0
    ):
        penalty += (after.front_blocker_count - before.front_blocker_count) * 8
    if (
        after.goal_track_blocker_count > before.goal_track_blocker_count
        and structural_progress_bonus <= 0
    ):
        penalty += (after.goal_track_blocker_count - before.goal_track_blocker_count) * 3
    if (
        _capacity_debt_total(after) > _capacity_debt_total(before) + 1e-9
        and structural_progress_bonus <= 0
    ):
        penalty += 8
    return penalty


def _structural_instability_penalty(
    *,
    candidate: MoveCandidate,
    before: Any,
    after: Any,
    before_route_pressure: int,
    after_route_pressure: int,
) -> int:
    if candidate.kind != "structural":
        return 0
    penalty = 0
    unfinished_growth = max(0, after.unfinished_count - before.unfinished_count)
    staging_growth = max(0, after.staging_debt_count - before.staging_debt_count)
    front_growth = max(0, after.front_blocker_count - before.front_blocker_count)
    goal_growth = max(0, after.goal_track_blocker_count - before.goal_track_blocker_count)
    route_improvement = max(0, before_route_pressure - after_route_pressure)
    if unfinished_growth >= 4:
        penalty += unfinished_growth * 3
    if staging_growth > 0:
        penalty += staging_growth * 8
    if front_growth > 0:
        penalty += front_growth * 8
    if goal_growth >= 2:
        penalty += goal_growth * 4
    if unfinished_growth > 0 and route_improvement <= 0:
        penalty += unfinished_growth * 2
    return penalty


def _capacity_debt_total(metrics: Any) -> float:
    capacity_debt_by_track = getattr(metrics, "capacity_debt_by_track", None)
    if not capacity_debt_by_track:
        return 0.0
    return sum(float(value) for value in capacity_debt_by_track.values())


def _carry_growth_penalty_for_move(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    if move.action_type != "ATTACH":
        return 0

    penalty = len(move.vehicle_nos) if state.loco_carry else 0
    projected_carry_count = len(state.loco_carry) + len(move.vehicle_nos)
    if projected_carry_count <= 10:
        return penalty

    if _is_soft_random_area_attach(move=move, vehicle_by_no=vehicle_by_no):
        penalty += (projected_carry_count - 10) * 2
    return penalty


def _is_soft_random_area_attach(
    *,
    move: HookAction,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if not move.vehicle_nos:
        return False
    for vehicle_no in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        goal = vehicle.goal
        if goal.target_mode not in {"AREA", "SNAPSHOT"}:
            return False
        if goal.target_area_code is None or ":RANDOM" not in goal.target_area_code:
            return False
    return True


def _route_release_regression_penalty(
    *,
    state: ReplayState,
    route_oracle: RouteOracle | None,
    focus_tracks: frozenset[str] | set[str],
    focus_ttl: int,
) -> int:
    if route_oracle is None or focus_ttl <= 0 or not focus_tracks:
        return 0
    penalty = 0
    for focus_track in sorted(focus_tracks):
        if focus_track == state.loco_track_name:
            continue
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=focus_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
        )
        if not access_result.is_valid:
            penalty += max(1, len(access_result.blocking_tracks))
    return penalty * max(1, focus_ttl)


def _route_release_repark_penalty(
    *,
    move: HookAction,
    state: ReplayState,
    next_state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
    route_blockage_plan: RouteBlockagePlan | None,
    focus_tracks: frozenset[str] | set[str],
    focus_ttl: int,
) -> int:
    if (
        route_oracle is None
        or route_blockage_plan is None
        or focus_ttl <= 0
        or not focus_tracks
        or move.action_type != "DETACH"
        or not state.loco_carry
    ):
        return 0
    if move.target_track not in route_blockage_plan.facts_by_blocking_track:
        return 0
    before_pressure = route_blockage_plan.total_blockage_pressure
    after_plan = compute_route_blockage_plan(plan_input, next_state, route_oracle)
    after_pressure = after_plan.total_blockage_pressure
    if after_pressure <= 0:
        return 0
    blocked_focus_tracks = _blocked_focus_tracks(
        state=next_state,
        route_oracle=route_oracle,
        focus_tracks=focus_tracks,
    )
    if not blocked_focus_tracks:
        return 0
    return max(1, after_pressure + max(0, after_pressure - before_pressure)) * max(1, focus_ttl)


def _staging_churn_penalty(
    *,
    transitions: list[tuple[ReplayState, HookAction, ReplayState]],
    state: ReplayState,
    next_state: ReplayState,
) -> int:
    penalty = 0
    for current_state, step, after_state in transitions:
        if step.action_type != "DETACH":
            continue
        if current_state.loco_track_name not in STAGING_TRACKS:
            continue
        if step.source_track not in STAGING_TRACKS and step.target_track not in STAGING_TRACKS:
            continue
        penalty += 2
        if step.source_track in STAGING_TRACKS and step.target_track in STAGING_TRACKS:
            penalty += 4
    if state.loco_track_name in STAGING_TRACKS and next_state.loco_track_name in STAGING_TRACKS:
        penalty += 1
    return penalty


def _structural_candidate_owns_staging(candidate: MoveCandidate) -> bool:
    if candidate.kind != "structural":
        return False
    return (
        candidate.reason.startswith("route_release_")
        or candidate.reason.startswith("goal_frontier_")
        or candidate.reason.startswith("chain_macro_")
        or candidate.reason in {"resource_release", "work_position_source_opening", "work_position_window_repair"}
    )


def _blocked_focus_tracks(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    focus_tracks: frozenset[str] | set[str],
) -> set[str]:
    blocked: set[str] = set()
    for focus_track in sorted(focus_tracks):
        if focus_track == state.loco_track_name:
            continue
        access_result = route_oracle.validate_loco_access(
            loco_track=state.loco_track_name,
            target_track=focus_track,
            occupied_track_sequences=state.track_sequences,
            loco_node=state.loco_node,
        )
        if not access_result.is_valid:
            blocked.add(focus_track)
    return blocked


def _blocking_goal_target_bonus(
    *,
    state: ReplayState,
    move: HookAction,
    blocking_goal_targets_by_source: dict[str, set[str]],
    route_blockage_plan: RouteBlockagePlan | None = None,
    route_release_focus_tracks: frozenset[str] | set[str] | None = None,
    route_release_focus_bonus: int = 0,
) -> int:
    route_release_bonus = route_blockage_release_score(
        source_track=move.source_track,
        vehicle_nos=move.vehicle_nos,
        route_blockage_plan=route_blockage_plan,
    )
    continuation_bonus = route_release_continuation_bonus(
        state=state,
        move=move,
        focus_tracks=route_release_focus_tracks,
        focus_bonus=route_release_focus_bonus,
    )
    blocking_targets = blocking_goal_targets_by_source.get(move.source_track)
    if not blocking_targets:
        return route_release_bonus + continuation_bonus
    source_seq = state.track_sequences.get(move.source_track, [])
    if not move.vehicle_nos:
        return route_release_bonus + continuation_bonus
    if tuple(source_seq[: len(move.vehicle_nos)]) != tuple(move.vehicle_nos):
        return route_release_bonus + continuation_bonus
    if move.action_type == "ATTACH":
        return len(blocking_targets) + route_release_bonus + continuation_bonus
    if move.target_track not in blocking_targets:
        return route_release_bonus + continuation_bonus
    return len(blocking_targets) + route_release_bonus + continuation_bonus


def _carry_exposure_bonus(
    *,
    move: HookAction,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, Any],
) -> int:
    if move.action_type != "DETACH":
        return 0
    carry = list(state.loco_carry)
    tail_size = len(move.vehicle_nos)
    if not carry or not is_carried_tail_block(carry, move.vehicle_nos):
        return 0

    last_committed_index: int | None = None
    for index in range(len(carry) - 1, -1, -1):
        vehicle_no = carry[index]
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        if _is_critical_carry_vehicle(vehicle, state=state, plan_input=plan_input):
            last_committed_index = index
            break
    if last_committed_index is None:
        return 0
    tail_distance = len(carry) - 1 - last_committed_index
    if tail_distance == 0 or tail_size > tail_distance:
        return 0
    return 1


def _is_critical_carry_vehicle(
    vehicle: Any,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if getattr(vehicle, "need_weigh", False) and vehicle.vehicle_no not in state.weighed_vehicle_nos:
        return True
    if vehicle.goal.target_mode == "SPOT":
        return True
    if vehicle.goal.target_area_code not in {None, "大库:RANDOM", "大库外:RANDOM"}:
        return True
    return bool(vehicle.goal.preferred_target_tracks and not vehicle.goal.fallback_target_tracks)


def _prune_queue(
    queue: list[QueueItem],
    best_cost: dict[tuple, int],
    beam_width: int,
    *,
    plan_input: NormalizedPlanInput | None = None,
    enable_structural_diversity: bool = False,
    debug_stats: dict[str, Any] | None = None,
) -> None:
    if len(queue) <= beam_width:
        return
    ranked = sorted(queue)
    if beam_width <= 1:
        kept = ranked[:beam_width]
    else:
        kept: list[QueueItem] = []
        kept_ids: set[int] = set()
        shallow_depth = min(len(item.plan) for item in ranked)
        shallow_candidates = [
            item
            for item in ranked
            if len(item.plan) == shallow_depth
        ]
        for item in shallow_candidates[:BEAM_SHALLOW_RESERVE]:
            kept.append(item)
            kept_ids.add(id(item))
        blocker_candidates = [
            item
            for item in ranked
            if item.blocker_reserve and id(item) not in kept_ids
        ]
        if blocker_candidates:
            blocker_item = blocker_candidates[0]
            kept.append(blocker_item)
            kept_ids.add(id(blocker_item))
        structural_reserve_candidates = [
            item
            for item in ranked
            if item.structural_reserve_tracks and id(item) not in kept_ids
        ]
        kept_structural_tracks: set[str] = set()
        for item in structural_reserve_candidates:
            tracks = sorted(item.structural_reserve_tracks - kept_structural_tracks)
            if not tracks:
                continue
            kept.append(item)
            kept_ids.add(id(item))
            kept_structural_tracks.update(item.structural_reserve_tracks)
            if len(kept) >= beam_width:
                break
        if enable_structural_diversity and _has_structural_churn_pressure(ranked[:beam_width]):
            frontier_limit = max(
                beam_width,
                beam_width * BEAM_STRUCTURAL_DIVERSITY_FRONTIER_MULTIPLIER,
            )
            low_debt_candidates = [
                item
                for item in ranked[:frontier_limit]
                if id(item) not in kept_ids
            ]
            if plan_input is not None:
                for item in low_debt_candidates:
                    if item.structural_key[0] < 0:
                        item.structural_key = _full_beam_structural_key(
                            plan_input,
                            item.plan,
                            item.state,
                        )
            low_debt_candidates.sort(key=lambda item: (item.structural_key, item.priority))
            for item in low_debt_candidates[:BEAM_LOW_STRUCTURAL_DEBT_RESERVE]:
                if len(kept) >= beam_width:
                    break
                kept.append(item)
                kept_ids.add(id(item))
        for item in ranked:
            if len(kept) >= beam_width:
                break
            if id(item) in kept_ids:
                continue
            kept.append(item)
            kept_ids.add(id(item))
        kept.sort()
    kept_ids = {id(item) for item in kept}
    pruned = [item for item in ranked if id(item) not in kept_ids]
    if debug_stats is not None:
        _record_beam_prune_trace(
            debug_stats,
            beam_width=beam_width,
            ranked=ranked,
            kept=kept,
            pruned=pruned,
        )
    queue[:] = kept
    heapify(queue)
    for item in pruned:
        if best_cost.get(item.state_key) == len(item.plan):
            del best_cost[item.state_key]


def _has_structural_churn_pressure(queue: list[QueueItem]) -> bool:
    return any(
        item.structural_key[2] >= BEAM_STRUCTURAL_DIVERSITY_MIN_STAGING_TO_STAGING
        or item.structural_key[3] >= BEAM_STRUCTURAL_DIVERSITY_MIN_REPEATED_STAGING_TOUCHES
        for item in queue
    )


def _beam_structural_key(
    plan: list[HookAction],
    state: ReplayState,
    *,
    target_sequence_defect_count: int = 0,
) -> tuple[int, ...]:
    staging_hooks = 0
    staging_to_staging_hooks = 0
    repeated_staging_touches = 0
    touched: dict[str, int] = {}
    for move in plan:
        source_is_staging = move.source_track in STAGING_TRACKS
        target_is_staging = move.target_track in STAGING_TRACKS
        if source_is_staging or target_is_staging:
            staging_hooks += 1
        if source_is_staging and target_is_staging:
            staging_to_staging_hooks += 1
        for vehicle_no in move.vehicle_nos:
            touched[vehicle_no] = touched.get(vehicle_no, 0) + 1
            if (source_is_staging or target_is_staging) and touched[vehicle_no] > 2:
                repeated_staging_touches += 1
    staging_debt = sum(
        len(seq)
        for track, seq in state.track_sequences.items()
        if track in STAGING_TRACKS
    )
    max_touch = max(touched.values(), default=0)
    return (
        target_sequence_defect_count,
        staging_debt,
        staging_to_staging_hooks,
        repeated_staging_touches,
        staging_hooks,
        max_touch,
        len(plan),
    )


def _approximate_beam_structural_key(plan: list[HookAction], state: ReplayState) -> tuple[int, ...]:
    structural_key = _beam_structural_key(plan, state)
    return (-1, *structural_key[1:])


def _full_beam_structural_key(
    plan_input: NormalizedPlanInput,
    plan: list[HookAction],
    state: ReplayState,
) -> tuple[int, ...]:
    structural = compute_structural_metrics(plan_input, state)
    return _beam_structural_key(
        plan,
        state,
        target_sequence_defect_count=structural.target_sequence_defect_count,
    )


def _initialize_debug_stats(
    debug_stats: dict[str, Any] | None,
    *,
    generated_nodes: int,
) -> None:
    if debug_stats is None:
        return
    debug_stats.clear()
    debug_stats.update(
        {
            "expanded_states": 0,
            "generated_nodes": generated_nodes,
            "closed_states": 0,
            "move_generation_calls": 0,
            "candidate_moves_total": 0,
            "candidate_steps_total": 0,
            "structural_intent_candidate_count": 0,
            "structural_candidate_steps_total": 0,
            "candidate_direct_moves": 0,
            "candidate_staging_moves": 0,
            "max_candidate_moves_per_state": 0,
            "max_candidate_steps_per_state": 0,
            "states_with_zero_moves": 0,
            "moves_by_target": {},
            "moves_by_source": {},
            "moves_by_block_size": {},
            "candidate_steps_by_kind": {},
            "structural_candidate_origin_counts": {},
            "structural_candidate_overlap_examples": [],
            "selected_candidate_origin_counts": {},
            "selected_structural_candidate_origin_counts": {},
            "selected_candidate_step_count_by_origin": {},
            "top_expansions": [],
            "expansion_candidate_traces": [],
            "beam_prune_traces": [],
            "candidate_topk_origin_counts": {},
            "candidate_selected_origin_counts_pre_prune": {},
        }
    )


def _accumulate_move_debug_stats(
    *,
    debug_stats: dict[str, Any],
    state: ReplayState,
    plan_len: int,
    move_stats: dict[str, Any],
) -> None:
    move_count = int(move_stats.get("total_moves", 0))
    step_count = int(move_stats.get("candidate_steps_total", move_count))
    debug_stats["move_generation_calls"] += 1
    debug_stats["candidate_moves_total"] += move_count
    debug_stats["candidate_steps_total"] += step_count
    debug_stats["structural_intent_candidate_count"] += int(
        move_stats.get("structural_intent_candidate_count", 0)
    )
    debug_stats["structural_candidate_steps_total"] += int(
        move_stats.get("structural_candidate_steps_total", 0)
    )
    debug_stats["candidate_direct_moves"] += int(move_stats.get("direct_moves", 0))
    debug_stats["candidate_staging_moves"] += int(move_stats.get("staging_moves", 0))
    debug_stats["max_candidate_moves_per_state"] = max(
        debug_stats["max_candidate_moves_per_state"],
        move_count,
    )
    debug_stats["max_candidate_steps_per_state"] = max(
        debug_stats["max_candidate_steps_per_state"],
        step_count,
    )
    if move_count == 0:
        debug_stats["states_with_zero_moves"] += 1
    _merge_counter_dict(debug_stats["moves_by_target"], move_stats.get("moves_by_target", {}))
    _merge_counter_dict(debug_stats["moves_by_source"], move_stats.get("moves_by_source", {}))
    _merge_counter_dict(debug_stats["moves_by_block_size"], move_stats.get("moves_by_block_size", {}))
    _merge_counter_dict(
        debug_stats["candidate_steps_by_kind"],
        move_stats.get("candidate_steps_by_kind", {}),
    )
    _merge_counter_dict(
        debug_stats["structural_candidate_origin_counts"],
        move_stats.get("structural_candidate_origin_counts", {}),
    )
    _merge_counter_dict(
        debug_stats.setdefault("structural_candidate_intent_counts", {}),
        move_stats.get("structural_candidate_intent_counts", {}),
    )
    if move_stats.get("structural_primary_problem_signature") is not None:
        debug_stats["last_primary_problem_signature"] = list(
            move_stats.get("structural_primary_problem_signature") or []
        )
    if move_stats.get("gated_problem_set") is not None:
        debug_stats["last_gated_problem_set"] = dict(
            move_stats.get("gated_problem_set") or {}
        )
    if move_stats.get("structural_primary_problems") is not None:
        debug_stats["last_primary_problem_details"] = list(
            move_stats.get("structural_primary_problems") or []
        )
    if move_stats.get("structural_candidates_by_problem") is not None:
        debug_stats["last_structural_candidates_by_problem"] = list(
            move_stats.get("structural_candidates_by_problem") or []
        )
    _merge_counter_dict(
        debug_stats.setdefault("structural_candidate_competition_origin_counts", {}),
        move_stats.get("structural_candidate_competition_origin_counts", {}),
    )
    _merge_counter_dict(
        debug_stats.setdefault("structural_candidate_competition_intent_counts", {}),
        move_stats.get("structural_candidate_competition_intent_counts", {}),
    )
    debug_stats["structural_candidate_competition_group_count"] = (
        debug_stats.get("structural_candidate_competition_group_count", 0)
        + int(move_stats.get("structural_candidate_competition_group_count", 0))
    )
    overlap_examples = move_stats.get("structural_candidate_overlap_examples", ())
    if overlap_examples:
        debug_stats["structural_candidate_overlap_examples"].extend(overlap_examples)
    competition_examples = move_stats.get("structural_candidate_competition_examples", ())
    if competition_examples:
        existing = debug_stats.setdefault("structural_candidate_competition_examples", [])
        existing.extend(competition_examples)
        del existing[48:]
    top_expansions = debug_stats["top_expansions"]
    top_expansions.append(
        {
            "plan_len": plan_len,
            "loco_track": state.loco_track_name,
            "move_count": move_count,
            "step_count": step_count,
            "direct_moves": int(move_stats.get("direct_moves", 0)),
            "staging_moves": int(move_stats.get("staging_moves", 0)),
            "sources": move_stats.get("moves_by_source", {}),
            "targets": move_stats.get("moves_by_target", {}),
        }
    )
    top_expansions.sort(key=lambda item: item["move_count"], reverse=True)
    del top_expansions[10:]


def _merge_counter_dict(target: dict[str, int], incoming: dict[Any, int]) -> None:
    for key, value in incoming.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _record_selected_candidate_debug_stats(
    debug_stats: dict[str, Any],
    *,
    candidate: MoveCandidate,
    step_count: int,
) -> None:
    origin = candidate.origin or candidate.reason or candidate.kind
    intent_type = candidate.intent_type or "UNCLASSIFIED"
    debug_stats["selected_candidate_origin_counts"][origin] = (
        debug_stats["selected_candidate_origin_counts"].get(origin, 0) + 1
    )
    selected_intent_counts = debug_stats.setdefault("selected_candidate_intent_counts", {})
    selected_intent_counts[intent_type] = selected_intent_counts.get(intent_type, 0) + 1
    debug_stats["selected_candidate_step_count_by_origin"][origin] = (
        debug_stats["selected_candidate_step_count_by_origin"].get(origin, 0) + step_count
    )
    selected_sequence = debug_stats.setdefault("selected_candidate_origin_sequence", [])
    if len(selected_sequence) < 256:
        selected_sequence.append(origin)
    selected_reason_sequence = debug_stats.setdefault("selected_candidate_reason_sequence", [])
    if len(selected_reason_sequence) < 256:
        selected_reason_sequence.append(candidate.reason or candidate.kind)
    selected_intent_sequence = debug_stats.setdefault("selected_candidate_intent_sequence", [])
    if len(selected_intent_sequence) < 256:
        selected_intent_sequence.append(
            {
                "intent_type": candidate.intent_type,
                "intent_anchor": candidate.intent_anchor,
                "intent_target": candidate.intent_target,
                "problem_type": candidate.problem_type,
                "problem_track": candidate.problem_track,
                "problem_stage": candidate.problem_stage,
            }
        )
    if candidate.kind == "structural":
        debug_stats["selected_structural_candidate_origin_counts"][origin] = (
            debug_stats["selected_structural_candidate_origin_counts"].get(origin, 0) + 1
        )


def _state_diagnostic_snapshot(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle | None,
) -> dict[str, Any]:
    purity = compute_state_purity(plan_input, state)
    structural = compute_structural_metrics(plan_input, state)
    route_pressure = (
        compute_route_blockage_plan(plan_input, state, route_oracle).total_blockage_pressure
        if route_oracle is not None
        else 0
    )
    return {
        "unfinished_count": purity.unfinished_count,
        "preferred_violation_count": purity.preferred_violation_count,
        "staging_pollution_count": purity.staging_pollution_count,
        "target_sequence_defect_count": structural.target_sequence_defect_count,
        "work_position_unfinished_count": structural.work_position_unfinished_count,
        "front_blocker_count": structural.front_blocker_count,
        "goal_track_blocker_count": structural.goal_track_blocker_count,
        "capacity_overflow_track_count": structural.capacity_overflow_track_count,
        "carry_count": len(state.loco_carry),
        "route_blockage_pressure": route_pressure,
    }


def _diagnostic_delta(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "unfinished_count",
        "preferred_violation_count",
        "staging_pollution_count",
        "target_sequence_defect_count",
        "work_position_unfinished_count",
        "front_blocker_count",
        "goal_track_blocker_count",
        "capacity_overflow_track_count",
        "carry_count",
        "route_blockage_pressure",
    )
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in keys
    }


def _record_expansion_candidate_trace(
    debug_stats: dict[str, Any],
    *,
    plan_input: NormalizedPlanInput,
    current_plan_len: int,
    current_state: ReplayState,
    branch_records: list[CandidateBranchRecord],
    route_oracle: RouteOracle | None,
) -> None:
    traces = debug_stats.setdefault("expansion_candidate_traces", [])
    if len(traces) >= 48 or not branch_records:
        return
    before_snapshot = _state_diagnostic_snapshot(
        plan_input=plan_input,
        state=current_state,
        route_oracle=route_oracle,
    )
    topk_counter = debug_stats.setdefault("candidate_topk_origin_counts", {})
    for record in branch_records[:5]:
        origin = record.candidate.origin or record.candidate.reason or record.candidate.kind
        topk_counter[origin] = topk_counter.get(origin, 0) + 1
    traces.append(
        {
            "plan_len": current_plan_len,
            "loco_track": current_state.loco_track_name,
            "candidate_count": len(branch_records),
            "primary_problem_signature": list(
                debug_stats.get("last_primary_problem_signature") or []
            ),
            "gated_problem_set": dict(
                debug_stats.get("last_gated_problem_set") or {}
            ),
            "primary_problem_details": list(
                debug_stats.get("last_primary_problem_details") or []
            ),
            "problem_candidate_rows": list(
                debug_stats.get("last_structural_candidates_by_problem") or []
            ),
            "before_state": before_snapshot,
            "top_candidates": [
                {
                    "origin": (
                        record.candidate.origin
                        or record.candidate.reason
                        or record.candidate.kind
                    ),
                    "kind": record.candidate.kind,
                    "reason": record.candidate.reason,
                    "focus_tracks": list(record.candidate.focus_tracks),
                    "intent_type": record.candidate.intent_type,
                    "intent_anchor": record.candidate.intent_anchor,
                    "intent_target": record.candidate.intent_target,
                    "staging_mode": record.candidate.staging_mode,
                    "problem_type": record.candidate.problem_type,
                    "problem_track": record.candidate.problem_track,
                    "problem_stage": record.candidate.problem_stage,
                    "step_count": len(record.applied_candidate.steps),
                    "heuristic": record.heuristic,
                    "blocker_bonus": record.scoring.blocker_bonus,
                    "structural_progress_bonus": record.scoring.structural_progress_bonus,
                    "intent_continuation_bonus": (
                        record.scoring.intent_continuation_bonus
                    ),
                    "intent_switch_penalty": record.scoring.intent_switch_penalty,
                    "route_release_regression_penalty": (
                        record.scoring.route_release_regression_penalty
                    ),
                    "staging_churn_penalty": record.scoring.staging_churn_penalty,
                    "carry_fragmentation_penalty": (
                        record.scoring.carry_fragmentation_penalty
                    ),
                    "carry_growth_penalty": record.scoring.carry_growth_penalty,
                    "soft_penalty": record.scoring.soft_penalty,
                    "after_state": (
                        after_snapshot := _state_diagnostic_snapshot(
                            plan_input=plan_input,
                            state=record.applied_candidate.final_state,
                            route_oracle=route_oracle,
                        )
                    ),
                    "delta": _diagnostic_delta(
                        before=before_snapshot,
                        after=after_snapshot,
                    ),
                }
                for record in branch_records[:5]
            ],
        }
    )


def _record_beam_prune_trace(
    debug_stats: dict[str, Any],
    *,
    beam_width: int,
    ranked: list[QueueItem],
    kept: list[QueueItem],
    pruned: list[QueueItem],
) -> None:
    traces = debug_stats.setdefault("beam_prune_traces", [])
    if len(traces) >= 48 or not pruned:
        return
    traces.append(
        {
            "beam_width": beam_width,
            "queue_size_before": len(ranked),
            "kept": [
                {
                    "origin": item.last_candidate_origin,
                    "plan_len": len(item.plan),
                    "priority": list(item.priority),
                    "blocker_reserve": item.blocker_reserve,
                    "structural_reserve_tracks": list(item.structural_reserve_tracks),
                }
                for item in kept[:8]
            ],
            "pruned": [
                {
                    "origin": item.last_candidate_origin,
                    "plan_len": len(item.plan),
                    "priority": list(item.priority),
                    "blocker_reserve": item.blocker_reserve,
                    "structural_reserve_tracks": list(item.structural_reserve_tracks),
                }
                for item in pruned[:8]
            ],
        }
    )
