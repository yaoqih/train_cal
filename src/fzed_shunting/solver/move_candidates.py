from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import (
    WORK_POSITION_TRACKS,
    is_work_position_track,
    preview_work_positions_after_prepend,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.candidate_compiler import replay_candidate_steps
from fzed_shunting.solver.debt_chain import DebtChainSummary, summarize_debt_chains
from fzed_shunting.solver.debt_graph import DebtGraphView, build_debt_graph
from fzed_shunting.solver.exact_spot import (
    exact_spot_clearance_bonus,
    exact_spot_seeker_exposure_bonus,
)
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import generate_real_hook_moves
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.route_blockage import (
    RouteBlockagePlan,
    compute_route_blockage_plan,
    route_blockage_release_score,
)
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.solver.structural_intent import (
    BufferLease,
    StructuralIntent,
    build_debt_clusters_by_track,
    build_structural_intent,
)
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


STRICT_WORK_POSITION_KINDS = frozenset(
    {"SPOTTING", "EXACT_NORTH_RANK", "EXACT_WORK_SLOT"}
)

STRUCTURAL_PRIORITY_REASON_GROUPS = {
    "route_release_frontier": 0,
    "goal_frontier_source_opening": 1,
    "resource_release": 1,
    "capacity_release": 1,
}

INTENT_PRIORITY = {
    "ADVANCE_WORK_SLOT": 0,
    "ADVANCE_ORDER": 1,
    "DELIVER": 2,
    "OPEN_GOAL_ACCESS": 3,
    "OPEN_SOURCE_ACCESS": 4,
    "RELIEVE_CAPACITY": 5,
    "BREAK_CHAIN": 6,
    "BORROW_BUFFER": 7,
    "RESTORE": 8,
}

PROBLEM_TO_INTENT_TYPE = {
    "DIRECT_GOAL_WINDOW": "OPEN_GOAL_ACCESS",
    "BLOCKER_RELEASE": "OPEN_SOURCE_ACCESS",
    "SEQUENCE_ADVANCE": "ADVANCE_ORDER",
    "CAPACITY_RELEASE": "RELIEVE_CAPACITY",
}


@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]
    kind: str = "primitive"
    reason: str = ""
    focus_tracks: tuple[str, ...] = ()
    intent_type: str = ""
    intent_anchor: str = ""
    intent_target: str = ""
    staging_mode: str = "NONE"
    problem_type: str = ""
    problem_track: str = ""
    problem_stage: str = ""
    structural_reserve: bool = False
    soft_penalty: int = 0
    origin: str = ""


@dataclass(frozen=True)
class ProblemDescriptor:
    problem_type: str
    track_name: str
    pressure: float
    anchor_track: str = ""
    resource_kind: str = ""
    goal_track: str = ""
    blocked_track: str = ""
    progress_value: float = 0.0
    legal_risk: float = 0.0
    needs_staging_only: bool = False
    source_family: str = ""


@dataclass(frozen=True)
class GatedProblemSet:
    primary_problem: ProblemDescriptor | None
    secondary_problem: ProblemDescriptor | None = None


@dataclass(frozen=True)
class IntentRequest:
    intent_type: str
    anchor_track: str
    priority: float
    business_target: str = ""
    source_problem_type: str = ""
    source_problem_track: str = ""
    source_problem_stage: str = ""


StructuralProblem = ProblemDescriptor


def _structural_candidate(
    *,
    steps: tuple[HookAction, ...] | list[HookAction],
    reason: str,
    focus_tracks: tuple[str, ...],
    problem_type: str = "",
    problem_track: str = "",
    problem_stage: str = "",
    structural_reserve: bool = True,
    origin: str = "",
) -> MoveCandidate:
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason=reason,
        focus_tracks=focus_tracks,
        intent_type=problem_type_to_intent_type(problem_type),
        intent_anchor=problem_track,
        problem_type=problem_type,
        problem_track=problem_track,
        problem_stage=problem_stage,
        structural_reserve=structural_reserve,
        soft_penalty=0,
        origin=origin or reason,
    )


def _with_candidate_origin(candidate: MoveCandidate | None, origin: str) -> MoveCandidate | None:
    if candidate is None:
        return None
    return replace(candidate, origin=origin)


def _replay_noncarry_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    steps: tuple[HookAction, ...] | list[HookAction],
    route_oracle: RouteOracle,
) -> Any | None:
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=list(steps),
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    return compiled


def generate_move_candidates(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
    debug_stats: dict[str, Any] | None = None,
    active_problem_type: str = "",
    active_problem_track: str = "",
) -> list[MoveCandidate]:
    primitive_debug: dict[str, Any] | None = {} if debug_stats is not None else None
    primitive_moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
        debug_stats=primitive_debug,
    )
    intent = build_structural_intent(
        plan_input,
        state,
        route_oracle=route_oracle,
    )
    debt_graph = build_debt_graph(plan_input, state, route_oracle=route_oracle, intent=intent)
    debt_chain_summary = debt_graph.chain_summary
    diagnosed_problems = diagnose_problems(
        intent=intent,
        plan_input=plan_input,
        state=state,
        debt_chain_summary=debt_chain_summary,
    )
    gated_problem_set = gate_problem_candidates(
        diagnosed_problems,
        active_problem_type=active_problem_type,
        active_problem_track=active_problem_track,
    )
    intent_requests = enumerate_intent_requests(
        diagnosed_problems,
        active_problem_type=active_problem_type,
        active_problem_track=active_problem_track,
    )
    protected_rejected_count = 0
    protected_allowed_by_debt_count = 0
    mixed_resource_primitive_rejected_count = 0
    delayed_staging_churn_rejected_count = 0
    filtered_primitive_moves: list[HookAction] = []
    for move in primitive_moves:
        if _move_mixes_resource_debt_groups(
            move,
            intent=intent,
            state=state,
            plan_input=plan_input,
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles},
        ):
            mixed_resource_primitive_rejected_count += 1
            continue
        if _move_is_unowned_staging_churn(
            move,
            intent=intent,
            state=state,
            plan_input=plan_input,
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles},
        ):
            delayed_staging_churn_rejected_count += 1
            continue
        if _move_attaches_unowned_delayed_staging_buffer(move, intent=intent):
            delayed_staging_churn_rejected_count += 1
            continue
        if _move_breaks_protected_commitment(move, intent=intent):
            if _move_is_required_by_resource_debt(move, intent=intent):
                protected_allowed_by_debt_count += 1
                filtered_primitive_moves.append(move)
            else:
                protected_rejected_count += 1
            continue
        filtered_primitive_moves.append(move)
    structural_candidates = _generate_structural_candidates(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        intent=intent,
        debt_graph=debt_graph,
        debt_chain_summary=debt_chain_summary,
        diagnosed_problems=diagnosed_problems,
        intent_requests=intent_requests,
    )
    primitive_candidates = _build_problem_primitive_candidates(
        filtered_primitive_moves,
        intent_requests=intent_requests,
        gated_problem_set=gated_problem_set,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )
    candidate_pool = structural_candidates + primitive_candidates
    trimmed_candidates = sorted(
        (
            trimmed
            for candidate in candidate_pool
            if (
                trimmed := _trim_candidate_before_delayed_commitment(
                    candidate,
                    vehicle_by_no={
                        vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles
                    },
                    intent=intent,
                )
            )
            is not None
        ),
        key=lambda candidate: _candidate_search_sort_key(
            candidate,
            debt_chain_summary=debt_chain_summary,
            intent=intent,
        ),
    )
    all_candidates = _dedup_candidates(tuple(trimmed_candidates))
    if debug_stats is not None:
        debug_stats.clear()
        debug_stats.update(primitive_debug or {})
        primary_problems = diagnosed_problems
        debug_stats["primitive_move_count"] = len(primitive_moves)
        debug_stats["gated_primitive_move_count"] = len(primitive_candidates)
        debug_stats["protected_primitive_rejected_count"] = protected_rejected_count
        debug_stats["protected_primitive_allowed_by_debt_count"] = protected_allowed_by_debt_count
        debug_stats["mixed_resource_primitive_rejected_count"] = (
            mixed_resource_primitive_rejected_count
        )
        debug_stats["delayed_staging_churn_rejected_count"] = (
            delayed_staging_churn_rejected_count
        )
        debug_stats["structural_intent_candidate_count"] = len(structural_candidates)
        debug_stats["structural_debt_cluster_count"] = len(intent.debt_clusters_by_track)
        debug_stats["structural_debt_chain_count"] = debt_chain_summary.chain_count
        debug_stats["structural_debt_chain_max_pressure"] = (
            debt_chain_summary.max_chain_pressure
        )
        debug_stats["structural_debt_graph_multi_track_component_count"] = (
            debt_graph.multi_track_component_count
        )
        debug_stats["structural_debt_graph_max_multi_track_pressure"] = (
            debt_graph.max_multi_track_pressure
        )
        debug_stats["structural_primary_problems"] = [
            {
                "problem_type": problem.problem_type,
                "track_name": problem.track_name,
                "pressure": problem.pressure,
                "anchor_track": problem.anchor_track,
                "resource_kind": problem.resource_kind,
                "goal_track": problem.goal_track,
                "blocked_track": problem.blocked_track,
                "progress_value": problem.progress_value,
                "legal_risk": problem.legal_risk,
                "needs_staging_only": problem.needs_staging_only,
            }
            for problem in primary_problems
        ]
        debug_stats["structural_primary_problem_signature"] = [
            f"{problem.problem_type}:{problem.track_name}:{problem.resource_kind or '-'}"
            for problem in primary_problems
        ]
        debug_stats["gated_problem_set"] = {
            "primary": (
                f"{gated_problem_set.primary_problem.problem_type}:"
                f"{gated_problem_set.primary_problem.track_name}:"
                f"{gated_problem_set.primary_problem.resource_kind or '-'}"
                if gated_problem_set.primary_problem is not None
                else None
            ),
            "secondary": (
                f"{gated_problem_set.secondary_problem.problem_type}:"
                f"{gated_problem_set.secondary_problem.track_name}:"
                f"{gated_problem_set.secondary_problem.resource_kind or '-'}"
                if gated_problem_set.secondary_problem is not None
                else None
            ),
        }
        debug_stats["gated_problem_state"] = {
            "primary_problem_type": (
                gated_problem_set.primary_problem.problem_type
                if gated_problem_set.primary_problem is not None
                else ""
            ),
            "primary_problem_track": (
                gated_problem_set.primary_problem.track_name
                if gated_problem_set.primary_problem is not None
                else ""
            ),
            "secondary_problem_type": (
                gated_problem_set.secondary_problem.problem_type
                if gated_problem_set.secondary_problem is not None
                else ""
            ),
            "secondary_problem_track": (
                gated_problem_set.secondary_problem.track_name
                if gated_problem_set.secondary_problem is not None
                else ""
            ),
        }
        debug_stats["structural_candidate_steps_total"] = sum(
            len(candidate.steps) for candidate in structural_candidates
        )
        debug_stats["structural_candidate_origin_counts"] = _candidate_origin_counts(
            structural_candidates
        )
        debug_stats["structural_candidate_intent_counts"] = _candidate_intent_counts(
            structural_candidates
        )
        debug_stats["intent_requests"] = [
            {
                "intent_type": request.intent_type,
                "anchor_track": request.anchor_track,
                "business_target": request.business_target,
                "priority": request.priority,
                "source_problem_type": request.source_problem_type,
            }
            for request in intent_requests
        ]
        debug_stats["structural_candidates_by_problem"] = _structural_candidates_by_problem(
            structural_candidates,
            primary_problems=primary_problems,
        )
        debug_stats["task_bundles"] = []
        debug_stats["structural_candidate_overlap_examples"] = _candidate_overlap_examples(
            structural_candidates
        )
        debug_stats["structural_candidate_competition_examples"] = (
            _structural_candidate_competition_examples(structural_candidates)
        )
        debug_stats["structural_candidate_competition_group_count"] = (
            _structural_candidate_competition_group_count(structural_candidates)
        )
        debug_stats["structural_candidate_competition_origin_counts"] = (
            _structural_candidate_competition_origin_counts(structural_candidates)
        )
        debug_stats["structural_candidate_competition_intent_counts"] = (
            _structural_candidate_competition_intent_counts(structural_candidates)
        )
        debug_stats["total_moves"] = len(all_candidates)
        debug_stats["total_candidates"] = len(all_candidates)
        debug_stats["candidate_steps_total"] = sum(
            len(candidate.steps) for candidate in all_candidates
        )
        debug_stats["candidate_steps_by_kind"] = _candidate_steps_by_kind(all_candidates)
        debug_stats["candidate_intent_counts"] = _candidate_intent_counts(all_candidates)
    return list(all_candidates)


def _trim_candidate_before_delayed_commitment(
    candidate: MoveCandidate,
    *,
    vehicle_by_no: dict[str, NormalizedVehicle],
    intent: StructuralIntent,
) -> MoveCandidate | None:
    if not intent.delayed_commitments:
        return candidate
    if candidate.origin == "route_release_frontier":
        return candidate
    if _work_position_free_fill_advances_delayed_order_prefix(
        candidate,
        intent=intent,
    ):
        return candidate
    delayed_targets = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }
    kept_steps: list[HookAction] = []
    pending_attach: HookAction | None = None
    for step in candidate.steps:
        if step.action_type == "ATTACH":
            if (
                candidate.kind == "structural"
                and not candidate.reason.startswith("work_position_")
                and step.source_track in STAGING_TRACKS
                and any(
                    (vehicle_no, delayed.target_track) in delayed_targets
                    for vehicle_no in step.vehicle_nos
                    for delayed in intent.delayed_commitments
                    if delayed.vehicle_no == vehicle_no
                )
                and not _move_is_required_by_resource_debt(step, intent=intent)
            ):
                return None
            kept_steps.append(step)
            pending_attach = step
            continue
        delayed_hit = False
        strict_delayed_hit = False
        for vehicle_no in step.vehicle_nos:
            if (vehicle_no, step.target_track) in delayed_targets:
                delayed_hit = True
                vehicle = vehicle_by_no.get(vehicle_no)
                if (
                    vehicle is not None
                    and vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS
                ):
                    strict_delayed_hit = True
        if delayed_hit and strict_delayed_hit and candidate.reason.startswith("work_position_"):
            kept_steps.append(step)
            pending_attach = None
            continue
        if delayed_hit:
            if pending_attach is not None:
                kept_steps = kept_steps[:-1]
            if not kept_steps or candidate.kind != "structural":
                return None
            return MoveCandidate(
                steps=tuple(kept_steps),
                kind=candidate.kind,
                reason=candidate.reason,
                focus_tracks=candidate.focus_tracks,
                structural_reserve=candidate.structural_reserve,
                soft_penalty=candidate.soft_penalty,
            )
        kept_steps.append(step)
        pending_attach = None
    return candidate


def _work_position_free_fill_advances_delayed_order_prefix(
    candidate: MoveCandidate,
    *,
    intent: StructuralIntent,
) -> bool:
    if candidate.reason != "work_position_free_fill":
        return False
    delayed_targets = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }
    detach_steps = [step for step in candidate.steps if step.action_type == "DETACH"]
    if not detach_steps:
        return False
    target_tracks = {step.target_track for step in detach_steps}
    if len(target_tracks) != 1:
        return False
    target_track = next(iter(target_tracks))
    order_debt = intent.order_debts_by_track.get(target_track)
    if order_debt is None or not order_debt.pending_vehicle_nos:
        return False
    moved_vehicle_nos = [
        vehicle_no
        for step in detach_steps
        for vehicle_no in step.vehicle_nos
    ]
    if not moved_vehicle_nos:
        return False
    if len(moved_vehicle_nos) <= 1:
        return False
    if any((vehicle_no, target_track) not in delayed_targets for vehicle_no in moved_vehicle_nos):
        return False
    return tuple(moved_vehicle_nos) == order_debt.pending_vehicle_nos[: len(moved_vehicle_nos)]


def _primitive_move_soft_penalty(
    move: HookAction,
    *,
    intent: StructuralIntent,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    penalty = 0
    if _move_breaks_protected_commitment(move, intent=intent) and not _move_is_required_by_resource_debt(move, intent=intent):
        penalty += 18
    if _move_mixes_resource_debt_groups(
        move,
        intent=intent,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        penalty += 14
    if _move_is_unowned_staging_churn(
        move,
        intent=intent,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        penalty += 12
    if _move_attaches_unowned_delayed_staging_buffer(move, intent=intent):
        penalty += 12
    return penalty


def _move_breaks_protected_commitment(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    if not move.vehicle_nos:
        return False
    protected_source_blocks = intent.committed_blocks_by_track.get(move.source_track, ())
    protected_target_blocks = intent.committed_blocks_by_track.get(move.target_track, ())
    moved = set(move.vehicle_nos)
    if move.action_type == "ATTACH":
        return any(
            moved.intersection(block.vehicle_nos)
            for block in protected_source_blocks
        )
    if move.action_type == "DETACH":
        if move.target_track in WORK_POSITION_TRACKS and protected_target_blocks:
            return True
        return any(
            moved.intersection(block.vehicle_nos)
            for block in protected_target_blocks
        )
    return False


def _move_is_unowned_staging_churn(
    move: HookAction,
    *,
    intent: StructuralIntent,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if move.action_type != "DETACH" or not move.vehicle_nos:
        return False
    if state.loco_track_name not in STAGING_TRACKS or move.target_track not in STAGING_TRACKS:
        return False
    delayed_vehicle_nos = {
        delayed.vehicle_no
        for delayed in intent.delayed_commitments
    }
    if any(vehicle_no in delayed_vehicle_nos for vehicle_no in move.vehicle_nos):
        return True
    for vehicle_no in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if not goal_is_satisfied(
            vehicle,
            track_name=state.loco_track_name,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _move_attaches_unowned_delayed_staging_buffer(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    if move.action_type != "ATTACH" or move.source_track not in STAGING_TRACKS:
        return False
    delayed_vehicle_nos = {
        delayed.vehicle_no
        for delayed in intent.delayed_commitments
    }
    return any(vehicle_no in delayed_vehicle_nos for vehicle_no in move.vehicle_nos)


def _move_mixes_resource_debt_groups(
    move: HookAction,
    *,
    intent: StructuralIntent,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if move.action_type != "ATTACH" or len(move.vehicle_nos) <= 1:
        return False
    source_seq = list(state.track_sequences.get(move.source_track, []))
    if source_seq[: len(move.vehicle_nos)] != list(move.vehicle_nos):
        return False
    moved = list(move.vehicle_nos)
    for debt in intent.resource_debts:
        if debt.track_name != move.source_track:
            continue
        debt_prefix = list(debt.vehicle_nos)
        if not debt_prefix:
            continue
        if len(moved) <= len(debt_prefix):
            if moved != debt_prefix[: len(moved)]:
                continue
        elif moved[: len(debt_prefix)] != debt_prefix:
            continue
        groups = _resource_debt_prefix_group_keys(
            source_track=move.source_track,
            vehicle_nos=moved,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        return len(groups) > 1
    return False


def _resource_debt_prefix_group_keys(
    *,
    source_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for vehicle_no in vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            keys.append(("UNKNOWN", ""))
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            key = ("KEEP", source_track)
        else:
            target = _single_allowed_target(vehicle, source_track=source_track)
            key = ("TARGET", target) if target is not None else ("BUFFER", "")
        if not keys or keys[-1] != key:
            keys.append(key)
    return keys


def _move_is_required_by_resource_debt(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    moved = set(move.vehicle_nos)
    if not moved:
        return False
    for debt in intent.resource_debts:
        if debt.track_name != move.source_track:
            continue
        if moved.intersection(debt.vehicle_nos):
            return True
    return False


def _generate_structural_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    intent: StructuralIntent,
    debt_graph: DebtGraphView | None = None,
    debt_chain_summary: DebtChainSummary | None = None,
    allow_chain_macros: bool = True,
    diagnosed_problems: tuple[ProblemDescriptor, ...] | None = None,
    intent_requests: tuple[IntentRequest, ...] | None = None,
) -> tuple[MoveCandidate, ...]:
    if debt_graph is None:
        debt_graph = build_debt_graph(
            plan_input,
            state,
            route_oracle=route_oracle,
            intent=intent,
            chain_summary=debt_chain_summary,
        )
    debt_chain_summary = debt_graph.chain_summary
    if master is None or route_oracle is None:
        return ()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    track_by_vehicle = _vehicle_track_lookup(state)
    delayed_target_pairs = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }
    if diagnosed_problems is None:
        diagnosed_problems = diagnose_problems(
            intent=intent,
            plan_input=plan_input,
            state=state,
            debt_chain_summary=debt_chain_summary,
        )
    if intent_requests is None:
        intent_requests = enumerate_intent_requests(diagnosed_problems)
    candidates: list[MoveCandidate] = []
    clusters_by_track = {
        cluster.track_name: cluster for cluster in _ordered_debt_clusters(intent)
    }
    for request in intent_requests:
        cluster = clusters_by_track.get(request.anchor_track)
        if cluster is None:
            continue
        problem = _problem_for_intent_request(
            request,
            diagnosed_problems=diagnosed_problems,
            cluster=cluster,
        )
        if problem is None:
            continue
        candidates.extend(
            candidate
            for candidate in build_candidates_for_intent_request(
                request=request,
                problem=problem,
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                track_by_vehicle=track_by_vehicle,
                cluster=cluster,
                intent=intent,
                delayed_target_pairs=delayed_target_pairs,
            )
            if _request_accepts_candidate(request, candidate)
        )
    if allow_chain_macros:
        candidates.extend(
            _generate_chain_macro_candidates(
                plan_input=plan_input,
                state=state,
                master=master,
                route_oracle=route_oracle,
                intent=intent,
                debt_chain_summary=debt_chain_summary,
                base_candidates=tuple(candidates),
                vehicle_by_no=vehicle_by_no,
                track_by_vehicle=track_by_vehicle,
                delayed_target_pairs=delayed_target_pairs,
                allowed_tracks={
                    request.anchor_track
                    for request in intent_requests
                    if request.anchor_track
                },
            )
        )
    candidates = _select_intent_request_candidates(candidates, intent_requests=intent_requests)
    ranked_candidates = _rank_release_structural_candidates(
        _budget_release_generation_candidates(candidates)
    )
    return _select_structural_candidates(
        ranked_candidates,
        limit=_structural_candidate_limit_for_intent_requests(
            debt_graph,
            intent_requests=intent_requests,
        ),
        debt_chain_summary=debt_chain_summary,
    )


def _ordered_debt_clusters(intent: StructuralIntent):
    debt_clusters_by_track = intent.debt_clusters_by_track or build_debt_clusters_by_track(
        order_debts_by_track=intent.order_debts_by_track,
        resource_debts=intent.resource_debts,
    )
    return tuple(
        sorted(
            debt_clusters_by_track.values(),
            key=lambda cluster: (
                0 if cluster.order_debt is not None else 1,
                -cluster.pressure,
                len(cluster.resource_debts),
                cluster.track_name,
            ),
        )
    )


def diagnose_problems(
    *,
    intent: StructuralIntent,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    debt_chain_summary: DebtChainSummary,
) -> tuple[ProblemDescriptor, ...]:
    problems: list[ProblemDescriptor] = []
    clusters = _ordered_debt_clusters(intent)
    for cluster in clusters:
        cluster_pressure = float(getattr(cluster, "pressure", 0.0))
        progress_value = _cluster_progress_value(
            plan_input=plan_input,
            state=state,
            track_name=cluster.track_name,
        )
        legal_risk = _cluster_legal_risk(
            plan_input=plan_input,
            state=state,
            track_name=cluster.track_name,
        )
        if cluster.order_debt is not None:
            problems.append(
                ProblemDescriptor(
                    problem_type="SEQUENCE_ADVANCE",
                    track_name=cluster.track_name,
                    pressure=cluster_pressure + 18.0 + progress_value,
                    goal_track=cluster.track_name,
                    progress_value=progress_value,
                    legal_risk=legal_risk,
                    source_family="order_debt",
                )
            )
        for debt in cluster.resource_debts:
            debt_pressure = float(getattr(debt, "pressure", 0.0))
            if debt.kind == "ROUTE_RELEASE":
                problems.append(
                    ProblemDescriptor(
                        problem_type="DIRECT_GOAL_WINDOW",
                        track_name=cluster.track_name,
                        pressure=cluster_pressure + debt_pressure * 2.5 + progress_value,
                        resource_kind=debt.kind,
                        blocked_track=debt.track_name,
                        progress_value=progress_value,
                        legal_risk=legal_risk,
                        source_family="route_release",
                    )
                )
            elif debt.kind == "FRONT_CLEARANCE":
                problems.append(
                    ProblemDescriptor(
                        problem_type="BLOCKER_RELEASE",
                        track_name=cluster.track_name,
                        pressure=cluster_pressure + debt_pressure + progress_value,
                        resource_kind=debt.kind,
                        blocked_track=debt.track_name,
                        progress_value=progress_value,
                        legal_risk=legal_risk,
                        source_family="front_clearance",
                    )
                )
            elif debt.kind == "CAPACITY_RELEASE":
                problems.append(
                    ProblemDescriptor(
                        problem_type="CAPACITY_RELEASE",
                        track_name=cluster.track_name,
                        pressure=cluster_pressure + debt_pressure,
                        resource_kind=debt.kind,
                        blocked_track=debt.track_name,
                        legal_risk=legal_risk,
                        source_family="capacity_release",
                    )
                )
    ranked = sorted(
        problems,
        key=lambda problem: (
            0 if not problem.needs_staging_only else 1,
            -problem.pressure,
            {
                "DIRECT_GOAL_WINDOW": 0,
                "BLOCKER_RELEASE": 1,
                "SEQUENCE_ADVANCE": 2,
                "CAPACITY_RELEASE": 3,
                "STAGING_NEED": 4,
            }.get(problem.problem_type, 9),
            problem.legal_risk,
            problem.track_name,
        ),
    )
    slot_order = (
        "DIRECT_GOAL_WINDOW",
        "BLOCKER_RELEASE",
        "SEQUENCE_ADVANCE",
        "CAPACITY_RELEASE",
        "STAGING_NEED",
    )
    selected: list[ProblemDescriptor] = []
    selected_keys: set[tuple[str, str, str]] = set()
    for slot in slot_order:
        for problem in ranked:
            if problem.problem_type != slot:
                continue
            key = (problem.problem_type, problem.track_name, problem.anchor_track)
            if key in selected_keys:
                continue
            selected.append(problem)
            selected_keys.add(key)
            break
        if len(selected) >= 2:
            break
    if not selected:
        return tuple(ranked[:1])
    sequence_problem = next(
        (problem for problem in ranked if problem.problem_type == "SEQUENCE_ADVANCE"),
        None,
    )
    if sequence_problem is not None and all(
        not (
            problem.problem_type == "SEQUENCE_ADVANCE"
            and problem.track_name == sequence_problem.track_name
        )
        for problem in selected
    ):
        selected.append(sequence_problem)
    return tuple(selected[:2])


def gate_problem_candidates(
    problems: tuple[ProblemDescriptor, ...],
    *,
    active_problem_type: str = "",
    active_problem_track: str = "",
) -> GatedProblemSet:
    if not problems:
        return GatedProblemSet(primary_problem=None, secondary_problem=None)
    primary = problems[0]
    if active_problem_type and active_problem_track:
        continued = next(
            (
                problem
                for problem in problems
                if problem.problem_type == active_problem_type
                and problem.track_name == active_problem_track
            ),
            None,
        )
        if continued is not None:
            primary = continued
    secondary = None
    for candidate in problems[1:]:
        if candidate.problem_type == primary.problem_type and candidate.track_name == primary.track_name:
            continue
        secondary = candidate
        break
    if secondary is not None and secondary.problem_type == "STAGING_FALLBACK":
        secondary = None
    return GatedProblemSet(primary_problem=primary, secondary_problem=secondary)


def problem_type_to_intent_type(problem_type: str) -> str:
    return PROBLEM_TO_INTENT_TYPE.get(problem_type, "")


def enumerate_intent_requests(
    problems: tuple[ProblemDescriptor, ...],
    *,
    active_problem_type: str = "",
    active_problem_track: str = "",
) -> tuple[IntentRequest, ...]:
    if not problems:
        return ()
    requests: list[IntentRequest] = []
    seen: set[tuple[str, str, str]] = set()
    for problem in problems:
        business_target = problem.goal_track or problem.blocked_track or ""
        for intent_type, stage, bonus in _intent_requests_for_problem(problem):
            key = (intent_type, problem.track_name, business_target)
            if key in seen:
                continue
            priority = problem.pressure + bonus
            if (
                problem.problem_type == active_problem_type
                and problem.track_name == active_problem_track
            ):
                priority += 1000.0
            requests.append(
                IntentRequest(
                    intent_type=intent_type,
                    anchor_track=problem.track_name,
                    business_target=business_target,
                    priority=priority,
                    source_problem_type=problem.problem_type,
                    source_problem_track=problem.track_name,
                    source_problem_stage=stage,
                )
            )
            seen.add(key)
    requests.sort(
        key=lambda request: (
            -request.priority,
            INTENT_PRIORITY.get(request.intent_type, 99),
            request.anchor_track,
            request.business_target,
        )
    )
    return tuple(requests[:4])


def _intent_requests_for_problem(
    problem: ProblemDescriptor,
) -> tuple[tuple[str, str, float], ...]:
    if problem.problem_type == "SEQUENCE_ADVANCE":
        if is_work_position_track(problem.track_name):
            return (("ADVANCE_WORK_SLOT", "work_slot_primary", 24.0),)
        return (("ADVANCE_ORDER", "order_primary", 18.0),)
    if problem.problem_type == "DIRECT_GOAL_WINDOW":
        requests: list[tuple[str, str, float]] = [
            ("OPEN_GOAL_ACCESS", "goal_access_primary", 16.0),
            ("DELIVER", "deliver_backup", 6.0),
        ]
        if problem.pressure >= 80.0:
            requests.append(("BREAK_CHAIN", "chain_pressure_backup", 4.0))
        return tuple(requests)
    if problem.problem_type == "BLOCKER_RELEASE":
        requests: list[tuple[str, str, float]] = [
            ("OPEN_SOURCE_ACCESS", "source_access_primary", 14.0),
        ]
        if problem.needs_staging_only:
            requests.append(("BORROW_BUFFER", "staging_only_backup", 7.0))
        if is_work_position_track(problem.track_name):
            requests.append(("RESTORE", "work_slot_restore_backup", 2.0))
        return tuple(requests)
    if problem.problem_type == "CAPACITY_RELEASE":
        requests = [("RELIEVE_CAPACITY", "capacity_primary", 12.0)]
        if problem.pressure >= 70.0:
            requests.append(("BORROW_BUFFER", "capacity_buffer_backup", 3.0))
        return tuple(requests)
    intent_type = problem_type_to_intent_type(problem.problem_type)
    if not intent_type:
        return ()
    return ((intent_type, "primary", 0.0),)


def _build_problem_primitive_candidates(
    primitive_moves: list[HookAction],
    *,
    intent_requests: tuple[IntentRequest, ...],
    gated_problem_set: GatedProblemSet,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
) -> tuple[MoveCandidate, ...]:
    if not intent_requests:
        return tuple(
            MoveCandidate(steps=(move,), kind="primitive")
            for move in primitive_moves[:4]
        )
    selected: list[MoveCandidate] = []
    used_signatures: set[tuple] = set()
    problems_by_key = {
        (problem_type_to_intent_type(problem.problem_type), problem.track_name): problem
        for problem in (
            gated_problem_set.primary_problem,
            gated_problem_set.secondary_problem,
        )
        if problem is not None
    }
    for request in intent_requests:
        problem = problems_by_key.get((request.intent_type, request.anchor_track))
        if problem is None:
            continue
        matched = 0
        for move in primitive_moves:
            if not _primitive_matches_problem(
                move,
                problem=problem,
                state=state,
                plan_input=plan_input,
            ):
                continue
            signature = (
                move.action_type,
                move.source_track,
                move.target_track,
                tuple(move.vehicle_nos),
                tuple(move.path_tracks),
            )
            if signature in used_signatures:
                continue
            selected.append(
                MoveCandidate(
                    steps=(move,),
                    kind="primitive",
                    reason="primitive",
                    focus_tracks=tuple(
                        track
                        for track in (problem.track_name, problem.blocked_track)
                        if track
                    ),
                    intent_type=request.intent_type,
                    intent_anchor=request.anchor_track,
                    intent_target=request.business_target,
                    staging_mode=_candidate_staging_mode_from_steps((move,)),
                    problem_type=problem.problem_type,
                    problem_track=problem.track_name,
                    problem_stage="primitive_support",
                    soft_penalty=6,
                )
            )
            used_signatures.add(signature)
            matched += 1
            if matched >= 1:
                break
    if selected:
        used_signatures.update(
            (
                candidate.steps[0].action_type,
                candidate.steps[0].source_track,
                candidate.steps[0].target_track,
                tuple(candidate.steps[0].vehicle_nos),
                tuple(candidate.steps[0].path_tracks),
            )
            for candidate in selected
            if candidate.steps
        )
        for candidate in _build_global_signal_primitive_candidates(
            primitive_moves,
            state=state,
            plan_input=plan_input,
            route_oracle=route_oracle,
            used_signatures=used_signatures,
        ):
            selected.append(candidate)
        return tuple(selected)
    fallback = _build_global_signal_primitive_candidates(
        primitive_moves,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
        used_signatures=set(),
    )
    if fallback:
        return fallback
    return tuple(
        MoveCandidate(steps=(move,), kind="primitive", soft_penalty=10)
        for move in primitive_moves[:2]
    )


def _build_global_signal_primitive_candidates(
    primitive_moves: list[HookAction],
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle | None,
    used_signatures: set[tuple],
) -> tuple[MoveCandidate, ...]:
    if route_oracle is None:
        return ()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    scored: list[tuple[int, int, MoveCandidate]] = []
    for move in primitive_moves:
        signature = (
            move.action_type,
            move.source_track,
            move.target_track,
            tuple(move.vehicle_nos),
            tuple(move.path_tracks),
        )
        if signature in used_signatures:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=[move],
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        signal = exact_spot_clearance_bonus(
            plan_input=plan_input,
            state=state,
            move=move,
            next_state=compiled.final_state,
        ) + exact_spot_seeker_exposure_bonus(
            plan_input=plan_input,
            state=state,
            move=move,
            next_state=compiled.final_state,
        )
        if signal <= 0:
            continue
        scored.append(
            (
                -signal,
                -len(move.vehicle_nos),
                MoveCandidate(
                    steps=(move,),
                    kind="primitive",
                    reason="primitive",
                    focus_tracks=tuple(
                        track for track in (move.source_track, move.target_track) if track
                    ),
                    problem_stage="global_exact_spot_signal",
                    soft_penalty=2,
                ),
            )
        )
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2].focus_tracks,
        )
    )
    return tuple(candidate for *_prefix, candidate in scored[:2])


def _primitive_matches_problem(
    move: HookAction,
    *,
    problem: ProblemDescriptor,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    if problem.problem_type == "DIRECT_GOAL_WINDOW":
        if move.source_track not in {problem.track_name, problem.blocked_track}:
            return False
        return move.target_track not in {problem.track_name, problem.blocked_track}
    if problem.problem_type == "BLOCKER_RELEASE":
        return move.source_track in {problem.track_name, problem.blocked_track}
    if problem.problem_type == "SEQUENCE_ADVANCE":
        return move.source_track == problem.track_name
    if problem.problem_type == "CAPACITY_RELEASE":
        return move.source_track == problem.track_name
    return move.target_track in STAGING_TRACKS or move.source_track in STAGING_TRACKS


def _request_accepts_candidate(
    request: IntentRequest,
    candidate: MoveCandidate,
) -> bool:
    origin = candidate.origin or candidate.reason
    if request.intent_type == "BORROW_BUFFER":
        return candidate.staging_mode == "USES_STAGING" or "stage" in origin or "frontier" in origin
    if request.intent_type == "RESTORE":
        return "restore" in origin or candidate.reason.startswith("work_position_")
    if request.intent_type == "BREAK_CHAIN":
        return origin.startswith("route_release_frontier") or origin.startswith("chain_macro_")
    if request.intent_type == "DELIVER":
        return origin.startswith("resource_release_direct_target") or origin.startswith("work_position_")
    if request.intent_type == "OPEN_GOAL_ACCESS":
        return not origin.startswith("chain_macro_")
    if request.intent_type == "OPEN_SOURCE_ACCESS":
        return not origin.startswith("route_release_frontier")
    return True


def _cluster_progress_value(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_name: str,
) -> float:
    sequence = list(state.track_sequences.get(track_name, ()))
    if not sequence:
        return 0.0
    matched_prefix = 0
    for vehicle_no in sequence:
        vehicle = next((v for v in plan_input.vehicles if v.vehicle_no == vehicle_no), None)
        if vehicle is None:
            break
        allowed = set(goal_effective_allowed_tracks(vehicle, state=state, plan_input=plan_input))
        if track_name not in allowed:
            break
        matched_prefix += 1
    return float(matched_prefix * 4)


def _cluster_legal_risk(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_name: str,
) -> float:
    risk = 0.0
    for vehicle_no in state.track_sequences.get(track_name, ()):
        vehicle = next((v for v in plan_input.vehicles if v.vehicle_no == vehicle_no), None)
        if vehicle is None:
            continue
        allowed = set(goal_effective_allowed_tracks(vehicle, state=state, plan_input=plan_input))
        if track_name not in allowed:
            risk += 1.0
    return risk


def build_candidates_for_problem(
    *,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    cluster: Any,
    intent: StructuralIntent,
    delayed_target_pairs: set[tuple[str, str]],
) -> tuple[MoveCandidate, ...]:
    request = IntentRequest(
        intent_type=problem_type_to_intent_type(problem.problem_type),
        anchor_track=problem.track_name,
        business_target=problem.goal_track or problem.blocked_track or "",
        priority=problem.pressure,
        source_problem_type=problem.problem_type,
        source_problem_track=problem.track_name,
    )
    return build_candidates_for_intent_request(
        request=request,
        problem=problem,
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
        cluster=cluster,
        intent=intent,
        delayed_target_pairs=delayed_target_pairs,
    )


def build_candidates_for_intent_request(
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    cluster: Any,
    intent: StructuralIntent,
    delayed_target_pairs: set[tuple[str, str]],
) -> tuple[MoveCandidate, ...]:
    if request.intent_type in {"ADVANCE_ORDER", "ADVANCE_WORK_SLOT"}:
        return _build_sequence_advance_candidates(
            request=request,
            problem=problem,
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            cluster=cluster,
        )
    if request.intent_type == "RELIEVE_CAPACITY":
        return _build_capacity_release_candidates(
            request=request,
            problem=problem,
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            cluster=cluster,
            delayed_target_pairs=delayed_target_pairs,
            intent=intent,
        )
    if request.intent_type in {"DELIVER", "OPEN_GOAL_ACCESS", "BREAK_CHAIN"}:
        return _build_direct_goal_candidates(
            request=request,
            problem=problem,
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            cluster=cluster,
            intent=intent,
            delayed_target_pairs=delayed_target_pairs,
        )
    if request.intent_type in {"OPEN_SOURCE_ACCESS", "BORROW_BUFFER", "RESTORE"}:
        return _build_blocker_clear_candidates(
            request=request,
            problem=problem,
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            cluster=cluster,
            intent=intent,
            delayed_target_pairs=delayed_target_pairs,
        )
    return ()


def _problem_for_intent_request(
    request: IntentRequest,
    *,
    diagnosed_problems: tuple[ProblemDescriptor, ...],
    cluster: Any,
) -> ProblemDescriptor | None:
    for problem in diagnosed_problems:
        if (
            problem_type_to_intent_type(problem.problem_type) == request.intent_type
            and problem.track_name == request.anchor_track
        ):
            return problem
    fallback_problem_type = {
        "DELIVER": "DIRECT_GOAL_WINDOW",
        "OPEN_GOAL_ACCESS": "DIRECT_GOAL_WINDOW",
        "BREAK_CHAIN": "DIRECT_GOAL_WINDOW",
        "OPEN_SOURCE_ACCESS": "BLOCKER_RELEASE",
        "BORROW_BUFFER": "BLOCKER_RELEASE",
        "RESTORE": "BLOCKER_RELEASE",
        "ADVANCE_ORDER": "SEQUENCE_ADVANCE",
        "ADVANCE_WORK_SLOT": "SEQUENCE_ADVANCE",
        "RELIEVE_CAPACITY": "CAPACITY_RELEASE",
    }.get(request.intent_type, "")
    if not fallback_problem_type:
        return None
    return ProblemDescriptor(
        problem_type=fallback_problem_type,
        track_name=request.anchor_track,
        pressure=request.priority,
        blocked_track=request.anchor_track,
        goal_track=request.business_target,
        source_family="intent_request",
    )


def _decorate_candidate_labels(
    candidate: MoveCandidate | None,
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    stage: str,
) -> MoveCandidate | None:
    if candidate is None:
        return None
    business_target = request.business_target or _candidate_business_target(candidate)
    return replace(
        candidate,
        intent_type=request.intent_type,
        intent_anchor=request.anchor_track or candidate.problem_track,
        intent_target=business_target,
        staging_mode=_candidate_staging_mode_from_steps(candidate.steps),
        problem_type=problem.problem_type,
        problem_track=problem.track_name,
        problem_stage=stage,
    )


def _candidate_is_same_intent(
    first: MoveCandidate | None,
    second: MoveCandidate | None,
) -> bool:
    if first is None or second is None:
        return False
    if tuple(first.focus_tracks) != tuple(second.focus_tracks):
        return False
    if first.reason != second.reason:
        return False
    first_targets = tuple(
        (step.source_track, step.target_track, tuple(step.vehicle_nos))
        for step in first.steps
        if step.action_type == "DETACH"
    )
    second_targets = tuple(
        (step.source_track, step.target_track, tuple(step.vehicle_nos))
        for step in second.steps
        if step.action_type == "DETACH"
    )
    return first_targets == second_targets


def _candidate_business_target(candidate: MoveCandidate) -> str:
    source_track = _candidate_primary_source_track(candidate)
    if source_track is None:
        return ""
    targets = _candidate_business_target_tracks(candidate, source_track=source_track)
    return targets[0] if targets else ""


def _candidate_staging_mode_from_steps(
    steps: tuple[HookAction, ...] | list[HookAction],
) -> str:
    return (
        "USES_STAGING"
        if any(
            step.target_track in STAGING_TRACKS or step.source_track in STAGING_TRACKS
            for step in steps
        )
        else "NONE"
    )


def _candidate_intent_signature(candidate: MoveCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.intent_type or problem_type_to_intent_type(candidate.problem_type),
        candidate.intent_anchor or candidate.problem_track or _candidate_primary_source_track(candidate) or "",
        candidate.intent_target or _candidate_business_target(candidate),
        candidate.staging_mode or _candidate_staging_mode_from_steps(candidate.steps),
    )


def _select_intent_request_candidates(
    candidates: list[MoveCandidate],
    *,
    intent_requests: tuple[IntentRequest, ...],
) -> list[MoveCandidate]:
    if not candidates:
        return []
    best_by_signature: dict[tuple[str, str, str, str], MoveCandidate] = {}
    backup_by_signature: dict[tuple[str, str, str, str], MoveCandidate] = {}
    for candidate in candidates:
        signature = _candidate_intent_signature(candidate)
        current = best_by_signature.get(signature)
        if current is None:
            best_by_signature[signature] = candidate
            continue
        if _prefer_intent_signature_candidate(candidate, current):
            backup_by_signature[signature] = current
            best_by_signature[signature] = candidate
        elif _prefer_intent_signature_candidate(
            candidate,
            backup_by_signature.get(signature),
        ):
            backup_by_signature[signature] = candidate
    selected: list[MoveCandidate] = []
    seen_steps: set[tuple] = set()
    for request in intent_requests:
        request_candidates: list[MoveCandidate] = []
        for staging_mode in ("NONE", "USES_STAGING"):
            signature = (
                request.intent_type,
                request.anchor_track,
                request.business_target,
                staging_mode,
            )
            current = best_by_signature.get(signature)
            backup = backup_by_signature.get(signature)
            if current is not None:
                request_candidates.append(current)
            if backup is not None:
                request_candidates.append(backup)
        if not request_candidates:
            continue
        request_candidates.sort(key=_intent_candidate_priority_key)
        primary = request_candidates[0]
        for candidate in (primary,):
            step_signature = _candidate_step_signature(candidate)
            if step_signature in seen_steps:
                continue
            seen_steps.add(step_signature)
            selected.append(candidate)
        for backup in request_candidates[1:]:
            if _candidate_same_implementation_lane(primary, backup):
                continue
            step_signature = _candidate_step_signature(backup)
            if step_signature in seen_steps:
                continue
            seen_steps.add(step_signature)
            selected.append(backup)
            break
    return selected


def _prefer_intent_signature_candidate(
    candidate: MoveCandidate | None,
    existing: MoveCandidate | None,
) -> bool:
    if candidate is None:
        return False
    if existing is None:
        return True
    return _intent_candidate_priority_key(candidate) < _intent_candidate_priority_key(existing)


def _intent_candidate_priority_key(candidate: MoveCandidate) -> tuple[Any, ...]:
    origin = candidate.origin or candidate.reason
    return (
        INTENT_PRIORITY.get(candidate.intent_type, 99),
        _release_budget_origin_priority(origin),
        0 if candidate.staging_mode == "NONE" else 1,
        0 if candidate.reason.startswith("work_position_source_opening") else 1,
        len(candidate.steps),
        candidate.soft_penalty,
        tuple(candidate.focus_tracks),
        origin,
    )


def _candidate_same_implementation_lane(
    first: MoveCandidate,
    second: MoveCandidate,
) -> bool:
    return _candidate_lane(first) == _candidate_lane(second)


def _candidate_lane(candidate: MoveCandidate) -> str:
    origin = candidate.origin or candidate.reason
    if origin.startswith("resource_release_direct_target"):
        return "direct"
    if origin.startswith("resource_release_stage_target"):
        return "stage_target"
    if origin.startswith("goal_frontier_source_opening"):
        return "frontier_source"
    if origin.startswith("route_release_frontier"):
        return "frontier_route"
    if origin.startswith("work_position_"):
        return "work_position"
    if origin.startswith("chain_macro_"):
        return "chain_macro"
    if candidate.kind == "primitive":
        return "primitive"
    return origin


def _candidate_with_problem(
    candidate: MoveCandidate | None,
    *,
    request: IntentRequest | None = None,
    problem: ProblemDescriptor,
    stage: str,
) -> MoveCandidate | None:
    if request is None:
        request = IntentRequest(
            intent_type=problem_type_to_intent_type(problem.problem_type),
            anchor_track=problem.track_name,
            business_target=problem.goal_track or problem.blocked_track or "",
            priority=problem.pressure,
            source_problem_type=problem.problem_type,
            source_problem_track=problem.track_name,
        )
    return _decorate_candidate_labels(
        candidate,
        request=request,
        problem=problem,
        stage=stage,
    )


def _build_sequence_advance_candidates(
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    cluster: Any,
) -> tuple[MoveCandidate, ...]:
    debt = cluster.order_debt
    if debt is None:
        return ()
    candidates: list[MoveCandidate] = []
    source_opening = _candidate_with_problem(
        _build_work_position_source_opening_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
        target_track=cluster.track_name,
        pending_vehicle_nos=list(debt.pending_vehicle_nos),
        blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
        ),
        request=request,
        problem=problem,
        stage="sequence_primary",
    )
    if source_opening is not None:
        _append_generation_candidate(candidates, source_opening)
    for backup in (
        _candidate_with_problem(
            _build_work_position_window_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            target_track=cluster.track_name,
            pending_vehicle_nos=list(debt.pending_vehicle_nos),
            blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
            ),
            request=request,
            problem=problem,
            stage="sequence_backup_window",
        ),
        _candidate_with_problem(
            _build_work_position_window_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            target_track=cluster.track_name,
            pending_vehicle_nos=list(debt.pending_vehicle_nos),
            blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
            front_block_only=True,
            ),
            request=request,
            problem=problem,
            stage="sequence_backup_front_window",
        ),
        _candidate_with_problem(
            _build_work_position_free_fill_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            target_track=cluster.track_name,
            pending_vehicle_nos=list(debt.pending_vehicle_nos),
            ),
            request=request,
            problem=problem,
            stage="sequence_backup_fill",
        ),
    ):
        if backup is None:
            continue
        if any(_candidate_is_same_intent(backup, existing) for existing in candidates):
            continue
        _append_generation_candidate(candidates, backup, sibling_candidates=candidates)
    return tuple(candidates[:2])


def _build_capacity_release_candidates(
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    cluster: Any,
    delayed_target_pairs: set[tuple[str, str]],
    intent: StructuralIntent,
) -> tuple[MoveCandidate, ...]:
    candidates: list[MoveCandidate] = []
    primary = _candidate_with_problem(
        _build_capacity_closure_primary_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        cluster=cluster,
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=intent.buffer_leases,
        ),
        request=request,
        problem=problem,
        stage="capacity_primary",
    )
    if primary is not None:
        _append_generation_candidate(candidates, primary)
    for debt in cluster.resource_debts:
        if debt.kind != "CAPACITY_RELEASE":
            continue
        backup = _candidate_with_problem(
            _build_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt=debt,
            delayed_target_pairs=delayed_target_pairs,
            order_debts_by_track=getattr(intent, "order_debts_by_track", None),
            buffer_leases=intent.buffer_leases,
            ),
            request=request,
            problem=problem,
            stage="capacity_backup_release",
        )
        if backup is not None and not any(
            _candidate_is_same_intent(backup, existing) for existing in candidates
        ):
            _append_generation_candidate(candidates, backup, sibling_candidates=candidates)
        break
    return tuple(_rank_release_structural_candidates(_budget_release_generation_candidates(candidates))[:2])


def _build_direct_goal_candidates(
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    cluster: Any,
    intent: StructuralIntent,
    delayed_target_pairs: set[tuple[str, str]],
) -> tuple[MoveCandidate, ...]:
    candidates: list[MoveCandidate] = []
    for debt in cluster.resource_debts:
        if debt.kind != "ROUTE_RELEASE":
            continue
        primary = _candidate_with_problem(
            _build_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt=debt,
            delayed_target_pairs=delayed_target_pairs,
            order_debts_by_track=getattr(intent, "order_debts_by_track", None),
            buffer_leases=intent.buffer_leases,
            ),
            request=request,
            problem=problem,
            stage="goal_window_primary",
        )
        primary = _extend_same_track_structural_followup_candidate(
            primary,
            followup_kind="ROUTE_RELEASE",
            request=request,
            problem=problem,
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            delayed_target_pairs=delayed_target_pairs,
            intent=intent,
        )
        if primary is not None and not any(
            _candidate_is_same_intent(primary, existing) for existing in candidates
        ):
            _append_generation_candidate(
                candidates,
                primary,
                sibling_candidates=candidates,
            )
        if not _direct_goal_primary_covers_source_opening(primary):
            source_opening = _candidate_with_problem(
                _build_goal_frontier_source_opening_candidate(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=problem.track_name,
                ),
                request=request,
                problem=problem,
                stage="goal_window_backup_frontier_source",
            )
            if source_opening is not None and not any(
                _candidate_is_same_intent(source_opening, existing) for existing in candidates
            ):
                _append_generation_candidate(
                    candidates,
                    source_opening,
                    sibling_candidates=candidates,
                )
        backup = _candidate_with_problem(
            _build_route_release_frontier_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt=debt,
            delayed_target_pairs=delayed_target_pairs,
            buffer_leases=intent.buffer_leases,
            ),
            request=request,
            problem=problem,
            stage="goal_window_backup_frontier",
        )
        if backup is not None and not any(
            _candidate_is_same_intent(backup, existing) for existing in candidates
        ):
            _append_generation_candidate(
                candidates,
                backup,
                sibling_candidates=candidates,
            )
        break
    return tuple(_rank_release_structural_candidates(_budget_release_generation_candidates(candidates))[:2])


def _build_blocker_clear_candidates(
    *,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    cluster: Any,
    intent: StructuralIntent,
    delayed_target_pairs: set[tuple[str, str]],
) -> tuple[MoveCandidate, ...]:
    candidates: list[MoveCandidate] = []
    for debt in cluster.resource_debts:
        if debt.kind != "FRONT_CLEARANCE":
            continue
        primary = _candidate_with_problem(
            _build_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt=debt,
            delayed_target_pairs=delayed_target_pairs,
            order_debts_by_track=getattr(intent, "order_debts_by_track", None),
            buffer_leases=intent.buffer_leases,
            ),
            request=request,
            problem=problem,
            stage="blocker_primary",
        )
        if primary is not None and not any(
            _candidate_is_same_intent(primary, existing) for existing in candidates
        ):
            _append_generation_candidate(
                candidates,
                primary,
                sibling_candidates=candidates,
            )
        if not _blocker_primary_covers_source_opening(primary):
            source_opening = _candidate_with_problem(
                _build_goal_frontier_source_opening_candidate(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=problem.track_name,
                ),
                request=request,
                problem=problem,
                stage="blocker_backup_frontier_source",
            )
            if source_opening is not None and not any(
                _candidate_is_same_intent(source_opening, existing) for existing in candidates
            ):
                _append_generation_candidate(
                    candidates,
                    source_opening,
                    sibling_candidates=candidates,
                )
        break
    return tuple(_rank_release_structural_candidates(_budget_release_generation_candidates(candidates))[:2])


def _structural_candidates_by_problem(
    candidates: tuple[MoveCandidate, ...],
    *,
    primary_problems: tuple[StructuralProblem, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem in primary_problems:
        problem_candidates = [
            candidate
            for candidate in candidates
            if problem.track_name in candidate.focus_tracks
        ]
        rows.append(
            {
                "problem_type": problem.problem_type,
                "track_name": problem.track_name,
                "resource_kind": problem.resource_kind,
                "candidate_count": len(problem_candidates),
                "candidate_origins": [
                    candidate.origin or candidate.reason or candidate.kind
                    for candidate in problem_candidates[:8]
                ],
                "candidate_reasons": [
                    candidate.reason
                    for candidate in problem_candidates[:8]
                ],
            }
        )
    return rows


def _build_route_release_frontier_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt: Any,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    blocker_block = list(debt.vehicle_nos)
    if not blocker_block:
        return None
    blocking_track = debt.track_name
    if list(state.track_sequences.get(blocking_track, []))[: len(blocker_block)] != blocker_block:
        return None

    frontier = _route_release_frontier_block(
        state=state,
        vehicle_by_no=vehicle_by_no,
        blocked_vehicle_nos=list(debt.blocked_vehicle_nos),
        source_tracks=list(getattr(debt, "source_tracks", ())),
        target_tracks=list(getattr(debt, "target_tracks", ())),
        route_blockage_plan=compute_route_blockage_plan(plan_input, state, route_oracle),
    )
    if frontier is None:
        return None
    source_track, target_track, frontier_block = frontier

    blocker_clear = _build_resource_release_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        source_track=blocking_track,
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=buffer_leases,
    )
    if blocker_clear is None:
        blocker_clear = _build_blocker_direct_goal_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=blocker_block,
            source_track=blocking_track,
            delayed_target_pairs=delayed_target_pairs,
        )
    if blocker_clear is None:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=blocking_track,
            block=blocker_block,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks={blocking_track, source_track, target_track},
            prefer_low_route_pressure=True,
            buffer_leases=buffer_leases,
        )
        if stage_track is None:
            return None
        blocker_steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=blocking_track,
            target_track=stage_track,
            block=blocker_block,
            action_source_track=blocking_track,
        )
        if blocker_steps is None:
            return None
    else:
        blocker_steps = list(blocker_clear.steps)

    compiled_clear = _replay_noncarry_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=blocker_steps,
        route_oracle=route_oracle,
    )
    if compiled_clear is None:
        return None

    steps = [*compiled_clear.steps]
    current_state = compiled_clear.final_state
    moved_frontier_count = 0
    used_frontier_signatures: set[tuple[str, str, tuple[str, ...]]] = set()
    while moved_frontier_count < 3:
        frontier = _route_release_frontier_block(
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            blocked_vehicle_nos=list(debt.blocked_vehicle_nos),
            source_tracks=list(getattr(debt, "source_tracks", ())),
            target_tracks=list(getattr(debt, "target_tracks", ())),
            route_blockage_plan=compute_route_blockage_plan(
                plan_input,
                current_state,
                route_oracle,
            ),
        )
        if frontier is None:
            break
        source_track, target_track, frontier_block = frontier
        frontier_signature = (source_track, target_track, tuple(frontier_block))
        if frontier_signature in used_frontier_signatures:
            break
        frontier_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=frontier_block,
            action_source_track=source_track,
        )
        if frontier_steps is None:
            break
        compiled_frontier = _replay_noncarry_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=frontier_steps,
            route_oracle=route_oracle,
        )
        if compiled_frontier is None:
            break
        steps.extend(compiled_frontier.steps)
        current_state = compiled_frontier.final_state
        used_frontier_signatures.add(frontier_signature)
        moved_frontier_count += 1
    if moved_frontier_count <= 0:
        return None
    restore_steps = _restore_committed_blocker_steps(
        plan_input=plan_input,
        state=current_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        original_track=blocking_track,
        clear_steps=list(compiled_clear.steps),
    )
    if _block_is_satisfied_on_track(
        plan_input=plan_input,
        state=current_state,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        track_name=blocking_track,
    ):
        if not restore_steps:
            return None
        compiled_restore = _replay_noncarry_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None:
            return None
        steps.extend(compiled_restore.steps)

    return _structural_candidate(
        steps=steps,
        reason="route_release_frontier",
        focus_tracks=(blocking_track, source_track, target_track),
        origin="route_release_frontier",
    )


def _build_blocker_direct_goal_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
) -> MoveCandidate | None:
    for target_track in _resource_release_target_tracks(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
    ):
        if target_track in STAGING_TRACKS:
            continue
        steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if steps is None:
            continue
        compiled = _replay_noncarry_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        return _structural_candidate(
            steps=compiled.steps,
            reason="resource_release",
            focus_tracks=(source_track,),
            origin="route_release_frontier_blocker_direct_goal",
        )
    return None


def _block_is_satisfied_on_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    track_name: str,
) -> bool:
    return bool(block) and all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in block
    )


def _restore_committed_blocker_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    original_track: str,
    clear_steps: list[HookAction],
) -> list[HookAction]:
    if not _block_is_satisfied_on_track(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        track_name=original_track,
    ):
        return []
    staging_track = None
    for step in clear_steps:
        if (
            step.action_type == "DETACH"
            and step.vehicle_nos == block
            and step.target_track != original_track
        ):
            staging_track = step.target_track
            break
    if staging_track is None:
        return []
    if list(state.track_sequences.get(staging_track, []))[: len(block)] != block:
        return []
    return _attach_detach_steps(
        state=state,
        route_oracle=route_oracle,
        source_track=staging_track,
        target_track=original_track,
        block=block,
        action_source_track=staging_track,
    ) or []


def _route_release_frontier_block(
    *,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    blocked_vehicle_nos: list[str],
    source_tracks: list[str],
    target_tracks: list[str],
    route_blockage_plan: RouteBlockagePlan | None = None,
) -> tuple[str, str, list[str]] | None:
    blocked = set(blocked_vehicle_nos)
    candidates: list[tuple[int, int, str, str, list[str]]] = []
    for source_track in source_tracks:
        source_seq = list(state.track_sequences.get(source_track, []))
        if not source_seq or source_seq[0] not in blocked:
            continue
        first_vehicle = vehicle_by_no.get(source_seq[0])
        if first_vehicle is None:
            continue
        target_track = next(
            (
                target
                for target in first_vehicle.goal.allowed_target_tracks
                if target in target_tracks and target != source_track
            ),
            None,
        )
        if target_track is None:
            continue
        block = [source_seq[0]]
        for vehicle_no in source_seq[1:]:
            if vehicle_no not in blocked:
                break
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None or target_track not in vehicle.goal.allowed_target_tracks:
                break
            block.append(vehicle_no)
        candidates.append(
            (
                route_blockage_release_score(
                    source_track=source_track,
                    vehicle_nos=block,
                    route_blockage_plan=route_blockage_plan,
                ),
                len(block),
                source_track,
                target_track,
                block,
            )
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
            item[3],
            tuple(item[4]),
        )
    )
    _score, _size, source_track, target_track, block = candidates[0]
    return source_track, target_track, block


def _select_structural_candidates(
    candidates: list[MoveCandidate],
    *,
    limit: int = 6,
    debt_chain_summary: DebtChainSummary | None = None,
) -> tuple[MoveCandidate, ...]:
    if len(candidates) <= limit:
        return tuple(candidates)

    selected: list[MoveCandidate] = []
    selected_ids: set[int] = set()
    covered_chain_slot_keys: set[tuple[tuple[str, ...], str]] = set()
    if debt_chain_summary is not None:
        chain_slot_candidates: dict[tuple[tuple[str, ...], str], MoveCandidate] = {}
        chain_metadata = {
            chain.track_names: (index, chain.anchor_track)
            for index, chain in enumerate(debt_chain_summary.chains)
        }
        for candidate in candidates:
            chain_key = _candidate_chain_key(candidate, debt_chain_summary)
            if chain_key is None:
                continue
            chain_index, anchor_track = chain_metadata.get(chain_key, (0, ""))
            slot_key = (
                chain_key,
                _structural_chain_slot(candidate, anchor_track=anchor_track),
            )
            current = chain_slot_candidates.get(slot_key)
            if current is None or _prefer_slot_candidate_over_existing(
                candidate=candidate,
                existing=current,
                chain_index=chain_index,
                anchor_track=anchor_track,
            ):
                chain_slot_candidates[slot_key] = candidate

        ordered_chain_slots: list[tuple[int, MoveCandidate]] = []
        for chain in debt_chain_summary.chains:
            slot_candidates = [
                chain_slot_candidates[(chain.track_names, slot_name)]
                for slot_name in (
                    "sequence",
                    "frontier_release",
                    "anchor_release",
                    "non_anchor_release",
                )
                if (chain.track_names, slot_name) in chain_slot_candidates
            ]
            slot_candidates.sort(
                key=lambda candidate: _structural_chain_candidate_priority(
                    candidate,
                    *_candidate_chain_metadata(candidate, debt_chain_summary),
                )
            )
            ordered_chain_slots.extend(
                (slot_index, candidate)
                for slot_index, candidate in enumerate(slot_candidates)
            )

        for _slot_index, candidate in sorted(
            ordered_chain_slots,
            key=lambda item: (
                item[0],
                -_candidate_chain_pressure(item[1], debt_chain_summary),
                _structural_chain_candidate_priority(
                    item[1],
                    *_candidate_chain_metadata(item[1], debt_chain_summary),
                ),
                tuple(item[1].focus_tracks),
                item[1].reason,
            ),
        ):
            chain_key = _candidate_chain_key(candidate, debt_chain_summary)
            if chain_key is None:
                continue
            chain_index, anchor_track = chain_metadata.get(chain_key, (0, ""))
            slot_key = (
                chain_key,
                _structural_chain_slot(candidate, anchor_track=anchor_track),
            )
            if slot_key in covered_chain_slot_keys:
                continue
            selected.append(candidate)
            selected_ids.add(id(candidate))
            covered_chain_slot_keys.add(slot_key)
            if len(selected) >= limit:
                return tuple(selected)

    covered_focus_tracks: set[tuple[str, ...]] = set()

    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        chain_slot_key = _candidate_chain_slot_key(candidate, debt_chain_summary)
        if chain_slot_key is not None and chain_slot_key in covered_chain_slot_keys:
            continue
        key = tuple(candidate.focus_tracks)
        if key in covered_focus_tracks:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if chain_slot_key is not None:
            covered_chain_slot_keys.add(chain_slot_key)
        covered_focus_tracks.add(key)
        if len(selected) >= limit:
            return tuple(selected)

    covered_reason_tracks: set[tuple[str, tuple[str, ...]]] = {
        (candidate.reason, tuple(candidate.focus_tracks))
        for candidate in selected
    }
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        chain_slot_key = _candidate_chain_slot_key(candidate, debt_chain_summary)
        if chain_slot_key is not None and chain_slot_key in covered_chain_slot_keys:
            continue
        key = (candidate.reason, tuple(candidate.focus_tracks))
        if key in covered_reason_tracks:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if chain_slot_key is not None:
            covered_chain_slot_keys.add(chain_slot_key)
        covered_reason_tracks.add(key)
        if len(selected) >= limit:
            return tuple(selected)
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        chain_slot_key = _candidate_chain_slot_key(candidate, debt_chain_summary)
        if chain_slot_key is not None and chain_slot_key in covered_chain_slot_keys:
            continue
        selected.append(candidate)
        if chain_slot_key is not None:
            covered_chain_slot_keys.add(chain_slot_key)
        if len(selected) >= limit:
            return tuple(selected)
    return tuple(selected)


def _structural_candidate_limit(debt_graph: DebtGraphView | DebtChainSummary | None) -> int:
    if debt_graph is None:
        return 6
    if isinstance(debt_graph, DebtChainSummary):
        chain_summary = debt_graph
        max_multi_track_pressure = 0.0
        multi_track_component_count = sum(1 for chain in chain_summary.chains if len(chain.track_names) > 1)
    else:
        chain_summary = debt_graph.chain_summary
        max_multi_track_pressure = debt_graph.max_multi_track_pressure
        multi_track_component_count = debt_graph.multi_track_component_count
    if chain_summary.chain_count <= 1:
        return 4
    if max_multi_track_pressure >= 120.0 or chain_summary.total_tracks >= 6:
        return 8
    if max_multi_track_pressure >= 60.0 or chain_summary.chain_count >= 3 or multi_track_component_count >= 3:
        return 7
    return 6


def _structural_candidate_limit_for_intent_requests(
    debt_graph: DebtGraphView | DebtChainSummary | None,
    *,
    intent_requests: tuple[IntentRequest, ...],
) -> int:
    if not intent_requests:
        return 0
    if len(intent_requests) <= 1:
        return 3
    return min(4, _structural_candidate_limit(debt_graph))


def _structural_chain_slot(candidate: MoveCandidate, *, anchor_track: str) -> str:
    if candidate.origin == "route_release_frontier":
        return "frontier_release"
    if candidate.reason.startswith("work_position_") or candidate.reason.startswith("chain_macro_"):
        return "sequence"
    if anchor_track and anchor_track in candidate.focus_tracks:
        return "anchor_release"
    return "non_anchor_release"


def _candidate_chain_slot_key(
    candidate: MoveCandidate,
    debt_chain_summary: DebtChainSummary | None,
) -> tuple[tuple[str, ...], str] | None:
    if debt_chain_summary is None:
        return None
    chain_key = _candidate_chain_key(candidate, debt_chain_summary)
    if chain_key is None:
        return None
    _chain_index, anchor_track = _candidate_chain_metadata(candidate, debt_chain_summary)
    return (chain_key, _structural_chain_slot(candidate, anchor_track=anchor_track))


def _candidate_chain_key(
    candidate: MoveCandidate,
    debt_chain_summary: DebtChainSummary,
) -> tuple[str, ...] | None:
    focus_tracks = set(candidate.focus_tracks)
    if not focus_tracks:
        return None
    for chain in debt_chain_summary.chains:
        if focus_tracks.intersection(chain.track_names):
            return chain.track_names
    return None


def _candidate_chain_metadata(
    candidate: MoveCandidate,
    debt_chain_summary: DebtChainSummary,
) -> tuple[int, str]:
    focus_tracks = set(candidate.focus_tracks)
    for index, chain in enumerate(debt_chain_summary.chains):
        if focus_tracks.intersection(chain.track_names):
            return index, chain.anchor_track
    return 0, ""


def _candidate_chain_pressure(
    candidate: MoveCandidate,
    debt_chain_summary: DebtChainSummary,
) -> float:
    chain_key = _candidate_chain_key(candidate, debt_chain_summary)
    if chain_key is None:
        return 0.0
    for chain in debt_chain_summary.chains:
        if chain.track_names == chain_key:
            return chain.total_pressure
    return 0.0


def _structural_chain_candidate_priority(
    candidate: MoveCandidate,
    chain_index: int,
    anchor_track: str,
) -> tuple[int, int, int, int, int, tuple[str, ...], str]:
    return (
        chain_index,
        STRUCTURAL_PRIORITY_REASON_GROUPS.get(candidate.reason, 2),
        0 if candidate.reason.startswith("chain_macro_") else 1,
        0 if candidate.reason.startswith("work_position_") else 1,
        0 if anchor_track in candidate.focus_tracks else 1,
        -len(candidate.focus_tracks),
        tuple(candidate.focus_tracks),
        candidate.reason,
    )


def _prefer_slot_candidate_over_existing(
    *,
    candidate: MoveCandidate,
    existing: MoveCandidate,
    chain_index: int,
    anchor_track: str,
) -> bool:
    candidate_is_direct_work = candidate.reason.startswith("work_position_")
    existing_is_direct_work = existing.reason.startswith("work_position_")
    candidate_is_macro = candidate.reason.startswith("chain_macro_")
    existing_is_macro = existing.reason.startswith("chain_macro_")
    if (
        candidate_is_direct_work
        and existing_is_macro
        and tuple(candidate.focus_tracks) == tuple(existing.focus_tracks)
    ):
        return True
    if (
        existing_is_direct_work
        and candidate_is_macro
        and tuple(candidate.focus_tracks) == tuple(existing.focus_tracks)
    ):
        return False
    return _structural_chain_candidate_priority(
        candidate,
        chain_index,
        anchor_track,
    ) < _structural_chain_candidate_priority(
        existing,
        chain_index,
        anchor_track,
    )


def _generate_chain_macro_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    intent: StructuralIntent,
    debt_chain_summary: DebtChainSummary,
    base_candidates: tuple[MoveCandidate, ...],
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    delayed_target_pairs: set[tuple[str, str]],
    allowed_tracks: set[str] | None = None,
) -> tuple[MoveCandidate, ...]:
    if master is None or route_oracle is None:
        return ()
    macro_candidates: list[MoveCandidate] = []
    ranked_chains = sorted(
        debt_chain_summary.chains,
        key=lambda chain: (-chain.total_pressure, -len(chain.track_names), chain.anchor_track),
    )
    for chain in ranked_chains[:3]:
        if allowed_tracks is not None and allowed_tracks:
            if not set(chain.track_names).intersection(allowed_tracks):
                continue
        chain_base_candidates = [
            candidate
            for candidate in base_candidates
            if set(candidate.focus_tracks).intersection(chain.track_names)
        ]
        if len(chain.track_names) < 3 or chain.total_pressure < 25.0:
            continue
        if not chain_base_candidates:
            continue
        seed = _best_chain_seed_candidate(base_candidates, chain)
        if seed is None:
            continue
        combined_steps = _build_chain_macro_steps(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            seed=seed,
            chain=chain,
            max_segments=4 if chain.total_pressure >= 120.0 else 3,
        )
        if combined_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=combined_steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        macro_candidates.append(
            MoveCandidate(
                steps=compiled.steps,
                kind="structural",
                reason=f"chain_macro_{chain.anchor_track}",
                focus_tracks=_macro_focus_tracks(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    chain_tracks=chain.track_names,
                    seed=seed,
                    combined_steps=compiled.steps,
                ),
                intent_type=seed.intent_type,
                intent_anchor=seed.intent_anchor,
                intent_target=seed.intent_target,
                staging_mode=_candidate_staging_mode_from_steps(compiled.steps),
                problem_type=seed.problem_type,
                problem_track=seed.problem_track,
                problem_stage=seed.problem_stage,
                structural_reserve=True,
            )
        )
    return tuple(macro_candidates)


def _build_chain_macro_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    seed: MoveCandidate,
    chain: DebtChainComponent,
    max_segments: int = 3,
) -> tuple[HookAction, ...] | None:
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=seed.steps,
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None

    combined_steps: list[HookAction] = list(seed.steps)
    used_signatures = {
        _candidate_signature(seed),
    }
    prior = seed
    current_state = compiled.final_state
    appended = 0
    while appended < max_segments - 1:
        next_intent = build_structural_intent(
            plan_input,
            current_state,
            route_oracle=route_oracle,
        )
        next_chain_summary = summarize_debt_chains(
            plan_input,
            current_state,
            intent=next_intent,
        )
        followups = _generate_structural_candidates(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            intent=next_intent,
            debt_chain_summary=next_chain_summary,
            allow_chain_macros=False,
        )
        followup = _best_chain_followup_candidate(
            followups,
            chain,
            prior,
            used_signatures=used_signatures,
        )
        if followup is None:
            break
        next_compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=followup.steps,
            route_oracle=route_oracle,
        )
        if next_compiled is None or next_compiled.final_state.loco_carry:
            break
        combined_steps.extend(followup.steps)
        used_signatures.add(_candidate_signature(followup))
        prior = followup
        current_state = next_compiled.final_state
        appended += 1

    if appended == 0:
        return None
    return tuple(combined_steps)


def _macro_focus_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    chain_tracks: tuple[str, ...],
    seed: MoveCandidate,
    combined_steps: tuple[HookAction, ...],
) -> tuple[str, ...]:
    focus_tracks = set(seed.focus_tracks)
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=combined_steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return tuple(sorted(focus_tracks))
    for step in combined_steps:
        if step.source_track in chain_tracks:
            focus_tracks.add(step.source_track)
        if step.target_track in chain_tracks:
            focus_tracks.add(step.target_track)
    next_intent = build_structural_intent(
        plan_input,
        compiled.final_state,
        route_oracle=route_oracle,
    )
    next_chain_summary = summarize_debt_chains(
        plan_input,
        compiled.final_state,
        intent=next_intent,
    )
    for next_chain in next_chain_summary.chains:
        if set(next_chain.track_names).intersection(chain_tracks):
            focus_tracks.add(next_chain.anchor_track)
            break
    return tuple(sorted(focus_tracks))


def _candidate_signature(candidate: MoveCandidate) -> tuple[str, tuple[str, ...], tuple[tuple[str, str, tuple[str, ...]], ...]]:
    return (
        candidate.reason,
        tuple(candidate.focus_tracks),
        tuple(
            (
                step.action_type,
                step.target_track,
                tuple(step.vehicle_nos),
            )
            for step in candidate.steps
        ),
    )


def _best_chain_seed_candidate(
    candidates: tuple[MoveCandidate, ...],
    chain: DebtChainComponent,
) -> MoveCandidate | None:
    chain_tracks = set(chain.track_names)
    seeded = [
        candidate
        for candidate in candidates
        if chain_tracks.intersection(candidate.focus_tracks)
    ]
    if not seeded:
        return None
    return sorted(
        seeded,
        key=lambda candidate: (
            0 if chain.anchor_track in candidate.focus_tracks else 1,
            0 if candidate.reason.startswith("chain_macro_") else 1,
            0 if candidate.reason.startswith("work_position_") else 1,
            0 if candidate.kind == "structural" else 1,
            len(candidate.steps),
            tuple(candidate.focus_tracks),
            candidate.reason,
        ),
    )[0]


def _best_chain_followup_candidate(
    candidates: tuple[MoveCandidate, ...],
    chain: DebtChainComponent | tuple[str, ...],
    seed: MoveCandidate,
    *,
    used_signatures: set[tuple[str, tuple[str, ...], tuple[tuple[str, str, tuple[str, ...]], ...]]] | None = None,
) -> MoveCandidate | None:
    if isinstance(chain, tuple):
        chain_tracks = set(chain)
        anchor_track = chain[0] if chain else ""
    else:
        chain_tracks = set(chain.track_names)
        anchor_track = chain.anchor_track
    used_signatures = used_signatures or set()
    followups = [
        candidate
        for candidate in candidates
        if chain_tracks.intersection(candidate.focus_tracks)
        and _candidate_signature(candidate) not in used_signatures
    ]
    if not followups:
        return None
    return sorted(
        followups,
        key=lambda candidate: (
            0 if anchor_track in candidate.focus_tracks else 1,
            0 if tuple(candidate.focus_tracks) != tuple(seed.focus_tracks) else 1,
            0 if candidate.reason.startswith("work_position_") else 1,
            0 if candidate.reason.startswith("chain_macro_") else 1,
            len(candidate.steps),
            tuple(candidate.focus_tracks),
            candidate.reason,
        ),
    )[0]

def _rank_release_structural_candidates(
    candidates: list[MoveCandidate],
) -> list[MoveCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            STRUCTURAL_PRIORITY_REASON_GROUPS.get(candidate.reason, 2),
            len(candidate.steps),
            0 if candidate.structural_reserve else 1,
            tuple(candidate.focus_tracks),
            candidate.reason,
        ),
    )


def _budget_release_generation_candidates(
    candidates: list[MoveCandidate],
) -> list[MoveCandidate]:
    passthrough: list[MoveCandidate] = []
    direct_slots: dict[tuple[str, str], MoveCandidate] = {}
    dispatch_slots: dict[tuple[str, tuple[str, ...]], MoveCandidate] = {}
    staging_slots: dict[str, MoveCandidate] = {}
    for candidate in candidates:
        if _is_weak_release_generation_candidate(candidate, existing_candidates=candidates):
            continue
        slot = _release_generation_slot(candidate)
        if slot is None:
            passthrough.append(candidate)
            continue
        slot_kind, slot_key = slot
        if slot_kind == "direct":
            existing = direct_slots.get(slot_key)
            if existing is None or _prefer_release_budget_candidate(candidate, existing):
                direct_slots[slot_key] = candidate
            continue
        if slot_kind == "dispatch":
            existing = dispatch_slots.get(slot_key)
            if existing is None or _prefer_release_budget_candidate(candidate, existing):
                dispatch_slots[slot_key] = candidate
            continue
        existing = staging_slots.get(slot_key)
        if existing is None or _prefer_release_budget_candidate(candidate, existing):
            staging_slots[slot_key] = candidate
    return [
        *passthrough,
        *direct_slots.values(),
        *dispatch_slots.values(),
        *staging_slots.values(),
    ]


def _is_weak_release_generation_candidate(
    candidate: MoveCandidate,
    *,
    existing_candidates: list[MoveCandidate],
) -> bool:
    origin = candidate.origin or candidate.reason
    if origin == "goal_frontier_source_opening":
        source_track = _candidate_primary_source_track(candidate)
        if source_track and any(
            _candidate_primary_source_track(existing) == source_track
            and _stronger_release_origin(existing) in {
                "resource_release_gateway_clear_direct_target",
                "resource_release_direct_target",
                "resource_release_protected_prefix",
            }
            for existing in existing_candidates
            if existing is not candidate
        ):
            return True
    if origin == "resource_release_stage_target":
        source_track = _candidate_primary_source_track(candidate)
        if source_track and any(
            _candidate_primary_source_track(existing) == source_track
            and _stronger_release_origin(existing) in {
                "resource_release_gateway_clear_direct_target",
                "resource_release_direct_target",
                "resource_release_protected_prefix",
                "goal_frontier_source_opening",
                "route_release_frontier",
            }
            for existing in existing_candidates
            if existing is not candidate
        ):
            return True
    return False


def _stronger_release_origin(candidate: MoveCandidate) -> str:
    return candidate.origin or candidate.reason or candidate.kind


def _direct_goal_primary_covers_source_opening(candidate: MoveCandidate | None) -> bool:
    if candidate is None:
        return False
    return _stronger_release_origin(candidate) in {
        "resource_release_gateway_clear_direct_target",
        "resource_release_direct_target",
        "resource_release_protected_prefix",
    }


def _blocker_primary_covers_source_opening(candidate: MoveCandidate | None) -> bool:
    if candidate is None:
        return False
    return _stronger_release_origin(candidate) in {
        "resource_release_gateway_clear_direct_target",
        "resource_release_direct_target",
        "resource_release_protected_prefix",
    }


def _release_generation_slot(
    candidate: MoveCandidate,
) -> tuple[str, Any] | None:
    origin = candidate.origin or candidate.reason
    if origin not in {
        "route_release_frontier",
        "goal_frontier_source_opening",
        "resource_release_direct_target",
        "resource_release_stage_target",
        "resource_release_dispatch",
        "resource_release_protected_prefix",
        "resource_release_carried_prefix",
    }:
        return None
    source_track = _candidate_primary_source_track(candidate)
    if source_track is None:
        return None
    target_tracks = _candidate_business_target_tracks(candidate, source_track=source_track)
    if not target_tracks:
        return None
    if all(target_track in STAGING_TRACKS for target_track in target_tracks):
        return ("staging", source_track)
    if len(target_tracks) == 1:
        return ("direct", (source_track, target_tracks[0]))
    return ("dispatch", (source_track, target_tracks))


def _prefer_release_budget_candidate(
    candidate: MoveCandidate,
    existing: MoveCandidate,
) -> bool:
    return _release_budget_candidate_key(candidate) < _release_budget_candidate_key(existing)


def _release_budget_candidate_key(candidate: MoveCandidate) -> tuple[Any, ...]:
    origin = candidate.origin or candidate.reason
    moved_vehicle_count = sum(
        len(step.vehicle_nos) for step in candidate.steps if step.action_type == "DETACH"
    )
    return (
        _release_budget_origin_priority(origin),
        -moved_vehicle_count,
        len(candidate.steps),
        0 if candidate.structural_reserve else 1,
        tuple(candidate.focus_tracks),
        origin,
    )


def _release_budget_origin_priority(origin: str) -> int:
    return {
        "resource_release_direct_target": 0,
        "resource_release_protected_prefix": 1,
        "goal_frontier_source_opening": 2,
        "route_release_frontier": 3,
        "resource_release_dispatch": 4,
        "resource_release_carried_prefix": 5,
        "resource_release_stage_target": 6,
    }.get(origin, 9)


def _candidate_primary_source_track(candidate: MoveCandidate) -> str | None:
    for step in candidate.steps:
        if step.source_track:
            return step.source_track
    return None


def _candidate_business_target_tracks(
    candidate: MoveCandidate,
    *,
    source_track: str,
) -> tuple[str, ...]:
    seen: set[str] = set()
    targets: list[str] = []
    for step in candidate.steps:
        if step.action_type != "DETACH":
            continue
        target_track = step.target_track
        if not target_track or target_track == source_track or target_track in seen:
            continue
        seen.add(target_track)
        targets.append(target_track)
    return tuple(targets)


def _build_goal_frontier_source_opening_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
) -> MoveCandidate | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if len(source_seq) < 2:
        return None
    first_unsatisfied = next(
        (
            index
            for index, vehicle_no in enumerate(source_seq)
            if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            )
        ),
        None,
    )
    if first_unsatisfied is None or first_unsatisfied <= 0:
        return None
    prefix = source_seq[:first_unsatisfied]
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in prefix
    ):
        return None
    target_track = _single_allowed_target(
        vehicle_by_no[source_seq[first_unsatisfied]],
        source_track=source_track,
    )
    if target_track is None:
        return None
    transfer_block = [source_seq[first_unsatisfied]]
    scan_index = first_unsatisfied + 1
    while scan_index < len(source_seq):
        vehicle = vehicle_by_no.get(source_seq[scan_index])
        if vehicle is None:
            break
        if _single_allowed_target(vehicle, source_track=source_track) != target_track:
            break
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            break
        transfer_block.append(source_seq[scan_index])
        scan_index += 1
    if not _prefix_group_can_detach_to_target(
        group=transfer_block,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    candidate = _build_source_prefix_split_detach_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=prefix,
        transfer_block=transfer_block,
        reason="goal_frontier_source_opening",
        focus_tracks=(source_track, target_track),
    )
    if candidate is None:
        return None
    final_state = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=candidate.steps,
        route_oracle=route_oracle,
    )
    if final_state is None:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=final_state.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=final_state.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return _with_candidate_origin(candidate, "goal_frontier_source_opening")


def _build_resource_release_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt: Any,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    order_debts_by_track: dict[str, Any] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    if debt.kind not in {"ROUTE_RELEASE", "CAPACITY_RELEASE", "FRONT_CLEARANCE", "EXACT_SPOT_RELEASE"}:
        return None
    block = list(debt.vehicle_nos)
    if not block:
        return None
    source_track = debt.track_name
    if list(state.track_sequences.get(source_track, []))[: len(block)] != block:
        return None
    variants: list[MoveCandidate] = []

    candidate = _build_resource_release_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=buffer_leases,
        allow_partial=debt.kind == "ROUTE_RELEASE" or len(block) >= 5,
        allow_carried_prefix=debt.kind != "ROUTE_RELEASE",
    )
    if candidate is not None:
        variants.append(_with_candidate_origin(candidate, "resource_release_dispatch"))
    if debt.kind == "CAPACITY_RELEASE":
        candidate = _build_protected_prefix_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=block,
            source_track=source_track,
            delayed_target_pairs=delayed_target_pairs,
            buffer_leases=buffer_leases,
        )
        if candidate is not None:
            variants.append(
                _with_candidate_origin(candidate, "resource_release_protected_prefix")
            )
    candidate = _build_resource_release_gateway_clear_direct_target_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
    )
    if candidate is not None:
        variants.append(
            _with_candidate_origin(
                candidate,
                "resource_release_gateway_clear_direct_target",
            )
        )
    target_tracks = _resource_release_target_tracks(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
        order_debts_by_track=order_debts_by_track,
    )
    staging_track = _best_staging_track(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks={source_track},
        prefer_low_route_pressure=debt.kind == "ROUTE_RELEASE",
        buffer_leases=buffer_leases if debt.kind == "ROUTE_RELEASE" else (),
    )
    for target_track in [*target_tracks, *([staging_track] if staging_track else [])]:
        if target_track is None or target_track == source_track:
            continue
        steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        variants.append(
            MoveCandidate(
                steps=compiled.steps,
                kind="structural",
                reason="resource_release",
                focus_tracks=(source_track,),
                structural_reserve=True,
                origin=(
                    "resource_release_direct_target"
                    if target_track not in STAGING_TRACKS
                    else "resource_release_stage_target"
                ),
            )
        )
    return _choose_best_resource_release_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        block=block,
        candidates=variants,
    )


def _extend_same_track_structural_followup_candidate(
    candidate: MoveCandidate | None,
    *,
    followup_kind: str,
    request: IntentRequest,
    problem: ProblemDescriptor,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    delayed_target_pairs: set[tuple[str, str]],
    intent: StructuralIntent,
) -> MoveCandidate | None:
    if candidate is None or not candidate.steps:
        return candidate
    if candidate.problem_type != "DIRECT_GOAL_WINDOW":
        return candidate
    if not _candidate_allows_direct_followup(candidate):
        return candidate
    compiled = _replay_noncarry_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=candidate.steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return candidate
    next_intent = build_structural_intent(
        plan_input,
        compiled.final_state,
        route_oracle=route_oracle,
    )
    next_debt = next(
        (
            debt
            for debt in next_intent.resource_debts
            if debt.kind == followup_kind and debt.track_name == problem.track_name
        ),
        None,
    )
    if next_debt is None:
        return candidate
    followup = _build_resource_release_candidate(
        plan_input=plan_input,
        state=compiled.final_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        debt=next_debt,
        delayed_target_pairs=delayed_target_pairs,
        order_debts_by_track=getattr(next_intent, "order_debts_by_track", None),
        buffer_leases=next_intent.buffer_leases,
    )
    followup = _decorate_candidate_labels(
        followup,
        request=request,
        problem=problem,
        stage=f"{candidate.problem_stage or request.source_problem_stage}_followup",
    )
    if followup is None or not followup.steps:
        return candidate
    if len(followup.steps) != 1 or not _candidate_allows_direct_followup(followup):
        return candidate
    combined_steps = tuple(candidate.steps) + tuple(followup.steps)
    if len(combined_steps) <= len(candidate.steps):
        return candidate
    combined = replace(
        followup,
        steps=combined_steps,
        focus_tracks=tuple(dict.fromkeys((*candidate.focus_tracks, *followup.focus_tracks))),
        structural_reserve=candidate.structural_reserve or followup.structural_reserve,
        soft_penalty=min(candidate.soft_penalty, followup.soft_penalty),
    )
    if _candidate_followup_is_meaningful(
        base_candidate=candidate,
        combined_candidate=combined,
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    ):
        return combined
    return candidate


def _candidate_followup_is_meaningful(
    *,
    base_candidate: MoveCandidate,
    combined_candidate: MoveCandidate,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle,
) -> bool:
    if len(combined_candidate.steps) <= len(base_candidate.steps):
        return False
    if any(
        step.target_track in STAGING_TRACKS or step.source_track in STAGING_TRACKS
        for step in combined_candidate.steps
    ):
        return False
    base_compiled = _replay_noncarry_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=base_candidate.steps,
        route_oracle=route_oracle,
    )
    combined_compiled = _replay_noncarry_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=combined_candidate.steps,
        route_oracle=route_oracle,
    )
    if base_compiled is None or combined_compiled is None:
        return False
    before = compute_structural_metrics(plan_input, base_compiled.final_state)
    after = compute_structural_metrics(plan_input, combined_compiled.final_state)
    if after.loco_carry_count > before.loco_carry_count:
        return False
    return (
        after.goal_track_blocker_count < before.goal_track_blocker_count
        or after.front_blocker_count < before.front_blocker_count
        or (
            after.unfinished_count + 1 < before.unfinished_count
            and after.capacity_overflow_track_count <= before.capacity_overflow_track_count
        )
    )


def _candidate_allows_direct_followup(candidate: MoveCandidate) -> bool:
    origin = candidate.origin or candidate.reason
    if origin not in {
        "resource_release_gateway_clear_direct_target",
        "resource_release_direct_target",
        "resource_release_protected_prefix",
    }:
        return False
    source_track = _candidate_primary_source_track(candidate)
    if source_track is None:
        return False
    target_tracks = _candidate_business_target_tracks(candidate, source_track=source_track)
    return bool(target_tracks) and all(target_track not in STAGING_TRACKS for target_track in target_tracks)


def _choose_best_resource_release_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    candidates: list[MoveCandidate],
) -> MoveCandidate | None:
    if not candidates:
        return None
    ranked: list[tuple[tuple[Any, ...], MoveCandidate]] = []
    for candidate in candidates:
        compiled = _replay_noncarry_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=candidate.steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        ranked.append(
            (
                _resource_release_candidate_local_key(
                    plan_input=plan_input,
                    initial_state=state,
                    final_state=compiled.final_state,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    block=block,
                    candidate=candidate,
                ),
                candidate,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _resource_release_candidate_local_key(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    final_state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    candidate: MoveCandidate,
) -> tuple[Any, ...]:
    moved_vehicle_nos = {
        vehicle_no
        for step in candidate.steps
        if step.action_type == "DETACH"
        for vehicle_no in step.vehicle_nos
    }
    moved_count = len(moved_vehicle_nos)
    satisfied_moved_count = 0
    non_staging_goal_detach_count = 0
    staging_detach_count = 0
    for step in candidate.steps:
        if step.action_type != "DETACH":
            continue
        if step.target_track in STAGING_TRACKS:
            staging_detach_count += len(step.vehicle_nos)
        else:
            for vehicle_no in step.vehicle_nos:
                vehicle = vehicle_by_no.get(vehicle_no)
                if vehicle is None:
                    continue
                if goal_is_satisfied(
                    vehicle,
                    track_name=step.target_track,
                    state=final_state,
                    plan_input=plan_input,
                ):
                    satisfied_moved_count += 1
                    non_staging_goal_detach_count += 1
    initial_unsatisfied_in_block = sum(
        1
        for vehicle_no in block
        if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=initial_state,
            plan_input=plan_input,
        )
    )
    final_source_seq = list(final_state.track_sequences.get(source_track, []))
    final_unsatisfied_prefix = 0
    for vehicle_no in final_source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            break
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=final_state,
            plan_input=plan_input,
        ):
            continue
        final_unsatisfied_prefix += 1
    origin = candidate.origin or candidate.reason or candidate.kind
    return (
        -non_staging_goal_detach_count,
        -satisfied_moved_count,
        -moved_count,
        final_unsatisfied_prefix,
        staging_detach_count,
        _resource_release_local_origin_priority(origin),
        len(candidate.steps),
        -initial_unsatisfied_in_block,
        tuple(candidate.focus_tracks),
        origin,
    )


def _resource_release_local_origin_priority(origin: str) -> int:
    return {
        "resource_release_gateway_clear_direct_target": 0,
        "resource_release_direct_target": 1,
        "resource_release_protected_prefix": 2,
        "resource_release_dispatch": 3,
        "resource_release_carried_prefix": 4,
        "resource_release_stage_target": 5,
    }.get(origin, 9)


def _build_capacity_closure_primary_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    cluster: Any,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    ranked: list[tuple[tuple[Any, ...], MoveCandidate]] = []
    for debt in cluster.resource_debts:
        if debt.kind != "CAPACITY_RELEASE":
            continue
        candidate = _build_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt=debt,
            delayed_target_pairs=delayed_target_pairs,
            buffer_leases=buffer_leases,
        )
        if candidate is None:
            continue
        compiled = _replay_noncarry_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=candidate.steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        ranked.append(
            (
                _capacity_closure_candidate_key(
                    plan_input=plan_input,
                    initial_state=state,
                    final_state=compiled.final_state,
                    vehicle_by_no=vehicle_by_no,
                    source_track=debt.track_name,
                    block=list(debt.vehicle_nos),
                    candidate=candidate,
                ),
                candidate,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    best = ranked[0][1]
    origin = best.origin or best.reason or best.kind
    if origin not in {
        "resource_release_gateway_clear_direct_target",
        "resource_release_direct_target",
        "resource_release_protected_prefix",
    }:
        return None
    return best


def _capacity_closure_candidate_key(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    final_state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    candidate: MoveCandidate,
) -> tuple[Any, ...]:
    origin = candidate.origin or candidate.reason or candidate.kind
    return (
        _capacity_closure_origin_priority(origin),
        *_resource_release_candidate_local_key(
            plan_input=plan_input,
            initial_state=initial_state,
            final_state=final_state,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            block=block,
            candidate=candidate,
        ),
    )


def _capacity_closure_origin_priority(origin: str) -> int:
    return {
        "resource_release_gateway_clear_direct_target": 0,
        "resource_release_direct_target": 1,
        "resource_release_protected_prefix": 2,
        "resource_release_dispatch": 3,
        "resource_release_carried_prefix": 4,
        "resource_release_stage_target": 5,
    }.get(origin, 9)


def _build_resource_release_gateway_clear_direct_target_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
) -> MoveCandidate | None:
    target_track = _shared_single_allowed_target(
        block=block,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
    )
    if target_track is None:
        return None
    if not _prefix_group_can_detach_to_target(
        group=block,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    source_node = _source_node_after_attach_for_block(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
    )
    path_tracks = route_oracle.resolve_path_tracks_for_endpoint_constraints(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=route_oracle.order_end_node(target_track),
    )
    if path_tracks is None:
        return None
    blocking_tracks = [
        track_name
        for track_name in path_tracks[1:-1]
        if state.track_sequences.get(track_name)
    ]
    if len(blocking_tracks) != 1:
        return None
    clearance = _build_single_gateway_blocker_clearance(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        blocker_track=blocking_tracks[0],
        forbidden_targets={source_track, target_track},
    )
    if clearance is None:
        return None
    clearance_steps, cleared_state = clearance
    direct_steps = _attach_detach_steps(
        state=cleared_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=block,
        action_source_track=source_track,
    )
    if direct_steps is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[*clearance_steps, *direct_steps],
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in block
    ):
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason="resource_release",
        focus_tracks=(source_track, target_track),
        structural_reserve=True,
        origin="resource_release_gateway_clear_direct_target",
    )


def _build_single_gateway_blocker_clearance(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    blocker_track: str,
    forbidden_targets: set[str],
) -> tuple[list[HookAction], ReplayState] | None:
    blocker_seq = list(state.track_sequences.get(blocker_track, []))
    if not blocker_seq:
        return None
    blocker_vehicle = vehicle_by_no.get(blocker_seq[0])
    if blocker_vehicle is None:
        return None
    for target_track in blocker_vehicle.goal.allowed_target_tracks:
        if target_track in forbidden_targets or target_track == blocker_track:
            continue
        steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=blocker_track,
            target_track=target_track,
            block=[blocker_seq[0]],
            action_source_track=blocker_track,
        )
        if steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        return list(compiled.steps), compiled.final_state
    return None


def _shared_single_allowed_target(
    *,
    block: list[str],
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> str | None:
    shared_target: str | None = None
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return None
        effective_targets = [
            target
            for target in goal_effective_allowed_tracks(
                vehicle,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
            )
            if target != source_track
        ]
        if len(effective_targets) != 1:
            preferred_targets = [
                target
                for target in vehicle.goal.preferred_target_tracks
                if target != source_track
            ]
            if len(preferred_targets) != 1:
                return None
            target_track = preferred_targets[0]
        else:
            target_track = effective_targets[0]
        if shared_target is None:
            shared_target = target_track
        elif target_track != shared_target:
            return None
    return shared_target


def _build_protected_prefix_resource_release_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    protected_prefix, release_suffix = _protected_prefix_resource_release_split(
        plan_input=plan_input,
        state=state,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
    )
    if not protected_prefix or not release_suffix:
        return None

    steps: list[HookAction] = []
    current_state = state
    staged_prefix: list[tuple[str, list[str]]] = []
    forbidden_tracks = {source_track}
    remaining_prefix = list(protected_prefix)
    while remaining_prefix:
        chunk_plan = _next_prefix_staging_chunk(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=source_track,
            transfer_block=release_suffix,
            remaining_prefix=remaining_prefix,
            forbidden_tracks=forbidden_tracks,
            allow_split_prefix=True,
        )
        if chunk_plan is None:
            return None
        stage_track, chunk, chunk_steps, next_state = chunk_plan
        steps.extend(chunk_steps)
        staged_prefix.append((stage_track, chunk))
        forbidden_tracks.add(stage_track)
        current_state = next_state
        remaining_prefix = remaining_prefix[len(chunk):]

    release_steps = _compile_resource_release_suffix_steps(
        plan_input=plan_input,
        state=current_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        release_suffix=release_suffix,
        delayed_target_pairs=delayed_target_pairs,
        forbidden_tracks=forbidden_tracks,
        buffer_leases=buffer_leases,
    )
    if release_steps is None:
        return None
    suffix_steps, current_state = release_steps
    steps.extend(suffix_steps)

    for stage_track, chunk in reversed(staged_prefix):
        restore_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=stage_track,
            target_track=source_track,
            block=chunk,
            action_source_track=stage_track,
        )
        if restore_steps is None:
            return None
        compiled_restore = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None or compiled_restore.final_state.loco_carry:
            return None
        steps.extend(compiled_restore.steps)
        current_state = compiled_restore.final_state

    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason="resource_release",
        focus_tracks=(source_track,),
        structural_reserve=True,
        origin="resource_release_protected_prefix",
    )


def _protected_prefix_resource_release_split(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> tuple[list[str], list[str]]:
    protected_prefix: list[str] = []
    index = 0
    while index < len(block):
        vehicle = vehicle_by_no.get(block[index])
        if vehicle is None:
            return [], []
        if not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            break
        protected_prefix.append(block[index])
        index += 1
    return protected_prefix, block[index:]


def _compile_resource_release_suffix_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    release_suffix: list[str],
    delayed_target_pairs: set[tuple[str, str]] | None,
    forbidden_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
) -> tuple[tuple[HookAction, ...], ReplayState] | None:
    groups = _resource_release_prefix_groups(
        block=release_suffix,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        route_oracle=route_oracle,
    )
    if not groups:
        return None

    steps: list[HookAction] = []
    current_state = state
    used_tracks = set(forbidden_tracks)
    for preferred_target, group in groups:
        compiled = None
        if (
            preferred_target is not None
            and preferred_target != source_track
            and not validate_hook_vehicle_group(
                [vehicle_by_no[vehicle_no] for vehicle_no in group]
            )
            and _prefix_group_can_detach_to_target(
                group=group,
                target_track=preferred_target,
                state=current_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        ):
            direct_steps = _attach_detach_steps(
                state=current_state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=preferred_target,
                block=group,
                action_source_track=source_track,
            )
            if direct_steps is not None:
                compiled = replay_candidate_steps(
                    plan_input=plan_input,
                    state=current_state,
                    vehicle_by_no=vehicle_by_no,
                    steps=direct_steps,
                    route_oracle=route_oracle,
                )
                if compiled is not None and compiled.final_state.loco_carry:
                    compiled = None
        if compiled is None:
            compiled = _compile_staging_group_with_source_reaccess(
                plan_input=plan_input,
                state=current_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                group=group,
                forbidden_staging_tracks=used_tracks,
                buffer_leases=buffer_leases,
                require_source_reaccess=False,
            )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state
        if compiled.steps:
            used_tracks.add(compiled.steps[-1].target_track)
    return tuple(steps), current_state


def _build_source_prefix_split_detach_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    reason: str,
    focus_tracks: tuple[str, ...],
) -> MoveCandidate | None:
    if not protected_prefix or not transfer_block:
        return None
    source_seq = list(state.track_sequences.get(source_track, []))
    combined_block = [*protected_prefix, *transfer_block]
    if source_seq[: len(combined_block)] != combined_block:
        return None

    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(combined_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    after_attach = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if after_attach is None:
        return None

    transfer_path = _clear_path_tracks_for_detach(
        route_oracle=route_oracle,
        state=after_attach.final_state,
        source_track=source_track,
        target_track=target_track,
    )
    if transfer_path is None:
        return None
    transfer_detach = HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(transfer_block),
        path_tracks=transfer_path,
        action_type="DETACH",
    )
    after_transfer = replay_candidate_steps(
        plan_input=plan_input,
        state=after_attach.final_state,
        vehicle_by_no=vehicle_by_no,
        steps=[transfer_detach],
        route_oracle=route_oracle,
    )
    if after_transfer is None or after_transfer.final_state.loco_carry != tuple(protected_prefix):
        return None

    restore_path = _clear_path_tracks_for_detach(
        route_oracle=route_oracle,
        state=after_transfer.final_state,
        source_track=target_track,
        target_track=source_track,
    )
    if restore_path is None:
        return None
    restore_detach = HookAction(
        source_track=target_track,
        target_track=source_track,
        vehicle_nos=list(protected_prefix),
        path_tracks=restore_path,
        action_type="DETACH",
    )
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach, transfer_detach, restore_detach],
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason=reason,
        focus_tracks=focus_tracks,
        structural_reserve=True,
    )


def _clear_path_tracks_for_detach(
    *,
    route_oracle: RouteOracle,
    state: ReplayState,
    source_track: str,
    target_track: str,
) -> list[str] | None:
    source_remaining = len(state.track_sequences.get(source_track, []))
    source_node = state.loco_node if source_remaining > 0 else None
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
    return path_tracks


def _build_resource_release_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
    allow_partial: bool = False,
    allow_carried_prefix: bool = True,
) -> MoveCandidate | None:
    groups = _resource_release_prefix_groups(
        block=block,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        route_oracle=route_oracle,
    )
    if not groups or len(groups) <= 1:
        return None
    return _build_group_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        groups=groups,
        block=block,
        source_track=source_track,
        buffer_leases=buffer_leases,
        reason="resource_release",
        focus_tracks=(source_track,),
        allow_partial=allow_partial,
        allow_carried_prefix=allow_carried_prefix,
    )


def _build_route_release_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    deferred_target_tracks: set[str],
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    groups = _resource_release_prefix_groups(
        block=block,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        deferred_target_tracks=deferred_target_tracks,
        route_oracle=route_oracle,
    )
    if not groups or len(groups) <= 1:
        return None
    return _build_group_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        groups=groups,
        block=block,
        source_track=source_track,
        buffer_leases=buffer_leases,
        reason="resource_release",
        focus_tracks=(source_track,),
        allow_partial=True,
        allow_carried_prefix=False,
    )


def _build_group_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    groups: list[tuple[str | None, list[str]]],
    block: list[str],
    source_track: str,
    buffer_leases: tuple[BufferLease, ...] = (),
    reason: str,
    focus_tracks: tuple[str, ...],
    allow_partial: bool = False,
    allow_carried_prefix: bool = True,
) -> MoveCandidate | None:
    if allow_carried_prefix:
        split_candidate = _build_carried_prefix_dispatch_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            groups=groups,
            source_track=source_track,
            buffer_leases=buffer_leases,
        )
        if split_candidate is not None:
            return MoveCandidate(
                steps=split_candidate.steps,
                kind="structural",
                reason=reason,
                focus_tracks=focus_tracks,
                structural_reserve=True,
            )

    steps: list[HookAction] = []
    current_state = state
    forbidden_staging_tracks = {source_track}
    best_prefix_candidate: MoveCandidate | None = None
    best_prefix_key: tuple[Any, ...] | None = None
    for group_index, (preferred_target, group) in enumerate(groups):
        require_source_reaccess = group_index < len(groups) - 1 or _source_tail_after_group_needs_access(
            state=current_state,
            source_track=source_track,
            group=group,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        compiled = None
        if (
            preferred_target is not None
            and preferred_target != source_track
            and _prefix_group_can_detach_to_target(
                group=group,
                target_track=preferred_target,
                state=current_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        ):
            direct_steps = _attach_detach_steps(
                state=current_state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=preferred_target,
                block=group,
                action_source_track=source_track,
            )
            if direct_steps is not None:
                compiled = replay_candidate_steps(
                    plan_input=plan_input,
                    state=current_state,
                    vehicle_by_no=vehicle_by_no,
                    steps=direct_steps,
                    route_oracle=route_oracle,
                )
                if compiled is not None and compiled.final_state.loco_carry:
                    compiled = None
                if (
                    compiled is not None
                    and require_source_reaccess
                    and not _loco_can_access_track(
                        state=compiled.final_state,
                        route_oracle=route_oracle,
                        target_track=source_track,
                    )
                ):
                    compiled = None
        if compiled is None:
            compiled = _compile_staging_group_with_source_reaccess(
                plan_input=plan_input,
                state=current_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                group=group,
                forbidden_staging_tracks=forbidden_staging_tracks,
                buffer_leases=buffer_leases,
                require_source_reaccess=require_source_reaccess,
            )
            if compiled is None:
                if allow_partial:
                    break
                return None
            stage_track = compiled.steps[-1].target_track
            forbidden_staging_tracks.add(stage_track)
        elif preferred_target in STAGING_TRACKS:
            forbidden_staging_tracks.add(preferred_target)
        steps.extend(compiled.steps)
        current_state = compiled.final_state
        if group_index >= 1:
            prefix_candidate = MoveCandidate(
                steps=tuple(steps),
                kind="structural",
                reason=reason,
                focus_tracks=focus_tracks,
                structural_reserve=True,
                origin=reason,
            )
            if compiled.steps and compiled.steps[-1].target_track not in STAGING_TRACKS:
                prefix_key = _group_dispatch_prefix_candidate_key(
                    plan_input=plan_input,
                    initial_state=state,
                    final_state=current_state,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    block=block,
                    candidate=prefix_candidate,
                )
                if best_prefix_key is None or prefix_key < best_prefix_key:
                    best_prefix_key = prefix_key
                    best_prefix_candidate = prefix_candidate
    if best_prefix_candidate is not None:
        return best_prefix_candidate
    if not steps:
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason=reason,
        focus_tracks=focus_tracks,
        structural_reserve=True,
        origin=reason,
    )


def _group_dispatch_prefix_candidate_key(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    final_state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    candidate: MoveCandidate,
) -> tuple[Any, ...]:
    staging_detach_count = 0
    direct_goal_detach_count = 0
    moved_count = 0
    distinct_targets: set[str] = set()
    for step in candidate.steps:
        if step.action_type != "DETACH":
            continue
        moved_count += len(step.vehicle_nos)
        distinct_targets.add(step.target_track)
        if step.target_track in STAGING_TRACKS:
            staging_detach_count += len(step.vehicle_nos)
            continue
        for vehicle_no in step.vehicle_nos:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            if goal_is_satisfied(
                vehicle,
                track_name=step.target_track,
                state=final_state,
                plan_input=plan_input,
            ):
                direct_goal_detach_count += 1
    final_source_seq = list(final_state.track_sequences.get(source_track, []))
    final_unsatisfied_prefix = 0
    for vehicle_no in final_source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            break
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=final_state,
            plan_input=plan_input,
        ):
            continue
        final_unsatisfied_prefix += 1
    initial_unsatisfied_in_block = sum(
        1
        for vehicle_no in block
        if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=initial_state,
            plan_input=plan_input,
        )
    )
    return (
        staging_detach_count,
        len(candidate.steps),
        len(distinct_targets),
        final_unsatisfied_prefix,
        -direct_goal_detach_count,
        -moved_count,
        -initial_unsatisfied_in_block,
    )


def _source_tail_after_group_needs_access(
    *,
    state: ReplayState,
    source_track: str,
    group: list[str],
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(group)] != list(group):
        return False
    for vehicle_no in source_seq[len(group):]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return True
        if not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _compile_staging_group_with_source_reaccess(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    group: list[str],
    forbidden_staging_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
    require_source_reaccess: bool,
) -> Any | None:
    for stage_track in _ranked_staging_tracks(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=group,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_staging_tracks,
        prefer_low_route_pressure=True,
        buffer_leases=buffer_leases,
    ):
        stage_steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=stage_track,
            block=group,
            action_source_track=source_track,
        )
        if stage_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=stage_steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        if require_source_reaccess and not _loco_can_access_track(
            state=compiled.final_state,
            route_oracle=route_oracle,
            target_track=source_track,
        ):
            continue
        return compiled
    return None


def _build_carried_prefix_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    groups: list[tuple[str | None, list[str]]],
    source_track: str,
    buffer_leases: tuple[BufferLease, ...],
) -> MoveCandidate | None:
    block = [vehicle_no for _target, group in groups for vehicle_no in group]
    if not block:
        return None
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    current = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if current is None:
        return None

    steps: list[HookAction] = [attach]
    used_staging_tracks: set[str] = {source_track}
    for preferred_target, group in reversed(groups):
        detach = _carried_group_detach_step(
            plan_input=plan_input,
            state=current.final_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            group=group,
            preferred_target=preferred_target,
            used_staging_tracks=used_staging_tracks,
            buffer_leases=buffer_leases,
        )
        if detach is None:
            return None
        if detach.target_track in STAGING_TRACKS:
            used_staging_tracks.add(detach.target_track)
        current = replay_candidate_steps(
            plan_input=plan_input,
            state=current.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[detach],
            route_oracle=route_oracle,
        )
        if current is None:
            return None
        steps.append(detach)
    if current.final_state.loco_carry:
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="resource_release",
        focus_tracks=(source_track,),
        structural_reserve=True,
        origin="resource_release_carried_prefix",
    )


def _carried_group_detach_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    group: list[str],
    preferred_target: str | None,
    used_staging_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
) -> HookAction | None:
    target_tracks: list[str] = []
    if preferred_target is not None:
        target_tracks.append(preferred_target)
    target_tracks.extend(
        track
        for track in _ranked_staging_tracks(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=state.loco_track_name,
            block=group,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=used_staging_tracks,
            prefer_low_route_pressure=True,
            buffer_leases=buffer_leases,
            block_is_carried=True,
        )
        if track not in target_tracks
    )
    for target_track in target_tracks:
        detach_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=state,
            source_track=state.loco_track_name,
            target_track=target_track,
        )
        if detach_path is None:
            continue
        return HookAction(
            source_track=state.loco_track_name,
            target_track=target_track,
            vehicle_nos=list(group),
            path_tracks=detach_path,
            action_type="DETACH",
        )
    return None


def _resource_release_prefix_groups(
    *,
    block: list[str],
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    deferred_target_tracks: set[str] | None = None,
    route_oracle: RouteOracle | None = None,
) -> list[tuple[str | None, list[str]]]:
    groups: list[tuple[str | None, list[str]]] = []
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            target_track = source_track
        else:
            target_track = _resource_release_group_target(
                vehicle=vehicle,
                vehicle_no=vehicle_no,
                source_track=source_track,
                state=state,
                plan_input=plan_input,
                delayed_target_pairs=delayed_target_pairs,
                deferred_target_tracks=deferred_target_tracks,
                route_oracle=route_oracle,
            )
        if groups and groups[-1][0] == target_track:
            groups[-1][1].append(vehicle_no)
        else:
            groups.append((target_track, [vehicle_no]))
    return groups


def _resource_release_group_target(
    *,
    vehicle: NormalizedVehicle,
    vehicle_no: str,
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    delayed_target_pairs: set[tuple[str, str]] | None,
    deferred_target_tracks: set[str] | None,
    route_oracle: RouteOracle | None,
) -> str | None:
    delayed_target_pairs = delayed_target_pairs or set()
    deferred_target_tracks = deferred_target_tracks or set()
    for target_track in goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    ):
        if target_track == source_track:
            continue
        if target_track in deferred_target_tracks:
            continue
        if (vehicle_no, target_track) in delayed_target_pairs:
            continue
        return target_track
    return None


def _block_contains_delayed_order_buffer(
    block: list[str],
    *,
    delayed_target_pairs: set[tuple[str, str]] | None,
) -> bool:
    if not delayed_target_pairs:
        return False
    delayed_vehicle_nos = {vehicle_no for vehicle_no, _target_track in delayed_target_pairs}
    return any(vehicle_no in delayed_vehicle_nos for vehicle_no in block)


def _resource_release_target_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    order_debts_by_track: dict[str, Any] | None = None,
) -> list[str]:
    delayed_target_pairs = delayed_target_pairs or set()
    targets: list[str] = []
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        for target_track in vehicle.goal.allowed_target_tracks:
            if (
                target_track == source_track
                or target_track in targets
                or (
                    (vehicle_no, target_track) in delayed_target_pairs
                    and not _delayed_target_block_can_advance_order_prefix(
                        plan_input=plan_input,
                        state=state,
                        vehicle_by_no=vehicle_by_no,
                        block=block,
                        target_track=target_track,
                        delayed_target_pairs=delayed_target_pairs,
                        order_debts_by_track=order_debts_by_track,
                    )
                )
            ):
                continue
            targets.append(target_track)
    if len(block) == 1:
        return targets
    compatible: list[str] = []
    for target_track in targets:
        if all(
            (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and target_track in vehicle.goal.allowed_target_tracks
            and not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            )
            for vehicle_no in block
        ):
            compatible.append(target_track)
    return compatible


def _delayed_target_block_can_advance_order_prefix(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    target_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None,
    order_debts_by_track: dict[str, Any] | None,
) -> bool:
    if not delayed_target_pairs or not order_debts_by_track:
        return False
    order_debt = order_debts_by_track.get(target_track)
    if order_debt is None:
        return False
    if any((vehicle_no, target_track) not in delayed_target_pairs for vehicle_no in block):
        return False
    preview_state = state.model_copy(deep=True)
    preview_state.track_sequences[target_track] = [
        *block,
        *preview_state.track_sequences.get(target_track, []),
    ]
    return all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=preview_state,
            plan_input=plan_input,
        )
        for vehicle_no in block
    )


def _build_work_position_window_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
    front_block_only: bool = False,
) -> MoveCandidate | None:
    commitment_vehicle_nos = [
        vehicle_no
        for vehicle_no in pending_vehicle_nos
        if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS
    ]
    if not commitment_vehicle_nos:
        return None
    source_track = track_by_vehicle.get(commitment_vehicle_nos[0])
    if source_track is None or source_track == target_track:
        return None
    source_seq = list(state.track_sequences.get(source_track, []))
    target_seq = list(state.track_sequences.get(target_track, []))
    if front_block_only:
        commitment_vehicle_nos = _front_work_position_commitment_block(
            source_seq=source_seq,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
        if not commitment_vehicle_nos:
            return None
    if _work_position_commitment_should_wait_for_free_buffers(
        commitment_vehicle_nos=commitment_vehicle_nos,
        pending_vehicle_nos=pending_vehicle_nos,
        target_seq=target_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    block_clear_count = _work_position_block_clear_count(
        target_track=target_track,
        target_seq=target_seq,
        incoming_vehicle_nos=commitment_vehicle_nos,
        vehicle_by_no=vehicle_by_no,
    )
    if block_clear_count is None:
        return None
    max_clear_count = max(len(blocking_prefix_vehicle_nos), block_clear_count)
    if _prefix_clearance_crosses_stable_work_window(
        target_seq=target_seq,
        clear_count=max_clear_count,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        state=state,
    ):
        return None
    blocking_prefix_vehicle_nos = target_seq[
        : max_clear_count
    ]
    source_indexes = [
        source_seq.index(vehicle_no)
        for vehicle_no in commitment_vehicle_nos
        if vehicle_no in source_seq
    ]
    if len(source_indexes) != len(commitment_vehicle_nos):
        return None
    first_index = min(source_indexes)
    last_index = max(source_indexes)
    if source_seq[first_index : last_index + 1] != commitment_vehicle_nos:
        return None
    order_buffer = source_seq[:first_index]
    if any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is None
        or (
            target_track in vehicle.goal.allowed_target_tracks
            and vehicle.goal.work_position_kind == "SPOTTING"
        )
        for vehicle_no in order_buffer
    ):
        return None

    steps: list[HookAction] = []
    current_state = state
    if blocking_prefix_vehicle_nos:
        clear_steps, current_state = _compile_prefix_clearance_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=target_track,
            vehicle_by_no=vehicle_by_no,
            block=blocking_prefix_vehicle_nos,
            forbidden_tracks={source_track, target_track},
        )
        if clear_steps is None:
            return None
        steps.extend(clear_steps)

    if order_buffer:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=order_buffer,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks={source_track, target_track},
        )
        if stage_track is None:
            return None
        buffer_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=stage_track,
            block=order_buffer,
            action_source_track=source_track,
        )
        if buffer_steps is None:
            return None
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=buffer_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state

    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=commitment_vehicle_nos,
        action_source_track=source_track,
    )
    if target_steps is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=current_state,
        vehicle_by_no=vehicle_by_no,
        steps=target_steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    final_state = compiled.final_state
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=final_state,
            plan_input=plan_input,
        )
        for vehicle_no in commitment_vehicle_nos
    ):
        return None
    steps.extend(compiled.steps)
    return _structural_candidate(
        steps=steps,
        reason="work_position_window_repair",
        focus_tracks=(target_track,),
        origin=(
            "work_position_window_front_block"
            if front_block_only
            else "work_position_window_general"
        ),
    )


def _build_work_position_source_opening_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
) -> MoveCandidate | None:
    commitments_by_source = _work_position_commitments_by_source(
        pending_vehicle_nos=pending_vehicle_nos,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
    )
    for source_track, source_spotting in sorted(
        commitments_by_source.items(),
        key=lambda item: _first_source_index(
            source_seq=list(state.track_sequences.get(item[0], [])),
            vehicle_nos=item[1],
        ),
    ):
        candidate = _build_source_opening_for_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            source_commitments=source_spotting,
            pending_vehicle_nos=pending_vehicle_nos,
            blocking_prefix_vehicle_nos=blocking_prefix_vehicle_nos,
        )
        if candidate is not None:
            return candidate
    return None


def _work_position_commitments_by_source(
    *,
    pending_vehicle_nos: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
) -> dict[str, list[str]]:
    strict_commitments_by_source: dict[str, list[str]] = {}
    free_commitments_by_source: dict[str, list[str]] = {}
    for vehicle_no in pending_vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.goal.work_position_kind is None:
            continue
        source_track = track_by_vehicle.get(vehicle_no)
        if source_track is None or source_track == target_track:
            continue
        if vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS:
            strict_commitments_by_source.setdefault(source_track, []).append(vehicle_no)
        elif vehicle.goal.work_position_kind == "FREE":
            free_commitments_by_source.setdefault(source_track, []).append(vehicle_no)
    commitments_by_source: dict[str, list[str]] = dict(strict_commitments_by_source)
    for source_track, free_commitments in free_commitments_by_source.items():
        commitments_by_source.setdefault(source_track, free_commitments)
    return commitments_by_source


def _build_source_opening_for_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    source_commitments: list[str],
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
) -> MoveCandidate | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if not source_seq:
        return None
    target_seq = list(state.track_sequences.get(target_track, []))
    first_spot_index = min(
        (
            source_seq.index(vehicle_no)
            for vehicle_no in source_commitments
            if vehicle_no in source_seq
        ),
        default=None,
    )
    if first_spot_index is None or first_spot_index == 0:
        return None
    dispatch_prefix = source_seq[:first_spot_index]
    front_commitments = _front_work_position_commitment_block(
        source_seq=source_seq[first_spot_index:],
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    if not front_commitments:
        front_commitments = _front_free_work_position_block(
            source_seq=source_seq[first_spot_index:],
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
    if not front_commitments:
        return None
    if _work_position_commitment_should_wait_for_free_buffers(
        commitment_vehicle_nos=front_commitments,
        pending_vehicle_nos=pending_vehicle_nos,
        target_seq=target_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    block_clear_count = _work_position_block_clear_count(
        target_track=target_track,
        target_seq=target_seq,
        incoming_vehicle_nos=front_commitments,
        vehicle_by_no=vehicle_by_no,
    )
    if block_clear_count is None:
        return None
    max_clear_count = max(len(blocking_prefix_vehicle_nos), block_clear_count)
    if _prefix_clearance_crosses_stable_work_window(
        target_seq=target_seq,
        clear_count=max_clear_count,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        state=state,
    ):
        return None
    blocking_prefix = target_seq[
        : max_clear_count
    ]
    dispatch_groups = _dispatchable_prefix_groups(
        dispatch_prefix=dispatch_prefix,
        source_track=source_track,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )
    if not dispatch_groups:
        prefix_split_candidate = _build_protected_prefix_work_position_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            protected_prefix=dispatch_prefix,
            transfer_block=front_commitments,
            blocking_prefix=blocking_prefix,
        )
        if prefix_split_candidate is not None:
            return prefix_split_candidate
        staged_prefix_candidate = _build_staged_prefix_work_position_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            protected_prefix=dispatch_prefix,
            transfer_block=front_commitments,
            blocking_prefix=blocking_prefix,
        )
        if staged_prefix_candidate is not None:
            return staged_prefix_candidate
        return None
    steps: list[HookAction] = []
    current_state = state
    if blocking_prefix:
        clear_steps, current_state = _compile_prefix_clearance_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=target_track,
            vehicle_by_no=vehicle_by_no,
            block=blocking_prefix,
            forbidden_tracks={source_track, target_track},
        )
        if clear_steps is None:
            return None
        steps.extend(clear_steps)
    for group in dispatch_groups:
        group_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=group[0],
            block=group[1],
            action_source_track=source_track,
        )
        if group_steps is None:
            return None
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=group_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=front_commitments,
        action_source_track=source_track,
    )
    if target_steps is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=current_state,
        vehicle_by_no=vehicle_by_no,
        steps=target_steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    final_state = compiled.final_state
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=final_state,
            plan_input=plan_input,
        )
        for vehicle_no in front_commitments
    ):
        return None
    steps.extend(compiled.steps)
    return _structural_candidate(
        steps=steps,
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
    )


def _build_work_position_free_fill_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
) -> MoveCandidate | None:
    if not _work_position_free_fill_is_allowed(
        state=state,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        pending_vehicle_nos=pending_vehicle_nos,
    ):
        return None
    source_tracks: list[str] = []
    for vehicle_no in pending_vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.goal.work_position_kind != "FREE":
            continue
        source_track = track_by_vehicle.get(vehicle_no)
        if source_track is None or source_track == target_track or source_track in source_tracks:
            continue
        source_tracks.append(source_track)
    if not source_tracks:
        return None
    steps: list[HookAction] = []
    current_state = state
    for source_track in sorted(source_tracks, key=lambda track: (track not in STAGING_TRACKS, track)):
        source_seq = list(current_state.track_sequences.get(source_track, []))
        block = _front_free_work_position_block(
            source_seq=source_seq,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
        if not block:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=block,
            existing_vehicle_nos=list(current_state.track_sequences.get(target_track, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        fill_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if fill_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=fill_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    if not steps:
        return None
    return _structural_candidate(
        steps=steps,
        reason="work_position_free_fill",
        focus_tracks=(target_track,),
    )


def _build_protected_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    blocking_prefix: list[str],
) -> MoveCandidate | None:
    if blocking_prefix:
        return None
    if not protected_prefix or not transfer_block:
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not _prefix_group_can_detach_to_target(
        group=transfer_block,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    candidate = _build_source_prefix_split_detach_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
    )
    if candidate is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=candidate.steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return _with_candidate_origin(candidate, "work_position_source_opening_protected_prefix")


def _build_staged_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    blocking_prefix: list[str],
) -> MoveCandidate | None:
    if not protected_prefix or not transfer_block:
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    candidate = _compile_staged_prefix_work_position_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        blocking_prefix=blocking_prefix,
        allow_split_prefix=False,
    )
    if candidate is not None:
        return _with_candidate_origin(candidate, "work_position_source_opening_staged_prefix")
    candidate = _compile_staged_prefix_work_position_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        blocking_prefix=blocking_prefix,
        allow_split_prefix=True,
    )
    return _with_candidate_origin(candidate, "work_position_source_opening_staged_split_prefix")


def _compile_staged_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    blocking_prefix: list[str],
    allow_split_prefix: bool,
) -> MoveCandidate | None:
    steps: list[HookAction] = []
    current_state = state
    staged_chunks: list[tuple[str, list[str]]] = []
    remaining_prefix = list(protected_prefix)
    forbidden_tracks = {source_track, target_track}

    if blocking_prefix:
        clear_steps, current_state = _compile_prefix_clearance_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=target_track,
            vehicle_by_no=vehicle_by_no,
            block=blocking_prefix,
            forbidden_tracks=forbidden_tracks,
        )
        if clear_steps is None:
            return None
        steps.extend(clear_steps)

    while remaining_prefix:
        chunk_plan = _next_prefix_staging_chunk(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            remaining_prefix=remaining_prefix,
            forbidden_tracks=forbidden_tracks,
            allow_split_prefix=allow_split_prefix,
        )
        if chunk_plan is None:
            return None
        stage_track, chunk, chunk_steps, next_state = chunk_plan
        steps.extend(chunk_steps)
        staged_chunks.append((stage_track, chunk))
        forbidden_tracks.add(stage_track)
        current_state = next_state
        remaining_prefix = remaining_prefix[len(chunk):]

    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=transfer_block,
        action_source_track=source_track,
    )
    if target_steps is None:
        protected_transfer = _compile_source_tail_protected_transfer_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            forbidden_tracks=forbidden_tracks,
        )
        if protected_transfer is None:
            return None
        transfer_steps, current_state = protected_transfer
        steps.extend(transfer_steps)
    else:
        compiled_target = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=target_steps,
            route_oracle=route_oracle,
        )
        if compiled_target is None or compiled_target.final_state.loco_carry:
            return None
        steps.extend(compiled_target.steps)
        current_state = compiled_target.final_state

    for stage_track, chunk in reversed(staged_chunks):
        restore_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=stage_track,
            target_track=source_track,
            block=chunk,
            action_source_track=stage_track,
        )
        if restore_steps is None:
            return None
        compiled_restore = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None or compiled_restore.final_state.loco_carry:
            return None
        steps.extend(compiled_restore.steps)
        current_state = compiled_restore.final_state

    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
        structural_reserve=True,
        origin=(
            "work_position_source_opening_staged_split_prefix"
            if allow_split_prefix
            else "work_position_source_opening_staged_prefix"
        ),
    )


def _next_prefix_staging_chunk(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    remaining_prefix: list[str],
    forbidden_tracks: set[str],
    allow_split_prefix: bool,
) -> tuple[str, list[str], tuple[HookAction, ...], ReplayState] | None:
    max_chunk_size = len(remaining_prefix) if allow_split_prefix else len(remaining_prefix)
    min_chunk_size = 1 if allow_split_prefix else len(remaining_prefix)
    for size in range(max_chunk_size, min_chunk_size - 1, -1):
        chunk = list(remaining_prefix[:size])
        for stage_track in _ranked_staging_tracks(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=chunk,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        ):
            chunk_steps = _attach_detach_steps(
                state=state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=stage_track,
                block=chunk,
                action_source_track=source_track,
            )
            if chunk_steps is None:
                continue
            compiled = replay_candidate_steps(
                plan_input=plan_input,
                state=state,
                vehicle_by_no=vehicle_by_no,
                steps=chunk_steps,
                route_oracle=route_oracle,
            )
            if compiled is None or compiled.final_state.loco_carry:
                continue
            if not _loco_can_access_track(
                state=compiled.final_state,
                route_oracle=route_oracle,
                target_track=source_track,
            ):
                continue
            if _source_front_is_transfer_ready(
                state=compiled.final_state,
                source_track=source_track,
                transfer_block=transfer_block,
            ) and not _source_to_target_route_is_open_or_tail_protectable(
                plan_input=plan_input,
                state=compiled.final_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                target_track=target_track,
                transfer_block=transfer_block,
                forbidden_tracks=forbidden_tracks | {stage_track},
            ):
                continue
            return stage_track, chunk, compiled.steps, compiled.final_state
    return None


def _source_front_is_transfer_ready(
    *,
    state: ReplayState,
    source_track: str,
    transfer_block: list[str],
) -> bool:
    if not transfer_block:
        return False
    return list(state.track_sequences.get(source_track, []))[: len(transfer_block)] == list(
        transfer_block
    )


def _source_to_target_route_is_open_or_tail_protectable(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    forbidden_tracks: set[str],
) -> bool:
    if _source_to_target_route_is_open(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        transfer_block=transfer_block,
    ):
        return True
    return (
        _compile_source_tail_protected_transfer_steps(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            forbidden_tracks=forbidden_tracks,
        )
        is not None
    )


def _source_to_target_route_is_open(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    transfer_block: list[str],
) -> bool:
    if not transfer_block:
        return True
    source_node = _source_node_after_attach_for_block(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=transfer_block,
    )
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=route_oracle.order_end_node(target_track),
    )
    return path_tracks is not None


def _compile_source_tail_protected_transfer_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    forbidden_tracks: set[str],
) -> tuple[tuple[HookAction, ...], ReplayState] | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(transfer_block)] != list(transfer_block):
        return None
    tail_block = _front_source_stay_block(
        source_seq=source_seq[len(transfer_block) :],
        source_track=source_track,
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    if not tail_block:
        return None

    combined_block = [*transfer_block, *tail_block]
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(combined_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    after_attach = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if after_attach is None:
        return None

    for buffer_track in _ranked_source_tail_buffer_tracks(
        plan_input=plan_input,
        state=after_attach.final_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        tail_block=tail_block,
        transfer_block=transfer_block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_tracks,
    ):
        tail_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_attach.final_state,
            source_track=source_track,
            target_track=buffer_track,
        )
        if tail_path is None:
            continue
        tail_detach = HookAction(
            source_track=source_track,
            target_track=buffer_track,
            vehicle_nos=list(tail_block),
            path_tracks=tail_path,
            action_type="DETACH",
        )
        after_tail = replay_candidate_steps(
            plan_input=plan_input,
            state=after_attach.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[tail_detach],
            route_oracle=route_oracle,
        )
        if after_tail is None or after_tail.final_state.loco_carry != tuple(transfer_block):
            continue

        target_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_tail.final_state,
            source_track=buffer_track,
            target_track=target_track,
        )
        if target_path is None:
            continue
        target_detach = HookAction(
            source_track=buffer_track,
            target_track=target_track,
            vehicle_nos=list(transfer_block),
            path_tracks=target_path,
            action_type="DETACH",
        )
        after_target = replay_candidate_steps(
            plan_input=plan_input,
            state=after_tail.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[target_detach],
            route_oracle=route_oracle,
        )
        if after_target is None or after_target.final_state.loco_carry:
            continue

        restore_tail_steps = _attach_detach_steps(
            state=after_target.final_state,
            route_oracle=route_oracle,
            source_track=buffer_track,
            target_track=source_track,
            block=tail_block,
            action_source_track=buffer_track,
        )
        if restore_tail_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=[attach, tail_detach, target_detach, *restore_tail_steps],
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        return compiled.steps, compiled.final_state
    return None


def _front_source_stay_block(
    *,
    source_seq: list[str],
    source_track: str,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in source_seq:
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
        block.append(vehicle_no)
    return block


def _ranked_source_tail_buffer_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    tail_block: list[str],
    transfer_block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
) -> list[str]:
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in tail_block)
    destination_tracks = {
        target
        for vehicle in plan_input.vehicles
        for target in vehicle.goal.allowed_target_tracks
    }
    candidates: list[tuple[tuple[int, int, int, int, int, float, str], str]] = []
    for track_name in sorted(set(capacity_by_track) - forbidden_tracks - {source_track, target_track}):
        if track_name not in STAGING_TRACKS and track_name not in destination_tracks:
            continue
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(track_name, [])
        )
        if current_length + block_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=tail_block,
            existing_vehicle_nos=list(state.track_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        tail_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=state,
            source_track=source_track,
            target_track=track_name,
        )
        if tail_path is None:
            continue
        tail_detach = HookAction(
            source_track=source_track,
            target_track=track_name,
            vehicle_nos=list(tail_block),
            path_tracks=tail_path,
            action_type="DETACH",
        )
        after_tail = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=[tail_detach],
            route_oracle=route_oracle,
        )
        if after_tail is None or after_tail.final_state.loco_carry != tuple(transfer_block):
            continue
        target_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_tail.final_state,
            source_track=track_name,
            target_track=target_track,
        )
        if target_path is None:
            continue
        restore_path = route_oracle.resolve_clear_path_tracks(
            target_track,
            track_name,
            occupied_track_sequences=after_tail.final_state.track_sequences,
            source_node=route_oracle.order_end_node(target_track),
            target_node=route_oracle.order_end_node(track_name),
        )
        candidates.append(
            (
                (
                    0 if track_name in STAGING_TRACKS else 1,
                    len(state.track_sequences.get(track_name, [])),
                    len(tail_path),
                    len(target_path),
                    capacity_by_track[track_name] - current_length,
                    track_name,
                ),
                track_name,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [track_name for _rank, track_name in candidates]


def _loco_can_access_track(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    target_track: str,
) -> bool:
    if state.loco_track_name == target_track:
        return True
    access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=target_track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    )
    return access.is_valid


def _work_position_free_fill_is_allowed(
    *,
    state: ReplayState,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    plan_input: NormalizedPlanInput,
    pending_vehicle_nos: list[str],
) -> bool:
    for vehicle_no in state.track_sequences.get(target_track, []):
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            continue
        if not goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            return False
    return True


def _prefix_clearance_crosses_stable_work_window(
    *,
    target_seq: list[str],
    clear_count: int,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> bool:
    for vehicle_no in target_seq[:clear_count]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
        ):
            continue
        if target_track not in vehicle.goal.allowed_target_tracks:
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _front_free_work_position_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind != "FREE"
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            break
        block.append(vehicle_no)
    return block


def _work_position_block_clear_count(
    *,
    target_track: str,
    target_seq: list[str],
    incoming_vehicle_nos: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int | None:
    for clear_count in range(len(target_seq) + 1):
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=incoming_vehicle_nos,
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        if all(
            (evaluation := preview.evaluations.get(vehicle_no)) is not None
            and evaluation.satisfied_now
            for vehicle_no in incoming_vehicle_nos
        ):
            return clear_count
    return None


def _compile_prefix_clearance_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    forbidden_tracks: set[str],
) -> tuple[list[HookAction], ReplayState] | tuple[None, ReplayState]:
    groups = _dispatchable_prefix_groups(
        dispatch_prefix=block,
        source_track=source_track,
        target_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )
    if not groups:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=block,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        )
        if stage_track is None:
            return None, state
        groups = [(stage_track, block)]
    steps: list[HookAction] = []
    current_state = state
    for target_track, group in groups:
        group_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=group,
            action_source_track=source_track,
        )
        if group_steps is None:
            return None, state
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=group_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None, state
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    return steps, current_state


def _front_work_position_commitment_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    first_kind: str | None = None
    previous_rank: int | None = None
    for vehicle_no in source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            break
        kind = vehicle.goal.work_position_kind
        if first_kind is None:
            first_kind = kind
        if first_kind == "SPOTTING":
            if kind != "SPOTTING":
                break
        elif kind == "SPOTTING":
            break
        else:
            rank = vehicle.goal.target_rank
            if rank is None:
                break
            if previous_rank is not None and rank < previous_rank:
                break
            previous_rank = rank
        block.append(vehicle_no)
    return block


def _front_spotting_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block = _front_work_position_commitment_block(
        source_seq=source_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    if not block:
        return []
    first_vehicle = vehicle_by_no.get(block[0])
    if first_vehicle is None or first_vehicle.goal.work_position_kind != "SPOTTING":
        return []
    return block


def _work_position_commitment_should_wait_for_free_buffers(
    *,
    commitment_vehicle_nos: list[str],
    pending_vehicle_nos: list[str],
    target_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if not any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind in {"EXACT_NORTH_RANK", "EXACT_WORK_SLOT"}
        for vehicle_no in commitment_vehicle_nos
    ):
        return False
    pending_free = [
        vehicle_no
        for vehicle_no in pending_vehicle_nos
        if vehicle_no not in commitment_vehicle_nos
        and (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind == "FREE"
        and target_track in vehicle.goal.allowed_target_tracks
    ]
    if not pending_free:
        return False
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=[*pending_free, *commitment_vehicle_nos],
        existing_vehicle_nos=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return True
    return not all(
        (evaluation := preview.evaluations.get(vehicle_no)) is not None
        and evaluation.satisfied_now
        for vehicle_no in commitment_vehicle_nos
    )


def _dispatchable_prefix_groups(
    *,
    dispatch_prefix: list[str],
    source_track: str,
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle,
) -> list[tuple[str | None, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(dispatch_prefix):
        vehicle = vehicle_by_no.get(dispatch_prefix[index])
        if vehicle is None:
            return []
        if (
            target_track in vehicle.goal.allowed_target_tracks
            and vehicle.goal.work_position_kind != "SPOTTING"
        ):
            group = [dispatch_prefix[index]]
            index += 1
            while index < len(dispatch_prefix):
                next_vehicle = vehicle_by_no.get(dispatch_prefix[index])
                if (
                    next_vehicle is None
                    or next_vehicle.goal.work_position_kind == "SPOTTING"
                    or target_track not in next_vehicle.goal.allowed_target_tracks
                ):
                    break
                group.append(dispatch_prefix[index])
                index += 1
            groups.append((None, group))
            continue
        group_target = _single_allowed_target(vehicle, source_track=source_track)
        if group_target is None:
            return []
        group = [dispatch_prefix[index]]
        index += 1
        while index < len(dispatch_prefix):
            next_vehicle = vehicle_by_no.get(dispatch_prefix[index])
            if next_vehicle is None:
                return []
            if _single_allowed_target(next_vehicle, source_track=source_track) != group_target:
                break
            group.append(dispatch_prefix[index])
            index += 1
        if not _prefix_group_can_detach_to_target(
            group=group,
            target_track=group_target,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        ):
            return []
        groups.append((group_target, group))
    if index == 0:
        return []
    return _assign_staging_buffers_to_order_groups(
        groups=groups,
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )


def _single_allowed_target(
    vehicle: NormalizedVehicle,
    *,
    source_track: str,
) -> str | None:
    targets = [
        target
        for target in vehicle.goal.allowed_target_tracks
        if target != source_track
    ]
    if len(targets) != 1:
        return None
    return targets[0]


def _prefix_group_can_detach_to_target(
    *,
    group: list[str],
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=group,
        existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return False
    return all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and target_track in vehicle.goal.allowed_target_tracks
        for vehicle_no in group
    )


def _first_source_index(*, source_seq: list[str], vehicle_nos: list[str]) -> int:
    return min(
        (source_seq.index(vehicle_no) for vehicle_no in vehicle_nos if vehicle_no in source_seq),
        default=10**9,
    )


def _assign_staging_buffers_to_order_groups(
    *,
    groups: list[tuple[str | None, list[str]]],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[tuple[str, list[str]]]:
    assigned: list[tuple[str, list[str]]] = []
    forbidden_tracks = {source_track, target_track}
    for group_target, group in groups:
        if group_target is not None:
            assigned.append((group_target, group))
            forbidden_tracks.add(group_target)
            continue
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=group,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        )
        if stage_track is None:
            return []
        assigned.append((stage_track, group))
        forbidden_tracks.add(stage_track)
    return assigned


def _attach_detach_steps(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    block: list[str],
    action_source_track: str,
) -> list[HookAction] | None:
    if not block:
        return []
    if list(state.track_sequences.get(source_track, []))[: len(block)] != list(block):
        return None
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    source_node = _source_node_after_attach_for_block(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
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
    detach = HookAction(
        source_track=action_source_track,
        target_track=target_track,
        vehicle_nos=list(block),
        path_tracks=path_tracks,
        action_type="DETACH",
    )
    return [attach, detach]


def _source_node_after_attach_for_block(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
) -> str | None:
    return (
        route_oracle.order_end_node(source_track)
        if len(state.track_sequences.get(source_track, [])) > len(block)
        else None
    )


def _best_staging_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
    prefer_low_route_pressure: bool = False,
    buffer_leases: tuple[BufferLease, ...] = (),
    block_is_carried: bool = False,
) -> str | None:
    ranked_tracks = _ranked_staging_tracks(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_tracks,
        prefer_low_route_pressure=prefer_low_route_pressure,
        buffer_leases=buffer_leases,
        block_is_carried=block_is_carried,
    )
    return ranked_tracks[0] if ranked_tracks else None


def _ranked_staging_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
    prefer_low_route_pressure: bool = False,
    buffer_leases: tuple[BufferLease, ...] = (),
    block_is_carried: bool = False,
) -> list[str]:
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in block)
    vehicle_by_no_all = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    candidates: list[tuple[tuple[int, int, int, int, float, str], str]] = []
    for track_name in sorted((STAGING_TRACKS & set(capacity_by_track)) - forbidden_tracks):
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(track_name, [])
        )
        if current_length + block_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=block,
            existing_vehicle_nos=list(state.track_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        path_tracks = route_oracle.resolve_clear_path_tracks(
            source_track,
            track_name,
            occupied_track_sequences=state.track_sequences,
            source_node=(
                state.loco_node
                if block_is_carried
                else _source_node_after_attach_for_block(
                    state=state,
                    route_oracle=route_oracle,
                    source_track=source_track,
                    block=block,
                )
            ),
            target_node=route_oracle.order_end_node(track_name),
        )
        if path_tracks is None:
            continue
        lease_violation_count = _buffer_lease_violation_count(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            target_track=track_name,
            block=block,
            vehicle_by_no=vehicle_by_no_all,
            capacity_by_track=capacity_by_track,
            length_by_vehicle=length_by_vehicle,
            buffer_leases=buffer_leases,
        )
        route_pressure = 0
        if prefer_low_route_pressure:
            if block_is_carried:
                if tuple(state.loco_carry[-len(block):]) != tuple(block):
                    continue
                steps = [
                    HookAction(
                        source_track=source_track,
                        target_track=track_name,
                        vehicle_nos=list(block),
                        path_tracks=path_tracks,
                        action_type="DETACH",
                    )
                ]
            else:
                steps = _attach_detach_steps(
                    state=state,
                    route_oracle=route_oracle,
                    source_track=source_track,
                    target_track=track_name,
                    block=block,
                    action_source_track=source_track,
                )
                if steps is None:
                    continue
            compiled = replay_candidate_steps(
                plan_input=plan_input,
                state=state,
                vehicle_by_no=vehicle_by_no_all,
                steps=steps,
                route_oracle=route_oracle,
            )
            if compiled is None:
                continue
            route_pressure = compute_route_blockage_plan(
                plan_input,
                compiled.final_state,
                route_oracle,
            ).total_blockage_pressure
        candidates.append(
            (
                (
                    lease_violation_count,
                    route_pressure,
                    len(state.track_sequences.get(track_name, [])),
                    len(path_tracks),
                    capacity_by_track[track_name] - current_length,
                    track_name,
                ),
                track_name,
            )
        )
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    return [track_name for _rank, track_name in candidates]


def _buffer_lease_violation_count(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    target_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
    buffer_leases: tuple[BufferLease, ...],
) -> int:
    if not buffer_leases:
        return 0
    block_set = set(block)
    candidate_sequences = {
        track: list(seq)
        for track, seq in state.track_sequences.items()
    }
    candidate_sequences[target_track] = [
        *block,
        *candidate_sequences.get(target_track, []),
    ]
    violations = 0
    for lease in buffer_leases:
        if block_set.issuperset(lease.vehicle_nos):
            continue
        if _lease_has_feasible_buffer_track(
            lease=lease,
            candidate_sequences=candidate_sequences,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
            length_by_vehicle=length_by_vehicle,
        ):
            continue
        violations += 1
    return violations


def _lease_has_feasible_buffer_track(
    *,
    lease: BufferLease,
    candidate_sequences: dict[str, list[str]],
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
) -> bool:
    lease_block = list(lease.vehicle_nos)
    if not lease_block:
        return True
    for track_name in sorted(STAGING_TRACKS & set(capacity_by_track)):
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in candidate_sequences.get(track_name, [])
        )
        if current_length + lease.required_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=lease_block,
            existing_vehicle_nos=list(candidate_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        path_tracks = route_oracle.resolve_clear_path_tracks(
            lease.source_track,
            track_name,
            occupied_track_sequences=candidate_sequences,
            source_node=route_oracle.order_end_node(lease.source_track),
            target_node=route_oracle.order_end_node(track_name),
        )
        if path_tracks is not None:
            return True
    return False


def _dedup_candidates(candidates: tuple[MoveCandidate, ...]) -> tuple[MoveCandidate, ...]:
    seen: set[tuple] = set()
    result: list[MoveCandidate] = []
    for candidate in candidates:
        key = tuple(
            (
                step.action_type,
                step.source_track,
                step.target_track,
                tuple(step.vehicle_nos),
                tuple(step.path_tracks),
            )
            for step in candidate.steps
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def _append_generation_candidate(
    collection: list[MoveCandidate],
    candidate: MoveCandidate,
    *,
    sibling_candidates: list[MoveCandidate] | None = None,
) -> None:
    if candidate.origin == "route_release_frontier":
        collection.append(candidate)
        return
    existing_candidates = list(collection)
    if sibling_candidates is not None:
        existing_candidates.extend(sibling_candidates)
    if _generation_candidate_is_shadowed(candidate, existing_candidates):
        return
    collection.append(candidate)


def _generation_candidate_is_shadowed(
    candidate: MoveCandidate,
    existing_candidates: list[MoveCandidate],
) -> bool:
    for existing in existing_candidates:
        if not _candidate_has_same_steps(candidate, existing):
            continue
        if _prefer_existing_generation_candidate(existing, candidate):
            return True
    return False


def _prefer_existing_generation_candidate(
    existing: MoveCandidate,
    candidate: MoveCandidate,
) -> bool:
    existing_origin = existing.origin or existing.reason
    candidate_origin = candidate.origin or candidate.reason
    return (existing_origin, candidate_origin) in {
        ("goal_frontier_source_opening", "resource_release_dispatch"),
        ("work_position_free_fill", "resource_release_direct_target"),
        ("work_position_source_opening", "goal_frontier_source_opening"),
    }


def _work_position_front_window_is_redundant(
    *,
    front_candidate: MoveCandidate,
    base_candidate: MoveCandidate | None,
) -> bool:
    if _candidate_has_same_steps(front_candidate, base_candidate):
        return True
    if base_candidate is None:
        return False
    base_origin = base_candidate.origin or base_candidate.reason
    front_origin = front_candidate.origin or front_candidate.reason
    if front_origin != "work_position_window_front_block":
        return False
    if not base_origin.startswith("work_position_"):
        return False
    return (
        tuple(front_candidate.focus_tracks) == tuple(base_candidate.focus_tracks)
        and len(front_candidate.steps) >= len(base_candidate.steps)
    )


def _candidate_has_same_steps(
    candidate: MoveCandidate,
    other: MoveCandidate | None,
) -> bool:
    if other is None:
        return False
    return _candidate_step_signature(candidate) == _candidate_step_signature(other)


def _candidate_step_signature(candidate: MoveCandidate) -> tuple:
    return tuple(
        (
            step.action_type,
            step.source_track,
            step.target_track,
            tuple(step.vehicle_nos),
            tuple(step.path_tracks),
        )
        for step in candidate.steps
    )


def _candidate_origin_counts(candidates: tuple[MoveCandidate, ...]) -> dict[str, int]:
    counter = Counter(
        candidate.origin or candidate.reason or candidate.kind
        for candidate in candidates
    )
    return dict(counter.most_common())


def _candidate_intent_counts(
    candidates: tuple[MoveCandidate, ...] | list[MoveCandidate],
) -> dict[str, int]:
    counter = Counter(
        candidate.intent_type or "UNCLASSIFIED"
        for candidate in candidates
    )
    return dict(counter.most_common())


def _structural_candidate_competition_examples(
    candidates: tuple[MoveCandidate, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[MoveCandidate]] = {}
    for candidate in candidates:
        source_track = _candidate_primary_source_track(candidate)
        if source_track is None:
            continue
        target_tracks = _candidate_business_target_tracks(
            candidate,
            source_track=source_track,
        )
        key = (source_track, target_tracks)
        grouped.setdefault(key, []).append(candidate)
    examples: list[dict[str, Any]] = []
    for (source_track, target_tracks), group in grouped.items():
        origins = sorted(
            {
                candidate.origin or candidate.reason or candidate.kind
                for candidate in group
            }
        )
        if len(origins) <= 1:
            continue
        examples.append(
            {
                "source_track": source_track,
                "target_tracks": list(target_tracks),
                "candidate_count": len(group),
                "origins": origins,
            }
        )
    examples.sort(
        key=lambda item: (-item["candidate_count"], item["source_track"], item["origins"])
    )
    return examples[:24]


def _structural_candidate_competition_group_count(
    candidates: tuple[MoveCandidate, ...],
) -> int:
    count = 0
    grouped: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for candidate in candidates:
        source_track = _candidate_primary_source_track(candidate)
        if source_track is None:
            continue
        target_tracks = _candidate_business_target_tracks(
            candidate,
            source_track=source_track,
        )
        key = (source_track, target_tracks)
        grouped.setdefault(key, set()).add(candidate.origin or candidate.reason or candidate.kind)
    for origins in grouped.values():
        if len(origins) > 1:
            count += 1
    return count


def _structural_candidate_competition_origin_counts(
    candidates: tuple[MoveCandidate, ...],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    grouped: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for candidate in candidates:
        source_track = _candidate_primary_source_track(candidate)
        if source_track is None:
            continue
        target_tracks = _candidate_business_target_tracks(
            candidate,
            source_track=source_track,
        )
        key = (source_track, target_tracks)
        grouped.setdefault(key, set()).add(candidate.origin or candidate.reason or candidate.kind)
    for origins in grouped.values():
        if len(origins) <= 1:
            continue
        for origin in origins:
            counter[origin] += 1
    return dict(counter.most_common())


def _structural_candidate_competition_intent_counts(
    candidates: tuple[MoveCandidate, ...],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    grouped: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for candidate in candidates:
        source_track = _candidate_primary_source_track(candidate)
        if source_track is None:
            continue
        target_tracks = _candidate_business_target_tracks(
            candidate,
            source_track=source_track,
        )
        key = (source_track, target_tracks)
        grouped.setdefault(key, set()).add(candidate.intent_type or "UNCLASSIFIED")
    for intents in grouped.values():
        if len(intents) <= 1:
            continue
        for intent_type in intents:
            counter[intent_type] += 1
    return dict(counter.most_common())


def _candidate_overlap_examples(
    candidates: tuple[MoveCandidate, ...],
    limit: int = 8,
) -> list[dict[str, Any]]:
    by_signature: dict[tuple, list[MoveCandidate]] = {}
    for candidate in candidates:
        by_signature.setdefault(_candidate_step_signature(candidate), []).append(candidate)
    overlaps: list[dict[str, Any]] = []
    for signature, group in by_signature.items():
        origins = sorted({candidate.origin or candidate.reason or candidate.kind for candidate in group})
        reasons = sorted({candidate.reason or candidate.kind for candidate in group})
        if len(origins) <= 1:
            continue
        overlaps.append(
            {
                "count": len(group),
                "origins": origins,
                "reasons": reasons,
                "step_count": len(signature),
                "focus_tracks": sorted(
                    {
                        track
                        for candidate in group
                        for track in candidate.focus_tracks
                    }
                ),
            }
        )
    overlaps.sort(
        key=lambda item: (
            -item["count"],
            -len(item["origins"]),
            -item["step_count"],
            tuple(item["origins"]),
        )
    )
    return overlaps[:limit]


def _candidate_steps_by_kind(candidates: tuple[MoveCandidate, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.kind] = counts.get(candidate.kind, 0) + len(candidate.steps)
    return dict(sorted(counts.items()))


def _candidate_search_sort_key(
    candidate: MoveCandidate,
    *,
    debt_chain_summary: DebtChainSummary | None = None,
    intent: StructuralIntent | None = None,
) -> tuple:
    kind_priority = 0 if candidate.kind == "structural" else 1
    intent_priority = INTENT_PRIORITY.get(candidate.intent_type, 99)
    if candidate.reason.startswith("work_position_"):
        reason_priority = 0
    elif candidate.reason.startswith("chain_macro_"):
        reason_priority = 1
    else:
        reason_priority = 2
    task_queue_priority = _candidate_task_queue_priority(
        candidate,
        debt_chain_summary=debt_chain_summary,
        intent=intent,
    )
    step_signature = tuple(
        (
            step.action_type,
            step.source_track,
            step.target_track,
            tuple(step.vehicle_nos),
            tuple(step.path_tracks),
        )
        for step in candidate.steps
    )
    return (
        kind_priority,
        intent_priority,
        0 if candidate.problem_stage.endswith("primary") else 1,
        reason_priority,
        candidate.soft_penalty,
        _candidate_goal_progress_penalty(candidate),
        _candidate_staging_only_penalty(candidate),
        len(candidate.steps),
        tuple(candidate.focus_tracks),
        0 if candidate.structural_reserve else 1,
        task_queue_priority,
        candidate.reason,
        step_signature,
    )


def _candidate_task_queue_priority(
    candidate: MoveCandidate,
    *,
    debt_chain_summary: DebtChainSummary | None,
    intent: StructuralIntent | None,
) -> tuple:
    if candidate.kind != "structural":
        return (1, 0, 0, 0, 0, ())
    chain_index, anchor_track = (
        _candidate_chain_metadata(candidate, debt_chain_summary)
        if debt_chain_summary is not None
        else (0, "")
    )
    role_priority = _candidate_cache_role_priority(candidate, intent=intent)
    return (
        0,
        chain_index,
        role_priority,
        0 if anchor_track in candidate.focus_tracks else 1,
        0
        if (
            candidate.reason.startswith("work_position_")
            or candidate.origin in {"route_release_frontier", "resource_release_direct_target"}
        )
        else 1,
        tuple(candidate.focus_tracks),
    )


def _candidate_cache_role_priority(
    candidate: MoveCandidate,
    *,
    intent: StructuralIntent | None,
) -> int:
    if intent is None or not candidate.focus_tracks:
        return 4
    roles: set[str] = set()
    for track_name in candidate.focus_tracks:
        cluster = intent.debt_clusters_by_track.get(track_name)
        if cluster is None:
            continue
        roles.update(cluster.buffer_roles)
    if "ORDER_BUFFER" in roles:
        return 0
    if "ROUTE_RELEASE" in roles:
        return 1
    if "CAPACITY_RELEASE" in roles:
        return 2
    if "SOURCE_REMAINDER" in roles:
        return 3
    return 4


def _candidate_goal_progress_penalty(candidate: MoveCandidate) -> int:
    detach_steps = [step for step in candidate.steps if step.action_type == "DETACH"]
    if not detach_steps:
        return 3
    if any(step.target_track not in STAGING_TRACKS for step in detach_steps):
        return 0
    return 2


def _candidate_staging_only_penalty(candidate: MoveCandidate) -> int:
    detach_steps = [step for step in candidate.steps if step.action_type == "DETACH"]
    if not detach_steps:
        return 0
    staging_targets = sum(1 for step in detach_steps if step.target_track in STAGING_TRACKS)
    business_targets = len(detach_steps) - staging_targets
    if business_targets > 0:
        return 0
    if candidate.origin == "route_release_frontier":
        return 1
    return 4
