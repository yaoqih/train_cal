"""Large Neighborhood Search (LNS) improvement layer for solver incumbents."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.purity import compute_state_purity
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.state import _is_goal, _locate_vehicle
from fzed_shunting.solver.types import HookAction
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.verify.replay import ReplayState, replay_plan


def _is_goal_after_prefix(
    *,
    plan_input: NormalizedPlanInput,
    start_state: ReplayState,
) -> bool:
    """True when the state at the cut point already satisfies the goal.

    Only in that case is it safe to accept a prefix + empty-repair cut:
    the prefix alone already solves the problem, so skipping the suffix
    is a strict improvement.
    """
    return _is_goal(plan_input, start_state)


def _solve_with_lns_result(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    heuristic_weight: float,
    beam_width: int | None,
    repair_passes: int,
    debug_stats: dict[str, Any] | None,
    solve_search_result,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult:
    started_at = perf_counter()
    seed_solver_mode = "beam" if beam_width is not None else "weighted"
    seed_debug_stats = debug_stats if debug_stats is not None else None
    incumbent = solve_search_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        solver_mode=seed_solver_mode,
        heuristic_weight=max(heuristic_weight, 1.5),
        beam_width=beam_width,
        debug_stats=seed_debug_stats,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    improved = _improve_incumbent_result(
        plan_input=plan_input,
        initial_state=initial_state,
        master=master,
        incumbent=incumbent,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        repair_passes=repair_passes,
        max_rounds=None,
        solve_search_result=solve_search_result,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    return SolverResult(
        plan=improved.plan,
        expanded_nodes=improved.expanded_nodes,
        generated_nodes=improved.generated_nodes,
        closed_nodes=improved.closed_nodes,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        is_complete=improved.is_complete,
        is_proven_optimal=improved.is_proven_optimal,
        fallback_stage=improved.fallback_stage,
        partial_plan=list(improved.partial_plan),
        partial_fallback_stage=improved.partial_fallback_stage,
        verification_report=improved.verification_report,
        partial_verification_report=improved.partial_verification_report,
        debug_stats=debug_stats,
    )


def _improve_incumbent_result(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    incumbent: SolverResult,
    heuristic_weight: float,
    beam_width: int | None,
    repair_passes: int,
    max_rounds: int | None,
    solve_search_result,
    time_budget_ms: float | None = None,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult:
    if repair_passes <= 0 or not incumbent.is_complete or not incumbent.plan:
        return incumbent

    from fzed_shunting.solver.budget import SearchBudget

    started_at = perf_counter()
    incumbent_plan = list(incumbent.plan)
    total_expanded = incumbent.expanded_nodes
    total_generated = incumbent.generated_nodes
    total_closed = incumbent.closed_nodes
    route_oracle = RouteOracle(master) if master is not None else None
    rounds = 0

    def _remaining_ms() -> float | None:
        if time_budget_ms is None:
            return None
        return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000)

    while max_rounds is None or rounds < max_rounds:
        rounds += 1
        if time_budget_ms is not None and _remaining_ms() <= 0:
            break
        incumbent_replay = replay_plan(
            initial_state,
            [_to_hook_dict(move) for move in incumbent_plan],
            plan_input=plan_input,
        )
        snapshots = incumbent_replay.snapshots
        incumbent_end_state = getattr(incumbent_replay, "final_state", snapshots[-1])
        improved = False
        for cut_index in _candidate_repair_cut_points(incumbent_plan, repair_passes):
            remaining = _remaining_ms()
            if remaining is not None and remaining <= 0:
                break
            prefix = incumbent_plan[:cut_index]
            start_state = snapshots[cut_index]
            if start_state.loco_carry:
                continue
            movable_vehicle_nos: set[str] = set()
            if route_oracle is not None:
                from fzed_shunting.solver.route_blockage import compute_route_blockage_plan

                route_blockage_plan = compute_route_blockage_plan(
                    plan_input,
                    start_state,
                    route_oracle,
                )
                movable_vehicle_nos.update(route_blockage_plan.blocking_vehicle_nos)
            repair_input = _build_repair_plan_input(
                plan_input,
                start_state,
                movable_vehicle_nos=movable_vehicle_nos,
            )
            per_call_budget = None
            if remaining is not None:
                # Spread remaining time across expected remaining calls
                # (passes × rounds_left). Floor at 100ms to avoid useless
                # micro-budgets.
                rounds_left = 1 if max_rounds is None else max(1, max_rounds - rounds + 1)
                per_call_budget = max(100.0, remaining / (repair_passes * rounds_left))
            try:
                repaired = solve_search_result(
                    plan_input=repair_input,
                    initial_state=start_state,
                    master=master,
                    solver_mode="exact" if beam_width is None else "beam",
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    debug_stats=None,
                    budget=SearchBudget(time_budget_ms=per_call_budget),
                    enable_depot_late_scheduling=enable_depot_late_scheduling,
                )
            except ValueError:
                continue
            # Reject timed-out sub-searches: a budget-exhausted inner call
            # returns plan=[] which, combined with the prefix, yields an
            # incomplete plan that would fool the length comparator into
            # accepting it. Only cuts that actually re-solve to goal win.
            if not repaired.is_complete and not _is_goal_after_prefix(
                plan_input=plan_input,
                start_state=start_state,
            ):
                continue
            total_expanded += repaired.expanded_nodes
            total_generated += repaired.generated_nodes
            total_closed += repaired.closed_nodes
            candidate_plan = prefix + repaired.plan
            candidate_replay = replay_plan(
                initial_state,
                [_to_hook_dict(move) for move in candidate_plan],
                plan_input=plan_input,
            )
            candidate_end_state = getattr(candidate_replay, "final_state", candidate_replay.snapshots[-1])
            if _is_better_plan(
                candidate_plan,
                incumbent_plan,
                route_oracle,
                depot_late=enable_depot_late_scheduling,
                candidate_purity=tuple(compute_state_purity(plan_input, candidate_end_state).__dict__.values()),
                incumbent_purity=tuple(compute_state_purity(plan_input, incumbent_end_state).__dict__.values()),
            ):
                incumbent_plan = candidate_plan
                improved = True
                break
        if not improved:
            break

    return SolverResult(
        plan=incumbent_plan,
        expanded_nodes=total_expanded,
        generated_nodes=total_generated,
        closed_nodes=total_closed,
        elapsed_ms=incumbent.elapsed_ms + (perf_counter() - started_at) * 1000,
        is_complete=True,
        is_proven_optimal=incumbent.is_proven_optimal,
        fallback_stage=incumbent.fallback_stage,
        debug_stats=incumbent.debug_stats,
    )


def _candidate_repair_cut_points(plan: list[HookAction], repair_passes: int) -> list[int]:
    """Generate diverse cut-point candidates for LNS repair.

    Combines four destroy operators (round-robin across the requested passes)
    to widen the search neighbourhood beyond pure hotspot detection:

    - **hotspot**: tracks touched many times (existing behavior).
    - **worst-cost**: hooks whose route is long or touches many branches.
    - **target-cluster**: first index of adjacent same-target runs (merge
      candidates).
    - **equidistant**: evenly spaced indexes spanning the plan (diversifier).

    Deduplicated and capped at ``repair_passes``.
    """
    if not plan:
        return []
    plan_length = len(plan)
    per_strategy = max(1, repair_passes // 4)
    touch_count_by_track: dict[str, int] = {}
    for move in plan:
        touch_count_by_track[move.source_track] = touch_count_by_track.get(move.source_track, 0) + 1
        touch_count_by_track[move.target_track] = touch_count_by_track.get(move.target_track, 0) + 1

    hotspot_cuts = _cut_points_hotspot(plan, touch_count_by_track, per_strategy)
    worst_cuts = _cut_points_worst_cost(plan, per_strategy)
    target_cuts = _cut_points_target_cluster(plan, per_strategy)
    equi_cuts = _cut_points_equidistant(plan_length, per_strategy)

    merged: list[int] = []
    seen: set[int] = set()
    for cut in (*hotspot_cuts, *worst_cuts, *target_cuts, *equi_cuts):
        if cut in seen or cut < 0 or cut >= plan_length:
            continue
        seen.add(cut)
        merged.append(cut)
        if len(merged) >= repair_passes:
            break
    return merged


def _cut_points_hotspot(
    plan: list[HookAction],
    touch_count_by_track: dict[str, int],
    limit: int,
) -> list[int]:
    scored = sorted(
        enumerate(plan),
        key=lambda item: _repair_cut_score(item[0], item[1], touch_count_by_track),
    )
    return [idx for idx, _ in scored[:limit]]


def _cut_points_worst_cost(plan: list[HookAction], limit: int) -> list[int]:
    scored = sorted(
        enumerate(plan),
        key=lambda item: -(len(item[1].path_tracks) * 100 + len(item[1].vehicle_nos)),
    )
    return [idx for idx, _ in scored[:limit]]


def _cut_points_target_cluster(plan: list[HookAction], limit: int) -> list[int]:
    clusters: list[int] = []
    i = 0
    while i < len(plan) and len(clusters) < limit:
        j = i + 1
        while j < len(plan) and plan[j].target_track == plan[i].target_track:
            j += 1
        if j - i >= 2:
            clusters.append(i)
        i = j
    return clusters


def _cut_points_equidistant(plan_length: int, limit: int) -> list[int]:
    if plan_length == 0 or limit == 0:
        return []
    step = max(1, plan_length // (limit + 1))
    return [step * i for i in range(1, limit + 1) if step * i < plan_length]


def _repair_cut_score(
    index: int,
    move: HookAction,
    touch_count_by_track: dict[str, int],
) -> tuple[int, int, int, int]:
    hotspot_score = max(
        touch_count_by_track.get(move.source_track, 0),
        touch_count_by_track.get(move.target_track, 0),
    )
    staging_flag = int(
        move.source_track.startswith("临")
        or move.target_track.startswith("临")
        or move.source_track == "存4南"
        or move.target_track == "存4南"
    )
    repeated_touch_flag = int(hotspot_score > 1)
    return (
        -repeated_touch_flag,
        -staging_flag,
        -hotspot_score,
        index,
    )


def _is_better_plan(
    candidate_plan: list[HookAction],
    incumbent_plan: list[HookAction],
    route_oracle: RouteOracle | None,
    *,
    depot_late: bool = False,
    candidate_purity: tuple[int, int, int] = (0, 0, 0),
    incumbent_purity: tuple[int, int, int] = (0, 0, 0),
) -> bool:
    candidate_metrics = _plan_quality(
        candidate_plan,
        route_oracle,
        depot_late=depot_late,
        purity_metrics=candidate_purity,
    )
    incumbent_metrics = _plan_quality(
        incumbent_plan,
        route_oracle,
        depot_late=depot_late,
        purity_metrics=incumbent_purity,
    )
    return candidate_metrics < incumbent_metrics


def _plan_quality(
    plan: list[HookAction],
    route_oracle: RouteOracle | None,
    *,
    depot_late: bool = False,
    purity_metrics: tuple[int, int, int] = (0, 0, 0),
) -> tuple:
    total_length_m = 0.0
    total_branch_count = 0
    if route_oracle is None:
        total_length_m = float(sum(len(move.path_tracks) for move in plan))
        total_branch_count = int(sum(max(len(move.path_tracks) - 1, 0) for move in plan))
    else:
        for move in plan:
            route = route_oracle.resolve_route(move.source_track, move.target_track)
            if route is not None:
                total_length_m += route.total_length_m
                total_branch_count += len(route.branch_codes)
            else:
                total_branch_count += max(len(move.path_tracks) - 1, 0)
                total_length_m += float(len(move.path_tracks))
    if depot_late:
        from fzed_shunting.solver.depot_late import depot_earliness
        return (len(plan), *purity_metrics, total_branch_count, total_length_m, depot_earliness(plan))
    return (len(plan), *purity_metrics, total_branch_count, total_length_m)


def _build_repair_plan_input(
    plan_input: NormalizedPlanInput,
    snapshot: ReplayState,
    *,
    movable_vehicle_nos: set[str] | frozenset[str] | None = None,
) -> NormalizedPlanInput:
    extra_movable_vehicle_nos = set(movable_vehicle_nos or set())
    reverse_spot_assignments = {
        spot_code: vehicle_no
        for vehicle_no, spot_code in snapshot.spot_assignments.items()
    }
    movable_exact_spot_blockers: set[str] = set()
    movable_front_blockers = _front_blocking_satisfied_vehicle_nos(
        plan_input=plan_input,
        snapshot=snapshot,
    )
    movable_capacity_release_front = _capacity_release_front_vehicle_nos(
        plan_input=plan_input,
        snapshot=snapshot,
    )
    for vehicle in plan_input.vehicles:
        current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
        if goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=snapshot,
            plan_input=plan_input,
        ):
            continue
        if vehicle.goal.target_mode != "SPOT":
            continue
        target_spot = vehicle.goal.target_spot_code
        if not target_spot:
            continue
        occupant_vehicle_no = reverse_spot_assignments.get(target_spot)
        if occupant_vehicle_no is None or occupant_vehicle_no == vehicle.vehicle_no:
            continue
        movable_exact_spot_blockers.add(occupant_vehicle_no)

    localized_vehicles: list[NormalizedVehicle] = []
    for vehicle in plan_input.vehicles:
        current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
        frozen_goal = vehicle.goal
        goal_satisfied = goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=snapshot,
            plan_input=plan_input,
        )
        if (
            goal_satisfied
            and vehicle.vehicle_no not in movable_exact_spot_blockers
            and vehicle.vehicle_no not in movable_front_blockers
            and vehicle.vehicle_no not in movable_capacity_release_front
            and vehicle.vehicle_no not in extra_movable_vehicle_nos
        ):
            if vehicle.goal.work_position_kind is not None:
                frozen_goal = GoalSpec(
                    target_mode="TRACK",
                    target_track=current_track,
                    allowed_target_tracks=[current_track],
                    preferred_target_tracks=[current_track],
                )
            else:
                frozen_goal = GoalSpec(
                    target_mode="TRACK",
                    target_track=current_track,
                    allowed_target_tracks=[current_track],
                    preferred_target_tracks=[current_track],
                    target_area_code=vehicle.goal.target_area_code,
                    target_spot_code=snapshot.spot_assignments.get(vehicle.vehicle_no),
                )
        localized_vehicles.append(
            vehicle.model_copy(
                update={
                    "current_track": current_track,
                    "goal": frozen_goal,
                }
            )
        )
    return plan_input.model_copy(
        update={
            "vehicles": localized_vehicles,
            "loco_track_name": snapshot.loco_track_name,
        }
    )


def _capacity_release_front_vehicle_nos(
    *,
    plan_input: NormalizedPlanInput,
    snapshot: ReplayState,
) -> set[str]:
    release_plan = compute_capacity_release_plan(plan_input, snapshot)
    movable: set[str] = set()
    for fact in release_plan.facts_by_track.values():
        if fact.release_pressure_length <= 1e-9:
            continue
        movable.update(fact.front_release_vehicle_nos)
    return movable


def _front_blocking_satisfied_vehicle_nos(
    *,
    plan_input: NormalizedPlanInput,
    snapshot: ReplayState,
) -> set[str]:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    blockers: set[str] = set()
    for track, seq in snapshot.track_sequences.items():
        prefix: list[str] = []
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            current_satisfied = (
                vehicle is not None
                and goal_is_satisfied(
                    vehicle,
                    track_name=track,
                    state=snapshot,
                    plan_input=plan_input,
                )
            )
            if not current_satisfied:
                if prefix:
                    blockers.update(prefix)
                break
            prefix.append(vehicle_no)
    return blockers


def _spot_goal_satisfied(
    vehicle: NormalizedVehicle,
    snapshot: ReplayState,
    plan_input: NormalizedPlanInput | None = None,
) -> bool:
    current_track = _locate_vehicle(snapshot, vehicle.vehicle_no)
    if current_track is None:
        return False
    return goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=snapshot,
        plan_input=plan_input,
    )


def _to_hook_dict(move: HookAction) -> dict:
    return {
        "hookNo": 1,
        "actionType": move.action_type,
        "sourceTrack": move.source_track,
        "targetTrack": move.target_track,
        "vehicleNos": move.vehicle_nos,
        "pathTracks": move.path_tracks,
    }
