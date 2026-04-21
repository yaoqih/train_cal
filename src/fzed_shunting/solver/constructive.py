"""Constructive baseline solver.

Priority-rule dispatcher that always returns a plan (or a best-effort partial
plan) for any legal input. Used as the rock-bottom SLA guarantee beneath the
A* / anytime-fallback optimisation layers.

The algorithm is a greedy event-driven loop:

1. If the current state already satisfies ``_is_goal``, return the plan.
2. Ask ``move_generator.generate_goal_moves`` for all legal candidates.
3. Score each candidate with a tuple-based priority and pick the minimum.
4. Apply the move, repeat until goal or the budget is exhausted.

Because candidate generation already honours every hard business constraint
(traction limits, path interference, close-door pruning, capacity tolerance,
spot availability, etc.), any plan the dispatcher emits is legal by
construction. Verification still happens on the solver entry point.

Bounded backtracking (W3-N): If greedy_forward gets stuck in a local minimum
(heuristic stale for ``stuck_threshold`` rounds) without reaching the goal,
``solve_constructive`` rewinds to a recent decision point and tries an
alternative (2nd/3rd best) move there, retrying up to ``max_backtracks``
times. This breaks the 12 known positive failures where greedy picks the
wrong move at some key step and oscillates in displacement cycles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.depot_late import DEPOT_INNER_TRACKS
from fzed_shunting.solver.heuristic import make_state_heuristic
from fzed_shunting.solver.move_generator import generate_goal_moves
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


STAGING_TRACKS = frozenset({"临1", "临2", "临3", "临4", "存4南"})
_INVERSE_GUARD_WINDOW = 12

# W3-N backtracking controls
REWIND_WINDOW = 30
MAX_ALTERNATIVES_PER_STEP = 3

# Stale-detection thresholds.
# Purposeful moves (tier < 5: goal-progress, blocker-clear, dig-out) may keep
# h flat for longer stretches during clearing chains without being "stuck".
# Regressive/lateral moves (tier >= 5) are held to the tighter threshold.
_PURPOSEFUL_STUCK_THRESHOLD = 60


@dataclass(frozen=True)
class ConstructiveResult:
    plan: list[HookAction]
    reached_goal: bool
    iterations: int
    elapsed_ms: float
    stuck_reason: str | None = None
    debug_stats: dict[str, Any] | None = None
    final_heuristic: float | None = None


def solve_constructive(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    *,
    max_iterations: int = 1000,
    stuck_threshold: int = 30,
    max_backtracks: int = 5,
    time_budget_ms: float | None = None,
    debug_stats: dict[str, Any] | None = None,
) -> ConstructiveResult:
    """Priority-rule dispatcher with bounded backtracking.

    Runs greedy-forward. If it gets stuck (h stale for ``stuck_threshold``
    rounds) without reaching goal, rewinds ~15-30 steps and tries an
    alternative move at one of the rewound decision points. Retries up to
    ``max_backtracks`` times. Always returns a ``ConstructiveResult`` — the
    best attempt seen across all retries (prefer reached_goal; among equal,
    prefer fewer hooks; among not reached, prefer lower final heuristic).

    ``stuck_threshold`` default 30 gives an identity-goal displacement detour
    room while still being far below ``max_iterations`` on genuinely stuck
    scenarios.
    """
    started_at = perf_counter()

    # ``alternatives`` maps step_index -> which alternative to pick at that step
    # (0 = best; 1 = 2nd-best; etc.). Starts empty (full greedy).
    alternatives: dict[int, int] = {}
    best_attempt: ConstructiveResult | None = None
    attempt_idx = 0

    for attempt_idx in range(max_backtracks + 1):
        remaining_budget = None
        if time_budget_ms is not None:
            elapsed = (perf_counter() - started_at) * 1000
            remaining_budget = time_budget_ms - elapsed
            if remaining_budget <= 0:
                break

        result = _greedy_forward(
            plan_input,
            initial_state,
            master,
            alternatives=alternatives,
            max_iterations=max_iterations,
            stuck_threshold=stuck_threshold,
            time_budget_ms=remaining_budget,
        )

        if best_attempt is None or _is_better_attempt(result, best_attempt):
            best_attempt = result

        if result.reached_goal:
            break

        # If we're out of budget, don't try more backtracks
        if time_budget_ms is not None:
            if (perf_counter() - started_at) * 1000 >= time_budget_ms:
                break

        # Find a step to rewind to and try an alternative there.
        stuck_at = len(result.plan)
        rewind_candidate: int | None = None

        # Preferred: LATEST step in the last REWIND_WINDOW that hasn't yet
        # exhausted its alternatives.
        window_start = max(0, stuck_at - REWIND_WINDOW)
        for step in range(stuck_at - 1, window_start - 1, -1):
            if alternatives.get(step, 0) + 1 >= MAX_ALTERNATIVES_PER_STEP:
                continue
            rewind_candidate = step
            break

        # Fallback: older steps (before window_start) that still have alternatives.
        if rewind_candidate is None:
            for step in range(window_start - 1, -1, -1):
                if alternatives.get(step, 0) + 1 >= MAX_ALTERNATIVES_PER_STEP:
                    continue
                rewind_candidate = step
                break

        if rewind_candidate is None:
            # Nothing left to try.
            break

        # Increment the alternative index for the chosen rewind step.
        alternatives[rewind_candidate] = alternatives.get(rewind_candidate, 0) + 1
        # Clear any later entries in `alternatives`: those steps will be
        # re-run from scratch so their prior bias no longer applies.
        for step in list(alternatives.keys()):
            if step > rewind_candidate:
                del alternatives[step]

    assert best_attempt is not None
    if debug_stats is not None:
        debug_stats["constructive_backtrack_count"] = attempt_idx
        if best_attempt.debug_stats:
            debug_stats.update(best_attempt.debug_stats)
        debug_stats["constructive_iterations"] = best_attempt.iterations
        debug_stats["constructive_reached_goal"] = best_attempt.reached_goal

    # Attach backtrack count to the returned result's debug_stats too.
    merged_stats = dict(best_attempt.debug_stats or {})
    merged_stats["constructive_backtrack_count"] = attempt_idx
    return ConstructiveResult(
        plan=best_attempt.plan,
        reached_goal=best_attempt.reached_goal,
        iterations=best_attempt.iterations,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        stuck_reason=best_attempt.stuck_reason,
        debug_stats=merged_stats,
        final_heuristic=best_attempt.final_heuristic,
    )


def _is_better_attempt(
    new_result: ConstructiveResult, old_result: ConstructiveResult
) -> bool:
    """Prefer reached_goal; among reached_goal prefer fewer hooks;
    among not reached prefer lower final_heuristic, then longer plan."""
    if new_result.reached_goal and not old_result.reached_goal:
        return True
    if not new_result.reached_goal and old_result.reached_goal:
        return False
    if new_result.reached_goal:
        return len(new_result.plan) < len(old_result.plan)
    # Both not reached. Prefer lower final heuristic (closer to goal).
    new_h = new_result.final_heuristic if new_result.final_heuristic is not None else float("inf")
    old_h = old_result.final_heuristic if old_result.final_heuristic is not None else float("inf")
    if new_h != old_h:
        return new_h < old_h
    # Tie-break: prefer longer plan (more work done without regressing).
    return len(new_result.plan) > len(old_result.plan)


def _greedy_forward(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    *,
    alternatives: dict[int, int],
    max_iterations: int,
    stuck_threshold: int,
    time_budget_ms: float | None,
) -> ConstructiveResult:
    """Single greedy forward sweep, optionally biased by ``alternatives``.

    At each step, if ``alternatives`` has an entry for the current step
    index, the move at that ordinal (0=best, 1=2nd-best, ...) is picked
    instead of the greedy best. The inverse-guard is applied after
    alternative selection to prevent visible oscillation.
    """
    from fzed_shunting.solver.astar_solver import _apply_move, _is_goal

    started_at = perf_counter()
    route_oracle = RouteOracle(master) if master is not None else None
    vehicle_by_no: dict[str, NormalizedVehicle] = {
        vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles
    }
    goal_tracks_needed = _collect_goal_tracks(plan_input)
    state_heuristic = make_state_heuristic(plan_input)

    state = initial_state
    plan: list[HookAction] = []
    best_heuristic = state_heuristic(state)
    stale_rounds = 0
    purposeful_stale = 0
    recent_moves: deque[tuple[str, str, tuple[str, ...]]] = deque(maxlen=_INVERSE_GUARD_WINDOW)
    stats: dict[str, int] = {
        "tier0_close_door_final": 0,
        "tier1_weigh_to_jiku": 0,
        "tier2_blocker_clearance": 0,
        "tier3_goal_satisfaction": 0,
        "tier4_dig_out_buried_seeker": 0,
        "tier5_lateral": 0,
        "tier6_staging": 0,
    }

    stuck_reason: str | None = None
    final_heuristic: float = best_heuristic

    for iteration in range(max_iterations):
        if _is_goal(plan_input, state):
            return _build_result(
                plan=plan,
                reached_goal=True,
                iterations=iteration,
                started_at=started_at,
                stats=stats,
                final_heuristic=state_heuristic(state),
            )
        if time_budget_ms is not None:
            if (perf_counter() - started_at) * 1000 > time_budget_ms:
                return _build_result(
                    plan=plan,
                    reached_goal=False,
                    iterations=iteration,
                    started_at=started_at,
                    stats=stats,
                    stuck_reason="time budget exhausted",
                    final_heuristic=state_heuristic(state),
                )

        moves = generate_goal_moves(
            plan_input,
            state,
            master=master,
            route_oracle=route_oracle,
        )
        if not moves:
            return _build_result(
                plan=plan,
                reached_goal=False,
                iterations=iteration,
                started_at=started_at,
                stats=stats,
                stuck_reason="no legal moves",
                final_heuristic=state_heuristic(state),
            )

        # Score and sort all moves (ascending — lowest tuple wins).
        scored: list[tuple[tuple, int, HookAction, bool]] = []
        for move in moves:
            score, tier = _score_move(
                move=move,
                state=state,
                vehicle_by_no=vehicle_by_no,
                goal_tracks_needed=goal_tracks_needed,
            )
            is_inverse = _is_inverse_of_recent(move, recent_moves)
            scored.append((score, tier, move, is_inverse))
        scored.sort(key=lambda entry: entry[0])

        # Apply non-inverse preference to the pool (retain existing behavior).
        non_inverse_pool = [entry for entry in scored if not entry[3]]
        pool = non_inverse_pool if non_inverse_pool else scored

        step_idx = len(plan)
        alt_idx = alternatives.get(step_idx, 0)
        if alt_idx >= len(pool):
            return _build_result(
                plan=plan,
                reached_goal=False,
                iterations=iteration,
                started_at=started_at,
                stats=stats,
                stuck_reason=f"alternative exhausted at step {step_idx}",
                final_heuristic=state_heuristic(state),
            )

        chosen_score, chosen_tier, chosen_move, _is_inv = pool[alt_idx]

        tier_key = f"tier{chosen_tier}_" + _TIER_NAMES[min(chosen_tier, 6)]
        stats[tier_key] = stats.get(tier_key, 0) + 1

        state = _apply_move(
            state=state,
            move=chosen_move,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        plan.append(chosen_move)
        recent_moves.append(
            (chosen_move.source_track, chosen_move.target_track, tuple(chosen_move.vehicle_nos))
        )
        current_heuristic = state_heuristic(state)
        final_heuristic = current_heuristic
        if current_heuristic < best_heuristic:
            best_heuristic = current_heuristic
            stale_rounds = 0
            purposeful_stale = 0
        elif chosen_tier < 5:
            # Purposeful move (goal-progress / blocker-clear / dig-out) but h
            # did not improve.  These are expected during clearing chains where
            # value only materialises after several consecutive steps.  Use the
            # higher purposeful threshold instead of the tight regressive one.
            purposeful_stale += 1
        else:
            stale_rounds += 1
        if stale_rounds >= stuck_threshold:
            return _build_result(
                plan=plan,
                reached_goal=False,
                iterations=iteration + 1,
                started_at=started_at,
                stats=stats,
                stuck_reason=f"heuristic stale for {stuck_threshold} regressive rounds",
                final_heuristic=final_heuristic,
            )
        if purposeful_stale >= _PURPOSEFUL_STUCK_THRESHOLD:
            return _build_result(
                plan=plan,
                reached_goal=False,
                iterations=iteration + 1,
                started_at=started_at,
                stats=stats,
                stuck_reason=f"heuristic stale for {_PURPOSEFUL_STUCK_THRESHOLD} purposeful rounds",
                final_heuristic=final_heuristic,
            )

    return _build_result(
        plan=plan,
        reached_goal=False,
        iterations=max_iterations,
        started_at=started_at,
        stats=stats,
        stuck_reason="max iterations reached",
        final_heuristic=final_heuristic,
    )


_TIER_NAMES = {
    0: "close_door_final",
    1: "weigh_to_jiku",
    2: "blocker_clearance",
    3: "goal_satisfaction",
    4: "dig_out_buried_seeker",
    5: "lateral",
    6: "staging",
}


def _build_result(
    *,
    plan: list[HookAction],
    reached_goal: bool,
    iterations: int,
    started_at: float,
    stats: dict[str, int],
    stuck_reason: str | None = None,
    final_heuristic: float | None = None,
) -> ConstructiveResult:
    elapsed_ms = (perf_counter() - started_at) * 1000
    return ConstructiveResult(
        plan=plan,
        reached_goal=reached_goal,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
        stuck_reason=stuck_reason,
        debug_stats=dict(stats),
        final_heuristic=final_heuristic,
    )


def _collect_goal_tracks(plan_input: NormalizedPlanInput) -> set[str]:
    tracks: set[str] = set()
    for vehicle in plan_input.vehicles:
        tracks.update(vehicle.goal.allowed_target_tracks)
    return tracks


def _choose_best_move(
    *,
    moves: list[HookAction],
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_tracks_needed: set[str],
    recent_moves: deque[tuple[str, str, tuple[str, ...]]] | None = None,
) -> tuple[HookAction, int]:
    """Legacy helper (retained for any external callers): returns the
    greedy-best move under the inverse-guard policy."""
    scored: list[tuple[tuple, int, HookAction, bool]] = []
    for move in moves:
        score, tier = _score_move(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
        )
        is_inverse = _is_inverse_of_recent(move, recent_moves)
        scored.append((score, tier, move, is_inverse))
    non_inverse = [entry for entry in scored if not entry[3]]
    pool = non_inverse if non_inverse else scored
    pool.sort(key=lambda entry: entry[0])
    score, tier, move, _ = pool[0]
    return move, tier


def _is_inverse_of_recent(
    move: HookAction,
    recent_moves: deque[tuple[str, str, tuple[str, ...]]] | None,
) -> bool:
    if not recent_moves:
        return False
    move_vehicles = set(move.vehicle_nos)
    for prev_source, prev_target, prev_vehicles in recent_moves:
        if prev_source != move.target_track or prev_target != move.source_track:
            continue
        prev_set = set(prev_vehicles)
        # Widened inverse: a move counts as inverse if it returns any non-trivial
        # overlap (not just the exact same set). Prevents dig-out/dig-back
        # oscillation when the return move carries a subset of the just-moved
        # block (e.g. 7 of 8 cars going back because one was buried deeper).
        overlap = move_vehicles & prev_set
        if not overlap:
            continue
        # At least half of either side must be in overlap to be considered
        # reversing the prior intent. Permits small reshuffles that happen to
        # share one car.
        if len(overlap) * 2 >= min(len(move_vehicles), len(prev_set)):
            return True
    return False


def _score_move(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_tracks_needed: set[str],
) -> tuple[tuple, int]:
    block_vehicles = [vehicle_by_no[vn] for vn in move.vehicle_nos]
    source_track = move.source_track
    target_track = move.target_track

    # Net change in goal-track placements induced by this hook.
    # +N if this move brings N vehicles closer to their target,
    # 0 if it is a pure staging / lateral motion,
    # <0 if it evicts already-placed vehicles off their target tracks.
    delta = sum(
        int(target_track in v.goal.allowed_target_tracks)
        - int(source_track in v.goal.allowed_target_tracks)
        for v in block_vehicles
    )

    is_close_door_final = (
        target_track == "存4北"
        and any(v.is_close_door for v in block_vehicles)
        and delta > 0
    )
    is_weigh_to_jiku = (
        target_track == "机库"
        and any(
            v.need_weigh and v.vehicle_no not in state.weighed_vehicle_nos
            for v in block_vehicles
        )
    )
    clears_blocker = (
        delta >= 0
        and _move_clears_goal_blocker(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
        )
    )
    exposes_buried_seeker = (
        delta <= 0
        and _move_exposes_buried_goal_seeker(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
        )
    )
    is_staging = target_track in STAGING_TRACKS

    tier: int
    if is_close_door_final:
        tier = 0
    elif is_weigh_to_jiku:
        tier = 1
    elif delta > 0:
        tier = 2  # forward progress (includes SPOT/AREA/TRACK satisfaction)
    elif clears_blocker:
        tier = 3  # evacuates a blocked goal track without lateral progress
    elif exposes_buried_seeker:
        tier = 4  # purposeful regression: digs out a buried goal seeker
    elif delta == 0 and not is_staging:
        tier = 5  # lateral, non-staging (rare; generally avoid)
    elif delta == 0 and is_staging:
        tier = 6  # staging — last resort for progress
    else:
        tier = 7  # delta < 0 with no buried seeker, actively regressing

    # Protect fully-satisfied vehicles from displacement. If a move takes
    # a vehicle away from its allowed target tracks (AND the vehicle was
    # fully satisfied at the source — meaning spot/area/weigh conditions
    # are also met), this is a pure loss: we pay one hook now and will
    # have to pay another later to put it back. Elevate tier so the
    # solver only picks such moves when literally nothing else works.
    displaces_satisfied = _move_displaces_satisfied_vehicle(
        move=move,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    if displaces_satisfied:
        tier += 100

    block_size = len(move.vehicle_nos)
    path_length = len(move.path_tracks)
    # SPOT/AREA finalizations are "pinpoint" goals: the vehicle only reaches
    # its objective if it lands on the exact spot/area. Prefer these over
    # block moves that merely inch other vehicles forward with bulk-delta
    # wins — otherwise a large return block can outrank a single SPOT finish.
    is_spot_or_area_finalization = delta > 0 and any(
        target_track in v.goal.allowed_target_tracks
        and (v.goal.target_mode == "SPOT" or v.goal.target_area_code is not None)
        for v in block_vehicles
    )

    score = (
        tier,
        0 if is_spot_or_area_finalization else 1,
        -delta,
        -block_size,
        path_length,
        source_track,
        target_track,
    )
    return score, min(tier, 5)


def _move_clears_goal_blocker(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_tracks_needed: set[str],
) -> bool:
    source_track = move.source_track
    if source_track not in goal_tracks_needed:
        return False
    source_seq = state.track_sequences.get(source_track, [])
    if not source_seq:
        return False
    block_set = set(move.vehicle_nos)
    prefix_matches = source_seq[: len(move.vehicle_nos)] == list(move.vehicle_nos)
    if not prefix_matches:
        return False
    remaining = [vn for vn in source_seq if vn not in block_set]
    for vn in remaining:
        vehicle = vehicle_by_no.get(vn)
        if vehicle is None:
            continue
        if source_track in vehicle.goal.allowed_target_tracks:
            return True
    return False

def _move_exposes_buried_goal_seeker(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    """True when the north-end block move reveals a buried vehicle whose goal
    cannot be satisfied while it remains at ``source_track`` — i.e. purposeful
    dig-out.

    Covers two categories of buried goal seekers:

    1. Vehicle whose target track is **elsewhere** (needs to leave source).
    2. Vehicle that needs weighing and has not yet been weighed. Even if its
       target track IS source_track, the weigh goal still requires visiting
       机库 — otherwise the goal is permanently unmet.
    """
    source_track = move.source_track
    source_seq = state.track_sequences.get(source_track, [])
    if not source_seq:
        return False
    block_set = set(move.vehicle_nos)
    prefix_matches = source_seq[: len(move.vehicle_nos)] == list(move.vehicle_nos)
    if not prefix_matches:
        return False
    remaining = [vn for vn in source_seq if vn not in block_set]
    if not remaining:
        return False
    for vn in remaining:
        vehicle = vehicle_by_no.get(vn)
        if vehicle is None:
            continue
        if source_track not in vehicle.goal.allowed_target_tracks:
            return True
        if vehicle.need_weigh and vehicle.vehicle_no not in state.weighed_vehicle_nos:
            return True
    return False


def _move_displaces_satisfied_vehicle(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    """True iff any vehicle in this move is currently fully satisfied AND
    the move takes it OUT of its allowed target tracks.

    "Fully satisfied" means: the vehicle's current track (=move.source_track,
    since this is the move that will take it out) is in allowed_target_tracks
    AND any mode-specific extras are met:
      - SPOT mode: spot_assignments[vno] == target_spot_code
      - 大库:RANDOM: spot_assignments[vno] is not None
      - WORK_AREA (调棚/洗南/油/抛/调棚预修): spot_assignments[vno] is not None
      - need_weigh: vno in weighed_vehicle_nos (only matters if vehicle is actually weighing-dependent)
      - close-door at 存4北: final seq index >= 3

    The check is per-vehicle on the block; if ANY vehicle in the block is
    fully satisfied and would be displaced, the whole move is flagged.
    """
    source_track = move.source_track
    target_track = move.target_track

    for vno in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vno)
        if vehicle is None:
            continue

        # Is the vehicle at goal right now?
        if source_track not in vehicle.goal.allowed_target_tracks:
            continue  # Not at goal, can't be "displaced" — skip

        # Mode-specific satisfaction checks (mirror _is_goal logic in state.py)
        if vehicle.goal.target_mode == "SPOT":
            if state.spot_assignments.get(vno) != vehicle.goal.target_spot_code:
                continue  # At correct track but wrong spot → not satisfied
        if vehicle.goal.target_area_code == "大库:RANDOM":
            if state.spot_assignments.get(vno) is None:
                continue
        if vehicle.goal.target_area_code in {
            "调棚:WORK", "调棚:PRE_REPAIR", "洗南:WORK", "油:WORK", "抛:WORK",
        }:
            if state.spot_assignments.get(vno) is None:
                continue
        if vehicle.need_weigh and vno not in state.weighed_vehicle_nos:
            continue  # Weighing pending → not yet fully satisfied even if at target track
        if vehicle.is_close_door and source_track == "存4北":
            final_seq = state.track_sequences.get("存4北", [])
            if vno in final_seq and final_seq.index(vno) < 3:
                continue  # Close-door not yet at required position

        # Vehicle IS fully satisfied at source. Is the move taking it away?
        if target_track not in vehicle.goal.allowed_target_tracks:
            return True  # Displacement!

    return False
