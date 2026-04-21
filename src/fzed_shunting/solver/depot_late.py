"""Depot-late-scheduling helpers.

Pure functions and metrics that express the preference for hooks touching
大库 (修N库内) appearing later in the plan's hook sequence. Consumers:

- search.py: maintains ``depot_index_sum`` incrementally and injects
  ``-depot_index_sum`` as a lexicographic secondary key.
- lns.py: compares plans with (hook_count, depot_earliness, ...) when the
  feature flag is on.
- constructive.py: adds ``depot_late_penalty`` intra-tier.
- astar_solver.py: applies ``reorder_depot_late`` as a final post-processing
  pass before verification.

The scope of "depot-touching" is strictly the four 修N库内 repair tracks;
修N库外 are gateway storage and are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.state import _apply_move
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


DEPOT_INNER_TRACKS: frozenset[str] = frozenset(
    {"修1库内", "修2库内", "修3库内", "修4库内"}
)


def is_depot_hook(hook: HookAction) -> bool:
    """True iff hook's source or target is one of the 修N库内 tracks."""
    return (
        hook.source_track in DEPOT_INNER_TRACKS
        or hook.target_track in DEPOT_INNER_TRACKS
    )


def depot_index_sum(plan: Sequence[HookAction]) -> int:
    """Sum of 1-indexed positions of depot-touching hooks.

    Used during search as an incremental secondary key. Larger = depot
    hooks appear later in the plan.
    """
    return sum(i for i, hook in enumerate(plan, start=1) if is_depot_hook(hook))


def depot_earliness(plan: Sequence[HookAction]) -> int:
    """Sum of (N - i) over depot-touching hook positions, where N = len(plan).

    Smaller = depot hooks appear later. Zero when no depot hooks exist OR
    when all depot hooks are at the tail. Terminal metric used for LNS
    comparison and post-processing decisions.
    """
    n = len(plan)
    return sum(n - i for i, hook in enumerate(plan, start=1) if is_depot_hook(hook))


def _is_early_depot_phase(
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    """True when at least one vehicle with a non-depot goal has not reached it.

    "Non-depot goal" means the vehicle's ``allowed_target_tracks`` contain at
    least one track outside ``DEPOT_INNER_TRACKS``. Vehicles whose allowed
    targets are exclusively depot tracks do not influence the phase.

    Note: this is a positional-only predicate — it ignores ``need_weigh``,
    close-door ordering, and SPOT exact-spot matching. A vehicle sitting on
    an allowed target track is considered "goal reached" for phase purposes
    even if its full goal semantics are not fully satisfied.
    """
    current_by_vehicle = {
        vno: track
        for track, seq in state.track_sequences.items()
        for vno in seq
    }
    for vehicle in plan_input.vehicles:
        allowed = set(vehicle.goal.allowed_target_tracks)
        non_depot = allowed - DEPOT_INNER_TRACKS
        if not non_depot:
            continue
        current = current_by_vehicle.get(vehicle.vehicle_no)
        if current not in allowed:
            return True
    return False


def reorder_depot_late(
    plan: Sequence[HookAction],
    initial_state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[HookAction]:
    """Best-of-two reorder: pure adjacent bubble vs semantic-topo seed +
    adjacent polish. Returns whichever produces smaller depot_earliness.

    Rationale (empirically observed on data/validation_inputs/truth, 127
    scenarios):

    - Pure adjacent bubble (v3) produced -5.9% earliness; it validates
      every swap incrementally so it never produces a worse plan.
    - Pure semantic topo (v4) produced -1.8%; pairwise dep graph is an
      over-approximation, 7/10 of pure-adjacent's top wins got 0 from
      topo because the composite order failed final replay and fell back.
    - Topo-then-adjacent hybrid (v5) produced -3.9%; bubble-sort from
      topo's output reached a worse local optimum than bubble from
      original in many cases.

    Best-of-two guarantees ``result <= pure_adjacent_result`` in earliness,
    and in the handful of scenarios where topo finds a jump adjacent
    can't reach (e.g., a hook must leap past a dep-blocked intermediate
    pair), the topo branch wins.

    Complexity: 2 × O(N^3), same asymptotic bound as v3. Either branch
    fails safe.
    """
    adj_result = _adjacent_swap_polish(plan, initial_state, plan_input)
    topo_result = _semantic_topo_reorder(plan, initial_state, plan_input)
    if topo_result != list(plan):
        # Polish topo output with adjacent bubble — it's always safe and
        # only improves.
        topo_polished = _adjacent_swap_polish(topo_result, initial_state, plan_input)
    else:
        topo_polished = adj_result

    # Return whichever has smallest earliness. Ties → prefer pure adjacent
    # (stable, more predictable for bench-to-bench reproducibility).
    if depot_earliness(topo_polished) < depot_earliness(adj_result):
        return topo_polished
    return adj_result


def _semantic_topo_reorder(
    plan: Sequence[HookAction],
    initial_state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[HookAction]:
    """Greedy topological sort with per-pair semantic dep detection."""
    result = list(plan)
    if not result:
        return result

    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}

    def simulate(sequence: list[HookAction]) -> ReplayState | None:
        state = initial_state
        for move in sequence:
            source_seq = state.track_sequences.get(move.source_track, [])
            if source_seq[: len(move.vehicle_nos)] != move.vehicle_nos:
                return None
            try:
                state = _apply_move(
                    state=state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except (ValueError, KeyError):
                return None
        return state

    baseline_state = simulate(result)
    if baseline_state is None:
        return list(plan)
    baseline_key = _canonicalize_state(baseline_state)

    n = len(result)

    def can_move_j_before_i(i: int, j: int) -> bool:
        modified = list(result[:i]) + [result[j]] + list(result[i:j]) + list(result[j + 1:])
        final = simulate(modified)
        if final is None:
            return False
        return _canonicalize_state(final) == baseline_key

    indegree = [0] * n
    successors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not can_move_j_before_i(i, j):
                successors[i].append(j)
                indegree[j] += 1

    ready = [i for i in range(n) if indegree[i] == 0]
    ordered: list[int] = []
    while ready:
        ready.sort(key=lambda idx: (is_depot_hook(result[idx]), idx))
        pick = ready.pop(0)
        ordered.append(pick)
        for j in successors[pick]:
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)

    if len(ordered) != n:
        return list(plan)

    candidate = [result[i] for i in ordered]
    final_state = simulate(candidate)
    if final_state is None or _canonicalize_state(final_state) != baseline_key:
        return list(plan)
    return candidate


def _adjacent_swap_polish(
    plan: Sequence[HookAction],
    initial_state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[HookAction]:
    """Bubble-sort pass: swap each adjacent (depot, non-depot) pair iff the
    swap replays legally AND reaches the same terminal state. Repeated until
    no more accepted swaps."""
    result = list(plan)
    if not result:
        return result

    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}

    def simulate(sequence: list[HookAction]) -> ReplayState | None:
        state = initial_state
        for move in sequence:
            source_seq = state.track_sequences.get(move.source_track, [])
            if source_seq[: len(move.vehicle_nos)] != move.vehicle_nos:
                return None
            try:
                state = _apply_move(
                    state=state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except (ValueError, KeyError):
                return None
        return state

    baseline_state = simulate(result)
    if baseline_state is None:
        return list(plan)
    baseline_key = _canonicalize_state(baseline_state)

    improved = True
    while improved:
        improved = False
        for i in range(len(result) - 1):
            a, b = result[i], result[i + 1]
            if is_depot_hook(a) and not is_depot_hook(b):
                candidate = list(result)
                candidate[i], candidate[i + 1] = b, a
                new_state = simulate(candidate)
                if new_state is None:
                    continue
                if _canonicalize_state(new_state) != baseline_key:
                    continue
                result = candidate
                improved = True
    return result


def _canonicalize_state(state: ReplayState) -> tuple:
    """Hashable snapshot of a ReplayState for equivalence comparison.

    Excludes ``loco_track_name`` because it trivially tracks the last
    hook's target and naturally differs under a permutation — equivalence
    here means "same vehicle placement, same weigh set, same spot
    assignments," not "identical ReplayState."
    """
    tracks = tuple(
        (name, tuple(seq)) for name, seq in sorted(state.track_sequences.items())
    )
    weighed = tuple(sorted(state.weighed_vehicle_nos))
    spots = tuple(sorted(state.spot_assignments.items()))
    return (tracks, weighed, spots)
