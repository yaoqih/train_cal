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
    """Semantic-dep topological reorder that pushes depot hooks toward the tail.

    Builds a semantic dependency graph over the plan's hooks (O(N^2) pair
    tests, each doing a full O(N) replay), then performs a greedy
    topological sort preferring non-depot hooks. This is strictly stronger
    than adjacent-swap bubble: any reorder reachable by a chain of
    state-preserving adjacent swaps is also a valid topological order under
    the semantic dep graph, and the topo approach can additionally jump a
    hook past a dep-blocked intermediate pair that adjacent-swap can't
    cross.

    Algorithm:
    1. Baseline simulate the original plan and canonicalize its terminal
       state. If the input is itself invalid, bail out unchanged.
    2. For each pair (i, j) with i < j, test whether hook j can be placed
       at position i (shifting plan[i..j-1] one slot right) while the
       modified plan still replays legally AND reaches the same terminal
       state. If not, add edge i -> j (j must stay after i).
    3. Greedy topological sort: at each step pick the ready hook with the
       lowest (is_depot, original_index) key — non-depot first, tiebreak
       by original position.
    4. Validate the final order by replaying from the initial state; if
       the terminal state diverges or replay fails, fall back to the
       original plan.

    Complexity: O(N^3). On any failure anywhere in the pipeline the
    function returns ``list(plan)`` unchanged.
    """
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
        """True iff hook at index j can be placed at position i while
        preserving terminal state (track_sequences, weighed, spots)."""
        modified = list(result[:i]) + [result[j]] + list(result[i:j]) + list(result[j + 1:])
        final = simulate(modified)
        if final is None:
            return False
        return _canonicalize_state(final) == baseline_key

    # Build semantic dep graph.
    indegree = [0] * n
    successors: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not can_move_j_before_i(i, j):
                successors[i].append(j)
                indegree[j] += 1

    # Greedy topological sort with non-depot preference.
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
        # Cycle — shouldn't happen with a valid DAG; fall back.
        return list(plan)

    candidate = [result[i] for i in ordered]

    # Final safety check.
    final_state = simulate(candidate)
    if final_state is None or _canonicalize_state(final_state) != baseline_key:
        return list(plan)
    return candidate


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
