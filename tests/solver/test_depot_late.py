"""Unit tests for depot-late-scheduling core helpers."""

from __future__ import annotations

from fzed_shunting.solver.depot_late import (
    DEPOT_INNER_TRACKS,
    depot_earliness,
    depot_index_sum,
    is_depot_hook,
)
from fzed_shunting.solver.types import HookAction


def _hook(source: str, target: str, vehicles: list[str] | None = None) -> HookAction:
    return HookAction(
        source_track=source,
        target_track=target,
        vehicle_nos=vehicles or ["V1"],
        path_tracks=[source, target],
        action_type="PUT",
    )


class TestIsDepotHook:
    def test_target_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("存5北", "修3库内")) is True

    def test_source_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("修1库内", "存4北")) is True

    def test_both_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("修1库内", "修2库内")) is True

    def test_non_depot_is_false(self) -> None:
        assert is_depot_hook(_hook("存1", "调北")) is False

    def test_depot_wai_is_false(self) -> None:
        # 修N库外 is gateway storage, NOT counted as depot.
        assert is_depot_hook(_hook("修1库外", "存4北")) is False

    def test_depot_wai_to_depot_nei_is_true(self) -> None:
        # Crossing into 修N库内 does count.
        assert is_depot_hook(_hook("修1库外", "修1库内")) is True

    def test_constant_exposes_all_four_tracks(self) -> None:
        assert DEPOT_INNER_TRACKS == frozenset(
            {"修1库内", "修2库内", "修3库内", "修4库内"}
        )


class TestDepotMetrics:
    def test_earliness_empty_plan_is_zero(self) -> None:
        assert depot_earliness([]) == 0
        assert depot_index_sum([]) == 0

    def test_earliness_no_depot_hooks_is_zero(self) -> None:
        plan = [_hook("存1", "调北"), _hook("调北", "存4北")]
        assert depot_earliness(plan) == 0
        assert depot_index_sum(plan) == 0

    def test_earliness_single_depot_hook_last(self) -> None:
        # N=3, depot at index 3 -> earliness = (3-3) = 0.
        plan = [
            _hook("存1", "调北"),
            _hook("调北", "预修"),
            _hook("修1库内", "存4北"),
        ]
        assert depot_earliness(plan) == 0
        assert depot_index_sum(plan) == 3

    def test_earliness_single_depot_hook_first(self) -> None:
        # N=3, depot at index 1 -> earliness = (3-1) = 2.
        plan = [
            _hook("修1库内", "存4北"),
            _hook("存1", "调北"),
            _hook("调北", "预修"),
        ]
        assert depot_earliness(plan) == 2
        assert depot_index_sum(plan) == 1

    def test_earliness_multiple_depot_hooks(self) -> None:
        # N=4, depot at indices 1 and 3.
        # earliness = (4-1) + (4-3) = 4; index_sum = 1 + 3 = 4.
        plan = [
            _hook("修1库内", "修1库外"),
            _hook("存1", "调北"),
            _hook("修2库内", "存4北"),
            _hook("调北", "预修"),
        ]
        assert depot_earliness(plan) == 4
        assert depot_index_sum(plan) == 4

    def test_earliness_and_index_sum_relation(self) -> None:
        plan = [
            _hook("修1库内", "存4北"),
            _hook("存1", "调北"),
            _hook("修2库内", "存4北"),
            _hook("调北", "预修"),
            _hook("修3库内", "存4北"),
        ]
        n = len(plan)
        k = sum(1 for h in plan if is_depot_hook(h))
        assert depot_earliness(plan) + depot_index_sum(plan) == k * n


from fzed_shunting.io.normalize_input import (
    GoalSpec,
    NormalizedPlanInput,
    NormalizedTrackInfo,
    NormalizedVehicle,
)
from fzed_shunting.solver.depot_late import _is_early_depot_phase
from fzed_shunting.verify.replay import ReplayState


def _vehicle(
    vno: str,
    current: str,
    targets: list[str],
    *,
    target_mode: str = "TRACK",
    area_code: str | None = None,
) -> NormalizedVehicle:
    return NormalizedVehicle(
        current_track=current,
        order=1,
        vehicle_model="敞车",
        vehicle_no=vno,
        repair_process="段修",
        vehicle_length=12.0,
        goal=GoalSpec(
            target_mode=target_mode,
            target_track=targets[0],
            allowed_target_tracks=targets,
            target_area_code=area_code,
        ),
    )


def _plan_input(vehicles: list[NormalizedVehicle]) -> NormalizedPlanInput:
    tracks = {v.current_track for v in vehicles} | {
        t for v in vehicles for t in v.goal.allowed_target_tracks
    }
    return NormalizedPlanInput(
        track_info=[NormalizedTrackInfo(track_name=t, track_distance=200.0) for t in tracks],
        vehicles=vehicles,
        loco_track_name="机库",
        yard_mode="NORMAL",
    )


def _state(track_sequences: dict[str, list[str]]) -> ReplayState:
    return ReplayState(
        track_sequences=track_sequences,
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )


class TestIsEarlyDepotPhase:
    def test_true_when_non_depot_goal_unmet(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存5北": ["V1"], "存1": ["V2"]})
        assert _is_early_depot_phase(state, plan_input) is True

    def test_false_when_all_non_depot_goals_met(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存4北": ["V1"], "存1": ["V2"]})
        assert _is_early_depot_phase(state, plan_input) is False

    def test_false_when_only_depot_goals_exist(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存5北": ["V1"]})
        assert _is_early_depot_phase(state, plan_input) is False

    def test_false_when_all_allowed_targets_are_depot(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内", "修2库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存1": ["V1"]})
        assert _is_early_depot_phase(state, plan_input) is False


from fzed_shunting.solver.depot_late import reorder_depot_late


class TestReorderDepotLate:
    def test_reorder_swaps_independent_depot_hook_later(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
            _vehicle("V2", "存2", ["存4北"]),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"]})
        plan = [
            _hook("存1", "修1库内", ["V1"]),
            _hook("存2", "存4北", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered[0].target_track == "存4北"
        assert reordered[1].target_track == "修1库内"
        assert depot_earliness(reordered) < depot_earliness(plan)

    def test_reorder_rejects_dependent_swap(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        initial = _state({"存1": ["V1"]})
        plan = [
            _hook("存1", "修1库外", ["V1"]),
            _hook("修1库外", "修1库内", ["V1"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered == plan  # unchanged

    def test_reorder_preserves_empty_plan(self) -> None:
        plan_input = _plan_input([_vehicle("V1", "存1", ["存4北"])])
        initial = _state({"存1": ["V1"]})
        assert reorder_depot_late([], initial, plan_input) == []

    def test_reorder_is_noop_when_all_depot_already_late(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"]})
        plan = [
            _hook("存1", "存4北", ["V1"]),
            _hook("存2", "修1库内", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered == plan

    def test_reorder_handles_two_random_depot_vehicles(self) -> None:
        """Two 大库:RANDOM vehicles; adjacent swap — verify _canonicalize_state
        handles per-vehicle spot assignment correctly.

        Setup: plan = [V1->修1库内, V3->存4北, V2->修1库内] where V1, V2 are
        大库:RANDOM-bound and V3 is non-depot. Baseline replay assigns
        V1=101, V2=102 (the first two spots in 修1库内 NORMAL). Only one
        adjacent (depot, non-depot) pair exists — indices (0, 1) — so the
        algorithm attempts to move V3 before V1.

        Observed behavior (committed branch): reorder SUCCEEDS. The swap
        preserves the relative order of V1 and V2, so allocate_spots_for_block
        deterministically re-assigns V1=101 and V2=102 in both the baseline
        and candidate replays. _canonicalize_state sees identical
        sorted(spot_assignments.items()) tuples and accepts the swap.

        Note on the review concern (Important #1): _canonicalize_state
        compares spot_assignments pair-wise, which WOULD reject any swap
        that flips the relative order of two 大库:RANDOM vehicles (same
        spots occupied, different vehicle->spot mapping). However, the
        current reorder algorithm only swaps (depot, non-depot) adjacent
        pairs, so it CANNOT swap two depot hooks relative to each other —
        making the concern theoretical for this algorithm. If a future
        change (e.g., Wave 3) adds depot-depot swap patterns or block-level
        moves, revisit _canonicalize_state to use a RANDOM-aware
        canonicalization like solver.state._state_key does.
        """
        plan_input = _plan_input([
            _vehicle(
                "V1", "存1", ["修1库内"],
                target_mode="AREA", area_code="大库:RANDOM",
            ),
            _vehicle(
                "V2", "存2", ["修1库内"],
                target_mode="AREA", area_code="大库:RANDOM",
            ),
            _vehicle("V3", "存3", ["存4北"]),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"], "存3": ["V3"]})
        plan = [
            _hook("存1", "修1库内", ["V1"]),
            _hook("存3", "存4北", ["V3"]),
            _hook("存2", "修1库内", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)

        # The swap succeeded: V3 (non-depot) is now first, pushing V1's
        # depot hook from index 1 to index 2. V2 stays at index 3.
        assert [h.target_track for h in reordered] == [
            "存4北",
            "修1库内",
            "修1库内",
        ]
        assert [h.vehicle_nos for h in reordered] == [["V3"], ["V1"], ["V2"]]
        # Earliness strictly improved (V1 moved from index 1 to index 2).
        assert depot_earliness(reordered) < depot_earliness(plan)

    def test_reorder_topological_jumps_past_blocked_swap(self) -> None:
        """Plan [A_depot, B_nondepot_dep_on_A, C_nondepot_independent].

        Adjacent swap cannot reorder: A-B is dep-blocked, B-C is same-class (both non-depot),
        so no swap fires. Topological sort: C is independent and ready first, B depends on A,
        so order becomes [C, A, B]. Depot A ends up at position 2 instead of 1.
        """
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
            _vehicle("V2", "存1", ["存4北"]),  # B moves V2 whose source is 存1 — depends on A removing V1 first
            _vehicle("V3", "存2", ["调北"]),  # independent
        ])
        initial = _state({"存1": ["V1", "V2"], "存2": ["V3"]})
        plan = [
            _hook("存1", "修1库内", ["V1"]),         # A: depot, prereq for B
            _hook("存1", "存4北", ["V2"]),           # B: non-depot, depends on A (source needs V1 gone)
            _hook("存2", "调北", ["V3"]),            # C: non-depot, independent
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        # Expect [C, A, B]: C first (independent), then A (B's prereq), then B
        assert reordered[0].vehicle_nos == ["V3"]
        assert reordered[1].vehicle_nos == ["V1"]
        assert reordered[2].vehicle_nos == ["V2"]
        assert depot_earliness(reordered) < depot_earliness(plan)

    def test_reorder_topological_handles_simple_swap(self) -> None:
        """The existing simple-swap case should still produce the same result."""
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
            _vehicle("V2", "存2", ["存4北"]),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"]})
        plan = [
            _hook("存1", "修1库内", ["V1"]),
            _hook("存2", "存4北", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered[0].vehicle_nos == ["V2"]
        assert reordered[1].vehicle_nos == ["V1"]


from fzed_shunting.solver.search import _priority


class TestSearchPriorityDepotSecondary:
    def test_flag_off_default_priority_unchanged_exact(self) -> None:
        # Baseline parity: neg_depot_index_sum=0 keeps the classic tuple shape.
        legacy = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
        )
        # Position 0: f=cost+h=8; position 1: cost; position 2: neg_depot_key
        # (0 when flag off); position 3: heuristic; position 4: -blocker_bonus.
        assert legacy[0] == 8
        assert legacy[1] == 3
        assert legacy[2] == 0
        assert legacy[3] == 5
        assert legacy[4] == 0

    def test_flag_on_smaller_neg_index_sum_preferred(self) -> None:
        # Two nodes same (cost, heuristic). Node A has depot at index 3
        # (index_sum=3, neg=-3); Node B at index 1 (index_sum=1, neg=-1).
        # A should have smaller priority tuple (more negative secondary).
        priority_a = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-3,
        )
        priority_b = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-1,
        )
        assert priority_a < priority_b

    def test_flag_on_does_not_override_cost(self) -> None:
        # Node A has later depot but higher cost; cost must still win.
        priority_a = _priority(
            cost=4, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-100,
        )
        priority_b = _priority(
            cost=3, heuristic=6, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=0,
        )
        # Both f = 9 (tied on position 0). Cost tie-breaks at position 1.
        # B has cost=3 < A's cost=4, so B < A.
        assert priority_b < priority_a

    def test_beam_mode_preserves_blocker_index(self) -> None:
        # Regression guard for _prune_queue: it reads priority[?] for blocker.
        # After secondary insert, blocker_bonus lives at index 4.
        p = _priority(
            cost=2, heuristic=3, blocker_bonus=1,
            solver_mode="beam", heuristic_weight=1.0,
            neg_depot_index_sum=0,
        )
        # beam: (f, cost, neg_depot, adj_h, -blocker, h)
        assert p[4] == -1


from fzed_shunting.solver.lns import _is_better_plan, _plan_quality


class TestLnsPlanQualityDepotAware:
    def test_flag_off_ignores_earliness(self) -> None:
        # Same hook count, same branch/length → equivalent when flag off.
        plan_early_depot = [_hook("修1库内", "存4北"), _hook("存1", "调北")]
        plan_late_depot = [_hook("存1", "调北"), _hook("修1库内", "存4北")]
        q_early = _plan_quality(plan_early_depot, route_oracle=None, depot_late=False)
        q_late = _plan_quality(plan_late_depot, route_oracle=None, depot_late=False)
        assert q_early == q_late

    def test_flag_on_prefers_later_depot(self) -> None:
        plan_early_depot = [_hook("修1库内", "存4北"), _hook("存1", "调北")]
        plan_late_depot = [_hook("存1", "调北"), _hook("修1库内", "存4北")]
        assert _is_better_plan(plan_late_depot, plan_early_depot, route_oracle=None, depot_late=True) is True
        assert _is_better_plan(plan_early_depot, plan_late_depot, route_oracle=None, depot_late=True) is False

    def test_flag_on_still_minimises_hook_count(self) -> None:
        # Shorter plan wins even if it has an earlier depot hook.
        plan_short_early_depot = [_hook("修1库内", "存4北")]
        plan_long_late_depot = [
            _hook("存1", "调北"),
            _hook("调北", "预修"),
            _hook("修1库内", "存4北"),
        ]
        assert _is_better_plan(
            plan_short_early_depot, plan_long_late_depot, route_oracle=None, depot_late=True
        ) is True


from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.verify.replay import build_initial_state


class TestAstarSolverDepotLateFlag:
    def test_result_exposes_depot_earliness_when_default(self) -> None:
        # Small trivially-solvable input. Default (flag on) populates fields.
        vehicles = [_vehicle("V1", "存1", ["存4北"])]
        plan_input = _plan_input(vehicles)
        initial = build_initial_state(plan_input)
        result = solve_with_simple_astar_result(
            plan_input=plan_input,
            initial_state=initial,
            time_budget_ms=5_000,
            verify=False,
        )
        assert result.depot_earliness is not None
        assert result.depot_hook_count is not None
        assert result.depot_hook_count == 0
        assert result.depot_earliness == 0

    def test_flag_toggle_preserves_hook_count(self) -> None:
        vehicles = [
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ]
        plan_input = _plan_input(vehicles)
        initial = build_initial_state(plan_input)
        off = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=5_000, verify=False,
            enable_depot_late_scheduling=False,
        )
        on = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=5_000, verify=False,
            enable_depot_late_scheduling=True,
        )
        assert len(on.plan) == len(off.plan)
        assert on.depot_earliness <= off.depot_earliness
