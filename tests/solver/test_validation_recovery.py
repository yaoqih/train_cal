from __future__ import annotations

from fzed_shunting.solver.partial_selection import partial_result_is_better
from fzed_shunting.solver.profile import (
    VALIDATION_TOTAL_TIMEOUT_SECONDS,
    prioritized_validation_recovery_beam_widths,
    validation_recovery_should_continue_after_success,
    validation_retry_time_budget_ms,
    validation_time_budget_ms,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.types import HookAction
from fzed_shunting.solver.validation_recovery import solve_with_validation_recovery_result


def _move(vehicle_no: str) -> HookAction:
    return HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=[vehicle_no],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )


def test_validation_recovery_profile_flags_staging_chain_churn() -> None:
    assert validation_recovery_should_continue_after_success(
        hook_count=191,
        max_vehicle_touch_count=40,
        staging_to_staging_hook_count=23,
        rehandled_vehicle_count=35,
    )


def test_validation_recovery_profile_ignores_high_hook_low_churn_plan() -> None:
    assert not validation_recovery_should_continue_after_success(
        hook_count=269,
        max_vehicle_touch_count=56,
        staging_to_staging_hook_count=2,
        rehandled_vehicle_count=20,
    )


def test_validation_profile_reserves_tail_recovery_budget_under_three_minute_cap() -> None:
    primary_budget = validation_time_budget_ms(VALIDATION_TOTAL_TIMEOUT_SECONDS)

    assert primary_budget == 75_000.0
    assert validation_retry_time_budget_ms(primary_budget) == 105_000.0


def test_validation_recovery_profile_prioritizes_widest_beam_under_three_minute_cap() -> None:
    primary_budget = validation_time_budget_ms(VALIDATION_TOTAL_TIMEOUT_SECONDS)

    assert prioritized_validation_recovery_beam_widths(
        [8, 16, 24, 32],
        base_beam_width=8,
        time_budget_ms=primary_budget,
    ) == [32, 24, 16, 8]


def test_validation_recovery_keeps_structurally_better_partial_over_longer_prefix():
    calls: list[dict] = []
    worse_longer = SolverResult(
        plan=[],
        partial_plan=[_move("WORSE")] * 100,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=100.0,
        is_complete=False,
        partial_fallback_stage="goal_frontier_tail_completion",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 38,
                "work_position_unfinished_count": 7,
                "front_blocker_count": 8,
                "goal_track_blocker_count": 12,
                "staging_debt_count": 2,
                "loco_carry_count": 9,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 5},
        },
    )
    better_shorter = SolverResult(
        plan=[],
        partial_plan=[_move("BETTER")] * 78,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=120_000.0,
        is_complete=False,
        partial_fallback_stage="constructive_partial",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 30,
                "work_position_unfinished_count": 7,
                "front_blocker_count": 6,
                "goal_track_blocker_count": 12,
                "staging_debt_count": 2,
                "loco_carry_count": 0,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 2},
        },
    )

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return worse_longer if len(calls) == 1 else better_shorter

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert result is better_shorter


def test_partial_selection_prefers_lower_target_sequence_defect_before_hook_count():
    move = _move("SEQ")
    incumbent = SolverResult(
        plan=[],
        partial_plan=[move] * 20,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=1.0,
        is_complete=False,
        partial_fallback_stage="goal_frontier_tail_completion",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 4,
                "target_sequence_defect_count": 3,
                "work_position_unfinished_count": 1,
                "front_blocker_count": 0,
                "goal_track_blocker_count": 0,
                "staging_debt_count": 0,
                "area_random_unfinished_count": 0,
                "capacity_overflow_track_count": 0,
                "loco_carry_count": 0,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 0},
        },
    )
    candidate = SolverResult(
        plan=[],
        partial_plan=[move] * 24,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=1.0,
        is_complete=False,
        partial_fallback_stage="goal_frontier_tail_completion",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 4,
                "target_sequence_defect_count": 0,
                "work_position_unfinished_count": 1,
                "front_blocker_count": 0,
                "goal_track_blocker_count": 0,
                "staging_debt_count": 0,
                "area_random_unfinished_count": 0,
                "capacity_overflow_track_count": 0,
                "loco_carry_count": 0,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 0},
        },
    )

    assert partial_result_is_better(candidate, incumbent)


def test_partial_selection_charges_unresolved_route_pressure_as_real_tail_work():
    move = _move("ROUTE")
    incumbent = SolverResult(
        plan=[],
        partial_plan=[move] * 12,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=1.0,
        is_complete=False,
        partial_fallback_stage="goal_frontier_tail_completion",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 12,
                "target_sequence_defect_count": 0,
                "work_position_unfinished_count": 0,
                "front_blocker_count": 2,
                "goal_track_blocker_count": 2,
                "staging_debt_count": 0,
                "area_random_unfinished_count": 0,
                "capacity_overflow_track_count": 0,
                "loco_carry_count": 0,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 12},
        },
    )
    candidate = SolverResult(
        plan=[],
        partial_plan=[move] * 14,
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=1.0,
        is_complete=False,
        partial_fallback_stage="goal_frontier_tail_completion",
        debug_stats={
            "partial_structural_metrics": {
                "unfinished_count": 13,
                "target_sequence_defect_count": 0,
                "work_position_unfinished_count": 0,
                "front_blocker_count": 2,
                "goal_track_blocker_count": 2,
                "staging_debt_count": 0,
                "area_random_unfinished_count": 0,
                "capacity_overflow_track_count": 0,
                "loco_carry_count": 0,
            },
            "partial_route_blockage_plan": {"total_blockage_pressure": 0},
        },
    )

    assert partial_result_is_better(candidate, incumbent)


def test_validation_recovery_uses_bounded_remaining_retry_budget():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[],
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=54_500.0 if len(calls) == 1 else 100.0,
            is_complete=False,
            fallback_stage=kwargs.get("solver_mode"),
        )

    solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert [call["beam_width"] for call in calls] == [8, 8, 16, 24, 32]
    assert calls[1]["time_budget_ms"] == 110_000.0
    assert calls[2]["time_budget_ms"] == 109_900.0
    assert calls[3]["time_budget_ms"] == 109_800.0
    assert calls[4]["time_budget_ms"] == 109_700.0


def test_validation_recovery_tries_same_beam_first_for_route_clean_tail():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) == 1:
            return SolverResult(
                plan=[],
                partial_plan=[_move("TAIL")] * 42,
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=75_000.0,
                is_complete=False,
                fallback_stage=kwargs.get("solver_mode"),
                partial_fallback_stage="route_blockage_tail_clearance",
                debug_stats={
                    "partial_structural_metrics": {
                        "unfinished_count": 8,
                        "work_position_unfinished_count": 6,
                        "front_blocker_count": 2,
                        "staging_debt_count": 0,
                        "capacity_overflow_track_count": 1,
                        "loco_carry_count": 0,
                    },
                    "partial_route_blockage_plan": {
                        "total_blockage_pressure": 0,
                    },
                },
            )
        return SolverResult(
            plan=[_move("DONE")] * 60,
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=12.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={"plan_shape_metrics": {"max_vehicle_touch_count": 8}},
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=validation_time_budget_ms(VALIDATION_TOTAL_TIMEOUT_SECONDS),
        solve_result_fn=fake_solve_result_fn,
    )

    assert [call["beam_width"] for call in calls] == [8, 8]
    assert calls[1]["near_goal_partial_resume_max_final_heuristic"] == 10
    assert result.is_complete is True
    assert len(result.plan) == 60


def test_validation_recovery_caps_total_retry_budget_at_three_minutes():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[],
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=100.0,
            is_complete=False,
            fallback_stage=kwargs.get("solver_mode"),
        )

    solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=110_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert calls[1]["time_budget_ms"] == 70_000.0


def test_validation_recovery_does_not_starve_retries_after_primary_budget_is_spent():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[],
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=56_000.0,
            is_complete=False,
            fallback_stage=kwargs.get("solver_mode"),
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert [call["beam_width"] for call in calls] == [8, 8, 16]
    assert calls[1]["time_budget_ms"] == 110_000.0
    assert calls[2]["time_budget_ms"] == 54_000.0
    assert result.is_complete is False


def test_validation_recovery_stops_after_recovery_budget_is_spent():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[],
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=120_000.0,
            is_complete=False,
            fallback_stage=kwargs.get("solver_mode"),
        )

    solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert [call["beam_width"] for call in calls] == [8, 8]
    assert calls[1]["time_budget_ms"] == 110_000.0


def test_validation_recovery_keeps_initial_solve_on_solver_default_profile():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[],
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=1.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
        )

    solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert "near_goal_partial_resume_max_final_heuristic" not in calls[0]


def test_validation_recovery_does_not_retry_pathological_success_by_default():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return SolverResult(
            plan=[_move("PATHOLOGICAL")] * 120,
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=1.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 120,
                    "staging_to_staging_hook_count": 20,
                }
            },
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
    )

    assert [call["beam_width"] for call in calls] == [8]
    assert result.is_complete is True


def test_validation_recovery_retries_pathological_success_when_requested():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        hook_count = 120 if len(calls) == 1 else 40
        return SolverResult(
            plan=[_move("PATHOLOGICAL")] * hook_count,
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=1.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 120 if len(calls) == 1 else 8,
                    "staging_to_staging_hook_count": 20 if len(calls) == 1 else 0,
                }
            },
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
        improve_pathological_success=True,
    )

    assert [call["beam_width"] for call in calls] == [8, 8]
    assert len(result.plan) == 40


def test_validation_recovery_keeps_clean_success_over_shorter_churny_retry():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) == 1:
            return SolverResult(
                plan=[_move("CLEAN")] * 150,
                partial_plan=[],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=1.0,
                is_complete=True,
                fallback_stage=kwargs.get("solver_mode"),
                debug_stats={
                    "plan_shape_metrics": {
                        "max_vehicle_touch_count": 22,
                        "staging_to_staging_hook_count": 8,
                        "rehandled_vehicle_count": 18,
                    }
                },
            )
        return SolverResult(
            plan=[_move("CHURN")] * 120,
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=1.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 55,
                    "staging_to_staging_hook_count": 7,
                    "rehandled_vehicle_count": 36,
                }
            },
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=55_000.0,
        solve_result_fn=fake_solve_result_fn,
        improve_pathological_success=True,
    )

    assert [call["beam_width"] for call in calls] == [8, 8]
    assert len(result.plan) == 150


def test_validation_recovery_continues_after_churny_recovery_success_in_quality_mode():
    calls: list[dict] = []

    def fake_solve_result_fn(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) == 1:
            return SolverResult(
                plan=[],
                partial_plan=[_move("TAIL")] * 254,
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=75_000.0,
                is_complete=False,
                fallback_stage=kwargs.get("solver_mode"),
                partial_fallback_stage="route_blockage_tail_clearance",
                debug_stats={
                    "partial_structural_metrics": {
                        "unfinished_count": 9,
                        "staging_debt_count": 3,
                        "work_position_unfinished_count": 2,
                        "front_blocker_count": 2,
                        "target_sequence_defect_count": 3,
                        "goal_track_blocker_count": 0,
                        "loco_carry_count": 0,
                    },
                    "partial_route_blockage_plan": {"total_blockage_pressure": 0},
                },
            )
        if len(calls) == 2:
            return SolverResult(
                plan=[_move("CHURN")] * 333,
                partial_plan=[],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=94_000.0,
                is_complete=True,
                fallback_stage=kwargs.get("solver_mode"),
                debug_stats={
                    "plan_shape_metrics": {
                        "staging_hook_count": 76,
                        "staging_to_staging_hook_count": 37,
                        "rehandled_vehicle_count": 45,
                        "max_vehicle_touch_count": 62,
                    }
                },
            )
        return SolverResult(
            plan=[_move("COMPACT")] * 156,
            partial_plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=71_000.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={
                "plan_shape_metrics": {
                    "staging_hook_count": 36,
                    "staging_to_staging_hook_count": 18,
                    "rehandled_vehicle_count": 35,
                    "max_vehicle_touch_count": 40,
                }
            },
        )

    result = solve_with_validation_recovery_result(
        plan_input=None,  # type: ignore[arg-type]
        initial_state=None,  # type: ignore[arg-type]
        master=None,
        solver_mode="beam",
        heuristic_weight=1.0,
        beam_width=8,
        time_budget_ms=75_000.0,
        solve_result_fn=fake_solve_result_fn,
        improve_pathological_success=True,
    )

    assert [call["beam_width"] for call in calls] == [8, 8, 16]
    assert len(result.plan) == 156
