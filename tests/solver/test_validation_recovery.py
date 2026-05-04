from __future__ import annotations

from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.validation_recovery import solve_with_validation_recovery_result


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
    assert calls[1]["time_budget_ms"] == 55_000.0
    assert calls[2]["time_budget_ms"] == 55_000.0
    assert calls[3]["time_budget_ms"] == 55_000.0
    assert calls[4]["time_budget_ms"] == 55_000.0


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

    assert [call["beam_width"] for call in calls] == [8, 8]
    assert calls[1]["time_budget_ms"] == 54_000.0
    assert result.is_complete is False


def test_validation_recovery_stops_when_total_retry_budget_is_spent():
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

    assert [call["beam_width"] for call in calls] == [8]


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
