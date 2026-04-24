"""Unit tests for SolverResult data shape."""

from __future__ import annotations

from fzed_shunting.solver.result import SolverResult


def test_solver_result_defaults_depot_fields_to_none() -> None:
    result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
    )
    assert result.is_complete is False
    assert result.partial_plan == []
    assert result.depot_earliness is None
    assert result.depot_hook_count is None


def test_solver_result_accepts_depot_fields() -> None:
    result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
        depot_earliness=4,
        depot_hook_count=2,
    )
    assert result.depot_earliness == 4
    assert result.depot_hook_count == 2


def test_solver_result_supports_complete_zero_hook_solution() -> None:
    result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
        is_complete=True,
        fallback_stage="exact",
    )
    assert result.is_complete is True
    assert result.plan == []
    assert result.partial_plan == []
