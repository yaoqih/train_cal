from pathlib import Path
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import (
    PlanVerificationError,
    SolverResult,
    solve_with_simple_astar_result,
)
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _payload():
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }


def test_exact_solve_marks_proven_optimal_when_budget_not_exhausted():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload(), master)
    initial = build_initial_state(normalized)
    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
    )
    assert result.plan
    assert result.is_proven_optimal is True
    assert result.fallback_stage == "exact"
    assert result.verification_report is not None
    assert result.verification_report.is_valid is True


def test_budget_exhaustion_triggers_anytime_fallback_chain_and_returns_valid_plan():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload(), master)
    initial = build_initial_state(normalized)

    from fzed_shunting.solver import astar_solver as astar_module
    original_search = astar_module._solve_search_result
    call_kwargs_history: list[dict] = []

    def tracking_search(*args, **kwargs):
        call_kwargs_history.append(dict(kwargs))
        solver_mode = kwargs.get("solver_mode")
        if solver_mode == "exact":
            budget = kwargs.get("budget")
            if budget is not None:
                budget.time_budget_ms = 0.0
        return original_search(*args, **kwargs)

    with patch.object(astar_module, "_solve_search_result", side_effect=tracking_search):
        result = solve_with_simple_astar_result(
            normalized,
            initial,
            master=master,
            time_budget_ms=5000.0,
            enable_anytime_fallback=True,
        )
    assert result.plan, "fallback chain must return some plan"
    assert result.verification_report is not None
    assert result.verification_report.is_valid is True
    solver_modes_invoked = [entry.get("solver_mode") for entry in call_kwargs_history]
    assert "exact" in solver_modes_invoked
    assert any(mode in {"weighted", "beam"} for mode in solver_modes_invoked)


def test_verify_false_skips_verification_report():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload(), master)
    initial = build_initial_state(normalized)
    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        verify=False,
    )
    assert result.plan
    assert result.verification_report is None


def test_verify_requires_master():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload(), master)
    initial = build_initial_state(normalized)
    try:
        solve_with_simple_astar_result(
            normalized,
            initial,
            master=None,
            verify=True,
        )
    except ValueError as exc:
        assert "verify=True requires master" in str(exc)
    else:
        raise AssertionError("expected ValueError for verify=True without master")


def test_plan_verification_error_raised_on_injected_bad_plan():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload(), master)
    initial = build_initial_state(normalized)

    from fzed_shunting.solver.types import HookAction

    def fake_search(*args, **kwargs):
        return SolverResult(
            plan=[
                HookAction(
                    source_track="存5北",
                    target_track="机库",
                    vehicle_nos=["DOES_NOT_EXIST"],
                    path_tracks=["存5北", "机库"],
                    action_type="DETACH",
                )
            ],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            is_proven_optimal=True,
            fallback_stage="exact",
        )

    with patch(
        "fzed_shunting.solver.astar_solver._solve_search_result",
        side_effect=fake_search,
    ):
        raised = False
        try:
            solve_with_simple_astar_result(
                normalized,
                initial,
                master=master,
                verify=True,
            )
        except PlanVerificationError:
            raised = True
    assert raised, "PlanVerificationError expected when verify=True on invalid plan"
