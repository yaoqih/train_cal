import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import (
    SolverResult,
    solve_with_simple_astar_result,
)
from fzed_shunting.solver import astar_solver as astar_module
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _simple_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
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
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }


def test_anytime_with_constructive_seed_never_returns_empty_plan_on_solvable_input():
    """SLA regression: simple inputs must always come back with a non-empty plan."""
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_simple_payload(), master)
    initial = build_initial_state(normalized)
    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        solver_mode="exact",
        time_budget_ms=30_000,
        enable_anytime_fallback=True,
        enable_constructive_seed=True,
    )
    assert len(result.plan) > 0
    assert result.verification_report is not None


def test_constructive_seed_salvages_when_exact_and_fallback_all_fail():
    """When every search stage returns no complete plan, a complete constructive seed still wins."""
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_simple_payload(), master)
    initial = build_initial_state(normalized)

    def fake_exhausted_search(*args, **kwargs):
        # Every search stage pretends to find nothing.
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_proven_optimal=False,
            fallback_stage=kwargs.get("solver_mode", "unknown"),
        )

    with patch.object(astar_module, "_solve_search_result", side_effect=fake_exhausted_search):
        result = solve_with_simple_astar_result(
            normalized,
            initial,
            master=master,
            solver_mode="exact",
            time_budget_ms=30_000,
            enable_anytime_fallback=True,
            enable_constructive_seed=True,
        )

    assert result.is_complete is True
    assert len(result.plan) > 0
    assert result.fallback_stage == "constructive"


def test_constructive_seed_disabled_preserves_legacy_no_solution_error():
    """Opt-out: callers can disable the seed to get the original ValueError signal."""
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DUP_A",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "DUP_B",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    with pytest.raises(ValueError, match="No solution found"):
        solve_with_simple_astar_result(
            normalized,
            initial,
            master=master,
            enable_constructive_seed=False,
            enable_anytime_fallback=False,
        )


def test_constructive_partial_attaches_partial_verification_report_without_raising():
    """Partial artifacts surface verifier report without becoming a solved plan."""
    from fzed_shunting.solver.astar_solver import _attach_verification
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_simple_payload(), master)
    initial = build_initial_state(normalized)

    # Craft a deliberate partial: a hook that completes no goal.
    partial_move = HookAction(
        source_track="存5北",
        target_track="机库",
        vehicle_nos=["E1"],
        path_tracks=["存5北", "机库"],
        action_type="DETACH",
    )
    partial = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
        is_proven_optimal=False,
        partial_plan=[partial_move],
        partial_fallback_stage="constructive_partial",
    )
    attached = _attach_verification(
        partial,
        plan_input=normalized,
        master=master,
        initial_state=initial,
    )
    assert attached.verification_report is None
    assert attached.partial_verification_report is not None
    # The partial places E1 on 机库 but goal is 存4北, so verifier must flag it.
    assert attached.partial_verification_report.is_valid is False
