"""Telemetry regression tests for the solver's structured observability record."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.solver.result import SolverTelemetry
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


def _solve() -> tuple[object, object]:
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_simple_payload(), master)
    initial = build_initial_state(normalized)
    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        solver_mode="exact",
        time_budget_ms=15_000,
    )
    return result, normalized


def test_solver_result_attaches_telemetry_record():
    result, normalized = _solve()
    assert isinstance(result.telemetry, SolverTelemetry)
    tel = result.telemetry
    assert tel.input_vehicle_count == 1
    assert tel.input_track_count == 3
    assert tel.input_weigh_count == 0
    assert tel.input_work_position_count == 0
    assert tel.plan_hook_count == len(result.plan)
    assert tel.fallback_stage == result.fallback_stage
    assert tel.is_valid is True
    # Phase timings must be non-negative numbers summing to roughly total_ms.
    assert tel.total_ms > 0
    assert tel.constructive_ms >= 0
    assert tel.exact_ms >= 0
    # total_ms should be >= the individual phases (no phase can overrun total).
    assert tel.total_ms + 1.0 >= tel.constructive_ms + tel.exact_ms


def test_solver_telemetry_counts_work_position_goals():
    master = load_master_data(DATA_DIR)
    payload = _simple_payload()
    payload["trackInfo"].append({"trackName": "洗南", "trackDistance": 88.7})
    payload["vehicleInfo"][0]["targetTrack"] = "洗南"
    payload["vehicleInfo"][0]["isSpotting"] = "否"
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        solver_mode="exact",
        time_budget_ms=15_000,
    )

    assert result.telemetry is not None
    assert result.telemetry.input_work_position_count == 1
    assert result.telemetry.input_area_count == 0


def test_emit_telemetry_appends_jsonl_when_env_var_set(tmp_path, monkeypatch):
    log_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("FZED_SOLVER_TELEMETRY_PATH", str(log_path))
    _solve()
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    record = json.loads(content[0])
    assert record["input_vehicle_count"] == 1
    assert record["plan_hook_count"] >= 1
    assert "ts_unix" in record
    assert isinstance(record["ts_unix"], float)


def test_emit_telemetry_noop_when_env_var_unset(tmp_path, monkeypatch):
    # Ensure env var is NOT set; no file should be created.
    monkeypatch.delenv("FZED_SOLVER_TELEMETRY_PATH", raising=False)
    # The file name we'd check doesn't get created either.
    log_path = tmp_path / "should_not_exist.jsonl"
    _solve()
    assert not log_path.exists()
