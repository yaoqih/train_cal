from pathlib import Path
import json
import subprocess
import sys

from typer.testing import CliRunner

from fzed_shunting.cli import app, _solve_payload
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.solver.profile import (
    VALIDATION_DEFAULT_BEAM_WIDTH,
    VALIDATION_DEFAULT_SOLVER,
    VALIDATION_DEFAULT_TIMEOUT_SECONDS,
    validation_retry_time_budget_ms,
    validation_time_budget_ms,
)
from fzed_shunting.solver.astar_solver import (
    RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.types import HookAction


runner = CliRunner()


def _mock_hook(source_track: str, target_track: str, vehicle_nos: list[str]) -> HookAction:
    return HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        path_tracks=[source_track, target_track],
        action_type="DETACH",
    )


def _native_direct_plan(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    detach_path_tracks: list[str],
) -> list[dict]:
    return [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": [source_track],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": source_track,
            "targetTrack": target_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": detach_path_tracks,
        },
    ]


def test_generate_and_solve_cli_flow(tmp_path: Path):
    scenario_path = tmp_path / "scenario.json"
    result = runner.invoke(
        app,
        [
            "generate-micro",
            "--output",
            str(scenario_path),
            "--seed",
            "21",
            "--vehicle-count",
            "4",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert scenario_path.exists()

    solve_result = runner.invoke(app, ["solve", "--input", str(scenario_path)])
    assert solve_result.exit_code == 0, solve_result.stdout
    payload = json.loads(solve_result.stdout)
    assert "hook_plan" in payload
    assert "final_state" in payload
    if payload["hook_plan"]:
        first_hook = payload["hook_plan"][0]
        assert first_hook["vehicleCount"] == len(first_hook["vehicleNos"])
        assert "remark" in first_hook


def test_solve_cli_defaults_match_validation_runner_profile(monkeypatch):
    master = load_master_data(Path(__file__).resolve().parents[2] / "data" / "master")
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CLI_PROFILE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    captured = {}

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
        )

    monkeypatch.setattr("fzed_shunting.cli.solve_with_simple_astar_result", fake_solve)

    result = _solve_payload(
        master=master,
        payload=payload,
        solver=None,
        heuristic_weight=1.0,
        beam_width=None,
        time_budget_ms=None,
    )

    assert result["solver"] == VALIDATION_DEFAULT_SOLVER
    assert result["hook_count"] == 0
    assert captured["solver_mode"] == VALIDATION_DEFAULT_SOLVER
    assert captured["beam_width"] == VALIDATION_DEFAULT_BEAM_WIDTH
    assert captured["time_budget_ms"] == validation_time_budget_ms(
        VALIDATION_DEFAULT_TIMEOUT_SECONDS
    )
    assert captured["enable_depot_late_scheduling"] is True


def test_solve_cli_retries_beam_like_validation_runner(monkeypatch):
    master = load_master_data(Path(__file__).resolve().parents[2] / "data" / "master")
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CLI_RETRY",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    calls = []

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) < 3:
            return SolverResult(
                plan=[],
                partial_plan=[],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=0.0,
                is_complete=False,
                fallback_stage=kwargs.get("solver_mode"),
            )
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
        )

    monkeypatch.setattr("fzed_shunting.cli.solve_with_simple_astar_result", fake_solve)

    result = _solve_payload(
        master=master,
        payload=payload,
        solver=None,
        heuristic_weight=1.0,
        beam_width=None,
        time_budget_ms=None,
    )

    assert result["is_valid"] is True
    assert result["solver_errors"] == []
    assert [call["beam_width"] for call in calls] == [8, 8, 16]
    primary_budget = validation_time_budget_ms(VALIDATION_DEFAULT_TIMEOUT_SECONDS)
    assert calls[1]["time_budget_ms"] == validation_retry_time_budget_ms(primary_budget)
    assert calls[1]["near_goal_partial_resume_max_final_heuristic"] == (
        RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )


def test_solve_cli_retries_pathological_complete_result(monkeypatch):
    master = load_master_data(Path(__file__).resolve().parents[2] / "data" / "master")
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CLI_PATHO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    calls = []

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) == 1:
            return SolverResult(
                plan=[
                    _mock_hook("存5北", "存5北", ["CLI_PATHO"])
                    for _ in range(130)
                ],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=0.0,
                is_complete=True,
                fallback_stage=kwargs.get("solver_mode"),
                debug_stats={
                    "plan_shape_metrics": {
                        "max_vehicle_touch_count": 90,
                    }
                },
            )
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={"plan_shape_metrics": {"max_vehicle_touch_count": 20}},
        )

    monkeypatch.setattr("fzed_shunting.cli.solve_with_simple_astar_result", fake_solve)

    result = _solve_payload(
        master=master,
        payload=payload,
        solver=None,
        heuristic_weight=1.0,
        beam_width=None,
        time_budget_ms=None,
    )

    assert result["is_valid"] is True
    assert result["hook_count"] == 0
    assert [call["beam_width"] for call in calls] == [8, 8]
    assert calls[1]["near_goal_partial_resume_max_final_heuristic"] == (
        RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )


def test_solve_cli_supports_weighted_solver(tmp_path: Path):
    scenario_path = tmp_path / "scenario_weighted.json"
    runner.invoke(
        app,
        [
            "generate-micro",
            "--output",
            str(scenario_path),
            "--seed",
            "21",
            "--vehicle-count",
            "4",
        ],
    )

    solve_result = runner.invoke(
        app,
        [
            "solve",
            "--input",
            str(scenario_path),
            "--solver",
            "weighted",
            "--heuristic-weight",
            "2.0",
        ],
    )

    assert solve_result.exit_code == 0, solve_result.stdout
    payload = json.loads(solve_result.stdout)
    assert payload["solver"] == "weighted"


def test_python_m_cli_executes_typer_app(tmp_path: Path):
    scenario_path = tmp_path / "scenario_module.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fzed_shunting.cli",
            "generate-micro",
            "--output",
            str(scenario_path),
            "--seed",
            "11",
            "--vehicle-count",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert scenario_path.exists()


def test_generate_scenario_cli_supports_profile(tmp_path: Path):
    scenario_path = tmp_path / "scenario_profile.json"

    result = runner.invoke(
        app,
        [
            "generate-scenario",
            "--output",
            str(scenario_path),
            "--profile",
            "adversarial",
            "--seed",
            "17",
            "--vehicle-count",
            "4",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(scenario_path.read_text())
    assert payload["scenarioMeta"]["profile"] == "adversarial"


def test_benchmark_cli_supports_beam_solver(tmp_path: Path):
    output_dir = tmp_path / "benchmark_beam"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--output-dir",
            str(output_dir),
            "--scenario-count",
            "2",
            "--vehicle-count",
            "4",
            "--solver",
            "beam",
            "--beam-width",
            "8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["solver"] == "beam"


def test_solve_cli_supports_lns_solver(tmp_path: Path):
    scenario_path = tmp_path / "scenario_lns.json"
    runner.invoke(
        app,
        [
            "generate-micro",
            "--output",
            str(scenario_path),
            "--seed",
            "21",
            "--vehicle-count",
            "4",
        ],
    )

    solve_result = runner.invoke(
        app,
        [
            "solve",
            "--input",
            str(scenario_path),
            "--solver",
            "lns",
            "--beam-width",
            "8",
        ],
    )

    assert solve_result.exit_code == 0, solve_result.stdout
    payload = json.loads(solve_result.stdout)
    assert payload["solver"] == "lns"


def test_verify_cli_returns_hook_reports(tmp_path: Path):
    scenario_path = tmp_path / "scenario_verify.json"
    scenario_path.write_text(
        json.dumps(
            {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": 367},
                    {"trackName": "存5南", "trackDistance": 156},
                    {"trackName": "修1", "trackDistance": 151.7},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "C1",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "targetTrack": "大库",
                        "isSpotting": "101",
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5南",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "C2",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存5南",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        )
    )
    plan_path = tmp_path / "plan_verify.json"
    plan_path.write_text(
        json.dumps(
            _native_direct_plan(
                source_track="存5北",
                target_track="修1",
                vehicle_nos=["C1"],
                detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1"],
            ),
            ensure_ascii=False,
        )
    )

    result = runner.invoke(app, ["verify", "--input", str(scenario_path), "--plan", str(plan_path)])

    assert result.exit_code == 0, result.stdout


def test_solve_suite_cli_solves_paint_and_shot_scenarios(tmp_path: Path):
    suite_path = tmp_path / "typical_suite.json"
    output_dir = tmp_path / "typical_suite_solved"

    generate_result = runner.invoke(
        app,
        [
            "generate-typical-suite",
            "--output",
            str(suite_path),
        ],
    )
    assert generate_result.exit_code == 0, generate_result.stdout

    solve_result = runner.invoke(
        app,
        [
            "solve-suite",
            "--input",
            str(suite_path),
            "--output-dir",
            str(output_dir),
            "--solver",
            "lns",
            "--beam-width",
            "8",
        ],
    )

    assert solve_result.exit_code == 0, solve_result.stdout
    payload = json.loads(solve_result.stdout)
    assert payload["scenarioCount"] == 13
    by_name = {item["name"]: item for item in payload["results"]}
    assert by_name["cun4nan_staging"]["isValid"] is True
    assert by_name["wash_work_area"]["isValid"] is True
    assert by_name["wheel_operate"]["isValid"] is True
    assert by_name["paint_work_area"]["isValid"] is True
    assert by_name["shot_work_area"]["isValid"] is True
    assert by_name["paint_work_area"]["solverErrors"] == []
    assert by_name["shot_work_area"]["solverErrors"] == []

    plan_payload = json.loads((output_dir / "paint_work_area.plan.json").read_text(encoding="utf-8"))
    assert plan_payload["is_valid"] is True
    assert plan_payload["solver_errors"] == []


def test_solve_suite_cli_exports_per_scenario_results(tmp_path: Path):
    suite_path = tmp_path / "typical_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite": "typical",
                "scenarioCount": 2,
                "scenarios": [
                    {
                        "name": "single_direct",
                        "description": "single vehicle direct move",
                        "payload": {
                            "trackInfo": [
                                {"trackName": "存5北", "trackDistance": 367},
                                {"trackName": "机库", "trackDistance": 71.6},
                            ],
                            "vehicleInfo": [
                                {
                                    "trackName": "存5北",
                                    "order": "1",
                                    "vehicleModel": "棚车",
                                    "vehicleNo": "S1",
                                    "repairProcess": "段修",
                                    "vehicleLength": 14.3,
                                    "targetTrack": "机库",
                                    "isSpotting": "",
                                    "vehicleAttributes": "",
                                }
                            ],
                            "locoTrackName": "机库",
                        },
                    },
                    {
                        "name": "weigh_then_store",
                        "description": "weigh before final storage",
                        "payload": {
                            "trackInfo": [
                                {"trackName": "存5北", "trackDistance": 367},
                                {"trackName": "机库", "trackDistance": 71.6},
                                {"trackName": "存4北", "trackDistance": 317.8},
                            ],
                            "vehicleInfo": [
                                {
                                    "trackName": "存5北",
                                    "order": "1",
                                    "vehicleModel": "棚车",
                                    "vehicleNo": "S2",
                                    "repairProcess": "段修",
                                    "vehicleLength": 14.3,
                                    "targetTrack": "存4北",
                                    "isSpotting": "",
                                    "vehicleAttributes": "称重",
                                }
                            ],
                            "locoTrackName": "机库",
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite_results"

    result = runner.invoke(
        app,
        [
            "solve-suite",
            "--input",
            str(suite_path),
            "--output-dir",
            str(output_dir),
            "--solver",
            "lns",
            "--beam-width",
            "8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["suite"] == "typical"
    assert payload["scenarioCount"] == 2
    assert payload["solver"] == "lns"
    assert len(payload["results"]) == 2
    assert {item["name"] for item in payload["results"]} == {"single_direct", "weigh_then_store"}
    assert (output_dir / "suite_report.json").exists()
    assert (output_dir / "single_direct.plan.json").exists()
    assert (output_dir / "weigh_then_store.plan.json").exists()


def test_solve_workflow_cli_executes_stage_sequence(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": 367},
                    {"trackName": "调棚", "trackDistance": 174.3},
                    {"trackName": "修1", "trackDistance": 151.7},
                    {"trackName": "存4北", "trackDistance": 317.8},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "CLIW1",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "dispatch_work",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW1", "targetTrack": "调棚", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "depot_spot",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW1", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "departure",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW1", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["solve-workflow", "--input", str(workflow_path)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["stageCount"] == 3
    assert [stage["name"] for stage in payload["stages"]] == ["dispatch_work", "depot_spot", "departure"]
    assert payload["stages"][0]["isValid"] is True
    assert payload["stages"][1]["finalSpotAssignments"] == {"CLIW1": "101"}
    assert payload["stages"][2]["finalTracks"] == ["存4北"]
    assert payload["stages"][0]["verifierErrors"] == []


def test_solve_suite_cli_supports_workflow_suite_input(tmp_path: Path):
    suite_path = tmp_path / "typical_workflow_suite.json"
    output_dir = tmp_path / "workflow_suite_results"

    generate_result = runner.invoke(
        app,
        [
            "generate-typical-workflow-suite",
            "--output",
            str(suite_path),
        ],
    )
    assert generate_result.exit_code == 0, generate_result.stdout

    result = runner.invoke(
        app,
        [
            "solve-suite",
            "--input",
            str(suite_path),
            "--output-dir",
            str(output_dir),
            "--solver",
            "lns",
            "--beam-width",
            "8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["suite"] == "typical_workflow"
    assert payload["scenarioCount"] == 17
    by_name = {item["name"]: item for item in payload["results"]}
    assert by_name["dispatch_depot_departure"]["isValid"] is True
    assert by_name["dispatch_depot_departure"]["stageCount"] == 3
    assert by_name["weigh_then_departure"]["stageCount"] == 2
    assert by_name["multi_vehicle_dispatch_merge"]["hookCount"] >= 4
    assert by_name["wash_depot_departure"]["stageCount"] == 3
    assert by_name["wheel_departure"]["stageCount"] == 2
    assert by_name["paint_depot_departure"]["stageCount"] == 3
    assert by_name["shot_depot_departure"]["stageCount"] == 3
    assert by_name["dispatch_jiku_final"]["stageCount"] == 2
    assert by_name["weigh_jiku_final"]["stageCount"] == 1
    assert by_name["pre_repair_departure"]["stageCount"] == 2
    assert by_name["main_pre_repair_departure"]["stageCount"] == 2
    assert by_name["jipeng_departure"]["stageCount"] == 2
    assert by_name["depot_wheel_departure"]["stageCount"] == 3
    assert by_name["tank_wash_direct_departure"]["stageCount"] == 2
    assert by_name["close_door_departure"]["stageCount"] == 1
    assert by_name["inspection_departure"]["stageCount"] == 2

    plan_payload = json.loads((output_dir / "multi_vehicle_dispatch_merge.plan.json").read_text(encoding="utf-8"))
    assert plan_payload["scenario_type"] == "workflow"
    assert plan_payload["stage_count"] == 3
    assert plan_payload["stages"][-1]["finalTracks"] == ["存4北"]
    assert plan_payload["stages"][0]["verifierErrors"] == []


def test_compare_suite_cli_exports_multi_solver_report(tmp_path: Path):
    suite_path = tmp_path / "typical_suite.json"
    output_dir = tmp_path / "compare_suite"

    generate_result = runner.invoke(
        app,
        [
            "generate-typical-suite",
            "--output",
            str(suite_path),
        ],
    )
    assert generate_result.exit_code == 0, generate_result.stdout

    result = runner.invoke(
        app,
        [
            "compare-suite",
            "--input",
            str(suite_path),
            "--output-dir",
            str(output_dir),
            "--solver",
            "exact",
            "--solver",
            "weighted",
            "--solver",
            "beam",
            "--solver",
            "lns",
            "--weighted-heuristic-weight",
            "2.0",
            "--beam-width",
            "8",
            "--lns-beam-width",
            "8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["scenario_count"] == 13
    assert [item["solver"] for item in payload["solvers"]] == ["exact", "weighted", "beam", "lns"]
    assert payload["solvers"][-1]["validCount"] == 13
    assert (output_dir / "comparison_report.json").exists()


def test_solve_workflow_cli_supports_multi_vehicle_workflow(tmp_path: Path):
    workflow_path = tmp_path / "workflow_multi.json"
    workflow_path.write_text(
        json.dumps(
            {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": 367},
                    {"trackName": "调棚", "trackDistance": 174.3},
                    {"trackName": "修1", "trackDistance": 151.7},
                    {"trackName": "存4北", "trackDistance": 317.8},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "CLIW2A",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "2",
                        "vehicleModel": "棚车",
                        "vehicleNo": "CLIW2B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                ],
                "workflowStages": [
                    {
                        "name": "dispatch_work",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW2A", "targetTrack": "调棚", "isSpotting": ""},
                            {"vehicleNo": "CLIW2B", "targetTrack": "存5北", "isSpotting": ""},
                        ],
                    },
                    {
                        "name": "depot_hold",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW2A", "targetTrack": "大库", "isSpotting": "101"},
                            {"vehicleNo": "CLIW2B", "targetTrack": "存5北", "isSpotting": ""},
                        ],
                    },
                    {
                        "name": "departure",
                        "vehicleGoals": [
                            {"vehicleNo": "CLIW2A", "targetTrack": "存4北", "isSpotting": ""},
                            {"vehicleNo": "CLIW2B", "targetTrack": "存4北", "isSpotting": ""},
                        ],
                    },
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["solve-workflow", "--input", str(workflow_path)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["stageCount"] == 3
    assert payload["stages"][0]["finalSpotAssignments"] == {}
    assert payload["stages"][1]["finalSpotAssignments"] == {"CLIW2A": "101"}
    assert payload["stages"][2]["hookCount"] == 3
    assert payload["stages"][2]["finalTracks"] == ["存4北"]
