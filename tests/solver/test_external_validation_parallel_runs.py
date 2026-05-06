from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from shutil import copy2
import json
from types import SimpleNamespace
from unittest.mock import patch

from fzed_shunting.tools.convert_external_validation_inputs import convert_external_validation_inputs
from fzed_shunting.solver.profile import (
    VALIDATION_DEFAULT_BEAM_WIDTH,
    VALIDATION_DEFAULT_MAX_WORKERS,
    VALIDATION_DEFAULT_SOLVER,
    VALIDATION_DEFAULT_TIMEOUT_SECONDS,
    VALIDATION_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
    validation_time_budget_ms,
)
from fzed_shunting.solver.astar_solver import (
    RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.types import HookAction


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_external_validation_parallel.py"
spec = spec_from_file_location("run_external_validation_parallel", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)
solve_one = module.solve_one


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_WORKBOOK = ROOT_DIR / "取送车计划" / "2月-取送车计划" / "取送车计划_20260227W.xlsx"


def test_solve_one_external_validation_returns_debug_stats(tmp_path: Path):
    source_root = tmp_path / "取送车计划"
    february_dir = source_root / "2月-取送车计划"
    february_dir.mkdir(parents=True)
    copy2(SAMPLE_WORKBOOK, february_dir / SAMPLE_WORKBOOK.name)
    output_dir = tmp_path / "external_validation_inputs"
    convert_external_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )
    scenario_path = output_dir / "validation_20260227W.json"

    result = solve_one(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
    )

    assert result["scenario"] == "validation_20260227W.json"
    assert "debug_stats" in result
    if result["solved"]:
        assert result["expanded_nodes"] >= 1
        assert result["generated_nodes"] >= 1
    else:
        assert "error" in result


def test_solve_one_reports_partial_artifact_as_unsolved(tmp_path: Path):
    scenario_path = tmp_path / "validation_partial.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "PX1",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    partial_result = SolverResult(
        plan=[],
        expanded_nodes=2,
        generated_nodes=3,
        closed_nodes=1,
        elapsed_ms=5.0,
        is_complete=False,
        partial_plan=[
            HookAction(
                source_track="存5北",
                target_track="机库",
                vehicle_nos=["PX1"],
                path_tracks=["存5北", "机库"],
                action_type="DETACH",
            )
        ],
        partial_fallback_stage="constructive_partial",
    )

    with patch.object(module, "solve_with_simple_astar_result", return_value=partial_result):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
        )

    assert result["scenario"] == "validation_partial.json"
    assert result["solved"] is False
    assert result["error"] == "no solution within budget"
    assert result["is_complete"] is False
    assert result["partial_hook_count"] == 1
    assert result["partial_fallback_stage"] == "constructive_partial"


def test_solve_one_can_opt_into_beam_primary_before_constructive_rescue(tmp_path: Path):
    scenario_path = tmp_path / "validation_primary.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "PRIMARY_FIRST",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured: list[bool | None] = []
    primary_result = SolverResult(
        plan=[
            HookAction(
                source_track="存5北",
                target_track="存5北",
                vehicle_nos=["PRIMARY_FIRST"],
                path_tracks=["存5北"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="存5北",
                target_track="存4北",
                vehicle_nos=["PRIMARY_FIRST"],
                path_tracks=["存5北", "存4北"],
                action_type="DETACH",
            ),
        ],
        expanded_nodes=2,
        generated_nodes=2,
        closed_nodes=1,
        elapsed_ms=5.0,
        is_complete=True,
        fallback_stage="beam",
        verification_report=SimpleNamespace(is_valid=True, errors=[]),
    )

    def fake_solve_with_simple_astar_result(*_args, **kwargs):  # noqa: ANN003
        captured.append(kwargs.get("enable_constructive_seed"))
        return primary_result

    with patch.object(
        module,
        "solve_with_simple_astar_result",
        side_effect=fake_solve_with_simple_astar_result,
    ):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            time_budget_ms=50_000.0,
            primary_first_beam=True,
        )

    assert captured == [False]
    assert result["solved"] is True
    assert result["hook_count"] == 2


def test_solve_one_retries_with_constructive_rescue_after_primary_beam_failure(tmp_path: Path):
    scenario_path = tmp_path / "validation_rescue.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "RESCUE_AFTER_PRIMARY",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[tuple[bool | None, float | None, int | None]] = []
    failed_primary = SolverResult(
        plan=[],
        expanded_nodes=10,
        generated_nodes=20,
        closed_nodes=10,
        elapsed_ms=20_000.0,
        is_complete=False,
        fallback_stage="beam",
        debug_stats={"primary": True},
    )
    rescued = SolverResult(
        plan=[
            HookAction(
                source_track="存5北",
                target_track="存5北",
                vehicle_nos=["RESCUE_AFTER_PRIMARY"],
                path_tracks=["存5北"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="存5北",
                target_track="存4北",
                vehicle_nos=["RESCUE_AFTER_PRIMARY"],
                path_tracks=["存5北", "存4北"],
                action_type="DETACH",
            ),
        ],
        expanded_nodes=2,
        generated_nodes=2,
        closed_nodes=1,
        elapsed_ms=8_000.0,
        is_complete=True,
        fallback_stage="constructive",
        verification_report=SimpleNamespace(is_valid=True, errors=[]),
        debug_stats={"rescued": True},
    )

    def fake_solve_with_simple_astar_result(*_args, **kwargs):  # noqa: ANN003
        calls.append((
            kwargs.get("enable_constructive_seed"),
            kwargs.get("time_budget_ms"),
            kwargs.get("near_goal_partial_resume_max_final_heuristic"),
        ))
        return failed_primary if len(calls) == 1 else rescued

    with patch.object(
        module,
        "solve_with_simple_astar_result",
        side_effect=fake_solve_with_simple_astar_result,
    ):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            time_budget_ms=50_000.0,
            primary_first_beam=True,
        )

    assert calls == [
        (False, 20_000.0, None),
        (True, 30_000.0, RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC),
    ]
    assert result["solved"] is True
    assert result["debug_stats"] == {"rescued": True}


def test_solve_one_primary_first_keeps_structurally_better_partial(tmp_path: Path):
    scenario_path = tmp_path / "validation_structural_partial.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "STRUCTURAL_PARTIAL",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    move = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["STRUCTURAL_PARTIAL"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    worse_longer_primary = SolverResult(
        plan=[],
        partial_plan=[move] * 100,
        expanded_nodes=10,
        generated_nodes=20,
        closed_nodes=10,
        elapsed_ms=20_000.0,
        is_complete=False,
        fallback_stage="beam",
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
    better_shorter_rescue = SolverResult(
        plan=[],
        partial_plan=[move] * 78,
        expanded_nodes=2,
        generated_nodes=2,
        closed_nodes=1,
        elapsed_ms=8_000.0,
        is_complete=False,
        fallback_stage="constructive_partial",
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
    calls: list[dict] = []

    def fake_solve_with_simple_astar_result(*_args, **kwargs):  # noqa: ANN003
        calls.append(kwargs)
        return worse_longer_primary if len(calls) == 1 else better_shorter_rescue

    with patch.object(
        module,
        "solve_with_simple_astar_result",
        side_effect=fake_solve_with_simple_astar_result,
    ):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            time_budget_ms=50_000.0,
            primary_first_beam=True,
        )

    assert result["solved"] is False
    assert result["partial_hook_count"] == 78
    assert result["partial_fallback_stage"] == "constructive_partial"
    assert result["debug_stats"] is better_shorter_rescue.debug_stats


def test_solve_one_can_disable_worker_recovery_for_single_attempt(tmp_path: Path):
    scenario_path = tmp_path / "validation_worker_single_attempt.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "WORKER_SINGLE_ATTEMPT",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    partial = SolverResult(
        plan=[],
        expanded_nodes=2,
        generated_nodes=2,
        closed_nodes=1,
        elapsed_ms=49_500.0,
        is_complete=False,
        fallback_stage="beam",
        partial_plan=[
            HookAction(
                source_track="存5北",
                target_track="存5北",
                vehicle_nos=["WORKER_SINGLE_ATTEMPT"],
                path_tracks=["存5北"],
                action_type="ATTACH",
            ),
        ],
        partial_fallback_stage="beam",
        debug_stats={"partial_structural_metrics": {"unfinished_count": 1}},
    )
    captured: list[dict] = []

    def fake_solve_with_simple_astar_result(*_args, **kwargs):  # noqa: ANN003
        captured.append(kwargs)
        return partial

    with patch.object(
        module,
        "solve_with_simple_astar_result",
        side_effect=fake_solve_with_simple_astar_result,
    ):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            time_budget_ms=50_000.0,
            enable_worker_recovery=False,
        )

    assert result["solved"] is False
    assert result["error_category"] == "no_solution"
    assert result["partial_hook_count"] == 1
    assert len(captured) == 1
    assert captured[0]["solver_mode"] == "beam"
    assert captured[0]["beam_width"] == 8
    assert captured[0]["time_budget_ms"] == 50_000.0


def test_solve_one_can_run_validation_recovery_inside_worker(tmp_path: Path):
    scenario_path = tmp_path / "validation_worker_recovery.json"
    scenario_path.write_text(
        json.dumps(
            {
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
                        "vehicleNo": "WORKER_RECOVERY",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    recovered = SolverResult(
        plan=[
            HookAction(
                source_track="存5北",
                target_track="存5北",
                vehicle_nos=["WORKER_RECOVERY"],
                path_tracks=["存5北"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="存5北",
                target_track="存4北",
                vehicle_nos=["WORKER_RECOVERY"],
                path_tracks=["存5北", "存4北"],
                action_type="DETACH",
            ),
        ],
        expanded_nodes=2,
        generated_nodes=2,
        closed_nodes=1,
        elapsed_ms=42.0,
        is_complete=True,
        fallback_stage="beam",
        verification_report=SimpleNamespace(is_valid=True, errors=[]),
        debug_stats={"validation_recovery": {"recovery_beam_width": 8}},
    )
    captured: dict = {}

    def fake_validation_recovery(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return recovered

    with patch.object(
        module,
        "solve_with_validation_recovery_result",
        side_effect=fake_validation_recovery,
    ):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            time_budget_ms=75_000.0,
            enable_worker_recovery=True,
        )

    assert result["solved"] is True
    assert result["hook_count"] == 2
    assert result["recovery_beam_width"] == 8
    assert captured["solver_mode"] == "beam"
    assert captured["beam_width"] == 8
    assert captured["time_budget_ms"] == 75_000.0
    assert captured["enable_depot_late_scheduling"] is True


def test_run_worker_passes_primary_first_beam_flag(tmp_path: Path, capsys):
    scenario_path = tmp_path / "validation_primary_worker.json"
    scenario_path.write_text("{}", encoding="utf-8")
    captured: list[bool] = []

    def fake_solve_one(**kwargs):  # noqa: ANN003
        captured.append(kwargs["primary_first_beam"])
        return {"scenario": scenario_path.name, "solved": True}

    with patch.object(module, "solve_one", side_effect=fake_solve_one):
        module._run_worker(
            SimpleNamespace(
                master_dir=DATA_DIR,
                scenario=scenario_path,
                solver="beam",
                beam_width=8,
                heuristic_weight=1.0,
                solver_time_budget_ms=50_000.0,
                enable_anytime_fallback=True,
                enable_depot_late_scheduling=False,
                near_goal_partial_resume_max_final_heuristic=None,
                primary_first_beam=True,
                enable_worker_recovery=False,
            )
        )

    assert captured == [True]
    assert json.loads(capsys.readouterr().out)["solved"] is True


def test_build_worker_command_passes_depot_late_disable_only_when_requested(tmp_path: Path):
    scenario_path = tmp_path / "validation_a.json"
    default_cmd = module._build_worker_command(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
        time_budget_ms=50_000.0,
    )
    disabled_cmd = module._build_worker_command(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
        time_budget_ms=50_000.0,
        enable_depot_late_scheduling=False,
    )

    assert "--disable-depot-late-scheduling" not in default_cmd
    assert "--disable-depot-late-scheduling" in disabled_cmd


def test_build_worker_command_passes_worker_recovery_enable_only_when_requested(tmp_path: Path):
    scenario_path = tmp_path / "validation_a.json"
    default_cmd = module._build_worker_command(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
        time_budget_ms=75_000.0,
    )
    disabled_cmd = module._build_worker_command(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
        time_budget_ms=75_000.0,
        enable_worker_recovery=True,
    )

    assert "--enable-worker-recovery" not in default_cmd
    assert "--enable-worker-recovery" in disabled_cmd


def test_worker_timeout_covers_unified_recovery_budget():
    timeout_seconds = module._effective_worker_timeout_seconds(
        timeout_seconds=180.0,
        time_budget_ms=75_000.0,
        enable_worker_recovery=True,
    )

    assert timeout_seconds == 190.0


def test_worker_timeout_keeps_single_attempt_budget_when_recovery_is_disabled():
    timeout_seconds = module._effective_worker_timeout_seconds(
        timeout_seconds=180.0,
        time_budget_ms=75_000.0,
        enable_worker_recovery=False,
    )

    assert timeout_seconds == 85.0


def test_solve_one_treats_final_capacity_overflow_as_warning(tmp_path: Path):
    scenario_path = tmp_path / "validation_capacity.json"
    scenario_path.write_text(
        json.dumps(
            {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": 367},
                    {"trackName": "存5南", "trackDistance": 20},
                    {"trackName": "机库", "trackDistance": 71.6},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": str(idx),
                        "vehicleModel": "棚车",
                        "vehicleNo": f"CAP{idx}",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存5南",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                    for idx in range(1, 3)
                ],
                "locoTrackName": "机库",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    solved_result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=1.0,
        is_complete=True,
        fallback_stage="capacity_warning_smoke",
        verification_report=SimpleNamespace(is_valid=True, errors=[]),
        debug_stats={},
    )

    with patch.object(module, "solve_with_simple_astar_result", return_value=solved_result):
        result = solve_one(
            master_dir=DATA_DIR,
            scenario_path=scenario_path,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
        )

    assert result["scenario"] == "validation_capacity.json"
    assert result["solved"] is True
    assert result["capacity_warnings"] == [
        {
            "track": "存5南",
            "required_length": 28.6,
            "capacity": 20.0,
            "effective_capacity": 20.0,
        }
    ]


def test_parse_args_defaults_to_baseline_settings():
    original_argv = module.sys.argv
    module.sys.argv = ["run_external_validation_parallel.py"]
    try:
        args = module.parse_args()
    finally:
        module.sys.argv = original_argv

    assert args.beam_width == 8
    assert args.solver == VALIDATION_DEFAULT_SOLVER
    assert args.beam_width == VALIDATION_DEFAULT_BEAM_WIDTH
    assert args.max_workers == VALIDATION_DEFAULT_MAX_WORKERS
    assert args.timeout_seconds == int(VALIDATION_DEFAULT_TIMEOUT_SECONDS)
    assert args.retry_no_solution_beam_width is None


def test_validation_runner_uses_shared_solver_time_budget_default(monkeypatch, tmp_path: Path):
    scenario = tmp_path / "validation_budget.json"
    scenario.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        captured["first"] = kwargs
        return [{"scenario": scenario.name, "solved": True}]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        captured["recover"] = kwargs
        return kwargs["initial_results"]

    monkeypatch.setattr(module, "run_parallel_scenarios", fake_run_parallel_scenarios)
    monkeypatch.setattr(module, "recover_no_solution_results", fake_recover_no_solution_results)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_external_validation_parallel.py",
            "--scenario",
            str(scenario),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    expected_budget = validation_time_budget_ms(VALIDATION_DEFAULT_TIMEOUT_SECONDS)
    assert captured["first"]["time_budget_ms"] == expected_budget
    assert captured["recover"]["time_budget_ms"] == expected_budget
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["solver_time_budget_ms"] == expected_budget
    assert summary["generated_cases"] == 1
    assert summary["scenario_count"] == 1
    assert summary["solved_cases"] == 1
    assert summary["unsolved_cases"] == 0


def test_validation_runner_keeps_recovery_in_parent_scheduler(
    monkeypatch,
    tmp_path: Path,
):
    scenario = tmp_path / "validation_budget.json"
    scenario.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        captured["first"] = kwargs
        return [{"scenario": scenario.name, "solved": True}]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        captured["recover"] = kwargs
        return kwargs["initial_results"]

    monkeypatch.setattr(module, "run_parallel_scenarios", fake_run_parallel_scenarios)
    monkeypatch.setattr(module, "recover_no_solution_results", fake_recover_no_solution_results)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_external_validation_parallel.py",
            "--scenario",
            str(scenario),
            "--output-dir",
            str(output_dir),
            "--timeout-seconds",
            "180",
        ],
    )

    module.main()

    assert captured["first"]["enable_worker_recovery"] is False
    assert captured["recover"]["deadline_at"] is None
    assert captured["recover"]["enable_worker_recovery"] is False
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["enable_worker_recovery"] is False


def test_validation_runner_enables_worker_recovery_only_when_requested(
    monkeypatch,
    tmp_path: Path,
):
    scenario = tmp_path / "validation_budget.json"
    scenario.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        captured["first"] = kwargs
        return [{"scenario": scenario.name, "solved": True}]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        return kwargs["initial_results"]

    monkeypatch.setattr(module, "run_parallel_scenarios", fake_run_parallel_scenarios)
    monkeypatch.setattr(module, "recover_no_solution_results", fake_recover_no_solution_results)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_external_validation_parallel.py",
            "--scenario",
            str(scenario),
            "--output-dir",
            str(output_dir),
            "--enable-worker-recovery",
        ],
    )

    module.main()

    assert captured["first"]["enable_worker_recovery"] is True


def test_recover_no_solution_results_keeps_worker_recovery_disabled_on_retries(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_recovery.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_recovery.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "partial_hook_count": 47,
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 24,
                    "work_position_unfinished_count": 0,
                    "front_blocker_count": 2,
                    "staging_debt_count": 0,
                    "capacity_overflow_track_count": 0,
                    "loco_carry_count": 8,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 1},
            },
        },
    ]
    calls: list[bool] = []

    def fake_run_parallel_scenarios(
        *,
        enable_worker_recovery,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(enable_worker_recovery)
        return [
            {
                "scenario": "validation_recovery.json",
                "solved": True,
                "hook_count": 139,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 40_000.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 7}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=4,
            retry_no_solution_beam_width=None,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [False]
    assert results[0]["solved"] is True


def test_recover_no_solution_results_passes_worker_recovery_when_requested(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_recovery.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_recovery.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "debug_stats": {},
        },
    ]
    calls: list[bool] = []

    def fake_run_parallel_scenarios(
        *,
        enable_worker_recovery,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(enable_worker_recovery)
        return [
            {
                "scenario": "validation_recovery.json",
                "solved": True,
                "hook_count": 139,
                "is_valid": True,
                "verifier_errors": [],
                "elapsed_ms": 40_000.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 7}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=4,
            retry_no_solution_beam_width=None,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
            enable_worker_recovery=True,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [True]


def test_validation_runner_summary_counts_recovered_results(
    monkeypatch,
    tmp_path: Path,
):
    scenario_a = tmp_path / "validation_a.json"
    scenario_b = tmp_path / "validation_b.json"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"

    def fake_run_parallel_scenarios(**_kwargs):  # noqa: ANN003
        return [
            {"scenario": scenario_a.name, "solved": False},
            {"scenario": scenario_b.name, "solved": False},
        ]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        initial_results = kwargs["initial_results"]
        return [
            {**initial_results[0], "solved": True},
            initial_results[1],
        ]

    monkeypatch.setattr(module, "run_parallel_scenarios", fake_run_parallel_scenarios)
    monkeypatch.setattr(module, "recover_no_solution_results", fake_recover_no_solution_results)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_external_validation_parallel.py",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generated_cases"] == 2
    assert summary["scenario_count"] == 2
    assert summary["solved_cases"] == 1
    assert summary["unsolved_cases"] == 1


def test_validation_runner_resolves_relative_scenario_against_input_dir(
    monkeypatch,
    tmp_path: Path,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    scenario = input_dir / "case_a.json"
    scenario.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    captured = {}

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        captured["scenario_paths"] = kwargs["scenario_paths"]
        return [{"scenario": scenario.name, "solved": True}]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        return kwargs["initial_results"]

    monkeypatch.setattr(module, "run_parallel_scenarios", fake_run_parallel_scenarios)
    monkeypatch.setattr(module, "recover_no_solution_results", fake_recover_no_solution_results)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_external_validation_parallel.py",
            "--input-dir",
            str(input_dir),
            "--scenario",
            scenario.name,
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    assert captured["scenario_paths"] == [scenario]


def test_run_parallel_scenarios_uses_subprocesses_and_collects_results(tmp_path: Path):
    scenario_a = tmp_path / "validation_a.json"
    scenario_b = tmp_path / "validation_b.json"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, scenario_name: str):
            self.pid = 4321
            self.returncode = 0
            self._stdout = json.dumps({"scenario": scenario_name, "solved": True})
            self._stderr = ""

        def communicate(self, timeout):  # noqa: ANN001, ARG002
            return self._stdout, self._stderr

        def wait(self, timeout=None):  # noqa: ANN001, ARG002
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, start_new_session):  # noqa: ANN001
        assert stdout == module.subprocess.PIPE
        assert stderr == module.subprocess.PIPE
        assert text is True
        assert start_new_session is True
        calls.append(cmd)
        scenario_name = Path(cmd[cmd.index("--scenario") + 1]).name
        return FakeProcess(scenario_name)

    original_popen = module.subprocess.Popen
    module.subprocess.Popen = fake_popen
    try:
        results = module.run_parallel_scenarios(
            master_dir=Path("data/master"),
            scenario_paths=[scenario_a, scenario_b],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=5,
            max_workers=2,
        )
    finally:
        module.subprocess.Popen = original_popen

    assert [item["scenario"] for item in results] == ["validation_a.json", "validation_b.json"]
    assert len(calls) == 2
    assert all("--scenario" in call for call in calls)


def test_run_parallel_scenarios_marks_timeout_and_cleans_worker_process_group(tmp_path: Path):
    scenario = tmp_path / "validation_timeout.json"
    scenario.write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    class FakeProcess:
        pid = 9876
        returncode = None

        def communicate(self, timeout):  # noqa: ANN001
            calls.append(("communicate", timeout))
            raise module.subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)

        def wait(self, timeout=None):  # noqa: ANN001
            calls.append(("wait", timeout))
            self.returncode = -9
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, start_new_session):  # noqa: ANN001, ARG001
        calls.append(("popen", start_new_session))
        return FakeProcess()

    def fake_killpg(pgid, sig):  # noqa: ANN001
        calls.append(("killpg", pgid, sig))

    original_popen = module.subprocess.Popen
    original_killpg = module.os.killpg
    module.subprocess.Popen = fake_popen
    module.os.killpg = fake_killpg
    try:
        results = module.run_parallel_scenarios(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=3,
            max_workers=1,
        )
    finally:
        module.subprocess.Popen = original_popen
        module.os.killpg = original_killpg

    assert results[0]["scenario"] == "validation_timeout.json"
    assert results[0]["solved"] is False
    assert results[0]["error"] == "timeout"
    assert ("popen", True) in calls
    assert ("communicate", 13.0) in calls
    assert ("killpg", 9876, module.signal.SIGKILL) in calls
    assert ("wait", None) in calls


def test_run_parallel_scenarios_gives_single_attempt_worker_timeout_solver_grace(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_timeout_budget.json"
    scenario.write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    class FakeProcess:
        pid = 9877
        returncode = None

        def communicate(self, timeout):  # noqa: ANN001
            calls.append(("communicate", timeout))
            raise module.subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)

        def wait(self, timeout=None):  # noqa: ANN001
            calls.append(("wait", timeout))
            self.returncode = -9
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, start_new_session):  # noqa: ANN001, ARG001
        return FakeProcess()

    def fake_killpg(pgid, sig):  # noqa: ANN001
        calls.append(("killpg", pgid, sig))

    original_popen = module.subprocess.Popen
    original_killpg = module.os.killpg
    module.subprocess.Popen = fake_popen
    module.os.killpg = fake_killpg
    try:
        module.run_parallel_scenarios(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=10,
            max_workers=1,
            time_budget_ms=30_000,
        )
    finally:
        module.subprocess.Popen = original_popen
        module.os.killpg = original_killpg

    assert ("communicate", 40.0) in calls


def test_run_parallel_scenarios_worker_timeout_uses_reserved_solver_budget(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_timeout_budget.json"
    scenario.write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    class FakeProcess:
        pid = 9878
        returncode = None

        def communicate(self, timeout):  # noqa: ANN001
            calls.append(("communicate", timeout))
            raise module.subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)

        def wait(self, timeout=None):  # noqa: ANN001
            self.returncode = -9
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, start_new_session):  # noqa: ANN001, ARG001
        return FakeProcess()

    def fake_killpg(pgid, sig):  # noqa: ANN001, ARG002
        calls.append(("killpg", pgid))

    original_popen = module.subprocess.Popen
    original_killpg = module.os.killpg
    module.subprocess.Popen = fake_popen
    module.os.killpg = fake_killpg
    try:
        module.run_parallel_scenarios(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=1,
            time_budget_ms=75_000,
            enable_worker_recovery=True,
        )
    finally:
        module.subprocess.Popen = original_popen
        module.os.killpg = original_killpg

    assert ("communicate", 190.0) in calls


def test_recover_no_solution_results_only_replaces_successful_retries(tmp_path: Path):
    scenario_a = tmp_path / "validation_a.json"
    scenario_b = tmp_path / "validation_b.json"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "no solution within budget", "debug_stats": {}},
        {
            "scenario": "validation_b.json",
            "solved": False,
            "error": "timeout",
            "error_category": "timeout",
            "debug_stats": {},
        },
    ]
    calls: list[tuple[list[str], int | None, int, float | None]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(([path.name for path in scenario_paths], beam_width, max_workers, time_budget_ms))
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 9,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario_a, scenario_b],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=12,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [
        (["validation_a.json", "validation_b.json"], 8, 2, 90_000.0),
        (["validation_b.json"], 12, 1, 90_000.0),
    ]
    assert results[0]["scenario"] == "validation_a.json"
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 8
    assert results[0]["recovery_time_budget_ms"] == 90_000.0
    assert results[1] == initial_results[1]


def test_recover_no_solution_results_uses_full_recovery_budget_for_first_retry(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 20_000.0,
            "worker_elapsed_ms": 22_000.0,
            "partial_hook_count": 1,
            "debug_stats": {
                "partial_structural_metrics": {"unfinished_count": 9},
                "partial_route_blockage_plan": {"total_blockage_pressure": 3},
            },
        },
    ]
    captured: list[float | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append(time_budget_ms)
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "partial_hook_count": 10,
                "debug_stats": {
                    "partial_structural_metrics": {"unfinished_count": 2},
                    "partial_route_blockage_plan": {"total_blockage_pressure": 0},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(60),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert captured == [100_000.0]
    assert results[0]["recovery_time_budget_ms"] == 100_000.0


def test_recover_no_solution_results_caps_total_worker_time_including_retry(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "timeout",
            "error_category": "timeout",
            "elapsed_ms": 119_000.0,
            "worker_elapsed_ms": 120_000.0,
            "debug_stats": {},
        },
    ]
    captured: list[tuple[float | None, float]] = []

    def fake_run_parallel_scenarios(
        *,
        timeout_seconds,  # noqa: ANN001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append((time_budget_ms, timeout_seconds))
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "elapsed_ms": 1.0,
                "worker_elapsed_ms": 1.0,
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=120,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=110_000.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    # 180s total wall cap - 120s already spent - 10s worker grace.
    assert captured == [(50_000.0, 50.0)]


def test_recover_no_solution_results_caps_retry_budget_to_runner_deadline(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "elapsed_ms": 75_000.0,
            "worker_elapsed_ms": 76_000.0,
            "debug_stats": {},
        },
    ]
    captured: list[tuple[float | None, float]] = []

    def fake_run_parallel_scenarios(
        *,
        timeout_seconds,  # noqa: ANN001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append((time_budget_ms, timeout_seconds))
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "elapsed_ms": 1.0,
                "worker_elapsed_ms": 1.0,
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    original_perf_counter = module.perf_counter
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    module.perf_counter = lambda: 45.0
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=75_000.0,
            deadline_at=100.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios
        module.perf_counter = original_perf_counter

    assert captured == [(45_000.0, 45.0)]


def test_recover_no_solution_results_retries_external_timeouts(tmp_path: Path):
    scenario = tmp_path / "validation_timeout.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_timeout.json",
            "solved": False,
            "error": "timeout",
            "error_category": "timeout",
            "worker_elapsed_ms": 70_000.0,
            "debug_stats": {},
        },
    ]
    calls: list[tuple[str, float | None]] = []

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        calls.append(("retry", kwargs["time_budget_ms"]))
        return [
            {
                "scenario": "validation_timeout.json",
                "solved": True,
                "hook_count": 9,
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(60),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [("retry", 100_000.0)]
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 8
    assert results[0]["recovery_time_budget_ms"] == 100_000.0


def test_recover_no_solution_results_gives_recovery_independent_budget_after_spent_primary(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 54_500.0,
            "worker_elapsed_ms": 58_000.0,
            "debug_stats": {},
        },
    ]
    calls: list[tuple[float | None, float]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001
        max_workers,  # noqa: ANN001, ARG001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((time_budget_ms, timeout_seconds))
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(60),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [(100_000.0, 100.0)]


def test_recover_no_solution_results_allows_small_retry_after_primary_budget_is_spent(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 50_250.0,
            "worker_elapsed_ms": 50_400.0,
            "debug_stats": {},
        },
    ]
    captured: list[tuple[float | None, float]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001
        max_workers,  # noqa: ANN001, ARG001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append((time_budget_ms, timeout_seconds))
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=50_000.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert captured == [(100_000.0, 100.0)]


def test_recover_no_solution_results_skips_retry_when_recovery_budget_is_spent(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 1_000.0,
            "worker_elapsed_ms": 1_000.0,
            "total_solver_elapsed_ms": 95_000.0,
            "total_worker_elapsed_ms": 98_000.0,
            "debug_stats": {},
        },
    ]
    calls: list[tuple[float | None, float]] = []

    def fake_run_parallel_scenarios(
        *,
        timeout_seconds,  # noqa: ANN001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((time_budget_ms, timeout_seconds))
        return []

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(60),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == []


def test_recover_no_solution_results_keeps_best_failed_retry_partial(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "partial_hook_count": 20,
            "debug_stats": {
                "partial_structural_metrics": {"unfinished_count": 8},
                "partial_route_blockage_plan": {"total_blockage_pressure": 4},
            },
        },
    ]

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "partial_hook_count": 77,
                "debug_stats": {
                    "partial_structural_metrics": {"unfinished_count": 2},
                    "partial_route_blockage_plan": {"total_blockage_pressure": 1},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert results[0]["solved"] is False
    assert results[0]["partial_hook_count"] == 77
    assert results[0]["recovery_beam_width"] == 8
    assert results[0]["debug_stats"]["partial_structural_metrics"]["unfinished_count"] == 2
    assert results[0]["debug_stats"]["partial_route_blockage_plan"]["total_blockage_pressure"] == 1


def test_recover_no_solution_results_prefers_route_clean_business_partial(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "partial_hook_count": 76,
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 28,
                    "staging_debt_count": 0,
                    "work_position_unfinished_count": 2,
                    "front_blocker_count": 5,
                    "goal_track_blocker_count": 19,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 0},
            },
        },
    ]

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "partial_hook_count": 78,
                "debug_stats": {
                    "partial_structural_metrics": {
                        "unfinished_count": 30,
                        "staging_debt_count": 2,
                        "work_position_unfinished_count": 7,
                        "front_blocker_count": 6,
                        "goal_track_blocker_count": 12,
                    },
                    "partial_route_blockage_plan": {"total_blockage_pressure": 2},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert results[0]["partial_hook_count"] == 76
    assert results[0]["debug_stats"]["partial_route_blockage_plan"]["total_blockage_pressure"] == 0


def test_recover_no_solution_results_prefers_structural_progress_before_hook_count(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "partial_hook_count": 100,
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 38,
                    "staging_debt_count": 2,
                    "work_position_unfinished_count": 7,
                    "front_blocker_count": 8,
                    "goal_track_blocker_count": 12,
                    "loco_carry_count": 9,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 5},
            },
        },
    ]

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "partial_hook_count": 78,
                "debug_stats": {
                    "partial_structural_metrics": {
                        "unfinished_count": 30,
                        "staging_debt_count": 2,
                        "work_position_unfinished_count": 7,
                        "front_blocker_count": 6,
                        "goal_track_blocker_count": 12,
                        "loco_carry_count": 0,
                    },
                    "partial_route_blockage_plan": {"total_blockage_pressure": 2},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert results[0]["partial_hook_count"] == 78
    assert results[0]["debug_stats"]["partial_structural_metrics"]["unfinished_count"] == 30
    assert results[0]["debug_stats"]["partial_structural_metrics"]["loco_carry_count"] == 0


def test_recover_no_solution_results_retries_progressively_until_success(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "No solution found", "debug_stats": {}},
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        if beam_width in {8, 16}:
            return [
                {
                    "scenario": "validation_a.json",
                    "solved": False,
                    "error": "no solution within budget",
                    "debug_stats": {},
                }
            ]
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 9,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=24,
            initial_results=initial_results,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8, 16, 24]
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 24


def test_recover_no_solution_results_reaches_wide_recovery_beam_by_default(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 41,
                    "capacity_overflow_track_count": 0,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 5},
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        if beam_width in {8, 16, 24}:
            return [
                {
                    "scenario": "validation_a.json",
                    "solved": False,
                    "error": "no solution within budget",
                    "error_category": "no_solution",
                    "partial_hook_count": 65,
                    "debug_stats": {
                        "partial_structural_metrics": {
                            "unfinished_count": 41,
                            "capacity_overflow_track_count": 0,
                        },
                        "partial_route_blockage_plan": {"total_blockage_pressure": 5},
                    },
                }
            ]
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 92,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 20}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=4,
            retry_no_solution_beam_width=None,
            initial_results=initial_results,
            time_budget_ms=50_000.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8, 16, 24, 32]
    assert results[0]["solved"] is True
    assert results[0]["hook_count"] == 92
    assert results[0]["recovery_beam_width"] == 32


def test_recover_no_solution_results_prioritizes_widest_beam_under_three_minute_primary_budget(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "elapsed_ms": 119_000.0,
            "worker_elapsed_ms": 119_500.0,
            "debug_stats": {
                "partial_structural_metrics": {"unfinished_count": 25},
                "partial_route_blockage_plan": {"total_blockage_pressure": 7},
            },
        },
    ]
    calls: list[tuple[int | None, float | None]] = []

    def fake_run_parallel_scenarios(
        *,
        beam_width,  # noqa: ANN001
        time_budget_ms=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((beam_width, time_budget_ms))
        return [
            {
                "scenario": "validation_a.json",
                "solved": False,
                "error": "no solution within budget",
                "error_category": "no_solution",
                "partial_hook_count": 30,
                "debug_stats": {
                    "partial_structural_metrics": {"unfinished_count": 20},
                    "partial_route_blockage_plan": {"total_blockage_pressure": 4},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=2,
            retry_no_solution_beam_width=24,
            initial_results=initial_results,
            time_budget_ms=120_000.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls[0][0] == 24


def test_recover_no_solution_results_limits_wide_retry_parallelism(tmp_path: Path):
    scenarios = [
        tmp_path / f"validation_{idx}.json"
        for idx in range(3)
    ]
    for scenario in scenarios:
        scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": scenario.name, "solved": False, "error": "no solution within budget", "debug_stats": {}}
        for scenario in scenarios
    ]
    calls: list[tuple[int | None, int, list[str]]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((beam_width, max_workers, [path.name for path in scenario_paths]))
        return [
            {
                "scenario": path.name,
                "solved": False,
                "error": "no solution within budget",
                "debug_stats": {},
            }
            for path in scenario_paths
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=scenarios,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=8,
            retry_no_solution_beam_width=24,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [
        (8, 4, [scenario.name for scenario in scenarios]),
        (16, 4, [scenario.name for scenario in scenarios]),
        (24, 2, [scenario.name for scenario in scenarios]),
    ]


def test_recover_no_solution_results_caps_same_beam_recovery_parallelism(
    tmp_path: Path,
):
    scenarios = [tmp_path / f"validation_{idx}.json" for idx in range(5)]
    for scenario in scenarios:
        scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": scenario.name,
            "solved": False,
            "error": "no solution within budget",
            "debug_stats": {},
        }
        for scenario in scenarios
    ]
    calls: list[int] = []

    def fake_run_parallel_scenarios(
        *,
        max_workers,  # noqa: ANN001
        scenario_paths,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(max_workers)
        return [
            {
                "scenario": path.name,
                "solved": False,
                "error": "no solution within budget",
                "debug_stats": {},
            }
            for path in scenario_paths
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=scenarios,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=16,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [4]


def test_recover_no_solution_results_buckets_similar_remaining_retry_budgets(
    tmp_path: Path,
):
    scenarios = [tmp_path / f"validation_{idx}.json" for idx in range(2)]
    for scenario in scenarios:
        scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": scenarios[0].name,
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 50_100.0,
            "worker_elapsed_ms": 50_200.0,
            "debug_stats": {},
        },
        {
            "scenario": scenarios[1].name,
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 50_450.0,
            "worker_elapsed_ms": 50_550.0,
            "debug_stats": {},
        },
    ]
    calls: list[list[str]] = []

    def fake_run_parallel_scenarios(
        *,
        scenario_paths,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append([path.name for path in scenario_paths])
        return [
            {
                "scenario": path.name,
                "solved": False,
                "error": "no solution within budget",
                "debug_stats": {},
            }
            for path in scenario_paths
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=scenarios,
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=4,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=50_000.0,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [[scenario.name for scenario in scenarios]]


def test_recover_no_solution_results_continues_after_pathological_success(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "no solution within budget", "debug_stats": {}},
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 807 if beam_width == 8 else 102,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {
                    "plan_shape_metrics": {
                        "max_vehicle_touch_count": 336 if beam_width == 8 else 28,
                    }
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=30_000,
            improve_pathological_success=True,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8, 16]
    assert results[0]["solved"] is True
    assert results[0]["hook_count"] == 102
    assert results[0]["recovery_beam_width"] == 16


def test_recover_no_solution_results_retries_pathological_initial_success(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": True,
            "hook_count": 1009,
            "is_valid": True,
            "debug_stats": {
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 210,
                }
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 102,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 24}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=30_000,
            improve_pathological_success=True,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8]
    assert results[0]["hook_count"] == 102
    assert results[0]["recovery_beam_width"] == 8


def test_recover_no_solution_results_does_not_retry_pathological_success_by_default(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": True,
            "hook_count": 1009,
            "is_valid": True,
            "debug_stats": {
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 210,
                }
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        calls.append(kwargs["beam_width"])
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 102,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 24}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == []
    assert results[0] == initial_results[0]


def test_recover_no_solution_results_retries_staging_churn_success_when_requested(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": True,
            "hook_count": 191,
            "is_valid": True,
            "debug_stats": {
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 40,
                    "staging_to_staging_hook_count": 23,
                    "rehandled_vehicle_count": 35,
                }
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        calls.append(kwargs["beam_width"])
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 61,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {
                    "plan_shape_metrics": {
                        "max_vehicle_touch_count": 12,
                        "staging_to_staging_hook_count": 2,
                        "rehandled_vehicle_count": 20,
                    }
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=30_000,
            improve_pathological_success=True,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8]
    assert results[0]["hook_count"] == 61
    assert results[0]["recovery_beam_width"] == 8


def test_recover_no_solution_results_does_not_retry_high_hook_low_churn_success(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": True,
            "hook_count": 269,
            "is_valid": True,
            "debug_stats": {
                "plan_shape_metrics": {
                    "max_vehicle_touch_count": 56,
                }
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(**kwargs):  # noqa: ANN003
        calls.append(kwargs["beam_width"])
        return []

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == []
    assert results[0] == initial_results[0]


def test_recover_no_solution_results_stops_after_healthy_success(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "no solution within budget", "debug_stats": {}},
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 95,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 36}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=24,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8]
    assert results[0]["solved"] is True
    assert results[0]["hook_count"] == 95
    assert results[0]["recovery_beam_width"] == 8


def test_recover_no_solution_results_continues_after_churny_recovery_success(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_sequence_tail.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_sequence_tail.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "partial_hook_count": 254,
            "debug_stats": {
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
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        beam_width,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        if beam_width == 8:
            return [
                {
                    "scenario": "validation_sequence_tail.json",
                    "solved": True,
                    "hook_count": 333,
                    "is_valid": True,
                    "verifier_errors": [],
                    "elapsed_ms": 94_000.0,
                    "debug_stats": {
                        "plan_shape_metrics": {
                            "staging_hook_count": 76,
                            "staging_to_staging_hook_count": 37,
                            "rehandled_vehicle_count": 45,
                            "max_vehicle_touch_count": 62,
                        }
                    },
                }
            ]
        return [
            {
                "scenario": "validation_sequence_tail.json",
                "solved": True,
                "hook_count": 156,
                "is_valid": True,
                "verifier_errors": [],
                "elapsed_ms": 71_000.0,
                "debug_stats": {
                    "plan_shape_metrics": {
                        "staging_hook_count": 36,
                        "staging_to_staging_hook_count": 18,
                        "rehandled_vehicle_count": 35,
                        "max_vehicle_touch_count": 40,
                    }
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=4,
            retry_no_solution_beam_width=16,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [8, 16]
    assert results[0]["solved"] is True
    assert results[0]["hook_count"] == 156
    assert results[0]["recovery_beam_width"] == 16


def test_recover_no_solution_results_enables_recovery_partial_resume_profile(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "no solution within budget", "debug_stats": {}},
    ]
    captured: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001, ARG001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001, ARG001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        near_goal_partial_resume_max_final_heuristic=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append(near_goal_partial_resume_max_final_heuristic)
        return [
            {
                "scenario": "validation_a.json",
                "solved": True,
                "hook_count": 88,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 20}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=60,
            max_workers=2,
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=30_000,
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert captured == [RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC]


def test_recover_no_solution_results_tries_same_beam_first_for_route_clean_tail(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_tail.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_tail.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "partial_hook_count": 42,
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 8,
                    "work_position_unfinished_count": 6,
                    "front_blocker_count": 2,
                    "staging_debt_count": 0,
                    "capacity_overflow_track_count": 1,
                    "loco_carry_count": 0,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 0},
            },
        },
    ]
    calls: list[tuple[int | None, int | None]] = []

    def fake_run_parallel_scenarios(
        *,
        beam_width,  # noqa: ANN001
        near_goal_partial_resume_max_final_heuristic=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((beam_width, near_goal_partial_resume_max_final_heuristic))
        return [
            {
                "scenario": "validation_tail.json",
                "solved": True,
                "hook_count": 60,
                "is_valid": True,
                "verifier_errors": [],
                "expanded_nodes": 10,
                "generated_nodes": 20,
                "closed_nodes": 10,
                "elapsed_ms": 12.0,
                "debug_stats": {"plan_shape_metrics": {"max_vehicle_touch_count": 8}},
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        results = module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=4,
            retry_no_solution_beam_width=None,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [
        (8, RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC),
    ]
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 8


def test_recover_no_solution_results_keeps_wide_first_for_route_blocked_partial(
    tmp_path: Path,
):
    scenario = tmp_path / "validation_blocked.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_blocked.json",
            "solved": False,
            "error": "no solution within budget",
            "error_category": "no_solution",
            "partial_hook_count": 42,
            "debug_stats": {
                "partial_structural_metrics": {
                    "unfinished_count": 25,
                    "work_position_unfinished_count": 6,
                    "front_blocker_count": 8,
                    "staging_debt_count": 0,
                    "capacity_overflow_track_count": 1,
                    "loco_carry_count": 0,
                },
                "partial_route_blockage_plan": {"total_blockage_pressure": 7},
            },
        },
    ]
    calls: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        beam_width,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(beam_width)
        return [
            {
                "scenario": "validation_blocked.json",
                "solved": False,
                "error": "no solution within budget",
                "error_category": "no_solution",
                "partial_hook_count": 43,
                "debug_stats": {
                    "partial_structural_metrics": {"unfinished_count": 24},
                    "partial_route_blockage_plan": {"total_blockage_pressure": 6},
                },
            }
        ]

    original_run_parallel_scenarios = module.run_parallel_scenarios
    module.run_parallel_scenarios = fake_run_parallel_scenarios
    try:
        module.recover_no_solution_results(
            master_dir=Path("data/master"),
            scenario_paths=[scenario],
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            timeout_seconds=180,
            max_workers=4,
            retry_no_solution_beam_width=None,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(180),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls[0] == 32


def test_main_honors_scenario_and_recovery_settings(tmp_path: Path):
    scenario = tmp_path / "validation_single.json"
    scenario.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, list[str], int | None]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        near_goal_partial_resume_max_final_heuristic=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append((
            "run",
            [path.name for path in scenario_paths],
            beam_width,
            near_goal_partial_resume_max_final_heuristic,
        ))
        return [{"scenario": "validation_single.json", "solved": True}]

    def fake_recover_no_solution_results(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
        retry_no_solution_beam_width,  # noqa: ANN001
        initial_results,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        calls.append(("recover", [path.name for path in scenario_paths], retry_no_solution_beam_width))
        return initial_results

    original_parse_args = module.parse_args
    original_run_parallel_scenarios = module.run_parallel_scenarios
    original_recover_no_solution_results = module.recover_no_solution_results
    try:
        module.parse_args = lambda: module.argparse.Namespace(
            input_dir=Path("ignored"),
            output_dir=tmp_path / "out",
            master_dir=Path("data/master"),
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            max_workers=2,
            timeout_seconds=60,
            scenario=scenario,
            retry_no_solution_beam_width=12,
            improve_pathological_success=False,
            near_goal_partial_resume_max_final_heuristic=10,
            worker=False,
        )
        module.run_parallel_scenarios = fake_run_parallel_scenarios
        module.recover_no_solution_results = fake_recover_no_solution_results
        module.main()
    finally:
        module.parse_args = original_parse_args
        module.run_parallel_scenarios = original_run_parallel_scenarios
        module.recover_no_solution_results = original_recover_no_solution_results

    assert calls == [
        ("run", ["validation_single.json"], 8, 10),
        ("recover", ["validation_single.json"], 12),
    ]


def test_main_leaves_initial_run_on_solver_default_near_goal_profile(tmp_path: Path):
    scenario = tmp_path / "validation_single.json"
    scenario.write_text("{}", encoding="utf-8")
    captured: list[int | None] = []

    def fake_run_parallel_scenarios(
        *,
        near_goal_partial_resume_max_final_heuristic=None,  # noqa: ANN001
        **_kwargs,  # noqa: ANN003
    ):
        captured.append(near_goal_partial_resume_max_final_heuristic)
        return [{"scenario": "validation_single.json", "solved": True}]

    def fake_recover_no_solution_results(**kwargs):  # noqa: ANN003
        return kwargs["initial_results"]

    original_parse_args = module.parse_args
    original_run_parallel_scenarios = module.run_parallel_scenarios
    original_recover_no_solution_results = module.recover_no_solution_results
    try:
        module.parse_args = lambda: module.argparse.Namespace(
            input_dir=Path("ignored"),
            output_dir=tmp_path / "out",
            master_dir=Path("data/master"),
            solver="beam",
            beam_width=8,
            heuristic_weight=1.0,
            max_workers=2,
            timeout_seconds=60,
            scenario=scenario,
            retry_no_solution_beam_width=0,
            improve_pathological_success=False,
            near_goal_partial_resume_max_final_heuristic=None,
            worker=False,
        )
        module.run_parallel_scenarios = fake_run_parallel_scenarios
        module.recover_no_solution_results = fake_recover_no_solution_results
        module.main()
    finally:
        module.parse_args = original_parse_args
        module.run_parallel_scenarios = original_run_parallel_scenarios
        module.recover_no_solution_results = original_recover_no_solution_results

    assert captured == [None]


def test_truth_20260227w_solves_under_validation_profile():
    result = solve_one(
        master_dir=DATA_DIR,
        scenario_path=ROOT_DIR / "data" / "validation_inputs" / "truth" / "validation_20260227W.json",
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
        time_budget_ms=10_000.0,
    )

    assert result["solved"] is True, result
    assert result["is_valid"] is True
