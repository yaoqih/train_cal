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
    validation_retry_time_budget_ms,
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


def test_solve_one_keeps_worker_to_single_attempt_for_parent_recovery(tmp_path: Path):
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
        )

    assert result["solved"] is False
    assert result["error_category"] == "no_solution"
    assert result["partial_hook_count"] == 1
    assert len(captured) == 1
    assert captured[0]["solver_mode"] == "beam"
    assert captured[0]["beam_width"] == 8
    assert captured[0]["time_budget_ms"] == 50_000.0


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
            )
        )

    assert captured == [True]
    assert json.loads(capsys.readouterr().out)["solved"] is True


def test_solve_one_classifies_final_capacity_infeasible(tmp_path: Path):
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

    result = solve_one(
        master_dir=DATA_DIR,
        scenario_path=scenario_path,
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
    )

    assert result["scenario"] == "validation_capacity.json"
    assert result["solved"] is False
    assert result["error_category"] == "capacity_infeasible"
    assert result["retryable"] is False
    assert "final arrangement exceeds track capacity" in result["error"]


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


def test_run_parallel_scenarios_gives_worker_timeout_solver_grace(tmp_path: Path):
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
        (["validation_a.json"], 8, 2, 30_000.0),
    ]
    assert results[0]["scenario"] == "validation_a.json"
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 8
    assert results[0]["recovery_time_budget_ms"] == 30_000.0
    assert results[1] == initial_results[1]


def test_recover_no_solution_results_caps_recovery_attempt_by_remaining_total_budget(tmp_path: Path):
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

    assert captured == [50_000.0]
    assert results[0]["recovery_time_budget_ms"] == 50_000.0


def test_recover_no_solution_results_does_not_retry_external_timeouts(tmp_path: Path):
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
    calls: list[str] = []

    def fake_run_parallel_scenarios(**_kwargs):  # noqa: ANN003
        calls.append("retry")
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
            retry_no_solution_beam_width=8,
            initial_results=initial_results,
            time_budget_ms=validation_time_budget_ms(60),
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == []
    assert results == initial_results


def test_recover_no_solution_results_subtracts_spent_time_from_total_retry_budget(tmp_path: Path):
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

    assert calls == [(42_000.0, 42.0)]


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

    assert captured == [(49_000.0, 49.0)]


def test_recover_no_solution_results_skips_retry_when_total_retry_budget_is_spent(tmp_path: Path):
    scenario = tmp_path / "validation_a.json"
    scenario.write_text("{}", encoding="utf-8")
    initial_results = [
        {
            "scenario": "validation_a.json",
            "solved": False,
            "error": "no solution within budget",
            "elapsed_ms": 95_000.0,
            "worker_elapsed_ms": 98_000.0,
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
        (8, 8, [scenario.name for scenario in scenarios]),
        (16, 4, [scenario.name for scenario in scenarios]),
        (24, 2, [scenario.name for scenario in scenarios]),
    ]


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


def test_main_leaves_initial_run_on_solver_default_near_goal_threshold(tmp_path: Path):
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
