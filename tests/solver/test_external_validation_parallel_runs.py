from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from shutil import copy2
import json

from fzed_shunting.tools.convert_external_validation_inputs import convert_external_validation_inputs


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
    assert ("communicate", 3) in calls
    assert ("killpg", 9876, module.signal.SIGKILL) in calls
    assert ("wait", None) in calls


def test_recover_no_solution_results_only_replaces_successful_retries(tmp_path: Path):
    scenario_a = tmp_path / "validation_a.json"
    scenario_b = tmp_path / "validation_b.json"
    scenario_a.write_text("{}", encoding="utf-8")
    scenario_b.write_text("{}", encoding="utf-8")
    initial_results = [
        {"scenario": "validation_a.json", "solved": False, "error": "No solution found", "debug_stats": {}},
        {"scenario": "validation_b.json", "solved": False, "error": "timeout", "debug_stats": {}},
    ]
    calls: list[tuple[list[str], int | None]] = []

    def fake_run_parallel_scenarios(
        *,
        master_dir,  # noqa: ANN001, ARG001
        scenario_paths,  # noqa: ANN001
        solver,  # noqa: ANN001, ARG001
        beam_width,  # noqa: ANN001
        heuristic_weight,  # noqa: ANN001, ARG001
        timeout_seconds,  # noqa: ANN001, ARG001
        max_workers,  # noqa: ANN001, ARG001
    ):
        calls.append(([path.name for path in scenario_paths], beam_width))
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
        )
    finally:
        module.run_parallel_scenarios = original_run_parallel_scenarios

    assert calls == [(["validation_a.json"], 12)]
    assert results[0]["scenario"] == "validation_a.json"
    assert results[0]["solved"] is True
    assert results[0]["recovery_beam_width"] == 12
    assert results[1] == initial_results[1]


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
    ):
        calls.append(("run", [path.name for path in scenario_paths], beam_width))
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
        ("run", ["validation_single.json"], 8),
        ("recover", ["validation_single.json"], 12),
    ]
