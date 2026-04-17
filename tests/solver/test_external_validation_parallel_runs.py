from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_external_validation_parallel.py"
spec = spec_from_file_location("run_external_validation_parallel", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)
solve_one = module.solve_one


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
INPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "external_validation_inputs"


def test_solve_one_external_validation_returns_debug_stats():
    result = solve_one(
        master_dir=DATA_DIR,
        scenario_path=INPUT_DIR / "validation_2025-09-04_am_to_pm.json",
        solver="beam",
        beam_width=8,
        heuristic_weight=1.0,
    )

    assert result["scenario"] == "validation_2025-09-04_am_to_pm.json"
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

    class FakeCompleted:
        def __init__(self, scenario_name: str):
            self.stdout = json.dumps({"scenario": scenario_name, "solved": True})
            self.stderr = ""
            self.returncode = 0

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001
        calls.append(cmd)
        scenario_name = Path(cmd[cmd.index("--scenario") + 1]).name
        return FakeCompleted(scenario_name)

    original_run = module.subprocess.run
    module.subprocess.run = fake_run
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
        module.subprocess.run = original_run

    assert [item["scenario"] for item in results] == ["validation_a.json", "validation_b.json"]
    assert len(calls) == 2
    assert all("--scenario" in call for call in calls)


def test_run_parallel_scenarios_marks_timeout(tmp_path: Path):
    scenario = tmp_path / "validation_timeout.json"
    scenario.write_text("{}", encoding="utf-8")

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001, ARG001
        raise module.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    original_run = module.subprocess.run
    module.subprocess.run = fake_run
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
        module.subprocess.run = original_run

    assert results[0]["scenario"] == "validation_timeout.json"
    assert results[0]["solved"] is False
    assert results[0]["error"] == "timeout"
