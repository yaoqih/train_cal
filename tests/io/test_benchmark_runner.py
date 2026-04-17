from pathlib import Path
import json

from fzed_shunting.benchmark.runner import run_benchmark, run_solver_comparison_suite
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.sim.generator import generate_typical_suite


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_run_benchmark_on_generated_micro_cases(tmp_path: Path):
    report = run_benchmark(
        output_dir=tmp_path,
        scenario_count=3,
        vehicle_count=4,
        seed_start=100,
    )

    assert report["scenario_count"] == 3
    assert report["solved_count"] == 3
    assert len(report["results"]) == 3


def test_run_benchmark_reports_rates_and_error_summary(tmp_path: Path):
    report = run_benchmark(
        output_dir=tmp_path,
        scenario_count=4,
        vehicle_count=4,
        seed_start=200,
        direct_only=False,
    )

    assert report["scenario_count"] == 4
    assert report["unsolved_count"] == report["scenario_count"] - report["solved_count"]
    assert report["solved_rate"] == report["solved_count"] / report["scenario_count"]
    assert "average_hook_count_on_solved" in report
    assert "average_path_length_m_on_solved" in report
    assert "average_branch_count_on_solved" in report
    assert "average_expanded_nodes_on_solved" in report
    assert "average_generated_nodes_on_solved" in report
    assert isinstance(report["error_summary"], dict)
    assert report["direct_only"] is False
    assert all("elapsed_ms" in item for item in report["results"])
    assert all("expanded_nodes" in item for item in report["results"])
    assert all("generated_nodes" in item for item in report["results"])
    assert all("path_length_m" in item for item in report["results"])
    assert all("branch_count" in item for item in report["results"])


def test_run_benchmark_keeps_profile_in_report(tmp_path: Path):
    report = run_benchmark(
        output_dir=tmp_path,
        scenario_count=2,
        vehicle_count=6,
        seed_start=50,
        direct_only=False,
        profile="adversarial",
    )

    assert report["profile"] == "adversarial"
    assert all(item["profile"] == "adversarial" for item in report["results"])


def test_run_benchmark_keeps_lns_solver_in_report(tmp_path: Path):
    report = run_benchmark(
        output_dir=tmp_path,
        scenario_count=2,
        vehicle_count=4,
        seed_start=20,
        solver="lns",
        beam_width=8,
    )

    assert report["solver"] == "lns"
    assert all(item["solver"] == "lns" for item in report["results"])


def test_run_benchmark_exports_csv_results(tmp_path: Path):
    report = run_benchmark(
        output_dir=tmp_path,
        scenario_count=2,
        vehicle_count=4,
        seed_start=10,
    )

    csv_path = tmp_path / "benchmark_results.csv"

    assert report["scenario_count"] == 2
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("seed,profile,scenario_path")


def test_run_solver_comparison_suite_exports_per_solver_summary(tmp_path: Path):
    master = load_master_data(DATA_DIR)
    suite = generate_typical_suite(master)
    suite_path = tmp_path / "typical_suite.json"
    suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")

    report = run_solver_comparison_suite(
        suite_path=suite_path,
        output_dir=tmp_path / "compare",
        solver_configs=[
            {"solver": "exact"},
            {"solver": "weighted", "heuristic_weight": 2.0},
            {"solver": "beam", "beam_width": 8},
            {"solver": "lns", "beam_width": 8},
        ],
    )

    assert report["scenario_count"] == 13
    assert [item["solver"] for item in report["solvers"]] == ["exact", "weighted", "beam", "lns"]
    assert all(item["validCount"] == 13 for item in report["solvers"])
    assert (tmp_path / "compare" / "comparison_report.json").exists()
