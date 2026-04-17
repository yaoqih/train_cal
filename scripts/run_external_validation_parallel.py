from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state


DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[1] / "data" / "master"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_inputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_parallel_runs"
DEFAULT_RETRY_NO_SOLUTION_BEAM_WIDTH = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--solver", default="beam")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--retry-no-solution-beam-width", type=int, default=DEFAULT_RETRY_NO_SOLUTION_BEAM_WIDTH)
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def solve_one(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    debug_stats: dict[str, Any] = {}
    try:
        result = solve_with_simple_astar_result(
            normalized,
            initial,
            master=master,
            solver_mode=solver,
            beam_width=beam_width,
            heuristic_weight=heuristic_weight,
            debug_stats=debug_stats,
        )
        hook_plan = [
            {
                "hookNo": idx,
                "actionType": move.action_type,
                "sourceTrack": move.source_track,
                "targetTrack": move.target_track,
                "vehicleNos": move.vehicle_nos,
                "pathTracks": move.path_tracks,
            }
            for idx, move in enumerate(result.plan, start=1)
        ]
        verify_report = verify_plan(master, normalized, hook_plan)
        return {
            "scenario": scenario_path.name,
            "solved": True,
            "hook_count": len(result.plan),
            "is_valid": verify_report.is_valid,
            "verifier_errors": verify_report.errors,
            "expanded_nodes": result.expanded_nodes,
            "generated_nodes": result.generated_nodes,
            "closed_nodes": result.closed_nodes,
            "elapsed_ms": result.elapsed_ms,
            "debug_stats": debug_stats,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": str(exc),
            "debug_stats": debug_stats,
        }


def _build_worker_command(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--master-dir",
        str(master_dir),
        "--scenario",
        str(scenario_path),
        "--solver",
        solver,
        "--heuristic-weight",
        str(heuristic_weight),
    ]
    if beam_width is not None:
        cmd.extend(["--beam-width", str(beam_width)])
    return cmd


def _run_scenario_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = _build_worker_command(
        master_dir=master_dir,
        scenario_path=scenario_path,
        solver=solver,
        beam_width=beam_width,
        heuristic_weight=heuristic_weight,
    )
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": "timeout",
            "debug_stats": {},
        }

    if completed.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": completed.stderr.strip() or f"worker exited with code {completed.returncode}",
            "debug_stats": {},
        }
    return json.loads(completed.stdout)


def run_parallel_scenarios(
    *,
    master_dir: Path,
    scenario_paths: list[Path],
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_scenario_subprocess,
                master_dir=master_dir,
                scenario_path=scenario_path,
                solver=solver,
                beam_width=beam_width,
                heuristic_weight=heuristic_weight,
                timeout_seconds=timeout_seconds,
            )
            for scenario_path in scenario_paths
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["scenario"])
    return results


def recover_no_solution_results(
    *,
    master_dir: Path,
    scenario_paths: list[Path],
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: int,
    max_workers: int,
    retry_no_solution_beam_width: int | None,
    initial_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if solver != "beam" or beam_width is None or retry_no_solution_beam_width is None:
        return initial_results
    if retry_no_solution_beam_width <= beam_width:
        return initial_results

    scenario_path_by_name = {scenario_path.name: scenario_path for scenario_path in scenario_paths}
    retry_scenarios = [
        scenario_path_by_name[result["scenario"]]
        for result in initial_results
        if (
            not result.get("solved")
            and "No solution found" in result.get("error", "")
            and result["scenario"] in scenario_path_by_name
        )
    ]
    if not retry_scenarios:
        return initial_results

    retry_results = run_parallel_scenarios(
        master_dir=master_dir,
        scenario_paths=retry_scenarios,
        solver=solver,
        beam_width=retry_no_solution_beam_width,
        heuristic_weight=heuristic_weight,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
    )
    retry_result_by_scenario = {
        result["scenario"]: result
        for result in retry_results
        if result.get("solved")
    }
    merged_results: list[dict[str, Any]] = []
    for result in initial_results:
        recovered = retry_result_by_scenario.get(result["scenario"])
        if recovered is None:
            merged_results.append(result)
            continue
        recovered = dict(recovered)
        recovered["recovery_beam_width"] = retry_no_solution_beam_width
        merged_results.append(recovered)
    return merged_results


def _run_worker(args: argparse.Namespace) -> None:
    if args.scenario is None:
        raise SystemExit("--worker requires --scenario")
    result = solve_one(
        master_dir=args.master_dir,
        scenario_path=args.scenario,
        solver=args.solver,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
    )
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.worker:
        _run_worker(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.scenario is not None:
        scenario_paths = [args.scenario]
    else:
        scenario_paths = sorted(args.input_dir.glob("validation_*.json"))
    results = run_parallel_scenarios(
        master_dir=args.master_dir,
        scenario_paths=scenario_paths,
        solver=args.solver,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        timeout_seconds=args.timeout_seconds,
        max_workers=args.max_workers,
    )
    results = recover_no_solution_results(
        master_dir=args.master_dir,
        scenario_paths=scenario_paths,
        solver=args.solver,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        timeout_seconds=args.timeout_seconds,
        max_workers=args.max_workers,
        retry_no_solution_beam_width=args.retry_no_solution_beam_width,
        initial_results=results,
    )
    summary = {
        "solver": args.solver,
        "beam_width": args.beam_width,
        "heuristic_weight": args.heuristic_weight,
        "timeout_seconds": args.timeout_seconds,
        "scenario_count": len(results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
