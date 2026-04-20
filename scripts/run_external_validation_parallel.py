from __future__ import annotations

import argparse
import json
import os
import signal
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
DEFAULT_RETRY_NO_SOLUTION_BEAM_WIDTH = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--solver", default="beam")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--solver-time-budget-ms",
        type=float,
        default=None,
        help="solver-internal budget (ms); default = timeout-seconds*1000 - 5000 grace",
    )
    parser.add_argument(
        "--no-anytime-fallback",
        dest="enable_anytime_fallback",
        action="store_false",
        default=True,
        help="disable exact->weighted->beam fallback chain",
    )
    parser.add_argument(
        "--enable-depot-late-scheduling",
        dest="enable_depot_late_scheduling",
        action="store_true",
        default=False,
        help="Opt in to depot-late scheduling (secondary lexicographic objective).",
    )
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
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = False,
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
            time_budget_ms=time_budget_ms,
            enable_anytime_fallback=enable_anytime_fallback,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        if not result.plan:
            return {
                "scenario": scenario_path.name,
                "solved": False,
                "error": "no solution within budget",
                "fallback_stage": result.fallback_stage,
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "elapsed_ms": result.elapsed_ms,
                "debug_stats": debug_stats,
            }
        report = result.verification_report
        if report is None:
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
            report = verify_plan(master, normalized, hook_plan)
        return {
            "scenario": scenario_path.name,
            "solved": True,
            "hook_count": len(result.plan),
            "is_valid": report.is_valid,
            "verifier_errors": report.errors,
            "is_proven_optimal": result.is_proven_optimal,
            "fallback_stage": result.fallback_stage,
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
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = False,
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
    if time_budget_ms is not None:
        cmd.extend(["--solver-time-budget-ms", str(time_budget_ms)])
    if not enable_anytime_fallback:
        cmd.append("--no-anytime-fallback")
    if enable_depot_late_scheduling:
        cmd.append("--enable-depot-late-scheduling")
    return cmd


def _run_scenario_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: int,
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = False,
) -> dict[str, Any]:
    cmd = _build_worker_command(
        master_dir=master_dir,
        scenario_path=scenario_path,
        solver=solver,
        beam_width=beam_width,
        heuristic_weight=heuristic_weight,
        time_budget_ms=time_budget_ms,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_worker_process_group(process)
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": "timeout",
            "debug_stats": {},
        }

    if process.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": stderr.strip() or f"worker exited with code {process.returncode}",
            "debug_stats": {},
        }
    return json.loads(stdout)


def _kill_worker_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def run_parallel_scenarios(
    *,
    master_dir: Path,
    scenario_paths: list[Path],
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: int,
    max_workers: int,
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = False,
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
                time_budget_ms=time_budget_ms,
                enable_anytime_fallback=enable_anytime_fallback,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
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
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = False,
) -> list[dict[str, Any]]:
    effective_retry_beam_width = _resolve_retry_no_solution_beam_width(
        beam_width=beam_width,
        retry_no_solution_beam_width=retry_no_solution_beam_width,
    )
    if solver != "beam" or beam_width is None or effective_retry_beam_width is None:
        return initial_results
    retry_beam_widths = _build_retry_beam_widths(
        beam_width=beam_width,
        retry_no_solution_beam_width=effective_retry_beam_width,
    )
    if not retry_beam_widths:
        return initial_results

    scenario_path_by_name = {scenario_path.name: scenario_path for scenario_path in scenario_paths}
    ordered_scenarios = [result["scenario"] for result in initial_results]
    merged_results = {
        result["scenario"]: dict(result)
        for result in initial_results
    }

    for retry_beam_width in retry_beam_widths:
        retry_scenarios = [
            scenario_path_by_name[scenario_name]
            for scenario_name in ordered_scenarios
            if (
                scenario_name in scenario_path_by_name
                and not merged_results[scenario_name].get("solved")
                and "No solution found" in merged_results[scenario_name].get("error", "")
            )
        ]
        if not retry_scenarios:
            break
        retry_results = run_parallel_scenarios(
            master_dir=master_dir,
            scenario_paths=retry_scenarios,
            solver=solver,
            beam_width=retry_beam_width,
            heuristic_weight=heuristic_weight,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            time_budget_ms=time_budget_ms,
            enable_anytime_fallback=enable_anytime_fallback,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
        )
        for retry_result in retry_results:
            if not retry_result.get("solved"):
                continue
            recovered = dict(retry_result)
            recovered["recovery_beam_width"] = retry_beam_width
            merged_results[retry_result["scenario"]] = recovered
    return [merged_results[scenario_name] for scenario_name in ordered_scenarios]


def _resolve_retry_no_solution_beam_width(
    *,
    beam_width: int | None,
    retry_no_solution_beam_width: int | None,
) -> int | None:
    if beam_width is None:
        return None
    if retry_no_solution_beam_width == 0:
        return None
    if retry_no_solution_beam_width is None:
        return beam_width * 3
    return retry_no_solution_beam_width


def _build_retry_beam_widths(
    *,
    beam_width: int,
    retry_no_solution_beam_width: int,
) -> list[int]:
    if retry_no_solution_beam_width <= beam_width:
        return []
    retry_beam_widths: list[int] = []
    multiplier = 2
    while multiplier * beam_width < retry_no_solution_beam_width:
        retry_beam_widths.append(multiplier * beam_width)
        multiplier += 1
    if not retry_beam_widths or retry_beam_widths[-1] != retry_no_solution_beam_width:
        retry_beam_widths.append(retry_no_solution_beam_width)
    return retry_beam_widths


def _run_worker(args: argparse.Namespace) -> None:
    if args.scenario is None:
        raise SystemExit("--worker requires --scenario")
    result = solve_one(
        master_dir=args.master_dir,
        scenario_path=args.scenario,
        solver=args.solver,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        time_budget_ms=getattr(args, "solver_time_budget_ms", None),
        enable_anytime_fallback=getattr(args, "enable_anytime_fallback", True),
        enable_depot_late_scheduling=getattr(args, "enable_depot_late_scheduling", False),
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
    time_budget_ms = getattr(args, "solver_time_budget_ms", None)
    enable_anytime_fallback = getattr(args, "enable_anytime_fallback", True)
    enable_depot_late_scheduling = getattr(args, "enable_depot_late_scheduling", False)
    if time_budget_ms is None:
        time_budget_ms = max(1000.0, args.timeout_seconds * 1000 - 5000)
    results = run_parallel_scenarios(
        master_dir=args.master_dir,
        scenario_paths=scenario_paths,
        solver=args.solver,
        beam_width=args.beam_width,
        heuristic_weight=args.heuristic_weight,
        timeout_seconds=args.timeout_seconds,
        max_workers=args.max_workers,
        time_budget_ms=time_budget_ms,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
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
        time_budget_ms=time_budget_ms,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    summary = {
        "solver": args.solver,
        "beam_width": args.beam_width,
        "heuristic_weight": args.heuristic_weight,
        "timeout_seconds": args.timeout_seconds,
        "solver_time_budget_ms": time_budget_ms,
        "enable_anytime_fallback": enable_anytime_fallback,
        "enable_depot_late_scheduling": enable_depot_late_scheduling,
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
