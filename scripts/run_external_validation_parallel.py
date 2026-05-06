from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import (
    RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
    SolverResult,
    solve_with_simple_astar_result,
)
from fzed_shunting.solver.profile import (
    VALIDATION_DEFAULT_BEAM_WIDTH,
    VALIDATION_DEFAULT_MAX_WORKERS,
    VALIDATION_DEFAULT_SOLVER,
    VALIDATION_DEFAULT_TIMEOUT_SECONDS,
    VALIDATION_MIN_RETRY_ATTEMPT_BUDGET_MS,
    VALIDATION_RETRY_TIMEOUT_SECONDS,
    VALIDATION_SOLVER_GRACE_SECONDS,
    VALIDATION_TOTAL_TIMEOUT_SECONDS,
    prioritized_validation_recovery_beam_widths,
    validation_recovery_should_continue_after_success,
    validation_retry_beam_widths,
    validation_retry_time_budget_ms,
    validation_time_budget_ms,
)
from fzed_shunting.solver.partial_selection import (
    partial_debug_stats_is_route_clean_tail_candidate,
    partial_dict_is_better,
    partial_dict_score,
    partial_result_is_better,
)
from fzed_shunting.solver.validation_recovery import solve_with_validation_recovery_result
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state


DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[1] / "data" / "master"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_inputs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_parallel_runs"
DEFAULT_RETRY_NO_SOLUTION_BEAM_WIDTH = None
RECOVERY_MAX_WORKERS = 4
RETRY_BUDGET_BUCKET_MS = 1_000.0
PRIMARY_FIRST_BEAM_BUDGET_MS = 20_000.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--solver", default=VALIDATION_DEFAULT_SOLVER)
    parser.add_argument("--beam-width", type=int, default=VALIDATION_DEFAULT_BEAM_WIDTH)
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=VALIDATION_DEFAULT_MAX_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=int(VALIDATION_DEFAULT_TIMEOUT_SECONDS))
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
        "--disable-depot-late-scheduling",
        dest="enable_depot_late_scheduling",
        action="store_false",
        default=True,
        help="Disable depot-late scheduling.",
    )
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--retry-no-solution-beam-width", type=int, default=DEFAULT_RETRY_NO_SOLUTION_BEAM_WIDTH)
    parser.add_argument(
        "--improve-pathological-success",
        action="store_true",
        default=False,
        help=(
            "continue recovery for already solved plans with very high repeated "
            "vehicle touches; useful for quality sweeps, disabled for default "
            "feasibility validation"
        ),
    )
    parser.add_argument(
        "--near-goal-partial-resume-max-final-heuristic",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--primary-first-beam",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-worker-recovery",
        dest="enable_worker_recovery",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-worker-recovery",
        dest="enable_worker_recovery",
        action="store_false",
        default=False,
        help=argparse.SUPPRESS,
    )
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
    enable_depot_late_scheduling: bool = True,
    near_goal_partial_resume_max_final_heuristic: int | None = None,
    primary_first_beam: bool = False,
    enable_worker_recovery: bool = False,
    improve_pathological_success: bool = False,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    debug_stats: dict[str, Any] = {}
    capacity_warnings = _final_track_goal_capacity_warnings(normalized)

    try:
        if enable_worker_recovery:
            result = _solve_with_worker_recovery_result(
                normalized=normalized,
                initial=initial,
                master=master,
                beam_width=beam_width,
                solver=solver,
                heuristic_weight=heuristic_weight,
                debug_stats=debug_stats,
                time_budget_ms=time_budget_ms,
                enable_anytime_fallback=enable_anytime_fallback,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                primary_first_beam=primary_first_beam,
                improve_pathological_success=improve_pathological_success,
            )
        else:
            result = _solve_with_primary_first_result(
                normalized=normalized,
                initial=initial,
                master=master,
                beam_width=beam_width,
                solver=solver,
                heuristic_weight=heuristic_weight,
                debug_stats=debug_stats,
                time_budget_ms=time_budget_ms,
                enable_anytime_fallback=enable_anytime_fallback,
                enable_depot_late_scheduling=enable_depot_late_scheduling,
                near_goal_partial_resume_max_final_heuristic=(
                    near_goal_partial_resume_max_final_heuristic
                ),
                primary_first_beam=primary_first_beam,
            )
        debug_stats = result.debug_stats or debug_stats
        if not result.is_complete:
            error = "no solution within budget"
            error_category = _classify_solve_error(error)
            return {
                "scenario": scenario_path.name,
                "solved": False,
                "error": error,
                "error_category": error_category,
                "retryable": _is_retryable_error_category(error_category),
                "fallback_stage": result.fallback_stage,
                "is_complete": False,
                "partial_hook_count": len(result.partial_plan),
                "partial_fallback_stage": result.partial_fallback_stage,
                "partial_is_valid": (
                    result.partial_verification_report.is_valid
                    if result.partial_verification_report is not None
                    else None
                ),
                "partial_verifier_errors": (
                    result.partial_verification_report.errors
                    if result.partial_verification_report is not None
                    else []
                ),
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "elapsed_ms": result.elapsed_ms,
                "capacity_warnings": capacity_warnings,
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
        summary = {
            "scenario": scenario_path.name,
            "solved": bool(result.is_complete and report.is_valid),
            "hook_count": len(result.plan),
            "is_complete": result.is_complete,
            "is_valid": report.is_valid,
            "verifier_errors": report.errors,
            "is_proven_optimal": result.is_proven_optimal,
            "fallback_stage": result.fallback_stage,
            "expanded_nodes": result.expanded_nodes,
            "generated_nodes": result.generated_nodes,
            "closed_nodes": result.closed_nodes,
            "elapsed_ms": result.elapsed_ms,
            "depot_earliness": result.depot_earliness,
            "depot_hook_count": result.depot_hook_count,
            "capacity_warnings": capacity_warnings,
            "debug_stats": debug_stats,
        }
        _attach_worker_recovery_summary(summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        error_category = _classify_solve_error(error)
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": error,
            "error_category": error_category,
            "retryable": _is_retryable_error_category(error_category),
            "capacity_warnings": capacity_warnings,
            "debug_stats": debug_stats,
        }


def _solve_with_primary_first_result(
    *,
    normalized,
    initial,
    master,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    debug_stats: dict[str, Any],
    time_budget_ms: float | None,
    enable_anytime_fallback: bool,
    enable_depot_late_scheduling: bool,
    near_goal_partial_resume_max_final_heuristic: int | None,
    primary_first_beam: bool,
) -> SolverResult:
    common_kwargs = dict(
        master=master,
        solver_mode=solver,
        beam_width=beam_width,
        heuristic_weight=heuristic_weight,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
    )
    if near_goal_partial_resume_max_final_heuristic is not None:
        common_kwargs["near_goal_partial_resume_max_final_heuristic"] = (
            near_goal_partial_resume_max_final_heuristic
        )
    if solver != "beam" or not enable_anytime_fallback or not primary_first_beam:
        return solve_with_simple_astar_result(
            normalized,
            initial,
            debug_stats=debug_stats,
            time_budget_ms=time_budget_ms,
            **common_kwargs,
        )

    primary_debug_stats: dict[str, Any] = {}
    primary_budget_ms = _primary_first_beam_budget_ms(time_budget_ms)
    primary_result = solve_with_simple_astar_result(
        normalized,
        initial,
        debug_stats=primary_debug_stats,
        time_budget_ms=primary_budget_ms,
        enable_constructive_seed=False,
        **common_kwargs,
    )
    if primary_result.is_complete:
        debug_stats.clear()
        debug_stats.update(primary_result.debug_stats or primary_debug_stats)
        return primary_result

    rescue_budget_ms = _remaining_primary_first_rescue_budget_ms(
        total_budget_ms=time_budget_ms,
        primary_budget_ms=primary_budget_ms,
        primary_result=primary_result,
    )
    rescue_debug_stats: dict[str, Any] = {}
    rescue_kwargs = dict(common_kwargs)
    rescue_kwargs["near_goal_partial_resume_max_final_heuristic"] = (
        near_goal_partial_resume_max_final_heuristic
        if near_goal_partial_resume_max_final_heuristic is not None
        else RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )
    rescue_result = solve_with_simple_astar_result(
        normalized,
        initial,
        debug_stats=rescue_debug_stats,
        time_budget_ms=rescue_budget_ms,
        enable_constructive_seed=True,
        **rescue_kwargs,
    )
    selected = _better_solver_result(primary_result, rescue_result)
    debug_stats.clear()
    debug_stats.update(selected.debug_stats or {})
    return selected


def _solve_with_worker_recovery_result(
    *,
    normalized,
    initial,
    master,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    debug_stats: dict[str, Any],
    time_budget_ms: float | None,
    enable_anytime_fallback: bool,
    enable_depot_late_scheduling: bool,
    primary_first_beam: bool,
    improve_pathological_success: bool,
) -> SolverResult:
    def solve_result_fn(
        plan_input,
        initial_state,
        *,
        master,
        solver_mode: str,
        heuristic_weight: float,
        beam_width: int | None,
        time_budget_ms: float | None,
        enable_depot_late_scheduling: bool,
        near_goal_partial_resume_max_final_heuristic: int | None = None,
    ) -> SolverResult:
        attempt_debug_stats: dict[str, Any] = {}
        return _solve_with_primary_first_result(
            normalized=plan_input,
            initial=initial_state,
            master=master,
            beam_width=beam_width,
            solver=solver_mode,
            heuristic_weight=heuristic_weight,
            debug_stats=attempt_debug_stats,
            time_budget_ms=time_budget_ms,
            enable_anytime_fallback=enable_anytime_fallback,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            near_goal_partial_resume_max_final_heuristic=(
                near_goal_partial_resume_max_final_heuristic
            ),
            primary_first_beam=primary_first_beam,
        )

    result = solve_with_validation_recovery_result(
        plan_input=normalized,
        initial_state=initial,
        master=master,
        solver_mode=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
        solve_result_fn=solve_result_fn,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
        improve_pathological_success=improve_pathological_success,
    )
    debug_stats.clear()
    debug_stats.update(result.debug_stats or {})
    return result


def _primary_first_beam_budget_ms(time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return PRIMARY_FIRST_BEAM_BUDGET_MS
    if time_budget_ms <= PRIMARY_FIRST_BEAM_BUDGET_MS:
        return time_budget_ms
    return PRIMARY_FIRST_BEAM_BUDGET_MS


def _remaining_primary_first_rescue_budget_ms(
    *,
    total_budget_ms: float | None,
    primary_budget_ms: float | None,
    primary_result: SolverResult,
) -> float | None:
    if total_budget_ms is None:
        return None
    spent_ms = max(
        _result_solver_elapsed_ms_result(primary_result),
        float(primary_budget_ms or 0.0),
    )
    return max(1.0, total_budget_ms - spent_ms)


def _result_solver_elapsed_ms_result(result: SolverResult) -> float:
    return float(result.elapsed_ms or 0.0)


def _better_solver_result(first: SolverResult, second: SolverResult) -> SolverResult:
    if second.is_complete and not first.is_complete:
        return second
    if first.is_complete and not second.is_complete:
        return first
    if first.is_complete and second.is_complete:
        return second if len(second.plan) < len(first.plan) else first
    return second if partial_result_is_better(second, first) else first


def _build_worker_command(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = True,
    near_goal_partial_resume_max_final_heuristic: int | None = None,
    primary_first_beam: bool = False,
    enable_worker_recovery: bool = False,
    improve_pathological_success: bool = False,
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
    if not enable_depot_late_scheduling:
        cmd.append("--disable-depot-late-scheduling")
    if near_goal_partial_resume_max_final_heuristic is not None:
        cmd.extend([
            "--near-goal-partial-resume-max-final-heuristic",
            str(near_goal_partial_resume_max_final_heuristic),
        ])
    if primary_first_beam:
        cmd.append("--primary-first-beam")
    if enable_worker_recovery:
        cmd.append("--enable-worker-recovery")
    if improve_pathological_success:
        cmd.append("--improve-pathological-success")
    return cmd


def _run_scenario_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: float,
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = True,
    near_goal_partial_resume_max_final_heuristic: int | None = None,
    primary_first_beam: bool = False,
    enable_worker_recovery: bool = False,
    improve_pathological_success: bool = False,
) -> dict[str, Any]:
    started_at = perf_counter()
    effective_timeout_seconds = _effective_worker_timeout_seconds(
        timeout_seconds=timeout_seconds,
        time_budget_ms=time_budget_ms,
        enable_worker_recovery=enable_worker_recovery,
    )
    cmd = _build_worker_command(
        master_dir=master_dir,
        scenario_path=scenario_path,
        solver=solver,
        beam_width=beam_width,
        heuristic_weight=heuristic_weight,
        time_budget_ms=time_budget_ms,
        enable_anytime_fallback=enable_anytime_fallback,
        enable_depot_late_scheduling=enable_depot_late_scheduling,
        near_goal_partial_resume_max_final_heuristic=(
            near_goal_partial_resume_max_final_heuristic
        ),
        primary_first_beam=primary_first_beam,
        enable_worker_recovery=enable_worker_recovery,
        improve_pathological_success=improve_pathological_success,
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=effective_timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_worker_process_group(process)
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": "timeout",
            "error_category": "timeout",
            "retryable": False,
            "worker_elapsed_ms": _elapsed_ms_since(started_at),
            "debug_stats": {},
        }
    worker_elapsed_ms = _elapsed_ms_since(started_at)

    if process.returncode != 0:
        error = stderr.strip() or f"worker exited with code {process.returncode}"
        error_category = _classify_solve_error(error)
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error": error,
            "error_category": error_category,
            "retryable": _is_retryable_error_category(error_category),
            "worker_elapsed_ms": worker_elapsed_ms,
            "debug_stats": {},
        }
    result = json.loads(stdout)
    result.setdefault("worker_elapsed_ms", worker_elapsed_ms)
    return result


def _elapsed_ms_since(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000.0, 3)


def _kill_worker_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def _effective_worker_timeout_seconds(
    *,
    timeout_seconds: float,
    time_budget_ms: float | None,
    enable_worker_recovery: bool = False,
) -> float:
    if time_budget_ms is None:
        return float(timeout_seconds) + VALIDATION_SOLVER_GRACE_SECONDS
    worker_budget_ms = max(0.0, float(time_budget_ms))
    if enable_worker_recovery:
        retry_budget_ms = validation_retry_time_budget_ms(time_budget_ms)
        if retry_budget_ms is not None:
            worker_budget_ms += max(0.0, float(retry_budget_ms))
    return worker_budget_ms / 1000.0 + VALIDATION_SOLVER_GRACE_SECONDS


def _retry_time_budget_ms(time_budget_ms: float | None) -> float | None:
    return validation_retry_time_budget_ms(time_budget_ms)


def run_parallel_scenarios(
    *,
    master_dir: Path,
    scenario_paths: list[Path],
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: float,
    max_workers: int,
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = True,
    near_goal_partial_resume_max_final_heuristic: int | None = None,
    primary_first_beam: bool = False,
    enable_worker_recovery: bool = False,
    improve_pathological_success: bool = False,
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
                near_goal_partial_resume_max_final_heuristic=(
                    near_goal_partial_resume_max_final_heuristic
                ),
                primary_first_beam=primary_first_beam,
                enable_worker_recovery=enable_worker_recovery,
                improve_pathological_success=improve_pathological_success,
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
    timeout_seconds: float,
    max_workers: int,
    retry_no_solution_beam_width: int | None,
    initial_results: list[dict[str, Any]],
    time_budget_ms: float | None = None,
    enable_anytime_fallback: bool = True,
    enable_depot_late_scheduling: bool = True,
    enable_worker_recovery: bool = False,
    improve_pathological_success: bool = False,
    deadline_at: float | None = None,
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
    retryable_scenario_names = {
        result["scenario"]
        for result in initial_results
        if (
            (
                not result.get("solved")
                and _is_retryable_result(result)
            )
            or (
                improve_pathological_success
                and
                result.get("solved")
                and _should_continue_recovery_after_success(result)
            )
        )
    }
    merged_results = {
        result["scenario"]: dict(result)
        for result in initial_results
    }
    retry_beam_widths_by_scenario = {
        result["scenario"]: _recovery_beam_widths_for_result(
            result,
            retry_beam_widths=retry_beam_widths,
            base_beam_width=beam_width,
            time_budget_ms=time_budget_ms,
        )
        for result in initial_results
    }
    used_retry_solver_ms_by_scenario = {
        result["scenario"]: _result_recovery_solver_elapsed_ms(result)
        for result in initial_results
    }
    used_retry_worker_ms_by_scenario = {
        result["scenario"]: _result_recovery_worker_elapsed_ms(result)
        for result in initial_results
    }
    used_total_solver_ms_by_scenario = {
        result["scenario"]: _result_total_solver_elapsed_ms(result)
        for result in initial_results
    }
    used_total_worker_ms_by_scenario = {
        result["scenario"]: _result_total_worker_elapsed_ms(result)
        for result in initial_results
    }

    while retryable_scenario_names:
        if (
            deadline_at is not None
            and _runner_deadline_remaining_budget_ms(deadline_at) is None
        ):
            break
        grouped_retry_scenarios = _group_next_retry_scenarios_by_width(
            ordered_scenarios=ordered_scenarios,
            scenario_path_by_name=scenario_path_by_name,
            retryable_scenario_names=retryable_scenario_names,
            retry_beam_widths_by_scenario=retry_beam_widths_by_scenario,
        )
        if not grouped_retry_scenarios:
            break
        for retry_beam_width, retry_scenarios in grouped_retry_scenarios:
            if (
                deadline_at is not None
                and _runner_deadline_remaining_budget_ms(deadline_at) is None
            ):
                break
            attempted_scenario_names = {scenario.name for scenario in retry_scenarios}
            for scenario_name in attempted_scenario_names:
                _consume_next_retry_beam_width(
                    retry_beam_widths_by_scenario,
                    scenario_name=scenario_name,
                    retry_beam_width=retry_beam_width,
                )
            retry_results = _run_retry_scenarios_with_remaining_budget(
                master_dir=master_dir,
                scenario_paths=retry_scenarios,
                solver=solver,
                beam_width=retry_beam_width,
                heuristic_weight=heuristic_weight,
                timeout_seconds=VALIDATION_RETRY_TIMEOUT_SECONDS,
                max_workers=_retry_max_workers(
                    base_beam_width=beam_width,
                    retry_beam_width=retry_beam_width,
                    max_workers=max_workers,
                ),
                base_time_budget_ms=_retry_time_budget_ms(time_budget_ms),
                used_retry_solver_ms_by_scenario=used_retry_solver_ms_by_scenario,
                used_retry_worker_ms_by_scenario=used_retry_worker_ms_by_scenario,
                used_total_solver_ms_by_scenario=used_total_solver_ms_by_scenario,
                used_total_worker_ms_by_scenario=used_total_worker_ms_by_scenario,
                deadline_at=deadline_at,
            enable_anytime_fallback=enable_anytime_fallback,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            enable_worker_recovery=enable_worker_recovery,
            near_goal_partial_resume_max_final_heuristic=(
                RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
            ),
            )
            returned_scenario_names = {result["scenario"] for result in retry_results}
            for scenario_name in attempted_scenario_names - returned_scenario_names:
                if not retry_beam_widths_by_scenario.get(scenario_name):
                    retryable_scenario_names.discard(scenario_name)
            for retry_result in retry_results:
                scenario_name = retry_result["scenario"]
                retry_solver_ms = _result_solver_elapsed_ms(retry_result)
                retry_worker_ms = _result_worker_elapsed_ms(retry_result)
                used_retry_solver_ms_by_scenario[scenario_name] = (
                    used_retry_solver_ms_by_scenario.get(scenario_name, 0.0)
                    + retry_solver_ms
                )
                used_retry_worker_ms_by_scenario[scenario_name] = (
                    used_retry_worker_ms_by_scenario.get(scenario_name, 0.0)
                    + retry_worker_ms
                )
                used_total_solver_ms_by_scenario[scenario_name] = (
                    used_total_solver_ms_by_scenario.get(scenario_name, 0.0)
                    + retry_solver_ms
                )
                used_total_worker_ms_by_scenario[scenario_name] = (
                    used_total_worker_ms_by_scenario.get(scenario_name, 0.0)
                    + retry_worker_ms
                )
                current = merged_results[retry_result["scenario"]]
                if not retry_result.get("solved"):
                    retry_failed = dict(retry_result)
                    retry_failed["recovery_beam_width"] = retry_beam_width
                    retry_failed["recovery_time_budget_ms"] = retry_result.get(
                        "recovery_attempt_time_budget_ms"
                    )
                    retry_failed["total_solver_elapsed_ms"] = round(
                        used_total_solver_ms_by_scenario[scenario_name],
                        3,
                    )
                    retry_failed["total_worker_elapsed_ms"] = round(
                        used_total_worker_ms_by_scenario[scenario_name],
                        3,
                    )
                    if _failed_retry_is_better_partial(retry_failed, current):
                        merged_results[retry_result["scenario"]] = retry_failed
                    if not retry_beam_widths_by_scenario.get(scenario_name):
                        retryable_scenario_names.discard(scenario_name)
                    continue
                recovered = dict(retry_result)
                recovered["recovery_beam_width"] = retry_beam_width
                recovered["recovery_time_budget_ms"] = retry_result.get(
                    "recovery_attempt_time_budget_ms"
                )
                recovered["total_solver_elapsed_ms"] = round(
                    used_total_solver_ms_by_scenario[scenario_name],
                    3,
                )
                recovered["total_worker_elapsed_ms"] = round(
                    used_total_worker_ms_by_scenario[scenario_name],
                    3,
                )
                if (
                    not current.get("solved")
                    or int(recovered.get("hook_count", 10**9))
                    < int(current.get("hook_count", 10**9))
                ):
                    merged_results[retry_result["scenario"]] = recovered
                should_keep_searching_complete = _should_continue_recovery_after_success(
                    recovered
                )
                if current.get("solved") and not improve_pathological_success:
                    should_keep_searching_complete = False
                if not should_keep_searching_complete:
                    retryable_scenario_names.discard(retry_result["scenario"])
                    continue
    return [merged_results[scenario_name] for scenario_name in ordered_scenarios]


def _run_retry_scenarios_with_remaining_budget(
    *,
    master_dir: Path,
    scenario_paths: list[Path],
    solver: str,
    beam_width: int | None,
    heuristic_weight: float,
    timeout_seconds: float,
    max_workers: int,
    base_time_budget_ms: float | None,
    used_retry_solver_ms_by_scenario: dict[str, float],
    used_retry_worker_ms_by_scenario: dict[str, float],
    used_total_solver_ms_by_scenario: dict[str, float],
    used_total_worker_ms_by_scenario: dict[str, float],
    deadline_at: float | None,
    enable_anytime_fallback: bool,
    enable_depot_late_scheduling: bool,
    enable_worker_recovery: bool,
    near_goal_partial_resume_max_final_heuristic: int,
) -> list[dict[str, Any]]:
    grouped_paths: dict[tuple[float | None, float], list[Path]] = {}
    for scenario_path in scenario_paths:
        remaining_budget = _remaining_retry_budget(
            scenario_name=scenario_path.name,
            timeout_seconds=timeout_seconds,
            base_time_budget_ms=base_time_budget_ms,
            used_retry_solver_ms_by_scenario=used_retry_solver_ms_by_scenario,
            used_retry_worker_ms_by_scenario=used_retry_worker_ms_by_scenario,
            used_total_solver_ms_by_scenario=used_total_solver_ms_by_scenario,
            used_total_worker_ms_by_scenario=used_total_worker_ms_by_scenario,
            deadline_at=deadline_at,
        )
        if remaining_budget is None:
            continue
        retry_time_budget_ms, retry_timeout_seconds = _bucket_retry_budget(
            *remaining_budget
        )
        grouped_paths.setdefault((retry_time_budget_ms, retry_timeout_seconds), []).append(
            scenario_path
        )

    retry_results: list[dict[str, Any]] = []
    for (retry_time_budget_ms, retry_timeout_seconds), grouped_scenarios in grouped_paths.items():
        for result in run_parallel_scenarios(
            master_dir=master_dir,
            scenario_paths=grouped_scenarios,
            solver=solver,
            beam_width=beam_width,
            heuristic_weight=heuristic_weight,
            timeout_seconds=retry_timeout_seconds,
            max_workers=max_workers,
            time_budget_ms=retry_time_budget_ms,
            enable_anytime_fallback=enable_anytime_fallback,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            enable_worker_recovery=enable_worker_recovery,
            near_goal_partial_resume_max_final_heuristic=(
                near_goal_partial_resume_max_final_heuristic
            ),
        ):
            result["recovery_attempt_time_budget_ms"] = retry_time_budget_ms
            result["recovery_attempt_timeout_seconds"] = retry_timeout_seconds
            retry_results.append(result)
    retry_results.sort(key=lambda item: item["scenario"])
    return retry_results


def _bucket_retry_budget(
    retry_time_budget_ms: float | None,
    retry_timeout_seconds: float,
) -> tuple[float | None, float]:
    if retry_time_budget_ms is None:
        return None, retry_timeout_seconds
    bucketed_ms = max(
        1.0,
        math.floor(retry_time_budget_ms / RETRY_BUDGET_BUCKET_MS)
        * RETRY_BUDGET_BUCKET_MS,
    )
    bucketed_seconds = max(0.001, bucketed_ms / 1000.0)
    return round(bucketed_ms, 3), bucketed_seconds


def _runner_deadline_remaining_budget_ms(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    remaining_ms = (
        float(deadline_at)
        - perf_counter()
        - VALIDATION_SOLVER_GRACE_SECONDS
    ) * 1000.0
    if remaining_ms < VALIDATION_MIN_RETRY_ATTEMPT_BUDGET_MS:
        return None
    return max(0.0, remaining_ms)


def _remaining_retry_budget(
    *,
    scenario_name: str,
    timeout_seconds: float,
    base_time_budget_ms: float | None,
    used_retry_solver_ms_by_scenario: dict[str, float],
    used_retry_worker_ms_by_scenario: dict[str, float],
    used_total_solver_ms_by_scenario: dict[str, float],
    used_total_worker_ms_by_scenario: dict[str, float],
    deadline_at: float | None = None,
) -> tuple[float | None, float] | None:
    retry_timeout_seconds = max(0.001, timeout_seconds)
    if base_time_budget_ms is None:
        return None, retry_timeout_seconds
    used_retry_ms = max(
        used_retry_solver_ms_by_scenario.get(scenario_name, 0.0),
        used_retry_worker_ms_by_scenario.get(scenario_name, 0.0),
    )
    used_total_ms = max(
        used_total_solver_ms_by_scenario.get(scenario_name, 0.0),
        used_total_worker_ms_by_scenario.get(scenario_name, 0.0),
    )
    total_cap_ms = VALIDATION_TOTAL_TIMEOUT_SECONDS * 1000.0
    retry_worker_budget_ms = max(
        0.0,
        total_cap_ms - used_total_ms - VALIDATION_SOLVER_GRACE_SECONDS * 1000.0,
    )
    deadline_budget_ms = _runner_deadline_remaining_budget_ms(deadline_at)
    if deadline_budget_ms is not None:
        retry_worker_budget_ms = min(retry_worker_budget_ms, deadline_budget_ms)
    remaining_ms = min(base_time_budget_ms - used_retry_ms, retry_worker_budget_ms)
    if remaining_ms < VALIDATION_MIN_RETRY_ATTEMPT_BUDGET_MS:
        return None
    return round(remaining_ms, 3), retry_timeout_seconds


def _result_solver_elapsed_ms(result: dict[str, Any]) -> float:
    return _as_float(result.get("elapsed_ms"))


def _result_worker_elapsed_ms(result: dict[str, Any]) -> float:
    worker_elapsed = _as_float(result.get("worker_elapsed_ms"))
    if worker_elapsed > 0:
        return worker_elapsed
    return _result_solver_elapsed_ms(result)


def _result_recovery_solver_elapsed_ms(result: dict[str, Any]) -> float:
    return _as_float(result.get("total_solver_elapsed_ms"))


def _result_recovery_worker_elapsed_ms(result: dict[str, Any]) -> float:
    worker_elapsed = _as_float(result.get("total_worker_elapsed_ms"))
    if worker_elapsed > 0:
        return worker_elapsed
    return _result_recovery_solver_elapsed_ms(result)


def _result_total_solver_elapsed_ms(result: dict[str, Any]) -> float:
    return max(_result_solver_elapsed_ms(result), _result_recovery_solver_elapsed_ms(result))


def _result_total_worker_elapsed_ms(result: dict[str, Any]) -> float:
    return max(_result_worker_elapsed_ms(result), _result_recovery_worker_elapsed_ms(result))


def _attach_worker_recovery_summary(result: dict[str, Any]) -> None:
    recovery = (result.get("debug_stats") or {}).get("validation_recovery") or {}
    if not recovery:
        return
    if "recovery_beam_width" in recovery:
        result["recovery_beam_width"] = recovery["recovery_beam_width"]
    if "recovery_time_budget_ms" in recovery:
        result["recovery_time_budget_ms"] = recovery["recovery_time_budget_ms"]


def _failed_retry_is_better_partial(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    if incumbent.get("solved"):
        return False
    return partial_dict_is_better(candidate, incumbent)


def _failed_partial_score(result: dict[str, Any]) -> tuple[int, ...]:
    return partial_dict_score(result)


def _should_continue_recovery_after_success(result: dict[str, Any]) -> bool:
    shape = (result.get("debug_stats") or {}).get("plan_shape_metrics") or {}
    return validation_recovery_should_continue_after_success(
        hook_count=_as_int(result.get("hook_count")),
        max_vehicle_touch_count=_as_int(shape.get("max_vehicle_touch_count")),
        staging_to_staging_hook_count=_as_int(shape.get("staging_to_staging_hook_count")),
        rehandled_vehicle_count=_as_int(shape.get("rehandled_vehicle_count")),
    )


def _retry_max_workers(
    *,
    base_beam_width: int,
    retry_beam_width: int,
    max_workers: int,
) -> int:
    return max(
        1,
        min(
            int(max_workers),
            RECOVERY_MAX_WORKERS,
            int(max_workers * base_beam_width / max(base_beam_width, retry_beam_width)),
        ),
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, parsed)


def _is_retryable_no_solution_error(error: str) -> bool:
    normalized = (error or "").strip().lower()
    return normalized in {"no solution found", "no solution within budget"}


def _is_retryable_result(result: dict[str, Any]) -> bool:
    error_category = result.get("error_category")
    if error_category in {"no_solution", "timeout"}:
        return True
    if error_category is not None:
        return False
    return _is_retryable_no_solution_error(result.get("error", ""))


def _classify_solve_error(error: str) -> str:
    normalized = (error or "").strip().lower()
    if _is_retryable_no_solution_error(normalized):
        return "no_solution"
    if normalized == "timeout":
        return "timeout"
    return "solver_error"


def _is_retryable_error_category(error_category: str) -> bool:
    return error_category in {"no_solution", "timeout"}


def _final_track_goal_capacity_warnings(normalized) -> list[dict[str, float | str]]:
    capacity_by_track = {
        info.track_name: info.track_distance
        for info in normalized.track_info
    }
    initial_occupation_by_track: dict[str, float] = {}
    for vehicle in normalized.vehicles:
        initial_occupation_by_track[vehicle.current_track] = (
            initial_occupation_by_track.get(vehicle.current_track, 0.0)
            + vehicle.vehicle_length
        )
    final_length_by_track: dict[str, float] = {}
    for vehicle in normalized.vehicles:
        if vehicle.goal.target_mode != "TRACK":
            continue
        final_length_by_track[vehicle.goal.target_track] = (
            final_length_by_track.get(vehicle.goal.target_track, 0.0)
            + vehicle.vehicle_length
        )
    warnings: list[dict[str, float | str]] = []
    for track_name, total_length in final_length_by_track.items():
        capacity = capacity_by_track.get(track_name)
        if capacity is None:
            continue
        effective_capacity = max(
            capacity,
            initial_occupation_by_track.get(track_name, 0.0),
        )
        if total_length > effective_capacity + 1e-9:
            warnings.append(
                {
                    "track": track_name,
                    "required_length": round(total_length, 1),
                    "capacity": round(capacity, 1),
                    "effective_capacity": round(effective_capacity, 1),
                }
            )
    return warnings


def _resolve_retry_no_solution_beam_width(
    *,
    beam_width: int | None,
    retry_no_solution_beam_width: int | None,
) -> int | None:
    if beam_width is None:
        return None
    retry_widths = validation_retry_beam_widths(
        beam_width=beam_width,
        retry_no_solution_beam_width=retry_no_solution_beam_width,
    )
    if not retry_widths:
        return None
    return retry_widths[-1]


def _build_retry_beam_widths(
    *,
    beam_width: int,
    retry_no_solution_beam_width: int,
) -> list[int]:
    return validation_retry_beam_widths(
        beam_width=beam_width,
        retry_no_solution_beam_width=retry_no_solution_beam_width,
    )


def _recovery_beam_widths_for_result(
    result: dict[str, Any],
    *,
    retry_beam_widths: list[int],
    base_beam_width: int,
    time_budget_ms: float | None,
) -> list[int]:
    if _result_is_route_clean_tail_candidate(result):
        return list(retry_beam_widths)
    return _prioritized_recovery_beam_widths(
        retry_beam_widths,
        base_beam_width=base_beam_width,
        time_budget_ms=time_budget_ms,
    )


def _result_is_route_clean_tail_candidate(result: dict[str, Any]) -> bool:
    if result.get("solved"):
        return False
    return partial_debug_stats_is_route_clean_tail_candidate(result.get("debug_stats"))


def _group_next_retry_scenarios_by_width(
    *,
    ordered_scenarios: list[str],
    scenario_path_by_name: dict[str, Path],
    retryable_scenario_names: set[str],
    retry_beam_widths_by_scenario: dict[str, list[int]],
) -> list[tuple[int, list[Path]]]:
    grouped: dict[int, list[Path]] = {}
    width_order: list[int] = []
    for scenario_name in ordered_scenarios:
        if (
            scenario_name not in retryable_scenario_names
            or scenario_name not in scenario_path_by_name
        ):
            continue
        widths = retry_beam_widths_by_scenario.get(scenario_name) or []
        if not widths:
            continue
        retry_beam_width = widths[0]
        if retry_beam_width not in grouped:
            grouped[retry_beam_width] = []
            width_order.append(retry_beam_width)
        grouped[retry_beam_width].append(scenario_path_by_name[scenario_name])
    return [(width, grouped[width]) for width in width_order]


def _consume_next_retry_beam_width(
    retry_beam_widths_by_scenario: dict[str, list[int]],
    *,
    scenario_name: str,
    retry_beam_width: int,
) -> None:
    widths = retry_beam_widths_by_scenario.get(scenario_name)
    if not widths:
        return
    if widths[0] == retry_beam_width:
        del widths[0]
        return
    try:
        widths.remove(retry_beam_width)
    except ValueError:
        return


def _prioritized_recovery_beam_widths(
    retry_beam_widths: list[int],
    *,
    base_beam_width: int,
    time_budget_ms: float | None,
) -> list[int]:
    return prioritized_validation_recovery_beam_widths(
        retry_beam_widths,
        base_beam_width=base_beam_width,
        time_budget_ms=time_budget_ms,
    )


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
        enable_depot_late_scheduling=getattr(args, "enable_depot_late_scheduling", True),
        near_goal_partial_resume_max_final_heuristic=getattr(
            args,
            "near_goal_partial_resume_max_final_heuristic",
            None,
        ),
        primary_first_beam=getattr(args, "primary_first_beam", False),
        enable_worker_recovery=getattr(args, "enable_worker_recovery", False),
        improve_pathological_success=getattr(args, "improve_pathological_success", False),
    )
    print(json.dumps(result, ensure_ascii=False))


def _resolve_scenario_path(input_dir: Path, scenario: Path) -> Path:
    if scenario.is_absolute() or scenario.exists():
        return scenario
    return input_dir / scenario


def main() -> None:
    args = parse_args()
    if args.worker:
        _run_worker(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.scenario is not None:
        scenario_paths = [_resolve_scenario_path(args.input_dir, args.scenario)]
    else:
        # Prefer validation_*.json (the original external_validation_inputs
        # naming) for backwards compatibility; fall back to all *.json when
        # none match (so data/validation_inputs/positive/case_*.json etc.
        # also work without a rename).
        scenario_paths = sorted(args.input_dir.glob("validation_*.json"))
        if not scenario_paths:
            scenario_paths = sorted(args.input_dir.glob("*.json"))
    time_budget_ms = getattr(args, "solver_time_budget_ms", None)
    enable_anytime_fallback = getattr(args, "enable_anytime_fallback", True)
    enable_depot_late_scheduling = getattr(args, "enable_depot_late_scheduling", True)
    primary_first_beam = getattr(args, "primary_first_beam", False)
    enable_worker_recovery = getattr(args, "enable_worker_recovery", False)
    if time_budget_ms is None:
        time_budget_ms = validation_time_budget_ms(args.timeout_seconds)
    deadline_at = None
    near_goal_partial_resume_max_final_heuristic = getattr(
        args,
        "near_goal_partial_resume_max_final_heuristic",
        None,
    )
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
        near_goal_partial_resume_max_final_heuristic=(
            near_goal_partial_resume_max_final_heuristic
        ),
        primary_first_beam=primary_first_beam,
        enable_worker_recovery=enable_worker_recovery,
        improve_pathological_success=args.improve_pathological_success,
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
        enable_worker_recovery=enable_worker_recovery,
        improve_pathological_success=args.improve_pathological_success,
        deadline_at=deadline_at,
    )
    solved_cases = sum(1 for result in results if result.get("solved"))
    unsolved_cases = len(results) - solved_cases
    summary = {
        "solver": args.solver,
        "beam_width": args.beam_width,
        "heuristic_weight": args.heuristic_weight,
        "timeout_seconds": args.timeout_seconds,
        "solver_time_budget_ms": time_budget_ms,
        "enable_anytime_fallback": enable_anytime_fallback,
        "enable_depot_late_scheduling": enable_depot_late_scheduling,
        "near_goal_partial_resume_max_final_heuristic": (
            near_goal_partial_resume_max_final_heuristic
        ),
        "primary_first_beam": primary_first_beam,
        "enable_worker_recovery": enable_worker_recovery,
        "scenario_count": len(results),
        "generated_cases": len(scenario_paths),
        "solved_cases": solved_cases,
        "unsolved_cases": unsolved_cases,
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
