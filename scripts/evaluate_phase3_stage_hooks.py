from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import (
    OPERATION_MODE_L7_CLOSED_TOPOLOGY,
    build_l7_closed_topology_workflow_payload,
)
from fzed_shunting.workflow.runner import WorkflowStageFailure, solve_workflow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_DIR = ROOT / "data" / "master"
DEFAULT_INPUT_ROOT = ROOT / "data" / "validation_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["truth"], choices=["truth", "positive", "online"])
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--solver-time-budget-ms", type=float, default=10_000.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None, "avg": None, "med": None}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p90": round(_percentile(ordered, 90), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
        "avg": round(mean(ordered), 3),
        "med": round(median(ordered), 3),
    }


def _percentile(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _build_phase3_payload(master_dir: Path, scenario_path: Path) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
    workflow = build_l7_closed_topology_workflow_payload(master, payload)
    return {
        "trackInfo": workflow["trackInfo"],
        "initialVehicleInfo": workflow["initialVehicleInfo"],
        "locoTrackName": workflow["locoTrackName"],
        "workflowStages": workflow["workflowStages"][:3],
    }


def _depot_goal_completion(stage_view: Any) -> dict[str, Any]:
    final_sequences = stage_view.steps[-1].track_sequences if stage_view.steps else {}
    final_track_by_vehicle = {
        vehicle_no: track
        for track, seq in final_sequences.items()
        for vehicle_no in seq
    }
    depot_targets = {
        vehicle_no: set(targets)
        for vehicle_no, targets in stage_view.vehicle_target_tracks.items()
        if {"修1", "修2", "修3", "修4", "轮"}.intersection(targets)
    }
    completed = [
        vehicle_no
        for vehicle_no, targets in depot_targets.items()
        if final_track_by_vehicle.get(vehicle_no) in targets
    ]
    return {
        "depotGoalCount": len(depot_targets),
        "depotGoalCompletedCount": len(completed),
        "depotGoalCompletionRate": round(len(completed) / len(depot_targets), 4) if depot_targets else 1.0,
        "uncompletedDepotGoalNos": sorted(set(depot_targets) - set(completed))[:20],
    }


def _run_worker(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = _build_phase3_payload(master_dir, scenario_path)
    started = perf_counter()
    try:
        workflow = solve_workflow(
            master,
            payload,
            solver="beam",
            heuristic_weight=1.0,
            beam_width=beam_width,
            time_budget_ms=solver_time_budget_ms,
            use_validation_recovery=True,
        )
        stage_valid = [bool(stage.view.summary.is_valid) for stage in workflow.stages]
        stage_hooks = [int(stage.view.summary.hook_count) for stage in workflow.stages]
        phase3_view = workflow.stages[2].view
        return {
            "scenario": scenario_path.name,
            "solved": all(stage_valid),
            "stage_valid": stage_valid,
            "stage_hook_counts": stage_hooks,
            "phase3_hook_count": stage_hooks[2],
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            **_depot_goal_completion(phase3_view),
        }
    except WorkflowStageFailure as exc:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_stage_name": exc.failed_stage_name,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }


def _run_one_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--master-dir",
        str(master_dir),
        "--scenario",
        str(scenario_path),
        "--beam-width",
        str(beam_width),
        "--solver-time-budget-ms",
        str(solver_time_budget_ms),
    ]
    started = perf_counter()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": "TimeoutExpired",
            "error": "timeout",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    if completed.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": "WorkerProcessError",
            "error": completed.stdout.strip() or completed.stderr.strip() or f"exit {completed.returncode}",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    row = json.loads(completed.stdout.strip())
    row["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return row


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.scenario is None:
            raise ValueError("--worker requires --scenario")
        print(json.dumps(_run_worker(
            master_dir=args.master_dir,
            scenario_path=args.scenario,
            beam_width=args.beam_width,
            solver_time_budget_ms=args.solver_time_budget_ms,
        ), ensure_ascii=False))
        return

    result: dict[str, Any] = {}
    for set_name in args.sets:
        rows: list[dict[str, Any]] = []
        paths = sorted(path for path in (args.input_root / set_name).glob("*.json") if path.name != "conversion_summary.json")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    _run_one_subprocess,
                    master_dir=args.master_dir,
                    scenario_path=path,
                    beam_width=args.beam_width,
                    solver_time_budget_ms=args.solver_time_budget_ms,
                    timeout_seconds=args.timeout_seconds,
                )
                for path in paths
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % 10 == 0 or index == len(paths):
                    print(f"[{set_name}] progress {index}/{len(paths)}", flush=True)
        rows.sort(key=lambda row: str(row["scenario"]))
        solved = [row for row in rows if row.get("solved")]
        result[set_name] = {
            "scenario_count": len(rows),
            "solved_count": len(solved),
            "failed_count": len(rows) - len(solved),
            "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
            "phase3_hook_distribution": _distribution([float(row["phase3_hook_count"]) for row in solved]),
            "depot_goal_completion_distribution": _distribution([
                float(row["depotGoalCompletionRate"])
                for row in solved
                if row.get("depotGoalCompletionRate") is not None
            ]),
            "failed_scenarios": [row for row in rows if not row.get("solved")],
            "rows": rows,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
