from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from statistics import mean, median
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
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["truth"],
        choices=["truth", "positive", "online"],
    )
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--solver-time-budget-ms", type=float, default=50_000.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario", type=Path)
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


def _build_phase2_payload(master_dir: Path, scenario_path: Path) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
    workflow = build_l7_closed_topology_workflow_payload(master, payload)
    return {
        "trackInfo": workflow["trackInfo"],
        "initialVehicleInfo": workflow["initialVehicleInfo"],
        "locoTrackName": workflow["locoTrackName"],
        "workflowStages": workflow["workflowStages"][:2],
    }


def _run_worker(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = _build_phase2_payload(master_dir, scenario_path)
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
        return {
            "scenario": scenario_path.name,
            "solved": bool(workflow.stages[1].view.summary.is_valid),
            "phase2_hook_count": int(workflow.stages[1].view.summary.hook_count),
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
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
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": "TimeoutExpired",
            "error": "timeout",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "error_type": "WorkerProcessError",
            "error": stdout or completed.stderr.strip() or f"exit {completed.returncode}",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    row = json.loads(stdout)
    row["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return row


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.scenario is None:
            raise ValueError("--worker requires --scenario")
        row = _run_worker(
            master_dir=args.master_dir,
            scenario_path=args.scenario,
            beam_width=args.beam_width,
            solver_time_budget_ms=args.solver_time_budget_ms,
        )
        print(json.dumps(row, ensure_ascii=False))
        return
    master = load_master_data(args.master_dir)
    result: dict[str, Any] = {}
    for set_name in args.sets:
        rows: list[dict[str, Any]] = []
        paths = sorted(path for path in (args.input_root / set_name).glob("*.json") if path.name != "conversion_summary.json")
        total = len(paths)
        completed = 0
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
            for future in as_completed(futures):
                rows.append(future.result())
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"[{set_name}] progress {completed}/{total}", flush=True)
        rows.sort(key=lambda row: str(row["scenario"]))
        solved = [row for row in rows if row.get("solved")]
        result[set_name] = {
            "scenario_count": len(rows),
            "solved_count": len(solved),
            "failed_count": len(rows) - len(solved),
            "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
            "phase2_hook_distribution": _distribution([float(row["phase2_hook_count"]) for row in solved]),
            "failed_scenarios": [row for row in rows if not row.get("solved")],
            "rows": rows,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
