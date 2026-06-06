from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
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
    parser.add_argument("--scenario-list", type=Path)
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


def _build_workflow(master_dir: Path, scenario_path: Path) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
    return build_l7_closed_topology_workflow_payload(master, payload)


def _phase2_prefix_payload(workflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "trackInfo": workflow["trackInfo"],
        "initialVehicleInfo": workflow["initialVehicleInfo"],
        "locoTrackName": workflow["locoTrackName"],
        "workflowStages": workflow["workflowStages"][:2],
    }


def _final_track_by_vehicle(stage_view: Any) -> dict[str, str]:
    if not stage_view or not stage_view.steps:
        return {}
    return {
        vehicle_no: track
        for track, seq in stage_view.steps[-1].track_sequences.items()
        for vehicle_no in seq
    }


def _run_worker(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    started = perf_counter()
    try:
        workflow = _build_workflow(master_dir, scenario_path)
        result = solve_workflow(
            master,
            _phase2_prefix_payload(workflow),
            solver="beam",
            heuristic_weight=1.0,
            beam_width=beam_width,
            time_budget_ms=solver_time_budget_ms,
            use_validation_recovery=True,
        )
        view = result.stages[1].view
        final_track = _final_track_by_vehicle(view)
        phase2_policy = dict(result.stages[1].input_payload.get("stagePolicy") or {})
        execution_plan = dict(phase2_policy.get("phase2ExecutionPlan") or {})
        must_pull_vehicle_nos = {
            str(item)
            for item in execution_plan.get("mustPullVehicleNos") or ()
        }
        predecessor_unlock_vehicle_nos = {
            str(item)
            for item in execution_plan.get("predecessorUnlockVehicleNos") or ()
        }
        deferred_tail_vehicle_nos = {
            str(item)
            for item in execution_plan.get("deferredTailVehicleNos") or ()
        }
        hard_required_vehicle_nos = must_pull_vehicle_nos | predecessor_unlock_vehicle_nos
        invalid_deferred_tail_vehicle_nos = sorted(deferred_tail_vehicle_nos & hard_required_vehicle_nos)
        must_pull_not_on_cun4 = sorted(
            vehicle_no
            for vehicle_no in must_pull_vehicle_nos
            if final_track.get(vehicle_no) != "存4北"
        )
        predecessor_unlock_not_on_cun4 = sorted(
            vehicle_no
            for vehicle_no in predecessor_unlock_vehicle_nos
            if final_track.get(vehicle_no) != "存4北"
        )
        loco_carry = list(getattr(view.steps[-1], "loco_carry_vehicle_nos", []) or []) if view.steps else []
        failures: list[str] = []
        if not bool(view.summary.is_valid):
            failures.append("phase2_summary_invalid")
        if loco_carry:
            failures.append("loco_carry_not_empty")
        if must_pull_not_on_cun4:
            failures.append("must_pull_not_on_cun4")
        if predecessor_unlock_not_on_cun4:
            failures.append("predecessor_unlock_not_on_cun4")
        if invalid_deferred_tail_vehicle_nos:
            failures.append("invalid_deferred_tail")
        return {
            "scenario": scenario_path.name,
            "solved": True,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "canEnterPhase3": not failures,
            "canEnterPhase3Failures": failures,
            "hookCount": int(view.summary.hook_count),
            "isValid": bool(view.summary.is_valid),
            "locoCarryVehicleNos": loco_carry,
            "mustPullVehicleCount": len(must_pull_vehicle_nos),
            "mustPullNotOnCun4VehicleNos": must_pull_not_on_cun4,
            "predecessorUnlockVehicleCount": len(predecessor_unlock_vehicle_nos),
            "predecessorUnlockNotOnCun4VehicleNos": predecessor_unlock_not_on_cun4,
            "invalidDeferredTailVehicleNos": invalid_deferred_tail_vehicle_nos,
            "deferredTailVehicleCount": len(deferred_tail_vehicle_nos),
            "collectionBatchCount": len(execution_plan.get("collectionBatches") or []),
        }
    except WorkflowStageFailure as exc:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "canEnterPhase3": False,
            "canEnterPhase3Failures": ["workflow_stage_failure"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_stage_name": exc.failed_stage_name,
            "failed_stage_index": exc.failed_stage_index,
            "stage_input_summary": exc.stage_input_summary,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "canEnterPhase3": False,
            "canEnterPhase3Failures": ["worker_exception"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_one_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = perf_counter()
    command = [
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
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved": False,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "canEnterPhase3": False,
            "canEnterPhase3Failures": ["worker_process_error"],
            "error_type": "WorkerProcessError",
            "error": completed.stdout.strip() or completed.stderr.strip() or f"exit {completed.returncode}",
        }
    row = json.loads(completed.stdout.strip())
    row["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return row


def _scenario_paths_for_set(args: argparse.Namespace, set_name: str) -> list[Path]:
    base_dir = args.input_root / set_name
    if args.scenario_list is None:
        return sorted(path for path in base_dir.glob("*.json") if path.name != "conversion_summary.json")
    names = [
        line.strip()
        for line in args.scenario_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    paths: list[Path] = []
    for name in names:
        path = Path(name)
        if not path.is_absolute():
            path = base_dir / path.name
        if not path.exists():
            raise FileNotFoundError(f"scenario not found: {path}")
        paths.append(path)
    return paths


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [row for row in rows if row.get("solved")]
    can_enter = [row for row in solved if row.get("canEnterPhase3")]
    failure_counter: Counter[str] = Counter()
    batch_counter: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("canEnterPhase3Failures") or ():
            failure_counter[str(reason)] += 1
        if row.get("collectionBatchCount") is not None:
            batch_counter[str(row["collectionBatchCount"])] += 1
    return {
        "scenario_count": len(rows),
        "solved_count": len(solved),
        "can_enter_phase3_count": len(can_enter),
        "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
        "can_enter_phase3_rate": round(len(can_enter) / len(solved), 4) if solved else None,
        "elapsed_ms_distribution": _distribution([float(row["elapsed_ms"]) for row in solved]),
        "hook_count_distribution": _distribution([
            float(row["hookCount"])
            for row in solved
            if row.get("hookCount") is not None
        ]),
        "collection_batch_count_histogram": dict(sorted(batch_counter.items(), key=lambda item: int(item[0]))),
        "failure_totals": dict(failure_counter.most_common()),
        "failed_scenarios": [
            row
            for row in rows
            if not row.get("canEnterPhase3")
        ],
    }


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
        paths = _scenario_paths_for_set(args, set_name)
        rows: list[dict[str, Any]] = []
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
                try:
                    rows.append(future.result())
                except subprocess.TimeoutExpired as exc:
                    scenario_name = "unknown"
                    if isinstance(exc.cmd, list) and "--scenario" in exc.cmd:
                        scenario_index = exc.cmd.index("--scenario") + 1
                        if scenario_index < len(exc.cmd):
                            scenario_name = Path(str(exc.cmd[scenario_index])).name
                    rows.append({
                        "scenario": scenario_name,
                        "solved": False,
                        "elapsed_ms": round(float(args.timeout_seconds) * 1000.0, 3),
                        "canEnterPhase3": False,
                        "canEnterPhase3Failures": ["worker_timeout"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                if index % 10 == 0 or index == len(paths):
                    print(f"[{set_name}] progress {index}/{len(paths)}", flush=True)
        rows.sort(key=lambda row: str(row["scenario"]))
        result[set_name] = {
            **_summarize_rows(rows),
            "rows": rows,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
