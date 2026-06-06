from __future__ import annotations

import argparse
import json
import signal
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
    DEPOT_OUTER_TRACKS,
    DEPOT_TARGET_TRACKS,
    JI_BUFFER_TRACKS,
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
    parser.add_argument("--prefix-timeout-seconds", type=int, default=60)
    parser.add_argument("--max-stage-count", type=int, default=4, choices=[1, 2, 3, 4])
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


def _build_workflow(master_dir: Path, scenario_path: Path) -> dict[str, Any]:
    master = load_master_data(master_dir)
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
    return build_l7_closed_topology_workflow_payload(master, payload)


def _prefix_payload(workflow: dict[str, Any], stage_count: int) -> dict[str, Any]:
    return {
        "trackInfo": workflow["trackInfo"],
        "initialVehicleInfo": workflow["initialVehicleInfo"],
        "locoTrackName": workflow["locoTrackName"],
        "workflowStages": workflow["workflowStages"][:stage_count],
    }


def _run_prefix(
    *,
    master: Any,
    workflow: dict[str, Any],
    stage_count: int,
    beam_width: int,
    solver_time_budget_ms: float,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = solve_workflow(
            master,
            _prefix_payload(workflow, stage_count),
            solver="beam",
            heuristic_weight=1.0,
            beam_width=beam_width,
            time_budget_ms=solver_time_budget_ms,
            use_validation_recovery=True,
        )
        return {
            "ok": True,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "stage_valid": [bool(stage.view.summary.is_valid) for stage in result.stages],
            "stage_hook_counts": [int(stage.view.summary.hook_count) for stage in result.stages],
            "workflow": result,
        }
    except WorkflowStageFailure as exc:
        return {
            "ok": False,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_stage_name": exc.failed_stage_name,
            "failed_stage_index": exc.failed_stage_index,
            "stage_input_summary": exc.stage_input_summary,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_prefix_with_timeout(
    *,
    master: Any,
    workflow: dict[str, Any],
    stage_count: int,
    beam_width: int,
    solver_time_budget_ms: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    def timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError(f"phase{stage_count} prefix timeout")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return _run_prefix(
            master=master,
            workflow=workflow,
            stage_count=stage_count,
            beam_width=beam_width,
            solver_time_budget_ms=solver_time_budget_ms,
        )
    except TimeoutError as exc:
        return {
            "ok": False,
            "elapsed_ms": round(float(timeout_seconds) * 1000.0, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_stage_name": f"phase{stage_count}_timeout",
            "failed_stage_index": stage_count,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _final_track_by_vehicle(stage_view: Any) -> dict[str, str]:
    if not stage_view or not stage_view.steps:
        return {}
    return {
        vehicle_no: track
        for track, seq in stage_view.steps[-1].track_sequences.items()
        for vehicle_no in seq
    }


def _target_tracks_by_vehicle(stage_view: Any) -> dict[str, set[str]]:
    return {
        str(vehicle_no): {str(track) for track in tracks}
        for vehicle_no, tracks in dict(getattr(stage_view, "vehicle_target_tracks", {}) or {}).items()
    }


def _phase1_planned_diagnostics(workflow: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (
        workflow["workflowStages"][0]
        .get("stagePolicy", {})
        .get("phase1Diagnostics", {})
    )
    non_depot = diagnostics.get("nonDepotRegionCompletion") or {}
    pending_reason = (non_depot.get("pendingByReason") or {}).get("counts") or {}
    return {
        "depotDemandVehicleCount": int(diagnostics.get("depotDemandVehicleCount") or 0),
        "depotCompiledVehicleCount": int(diagnostics.get("depotCompiledVehicleCount") or 0),
        "depotCompileRatio": float(diagnostics.get("depotCompileRatio") or 0.0),
        "cun4ClearRatio": float(diagnostics.get("cun4ClearRatio") or 0.0),
        "jiPurityRatio": float(diagnostics.get("jiPurityRatio") or 0.0),
        "selectedVehicleCount": int(diagnostics.get("selectedVehicleCount") or 0),
        "selectedTotalLengthM": float(diagnostics.get("selectedTotalLengthM") or 0.0),
        "selectedPackageCount": int(diagnostics.get("selectedPackageCount") or 0),
        "selectedPackageSourceTracks": list(diagnostics.get("selectedPackageSourceTracks") or []),
        "selectedExecutionLayerCounts": dict(diagnostics.get("selectedExecutionLayerCounts") or {}),
        "bufferVehicleCounts": dict(diagnostics.get("bufferVehicleCounts") or {}),
        "bufferLengthsM": dict(diagnostics.get("bufferLengthsM") or {}),
        "phase3BranchCount": int(diagnostics.get("phase3BranchCount") or 0),
        "nonDepotRegionTotal": int(non_depot.get("totalVehicleCount") or 0),
        "nonDepotRegionCompleted": int(non_depot.get("completedVehicleCount") or 0),
        "nonDepotRegionCompletionRatio": float(non_depot.get("completionRatio") or 0.0),
        "nonDepotRegionPendingCount": int((non_depot.get("counts") or {}).get("pending") or 0),
        "nonDepotRegionPendingByTrack": dict(non_depot.get("pendingByTrack") or {}),
        "nonDepotRegionPendingByReason": dict(pending_reason),
        "uncompiledDepotVehicleNos": list(diagnostics.get("uncompiledDepotVehicleNos") or []),
        "remainingCun4VehicleNos": list(diagnostics.get("remainingCun4VehicleNos") or []),
        "remainingJiNonDepotVehicleNos": list(diagnostics.get("remainingJiNonDepotVehicleNos") or []),
    }


def _phase1_actual_diagnostics(workflow_result: Any | None, workflow: dict[str, Any]) -> dict[str, Any]:
    if workflow_result is None:
        return {}
    view = workflow_result.stages[0].view
    final_track = _final_track_by_vehicle(view)
    initial_rows = workflow.get("initialVehicleInfo") or []
    depot_eligible = {
        str(row["vehicleNo"])
        for row in initial_rows
        if str(row.get("targetTrack") or "") in DEPOT_TARGET_TRACKS
        and str(row.get("trackName") or "") not in DEPOT_TARGET_TRACKS
        and str(row.get("trackName") or "") not in DEPOT_OUTER_TRACKS
    }
    buffered_depot = {
        vehicle_no
        for vehicle_no in depot_eligible
        if final_track.get(vehicle_no) in JI_BUFFER_TRACKS
    }
    return {
        "hookCount": int(view.summary.hook_count),
        "isValid": bool(view.summary.is_valid),
        "actualDepotEligibleVehicleCount": len(depot_eligible),
        "actualDepotBufferedVehicleCount": len(buffered_depot),
        "actualDepotBufferedRatio": round(len(buffered_depot) / len(depot_eligible), 4) if depot_eligible else 1.0,
        "actualBufferVehicleCounts": dict(sorted(Counter(
            track for track in final_track.values() if track in JI_BUFFER_TRACKS
        ).items())),
    }


def _phase2_planned_diagnostics(workflow: dict[str, Any]) -> dict[str, Any]:
    policy = workflow["workflowStages"][1].get("stagePolicy", {})
    plan = policy.get("phase2ExecutionPlan") or {}
    diagnostics = policy.get("phase2Diagnostics") or {}
    return {
        "depotOutboundVehicleCount": len(policy.get("depotOutboundVehicles") or []),
        "cun4FinalVehicleCount": len(policy.get("cun4FinalVehicles") or []),
        "depotStayVehicleCount": len(policy.get("depotStayVehicles") or []),
        "plannedTransferVehicleCount": len(plan.get("transferVehicleNos") or []),
        "plannedDeferredTailVehicleCount": len(plan.get("deferredTailVehicleNos") or []),
        "plannedTrackLayerCount": len(plan.get("trackLayers") or []),
        "plannedCollectionBatchCount": len(plan.get("collectionBatches") or []),
        "diagnostics": dict(diagnostics),
    }


def _phase2_actual_diagnostics(workflow_result: Any | None) -> dict[str, Any]:
    if workflow_result is None or len(workflow_result.stages) < 2:
        return {}
    view = workflow_result.stages[1].view
    final_track = _final_track_by_vehicle(view)
    runtime = dict((getattr(view, "diagnostics", {}) or {}).get("phase2Runtime") or {})
    planned_transfer = {str(item) for item in runtime.get("plannedTransferVehicleNos") or []}
    transferred = {vehicle_no for vehicle_no in planned_transfer if final_track.get(vehicle_no) == "存4北"}
    stage_payload = workflow_result.stages[1].input_payload
    stage_policy = dict(stage_payload.get("stagePolicy") or {})
    execution_plan = dict(stage_policy.get("phase2ExecutionPlan") or {})
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
    can_enter_phase3_failures = []
    if not bool(view.summary.is_valid):
        can_enter_phase3_failures.append("phase2_summary_invalid")
    if loco_carry:
        can_enter_phase3_failures.append("loco_carry_not_empty")
    if must_pull_not_on_cun4:
        can_enter_phase3_failures.append("must_pull_not_on_cun4")
    if predecessor_unlock_not_on_cun4:
        can_enter_phase3_failures.append("predecessor_unlock_not_on_cun4")
    if invalid_deferred_tail_vehicle_nos:
        can_enter_phase3_failures.append("invalid_deferred_tail")
    return {
        "hookCount": int(view.summary.hook_count),
        "isValid": bool(view.summary.is_valid),
        "actualTransferredVehicleCount": len(transferred),
        "actualTransferCompletionRatio": round(len(transferred) / len(planned_transfer), 4) if planned_transfer else 1.0,
        "locoCarryVehicleNos": loco_carry,
        "mustPullVehicleCount": len(must_pull_vehicle_nos),
        "mustPullNotOnCun4VehicleNos": must_pull_not_on_cun4,
        "predecessorUnlockVehicleCount": len(predecessor_unlock_vehicle_nos),
        "predecessorUnlockNotOnCun4VehicleNos": predecessor_unlock_not_on_cun4,
        "invalidDeferredTailVehicleNos": invalid_deferred_tail_vehicle_nos,
        "canEnterPhase3": not can_enter_phase3_failures,
        "canEnterPhase3Failures": can_enter_phase3_failures,
        "runtime": runtime,
    }


def _phase3_diagnostics(workflow_result: Any | None) -> dict[str, Any]:
    if workflow_result is None or len(workflow_result.stages) < 3:
        return {}
    view = workflow_result.stages[2].view
    final_track = _final_track_by_vehicle(view)
    depot_goals = {
        vehicle_no: tracks
        for vehicle_no, tracks in _target_tracks_by_vehicle(view).items()
        if tracks.intersection(DEPOT_TARGET_TRACKS)
    }
    completed = {
        vehicle_no
        for vehicle_no, tracks in depot_goals.items()
        if final_track.get(vehicle_no) in tracks
    }
    uncompleted = sorted(set(depot_goals) - completed)
    uncompleted_by_target = Counter(
        next(iter(sorted(depot_goals[vehicle_no].intersection(DEPOT_TARGET_TRACKS))), "unknown")
        for vehicle_no in uncompleted
    )
    return {
        "hookCount": int(view.summary.hook_count),
        "isValid": bool(view.summary.is_valid),
        "depotGoalCount": len(depot_goals),
        "depotGoalCompletedCount": len(completed),
        "depotGoalCompletionRate": round(len(completed) / len(depot_goals), 4) if depot_goals else 1.0,
        "uncompletedDepotGoalCount": len(uncompleted),
        "uncompletedDepotGoalNos": uncompleted[:30],
        "uncompletedByTargetTrack": dict(sorted(uncompleted_by_target.items())),
        "verifierErrorCount": len(getattr(view, "verifier_errors", []) or []),
        "verifierErrors": list(getattr(view, "verifier_errors", []) or [])[:10],
    }


def _phase4_diagnostics(workflow_result: Any | None) -> dict[str, Any]:
    if workflow_result is None or len(workflow_result.stages) < 4:
        return {}
    view = workflow_result.stages[3].view
    return {
        "hookCount": int(view.summary.hook_count),
        "isValid": bool(view.summary.is_valid),
        "verifierErrorCount": len(getattr(view, "verifier_errors", []) or []),
        "verifierErrors": list(getattr(view, "verifier_errors", []) or [])[:10],
    }


def _scenario_static_counts(workflow: dict[str, Any]) -> dict[str, Any]:
    rows = workflow.get("initialVehicleInfo") or []
    non_depot_to_depot = [
        row for row in rows
        if str(row.get("targetTrack") or "") in DEPOT_TARGET_TRACKS
        and str(row.get("trackName") or "") not in DEPOT_TARGET_TRACKS
        and str(row.get("trackName") or "") not in DEPOT_OUTER_TRACKS
    ]
    depot_area_outbound = [
        row for row in rows
        if (
            str(row.get("trackName") or "") in DEPOT_TARGET_TRACKS
            or str(row.get("trackName") or "") in DEPOT_OUTER_TRACKS
        )
        and str(row.get("targetTrack") or "") not in DEPOT_TARGET_TRACKS
    ]
    return {
        "initialNonDepotToDepotCount": len(non_depot_to_depot),
        "initialDepotAreaOutboundCount": len(depot_area_outbound),
    }


def _run_worker(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
    prefix_timeout_seconds: int,
    max_stage_count: int,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    workflow = _build_workflow(master_dir, scenario_path)
    started = perf_counter()
    row: dict[str, Any] = {
        "scenario": scenario_path.name,
        "static": _scenario_static_counts(workflow),
        "phase1Planned": _phase1_planned_diagnostics(workflow),
        "phase2Planned": _phase2_planned_diagnostics(workflow),
    }
    prefix_results: dict[int, dict[str, Any]] = {}
    for stage_count in range(1, max_stage_count + 1):
        prefix = _run_prefix_with_timeout(
            master=master,
            workflow=workflow,
            stage_count=stage_count,
            beam_width=beam_width,
            solver_time_budget_ms=solver_time_budget_ms,
            timeout_seconds=prefix_timeout_seconds,
        )
        prefix_results[stage_count] = prefix
        row[f"phase{stage_count}Prefix"] = {
            key: value
            for key, value in prefix.items()
            if key != "workflow"
        }
        if not prefix["ok"]:
            break
    phase1_workflow = prefix_results.get(1, {}).get("workflow")
    phase2_workflow = prefix_results.get(2, {}).get("workflow")
    phase3_workflow = prefix_results.get(3, {}).get("workflow")
    phase4_workflow = prefix_results.get(4, {}).get("workflow")
    row["phase1Actual"] = _phase1_actual_diagnostics(phase1_workflow, workflow)
    row["phase2Actual"] = _phase2_actual_diagnostics(phase2_workflow)
    row["phase3Actual"] = _phase3_diagnostics(phase3_workflow)
    row["phase4Actual"] = _phase4_diagnostics(phase4_workflow)
    for stage_count in range(max_stage_count + 1, 5):
        row[f"phase{stage_count}Prefix"] = {"ok": None, "skipped": True}
    row["solved123"] = bool(prefix_results.get(3, {}).get("ok"))
    row["solved1234"] = bool(prefix_results.get(4, {}).get("ok"))
    row["phasePathClass"] = _phase_path_class(prefix_results)
    row["failedAt"] = _failed_at(prefix_results)
    row["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return row


def _failed_at(prefix_results: dict[int, dict[str, Any]]) -> str | None:
    for stage_count in (1, 2, 3, 4):
        result = prefix_results.get(stage_count)
        if result is None:
            return f"phase{stage_count}_not_run"
        if not result.get("ok"):
            return str(result.get("failed_stage_name") or f"phase{stage_count}")
    return None


def _phase_path_class(prefix_results: dict[int, dict[str, Any]]) -> str:
    phase1_ok = bool(prefix_results.get(1, {}).get("ok"))
    phase2_ok = bool(prefix_results.get(2, {}).get("ok"))
    phase3_ok = bool(prefix_results.get(3, {}).get("ok"))
    phase4_result = prefix_results.get(4)
    phase4_ok = bool((phase4_result or {}).get("ok"))
    if not phase1_ok:
        return "phase1_failed"
    if not phase2_ok:
        return "phase1_ok_phase2_failed"
    if not phase3_ok:
        return "phase12_ok_phase3_failed"
    if phase4_result is None:
        return "phase123_ok_phase4_not_run"
    if not phase4_ok:
        failed_stage_index = phase4_result.get("failed_stage_index")
        if failed_stage_index == 1:
            return "phase123_ok_full4_phase1_failed"
        if failed_stage_index == 2:
            return "phase123_ok_full4_phase2_failed"
        if failed_stage_index == 3:
            return "phase123_ok_full4_phase3_failed"
        return "phase123_ok_phase4_failed"
    return "phase1234_solved"


def _run_one_subprocess(
    *,
    master_dir: Path,
    scenario_path: Path,
    beam_width: int,
    solver_time_budget_ms: float,
    timeout_seconds: int,
    max_stage_count: int,
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
        "--prefix-timeout-seconds",
        str(timeout_seconds),
        "--max-stage-count",
        str(max_stage_count),
    ]
    worker_timeout = timeout_seconds * max_stage_count + 10
    started = perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=worker_timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "scenario": scenario_path.name,
            "solved123": False,
            "solved1234": False,
            "failedAt": "timeout",
            "phasePathClass": "worker_timeout",
            "error_type": "TimeoutExpired",
            "error": "timeout",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    if completed.returncode != 0:
        return {
            "scenario": scenario_path.name,
            "solved123": False,
            "solved1234": False,
            "failedAt": "worker_error",
            "phasePathClass": "worker_error",
            "error_type": "WorkerProcessError",
            "error": completed.stdout.strip() or completed.stderr.strip() or f"exit {completed.returncode}",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    row = json.loads(completed.stdout.strip())
    row["elapsed_ms"] = round((perf_counter() - started) * 1000.0, 3)
    return row


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved123 = [row for row in rows if row.get("solved123")]
    solved1234 = [row for row in rows if row.get("solved1234")]
    phase1_ok = [row for row in rows if (row.get("phase1Prefix") or {}).get("ok")]
    phase2_ok = [row for row in rows if (row.get("phase2Prefix") or {}).get("ok")]
    phase3_ok = [row for row in rows if (row.get("phase3Prefix") or {}).get("ok")]
    phase4_ok = [row for row in rows if (row.get("phase4Prefix") or {}).get("ok")]
    failed_counter = Counter(str(row.get("failedAt") or "solved") for row in rows)
    phase_path_counter = Counter(str(row.get("phasePathClass") or "unknown") for row in rows)
    phase2_runtime_reason_counter: Counter[str] = Counter()
    phase2_blocking_track_counter: Counter[str] = Counter()
    phase2_enter_failure_counter: Counter[str] = Counter()
    phase1_pending_reason_counter: Counter[str] = Counter()
    phase1_pending_track_counter: Counter[str] = Counter()
    for row in rows:
        for reason, count in ((row.get("phase2Actual") or {}).get("runtime") or {}).get("runtimeDeferredReasons", {}).items():
            phase2_runtime_reason_counter[str(reason)] += int(count)
        for track, count in ((row.get("phase2Actual") or {}).get("runtime") or {}).get("runtimeDeferredBlockingTracks", {}).items():
            phase2_blocking_track_counter[str(track)] += int(count)
        for reason in (row.get("phase2Actual") or {}).get("canEnterPhase3Failures") or ():
            phase2_enter_failure_counter[str(reason)] += 1
        for reason, count in (row.get("phase1Planned") or {}).get("nonDepotRegionPendingByReason", {}).items():
            phase1_pending_reason_counter[str(reason)] += int(count)
        for track, count in (row.get("phase1Planned") or {}).get("nonDepotRegionPendingByTrack", {}).items():
            phase1_pending_track_counter[str(track)] += int(count)
    return {
        "scenario_count": len(rows),
        "phase1_ok_count": len(phase1_ok),
        "phase2_ok_count": len(phase2_ok),
        "phase2_can_enter_phase3_count": sum(
            1
            for row in phase2_ok
            if (row.get("phase2Actual") or {}).get("canEnterPhase3")
        ),
        "phase3_ok_count": len(phase3_ok),
        "phase4_ok_count": len(phase4_ok),
        "phase123_ok_count": len(solved123),
        "phase1234_ok_count": len(solved1234),
        "phase1_ok_rate": round(len(phase1_ok) / len(rows), 4) if rows else None,
        "phase2_ok_rate": round(len(phase2_ok) / len(rows), 4) if rows else None,
        "phase2_can_enter_phase3_rate": round(
            sum(
                1
                for row in phase2_ok
                if (row.get("phase2Actual") or {}).get("canEnterPhase3")
            ) / len(phase2_ok),
            4,
        ) if phase2_ok else None,
        "phase3_ok_rate": round(len(phase3_ok) / len(rows), 4) if rows else None,
        "phase4_ok_rate": round(len(phase4_ok) / len(rows), 4) if rows else None,
        "phase123_ok_rate": round(len(solved123) / len(rows), 4) if rows else None,
        "phase1234_ok_rate": round(len(solved1234) / len(rows), 4) if rows else None,
        "failed_stage_distribution": dict(sorted(failed_counter.items())),
        "phase_path_class_distribution": dict(sorted(phase_path_counter.items())),
        "stage1_hook_distribution": _distribution([
            float((row.get("phase1Actual") or {}).get("hookCount"))
            for row in phase1_ok
            if (row.get("phase1Actual") or {}).get("hookCount") is not None
        ]),
        "stage2_hook_distribution": _distribution([
            float((row.get("phase2Actual") or {}).get("hookCount"))
            for row in phase2_ok
            if (row.get("phase2Actual") or {}).get("hookCount") is not None
        ]),
        "stage3_hook_distribution": _distribution([
            float((row.get("phase3Actual") or {}).get("hookCount"))
            for row in phase3_ok
            if (row.get("phase3Actual") or {}).get("hookCount") is not None
        ]),
        "stage4_hook_distribution": _distribution([
            float((row.get("phase4Actual") or {}).get("hookCount"))
            for row in phase4_ok
            if (row.get("phase4Actual") or {}).get("hookCount") is not None
        ]),
        "phase1_depot_compile_ratio_distribution": _distribution([
            float((row.get("phase1Planned") or {}).get("depotCompileRatio"))
            for row in rows
            if (row.get("phase1Planned") or {}).get("depotCompileRatio") is not None
        ]),
        "phase1_actual_depot_buffered_ratio_distribution": _distribution([
            float((row.get("phase1Actual") or {}).get("actualDepotBufferedRatio"))
            for row in phase1_ok
            if (row.get("phase1Actual") or {}).get("actualDepotBufferedRatio") is not None
        ]),
        "phase1_non_depot_completion_distribution": _distribution([
            float((row.get("phase1Planned") or {}).get("nonDepotRegionCompletionRatio"))
            for row in rows
            if (row.get("phase1Planned") or {}).get("nonDepotRegionCompletionRatio") is not None
        ]),
        "phase2_actual_transfer_completion_distribution": _distribution([
            float((row.get("phase2Actual") or {}).get("actualTransferCompletionRatio"))
            for row in phase2_ok
            if (row.get("phase2Actual") or {}).get("actualTransferCompletionRatio") is not None
        ]),
        "phase2_runtime_deferred_vehicle_distribution": _distribution([
            float(((row.get("phase2Actual") or {}).get("runtime") or {}).get("runtimeDeferredVehicleCount"))
            for row in phase2_ok
            if ((row.get("phase2Actual") or {}).get("runtime") or {}).get("runtimeDeferredVehicleCount") is not None
        ]),
        "phase3_depot_goal_completion_distribution": _distribution([
            float((row.get("phase3Actual") or {}).get("depotGoalCompletionRate"))
            for row in phase3_ok
            if (row.get("phase3Actual") or {}).get("depotGoalCompletionRate") is not None
        ]),
        "phase1_pending_reason_totals": dict(phase1_pending_reason_counter.most_common()),
        "phase1_pending_track_totals": dict(phase1_pending_track_counter.most_common()),
        "phase2_enter_phase3_failure_totals": dict(phase2_enter_failure_counter.most_common()),
        "phase2_runtime_deferred_reason_totals": dict(phase2_runtime_reason_counter.most_common()),
        "phase2_runtime_deferred_blocking_track_totals": dict(phase2_blocking_track_counter.most_common()),
        "failed_scenarios": [
            {
                "scenario": row["scenario"],
                "failedAt": row.get("failedAt"),
                "phase1": row.get("phase1Prefix"),
                "phase2": row.get("phase2Prefix"),
                "phase3": row.get("phase3Prefix"),
                "phase4": row.get("phase4Prefix"),
                "phasePathClass": row.get("phasePathClass"),
            }
            for row in rows
            if not row.get("solved1234")
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
            prefix_timeout_seconds=args.prefix_timeout_seconds,
            max_stage_count=args.max_stage_count,
        ), ensure_ascii=False))
        return

    result: dict[str, Any] = {}
    for set_name in args.sets:
        paths = sorted(path for path in (args.input_root / set_name).glob("*.json") if path.name != "conversion_summary.json")
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
                    max_stage_count=args.max_stage_count,
                )
                for path in paths
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
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
