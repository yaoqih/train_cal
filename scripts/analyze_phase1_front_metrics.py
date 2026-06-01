from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.workflow.l7_closed_topology_mode import (
    DEPOT_OUTER_TRACKS,
    DEPOT_TARGET_TRACKS,
    JI_BUFFER_TRACKS,
    build_l7_closed_topology_workflow_payload,
)


DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[1] / "data" / "master"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            analyze_phase1_front_metrics(
                input_dir=args.input_dir,
                master_dir=args.master_dir,
                summary_path=args.summary,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def analyze_phase1_front_metrics(
    *,
    input_dir: Path,
    master_dir: Path,
    summary_path: Path | None,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    if not any(input_dir.glob("*.json")) and (input_dir / "data").exists():
        input_dir = input_dir / "data"
    workflow_by_scenario: dict[str, dict[str, Any]] = {}
    ratio_total = 0
    ratio_buffered = 0
    selected_counts: list[int] = []
    selected_lengths: list[float] = []
    region_completion_ratios: list[float] = []
    region_completed_counts: list[int] = []
    region_total_counts: list[int] = []
    region_pending_counts: list[int] = []
    region_bucket_counter: Counter[str] = Counter()
    region_pending_reason_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            normalized = normalize_plan_input(payload, master)
            workflow = build_l7_closed_topology_workflow_payload(master, payload)
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "scenario": path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        workflow_by_scenario[path.name] = workflow
        stage1 = workflow["workflowStages"][0]
        goals = stage1["vehicleGoals"]
        diagnostics = stage1["stagePolicy"]["phase1Diagnostics"]
        selected = [goal for goal in goals if goal.get("targetTrack") in set(JI_BUFFER_TRACKS)]
        selected_counts.append(len(selected))
        vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
        selected_lengths.append(
            round(sum(vehicle_by_no[str(goal["vehicleNo"])].vehicle_length for goal in selected), 1)
        )
        region_completion = diagnostics.get("nonDepotRegionCompletion") or {}
        ratio = region_completion.get("completionRatio")
        if ratio is not None:
            region_completion_ratios.append(float(ratio))
        region_completed_counts.append(int(region_completion.get("completedVehicleCount") or 0))
        region_total_counts.append(int(region_completion.get("totalVehicleCount") or 0))
        region_pending_counts.append(int((region_completion.get("counts") or {}).get("pending") or 0))
        for reason, count in ((region_completion.get("pendingByReason") or {}).get("counts") or {}).items():
            region_pending_reason_counter[reason] += int(count)
        if ratio is not None:
            if ratio >= 0.9:
                region_bucket_counter["ge_90pct"] += 1
            elif ratio >= 0.8:
                region_bucket_counter["80_to_90pct"] += 1
            else:
                region_bucket_counter["lt_80pct"] += 1
        for goal in selected:
            vehicle = vehicle_by_no[str(goal["vehicleNo"])]
            source_counter[vehicle.current_track] += 1
            target_counter[str(goal["targetTrack"])] += 1
        buffered_vehicle_nos = {str(goal["vehicleNo"]) for goal in selected}
        for vehicle in normalized.vehicles:
            if vehicle.current_track in DEPOT_TARGET_TRACKS or vehicle.current_track in DEPOT_OUTER_TRACKS:
                continue
            if not DEPOT_TARGET_TRACKS.intersection(vehicle.goal.allowed_target_tracks):
                continue
            ratio_total += 1
            if vehicle.vehicle_no in buffered_vehicle_nos:
                ratio_buffered += 1
    result: dict[str, Any] = {
        "scenario_count": len(workflow_by_scenario) + len(skipped),
        "valid_scenario_count": len(workflow_by_scenario),
        "skipped_rows": skipped,
        "phase1_front_compile_ratio": {
            "buffered_vehicle_count": ratio_buffered,
            "eligible_vehicle_count": ratio_total,
            "ratio": round(ratio_buffered / ratio_total, 4) if ratio_total else None,
        },
        "phase1_selected_vehicle_count": _stats(selected_counts),
        "phase1_selected_length_m": _stats(selected_lengths),
        "phase1_non_depot_region_completion_ratio": _stats(region_completion_ratios),
        "phase1_non_depot_region_completed_count": _stats(region_completed_counts),
        "phase1_non_depot_region_total_count": _stats(region_total_counts),
        "phase1_non_depot_region_pending_count": _stats(region_pending_counts),
        "phase1_non_depot_region_completion_buckets": dict(region_bucket_counter),
        "phase1_non_depot_region_pending_reason_totals": dict(region_pending_reason_counter),
        "phase1_selected_source_distribution": dict(source_counter.most_common()),
        "phase1_selected_target_distribution": dict(target_counter.most_common()),
    }
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results = summary.get("results", [])
        stage_counter: Counter[str] = Counter()
        solved_hooks: list[int] = []
        phase1_failed_diagnostics: list[dict[str, Any]] = []
        for row in results:
            scenario = str(row["scenario"])
            if row.get("solved"):
                stage_counter["solved"] += 1
                if row.get("hook_count") is not None:
                    solved_hooks.append(int(row["hook_count"]))
                continue
            failed_stage = ((row.get("debug_stats") or {}).get("failed_stage_name")) or "unknown"
            stage_counter[failed_stage] += 1
            if failed_stage == "phase1_pre_repair_buffering":
                workflow = workflow_by_scenario.get(scenario)
                if workflow is None:
                    continue
                phase1_diag = workflow["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
                failed_summary = ((row.get("debug_stats") or {}).get("failed_stage_input_summary")) or {}
                phase1_failed_diagnostics.append(
                    {
                        "scenario": scenario,
                        "active_move_count": failed_summary.get("active_move_count"),
                        "active_move_total_length_m": failed_summary.get("active_move_total_length_m"),
                        "nonDepotRegionCompletion": phase1_diag.get("nonDepotRegionCompletion"),
                        "selectedPackageIds": phase1_diag.get("selectedPackageIds"),
                        "deferredVehicleNos": phase1_diag.get("deferredVehicleNos"),
                        "hiddenVehicleNos": phase1_diag.get("hiddenVehicleNos"),
                    }
                )
        result["validation_summary"] = {
            "failed_stage_distribution": dict(stage_counter),
            "solved_hook_distribution": dict(sorted(Counter(solved_hooks).items())),
            "solved_hook_avg": round(mean(solved_hooks), 2) if solved_hooks else None,
            "solved_hook_max": max(solved_hooks) if solved_hooks else None,
            "phase1_failed_cases": phase1_failed_diagnostics,
        }
    return result


def _stats(values: list[int] | list[float]) -> dict[str, float | int] | dict[str, Any]:
    if not values:
        return {}
    return {
        "avg": round(mean(values), 2),
        "med": round(median(values), 2),
        "max": max(values),
    }


if __name__ == "__main__":
    main()
