from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import (
    JI_BUFFER_TRACKS,
    build_l7_closed_topology_workflow_payload,
)


DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[1] / "data" / "master"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "external_validation_inputs"
DEFAULT_SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "validation_inputs_truth_l7_workflow_front_only_phase_exact_stage_logged_20260527"
    / "summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_phase1_failures(
        summary_path=args.summary,
        input_dir=args.input_dir,
        master_dir=args.master_dir,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


def analyze_phase1_failures(
    *,
    summary_path: Path,
    input_dir: Path,
    master_dir: Path,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed_rows: list[dict[str, Any]] = []
    passed_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    for entry in summary.get("results", []):
        scenario_name = str(entry["scenario"])
        scenario_path = _resolve_scenario_path(input_dir, scenario_name)
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        try:
            workflow = build_l7_closed_topology_workflow_payload(master, payload)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({
                "scenario": scenario_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            continue
        row = _build_phase1_case_row(
            scenario_name=scenario_name,
            master=master,
            payload=payload,
            workflow=workflow,
            result_entry=entry,
        )
        if row["failed_stage_name"] == "phase1_pre_repair_buffering":
            failed_rows.append(row)
        elif row["phase1_buffer_vehicle_count"] > 0:
            passed_rows.append(row)
    return {
        "summary_path": str(summary_path),
        "input_dir": str(input_dir),
        "skipped_rows": skipped_rows,
        "phase1_failed_case_count": len(failed_rows),
        "phase1_passed_with_buffer_case_count": len(passed_rows),
        "phase1_failed_aggregate": _aggregate_rows(failed_rows),
        "phase1_passed_with_buffer_aggregate": _aggregate_rows(passed_rows),
        "phase1_failed_top_cases": _top_cases(failed_rows),
        "phase1_passed_top_cases": _top_cases(passed_rows),
    }


def _resolve_scenario_path(input_dir: Path, scenario_name: str) -> Path:
    direct = input_dir / scenario_name
    if direct.exists():
        return direct
    nested = input_dir / "data" / scenario_name
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Scenario not found under {input_dir}: {scenario_name}")


def _build_phase1_case_row(
    *,
    scenario_name: str,
    master,
    payload: dict[str, Any],
    workflow: dict[str, Any],
    result_entry: dict[str, Any],
) -> dict[str, Any]:
    phase1 = workflow["workflowStages"][0]
    goals = list(phase1["vehicleGoals"])
    vehicle_by_no = {
        str(item["vehicleNo"]): item
        for item in payload.get("vehicleInfo", [])
    }
    buffer_goals = [goal for goal in goals if goal.get("targetSource") == "PHASE1_BACKBONE_PLACE"]
    clear_ji_goals = [goal for goal in goals if goal.get("targetSource") == "PHASE1_BLOCKING_EVICT"]
    clear_cun4bei_goals = [goal for goal in goals if goal.get("targetSource") == "PHASE1_CLEAR_CUN4"]
    buffer_source_counts = Counter(
        str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"])
        for goal in buffer_goals
    )
    buffer_source_lengths = _sum_lengths_by_key(
        rows=buffer_goals,
        vehicle_by_no=vehicle_by_no,
        key_getter=lambda goal: str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"]),
    )
    buffer_target_lengths = _sum_lengths_by_key(
        rows=buffer_goals,
        vehicle_by_no=vehicle_by_no,
        key_getter=lambda goal: str(goal["targetTrack"]),
    )
    buffer_capacity_by_track = {
        track: round(float(master.tracks[track].effective_length_m), 1)
        for track in JI_BUFFER_TRACKS
    }
    overflow_by_track = {
        track: round(max(0.0, buffer_target_lengths.get(track, 0.0) - capacity), 1)
        for track, capacity in buffer_capacity_by_track.items()
    }
    failed_stage_name = (
        "solved"
        if result_entry.get("solved")
        else (result_entry.get("debug_stats") or {}).get("failed_stage_name")
    )
    return {
        "scenario": scenario_name,
        "failed_stage_name": failed_stage_name,
        "phase1_buffer_vehicle_count": len(buffer_goals),
        "phase1_buffer_total_length_m": round(
            sum(float(vehicle_by_no[str(goal["vehicleNo"])]["vehicleLength"]) for goal in buffer_goals),
            1,
        ),
        "phase1_clear_ji_vehicle_count": len(clear_ji_goals),
        "phase1_clear_cun4bei_vehicle_count": len(clear_cun4bei_goals),
        "phase1_wash_buffer_vehicle_count": sum(
            1
            for goal in buffer_goals
            if str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"]) in {"洗北", "洗南", "油"}
        ),
        "phase1_prerepair_buffer_vehicle_count": sum(
            1
            for goal in buffer_goals
            if str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"]) == "预修"
        ),
        "phase1_shed_buffer_vehicle_count": sum(
            1
            for goal in buffer_goals
            if str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"]) == "调棚"
        ),
        "phase1_storage_buffer_vehicle_count": sum(
            1
            for goal in buffer_goals
            if str(vehicle_by_no[str(goal["vehicleNo"])]["trackName"])
            in {"存1", "存2", "存3", "存4北", "存4南", "存5北", "存5南"}
        ),
        "phase1_existing_ji_vehicle_count": sum(
            1
            for item in payload.get("vehicleInfo", [])
            if str(item["trackName"]) in JI_BUFFER_TRACKS
        ),
        "phase1_buffer_source_counts": dict(sorted(buffer_source_counts.items())),
        "phase1_buffer_source_lengths_m": buffer_source_lengths,
        "phase1_buffer_target_lengths_m": buffer_target_lengths,
        "phase1_buffer_capacity_by_track_m": buffer_capacity_by_track,
        "phase1_buffer_overflow_by_track_m": overflow_by_track,
        "phase1_buffer_total_overflow_m": round(sum(overflow_by_track.values()), 1),
        "worker_elapsed_ms": result_entry.get("worker_elapsed_ms"),
    }


def _sum_lengths_by_key(
    *,
    rows: list[dict[str, Any]],
    vehicle_by_no: dict[str, dict[str, Any]],
    key_getter,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        key = key_getter(row)
        result[key] = round(
            result.get(key, 0.0) + float(vehicle_by_no[str(row["vehicleNo"])]["vehicleLength"]),
            1,
        )
    return result


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        "phase1_buffer_vehicle_count",
        "phase1_buffer_total_length_m",
        "phase1_clear_ji_vehicle_count",
        "phase1_clear_cun4bei_vehicle_count",
        "phase1_wash_buffer_vehicle_count",
        "phase1_prerepair_buffer_vehicle_count",
        "phase1_shed_buffer_vehicle_count",
        "phase1_storage_buffer_vehicle_count",
        "phase1_existing_ji_vehicle_count",
        "phase1_buffer_total_overflow_m",
    ]
    aggregate = {
        "case_count": len(rows),
        "cases_with_clear_ji": sum(1 for row in rows if row["phase1_clear_ji_vehicle_count"] > 0),
        "cases_with_capacity_overflow": sum(
            1 for row in rows if row["phase1_buffer_total_overflow_m"] > 0.0
        ),
        "cases_with_buffer_ge_12": sum(1 for row in rows if row["phase1_buffer_vehicle_count"] >= 12),
        "cases_with_buffer_ge_15": sum(1 for row in rows if row["phase1_buffer_vehicle_count"] >= 15),
        "cases_with_buffer_len_ge_200m": sum(
            1 for row in rows if row["phase1_buffer_total_length_m"] >= 200.0
        ),
        "cases_with_wash_buffer_ge_3": sum(
            1 for row in rows if row["phase1_wash_buffer_vehicle_count"] >= 3
        ),
        "cases_with_prerepair_plus_shed_ge_10": sum(
            1
            for row in rows
            if (
                row["phase1_prerepair_buffer_vehicle_count"]
                + row["phase1_shed_buffer_vehicle_count"]
            ) >= 10
        ),
        "buffer_source_counts_total": _merge_counter_maps(
            row["phase1_buffer_source_counts"] for row in rows
        ),
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        aggregate[key] = {
            "avg": round(mean(values), 2),
            "med": round(median(values), 2),
            "max": round(max(values), 2),
        }
    return aggregate


def _merge_counter_maps(counter_maps) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for counter_map in counter_maps:
        merged.update(counter_map)
    return dict(sorted(merged.items()))


def _top_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def pressure_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
        return (
            float(row["phase1_buffer_total_overflow_m"]),
            float(row["phase1_buffer_total_length_m"]),
            int(row["phase1_clear_ji_vehicle_count"]),
            int(row["phase1_buffer_vehicle_count"]),
        )

    keys = [
        "scenario",
        "failed_stage_name",
        "phase1_buffer_vehicle_count",
        "phase1_buffer_total_length_m",
        "phase1_clear_ji_vehicle_count",
        "phase1_wash_buffer_vehicle_count",
        "phase1_prerepair_buffer_vehicle_count",
        "phase1_shed_buffer_vehicle_count",
        "phase1_buffer_total_overflow_m",
        "phase1_buffer_source_counts",
        "phase1_buffer_target_lengths_m",
    ]
    top_rows = sorted(rows, key=pressure_key, reverse=True)[:15]
    return [{key: row[key] for key in keys} for row in top_rows]


if __name__ == "__main__":
    main()
