from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = []
    stage_counter = Counter()
    for result in summary.get("results", []):
        debug = result.get("debug_stats") or {}
        failed_stage = debug.get("failed_stage_name") if not result.get("solved") else "solved"
        stage_counter[failed_stage or "unknown_timeout_or_nonworkflow"] += 1
        phase1 = (debug.get("failed_stage_input_summary") or {}) if failed_stage == "phase1_pre_repair_buffering" else {}
        rows.append({
            "scenario": result.get("scenario"),
            "solved": bool(result.get("solved")),
            "failed_stage": failed_stage or "unknown_timeout_or_nonworkflow",
            "phase1_selected_vehicle_count": phase1.get("phase1_buffer_vehicle_count") or phase1.get("selectedVehicleCount"),
            "phase1_selected_total_length_m": phase1.get("phase1_buffer_total_length_m") or phase1.get("selectedTotalLengthM"),
            "phase1_deferred_vehicle_count": len(phase1.get("deferredVehicleNos") or []),
            "phase1_active_move_count": phase1.get("active_move_count"),
            "phase1_buffer_lengths_m": phase1.get("bufferLengthsM") or phase1.get("phase1_buffer_target_lengths_m"),
        })
    failed_phase1 = [row for row in rows if row["failed_stage"] == "phase1_pre_repair_buffering"]
    aggregate = _aggregate(failed_phase1)
    top_cases = sorted(
        failed_phase1,
        key=lambda row: (
            float(row["phase1_selected_total_length_m"] or 0.0),
            int(row["phase1_selected_vehicle_count"] or 0),
            int(row["phase1_deferred_vehicle_count"] or 0),
        ),
        reverse=True,
    )[:20]
    print(json.dumps({
        "scenario_count": len(rows),
        "failed_stage_distribution": dict(stage_counter),
        "phase1_failed_aggregate": aggregate,
        "phase1_failed_top_cases": top_cases,
    }, ensure_ascii=False, indent=2))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result = {"case_count": len(rows)}
    for key in [
        "phase1_selected_vehicle_count",
        "phase1_selected_total_length_m",
        "phase1_deferred_vehicle_count",
        "phase1_active_move_count",
    ]:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        result[key] = {
            "avg": round(mean(values), 2),
            "med": round(median(values), 2),
            "max": round(max(values), 2),
        }
    return result


if __name__ == "__main__":
    main()
