from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import (
    OPERATION_MODE_L7_CLOSED_TOPOLOGY,
    build_l7_closed_topology_workflow_payload,
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_DIR = ROOT / "data" / "master"
DEFAULT_INPUT_ROOT = ROOT / "data" / "validation_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["truth", "positive", "online"],
        choices=["truth", "positive", "online"],
    )
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_sets(
        sets=args.sets,
        master_dir=args.master_dir,
        input_root=args.input_root,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


def evaluate_sets(
    *,
    sets: list[str],
    master_dir: Path,
    input_root: Path,
) -> dict[str, Any]:
    master = load_master_data(master_dir)
    result: dict[str, Any] = {}
    for set_name in sets:
        input_dir = input_root / set_name
        rows: list[dict[str, Any]] = []
        for path in sorted(input_dir.glob("*.json")):
            if path.name == "conversion_summary.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
            started = perf_counter()
            try:
                workflow = build_l7_closed_topology_workflow_payload(master, payload)
                phase1_diag = workflow["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
                rows.append(
                    {
                        "scenario": path.name,
                        "solved": True,
                        "hook_count": None,
                        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                        "phase1_completion_ratio": float(
                            (phase1_diag.get("nonDepotRegionCompletion") or {}).get("completionRatio") or 0.0
                        ),
                        "phase1_selected_vehicle_count": int(phase1_diag.get("selectedVehicleCount") or 0),
                        "phase1_selected_package_sources": list(phase1_diag.get("selectedPackageSourceTracks") or []),
                        "phase1_deferred_vehicle_count": len(phase1_diag.get("deferredVehicleNos") or []),
                        "phase1_selected_package_count": int(phase1_diag.get("selectedPackageCount") or 0),
                        "phase1_depot_compile_ratio": float(phase1_diag.get("depotCompileRatio") or 0.0),
                        "phase1_cun4_clear_ratio": float(phase1_diag.get("cun4ClearRatio") or 0.0),
                        "phase1_ji_purity_ratio": float(phase1_diag.get("jiPurityRatio") or 0.0),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "scenario": path.name,
                        "solved": False,
                        "hook_count": None,
                        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        result[set_name] = summarize_rows(rows)
        result[set_name]["rows"] = rows
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [row for row in rows if row.get("solved")]
    failed = [row for row in rows if not row.get("solved")]
    hook_values = [int(row["hook_count"]) for row in solved if row.get("hook_count") is not None]
    elapsed_values = [float(row["elapsed_ms"]) for row in solved if row.get("elapsed_ms") is not None]
    phase1_completion_values = [
        float(row["phase1_completion_ratio"])
        for row in solved
        if row.get("phase1_completion_ratio") is not None
    ]
    phase1_selected_vehicle_values = [
        int(row["phase1_selected_vehicle_count"])
        for row in solved
        if row.get("phase1_selected_vehicle_count") is not None
    ]
    phase1_deferred_vehicle_values = [
        int(row["phase1_deferred_vehicle_count"])
        for row in solved
        if row.get("phase1_deferred_vehicle_count") is not None
    ]
    phase1_selected_package_values = [
        int(row["phase1_selected_package_count"])
        for row in solved
        if row.get("phase1_selected_package_count") is not None
    ]
    phase1_depot_compile_values = [
        float(row["phase1_depot_compile_ratio"])
        for row in solved
        if row.get("phase1_depot_compile_ratio") is not None
    ]
    phase1_cun4_clear_values = [
        float(row["phase1_cun4_clear_ratio"])
        for row in solved
        if row.get("phase1_cun4_clear_ratio") is not None
    ]
    phase1_ji_purity_values = [
        float(row["phase1_ji_purity_ratio"])
        for row in solved
        if row.get("phase1_ji_purity_ratio") is not None
    ]
    source_counter: Counter[str] = Counter()
    for row in solved:
        for source_track in row.get("phase1_selected_package_sources") or []:
            source_counter[str(source_track)] += 1
    return {
        "scenario_count": len(rows),
        "solved_count": len(solved),
        "failed_count": len(failed),
        "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
        "hook_count_distribution": distribution(hook_values),
        "elapsed_ms_distribution": distribution(elapsed_values),
        "phase1_completion_distribution": distribution(phase1_completion_values),
        "phase1_selected_vehicle_distribution": distribution(phase1_selected_vehicle_values),
        "phase1_deferred_vehicle_distribution": distribution(phase1_deferred_vehicle_values),
        "phase1_selected_package_distribution": distribution(phase1_selected_package_values),
        "phase1_depot_compile_distribution": distribution(phase1_depot_compile_values),
        "phase1_cun4_clear_distribution": distribution(phase1_cun4_clear_values),
        "phase1_ji_purity_distribution": distribution(phase1_ji_purity_values),
        "selected_package_source_distribution": dict(source_counter.most_common()),
        "failed_scenarios": [
            {
                "scenario": row["scenario"],
                "error_type": row.get("error_type"),
                "error": row.get("error"),
            }
            for row in failed
        ],
    }


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "avg": None,
        }
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "p50": round(percentile(ordered, 50), 3),
        "p90": round(percentile(ordered, 90), 3),
        "p95": round(percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
        "avg": round(mean(ordered), 3),
        "med": round(median(ordered), 3),
    }


def percentile(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


if __name__ == "__main__":
    main()
