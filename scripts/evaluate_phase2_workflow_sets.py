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


def main() -> None:
    args = parse_args()
    master = load_master_data(args.master_dir)
    result: dict[str, Any] = {}
    for set_name in args.sets:
        rows: list[dict[str, Any]] = []
        for path in sorted((args.input_root / set_name).glob("*.json")):
            if path.name == "conversion_summary.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
            started = perf_counter()
            try:
                workflow = build_l7_closed_topology_workflow_payload(master, payload)
                phase2 = workflow["workflowStages"][1]["stagePolicy"]["phase2Diagnostics"]
                rows.append(
                    {
                        "scenario": path.name,
                        "solved": True,
                        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                        "depot_stay_count": int(phase2.get("depotStayVehicleCount") or 0),
                        "depot_outbound_count": int(phase2.get("depotOutboundVehicleCount") or 0),
                        "cun4_final_count": int(phase2.get("cun4FinalVehicleCount") or 0),
                        "outbound_group_count": int(phase2.get("outboundGroupCount") or 0),
                        "staging_vehicle_count": int(phase2.get("stagingVehicleCount") or 0),
                        "final_segment_vehicle_count": int(phase2.get("finalSegmentVehicleCount") or 0),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "scenario": path.name,
                        "solved": False,
                        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        solved = [row for row in rows if row.get("solved")]
        group_counter = Counter()
        for row in solved:
            group_counter[str(row["outbound_group_count"])] += 1
        result[set_name] = {
            "scenario_count": len(rows),
            "solved_count": len(solved),
            "failed_count": len(rows) - len(solved),
            "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
            "elapsed_ms_distribution": _distribution([float(row["elapsed_ms"]) for row in solved]),
            "depot_stay_distribution": _distribution([float(row["depot_stay_count"]) for row in solved]),
            "depot_outbound_distribution": _distribution([float(row["depot_outbound_count"]) for row in solved]),
            "cun4_final_distribution": _distribution([float(row["cun4_final_count"]) for row in solved]),
            "outbound_group_distribution": _distribution([float(row["outbound_group_count"]) for row in solved]),
            "staging_vehicle_distribution": _distribution([float(row["staging_vehicle_count"]) for row in solved]),
            "final_segment_distribution": _distribution([float(row["final_segment_vehicle_count"]) for row in solved]),
            "group_count_histogram": dict(sorted(group_counter.items(), key=lambda item: int(item[0]))),
            "failed_scenarios": [
                {"scenario": row["scenario"], "error_type": row.get("error_type"), "error": row.get("error")}
                for row in rows
                if not row.get("solved")
            ],
            "rows": rows,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
