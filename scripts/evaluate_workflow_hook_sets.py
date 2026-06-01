from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import OPERATION_MODE_L7_CLOSED_TOPOLOGY
from fzed_shunting.workflow.runner import solve_workflow


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
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "avg": None,
            "med": None,
        }
    return {
        "min": round(min(values), 3),
        "p50": round(_percentile(values, 50) or 0.0, 3),
        "p90": round(_percentile(values, 90) or 0.0, 3),
        "p95": round(_percentile(values, 95) or 0.0, 3),
        "max": round(max(values), 3),
        "avg": round(mean(values), 3),
        "med": round(median(values), 3),
    }


def _solve_one(master_dir: str, scenario_path: str) -> dict[str, Any]:
    master = load_master_data(Path(master_dir))
    path = Path(scenario_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["operationMode"] = OPERATION_MODE_L7_CLOSED_TOPOLOGY
    try:
        workflow = solve_workflow(master, payload)
        stage_hooks = [int(stage.view.summary.hook_count) for stage in workflow.stages]
        stage_valid = [bool(stage.view.summary.is_valid) for stage in workflow.stages]
        return {
            "scenario": path.name,
            "solved": all(stage_valid),
            "hook_count": sum(stage_hooks),
            "stage_hook_counts": stage_hooks,
            "stage_valid": stage_valid,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "scenario": path.name,
            "solved": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def evaluate_set(
    *,
    set_name: str,
    master_dir: Path,
    input_root: Path,
    max_workers: int,
) -> dict[str, Any]:
    input_dir = input_root / set_name
    paths = sorted(p for p in input_dir.glob("*.json") if p.name != "conversion_summary.json")
    rows: list[dict[str, Any]] = []
    total = len(paths)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(_solve_one, str(master_dir), str(path))
            for path in paths
        ]
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[{set_name}] progress {completed}/{total}", flush=True)
    rows.sort(key=lambda row: str(row["scenario"]))
    solved = [row for row in rows if row.get("solved")]
    hook_counts = [int(row["hook_count"]) for row in solved]
    stage_count = max((len(row["stage_hook_counts"]) for row in solved), default=0)
    stage_rows: list[dict[str, Any]] = []
    for index in range(stage_count):
        values = [int(row["stage_hook_counts"][index]) for row in solved]
        stage_rows.append({
            "stage_index": index + 1,
            **_distribution(values),
        })
    return {
        "scenario_count": len(rows),
        "solved_count": len(solved),
        "failed_count": len(rows) - len(solved),
        "solve_rate": round(len(solved) / len(rows), 4) if rows else None,
        "hook_distribution": _distribution(hook_counts),
        "stage_hook_distributions": stage_rows,
        "top10_hook_cases": sorted(
            solved,
            key=lambda row: (-int(row["hook_count"]), str(row["scenario"])),
        )[:10],
        "failed_scenarios": [row for row in rows if not row.get("solved")],
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    summary = {
        set_name: evaluate_set(
            set_name=set_name,
            master_dir=args.master_dir,
            input_root=args.input_root,
            max_workers=args.max_workers,
        )
        for set_name in args.sets
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
