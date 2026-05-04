from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/summarize_validation_run.py <summary.json>")
    path = Path(sys.argv[1])
    summary = json.loads(path.read_text(encoding="utf-8"))
    results = list(summary.get("results", []))
    solved = [result for result in results if result.get("solved")]
    failed = [result for result in results if not result.get("solved")]
    print(f"path: {path}")
    print(f"scenario_count: {len(results)}")
    print(f"solved_count: {len(solved)}")
    print(f"unsolved_count: {len(failed)}")
    if results:
        print(f"solve_rate: {len(solved) / len(results):.3f}")
    print(f"error_categories: {dict(_failure_categories(failed))}")
    print(f"fallback_stages: {dict(_counter(solved, 'fallback_stage'))}")
    _print_distribution("hook_count", [_as_float(r.get("hook_count")) for r in solved])
    _print_distribution("elapsed_ms", [_as_float(r.get("elapsed_ms")) for r in solved])
    _print_distribution(
        "worker_elapsed_ms",
        [_as_float(r.get("worker_elapsed_ms")) for r in solved],
    )
    slow = sorted(
        solved,
        key=lambda r: _as_float(r.get("elapsed_ms")),
        reverse=True,
    )[:10]
    if slow:
        print("slowest_solved:")
        for result in slow:
            print(
                "  "
                f"{result.get('scenario')}: "
                f"hooks={result.get('hook_count')} "
                f"elapsed_ms={_as_float(result.get('elapsed_ms')):.1f} "
                f"stage={result.get('fallback_stage')}"
            )
    if failed:
        print("failed:")
        for result in failed[:30]:
            print(
                "  "
                f"{result.get('scenario')}: "
                f"category={_failure_category(result)} "
                f"partial_hooks={result.get('partial_hook_count')} "
                f"partial_stage={result.get('partial_fallback_stage')} "
                f"error={(result.get('error') or '')[:160]}"
            )


def _counter(results: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(result.get(key) or "unknown") for result in results)


def _failure_categories(results: list[dict[str, Any]]) -> Counter[str]:
    return Counter(_failure_category(result) for result in results)


def _failure_category(result: dict[str, Any]) -> str:
    return str(
        result.get("error_category")
        or result.get("error")
        or result.get("partial_fallback_stage")
        or "unknown"
    )


def _print_distribution(name: str, raw_values: list[float | None]) -> None:
    values = sorted(value for value in raw_values if value is not None)
    if not values:
        print(f"{name}: none")
        return
    print(
        f"{name}: "
        f"min={values[0]:.1f} "
        f"p50={_percentile(values, 50):.1f} "
        f"p90={_percentile(values, 90):.1f} "
        f"p95={_percentile(values, 95):.1f} "
        f"max={values[-1]:.1f} "
        f"avg={statistics.mean(values):.1f}"
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
