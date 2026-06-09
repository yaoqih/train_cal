from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python scripts/analyze_l7_phase1234_results.py <phase1234_summary.json>"
        )
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_name, dataset = _extract_dataset(payload)
    rows = list(dataset.get("rows") or [])
    summary = dict(dataset.get("summary") or {})
    print(f"path: {path}")
    print(f"dataset: {dataset_name}")
    print(f"scenario_count: {summary.get('scenario_count', len(rows))}")
    _print_core_counts(summary)
    _print_stage_failure_breakdown(summary)
    _print_hook_and_time_distribution(rows)
    _print_phase2_gate_breakdown(rows)
    _print_phase3_failure_breakdown(rows)
    _print_representative_failures(rows)


def _extract_dataset(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "truth" in payload and isinstance(payload["truth"], dict):
        raw = dict(payload["truth"])
        if "summary" in raw and "rows" in raw:
            merged = dict(raw.get("summary") or {})
            merged["rows"] = list(raw.get("rows") or [])
            return "truth", merged
        return "truth", raw
    if len(payload) == 1:
        key = next(iter(payload))
        value = payload[key]
        if isinstance(value, dict):
            raw = dict(value)
            if "summary" in raw and "rows" in raw:
                merged = dict(raw.get("summary") or {})
                merged["rows"] = list(raw.get("rows") or [])
                return str(key), merged
            return str(key), raw
    return "unknown", payload


def _print_core_counts(summary: dict[str, Any]) -> None:
    keys = [
        "phase1_ok_count",
        "phase2_ok_count",
        "phase2_can_enter_phase3_count",
        "phase3_ok_count",
        "phase4_ok_count",
        "phase1234_ok_count",
    ]
    for key in keys:
        print(f"{key}: {summary.get(key)}")


def _print_stage_failure_breakdown(summary: dict[str, Any]) -> None:
    print("failed_stage_distribution:")
    for key, value in sorted((summary.get("failed_stage_distribution") or {}).items()):
        print(f"  {key}: {value}")
    print("phase_path_class_distribution:")
    for key, value in sorted((summary.get("phase_path_class_distribution") or {}).items()):
        print(f"  {key}: {value}")


def _print_hook_and_time_distribution(rows: list[dict[str, Any]]) -> None:
    _print_distribution("elapsed_ms", [_as_float(row.get("elapsed_ms")) for row in rows])
    _print_distribution(
        "phase1_hook_count",
        [_as_float((row.get("phase1Actual") or {}).get("hookCount")) for row in rows],
    )
    _print_distribution(
        "phase2_hook_count",
        [_as_float((row.get("phase2Actual") or {}).get("hookCount")) for row in rows],
    )
    _print_distribution(
        "phase3_hook_count",
        [_as_float((row.get("phase3Actual") or {}).get("hookCount")) for row in rows],
    )
    _print_distribution(
        "phase4_hook_count",
        [_as_float((row.get("phase4Actual") or {}).get("hookCount")) for row in rows],
    )


def _print_phase2_gate_breakdown(rows: list[dict[str, Any]]) -> None:
    gate_rows = [
        row
        for row in rows
        if (row.get("phase2Prefix") or {}).get("ok")
    ]
    failures = [
        row
        for row in gate_rows
        if not (row.get("phase2Actual") or {}).get("canEnterPhase3")
    ]
    print(f"phase2_gate_fail_count: {len(failures)}")
    reason_counter: Counter[str] = Counter()
    blocking_track_counter: Counter[str] = Counter()
    runtime_reason_counter: Counter[str] = Counter()
    for row in failures:
        actual = row.get("phase2Actual") or {}
        runtime = actual.get("runtime") or {}
        for reason in actual.get("canEnterPhase3Failures") or ():
            reason_counter[str(reason)] += 1
        for track, count in (runtime.get("runtimeDeferredBlockingTracks") or {}).items():
            blocking_track_counter[str(track)] += int(count)
        for reason, count in (runtime.get("runtimeDeferredReasons") or {}).items():
            runtime_reason_counter[str(reason)] += int(count)
    print("phase2_gate_failure_reasons:")
    for key, value in reason_counter.most_common():
        print(f"  {key}: {value}")
    print("phase2_gate_blocking_tracks:")
    for key, value in blocking_track_counter.most_common(10):
        print(f"  {key}: {value}")
    print("phase2_gate_runtime_deferred_reasons:")
    for key, value in runtime_reason_counter.most_common():
        print(f"  {key}: {value}")


def _print_phase3_failure_breakdown(rows: list[dict[str, Any]]) -> None:
    phase3_failed = [
        row
        for row in rows
        if (row.get("phase1Prefix") or {}).get("ok")
        and (row.get("phase2Prefix") or {}).get("ok")
        and not (row.get("phase3Prefix") or {}).get("ok")
    ]
    print(f"phase3_failure_count: {len(phase3_failed)}")
    error_counter: Counter[str] = Counter()
    source_track_counter: Counter[str] = Counter()
    target_track_counter: Counter[str] = Counter()
    active_move_counts: list[float] = []
    for row in phase3_failed:
        prefix = row.get("phase3Prefix") or {}
        error_counter[_phase3_error_bucket(prefix.get("error"))] += 1
        stage_summary = prefix.get("stage_input_summary") or {}
        active_move_counts.append(_as_float(stage_summary.get("active_move_count")) or 0.0)
        for track, count in (stage_summary.get("active_source_tracks") or {}).items():
            source_track_counter[str(track)] += int(count)
        for track, count in (stage_summary.get("active_target_tracks") or {}).items():
            target_track_counter[str(track)] += int(count)
    print("phase3_failure_error_buckets:")
    for key, value in error_counter.most_common():
        print(f"  {key}: {value}")
    print("phase3_failure_active_source_tracks:")
    for key, value in source_track_counter.most_common(10):
        print(f"  {key}: {value}")
    print("phase3_failure_active_target_tracks:")
    for key, value in target_track_counter.most_common(10):
        print(f"  {key}: {value}")
    _print_distribution("phase3_failure_active_move_count", active_move_counts)


def _print_representative_failures(rows: list[dict[str, Any]]) -> None:
    print("representative_failures:")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("solved1234"):
            continue
        key = str(row.get("failedAt") or "unknown")
        groups[key].append(row)
    for failed_at, group_rows in sorted(groups.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: (
                _as_float(row.get("elapsed_ms")) or 0.0,
                str(row.get("scenario") or ""),
            ),
            reverse=True,
        )
        print(f"  {failed_at}:")
        for row in ordered[:5]:
            print(
                "    "
                f"{row.get('scenario')} "
                f"elapsed_ms={_fmt_float(row.get('elapsed_ms'))} "
                f"phase_path={row.get('phasePathClass')} "
                f"phase3_error={_short_text(((row.get('phase3Prefix') or {}).get('error')))}"
            )


def _phase3_error_bucket(error: Any) -> str:
    text = str(error or "").strip()
    if not text:
        return "unknown"
    if "cannot be assigned to 修1/修2/修3/修4" in text:
        return "assignment_infeasible"
    if "prefix timeout" in text:
        return "prefix_timeout"
    if "spot_realign_failed" in text:
        return "spot_realign_failed"
    return text[:120]


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


def _fmt_float(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.1f}"


def _short_text(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
