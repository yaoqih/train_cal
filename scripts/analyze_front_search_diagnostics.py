from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def _hook_count(result: dict[str, Any]) -> int | None:
    value = result.get("hook_count")
    if value is not None:
        return int(value)
    value = result.get("partial_hook_count")
    if value is not None:
        return int(value)
    value = result.get("effective_hook_count")
    if value is not None:
        return int(value)
    return None


def _first_primitive_heavy_index(debug_stats: dict[str, Any]) -> int | None:
    traces = debug_stats.get("expansion_candidate_traces") or []
    for index, trace in enumerate(traces):
        top_candidates = trace.get("top_candidates") or []
        primitive_count = sum(1 for item in top_candidates if item.get("origin") == "primitive")
        if primitive_count >= 3:
            return index
    return None


def _top_origin(counter: Counter[str], n: int) -> list[tuple[str, int]]:
    return counter.most_common(n)


def _is_release_origin(origin: str) -> bool:
    return origin.startswith("resource_release") or origin in {
        "goal_frontier_source_opening",
        "route_release_frontier",
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    top_n = int(args.top_n)

    selected_counter: Counter[str] = Counter()
    selected_pre_prune_counter: Counter[str] = Counter()
    topk_counter: Counter[str] = Counter()
    prune_loss_counter: Counter[str] = Counter()
    competition_counter: Counter[str] = Counter()
    primary_problem_counter: Counter[str] = Counter()
    selected_reason_counter: Counter[str] = Counter()
    release_overlap_counter: Counter[str] = Counter()
    problem_switch_cases: list[dict[str, Any]] = []
    primitive_heavy_cases: list[tuple[str, int]] = []
    long_unsolved: list[dict[str, Any]] = []

    for result in results:
        debug_stats = result.get("debug_stats") or {}
        selected_counter.update(debug_stats.get("selected_candidate_origin_counts") or {})
        selected_pre_prune_counter.update(
            debug_stats.get("candidate_selected_origin_counts_pre_prune") or {}
        )
        topk_counter.update(debug_stats.get("candidate_topk_origin_counts") or {})
        competition_counter.update(
            debug_stats.get("structural_candidate_competition_origin_counts") or {}
        )
        selected_reason_counter.update(debug_stats.get("selected_candidate_reason_sequence") or {})
        primary_sequences: list[tuple[str, ...]] = []
        for trace in debug_stats.get("expansion_candidate_traces") or []:
            signature = tuple(trace.get("primary_problem_signature") or ())
            if not signature:
                gated = trace.get("gated_problem_set") or {}
                primary = gated.get("primary")
                secondary = gated.get("secondary")
                signature = tuple(
                    item for item in (primary, secondary) if item
                )
            if signature:
                primary_problem_counter.update(signature)
                primary_sequences.append(signature)
            release_origins_by_track: dict[str, set[str]] = {}
            for item in trace.get("top_candidates") or []:
                origin = item.get("origin") or ""
                track = item.get("problem_track") or item.get("intent_anchor") or ""
                if not origin or not track or not _is_release_origin(origin):
                    continue
                release_origins_by_track.setdefault(track, set()).add(origin)
            for track, origins in release_origins_by_track.items():
                if len(origins) >= 2:
                    release_overlap_counter[track] += 1
        for trace in debug_stats.get("beam_prune_traces") or []:
            for item in trace.get("pruned") or []:
                origin = item.get("origin")
                if origin:
                    prune_loss_counter[origin] += 1
        primitive_index = _first_primitive_heavy_index(debug_stats)
        if primitive_index is not None:
            primitive_heavy_cases.append((result["scenario"], primitive_index))
        hook_count = _hook_count(result)
        if not result.get("solved") and hook_count is not None:
            long_unsolved.append(
                {
                    "scenario": result["scenario"],
                    "hook_count": hook_count,
                    "wall_ms": result.get("wall_ms"),
                    "first_primitive_heavy_index": primitive_index,
                    "selected_top": dict(
                        Counter(debug_stats.get("selected_candidate_origin_counts") or {}).most_common(6)
                    ),
                }
            )
        switch_count = 0
        last_signature: tuple[str, ...] | None = None
        for signature in primary_sequences:
            if last_signature is not None and signature != last_signature:
                switch_count += 1
            last_signature = signature
        if primary_sequences:
            problem_switch_cases.append(
                {
                    "scenario": result["scenario"],
                    "switch_count": switch_count,
                    "sequence_count": len(primary_sequences),
                    "switch_ratio": round(switch_count / max(len(primary_sequences) - 1, 1), 3),
                    "solved": bool(result.get("solved")),
                    "hook_count": hook_count,
                    "first_signatures": [list(sig) for sig in primary_sequences[:8]],
                }
            )

    long_unsolved.sort(key=lambda item: (-item["hook_count"], item["scenario"]))
    primitive_heavy_cases.sort(key=lambda item: (item[1], item[0]))
    problem_switch_cases.sort(
        key=lambda item: (
            -int(item["switch_count"]),
            0 if not item["solved"] else 1,
            -(item["hook_count"] or 0),
            item["scenario"],
        )
    )

    report = {
        "summary_path": str(args.summary),
        "scenario_count": len(results),
        "solved_count": sum(1 for result in results if result.get("solved")),
        "selected_origin_top": _top_origin(selected_counter, top_n),
        "selected_reason_top": _top_origin(selected_reason_counter, top_n),
        "selected_pre_prune_origin_top": _top_origin(selected_pre_prune_counter, top_n),
        "topk_origin_top": _top_origin(topk_counter, top_n),
        "pruned_origin_top": _top_origin(prune_loss_counter, top_n),
        "competition_origin_top": _top_origin(competition_counter, top_n),
        "primary_problem_top": _top_origin(primary_problem_counter, top_n),
        "release_overlap_track_top": _top_origin(release_overlap_counter, top_n),
        "primitive_heavy_case_count": len(primitive_heavy_cases),
        "primitive_heavy_cases_earliest": primitive_heavy_cases[:top_n],
        "problem_switch_cases": problem_switch_cases[:top_n],
        "long_unsolved_cases": long_unsolved[:top_n],
        "long_unsolved_avg_hooks": (
            mean(item["hook_count"] for item in long_unsolved[:top_n])
            if long_unsolved
            else None
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
