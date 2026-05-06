from __future__ import annotations

from typing import Any

from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.structural_metrics import summarize_plan_shape


UNKNOWN_SCORE = 10**9


def partial_result_is_better(
    candidate: SolverResult,
    incumbent: SolverResult,
) -> bool:
    return partial_result_score(candidate) < partial_result_score(incumbent)


def partial_result_score(result: SolverResult) -> tuple[int, ...]:
    shape = (result.debug_stats or {}).get("plan_shape_metrics")
    if not shape:
        shape = summarize_plan_shape(result.partial_plan)
    return partial_score_from_debug_stats(
        result.debug_stats,
        partial_hook_count=len(result.partial_plan),
        fallback_stage=result.partial_fallback_stage or result.fallback_stage,
        plan_shape_metrics=shape,
    )


def partial_dict_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    return partial_dict_score(candidate) < partial_dict_score(incumbent)


def partial_dict_score(result: dict[str, Any]) -> tuple[int, ...]:
    return partial_score_from_debug_stats(
        result.get("debug_stats"),
        partial_hook_count=_as_int(result.get("partial_hook_count")),
        fallback_stage=result.get("partial_fallback_stage") or result.get("fallback_stage"),
    )


def partial_debug_stats_is_route_clean_tail_candidate(
    debug_stats: dict[str, Any] | None,
) -> bool:
    stats = debug_stats or {}
    structural = stats.get("partial_structural_metrics") or {}
    route_blockage = stats.get("partial_route_blockage_plan") or {}
    unfinished = _as_int(structural.get("unfinished_count"))
    route_pressure = _as_int(route_blockage.get("total_blockage_pressure"))
    front_blockers = _as_int(structural.get("front_blocker_count")) or 0
    loco_carry = _as_int(structural.get("loco_carry_count")) or 0
    if unfinished is None or route_pressure is None:
        return False
    return (
        route_pressure == 0
        and 0 < unfinished <= 12
        and front_blockers <= 4
        and loco_carry == 0
    )


def partial_result_is_route_clean_tail_candidate(result: SolverResult) -> bool:
    if result.is_complete:
        return False
    return partial_debug_stats_is_route_clean_tail_candidate(result.debug_stats)


def partial_score_from_debug_stats(
    debug_stats: dict[str, Any] | None,
    *,
    partial_hook_count: int | None,
    fallback_stage: str | None,
    plan_shape_metrics: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    stats = debug_stats or {}
    structural = stats.get("partial_structural_metrics") or {}
    route_blockage = stats.get("partial_route_blockage_plan") or {}
    shape = plan_shape_metrics or stats.get("plan_shape_metrics") or {}
    has_quality_metrics = bool(structural or route_blockage)
    hook_count = partial_hook_count if partial_hook_count is not None else UNKNOWN_SCORE
    unfinished = _metric(structural, "unfinished_count")
    route_pressure = _metric(route_blockage, "total_blockage_pressure")
    return (
        _unfinished_route_work_score(unfinished, route_pressure),
        route_pressure,
        unfinished,
        _metric(structural, "target_sequence_defect_count", default=0),
        _metric(structural, "work_position_unfinished_count"),
        _metric(structural, "front_blocker_count"),
        _metric(structural, "goal_track_blocker_count"),
        _metric(structural, "staging_debt_count"),
        _metric(structural, "area_random_unfinished_count"),
        _metric(structural, "capacity_overflow_track_count"),
        _metric(structural, "loco_carry_count"),
        _plan_churn_penalty(shape),
        hook_count if has_quality_metrics else -hook_count,
        _stage_rank(fallback_stage),
    )


def _unfinished_route_work_score(unfinished: int, route_pressure: int) -> int:
    if unfinished >= UNKNOWN_SCORE:
        return UNKNOWN_SCORE
    if route_pressure >= UNKNOWN_SCORE:
        return unfinished + UNKNOWN_SCORE // 2
    return unfinished + max(0, route_pressure - 3)


def _metric(values: dict[str, Any], key: str, *, default: int = UNKNOWN_SCORE) -> int:
    value = _as_int(values.get(key))
    return value if value is not None else default


def _plan_churn_penalty(shape: dict[str, Any]) -> int:
    max_touch = _as_int(shape.get("max_vehicle_touch_count")) or 0
    staging_to_staging = _as_int(shape.get("staging_to_staging_hook_count")) or 0
    rehandled = _as_int(shape.get("rehandled_vehicle_count")) or 0
    return max_touch + staging_to_staging * 2 + rehandled


def _stage_rank(stage: str | None) -> int:
    if stage is None:
        return 10
    if "beam" in stage:
        return 0
    if "constructive" in stage:
        return 1
    return 5


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
