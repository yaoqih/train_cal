from __future__ import annotations

from typing import Any

from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.structural_metrics import summarize_plan_shape
from fzed_shunting.solver.types import HookAction


COMPLETE_RESULT_DECISIVE_STAGING_GAP = 8
COMPLETE_RESULT_DECISIVE_TOUCH_GAP = 24


def complete_result_is_better(
    candidate: SolverResult,
    incumbent: SolverResult,
) -> bool:
    if not candidate.is_complete:
        return False
    if not incumbent.is_complete:
        return True
    candidate_score = complete_result_quality_score(
        candidate.debug_stats,
        candidate.plan,
    )
    incumbent_score = complete_result_quality_score(
        incumbent.debug_stats,
        incumbent.plan,
    )
    if complete_quality_gap_is_decisive(candidate_score, incumbent_score):
        return True
    if complete_quality_gap_is_decisive(incumbent_score, candidate_score):
        return False
    if len(candidate.plan) < len(incumbent.plan):
        return True
    return len(candidate.plan) == len(incumbent.plan) and (
        candidate.is_proven_optimal and not incumbent.is_proven_optimal
    )


def complete_dict_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> bool:
    if not candidate.get("solved"):
        return False
    if not incumbent.get("solved"):
        return True
    candidate_score = complete_dict_quality_score(candidate)
    incumbent_score = complete_dict_quality_score(incumbent)
    if complete_quality_gap_is_decisive(candidate_score, incumbent_score):
        return True
    if complete_quality_gap_is_decisive(incumbent_score, candidate_score):
        return False
    candidate_hooks = _as_int(candidate.get("hook_count"))
    incumbent_hooks = _as_int(incumbent.get("hook_count"))
    if candidate_hooks is None:
        return False
    if incumbent_hooks is None:
        return True
    return candidate_hooks < incumbent_hooks


def complete_dict_quality_score(result: dict[str, Any]) -> tuple[int, int]:
    shape = (result.get("debug_stats") or {}).get("plan_shape_metrics") or {}
    return _quality_score_from_shape(shape)


def complete_result_quality_score(
    debug_stats: dict[str, Any] | None,
    plan: list[HookAction],
) -> tuple[int, int]:
    shape = (debug_stats or {}).get("plan_shape_metrics")
    if not shape:
        shape = summarize_plan_shape(plan)
    return _quality_score_from_shape(shape)


def complete_quality_gap_is_decisive(
    candidate_score: tuple[int, int],
    incumbent_score: tuple[int, int],
) -> bool:
    candidate_staging, candidate_touch = candidate_score
    incumbent_staging, incumbent_touch = incumbent_score
    return (
        candidate_staging + COMPLETE_RESULT_DECISIVE_STAGING_GAP <= incumbent_staging
        or candidate_touch + COMPLETE_RESULT_DECISIVE_TOUCH_GAP <= incumbent_touch
    )


def _quality_score_from_shape(shape: dict[str, Any]) -> tuple[int, int]:
    return (
        _as_int(shape.get("staging_to_staging_hook_count")) or 0,
        _as_int(shape.get("max_vehicle_touch_count")) or 0,
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
