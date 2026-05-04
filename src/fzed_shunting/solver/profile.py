from __future__ import annotations


VALIDATION_DEFAULT_SOLVER = "beam"
VALIDATION_DEFAULT_BEAM_WIDTH = 8
VALIDATION_DEFAULT_MAX_WORKERS = 4
VALIDATION_DEFAULT_TIMEOUT_SECONDS = 60.0
VALIDATION_SOLVER_GRACE_SECONDS = 10.0
VALIDATION_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 10
VALIDATION_RETRY_TIMEOUT_SECONDS = 60.0
VALIDATION_MIN_RETRY_ATTEMPT_BUDGET_MS = 5_000.0
VALIDATION_RETRY_TIME_BUDGET_MULTIPLIER = 2.0
VALIDATION_RETRY_MIN_TIME_BUDGET_MS = 90_000.0
VALIDATION_RETRY_BEAM_WIDTH_MULTIPLIER = 4
VALIDATION_RECOVERY_RISKY_MAX_VEHICLE_TOUCH_COUNT = 80


def validation_time_budget_ms(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    return max(
        1_000.0,
        (float(timeout_seconds) - VALIDATION_SOLVER_GRACE_SECONDS) * 1000.0,
    )


def validation_retry_time_budget_ms(time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return None
    return max(
        time_budget_ms * VALIDATION_RETRY_TIME_BUDGET_MULTIPLIER,
        VALIDATION_RETRY_MIN_TIME_BUDGET_MS,
    )


def validation_retry_beam_widths(
    *,
    beam_width: int | None,
    retry_no_solution_beam_width: int | None = None,
) -> list[int]:
    if beam_width is None:
        return []
    if retry_no_solution_beam_width == 0:
        return []
    retry_limit = (
        beam_width * VALIDATION_RETRY_BEAM_WIDTH_MULTIPLIER
        if retry_no_solution_beam_width is None
        else retry_no_solution_beam_width
    )
    if retry_limit <= beam_width:
        return [beam_width]
    widths: list[int] = [beam_width]
    multiplier = 2
    while multiplier * beam_width < retry_limit:
        widths.append(multiplier * beam_width)
        multiplier += 1
    if widths[-1] != retry_limit:
        widths.append(retry_limit)
    return widths


def validation_recovery_should_continue_after_success(
    *,
    hook_count: int | None,
    max_vehicle_touch_count: int | None,
) -> bool:
    """Return True only when a complete recovery result is structurally risky.

    Wider recovery beams are expensive. They are still useful when an early
    complete result is a churn-heavy local basin, but not when the plan is only
    large because the scenario itself is large. Keep this deliberately
    data-independent: repeated touches identify avoidable churn better than
    total hooks, which naturally grows with vehicle count and required work.
    """
    if (
        max_vehicle_touch_count is not None
        and max_vehicle_touch_count > VALIDATION_RECOVERY_RISKY_MAX_VEHICLE_TOUCH_COUNT
    ):
        return True
    return False
