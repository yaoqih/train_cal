from __future__ import annotations


VALIDATION_DEFAULT_SOLVER = "beam"
VALIDATION_DEFAULT_BEAM_WIDTH = 8
VALIDATION_DEFAULT_MAX_WORKERS = 4
VALIDATION_DEFAULT_TIMEOUT_SECONDS = 60.0
VALIDATION_SOLVER_GRACE_SECONDS = 10.0
VALIDATION_TOTAL_TIMEOUT_SECONDS = 180.0
VALIDATION_PRIMARY_RESERVED_BUDGET_SECONDS = 75.0
VALIDATION_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC = 10
VALIDATION_RETRY_TIMEOUT_SECONDS = 60.0
VALIDATION_MIN_RETRY_ATTEMPT_BUDGET_MS = 5_000.0
VALIDATION_RETRY_TIME_BUDGET_MULTIPLIER = 2.0
VALIDATION_RETRY_MIN_TIME_BUDGET_MS = 90_000.0
VALIDATION_RESERVED_RETRY_BUDGET_MS = 60_000.0
VALIDATION_RETRY_BEAM_WIDTH_MULTIPLIER = 4
VALIDATION_RECOVERY_RISKY_MAX_VEHICLE_TOUCH_COUNT = 80
VALIDATION_RECOVERY_RISKY_STAGING_TO_STAGING_HOOK_COUNT = 8
VALIDATION_RECOVERY_RISKY_REHANDLED_VEHICLE_COUNT = 80
VALIDATION_RECOVERY_ESCALATE_MAX_VEHICLE_TOUCH_COUNT = 80
VALIDATION_RECOVERY_ESCALATE_STAGING_TO_STAGING_HOOK_COUNT = 32
VALIDATION_RECOVERY_ESCALATE_REHANDLED_VEHICLE_COUNT = 80


def validation_time_budget_ms(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    budget_ms = max(
        1_000.0,
        (float(timeout_seconds) - VALIDATION_SOLVER_GRACE_SECONDS) * 1000.0,
    )
    if float(timeout_seconds) >= VALIDATION_TOTAL_TIMEOUT_SECONDS:
        budget_ms = min(
            budget_ms,
            max(
                1_000.0,
                VALIDATION_PRIMARY_RESERVED_BUDGET_SECONDS * 1000.0,
            ),
        )
    return budget_ms


def validation_retry_time_budget_ms(time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return None
    retry_budget = max(
        time_budget_ms * VALIDATION_RETRY_TIME_BUDGET_MULTIPLIER,
        VALIDATION_RETRY_MIN_TIME_BUDGET_MS,
    )
    total_solver_budget = VALIDATION_TOTAL_TIMEOUT_SECONDS * 1000.0
    return max(0.0, min(retry_budget, total_solver_budget - time_budget_ms))


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


def prioritized_validation_recovery_beam_widths(
    retry_beam_widths: list[int],
    *,
    base_beam_width: int,
    time_budget_ms: float | None,
) -> list[int]:
    if len(retry_beam_widths) <= 1:
        return retry_beam_widths
    if time_budget_ms is None:
        return retry_beam_widths
    validation_primary_budget_ms = validation_time_budget_ms(
        VALIDATION_TOTAL_TIMEOUT_SECONDS
    )
    if (
        validation_primary_budget_ms is None
        or float(time_budget_ms) < validation_primary_budget_ms
    ):
        return retry_beam_widths
    wider = sorted(
        (width for width in retry_beam_widths if width > base_beam_width),
        reverse=True,
    )
    same_or_lower = sorted(
        (width for width in retry_beam_widths if width <= base_beam_width),
        reverse=True,
    )
    return [*wider, *same_or_lower]


def validation_recovery_should_continue_after_success(
    *,
    hook_count: int | None,
    max_vehicle_touch_count: int | None,
    staging_to_staging_hook_count: int | None = None,
    rehandled_vehicle_count: int | None = None,
) -> bool:
    """Return True only when a complete recovery result is structurally risky.

    Wider recovery beams are expensive. They are still useful when an early
    complete result is a churn-heavy local basin, but not when the plan is only
    large because the scenario itself is large. Keep this deliberately
    data-independent: repeated touches and staging chains identify avoidable
    churn better than total hooks, which naturally grows with vehicle count
    and required work.
    """
    if (
        max_vehicle_touch_count is not None
        and max_vehicle_touch_count > VALIDATION_RECOVERY_RISKY_MAX_VEHICLE_TOUCH_COUNT
    ):
        return True
    if (
        staging_to_staging_hook_count is not None
        and staging_to_staging_hook_count
        >= VALIDATION_RECOVERY_RISKY_STAGING_TO_STAGING_HOOK_COUNT
    ):
        return True
    if (
        rehandled_vehicle_count is not None
        and rehandled_vehicle_count > VALIDATION_RECOVERY_RISKY_REHANDLED_VEHICLE_COUNT
    ):
        return True
    return False


def validation_recovery_should_escalate_after_success(
    *,
    hook_count: int | None,
    max_vehicle_touch_count: int | None,
    staging_to_staging_hook_count: int | None = None,
    rehandled_vehicle_count: int | None = None,
) -> bool:
    """Return True for complete plans that are likely trapped in a bad basin.

    ``continue`` is intentionally sensitive because a cheap same-beam retry can
    clean up modest churn. Beam widening is different: it spends the reserved
    recovery budget, so use only structural signals that show repeated
    rehandling or staging loops are dominating the plan.
    """
    if (
        max_vehicle_touch_count is not None
        and max_vehicle_touch_count
        > VALIDATION_RECOVERY_ESCALATE_MAX_VEHICLE_TOUCH_COUNT
    ):
        return True
    if (
        staging_to_staging_hook_count is not None
        and staging_to_staging_hook_count
        >= VALIDATION_RECOVERY_ESCALATE_STAGING_TO_STAGING_HOOK_COUNT
    ):
        return True
    if (
        rehandled_vehicle_count is not None
        and rehandled_vehicle_count
        > VALIDATION_RECOVERY_ESCALATE_REHANDLED_VEHICLE_COUNT
    ):
        return True
    return False
