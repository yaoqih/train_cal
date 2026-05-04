"""Solver result data structures, verification exception, and telemetry."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from fzed_shunting.solver.types import HookAction


@dataclass(frozen=True)
class SolverTelemetry:
    """Structured observability record for a single solver invocation.

    Designed for production metrics collection: timings are in milliseconds,
    counts are plain integers, and the record serialises cleanly to JSON via
    ``asdict``. Emitted to ``FZED_SOLVER_TELEMETRY_PATH`` (JSON-lines append)
    when the env var is set — for collection by a log tailer or sidecar.
    """

    # Input shape
    input_vehicle_count: int = 0
    input_track_count: int = 0
    input_weigh_count: int = 0
    input_close_door_count: int = 0
    input_spot_count: int = 0
    input_area_count: int = 0
    input_work_position_count: int = 0

    # Phase timings (ms)
    constructive_ms: float = 0.0
    exact_ms: float = 0.0
    anytime_ms: float = 0.0
    lns_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0

    # Outcome
    is_complete: bool = False
    plan_hook_count: int = 0
    fallback_stage: str | None = None
    is_valid: bool | None = None
    partial_hook_count: int = 0
    partial_fallback_stage: str | None = None
    partial_is_valid: bool | None = None
    is_proven_optimal: bool = False

    # Solver budget
    time_budget_ms: float | None = None
    node_budget: int | None = None


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a solver invocation.

    Attributes:
        plan: ordered list of hooks for a complete deliverable solution.
            A complete zero-hook solution is represented as ``is_complete=True``
            and ``plan=[]``.
        is_complete: True only when ``plan`` is a complete deliverable
            solution. False means the solve did not finish.
        partial_plan: best-effort artifact for debugging/analysis when the
            solver failed to finish. Never treated as a solved result.
        expanded_nodes / generated_nodes / closed_nodes: search telemetry.
        elapsed_ms: wall-clock time spent producing this result.
        is_proven_optimal: True only when exact search ran to completion.
        fallback_stage: which anytime stage produced the final plan
            (``"exact" | "weighted" | "beam" | "weighted_greedy" |
            "beam_greedy_128" | "beam_greedy_256" | "weighted_very_greedy" |
            "constructive" | "constructive_warm_start"``).
        partial_fallback_stage: which stage produced ``partial_plan``.
        verification_report: verifier result for the complete plan.
        partial_verification_report: verifier result for ``partial_plan``.
        debug_stats: optional telemetry bag.
        telemetry: structured per-phase metrics for production observability.
        depot_earliness / depot_hook_count: populated for observability when enable_depot_late_scheduling is available; may be None.
    """

    plan: list[HookAction]
    expanded_nodes: int
    generated_nodes: int
    closed_nodes: int
    elapsed_ms: float
    is_complete: bool = False
    is_proven_optimal: bool = False
    fallback_stage: str | None = None
    partial_plan: list[HookAction] = field(default_factory=list)
    partial_fallback_stage: str | None = None
    verification_report: Any | None = None
    partial_verification_report: Any | None = None
    debug_stats: dict[str, Any] | None = None
    telemetry: SolverTelemetry | None = None
    depot_earliness: int | None = None
    depot_hook_count: int | None = None


class PlanVerificationError(Exception):
    """Raised when verify=True and the produced plan fails verifier checks.

    Partial artifacts never raise this; callers inspect
    ``SolverResult.partial_verification_report``.
    """

    def __init__(self, report: Any) -> None:
        self.report = report
        errors = getattr(report, "errors", None) or []
        first = errors[0] if errors else "unknown verification failure"
        suffix = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        super().__init__(f"plan verification failed: {first}{suffix}")


def emit_telemetry(telemetry: SolverTelemetry) -> None:
    """Append a JSON-lines record to ``FZED_SOLVER_TELEMETRY_PATH`` if set.

    Silently no-ops when the env var is unset (production-safe default). Any
    I/O failure is swallowed to avoid breaking the solver's primary path —
    telemetry is best-effort.
    """
    path = os.environ.get("FZED_SOLVER_TELEMETRY_PATH")
    if not path:
        return
    try:
        record = asdict(telemetry)
        record["ts_unix"] = time.time()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Telemetry failures must never break solver output.
        pass
