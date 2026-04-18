"""Solver result data structures and the verification exception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.solver.types import HookAction


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a solver invocation.

    Attributes:
        plan: ordered list of hooks. Empty when no plan was produced.
        expanded_nodes / generated_nodes / closed_nodes: search telemetry.
        elapsed_ms: wall-clock time spent producing this result.
        is_proven_optimal: True only when exact search ran to completion.
        fallback_stage: which anytime stage produced the final plan
            (``"exact" | "weighted" | "beam" | "weighted_greedy" |
            "beam_greedy_128" | "beam_greedy_256" | "weighted_very_greedy" |
            "constructive" | "constructive_partial"``).
        verification_report: populated after _attach_verification.
        debug_stats: optional telemetry bag.
    """

    plan: list[HookAction]
    expanded_nodes: int
    generated_nodes: int
    closed_nodes: int
    elapsed_ms: float
    is_proven_optimal: bool = False
    fallback_stage: str | None = None
    verification_report: Any | None = None
    debug_stats: dict[str, Any] | None = None


class PlanVerificationError(Exception):
    """Raised when verify=True and the produced plan fails verifier checks.

    Partial plans flagged as best-effort (``fallback_stage="constructive_partial"``)
    never raise this; callers inspect ``SolverResult.verification_report.is_valid``.
    """

    def __init__(self, report: Any) -> None:
        self.report = report
        errors = getattr(report, "errors", None) or []
        first = errors[0] if errors else "unknown verification failure"
        suffix = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        super().__init__(f"plan verification failed: {first}{suffix}")
