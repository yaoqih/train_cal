"""Anytime fallback chain and budget helpers.

The fallback chain widens heuristic weights, swaps in beam search, and
increases beam width across sequential stages until one returns a non-empty
plan. Each stage gets a fraction of the remaining wall-clock budget so the
total chain respects ``time_budget_ms``.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.budget import SearchBudget
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.verify.replay import ReplayState


def _run_anytime_fallback_chain(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None,
    incumbent: SolverResult,
    started_at: float,
    time_budget_ms: float | None,
    node_budget: int | None,
    heuristic_weight: float,
    beam_width: int | None,
    debug_stats: dict[str, Any] | None,
    solve_search_result,
) -> SolverResult:
    fallback_stages: list[tuple[str, dict[str, Any], float]] = [
        (
            "weighted",
            {
                "solver_mode": "weighted",
                "heuristic_weight": max(heuristic_weight, 1.5),
                "beam_width": None,
            },
            0.08,
        ),
        (
            "beam",
            {
                "solver_mode": "beam",
                "heuristic_weight": max(heuristic_weight, 1.5),
                "beam_width": beam_width or 64,
            },
            0.08,
        ),
        (
            "beam_greedy_64",
            {
                "solver_mode": "beam",
                "heuristic_weight": 5.0,
                "beam_width": beam_width or 64,
            },
            0.08,
        ),
        (
            "weighted_greedy",
            {
                "solver_mode": "weighted",
                "heuristic_weight": 5.0,
                "beam_width": None,
            },
            0.12,
        ),
        (
            "beam_greedy_128",
            {
                "solver_mode": "beam",
                "heuristic_weight": 5.0,
                "beam_width": max(beam_width or 0, 128),
            },
            0.08,
        ),
        (
            "beam_greedy_256",
            {
                "solver_mode": "beam",
                "heuristic_weight": 5.0,
                "beam_width": max(beam_width or 0, 256),
            },
            0.08,
        ),
        (
            "weighted_very_greedy",
            {
                "solver_mode": "weighted",
                "heuristic_weight": 10.0,
                "beam_width": None,
            },
            0.08,
        ),
    ]
    current = incumbent
    for stage_name, stage_kwargs, budget_share in fallback_stages:
        if current.plan:
            break
        remaining_ms = _remaining_budget_ms(started_at, time_budget_ms)
        stage_time_budget_ms: float | None
        if time_budget_ms is None:
            stage_time_budget_ms = None
        else:
            share_ms = max(1.0, time_budget_ms * budget_share)
            stage_time_budget_ms = share_ms if remaining_ms is None else min(remaining_ms, share_ms)
            if stage_time_budget_ms <= 0:
                break
        remaining_nodes = _remaining_budget_nodes(node_budget, current.expanded_nodes)
        if remaining_nodes is not None and remaining_nodes <= 0:
            break
        try:
            candidate = solve_search_result(
                plan_input=plan_input,
                initial_state=initial_state,
                master=master,
                debug_stats=debug_stats,
                budget=SearchBudget(
                    time_budget_ms=stage_time_budget_ms,
                    node_budget=remaining_nodes,
                ),
                **stage_kwargs,
            )
        except ValueError:
            continue
        if not candidate.plan:
            continue
        if not current.plan or len(candidate.plan) < len(current.plan):
            current = replace(candidate, fallback_stage=stage_name)
    return current


def _remaining_budget_ms(started_at: float, time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return None
    elapsed = (perf_counter() - started_at) * 1000
    return max(0.0, time_budget_ms - elapsed)


def _remaining_budget_nodes(node_budget: int | None, expanded: int) -> int | None:
    if node_budget is None:
        return None
    return max(0, node_budget - expanded)
