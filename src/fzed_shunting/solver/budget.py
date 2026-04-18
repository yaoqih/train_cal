from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class SearchBudget:
    time_budget_ms: float | None = None
    node_budget: int | None = None
    started_at: float = field(default_factory=perf_counter)
    nodes_expanded: int = 0

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000

    def remaining_ms(self) -> float | None:
        if self.time_budget_ms is None:
            return None
        return max(0.0, self.time_budget_ms - self.elapsed_ms())

    def exhausted(self) -> bool:
        if self.time_budget_ms is not None and self.elapsed_ms() >= self.time_budget_ms:
            return True
        if self.node_budget is not None and self.nodes_expanded >= self.node_budget:
            return True
        return False

    def tick_expand(self) -> None:
        self.nodes_expanded += 1

    def reset(self) -> None:
        self.started_at = perf_counter()
        self.nodes_expanded = 0

    def child_budget(self, time_ms: float | None = None) -> "SearchBudget":
        remaining_time = self.remaining_ms() if self.time_budget_ms is not None else None
        if time_ms is not None and remaining_time is not None:
            remaining_time = min(remaining_time, time_ms)
        elif time_ms is not None and remaining_time is None:
            remaining_time = time_ms
        remaining_nodes: int | None = None
        if self.node_budget is not None:
            remaining_nodes = max(0, self.node_budget - self.nodes_expanded)
        return SearchBudget(
            time_budget_ms=remaining_time,
            node_budget=remaining_nodes,
        )


class SearchBudgetExhausted(Exception):
    pass
