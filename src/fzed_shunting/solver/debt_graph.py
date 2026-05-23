from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.debt_chain import DebtChainComponent, DebtChainSummary, summarize_debt_chains
from fzed_shunting.solver.structural_intent import StructuralIntent, build_structural_intent
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class DebtGraphComponent:
    anchor_track: str
    track_names: tuple[str, ...]
    total_pressure: float
    multi_track_pressure: float
    track_count: int
    delayed_commitment_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebtGraphView:
    intent: StructuralIntent
    chain_summary: DebtChainSummary
    components: tuple[DebtGraphComponent, ...]

    @property
    def max_multi_track_pressure(self) -> float:
        return max((component.multi_track_pressure for component in self.components), default=0.0)

    @property
    def multi_track_component_count(self) -> int:
        return sum(1 for component in self.components if component.track_count > 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "chain_summary": self.chain_summary.to_dict(),
            "components": [component.to_dict() for component in self.components],
        }


def build_debt_graph(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    route_oracle: RouteOracle | None = None,
    intent: StructuralIntent | None = None,
    chain_summary: DebtChainSummary | None = None,
) -> DebtGraphView:
    intent = intent or build_structural_intent(plan_input, state, route_oracle=route_oracle)
    chain_summary = chain_summary or summarize_debt_chains(plan_input, state, intent=intent)
    components = tuple(
        DebtGraphComponent(
            anchor_track=chain.anchor_track,
            track_names=chain.track_names,
            total_pressure=chain.total_pressure,
            multi_track_pressure=round(
                sum(track.cluster_pressure for track in chain.track_summaries if len(chain.track_names) > 1),
                3,
            ),
            track_count=len(chain.track_names),
            delayed_commitment_count=chain.delayed_commitment_count,
        )
        for chain in chain_summary.chains
    )
    return DebtGraphView(intent=intent, chain_summary=chain_summary, components=components)
