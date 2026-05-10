from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.solver.structural_intent import StructuralIntent, build_structural_intent
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class DebtChainTrackSummary:
    track_name: str
    cluster_pressure: float
    debt_kinds: tuple[str, ...]
    pending_vehicle_count: int
    blocking_prefix_count: int
    delayed_commitment_count: int
    capacity_release_length: float
    route_blocked_vehicle_count: int
    route_blocking_vehicle_count: int
    source_tracks: tuple[str, ...]
    target_tracks: tuple[str, ...]
    buffer_roles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebtChainComponent:
    anchor_track: str
    track_names: tuple[str, ...]
    total_pressure: float
    order_debt_track_count: int
    route_blockage_track_count: int
    capacity_release_track_count: int
    delayed_commitment_count: int
    track_summaries: tuple[DebtChainTrackSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_track": self.anchor_track,
            "track_names": list(self.track_names),
            "total_pressure": self.total_pressure,
            "order_debt_track_count": self.order_debt_track_count,
            "route_blockage_track_count": self.route_blockage_track_count,
            "capacity_release_track_count": self.capacity_release_track_count,
            "delayed_commitment_count": self.delayed_commitment_count,
            "track_summaries": [track.to_dict() for track in self.track_summaries],
        }


@dataclass(frozen=True)
class DebtChainSummary:
    chain_count: int
    total_tracks: int
    max_chain_pressure: float
    chains: tuple[DebtChainComponent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_count": self.chain_count,
            "total_tracks": self.total_tracks,
            "max_chain_pressure": self.max_chain_pressure,
            "chains": [chain.to_dict() for chain in self.chains],
        }


def summarize_debt_chains(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    intent: StructuralIntent | None = None,
) -> DebtChainSummary:
    intent = intent or build_structural_intent(plan_input, state)
    track_by_vehicle = _vehicle_track_lookup(state)

    graph: dict[str, set[str]] = defaultdict(set)
    delayed_by_track: Counter[str] = Counter()
    all_tracks: set[str] = set(intent.debt_clusters_by_track)

    for cluster in intent.debt_clusters_by_track.values():
        graph.setdefault(cluster.track_name, set())
        order_debt = cluster.order_debt
        if order_debt is not None:
            for vehicle_no in order_debt.pending_vehicle_nos:
                current_track = track_by_vehicle.get(vehicle_no)
                if current_track is not None and current_track != cluster.track_name:
                    _link_tracks(graph, cluster.track_name, current_track)
                    all_tracks.add(current_track)
        for debt in cluster.resource_debts:
            for source_track in debt.source_tracks:
                _link_tracks(graph, cluster.track_name, source_track)
                all_tracks.add(source_track)
            for target_track in debt.target_tracks:
                _link_tracks(graph, cluster.track_name, target_track)
                all_tracks.add(target_track)

    for delayed in intent.delayed_commitments:
        delayed_by_track[delayed.target_track] += 1
        all_tracks.add(delayed.target_track)
        current_track = track_by_vehicle.get(delayed.vehicle_no)
        if current_track is not None and current_track != delayed.target_track:
            _link_tracks(graph, current_track, delayed.target_track)
            all_tracks.add(current_track)

    return _build_chain_summary(
        intent=intent,
        graph=graph,
        all_tracks=all_tracks,
        delayed_by_track=delayed_by_track,
        route_facts={},
        capacity_facts={},
    )


def analyze_debt_chains(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    route_oracle: RouteOracle | None = None,
    intent: StructuralIntent | None = None,
) -> DebtChainSummary:
    intent = intent or build_structural_intent(
        plan_input,
        state,
        route_oracle=route_oracle,
    )
    route_plan = compute_route_blockage_plan(plan_input, state, route_oracle)
    capacity_plan = compute_capacity_release_plan(plan_input, state)
    track_by_vehicle = _vehicle_track_lookup(state)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}

    graph: dict[str, set[str]] = defaultdict(set)
    delayed_by_track: Counter[str] = Counter()
    all_tracks: set[str] = set(intent.debt_clusters_by_track)

    for cluster in intent.debt_clusters_by_track.values():
        graph.setdefault(cluster.track_name, set())
        order_debt = cluster.order_debt
        if order_debt is not None:
            for vehicle_no in order_debt.pending_vehicle_nos:
                current_track = track_by_vehicle.get(vehicle_no)
                if current_track is not None and current_track != cluster.track_name:
                    _link_tracks(graph, cluster.track_name, current_track)
                    all_tracks.add(current_track)
        for debt in cluster.resource_debts:
            for source_track in debt.source_tracks:
                _link_tracks(graph, cluster.track_name, source_track)
                all_tracks.add(source_track)
            for target_track in debt.target_tracks:
                _link_tracks(graph, cluster.track_name, target_track)
                all_tracks.add(target_track)

    for delayed in intent.delayed_commitments:
        delayed_by_track[delayed.target_track] += 1
        all_tracks.add(delayed.target_track)
        current_track = track_by_vehicle.get(delayed.vehicle_no)
        if current_track is not None and current_track != delayed.target_track:
            _link_tracks(graph, current_track, delayed.target_track)
            all_tracks.add(current_track)

    route_facts = route_plan.to_dict().get("facts_by_blocking_track", {})
    for blocking_track, fact in route_facts.items():
        all_tracks.add(blocking_track)
        graph.setdefault(blocking_track, set())
        for source_track in fact.get("source_tracks", []):
            _link_tracks(graph, blocking_track, source_track)
            all_tracks.add(source_track)
        for target_track in fact.get("target_tracks", []):
            _link_tracks(graph, blocking_track, target_track)
            all_tracks.add(target_track)

    capacity_facts = capacity_plan.to_dict().get("facts_by_track", {})
    for track_name, fact in capacity_facts.items():
        if fact.get("release_pressure_length", 0) > 0 or fact.get("front_release_length", 0) > 0:
            all_tracks.add(track_name)
            graph.setdefault(track_name, set())

    return _build_chain_summary(
        intent=intent,
        graph=graph,
        all_tracks=all_tracks,
        delayed_by_track=delayed_by_track,
        route_facts=route_facts,
        capacity_facts=capacity_facts,
    )


def _link_tracks(graph: dict[str, set[str]], left: str, right: str) -> None:
    if left == right:
        graph.setdefault(left, set())
        return
    graph[left].add(right)
    graph[right].add(left)


def _build_chain_summary(
    *,
    intent: StructuralIntent,
    graph: dict[str, set[str]],
    all_tracks: set[str],
    delayed_by_track: Counter[str],
    route_facts: dict[str, Any],
    capacity_facts: dict[str, Any],
) -> DebtChainSummary:
    components: list[DebtChainComponent] = []
    visited: set[str] = set()
    for start in sorted(all_tracks):
        if start in visited:
            continue
        stack = [start]
        component_tracks: list[str] = []
        while stack:
            track = stack.pop()
            if track in visited:
                continue
            visited.add(track)
            component_tracks.append(track)
            stack.extend(sorted(graph.get(track, ())))
        track_summaries = tuple(
            sorted(
                (
                    _build_track_summary(
                        track_name=track,
                        intent=intent,
                        delayed_commitment_count=delayed_by_track.get(track, 0),
                        capacity_facts=capacity_facts.get(track, {}),
                        route_fact=route_facts.get(track, {}),
                    )
                    for track in component_tracks
                ),
                key=lambda item: (-item.cluster_pressure, item.track_name),
            )
        )
        total_pressure = round(sum(item.cluster_pressure for item in track_summaries), 3)
        components.append(
            DebtChainComponent(
                anchor_track=track_summaries[0].track_name,
                track_names=tuple(sorted(component_tracks)),
                total_pressure=total_pressure,
                order_debt_track_count=sum(
                    1
                    for track in component_tracks
                    if track in intent.order_debts_by_track
                ),
                route_blockage_track_count=sum(
                    1 for track in component_tracks if track in route_facts
                ),
                capacity_release_track_count=sum(
                    1
                    for track in component_tracks
                    if (capacity_facts.get(track, {}) or {}).get("release_pressure_length", 0) > 0
                    or (capacity_facts.get(track, {}) or {}).get("front_release_length", 0) > 0
                ),
                delayed_commitment_count=sum(delayed_by_track.get(track, 0) for track in component_tracks),
                track_summaries=track_summaries,
            )
        )

    components = sorted(
        components,
        key=lambda item: (-item.total_pressure, -len(item.track_names), item.anchor_track),
    )
    return DebtChainSummary(
        chain_count=len(components),
        total_tracks=sum(len(component.track_names) for component in components),
        max_chain_pressure=max((component.total_pressure for component in components), default=0.0),
        chains=tuple(components),
    )


def _build_track_summary(
    *,
    track_name: str,
    intent: StructuralIntent,
    delayed_commitment_count: int,
    capacity_facts: dict[str, Any],
    route_fact: dict[str, Any],
) -> DebtChainTrackSummary:
    cluster = intent.debt_clusters_by_track.get(track_name)
    order_debt = cluster.order_debt if cluster is not None else None
    resource_debts = cluster.resource_debts if cluster is not None else ()
    return DebtChainTrackSummary(
        track_name=track_name,
        cluster_pressure=round(cluster.pressure if cluster is not None else 0.0, 3),
        debt_kinds=tuple(
            sorted(
                {
                    *(["ORDER"] if order_debt is not None else []),
                    *(debt.kind for debt in resource_debts),
                    *(["ROUTE_BLOCKAGE"] if route_fact else []),
                    *(
                        ["CAPACITY_RELEASE"]
                        if (capacity_facts.get("release_pressure_length", 0) > 0 or capacity_facts.get("front_release_length", 0) > 0)
                        else []
                    ),
                }
            )
        ),
        pending_vehicle_count=len(order_debt.pending_vehicle_nos) if order_debt is not None else 0,
        blocking_prefix_count=len(order_debt.blocking_prefix_vehicle_nos) if order_debt is not None else 0,
        delayed_commitment_count=delayed_commitment_count,
        capacity_release_length=round(
            float(capacity_facts.get("release_pressure_length", 0.0)),
            3,
        ),
        route_blocked_vehicle_count=len(route_fact.get("blocked_vehicle_nos", ())),
        route_blocking_vehicle_count=len(route_fact.get("blocking_vehicle_nos", ())),
        source_tracks=tuple(sorted({*(route_fact.get("source_tracks", ())), *(source for debt in resource_debts for source in debt.source_tracks)})),
        target_tracks=tuple(sorted({*(route_fact.get("target_tracks", ())), *(target for debt in resource_debts for target in debt.target_tracks)})),
        buffer_roles=cluster.buffer_roles if cluster is not None else (),
    )
