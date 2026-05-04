from __future__ import annotations

from dataclasses import asdict, dataclass

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.state import _state_key, _vehicle_track_lookup
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


ROUTE_RELEASE_CONTINUATION_BONUS = 8
ROUTE_RELEASE_FOCUS_TTL = 3


@dataclass(frozen=True)
class RouteBlockageFact:
    blocking_track: str
    blocking_vehicle_nos: list[str]
    blocked_vehicle_nos: list[str]
    source_tracks: list[str]
    target_tracks: list[str]
    blockage_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RouteBlockagePlan:
    facts_by_blocking_track: dict[str, RouteBlockageFact]

    @property
    def total_blockage_pressure(self) -> int:
        return sum(fact.blockage_count for fact in self.facts_by_blocking_track.values())

    @property
    def blocked_vehicle_nos(self) -> list[str]:
        vehicle_nos: set[str] = set()
        for fact in self.facts_by_blocking_track.values():
            vehicle_nos.update(fact.blocked_vehicle_nos)
        return sorted(vehicle_nos)

    @property
    def blocking_vehicle_nos(self) -> list[str]:
        vehicle_nos: set[str] = set()
        for fact in self.facts_by_blocking_track.values():
            vehicle_nos.update(fact.blocking_vehicle_nos)
        return sorted(vehicle_nos)

    def to_dict(self) -> dict:
        return {
            "total_blockage_pressure": self.total_blockage_pressure,
            "blocked_vehicle_nos": self.blocked_vehicle_nos,
            "blocking_vehicle_nos": self.blocking_vehicle_nos,
            "facts_by_blocking_track": {
                track: fact.to_dict()
                for track, fact in sorted(self.facts_by_blocking_track.items())
            },
        }


def compute_route_blockage_plan(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle | None,
    *,
    blocked_source_tracks: set[str] | frozenset[str] | None = None,
) -> RouteBlockagePlan:
    if route_oracle is None:
        return RouteBlockagePlan(facts_by_blocking_track={})
    cache_key = _route_blockage_cache_key(
        plan_input=plan_input,
        state=state,
        blocked_source_tracks=blocked_source_tracks,
    )
    cached = route_oracle._route_blockage_plan_cache.get(cache_key)
    if cached is not None:
        return cached

    track_by_vehicle = _vehicle_track_lookup(state)
    fact_builders: dict[str, _RouteBlockageFactBuilder] = {}
    for vehicle in plan_input.vehicles:
        source_track = track_by_vehicle.get(vehicle.vehicle_no)
        if source_track is None and vehicle.vehicle_no in state.loco_carry:
            source_track = state.loco_track_name
        if source_track is None:
            continue
        if blocked_source_tracks is not None and source_track not in blocked_source_tracks:
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        target_tracks = goal_effective_allowed_tracks(
            vehicle,
            state=state,
            plan_input=plan_input,
        )
        if source_track != state.loco_track_name:
            access_result = route_oracle.validate_loco_access(
                loco_track=state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=state.track_sequences,
                loco_node=state.loco_node,
            )
            if not access_result.is_valid:
                target_label = next(
                    (track for track in target_tracks if track != source_track),
                    source_track,
                )
                for blocking_track in access_result.blocking_tracks:
                    if blocking_track == source_track:
                        continue
                    blocking_vehicle_nos = list(state.track_sequences.get(blocking_track, []))
                    if not blocking_vehicle_nos:
                        continue
                    builder = fact_builders.setdefault(
                        blocking_track,
                        _RouteBlockageFactBuilder(blocking_track=blocking_track),
                    )
                    builder.add(
                        blocking_vehicle_nos=blocking_vehicle_nos,
                        blocked_vehicle_no=vehicle.vehicle_no,
                        source_track=source_track,
                        target_track=target_label,
                    )
        for target_track in target_tracks:
            if target_track == source_track:
                continue
            path_tracks = route_oracle.resolve_path_tracks_for_endpoint_constraints(
                source_track,
                target_track,
                occupied_track_sequences=state.track_sequences,
                source_node=state.loco_node if source_track == state.loco_track_name else None,
                target_node=route_oracle.order_end_node(target_track),
            )
            if path_tracks is None:
                path_tracks = route_oracle.resolve_path_tracks(source_track, target_track)
            if path_tracks is None:
                continue
            for blocking_track in path_tracks[1:-1]:
                blocking_vehicle_nos = list(state.track_sequences.get(blocking_track, []))
                if not blocking_vehicle_nos:
                    continue
                builder = fact_builders.setdefault(
                    blocking_track,
                    _RouteBlockageFactBuilder(blocking_track=blocking_track),
                )
                builder.add(
                    blocking_vehicle_nos=blocking_vehicle_nos,
                    blocked_vehicle_no=vehicle.vehicle_no,
                    source_track=source_track,
                    target_track=target_track,
                )

    plan = RouteBlockagePlan(
        facts_by_blocking_track={
            track: builder.build()
            for track, builder in sorted(fact_builders.items())
        }
    )
    route_oracle._route_blockage_plan_cache[cache_key] = plan
    return plan


def _route_blockage_cache_key(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    blocked_source_tracks: set[str] | frozenset[str] | None,
) -> tuple:
    return (
        id(plan_input),
        _state_key(state, plan_input),
        tuple(sorted(blocked_source_tracks or ())),
    )


def route_blockage_release_score(
    *,
    source_track: str,
    vehicle_nos: list[str] | tuple[str, ...],
    route_blockage_plan: RouteBlockagePlan | None,
) -> int:
    if route_blockage_plan is None:
        return 0
    fact = route_blockage_plan.facts_by_blocking_track.get(source_track)
    if fact is None:
        return 0
    source_blockers = set(fact.blocking_vehicle_nos)
    if not source_blockers:
        return 0
    moved = set(vehicle_nos)
    if not moved:
        return 0
    source_release_fraction = len(moved & source_blockers) / len(source_blockers)
    if source_release_fraction <= 0.0:
        return 0
    return max(1, round(fact.blockage_count * source_release_fraction))


def route_release_continuation_bonus(
    *,
    state: ReplayState,
    move: HookAction,
    focus_tracks: frozenset[str] | set[str] | None,
    focus_bonus: int,
) -> int:
    if not focus_tracks or move.action_type != "ATTACH" or state.loco_carry:
        return 0
    if move.source_track not in focus_tracks or not move.vehicle_nos:
        return 0
    source_seq = state.track_sequences.get(move.source_track, [])
    if tuple(source_seq[: len(move.vehicle_nos)]) != tuple(move.vehicle_nos):
        return 0
    return max(ROUTE_RELEASE_CONTINUATION_BONUS, focus_bonus + 2)


def route_release_focus_after_move(
    *,
    prior_focus_tracks: frozenset[str] | set[str],
    prior_focus_bonus: int,
    prior_focus_ttl: int,
    move: HookAction,
    route_blockage_plan: RouteBlockagePlan | None,
) -> tuple[frozenset[str], int, int]:
    if route_blockage_plan is not None and move.action_type == "ATTACH":
        fact = route_blockage_plan.facts_by_blocking_track.get(move.source_track)
        if fact is not None:
            moved = set(move.vehicle_nos)
            if moved and moved & set(fact.blocking_vehicle_nos):
                return (
                    frozenset(fact.source_tracks),
                    fact.blockage_count,
                    max(ROUTE_RELEASE_FOCUS_TTL, len(fact.blocking_vehicle_nos) + 1),
                )
    if prior_focus_ttl <= 0 or not prior_focus_tracks:
        return frozenset(), 0, 0
    if move.action_type == "ATTACH" and move.source_track in prior_focus_tracks:
        return frozenset(), 0, 0
    ttl = prior_focus_ttl - 1
    if ttl <= 0:
        return frozenset(), 0, 0
    return frozenset(prior_focus_tracks), prior_focus_bonus, ttl


@dataclass
class _RouteBlockageFactBuilder:
    blocking_track: str
    blocking_vehicle_nos: set[str] | None = None
    blocked_vehicle_nos: set[str] | None = None
    source_tracks: set[str] | None = None
    target_tracks: set[str] | None = None
    blockage_count: int = 0

    def add(
        self,
        *,
        blocking_vehicle_nos: list[str],
        blocked_vehicle_no: str,
        source_track: str,
        target_track: str,
    ) -> None:
        if self.blocking_vehicle_nos is None:
            self.blocking_vehicle_nos = set()
        if self.blocked_vehicle_nos is None:
            self.blocked_vehicle_nos = set()
        if self.source_tracks is None:
            self.source_tracks = set()
        if self.target_tracks is None:
            self.target_tracks = set()
        self.blocking_vehicle_nos.update(blocking_vehicle_nos)
        self.blocked_vehicle_nos.add(blocked_vehicle_no)
        self.source_tracks.add(source_track)
        self.target_tracks.add(target_track)
        self.blockage_count += 1

    def build(self) -> RouteBlockageFact:
        return RouteBlockageFact(
            blocking_track=self.blocking_track,
            blocking_vehicle_nos=sorted(self.blocking_vehicle_nos or set()),
            blocked_vehicle_nos=sorted(self.blocked_vehicle_nos or set()),
            source_tracks=sorted(self.source_tracks or set()),
            target_tracks=sorted(self.target_tracks or set()),
            blockage_count=self.blockage_count,
        )
