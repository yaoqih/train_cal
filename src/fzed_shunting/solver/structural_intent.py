from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import (
    allowed_spotting_south_ranks,
    preview_work_positions_after_prepend,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class CommittedBlock:
    track_name: str
    vehicle_nos: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderDebt:
    track_name: str
    defect_count: int
    pending_vehicle_nos: tuple[str, ...]
    blocking_prefix_vehicle_nos: tuple[str, ...]
    kind_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceDebt:
    kind: str
    track_name: str
    vehicle_nos: tuple[str, ...]
    blocked_vehicle_nos: tuple[str, ...] = ()
    source_tracks: tuple[str, ...] = ()
    target_tracks: tuple[str, ...] = ()
    pressure: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StagingBuffer:
    track_name: str
    free_length: float
    occupied_vehicle_count: int
    role_scores: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DelayedCommitment:
    vehicle_no: str
    target_track: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BufferLease:
    role: str
    vehicle_nos: tuple[str, ...]
    source_track: str
    target_track: str
    required_length: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralIntent:
    committed_blocks_by_track: dict[str, tuple[CommittedBlock, ...]]
    order_debts_by_track: dict[str, OrderDebt]
    resource_debts: tuple[ResourceDebt, ...]
    staging_buffers: tuple[StagingBuffer, ...]
    delayed_commitments: tuple[DelayedCommitment, ...]
    buffer_leases: tuple[BufferLease, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "committed_blocks_by_track": {
                track: [block.to_dict() for block in blocks]
                for track, blocks in sorted(self.committed_blocks_by_track.items())
            },
            "order_debts_by_track": {
                track: debt.to_dict()
                for track, debt in sorted(self.order_debts_by_track.items())
            },
            "resource_debts": [debt.to_dict() for debt in self.resource_debts],
            "staging_buffers": [buffer.to_dict() for buffer in self.staging_buffers],
            "delayed_commitments": [
                delayed.to_dict() for delayed in self.delayed_commitments
            ],
            "buffer_leases": [lease.to_dict() for lease in self.buffer_leases],
        }


def build_structural_intent(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    route_oracle: RouteOracle | None = None,
) -> StructuralIntent:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    track_by_vehicle = _vehicle_track_lookup(state)
    metrics = compute_structural_metrics(plan_input, state)
    route_plan = compute_route_blockage_plan(plan_input, state, route_oracle)
    capacity_plan = compute_capacity_release_plan(plan_input, state)
    delayed_commitments = _delayed_commitments(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
    )

    return StructuralIntent(
        committed_blocks_by_track=_committed_blocks_by_track(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
        ),
        order_debts_by_track=_order_debts_by_track(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            defect_by_track=metrics.target_sequence_defect_by_track,
        ),
        resource_debts=_resource_debts(
            plan_input=plan_input,
            route_plan=route_plan,
            capacity_plan=capacity_plan,
            front_blocker_by_track=metrics.front_blocker_by_track,
            state=state,
            vehicle_by_no=vehicle_by_no,
            work_position_unfinished_count=metrics.work_position_unfinished_count,
            target_sequence_defect_count=metrics.target_sequence_defect_count,
        ),
        staging_buffers=_staging_buffers(plan_input=plan_input, state=state),
        delayed_commitments=delayed_commitments,
        buffer_leases=_buffer_leases(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            track_by_vehicle=track_by_vehicle,
            delayed_commitments=delayed_commitments,
        ),
    )


def _committed_blocks_by_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> dict[str, tuple[CommittedBlock, ...]]:
    committed: dict[str, tuple[CommittedBlock, ...]] = {}
    for track_name, seq in state.track_sequences.items():
        blocks: list[CommittedBlock] = []
        current_block: list[str] = []
        for vehicle_no in seq:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is not None and goal_is_satisfied(
                vehicle,
                track_name=track_name,
                state=state,
                plan_input=plan_input,
            ):
                current_block.append(vehicle_no)
                continue
            if current_block:
                blocks.append(
                    CommittedBlock(
                        track_name=track_name,
                        vehicle_nos=tuple(current_block),
                        reason="goal_satisfied_contiguous_block",
                    )
                )
                current_block = []
        if current_block:
            blocks.append(
                CommittedBlock(
                    track_name=track_name,
                    vehicle_nos=tuple(current_block),
                    reason="goal_satisfied_contiguous_block",
                )
            )
        blocks.extend(
            _stable_work_position_window_blocks(
                plan_input=plan_input,
                state=state,
                track_name=track_name,
                seq=seq,
                vehicle_by_no=vehicle_by_no,
            )
        )
        if blocks:
            committed[track_name] = tuple(blocks)
    return dict(sorted(committed.items()))


def _stable_work_position_window_blocks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_name: str,
    seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[CommittedBlock]:
    track_spotting = [
        vehicle_no
        for vehicle_no in seq
        if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind == "SPOTTING"
        and track_name in vehicle.goal.allowed_target_tracks
    ]
    if not track_spotting:
        return []
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in track_spotting
    ):
        return []

    blocks: list[CommittedBlock] = []
    current_block: list[str] = []
    for vehicle_no in seq:
        if vehicle_no in track_spotting:
            current_block.append(vehicle_no)
            continue
        if current_block:
            blocks.append(
                CommittedBlock(
                    track_name=track_name,
                    vehicle_nos=tuple(current_block),
                    reason="stable_work_position_window",
                )
            )
            current_block = []
    if current_block:
        blocks.append(
            CommittedBlock(
                track_name=track_name,
                vehicle_nos=tuple(current_block),
                reason="stable_work_position_window",
            )
        )
    return blocks


def _order_debts_by_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    defect_by_track: dict[str, int],
) -> dict[str, OrderDebt]:
    debts: dict[str, OrderDebt] = {}
    work_position_target_tracks = _unfinished_work_position_target_tracks(
        plan_input=plan_input,
        state=state,
        track_by_vehicle=track_by_vehicle,
    )
    for target_track in sorted(set(defect_by_track) | work_position_target_tracks):
        pending = [
            vehicle
            for vehicle in plan_input.vehicles
            if vehicle.goal.work_position_kind is not None
            and target_track in vehicle.goal.allowed_target_tracks
            and not _vehicle_satisfied_on_current_track(
                vehicle=vehicle,
                state=state,
                plan_input=plan_input,
                track_by_vehicle=track_by_vehicle,
            )
        ]
        if not pending:
            continue
        defect_count = defect_by_track.get(target_track, 0)
        max_clear_count = 0
        for vehicle in pending:
            max_clear_count = max(
                max_clear_count,
                _insertion_clear_count(
                    vehicle=vehicle,
                    target_track=target_track,
                    state=state,
                    vehicle_by_no=vehicle_by_no,
                ),
            )
        kind_counts = Counter(vehicle.goal.work_position_kind or "UNKNOWN" for vehicle in pending)
        debts[target_track] = OrderDebt(
            track_name=target_track,
            defect_count=defect_count,
            pending_vehicle_nos=tuple(vehicle.vehicle_no for vehicle in pending),
            blocking_prefix_vehicle_nos=tuple(
                state.track_sequences.get(target_track, [])[:max_clear_count]
            ),
            kind_counts=tuple(sorted(kind_counts.items())),
        )
    return debts


def _unfinished_work_position_target_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_by_vehicle: dict[str, str],
) -> set[str]:
    target_tracks: set[str] = set()
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind is None:
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is not None and goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        target_tracks.update(vehicle.goal.allowed_target_tracks)
    return target_tracks


def _vehicle_satisfied_on_current_track(
    *,
    vehicle: NormalizedVehicle,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    track_by_vehicle: dict[str, str],
) -> bool:
    current_track = track_by_vehicle.get(vehicle.vehicle_no)
    return current_track is not None and goal_is_satisfied(
        vehicle,
        track_name=current_track,
        state=state,
        plan_input=plan_input,
    )


def _insertion_clear_count(
    *,
    vehicle: NormalizedVehicle,
    target_track: str,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    target_seq = list(state.track_sequences.get(target_track, []))
    kind = vehicle.goal.work_position_kind
    for clear_count in range(len(target_seq) + 1):
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=[vehicle.vehicle_no],
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        evaluation = preview.evaluations.get(vehicle.vehicle_no)
        if evaluation is None:
            continue
        if kind == "SPOTTING" and not evaluation.satisfied_now:
            continue
        return clear_count
    if kind == "SPOTTING" and not allowed_spotting_south_ranks(target_track):
        return 0
    return 0


def _resource_debts(
    *,
    plan_input: NormalizedPlanInput,
    route_plan: Any,
    capacity_plan: Any,
    front_blocker_by_track: dict[str, int],
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    work_position_unfinished_count: int,
    target_sequence_defect_count: int,
) -> tuple[ResourceDebt, ...]:
    debts: list[ResourceDebt] = []
    for track_name, fact in sorted(route_plan.facts_by_blocking_track.items()):
        debts.append(
            ResourceDebt(
                kind="ROUTE_RELEASE",
                track_name=track_name,
                vehicle_nos=tuple(fact.blocking_vehicle_nos),
                blocked_vehicle_nos=tuple(fact.blocked_vehicle_nos),
                source_tracks=tuple(fact.source_tracks),
                target_tracks=tuple(fact.target_tracks),
                pressure=float(fact.blockage_count),
            )
        )
    for track_name, fact in sorted(capacity_plan.facts_by_track.items()):
        if fact.release_pressure_length <= 1e-9:
            continue
        debts.append(
            ResourceDebt(
                kind="CAPACITY_RELEASE",
                track_name=track_name,
                vehicle_nos=tuple(fact.front_release_vehicle_nos),
                pressure=float(fact.release_pressure_length),
            )
        )
    for track_name, pressure in sorted(front_blocker_by_track.items()):
        prefix = _front_clearance_prefix(
            plan_input=plan_input,
            state=state,
            track_name=track_name,
            vehicle_by_no=vehicle_by_no,
            prefer_source_prefix_groups=(
                work_position_unfinished_count <= 0
                and target_sequence_defect_count <= 0
            ),
        )
        if not prefix:
            continue
        debts.append(
            ResourceDebt(
                kind="FRONT_CLEARANCE",
                track_name=track_name,
                vehicle_nos=tuple(prefix),
                pressure=float(pressure),
            )
        )
    debts.extend(_exact_spot_release_debts(plan_input=plan_input, state=state))
    return tuple(debts)


def _front_clearance_prefix(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_name: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    prefer_source_prefix_groups: bool,
) -> list[str]:
    seq = list(state.track_sequences.get(track_name, []))
    if not prefer_source_prefix_groups:
        return seq[:1]
    last_unfinished_index: int | None = None
    for index, vehicle_no in enumerate(seq):
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            break
        satisfied_here = goal_is_satisfied(
            vehicle,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        if not satisfied_here:
            last_unfinished_index = index
    if last_unfinished_index is None:
        return []
    return seq[: last_unfinished_index + 1]


def _exact_spot_release_debts(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> list[ResourceDebt]:
    spot_owner = {spot: vehicle_no for vehicle_no, spot in state.spot_assignments.items()}
    debts: list[ResourceDebt] = []
    for vehicle in plan_input.vehicles:
        goal = vehicle.goal
        if goal.target_mode != "SPOT" or goal.target_spot_code is None:
            continue
        if state.spot_assignments.get(vehicle.vehicle_no) == goal.target_spot_code:
            continue
        occupant = spot_owner.get(goal.target_spot_code)
        if occupant is None or occupant == vehicle.vehicle_no:
            continue
        debts.append(
            ResourceDebt(
                kind="EXACT_SPOT_RELEASE",
                track_name=goal.target_track,
                vehicle_nos=(occupant,),
                blocked_vehicle_nos=(vehicle.vehicle_no,),
                pressure=1.0,
            )
        )
    return debts


def _staging_buffers(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> tuple[StagingBuffer, ...]:
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    buffers: list[StagingBuffer] = []
    for track_name in sorted(STAGING_TRACKS & set(capacity_by_track)):
        seq = list(state.track_sequences.get(track_name, []))
        occupied_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in seq)
        free_length = max(0.0, capacity_by_track[track_name] - occupied_length)
        empty_bonus = 2 if not seq else 0
        buffers.append(
            StagingBuffer(
                track_name=track_name,
                free_length=round(free_length, 3),
                occupied_vehicle_count=len(seq),
                role_scores={
                    "ORDER_BUFFER": max(0, 3 + empty_bonus),
                    "ROUTE_RELEASE": max(0, 2 + empty_bonus),
                    "CAPACITY_RELEASE": max(0, 2 + empty_bonus),
                    "SOURCE_REMAINDER": max(0, 1 + empty_bonus),
                },
            )
        )
    return tuple(buffers)


def _delayed_commitments(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
) -> tuple[DelayedCommitment, ...]:
    delayed: list[DelayedCommitment] = []
    tracks_with_pending_spotting = _tracks_with_pending_spotting(
        plan_input=plan_input,
        state=state,
        track_by_vehicle=track_by_vehicle,
    )
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind is None:
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is not None and goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        for target_track in vehicle.goal.allowed_target_tracks:
            preview = preview_work_positions_after_prepend(
                target_track=target_track,
                incoming_vehicle_nos=[vehicle.vehicle_no],
                existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
                vehicle_by_no=vehicle_by_no,
            )
            evaluation = preview.evaluations.get(vehicle.vehicle_no)
            if not preview.valid or evaluation is None or not evaluation.satisfied_now:
                delayed.append(
                    DelayedCommitment(
                        vehicle_no=vehicle.vehicle_no,
                        target_track=target_track,
                        reason="work_position_window_not_ready",
                    )
                )
                continue
            if (
                target_track in tracks_with_pending_spotting
                and vehicle.goal.work_position_kind != "SPOTTING"
            ):
                delayed.append(
                    DelayedCommitment(
                        vehicle_no=vehicle.vehicle_no,
                        target_track=target_track,
                        reason="would_precede_unfinished_work_position_window",
                    )
                )
    return tuple(delayed)


def _tracks_with_pending_spotting(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_by_vehicle: dict[str, str],
) -> set[str]:
    tracks: set[str] = set()
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind != "SPOTTING":
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track is not None and goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            continue
        tracks.update(vehicle.goal.allowed_target_tracks)
    return tracks


def _buffer_leases(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    delayed_commitments: tuple[DelayedCommitment, ...],
) -> tuple[BufferLease, ...]:
    delayed_by_vehicle = {delayed.vehicle_no: delayed for delayed in delayed_commitments}
    if not delayed_by_vehicle:
        return ()

    leases: list[BufferLease] = []
    for source_track, seq in sorted(state.track_sequences.items()):
        current_block: list[str] = []
        current_target: str | None = None
        current_reason: str | None = None
        for vehicle_no in seq:
            delayed = delayed_by_vehicle.get(vehicle_no)
            if delayed is None or track_by_vehicle.get(vehicle_no) != source_track:
                if current_block:
                    leases.append(
                        _make_buffer_lease(
                            plan_input=plan_input,
                            vehicle_by_no=vehicle_by_no,
                            source_track=source_track,
                            vehicle_nos=current_block,
                            target_track=current_target or "",
                            reason=current_reason or "",
                        )
                    )
                    current_block = []
                    current_target = None
                    current_reason = None
                continue
            if current_block and (
                delayed.target_track != current_target
                or delayed.reason != current_reason
            ):
                leases.append(
                    _make_buffer_lease(
                        plan_input=plan_input,
                        vehicle_by_no=vehicle_by_no,
                        source_track=source_track,
                        vehicle_nos=current_block,
                        target_track=current_target or "",
                        reason=current_reason or "",
                    )
                )
                current_block = []
            current_block.append(vehicle_no)
            current_target = delayed.target_track
            current_reason = delayed.reason
        if current_block:
            leases.append(
                _make_buffer_lease(
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    vehicle_nos=current_block,
                    target_track=current_target or "",
                    reason=current_reason or "",
                )
            )
    return tuple(lease for lease in leases if lease.vehicle_nos)


def _make_buffer_lease(
    *,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    vehicle_nos: list[str],
    target_track: str,
    reason: str,
) -> BufferLease:
    required_length = sum(
        vehicle_by_no[vehicle_no].vehicle_length
        for vehicle_no in vehicle_nos
        if vehicle_no in vehicle_by_no
    )
    return BufferLease(
        role="ORDER_BUFFER",
        vehicle_nos=tuple(vehicle_nos),
        source_track=source_track,
        target_track=target_track,
        required_length=round(required_length, 3),
        reason=reason,
    )
