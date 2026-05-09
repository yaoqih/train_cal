from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import preview_work_positions_after_prepend
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.candidate_compiler import replay_candidate_steps
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import generate_real_hook_moves
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.state import _vehicle_track_lookup
from fzed_shunting.solver.structural_intent import (
    BufferLease,
    StructuralIntent,
    build_debt_clusters_by_track,
    build_structural_intent,
)
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


STRICT_WORK_POSITION_KINDS = frozenset(
    {"SPOTTING", "EXACT_NORTH_RANK", "EXACT_WORK_SLOT"}
)

STRUCTURAL_PRIORITY_REASON_GROUPS = {
    "route_release_frontier": 0,
    "resource_release": 1,
    "capacity_release": 1,
}


@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]
    kind: str = "primitive"
    reason: str = ""
    focus_tracks: tuple[str, ...] = ()
    structural_reserve: bool = False


def generate_move_candidates(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    master: MasterData | None = None,
    route_oracle: RouteOracle | None = None,
    debug_stats: dict[str, Any] | None = None,
) -> list[MoveCandidate]:
    primitive_debug: dict[str, Any] | None = {} if debug_stats is not None else None
    primitive_moves = generate_real_hook_moves(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
        debug_stats=primitive_debug,
    )
    intent = build_structural_intent(
        plan_input,
        state,
        route_oracle=route_oracle,
    )
    protected_rejected_count = 0
    protected_allowed_by_debt_count = 0
    mixed_resource_primitive_rejected_count = 0
    delayed_staging_churn_rejected_count = 0
    filtered_primitive_moves: list[HookAction] = []
    for move in primitive_moves:
        if _move_mixes_resource_debt_groups(
            move,
            intent=intent,
            state=state,
            plan_input=plan_input,
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles},
        ):
            mixed_resource_primitive_rejected_count += 1
            continue
        if _move_is_unowned_staging_churn(
            move,
            intent=intent,
            state=state,
            plan_input=plan_input,
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles},
        ):
            delayed_staging_churn_rejected_count += 1
            continue
        if _move_attaches_unowned_delayed_staging_buffer(move, intent=intent):
            delayed_staging_churn_rejected_count += 1
            continue
        if _move_breaks_protected_commitment(move, intent=intent):
            if _move_is_required_by_resource_debt(move, intent=intent):
                protected_allowed_by_debt_count += 1
                filtered_primitive_moves.append(move)
            else:
                protected_rejected_count += 1
            continue
        filtered_primitive_moves.append(move)
    candidates = tuple(
        MoveCandidate(steps=(move,), kind="primitive")
        for move in filtered_primitive_moves
    )
    structural_candidates = _generate_structural_candidates(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        intent=intent,
    )
    candidate_pool = (*structural_candidates, *candidates)
    all_candidates = _dedup_candidates(
        tuple(
            trimmed
            for candidate in candidate_pool
            if (
                trimmed := _trim_candidate_before_delayed_commitment(
                    candidate,
                    vehicle_by_no={
                        vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles
                    },
                    intent=intent,
                )
            )
            is not None
        )
    )
    if debug_stats is not None:
        debug_stats.clear()
        debug_stats.update(primitive_debug or {})
        debug_stats["primitive_move_count"] = len(primitive_moves)
        debug_stats["protected_primitive_rejected_count"] = protected_rejected_count
        debug_stats["protected_primitive_allowed_by_debt_count"] = protected_allowed_by_debt_count
        debug_stats["mixed_resource_primitive_rejected_count"] = (
            mixed_resource_primitive_rejected_count
        )
        debug_stats["delayed_staging_churn_rejected_count"] = (
            delayed_staging_churn_rejected_count
        )
        debug_stats["structural_intent_candidate_count"] = len(structural_candidates)
        debug_stats["structural_debt_cluster_count"] = len(intent.debt_clusters_by_track)
        debug_stats["structural_candidate_steps_total"] = sum(
            len(candidate.steps) for candidate in structural_candidates
        )
        debug_stats["total_moves"] = len(all_candidates)
        debug_stats["total_candidates"] = len(all_candidates)
        debug_stats["candidate_steps_total"] = sum(
            len(candidate.steps) for candidate in all_candidates
        )
        debug_stats["candidate_steps_by_kind"] = _candidate_steps_by_kind(all_candidates)
    return list(all_candidates)


def _trim_candidate_before_delayed_commitment(
    candidate: MoveCandidate,
    *,
    vehicle_by_no: dict[str, NormalizedVehicle],
    intent: StructuralIntent,
) -> MoveCandidate | None:
    if not intent.delayed_commitments:
        return candidate
    delayed_targets = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }
    kept_steps: list[HookAction] = []
    pending_attach: HookAction | None = None
    for step in candidate.steps:
        if step.action_type == "ATTACH":
            if (
                candidate.kind == "structural"
                and not candidate.reason.startswith("work_position_")
                and step.source_track in STAGING_TRACKS
                and any(
                    (vehicle_no, delayed.target_track) in delayed_targets
                    for vehicle_no in step.vehicle_nos
                    for delayed in intent.delayed_commitments
                    if delayed.vehicle_no == vehicle_no
                )
                and not _move_is_required_by_resource_debt(step, intent=intent)
            ):
                return None
            kept_steps.append(step)
            pending_attach = step
            continue
        delayed_hit = False
        strict_delayed_hit = False
        for vehicle_no in step.vehicle_nos:
            if (vehicle_no, step.target_track) in delayed_targets:
                delayed_hit = True
                vehicle = vehicle_by_no.get(vehicle_no)
                if (
                    vehicle is not None
                    and vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS
                ):
                    strict_delayed_hit = True
        if delayed_hit and strict_delayed_hit and candidate.reason.startswith("work_position_"):
            kept_steps.append(step)
            pending_attach = None
            continue
        if delayed_hit:
            if pending_attach is not None:
                kept_steps = kept_steps[:-1]
            if not kept_steps or candidate.kind != "structural":
                return None
            return MoveCandidate(
                steps=tuple(kept_steps),
                kind=candidate.kind,
                reason=candidate.reason,
                focus_tracks=candidate.focus_tracks,
                structural_reserve=candidate.structural_reserve,
            )
        kept_steps.append(step)
        pending_attach = None
    return candidate


def _move_breaks_protected_commitment(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    if move.action_type != "ATTACH" or not move.vehicle_nos:
        return False
    for block in intent.committed_blocks_by_track.get(move.source_track, ()):
        protected = set(block.vehicle_nos)
        if protected.intersection(move.vehicle_nos):
            return True
    return False


def _move_is_unowned_staging_churn(
    move: HookAction,
    *,
    intent: StructuralIntent,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if move.action_type != "DETACH" or not move.vehicle_nos:
        return False
    if state.loco_track_name not in STAGING_TRACKS or move.target_track not in STAGING_TRACKS:
        return False
    delayed_vehicle_nos = {
        delayed.vehicle_no
        for delayed in intent.delayed_commitments
    }
    if any(vehicle_no in delayed_vehicle_nos for vehicle_no in move.vehicle_nos):
        return True
    for vehicle_no in move.vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if not goal_is_satisfied(
            vehicle,
            track_name=state.loco_track_name,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _move_attaches_unowned_delayed_staging_buffer(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    if move.action_type != "ATTACH" or move.source_track not in STAGING_TRACKS:
        return False
    delayed_vehicle_nos = {
        delayed.vehicle_no
        for delayed in intent.delayed_commitments
    }
    return any(vehicle_no in delayed_vehicle_nos for vehicle_no in move.vehicle_nos)


def _move_mixes_resource_debt_groups(
    move: HookAction,
    *,
    intent: StructuralIntent,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if move.action_type != "ATTACH" or len(move.vehicle_nos) <= 1:
        return False
    source_seq = list(state.track_sequences.get(move.source_track, []))
    if source_seq[: len(move.vehicle_nos)] != list(move.vehicle_nos):
        return False
    moved = list(move.vehicle_nos)
    for debt in intent.resource_debts:
        if debt.track_name != move.source_track:
            continue
        debt_prefix = list(debt.vehicle_nos)
        if not debt_prefix:
            continue
        if len(moved) <= len(debt_prefix):
            if moved != debt_prefix[: len(moved)]:
                continue
        elif moved[: len(debt_prefix)] != debt_prefix:
            continue
        groups = _resource_debt_prefix_group_keys(
            source_track=move.source_track,
            vehicle_nos=moved,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        return len(groups) > 1
    return False


def _resource_debt_prefix_group_keys(
    *,
    source_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for vehicle_no in vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            keys.append(("UNKNOWN", ""))
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            key = ("KEEP", source_track)
        else:
            target = _single_allowed_target(vehicle, source_track=source_track)
            key = ("TARGET", target) if target is not None else ("BUFFER", "")
        if not keys or keys[-1] != key:
            keys.append(key)
    return keys


def _move_is_required_by_resource_debt(
    move: HookAction,
    *,
    intent: StructuralIntent,
) -> bool:
    moved = set(move.vehicle_nos)
    if not moved:
        return False
    for debt in intent.resource_debts:
        if debt.track_name != move.source_track:
            continue
        if moved.intersection(debt.vehicle_nos):
            return True
    return False


def _generate_structural_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData | None,
    route_oracle: RouteOracle | None,
    intent: StructuralIntent,
) -> tuple[MoveCandidate, ...]:
    if master is None or route_oracle is None or state.loco_carry:
        return ()
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    track_by_vehicle = _vehicle_track_lookup(state)
    delayed_target_pairs = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }
    candidates: list[MoveCandidate] = []
    resource_candidates: list[MoveCandidate] = []
    for cluster in _ordered_debt_clusters(intent):
        target_track = cluster.track_name
        debt = cluster.order_debt
        if debt is not None:
            candidate = _build_work_position_source_opening_candidate(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                track_by_vehicle=track_by_vehicle,
                target_track=target_track,
                pending_vehicle_nos=list(debt.pending_vehicle_nos),
                blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
            )
            if candidate is None:
                candidate = _build_work_position_window_candidate(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    track_by_vehicle=track_by_vehicle,
                    target_track=target_track,
                    pending_vehicle_nos=list(debt.pending_vehicle_nos),
                    blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
                )
            if candidate is not None:
                candidates.append(candidate)
            front_candidate = _build_work_position_window_candidate(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                track_by_vehicle=track_by_vehicle,
                target_track=target_track,
                pending_vehicle_nos=list(debt.pending_vehicle_nos),
                blocking_prefix_vehicle_nos=list(debt.blocking_prefix_vehicle_nos),
                front_block_only=True,
            )
            if front_candidate is not None:
                candidates.append(front_candidate)
            fill_candidate = _build_work_position_free_fill_candidate(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                track_by_vehicle=track_by_vehicle,
                target_track=target_track,
                pending_vehicle_nos=list(debt.pending_vehicle_nos),
            )
            if fill_candidate is not None:
                candidates.append(fill_candidate)
        for debt in cluster.resource_debts:
            if debt.kind == "ROUTE_RELEASE":
                route_candidate = _build_route_release_frontier_candidate(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    debt=debt,
                    delayed_target_pairs=delayed_target_pairs,
                    buffer_leases=intent.buffer_leases,
                )
                if route_candidate is not None:
                    resource_candidates.append(route_candidate)
            if debt.kind == "FRONT_CLEARANCE":
                frontier_candidate = _build_goal_frontier_source_opening_candidate(
                    plan_input=plan_input,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=debt.track_name,
                )
                if frontier_candidate is not None:
                    resource_candidates.append(frontier_candidate)
                    continue
            candidate = _build_resource_release_candidate(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                debt=debt,
                delayed_target_pairs=delayed_target_pairs,
                buffer_leases=intent.buffer_leases,
            )
            if candidate is not None:
                resource_candidates.append(candidate)
    candidates.extend(_rank_release_structural_candidates(resource_candidates))
    return _select_structural_candidates(candidates)


def _ordered_debt_clusters(intent: StructuralIntent):
    debt_clusters_by_track = intent.debt_clusters_by_track or build_debt_clusters_by_track(
        order_debts_by_track=intent.order_debts_by_track,
        resource_debts=intent.resource_debts,
    )
    return tuple(
        sorted(
            debt_clusters_by_track.values(),
            key=lambda cluster: (
                0 if cluster.order_debt is not None else 1,
                -cluster.pressure,
                len(cluster.resource_debts),
                cluster.track_name,
            ),
        )
    )


def _build_route_release_frontier_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt: Any,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    blocker_block = list(debt.vehicle_nos)
    if not blocker_block:
        return None
    blocking_track = debt.track_name
    if list(state.track_sequences.get(blocking_track, []))[: len(blocker_block)] != blocker_block:
        return None

    frontier = _route_release_frontier_block(
        state=state,
        vehicle_by_no=vehicle_by_no,
        blocked_vehicle_nos=list(debt.blocked_vehicle_nos),
        source_tracks=list(getattr(debt, "source_tracks", ())),
        target_tracks=list(getattr(debt, "target_tracks", ())),
    )
    if frontier is None:
        return None
    source_track, target_track, frontier_block = frontier

    blocker_clear = _build_resource_release_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        source_track=blocking_track,
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=buffer_leases,
    )
    if blocker_clear is None:
        blocker_clear = _build_blocker_direct_goal_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=blocker_block,
            source_track=blocking_track,
            delayed_target_pairs=delayed_target_pairs,
        )
    if blocker_clear is None:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=blocking_track,
            block=blocker_block,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks={blocking_track, source_track, target_track},
            prefer_low_route_pressure=True,
            buffer_leases=buffer_leases,
        )
        if stage_track is None:
            return None
        blocker_steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=blocking_track,
            target_track=stage_track,
            block=blocker_block,
            action_source_track=blocking_track,
        )
        if blocker_steps is None:
            return None
    else:
        blocker_steps = list(blocker_clear.steps)

    compiled_clear = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=blocker_steps,
        route_oracle=route_oracle,
    )
    if compiled_clear is None or compiled_clear.final_state.loco_carry:
        return None

    frontier_steps = _attach_detach_steps(
        state=compiled_clear.final_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=frontier_block,
        action_source_track=source_track,
    )
    if frontier_steps is None:
        return None
    compiled_frontier = replay_candidate_steps(
        plan_input=plan_input,
        state=compiled_clear.final_state,
        vehicle_by_no=vehicle_by_no,
        steps=frontier_steps,
        route_oracle=route_oracle,
    )
    if compiled_frontier is None or compiled_frontier.final_state.loco_carry:
        return None
    steps = [*compiled_clear.steps, *compiled_frontier.steps]
    restore_steps = _restore_committed_blocker_steps(
        plan_input=plan_input,
        state=compiled_frontier.final_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        original_track=blocking_track,
        clear_steps=list(compiled_clear.steps),
    )
    if _block_is_satisfied_on_track(
        plan_input=plan_input,
        state=compiled_frontier.final_state,
        vehicle_by_no=vehicle_by_no,
        block=blocker_block,
        track_name=blocking_track,
    ):
        if not restore_steps:
            return None
        compiled_restore = replay_candidate_steps(
            plan_input=plan_input,
            state=compiled_frontier.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None or compiled_restore.final_state.loco_carry:
            return None
        steps.extend(compiled_restore.steps)

    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="route_release_frontier",
        focus_tracks=(blocking_track, source_track, target_track),
        structural_reserve=True,
    )


def _build_blocker_direct_goal_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
) -> MoveCandidate | None:
    for target_track in _resource_release_target_tracks(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
    ):
        if target_track in STAGING_TRACKS:
            continue
        steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        return MoveCandidate(
            steps=compiled.steps,
            kind="structural",
            reason="resource_release",
            focus_tracks=(source_track,),
            structural_reserve=True,
        )
    return None


def _block_is_satisfied_on_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    track_name: str,
) -> bool:
    return bool(block) and all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=track_name,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in block
    )


def _restore_committed_blocker_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    original_track: str,
    clear_steps: list[HookAction],
) -> list[HookAction]:
    if not _block_is_satisfied_on_track(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        track_name=original_track,
    ):
        return []
    staging_track = None
    for step in clear_steps:
        if (
            step.action_type == "DETACH"
            and step.vehicle_nos == block
            and step.target_track != original_track
        ):
            staging_track = step.target_track
            break
    if staging_track is None:
        return []
    if list(state.track_sequences.get(staging_track, []))[: len(block)] != block:
        return []
    return _attach_detach_steps(
        state=state,
        route_oracle=route_oracle,
        source_track=staging_track,
        target_track=original_track,
        block=block,
        action_source_track=staging_track,
    ) or []


def _route_release_frontier_block(
    *,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    blocked_vehicle_nos: list[str],
    source_tracks: list[str],
    target_tracks: list[str],
) -> tuple[str, str, list[str]] | None:
    blocked = set(blocked_vehicle_nos)
    for source_track in source_tracks:
        source_seq = list(state.track_sequences.get(source_track, []))
        if not source_seq or source_seq[0] not in blocked:
            continue
        first_vehicle = vehicle_by_no.get(source_seq[0])
        if first_vehicle is None:
            continue
        target_track = next(
            (
                target
                for target in first_vehicle.goal.allowed_target_tracks
                if target in target_tracks and target != source_track
            ),
            None,
        )
        if target_track is None:
            continue
        block = [source_seq[0]]
        for vehicle_no in source_seq[1:]:
            if vehicle_no not in blocked:
                break
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None or target_track not in vehicle.goal.allowed_target_tracks:
                break
            block.append(vehicle_no)
        return source_track, target_track, block
    return None


def _select_structural_candidates(
    candidates: list[MoveCandidate],
    *,
    limit: int = 6,
) -> tuple[MoveCandidate, ...]:
    if len(candidates) <= limit:
        return tuple(candidates)

    selected: list[MoveCandidate] = []
    selected_ids: set[int] = set()
    covered_focus_tracks: set[tuple[str, ...]] = set()

    for candidate in candidates:
        key = tuple(candidate.focus_tracks)
        if key in covered_focus_tracks:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        covered_focus_tracks.add(key)
        if len(selected) >= limit:
            return tuple(selected)

    covered_reason_tracks: set[tuple[str, tuple[str, ...]]] = {
        (candidate.reason, tuple(candidate.focus_tracks))
        for candidate in selected
    }
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        key = (candidate.reason, tuple(candidate.focus_tracks))
        if key in covered_reason_tracks:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        covered_reason_tracks.add(key)
        if len(selected) >= limit:
            return tuple(selected)
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            return tuple(selected)
    return tuple(selected)


def _rank_release_structural_candidates(
    candidates: list[MoveCandidate],
) -> list[MoveCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            STRUCTURAL_PRIORITY_REASON_GROUPS.get(candidate.reason, 2),
            len(candidate.steps),
            0 if candidate.structural_reserve else 1,
            tuple(candidate.focus_tracks),
            candidate.reason,
        ),
    )


def _build_goal_frontier_source_opening_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
) -> MoveCandidate | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if len(source_seq) < 2:
        return None
    first_unsatisfied = next(
        (
            index
            for index, vehicle_no in enumerate(source_seq)
            if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            )
        ),
        None,
    )
    if first_unsatisfied is None or first_unsatisfied <= 0:
        return None
    prefix = source_seq[:first_unsatisfied]
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in prefix
    ):
        return None
    target_track = _single_allowed_target(
        vehicle_by_no[source_seq[first_unsatisfied]],
        source_track=source_track,
    )
    if target_track is None:
        return None
    transfer_block = [source_seq[first_unsatisfied]]
    scan_index = first_unsatisfied + 1
    while scan_index < len(source_seq):
        vehicle = vehicle_by_no.get(source_seq[scan_index])
        if vehicle is None:
            break
        if _single_allowed_target(vehicle, source_track=source_track) != target_track:
            break
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            break
        transfer_block.append(source_seq[scan_index])
        scan_index += 1
    if not _prefix_group_can_detach_to_target(
        group=transfer_block,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    candidate = _build_source_prefix_split_detach_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=prefix,
        transfer_block=transfer_block,
        reason="goal_frontier_source_opening",
        focus_tracks=(source_track, target_track),
    )
    if candidate is None:
        return None
    final_state = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=candidate.steps,
        route_oracle=route_oracle,
    )
    if final_state is None:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=final_state.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=final_state.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return candidate


def _build_resource_release_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt: Any,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    if debt.kind not in {"ROUTE_RELEASE", "CAPACITY_RELEASE", "FRONT_CLEARANCE", "EXACT_SPOT_RELEASE"}:
        return None
    block = list(debt.vehicle_nos)
    if not block:
        return None
    source_track = debt.track_name
    if list(state.track_sequences.get(source_track, []))[: len(block)] != block:
        return None
    candidate = _build_resource_release_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=buffer_leases,
    )
    if candidate is not None:
        return candidate
    if debt.kind == "CAPACITY_RELEASE":
        candidate = _build_protected_prefix_resource_release_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=block,
            source_track=source_track,
            delayed_target_pairs=delayed_target_pairs,
            buffer_leases=buffer_leases,
        )
        if candidate is not None:
            return candidate
    target_tracks = _resource_release_target_tracks(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        delayed_target_pairs=delayed_target_pairs,
    )
    staging_track = _best_staging_track(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks={source_track},
        prefer_low_route_pressure=debt.kind == "ROUTE_RELEASE",
        buffer_leases=buffer_leases if debt.kind == "ROUTE_RELEASE" else (),
    )
    for target_track in [*target_tracks, *([staging_track] if staging_track else [])]:
        if target_track is None or target_track == source_track:
            continue
        steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        return MoveCandidate(
            steps=compiled.steps,
            kind="structural",
            reason="resource_release",
            focus_tracks=(source_track,),
            structural_reserve=True,
        )
    return None


def _build_protected_prefix_resource_release_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    protected_prefix, release_suffix = _protected_prefix_resource_release_split(
        plan_input=plan_input,
        state=state,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
    )
    if not protected_prefix or not release_suffix:
        return None

    steps: list[HookAction] = []
    current_state = state
    staged_prefix: list[tuple[str, list[str]]] = []
    forbidden_tracks = {source_track}
    remaining_prefix = list(protected_prefix)
    while remaining_prefix:
        chunk_plan = _next_prefix_staging_chunk(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=source_track,
            transfer_block=release_suffix,
            remaining_prefix=remaining_prefix,
            forbidden_tracks=forbidden_tracks,
            allow_split_prefix=True,
        )
        if chunk_plan is None:
            return None
        stage_track, chunk, chunk_steps, next_state = chunk_plan
        steps.extend(chunk_steps)
        staged_prefix.append((stage_track, chunk))
        forbidden_tracks.add(stage_track)
        current_state = next_state
        remaining_prefix = remaining_prefix[len(chunk):]

    release_steps = _compile_resource_release_suffix_steps(
        plan_input=plan_input,
        state=current_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        release_suffix=release_suffix,
        delayed_target_pairs=delayed_target_pairs,
        forbidden_tracks=forbidden_tracks,
        buffer_leases=buffer_leases,
    )
    if release_steps is None:
        return None
    suffix_steps, current_state = release_steps
    steps.extend(suffix_steps)

    for stage_track, chunk in reversed(staged_prefix):
        restore_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=stage_track,
            target_track=source_track,
            block=chunk,
            action_source_track=stage_track,
        )
        if restore_steps is None:
            return None
        compiled_restore = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None or compiled_restore.final_state.loco_carry:
            return None
        steps.extend(compiled_restore.steps)
        current_state = compiled_restore.final_state

    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason="resource_release",
        focus_tracks=(source_track,),
        structural_reserve=True,
    )


def _protected_prefix_resource_release_split(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> tuple[list[str], list[str]]:
    protected_prefix: list[str] = []
    index = 0
    while index < len(block):
        vehicle = vehicle_by_no.get(block[index])
        if vehicle is None:
            return [], []
        if not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            break
        protected_prefix.append(block[index])
        index += 1
    return protected_prefix, block[index:]


def _compile_resource_release_suffix_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    release_suffix: list[str],
    delayed_target_pairs: set[tuple[str, str]] | None,
    forbidden_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
) -> tuple[tuple[HookAction, ...], ReplayState] | None:
    groups = _resource_release_prefix_groups(
        block=release_suffix,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        route_oracle=route_oracle,
    )
    if not groups:
        return None

    steps: list[HookAction] = []
    current_state = state
    used_tracks = set(forbidden_tracks)
    for preferred_target, group in groups:
        compiled = None
        if (
            preferred_target is not None
            and preferred_target != source_track
            and not validate_hook_vehicle_group(
                [vehicle_by_no[vehicle_no] for vehicle_no in group]
            )
            and _prefix_group_can_detach_to_target(
                group=group,
                target_track=preferred_target,
                state=current_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        ):
            direct_steps = _attach_detach_steps(
                state=current_state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=preferred_target,
                block=group,
                action_source_track=source_track,
            )
            if direct_steps is not None:
                compiled = replay_candidate_steps(
                    plan_input=plan_input,
                    state=current_state,
                    vehicle_by_no=vehicle_by_no,
                    steps=direct_steps,
                    route_oracle=route_oracle,
                )
                if compiled is not None and compiled.final_state.loco_carry:
                    compiled = None
        if compiled is None:
            compiled = _compile_staging_group_with_source_reaccess(
                plan_input=plan_input,
                state=current_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                group=group,
                forbidden_staging_tracks=used_tracks,
                buffer_leases=buffer_leases,
                require_source_reaccess=False,
            )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state
        if compiled.steps:
            used_tracks.add(compiled.steps[-1].target_track)
    return tuple(steps), current_state


def _build_source_prefix_split_detach_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    reason: str,
    focus_tracks: tuple[str, ...],
) -> MoveCandidate | None:
    if not protected_prefix or not transfer_block:
        return None
    source_seq = list(state.track_sequences.get(source_track, []))
    combined_block = [*protected_prefix, *transfer_block]
    if source_seq[: len(combined_block)] != combined_block:
        return None

    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(combined_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    after_attach = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if after_attach is None:
        return None

    transfer_path = _clear_path_tracks_for_detach(
        route_oracle=route_oracle,
        state=after_attach.final_state,
        source_track=source_track,
        target_track=target_track,
    )
    if transfer_path is None:
        return None
    transfer_detach = HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(transfer_block),
        path_tracks=transfer_path,
        action_type="DETACH",
    )
    after_transfer = replay_candidate_steps(
        plan_input=plan_input,
        state=after_attach.final_state,
        vehicle_by_no=vehicle_by_no,
        steps=[transfer_detach],
        route_oracle=route_oracle,
    )
    if after_transfer is None or after_transfer.final_state.loco_carry != tuple(protected_prefix):
        return None

    restore_path = _clear_path_tracks_for_detach(
        route_oracle=route_oracle,
        state=after_transfer.final_state,
        source_track=target_track,
        target_track=source_track,
    )
    if restore_path is None:
        return None
    restore_detach = HookAction(
        source_track=target_track,
        target_track=source_track,
        vehicle_nos=list(protected_prefix),
        path_tracks=restore_path,
        action_type="DETACH",
    )
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach, transfer_detach, restore_detach],
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason=reason,
        focus_tracks=focus_tracks,
        structural_reserve=True,
    )


def _clear_path_tracks_for_detach(
    *,
    route_oracle: RouteOracle,
    state: ReplayState,
    source_track: str,
    target_track: str,
) -> list[str] | None:
    source_remaining = len(state.track_sequences.get(source_track, []))
    source_node = state.loco_node if source_remaining > 0 else None
    target_node = route_oracle.order_end_node(target_track)
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=target_node,
    )
    if path_tracks is None:
        return None
    route = route_oracle.resolve_route_for_path_tracks(
        path_tracks,
        source_node=source_node,
        target_node=target_node,
    )
    if route is None:
        return None
    return path_tracks


def _build_resource_release_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    groups = _resource_release_prefix_groups(
        block=block,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        route_oracle=route_oracle,
    )
    if not groups or len(groups) <= 1:
        return None
    return _build_group_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        groups=groups,
        source_track=source_track,
        buffer_leases=buffer_leases,
        reason="resource_release",
        focus_tracks=(source_track,),
    )


def _build_route_release_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    deferred_target_tracks: set[str],
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    buffer_leases: tuple[BufferLease, ...] = (),
) -> MoveCandidate | None:
    groups = _resource_release_prefix_groups(
        block=block,
        source_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        delayed_target_pairs=delayed_target_pairs,
        deferred_target_tracks=deferred_target_tracks,
        route_oracle=route_oracle,
    )
    if not groups or len(groups) <= 1:
        return None
    return _build_group_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        groups=groups,
        source_track=source_track,
        buffer_leases=buffer_leases,
        reason="resource_release",
        focus_tracks=(source_track,),
    )


def _build_group_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    groups: list[tuple[str | None, list[str]]],
    source_track: str,
    buffer_leases: tuple[BufferLease, ...] = (),
    reason: str,
    focus_tracks: tuple[str, ...],
) -> MoveCandidate | None:
    split_candidate = _build_carried_prefix_dispatch_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        groups=groups,
        source_track=source_track,
        buffer_leases=buffer_leases,
    )
    if split_candidate is not None:
        return MoveCandidate(
            steps=split_candidate.steps,
            kind="structural",
            reason=reason,
            focus_tracks=focus_tracks,
            structural_reserve=True,
        )

    steps: list[HookAction] = []
    current_state = state
    forbidden_staging_tracks = {source_track}
    for group_index, (preferred_target, group) in enumerate(groups):
        require_source_reaccess = group_index < len(groups) - 1 or _source_tail_after_group_needs_access(
            state=current_state,
            source_track=source_track,
            group=group,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        compiled = None
        if (
            preferred_target is not None
            and preferred_target != source_track
            and _prefix_group_can_detach_to_target(
                group=group,
                target_track=preferred_target,
                state=current_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        ):
            direct_steps = _attach_detach_steps(
                state=current_state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=preferred_target,
                block=group,
                action_source_track=source_track,
            )
            if direct_steps is not None:
                compiled = replay_candidate_steps(
                    plan_input=plan_input,
                    state=current_state,
                    vehicle_by_no=vehicle_by_no,
                    steps=direct_steps,
                    route_oracle=route_oracle,
                )
                if compiled is not None and compiled.final_state.loco_carry:
                    compiled = None
                if (
                    compiled is not None
                    and require_source_reaccess
                    and not _loco_can_access_track(
                        state=compiled.final_state,
                        route_oracle=route_oracle,
                        target_track=source_track,
                    )
                ):
                    compiled = None
        if compiled is None:
            compiled = _compile_staging_group_with_source_reaccess(
                plan_input=plan_input,
                state=current_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                group=group,
                forbidden_staging_tracks=forbidden_staging_tracks,
                buffer_leases=buffer_leases,
                require_source_reaccess=require_source_reaccess,
            )
            if compiled is None:
                return None
            stage_track = compiled.steps[-1].target_track
            forbidden_staging_tracks.add(stage_track)
        elif preferred_target in STAGING_TRACKS:
            forbidden_staging_tracks.add(preferred_target)
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    if not steps:
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason=reason,
        focus_tracks=focus_tracks,
        structural_reserve=True,
    )


def _source_tail_after_group_needs_access(
    *,
    state: ReplayState,
    source_track: str,
    group: list[str],
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(group)] != list(group):
        return False
    for vehicle_no in source_seq[len(group):]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return True
        if not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _compile_staging_group_with_source_reaccess(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    group: list[str],
    forbidden_staging_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
    require_source_reaccess: bool,
) -> Any | None:
    for stage_track in _ranked_staging_tracks(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=group,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_staging_tracks,
        prefer_low_route_pressure=True,
        buffer_leases=buffer_leases,
    ):
        stage_steps = _attach_detach_steps(
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=stage_track,
            block=group,
            action_source_track=source_track,
        )
        if stage_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=stage_steps,
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        if require_source_reaccess and not _loco_can_access_track(
            state=compiled.final_state,
            route_oracle=route_oracle,
            target_track=source_track,
        ):
            continue
        return compiled
    return None


def _build_carried_prefix_dispatch_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    groups: list[tuple[str | None, list[str]]],
    source_track: str,
    buffer_leases: tuple[BufferLease, ...],
) -> MoveCandidate | None:
    block = [vehicle_no for _target, group in groups for vehicle_no in group]
    if not block:
        return None
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    current = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if current is None:
        return None

    steps: list[HookAction] = [attach]
    used_staging_tracks: set[str] = {source_track}
    for preferred_target, group in reversed(groups):
        detach = _carried_group_detach_step(
            plan_input=plan_input,
            state=current.final_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            group=group,
            preferred_target=preferred_target,
            used_staging_tracks=used_staging_tracks,
            buffer_leases=buffer_leases,
        )
        if detach is None:
            return None
        if detach.target_track in STAGING_TRACKS:
            used_staging_tracks.add(detach.target_track)
        current = replay_candidate_steps(
            plan_input=plan_input,
            state=current.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[detach],
            route_oracle=route_oracle,
        )
        if current is None:
            return None
        steps.append(detach)
    if current.final_state.loco_carry:
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="resource_release",
        focus_tracks=(source_track,),
        structural_reserve=True,
    )


def _carried_group_detach_step(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    group: list[str],
    preferred_target: str | None,
    used_staging_tracks: set[str],
    buffer_leases: tuple[BufferLease, ...],
) -> HookAction | None:
    target_tracks: list[str] = []
    if preferred_target is not None:
        target_tracks.append(preferred_target)
    target_tracks.extend(
        track
        for track in _ranked_staging_tracks(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=state.loco_track_name,
            block=group,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=used_staging_tracks,
            prefer_low_route_pressure=True,
            buffer_leases=buffer_leases,
            block_is_carried=True,
        )
        if track not in target_tracks
    )
    for target_track in target_tracks:
        detach_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=state,
            source_track=state.loco_track_name,
            target_track=target_track,
        )
        if detach_path is None:
            continue
        return HookAction(
            source_track=state.loco_track_name,
            target_track=target_track,
            vehicle_nos=list(group),
            path_tracks=detach_path,
            action_type="DETACH",
        )
    return None


def _resource_release_prefix_groups(
    *,
    block: list[str],
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    delayed_target_pairs: set[tuple[str, str]] | None = None,
    deferred_target_tracks: set[str] | None = None,
    route_oracle: RouteOracle | None = None,
) -> list[tuple[str | None, list[str]]]:
    groups: list[tuple[str | None, list[str]]] = []
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        if goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            target_track = source_track
        else:
            target_track = _resource_release_group_target(
                vehicle=vehicle,
                vehicle_no=vehicle_no,
                source_track=source_track,
                state=state,
                plan_input=plan_input,
                delayed_target_pairs=delayed_target_pairs,
                deferred_target_tracks=deferred_target_tracks,
                route_oracle=route_oracle,
            )
        if groups and groups[-1][0] == target_track:
            groups[-1][1].append(vehicle_no)
        else:
            groups.append((target_track, [vehicle_no]))
    return groups


def _resource_release_group_target(
    *,
    vehicle: NormalizedVehicle,
    vehicle_no: str,
    source_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    delayed_target_pairs: set[tuple[str, str]] | None,
    deferred_target_tracks: set[str] | None,
    route_oracle: RouteOracle | None,
) -> str | None:
    delayed_target_pairs = delayed_target_pairs or set()
    deferred_target_tracks = deferred_target_tracks or set()
    for target_track in goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    ):
        if target_track == source_track:
            continue
        if target_track in deferred_target_tracks:
            continue
        if (vehicle_no, target_track) in delayed_target_pairs:
            continue
        return target_track
    return None


def _block_contains_delayed_order_buffer(
    block: list[str],
    *,
    delayed_target_pairs: set[tuple[str, str]] | None,
) -> bool:
    if not delayed_target_pairs:
        return False
    delayed_vehicle_nos = {vehicle_no for vehicle_no, _target_track in delayed_target_pairs}
    return any(vehicle_no in delayed_vehicle_nos for vehicle_no in block)


def _resource_release_target_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    delayed_target_pairs: set[tuple[str, str]] | None = None,
) -> list[str]:
    delayed_target_pairs = delayed_target_pairs or set()
    targets: list[str] = []
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return []
        for target_track in vehicle.goal.allowed_target_tracks:
            if (
                target_track == source_track
                or target_track in targets
                or (vehicle_no, target_track) in delayed_target_pairs
            ):
                continue
            targets.append(target_track)
    if len(block) == 1:
        return targets
    compatible: list[str] = []
    for target_track in targets:
        if all(
            (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and target_track in vehicle.goal.allowed_target_tracks
            and not goal_is_satisfied(
                vehicle,
                track_name=source_track,
                state=state,
                plan_input=plan_input,
            )
            for vehicle_no in block
        ):
            compatible.append(target_track)
    return compatible


def _build_work_position_window_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
    front_block_only: bool = False,
) -> MoveCandidate | None:
    commitment_vehicle_nos = [
        vehicle_no
        for vehicle_no in pending_vehicle_nos
        if (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS
    ]
    if not commitment_vehicle_nos:
        return None
    source_track = track_by_vehicle.get(commitment_vehicle_nos[0])
    if source_track is None or source_track == target_track:
        return None
    source_seq = list(state.track_sequences.get(source_track, []))
    target_seq = list(state.track_sequences.get(target_track, []))
    if front_block_only:
        commitment_vehicle_nos = _front_work_position_commitment_block(
            source_seq=source_seq,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
        if not commitment_vehicle_nos:
            return None
    if _work_position_commitment_should_wait_for_free_buffers(
        commitment_vehicle_nos=commitment_vehicle_nos,
        pending_vehicle_nos=pending_vehicle_nos,
        target_seq=target_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    block_clear_count = _work_position_block_clear_count(
        target_track=target_track,
        target_seq=target_seq,
        incoming_vehicle_nos=commitment_vehicle_nos,
        vehicle_by_no=vehicle_by_no,
    )
    if block_clear_count is None:
        return None
    max_clear_count = max(len(blocking_prefix_vehicle_nos), block_clear_count)
    if _prefix_clearance_crosses_stable_work_window(
        target_seq=target_seq,
        clear_count=max_clear_count,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        state=state,
    ):
        return None
    blocking_prefix_vehicle_nos = target_seq[
        : max_clear_count
    ]
    source_indexes = [
        source_seq.index(vehicle_no)
        for vehicle_no in commitment_vehicle_nos
        if vehicle_no in source_seq
    ]
    if len(source_indexes) != len(commitment_vehicle_nos):
        return None
    first_index = min(source_indexes)
    last_index = max(source_indexes)
    if source_seq[first_index : last_index + 1] != commitment_vehicle_nos:
        return None
    order_buffer = source_seq[:first_index]
    if any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is None
        or (
            target_track in vehicle.goal.allowed_target_tracks
            and vehicle.goal.work_position_kind == "SPOTTING"
        )
        for vehicle_no in order_buffer
    ):
        return None

    steps: list[HookAction] = []
    current_state = state
    if blocking_prefix_vehicle_nos:
        clear_steps, current_state = _compile_prefix_clearance_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=target_track,
            vehicle_by_no=vehicle_by_no,
            block=blocking_prefix_vehicle_nos,
            forbidden_tracks={source_track, target_track},
        )
        if clear_steps is None:
            return None
        steps.extend(clear_steps)

    if order_buffer:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=order_buffer,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks={source_track, target_track},
        )
        if stage_track is None:
            return None
        buffer_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=stage_track,
            block=order_buffer,
            action_source_track=source_track,
        )
        if buffer_steps is None:
            return None
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=buffer_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state

    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=commitment_vehicle_nos,
        action_source_track=source_track,
    )
    if target_steps is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=current_state,
        vehicle_by_no=vehicle_by_no,
        steps=target_steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    final_state = compiled.final_state
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=final_state,
            plan_input=plan_input,
        )
        for vehicle_no in commitment_vehicle_nos
    ):
        return None
    steps.extend(compiled.steps)
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="work_position_window_repair",
        focus_tracks=(target_track,),
        structural_reserve=True,
    )


def _build_work_position_source_opening_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
) -> MoveCandidate | None:
    strict_commitments_by_source: dict[str, list[str]] = {}
    free_commitments_by_source: dict[str, list[str]] = {}
    for vehicle_no in pending_vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.goal.work_position_kind is None:
            continue
        source_track = track_by_vehicle.get(vehicle_no)
        if source_track is None or source_track == target_track:
            continue
        if vehicle.goal.work_position_kind in STRICT_WORK_POSITION_KINDS:
            strict_commitments_by_source.setdefault(source_track, []).append(vehicle_no)
        elif vehicle.goal.work_position_kind == "FREE":
            free_commitments_by_source.setdefault(source_track, []).append(vehicle_no)
    commitment_sources = [strict_commitments_by_source]
    if not strict_commitments_by_source:
        commitment_sources.append(free_commitments_by_source)
    for commitments_by_source in commitment_sources:
        for source_track, source_spotting in sorted(
            commitments_by_source.items(),
            key=lambda item: _first_source_index(
                source_seq=list(state.track_sequences.get(item[0], [])),
                vehicle_nos=item[1],
            ),
        ):
            candidate = _build_source_opening_for_track(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                target_track=target_track,
                source_commitments=source_spotting,
                pending_vehicle_nos=pending_vehicle_nos,
                blocking_prefix_vehicle_nos=blocking_prefix_vehicle_nos,
            )
            if candidate is not None:
                return candidate
    return None


def _build_source_opening_for_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    source_commitments: list[str],
    pending_vehicle_nos: list[str],
    blocking_prefix_vehicle_nos: list[str],
) -> MoveCandidate | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if not source_seq:
        return None
    target_seq = list(state.track_sequences.get(target_track, []))
    first_spot_index = min(
        (
            source_seq.index(vehicle_no)
            for vehicle_no in source_commitments
            if vehicle_no in source_seq
        ),
        default=None,
    )
    if first_spot_index is None or first_spot_index == 0:
        return None
    dispatch_prefix = source_seq[:first_spot_index]
    front_commitments = _front_work_position_commitment_block(
        source_seq=source_seq[first_spot_index:],
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    if not front_commitments:
        front_commitments = _front_free_work_position_block(
            source_seq=source_seq[first_spot_index:],
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
    if not front_commitments:
        return None
    if _work_position_commitment_should_wait_for_free_buffers(
        commitment_vehicle_nos=front_commitments,
        pending_vehicle_nos=pending_vehicle_nos,
        target_seq=target_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    block_clear_count = _work_position_block_clear_count(
        target_track=target_track,
        target_seq=target_seq,
        incoming_vehicle_nos=front_commitments,
        vehicle_by_no=vehicle_by_no,
    )
    if block_clear_count is None:
        return None
    max_clear_count = max(len(blocking_prefix_vehicle_nos), block_clear_count)
    if _prefix_clearance_crosses_stable_work_window(
        target_seq=target_seq,
        clear_count=max_clear_count,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        state=state,
    ):
        return None
    blocking_prefix = target_seq[
        : max_clear_count
    ]
    dispatch_groups = _dispatchable_prefix_groups(
        dispatch_prefix=dispatch_prefix,
        source_track=source_track,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )
    if not dispatch_groups:
        prefix_split_candidate = _build_protected_prefix_work_position_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            protected_prefix=dispatch_prefix,
            transfer_block=front_commitments,
            blocking_prefix=blocking_prefix,
        )
        if prefix_split_candidate is not None:
            return prefix_split_candidate
        staged_prefix_candidate = _build_staged_prefix_work_position_candidate(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            protected_prefix=dispatch_prefix,
            transfer_block=front_commitments,
            blocking_prefix=blocking_prefix,
        )
        if staged_prefix_candidate is not None:
            return staged_prefix_candidate
        return None
    steps: list[HookAction] = []
    current_state = state
    if blocking_prefix:
        clear_steps, current_state = _compile_prefix_clearance_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            source_track=target_track,
            vehicle_by_no=vehicle_by_no,
            block=blocking_prefix,
            forbidden_tracks={source_track, target_track},
        )
        if clear_steps is None:
            return None
        steps.extend(clear_steps)
    for group in dispatch_groups:
        group_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=group[0],
            block=group[1],
            action_source_track=source_track,
        )
        if group_steps is None:
            return None
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=group_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=front_commitments,
        action_source_track=source_track,
    )
    if target_steps is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=current_state,
        vehicle_by_no=vehicle_by_no,
        steps=target_steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    final_state = compiled.final_state
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=final_state,
            plan_input=plan_input,
        )
        for vehicle_no in front_commitments
    ):
        return None
    steps.extend(compiled.steps)
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
        structural_reserve=True,
    )


def _build_work_position_free_fill_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    target_track: str,
    pending_vehicle_nos: list[str],
) -> MoveCandidate | None:
    if not _work_position_free_fill_is_allowed(
        state=state,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
        plan_input=plan_input,
        pending_vehicle_nos=pending_vehicle_nos,
    ):
        return None
    source_tracks: list[str] = []
    for vehicle_no in pending_vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None or vehicle.goal.work_position_kind != "FREE":
            continue
        source_track = track_by_vehicle.get(vehicle_no)
        if source_track is None or source_track == target_track or source_track in source_tracks:
            continue
        source_tracks.append(source_track)
    if not source_tracks:
        return None
    steps: list[HookAction] = []
    current_state = state
    for source_track in sorted(source_tracks, key=lambda track: (track not in STAGING_TRACKS, track)):
        source_seq = list(current_state.track_sequences.get(source_track, []))
        block = _front_free_work_position_block(
            source_seq=source_seq,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
        if not block:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=block,
            existing_vehicle_nos=list(current_state.track_sequences.get(target_track, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        fill_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=block,
            action_source_track=source_track,
        )
        if fill_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=fill_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            continue
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    if not steps:
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="structural",
        reason="work_position_free_fill",
        focus_tracks=(target_track,),
        structural_reserve=True,
    )


def _build_protected_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    blocking_prefix: list[str],
) -> MoveCandidate | None:
    if blocking_prefix:
        return None
    if not protected_prefix or not transfer_block:
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not _prefix_group_can_detach_to_target(
        group=transfer_block,
        target_track=target_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    candidate = _build_source_prefix_split_detach_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
    )
    if candidate is None:
        return None
    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=candidate.steps,
        route_oracle=route_oracle,
    )
    if compiled is None:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return candidate


def _build_staged_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    blocking_prefix: list[str],
) -> MoveCandidate | None:
    if blocking_prefix or not protected_prefix or not transfer_block:
        return None
    if not all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    candidate = _compile_staged_prefix_work_position_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        allow_split_prefix=False,
    )
    if candidate is not None:
        return candidate
    return _compile_staged_prefix_work_position_candidate(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        protected_prefix=protected_prefix,
        transfer_block=transfer_block,
        allow_split_prefix=True,
    )


def _compile_staged_prefix_work_position_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    protected_prefix: list[str],
    transfer_block: list[str],
    allow_split_prefix: bool,
) -> MoveCandidate | None:
    steps: list[HookAction] = []
    current_state = state
    staged_chunks: list[tuple[str, list[str]]] = []
    remaining_prefix = list(protected_prefix)
    forbidden_tracks = {source_track, target_track}

    while remaining_prefix:
        chunk_plan = _next_prefix_staging_chunk(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            remaining_prefix=remaining_prefix,
            forbidden_tracks=forbidden_tracks,
            allow_split_prefix=allow_split_prefix,
        )
        if chunk_plan is None:
            return None
        stage_track, chunk, chunk_steps, next_state = chunk_plan
        steps.extend(chunk_steps)
        staged_chunks.append((stage_track, chunk))
        forbidden_tracks.add(stage_track)
        current_state = next_state
        remaining_prefix = remaining_prefix[len(chunk):]

    target_steps = _attach_detach_steps(
        state=current_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        block=transfer_block,
        action_source_track=source_track,
    )
    if target_steps is None:
        protected_transfer = _compile_source_tail_protected_transfer_steps(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            forbidden_tracks=forbidden_tracks,
        )
        if protected_transfer is None:
            return None
        transfer_steps, current_state = protected_transfer
        steps.extend(transfer_steps)
    else:
        compiled_target = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=target_steps,
            route_oracle=route_oracle,
        )
        if compiled_target is None or compiled_target.final_state.loco_carry:
            return None
        steps.extend(compiled_target.steps)
        current_state = compiled_target.final_state

    for stage_track, chunk in reversed(staged_chunks):
        restore_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=stage_track,
            target_track=source_track,
            block=chunk,
            action_source_track=stage_track,
        )
        if restore_steps is None:
            return None
        compiled_restore = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=restore_steps,
            route_oracle=route_oracle,
        )
        if compiled_restore is None or compiled_restore.final_state.loco_carry:
            return None
        steps.extend(compiled_restore.steps)
        current_state = compiled_restore.final_state

    compiled = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=route_oracle,
    )
    if compiled is None or compiled.final_state.loco_carry:
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=source_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in protected_prefix
    ):
        return None
    if not all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name=target_track,
            state=compiled.final_state,
            plan_input=plan_input,
        )
        for vehicle_no in transfer_block
    ):
        return None
    return MoveCandidate(
        steps=compiled.steps,
        kind="structural",
        reason="work_position_source_opening",
        focus_tracks=(target_track,),
        structural_reserve=True,
    )


def _next_prefix_staging_chunk(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    remaining_prefix: list[str],
    forbidden_tracks: set[str],
    allow_split_prefix: bool,
) -> tuple[str, list[str], tuple[HookAction, ...], ReplayState] | None:
    max_chunk_size = len(remaining_prefix) if allow_split_prefix else len(remaining_prefix)
    min_chunk_size = 1 if allow_split_prefix else len(remaining_prefix)
    for size in range(max_chunk_size, min_chunk_size - 1, -1):
        chunk = list(remaining_prefix[:size])
        for stage_track in _ranked_staging_tracks(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=chunk,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        ):
            chunk_steps = _attach_detach_steps(
                state=state,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=stage_track,
                block=chunk,
                action_source_track=source_track,
            )
            if chunk_steps is None:
                continue
            compiled = replay_candidate_steps(
                plan_input=plan_input,
                state=state,
                vehicle_by_no=vehicle_by_no,
                steps=chunk_steps,
                route_oracle=route_oracle,
            )
            if compiled is None or compiled.final_state.loco_carry:
                continue
            if not _loco_can_access_track(
                state=compiled.final_state,
                route_oracle=route_oracle,
                target_track=source_track,
            ):
                continue
            if _source_front_is_transfer_ready(
                state=compiled.final_state,
                source_track=source_track,
                transfer_block=transfer_block,
            ) and not _source_to_target_route_is_open_or_tail_protectable(
                plan_input=plan_input,
                state=compiled.final_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                target_track=target_track,
                transfer_block=transfer_block,
                forbidden_tracks=forbidden_tracks | {stage_track},
            ):
                continue
            return stage_track, chunk, compiled.steps, compiled.final_state
    return None


def _source_front_is_transfer_ready(
    *,
    state: ReplayState,
    source_track: str,
    transfer_block: list[str],
) -> bool:
    if not transfer_block:
        return False
    return list(state.track_sequences.get(source_track, []))[: len(transfer_block)] == list(
        transfer_block
    )


def _source_to_target_route_is_open_or_tail_protectable(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    forbidden_tracks: set[str],
) -> bool:
    if _source_to_target_route_is_open(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        transfer_block=transfer_block,
    ):
        return True
    return (
        _compile_source_tail_protected_transfer_steps(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            transfer_block=transfer_block,
            forbidden_tracks=forbidden_tracks,
        )
        is not None
    )


def _source_to_target_route_is_open(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    transfer_block: list[str],
) -> bool:
    if not transfer_block:
        return True
    source_node = _source_node_after_attach_for_block(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=transfer_block,
    )
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=route_oracle.order_end_node(target_track),
    )
    return path_tracks is not None


def _compile_source_tail_protected_transfer_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    transfer_block: list[str],
    forbidden_tracks: set[str],
) -> tuple[tuple[HookAction, ...], ReplayState] | None:
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(transfer_block)] != list(transfer_block):
        return None
    tail_block = _front_source_stay_block(
        source_seq=source_seq[len(transfer_block) :],
        source_track=source_track,
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
    )
    if not tail_block:
        return None

    combined_block = [*transfer_block, *tail_block]
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(combined_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    after_attach = replay_candidate_steps(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=[attach],
        route_oracle=route_oracle,
    )
    if after_attach is None:
        return None

    for buffer_track in _ranked_source_tail_buffer_tracks(
        plan_input=plan_input,
        state=after_attach.final_state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        tail_block=tail_block,
        transfer_block=transfer_block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_tracks,
    ):
        tail_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_attach.final_state,
            source_track=source_track,
            target_track=buffer_track,
        )
        if tail_path is None:
            continue
        tail_detach = HookAction(
            source_track=source_track,
            target_track=buffer_track,
            vehicle_nos=list(tail_block),
            path_tracks=tail_path,
            action_type="DETACH",
        )
        after_tail = replay_candidate_steps(
            plan_input=plan_input,
            state=after_attach.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[tail_detach],
            route_oracle=route_oracle,
        )
        if after_tail is None or after_tail.final_state.loco_carry != tuple(transfer_block):
            continue

        target_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_tail.final_state,
            source_track=buffer_track,
            target_track=target_track,
        )
        if target_path is None:
            continue
        target_detach = HookAction(
            source_track=buffer_track,
            target_track=target_track,
            vehicle_nos=list(transfer_block),
            path_tracks=target_path,
            action_type="DETACH",
        )
        after_target = replay_candidate_steps(
            plan_input=plan_input,
            state=after_tail.final_state,
            vehicle_by_no=vehicle_by_no,
            steps=[target_detach],
            route_oracle=route_oracle,
        )
        if after_target is None or after_target.final_state.loco_carry:
            continue

        restore_tail_steps = _attach_detach_steps(
            state=after_target.final_state,
            route_oracle=route_oracle,
            source_track=buffer_track,
            target_track=source_track,
            block=tail_block,
            action_source_track=buffer_track,
        )
        if restore_tail_steps is None:
            continue
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=[attach, tail_detach, target_detach, *restore_tail_steps],
            route_oracle=route_oracle,
        )
        if compiled is None or compiled.final_state.loco_carry:
            continue
        return compiled.steps, compiled.final_state
    return None


def _front_source_stay_block(
    *,
    source_seq: list[str],
    source_track: str,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            break
        if not goal_is_satisfied(
            vehicle,
            track_name=source_track,
            state=state,
            plan_input=plan_input,
        ):
            break
        block.append(vehicle_no)
    return block


def _ranked_source_tail_buffer_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    tail_block: list[str],
    transfer_block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
) -> list[str]:
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in tail_block)
    destination_tracks = {
        target
        for vehicle in plan_input.vehicles
        for target in vehicle.goal.allowed_target_tracks
    }
    candidates: list[tuple[tuple[int, int, int, int, int, float, str], str]] = []
    for track_name in sorted(set(capacity_by_track) - forbidden_tracks - {source_track, target_track}):
        if track_name not in STAGING_TRACKS and track_name not in destination_tracks:
            continue
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(track_name, [])
        )
        if current_length + block_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=tail_block,
            existing_vehicle_nos=list(state.track_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        tail_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=state,
            source_track=source_track,
            target_track=track_name,
        )
        if tail_path is None:
            continue
        tail_detach = HookAction(
            source_track=source_track,
            target_track=track_name,
            vehicle_nos=list(tail_block),
            path_tracks=tail_path,
            action_type="DETACH",
        )
        after_tail = replay_candidate_steps(
            plan_input=plan_input,
            state=state,
            vehicle_by_no=vehicle_by_no,
            steps=[tail_detach],
            route_oracle=route_oracle,
        )
        if after_tail is None or after_tail.final_state.loco_carry != tuple(transfer_block):
            continue
        target_path = _clear_path_tracks_for_detach(
            route_oracle=route_oracle,
            state=after_tail.final_state,
            source_track=track_name,
            target_track=target_track,
        )
        if target_path is None:
            continue
        restore_path = route_oracle.resolve_clear_path_tracks(
            target_track,
            track_name,
            occupied_track_sequences=after_tail.final_state.track_sequences,
            source_node=route_oracle.order_end_node(target_track),
            target_node=route_oracle.order_end_node(track_name),
        )
        candidates.append(
            (
                (
                    0 if track_name in STAGING_TRACKS else 1,
                    len(state.track_sequences.get(track_name, [])),
                    len(tail_path),
                    len(target_path),
                    capacity_by_track[track_name] - current_length,
                    track_name,
                ),
                track_name,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [track_name for _rank, track_name in candidates]


def _loco_can_access_track(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    target_track: str,
) -> bool:
    if state.loco_track_name == target_track:
        return True
    access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=target_track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    )
    return access.is_valid


def _work_position_free_fill_is_allowed(
    *,
    state: ReplayState,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    plan_input: NormalizedPlanInput,
    pending_vehicle_nos: list[str],
) -> bool:
    for vehicle_no in state.track_sequences.get(target_track, []):
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            continue
        if not goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            return False
    return True


def _prefix_clearance_crosses_stable_work_window(
    *,
    target_seq: list[str],
    clear_count: int,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
) -> bool:
    for vehicle_no in target_seq[:clear_count]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
        ):
            continue
        if target_track not in vehicle.goal.allowed_target_tracks:
            continue
        if goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _front_free_work_position_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind != "FREE"
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            break
        block.append(vehicle_no)
    return block


def _work_position_block_clear_count(
    *,
    target_track: str,
    target_seq: list[str],
    incoming_vehicle_nos: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int | None:
    for clear_count in range(len(target_seq) + 1):
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=incoming_vehicle_nos,
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        if all(
            (evaluation := preview.evaluations.get(vehicle_no)) is not None
            and evaluation.satisfied_now
            for vehicle_no in incoming_vehicle_nos
        ):
            return clear_count
    return None


def _compile_prefix_clearance_steps(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    forbidden_tracks: set[str],
) -> tuple[list[HookAction], ReplayState] | tuple[None, ReplayState]:
    groups = _dispatchable_prefix_groups(
        dispatch_prefix=block,
        source_track=source_track,
        target_track=source_track,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
        route_oracle=route_oracle,
    )
    if not groups:
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=block,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        )
        if stage_track is None:
            return None, state
        groups = [(stage_track, block)]
    steps: list[HookAction] = []
    current_state = state
    for target_track, group in groups:
        group_steps = _attach_detach_steps(
            state=current_state,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
            block=group,
            action_source_track=source_track,
        )
        if group_steps is None:
            return None, state
        compiled = replay_candidate_steps(
            plan_input=plan_input,
            state=current_state,
            vehicle_by_no=vehicle_by_no,
            steps=group_steps,
            route_oracle=route_oracle,
        )
        if compiled is None:
            return None, state
        steps.extend(compiled.steps)
        current_state = compiled.final_state
    return steps, current_state


def _front_work_position_commitment_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    first_kind: str | None = None
    previous_rank: int | None = None
    for vehicle_no in source_seq:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind not in STRICT_WORK_POSITION_KINDS
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            break
        kind = vehicle.goal.work_position_kind
        if first_kind is None:
            first_kind = kind
        if first_kind == "SPOTTING":
            if kind != "SPOTTING":
                break
        elif kind == "SPOTTING":
            break
        else:
            rank = vehicle.goal.target_rank
            if rank is None:
                break
            if previous_rank is not None and rank < previous_rank:
                break
            previous_rank = rank
        block.append(vehicle_no)
    return block


def _front_spotting_block(
    *,
    source_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block = _front_work_position_commitment_block(
        source_seq=source_seq,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    if not block:
        return []
    first_vehicle = vehicle_by_no.get(block[0])
    if first_vehicle is None or first_vehicle.goal.work_position_kind != "SPOTTING":
        return []
    return block


def _work_position_commitment_should_wait_for_free_buffers(
    *,
    commitment_vehicle_nos: list[str],
    pending_vehicle_nos: list[str],
    target_seq: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    if not any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind in {"EXACT_NORTH_RANK", "EXACT_WORK_SLOT"}
        for vehicle_no in commitment_vehicle_nos
    ):
        return False
    pending_free = [
        vehicle_no
        for vehicle_no in pending_vehicle_nos
        if vehicle_no not in commitment_vehicle_nos
        and (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.goal.work_position_kind == "FREE"
        and target_track in vehicle.goal.allowed_target_tracks
    ]
    if not pending_free:
        return False
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=[*pending_free, *commitment_vehicle_nos],
        existing_vehicle_nos=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return True
    return not all(
        (evaluation := preview.evaluations.get(vehicle_no)) is not None
        and evaluation.satisfied_now
        for vehicle_no in commitment_vehicle_nos
    )


def _dispatchable_prefix_groups(
    *,
    dispatch_prefix: list[str],
    source_track: str,
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
    route_oracle: RouteOracle,
) -> list[tuple[str | None, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(dispatch_prefix):
        vehicle = vehicle_by_no.get(dispatch_prefix[index])
        if vehicle is None:
            return []
        if (
            target_track in vehicle.goal.allowed_target_tracks
            and vehicle.goal.work_position_kind != "SPOTTING"
        ):
            group = [dispatch_prefix[index]]
            index += 1
            while index < len(dispatch_prefix):
                next_vehicle = vehicle_by_no.get(dispatch_prefix[index])
                if (
                    next_vehicle is None
                    or next_vehicle.goal.work_position_kind == "SPOTTING"
                    or target_track not in next_vehicle.goal.allowed_target_tracks
                ):
                    break
                group.append(dispatch_prefix[index])
                index += 1
            groups.append((None, group))
            continue
        group_target = _single_allowed_target(vehicle, source_track=source_track)
        if group_target is None:
            return []
        group = [dispatch_prefix[index]]
        index += 1
        while index < len(dispatch_prefix):
            next_vehicle = vehicle_by_no.get(dispatch_prefix[index])
            if next_vehicle is None:
                return []
            if _single_allowed_target(next_vehicle, source_track=source_track) != group_target:
                break
            group.append(dispatch_prefix[index])
            index += 1
        if not _prefix_group_can_detach_to_target(
            group=group,
            target_track=group_target,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        ):
            return []
        groups.append((group_target, group))
    if index == 0:
        return []
    return _assign_staging_buffers_to_order_groups(
        groups=groups,
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )


def _single_allowed_target(
    vehicle: NormalizedVehicle,
    *,
    source_track: str,
) -> str | None:
    targets = [
        target
        for target in vehicle.goal.allowed_target_tracks
        if target != source_track
    ]
    if len(targets) != 1:
        return None
    return targets[0]


def _prefix_group_can_detach_to_target(
    *,
    group: list[str],
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=group,
        existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return False
    return all(
        (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and target_track in vehicle.goal.allowed_target_tracks
        for vehicle_no in group
    )


def _first_source_index(*, source_seq: list[str], vehicle_nos: list[str]) -> int:
    return min(
        (source_seq.index(vehicle_no) for vehicle_no in vehicle_nos if vehicle_no in source_seq),
        default=10**9,
    )


def _assign_staging_buffers_to_order_groups(
    *,
    groups: list[tuple[str | None, list[str]]],
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[tuple[str, list[str]]]:
    assigned: list[tuple[str, list[str]]] = []
    forbidden_tracks = {source_track, target_track}
    for group_target, group in groups:
        if group_target is not None:
            assigned.append((group_target, group))
            forbidden_tracks.add(group_target)
            continue
        stage_track = _best_staging_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            source_track=source_track,
            block=group,
            vehicle_by_no=vehicle_by_no,
            forbidden_tracks=forbidden_tracks,
        )
        if stage_track is None:
            return []
        assigned.append((stage_track, group))
        forbidden_tracks.add(stage_track)
    return assigned


def _attach_detach_steps(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    block: list[str],
    action_source_track: str,
) -> list[HookAction] | None:
    if not block:
        return []
    if list(state.track_sequences.get(source_track, []))[: len(block)] != list(block):
        return None
    attach = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    source_node = _source_node_after_attach_for_block(
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
    )
    target_node = route_oracle.order_end_node(target_track)
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=target_node,
    )
    if path_tracks is None:
        return None
    route = route_oracle.resolve_route_for_path_tracks(
        path_tracks,
        source_node=source_node,
        target_node=target_node,
    )
    if route is None:
        return None
    detach = HookAction(
        source_track=action_source_track,
        target_track=target_track,
        vehicle_nos=list(block),
        path_tracks=path_tracks,
        action_type="DETACH",
    )
    return [attach, detach]


def _source_node_after_attach_for_block(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
) -> str | None:
    return (
        route_oracle.order_end_node(source_track)
        if len(state.track_sequences.get(source_track, [])) > len(block)
        else None
    )


def _best_staging_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
    prefer_low_route_pressure: bool = False,
    buffer_leases: tuple[BufferLease, ...] = (),
    block_is_carried: bool = False,
) -> str | None:
    ranked_tracks = _ranked_staging_tracks(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        source_track=source_track,
        block=block,
        vehicle_by_no=vehicle_by_no,
        forbidden_tracks=forbidden_tracks,
        prefer_low_route_pressure=prefer_low_route_pressure,
        buffer_leases=buffer_leases,
        block_is_carried=block_is_carried,
    )
    return ranked_tracks[0] if ranked_tracks else None


def _ranked_staging_tracks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    forbidden_tracks: set[str],
    prefer_low_route_pressure: bool = False,
    buffer_leases: tuple[BufferLease, ...] = (),
    block_is_carried: bool = False,
) -> list[str]:
    capacity_by_track = {
        info.track_name: float(info.track_distance) for info in plan_input.track_info
    }
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles
    }
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in block)
    vehicle_by_no_all = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    candidates: list[tuple[tuple[int, int, int, int, float, str], str]] = []
    for track_name in sorted((STAGING_TRACKS & set(capacity_by_track)) - forbidden_tracks):
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in state.track_sequences.get(track_name, [])
        )
        if current_length + block_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=block,
            existing_vehicle_nos=list(state.track_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        path_tracks = route_oracle.resolve_clear_path_tracks(
            source_track,
            track_name,
            occupied_track_sequences=state.track_sequences,
            source_node=(
                state.loco_node
                if block_is_carried
                else _source_node_after_attach_for_block(
                    state=state,
                    route_oracle=route_oracle,
                    source_track=source_track,
                    block=block,
                )
            ),
            target_node=route_oracle.order_end_node(track_name),
        )
        if path_tracks is None:
            continue
        lease_violation_count = _buffer_lease_violation_count(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            target_track=track_name,
            block=block,
            vehicle_by_no=vehicle_by_no_all,
            capacity_by_track=capacity_by_track,
            length_by_vehicle=length_by_vehicle,
            buffer_leases=buffer_leases,
        )
        route_pressure = 0
        if prefer_low_route_pressure:
            if block_is_carried:
                if tuple(state.loco_carry[-len(block):]) != tuple(block):
                    continue
                steps = [
                    HookAction(
                        source_track=source_track,
                        target_track=track_name,
                        vehicle_nos=list(block),
                        path_tracks=path_tracks,
                        action_type="DETACH",
                    )
                ]
            else:
                steps = _attach_detach_steps(
                    state=state,
                    route_oracle=route_oracle,
                    source_track=source_track,
                    target_track=track_name,
                    block=block,
                    action_source_track=source_track,
                )
                if steps is None:
                    continue
            compiled = replay_candidate_steps(
                plan_input=plan_input,
                state=state,
                vehicle_by_no=vehicle_by_no_all,
                steps=steps,
                route_oracle=route_oracle,
            )
            if compiled is None:
                continue
            route_pressure = compute_route_blockage_plan(
                plan_input,
                compiled.final_state,
                route_oracle,
            ).total_blockage_pressure
        candidates.append(
            (
                (
                    lease_violation_count,
                    route_pressure,
                    len(state.track_sequences.get(track_name, [])),
                    len(path_tracks),
                    capacity_by_track[track_name] - current_length,
                    track_name,
                ),
                track_name,
            )
        )
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    return [track_name for _rank, track_name in candidates]


def _buffer_lease_violation_count(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    target_track: str,
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
    buffer_leases: tuple[BufferLease, ...],
) -> int:
    if not buffer_leases:
        return 0
    block_set = set(block)
    candidate_sequences = {
        track: list(seq)
        for track, seq in state.track_sequences.items()
    }
    candidate_sequences[target_track] = [
        *block,
        *candidate_sequences.get(target_track, []),
    ]
    violations = 0
    for lease in buffer_leases:
        if block_set.issuperset(lease.vehicle_nos):
            continue
        if _lease_has_feasible_buffer_track(
            lease=lease,
            candidate_sequences=candidate_sequences,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
            length_by_vehicle=length_by_vehicle,
        ):
            continue
        violations += 1
    return violations


def _lease_has_feasible_buffer_track(
    *,
    lease: BufferLease,
    candidate_sequences: dict[str, list[str]],
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
) -> bool:
    lease_block = list(lease.vehicle_nos)
    if not lease_block:
        return True
    for track_name in sorted(STAGING_TRACKS & set(capacity_by_track)):
        current_length = sum(
            length_by_vehicle.get(vehicle_no, 0.0)
            for vehicle_no in candidate_sequences.get(track_name, [])
        )
        if current_length + lease.required_length > capacity_by_track[track_name] + 1e-9:
            continue
        preview = preview_work_positions_after_prepend(
            target_track=track_name,
            incoming_vehicle_nos=lease_block,
            existing_vehicle_nos=list(candidate_sequences.get(track_name, [])),
            vehicle_by_no=vehicle_by_no,
        )
        if not preview.valid:
            continue
        path_tracks = route_oracle.resolve_clear_path_tracks(
            lease.source_track,
            track_name,
            occupied_track_sequences=candidate_sequences,
            source_node=route_oracle.order_end_node(lease.source_track),
            target_node=route_oracle.order_end_node(track_name),
        )
        if path_tracks is not None:
            return True
    return False


def _dedup_candidates(candidates: tuple[MoveCandidate, ...]) -> tuple[MoveCandidate, ...]:
    seen: set[tuple] = set()
    result: list[MoveCandidate] = []
    for candidate in candidates:
        key = tuple(
            (
                step.action_type,
                step.source_track,
                step.target_track,
                tuple(step.vehicle_nos),
                tuple(step.path_tracks),
            )
            for step in candidate.steps
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def _candidate_steps_by_kind(candidates: tuple[MoveCandidate, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.kind] = counts.get(candidate.kind, 0) + len(candidate.steps)
    return dict(sorted(counts.items()))
