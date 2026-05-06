from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import preview_work_positions_after_prepend
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import generate_real_hook_moves
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.state import _apply_move, _state_key, _vehicle_track_lookup
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

WORK_POSITION_SEQUENCE_CANDIDATE_LIMIT = 3


@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]
    kind: str = "primitive"
    reason: str = ""


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
    if debug_stats is not None:
        debug_stats.clear()
        debug_stats.update(primitive_debug or {})
        debug_stats["primitive_move_count"] = len(primitive_moves)
    candidates = [
        MoveCandidate(steps=(move,), kind="primitive")
        for move in primitive_moves
    ]
    if master is None:
        return candidates
    if route_oracle is None:
        route_oracle = RouteOracle(master)
    sequence_candidates = generate_work_position_sequence_candidates(
        plan_input,
        state,
        master=master,
        route_oracle=route_oracle,
    )
    if debug_stats is not None:
        debug_stats["work_position_sequence_candidate_count"] = len(sequence_candidates)
    all_candidates = _dedup_candidates([*sequence_candidates, *candidates])
    if debug_stats is not None:
        debug_stats["total_moves"] = len(all_candidates)
        debug_stats["total_candidates"] = len(all_candidates)
        debug_stats["candidate_steps_total"] = sum(len(candidate.steps) for candidate in all_candidates)
        debug_stats["candidate_steps_by_kind"] = _candidate_steps_by_kind(all_candidates)
    return all_candidates


def generate_work_position_sequence_candidates(
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    *,
    master: MasterData,
    route_oracle: RouteOracle,
) -> list[MoveCandidate]:
    if state.loco_carry:
        return []
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    track_by_vehicle = _vehicle_track_lookup(state)
    spotting_jobs = _candidate_spotting_jobs(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
    )
    if not spotting_jobs:
        return []
    before_metrics = compute_structural_metrics(plan_input, state)
    if (
        before_metrics.target_sequence_defect_count <= 0
        and before_metrics.work_position_unfinished_count <= 0
    ):
        return []

    candidates: list[MoveCandidate] = []
    for vehicle, source_track, target_tracks in spotting_jobs:
        for target_track in target_tracks:
            for candidate in _build_spotting_sequence_candidates(
                plan_input=plan_input,
                state=state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle=vehicle,
                source_track=source_track,
                target_track=target_track,
                before_metrics=before_metrics,
            ):
                candidates.append(candidate)
    return _rank_work_position_sequence_candidates(
        plan_input=plan_input,
        state=state,
        candidates=_dedup_candidates(candidates),
        vehicle_by_no=vehicle_by_no,
    )


def _candidate_spotting_jobs(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
) -> list[tuple[NormalizedVehicle, str, list[str]]]:
    jobs: list[tuple[NormalizedVehicle, str, list[str]]] = []
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind != "SPOTTING":
            continue
        source_track = track_by_vehicle.get(vehicle.vehicle_no)
        if source_track is None:
            continue
        target_tracks = [
            target_track
            for target_track in vehicle.goal.allowed_target_tracks
            if target_track != source_track
        ]
        if not target_tracks:
            continue
        source_seq = list(state.track_sequences.get(source_track, []))
        if _is_nonleading_spotting_vehicle(
            vehicle=vehicle,
            source_seq=source_seq,
            vehicle_by_no=vehicle_by_no,
        ):
            continue
        jobs.append((vehicle, source_track, target_tracks))
    return jobs


def _rank_work_position_sequence_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    candidates: list[MoveCandidate],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[MoveCandidate]:
    ranked: list[tuple[tuple[Any, ...], MoveCandidate]] = []
    for candidate in candidates:
        next_state = _apply_candidate_for_ranking(
            candidate=candidate,
            state=state,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        if next_state is None:
            continue
        metrics = compute_structural_metrics(plan_input, next_state)
        ranked.append(
            (
                (
                    metrics.target_sequence_defect_count,
                    metrics.work_position_unfinished_count,
                    metrics.unfinished_count,
                    metrics.staging_debt_count,
                    -_max_spotting_goal_insert_size(candidate, vehicle_by_no),
                    len(candidate.steps),
                    _candidate_signature(candidate),
                ),
                candidate,
            )
        )
    ranked.sort(key=lambda item: item[0])
    return [
        candidate
        for _key, candidate in ranked[:WORK_POSITION_SEQUENCE_CANDIDATE_LIMIT]
    ]


def _apply_candidate_for_ranking(
    *,
    candidate: MoveCandidate,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> ReplayState | None:
    next_state = state
    for step in candidate.steps:
        try:
            next_state = _apply_move(
                state=next_state,
                move=step,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
    return next_state


def _max_spotting_goal_insert_size(
    candidate: MoveCandidate,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    max_size = 0
    for step in candidate.steps:
        if step.action_type != "DETACH":
            continue
        inserted = 0
        for vehicle_no in step.vehicle_nos:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None:
                continue
            if (
                vehicle.goal.work_position_kind == "SPOTTING"
                and step.target_track in vehicle.goal.allowed_target_tracks
            ):
                inserted += 1
        max_size = max(max_size, inserted)
    return max_size


def _candidate_signature(candidate: MoveCandidate) -> tuple[Any, ...]:
    return tuple(
        (
            step.action_type,
            step.source_track,
            step.target_track,
            tuple(step.vehicle_nos),
            tuple(step.path_tracks),
        )
        for step in candidate.steps
    )


def _build_spotting_sequence_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    vehicle: NormalizedVehicle,
    source_track: str,
    target_track: str,
    before_metrics: Any,
) -> list[MoveCandidate]:
    target_seq = list(state.track_sequences.get(target_track, []))
    source_seq = list(state.track_sequences.get(source_track, []))
    if vehicle.vehicle_no not in source_seq:
        return []
    target_index = source_seq.index(vehicle.vehicle_no)
    full_work_block = _contiguous_spotting_block(
        source_seq=source_seq,
        start_index=target_index,
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    if not full_work_block:
        return []
    candidates: list[MoveCandidate] = []
    single_candidate = _build_single_spotting_sequence_candidate(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
        before_metrics=before_metrics,
        target_seq=target_seq,
        source_seq=source_seq,
        target_index=target_index,
        vehicle_no=vehicle.vehicle_no,
    )
    if single_candidate is not None:
        candidates.append(single_candidate)
    if len(full_work_block) > 1 and full_work_block[0] == vehicle.vehicle_no:
        block_candidate = _build_spotting_sequence_candidate_for_block(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            before_metrics=before_metrics,
            target_seq=target_seq,
            source_seq=source_seq,
            target_index=target_index,
            work_block=full_work_block,
        )
        if block_candidate is not None:
            candidates.append(block_candidate)
    return candidates


def _build_single_spotting_sequence_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    before_metrics: Any,
    target_seq: list[str],
    source_seq: list[str],
    target_index: int,
    vehicle_no: str,
) -> MoveCandidate | None:
    vehicle = vehicle_by_no[vehicle_no]
    candidate_clear_counts = _spotting_clear_counts(
        vehicle=vehicle,
        target_track=target_track,
        target_seq=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    for clear_count in candidate_clear_counts:
        clear_block = target_seq[:clear_count]
        staged_options = _stage_prefix_options(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=target_track,
            block=clear_block,
            target_hints=(target_track, source_track),
        )
        for staged_state, clear_steps in staged_options:
            source_prefix = source_seq[: target_index + 1]
            prefix_state, prefix_steps = _attach_prefix(
                plan_input=plan_input,
                state=staged_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                prefix_block=source_prefix,
            )
            if prefix_state is None:
                continue
            target_state, target_steps = _detach_carried_tail_to_track(
                plan_input=plan_input,
                state=prefix_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle_nos=[vehicle_no],
                target_track=target_track,
            )
            if target_state is None:
                continue
            restored_state, source_dispatch_steps = _dispatch_carried_goal_blocks(
                plan_input=plan_input,
                state=target_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
            )
            if restored_state is None:
                continue
            restore_steps: list[HookAction] = []
            if clear_block:
                clear_staging_track = clear_steps[-1].target_track
                restored_state, clear_restore_steps = _dispatch_staged_block_to_goals(
                    plan_input=plan_input,
                    state=restored_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=clear_staging_track,
                    block=clear_block,
                    target_track=target_track,
                )
                if restored_state is None:
                    continue
                restore_steps.extend(clear_restore_steps)
            steps = [
                *clear_steps,
                *prefix_steps,
                *target_steps,
                *source_dispatch_steps,
                *restore_steps,
            ]
            if not steps or restored_state is None or restored_state.loco_carry:
                continue
            after_metrics = compute_structural_metrics(plan_input, restored_state)
            if not _sequence_candidate_improves(
                before_metrics=before_metrics,
                after_metrics=after_metrics,
            ):
                continue
            if not _work_block_satisfied(
                work_block=[vehicle_no],
                target_track=target_track,
                state=restored_state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            ):
                continue
            return MoveCandidate(
                steps=tuple(steps),
                kind="work_position_sequence",
                reason=f"{target_track} SPOTTING sequence repair",
            )
    return None


def _is_nonleading_spotting_vehicle(
    *,
    vehicle: NormalizedVehicle,
    source_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    try:
        index = source_seq.index(vehicle.vehicle_no)
    except ValueError:
        return False
    if index <= 0:
        return False
    previous = vehicle_by_no.get(source_seq[index - 1])
    return (
        previous is not None
        and previous.goal.work_position_kind == "SPOTTING"
        and previous.goal.target_track == vehicle.goal.target_track
    )


def _build_spotting_sequence_candidate_for_block(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    before_metrics: Any,
    target_seq: list[str],
    source_seq: list[str],
    target_index: int,
    work_block: list[str],
) -> MoveCandidate | None:
    candidate_clear_counts = _spotting_block_clear_counts(
        work_block=work_block,
        target_track=target_track,
        target_seq=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    for clear_count in candidate_clear_counts:
        clear_block = target_seq[:clear_count]
        staged_options = _stage_prefix_options(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=target_track,
            block=clear_block,
            target_hints=(target_track, source_track),
        )
        for staged_state, clear_steps in staged_options:
            source_ready_options = _clear_source_prefix_before_work_block(
                plan_input=plan_input,
                state=staged_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                source_blockers=source_seq[:target_index],
                target_track=target_track,
            )
            if not source_ready_options:
                continue
            for (
                source_ready_state,
                source_ready_steps,
                order_buffer_block,
                order_buffer_track,
            ) in source_ready_options:
                work_attach_state, work_attach_steps = _attach_prefix(
                    plan_input=plan_input,
                    state=source_ready_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    prefix_block=work_block,
                )
                if work_attach_state is None:
                    continue
                target_state, target_steps = _detach_carried_tail_to_track(
                    plan_input=plan_input,
                    state=work_attach_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    vehicle_nos=work_block,
                    target_track=target_track,
                )
                if target_state is None:
                    continue
                restored_state = target_state
                restore_steps: list[HookAction] = []
                if order_buffer_block and order_buffer_track is not None:
                    restored_state, order_restore_steps = _restore_staged_block(
                        plan_input=plan_input,
                        state=restored_state,
                        route_oracle=route_oracle,
                        vehicle_by_no=vehicle_by_no,
                        source_track=order_buffer_track,
                        block=order_buffer_block,
                        target_track=target_track,
                    )
                    if restored_state is None:
                        continue
                    restore_steps.extend(order_restore_steps)
                if clear_block:
                    clear_staging_track = clear_steps[-1].target_track
                    restored_state, clear_restore_steps = _dispatch_staged_block_to_goals(
                        plan_input=plan_input,
                        state=restored_state,
                        route_oracle=route_oracle,
                        vehicle_by_no=vehicle_by_no,
                        source_track=clear_staging_track,
                        block=clear_block,
                        target_track=target_track,
                    )
                    if restored_state is None:
                        continue
                    restore_steps.extend(clear_restore_steps)
                steps = [
                    *clear_steps,
                    *source_ready_steps,
                    *work_attach_steps,
                    *target_steps,
                    *restore_steps,
                ]
                if not steps or restored_state is None or restored_state.loco_carry:
                    continue
                after_metrics = compute_structural_metrics(plan_input, restored_state)
                if not _sequence_candidate_improves(
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                ):
                    continue
                if not _work_block_satisfied(
                    work_block=work_block,
                    target_track=target_track,
                    state=restored_state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                ):
                    continue
                return MoveCandidate(
                    steps=tuple(steps),
                    kind="work_position_sequence",
                    reason=f"{target_track} SPOTTING sequence repair",
                )
    return None


def _contiguous_spotting_block(
    *,
    source_seq: list[str],
    start_index: int,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in source_seq[start_index:]:
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.work_position_kind != "SPOTTING"
            or vehicle.goal.target_track != target_track
        ):
            break
        block.append(vehicle_no)
    return block


def _work_block_satisfied(
    *,
    work_block: list[str],
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    for vehicle_no in work_block:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if not goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        ):
            return False
    return True


def _spotting_clear_counts(
    *,
    vehicle: NormalizedVehicle,
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[int]:
    counts: list[int] = []
    for clear_count in range(len(target_seq) + 1):
        clear_block = target_seq[:clear_count]
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=[*clear_block, vehicle.vehicle_no],
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        evaluation = preview.evaluations.get(vehicle.vehicle_no)
        if preview.valid and evaluation is not None and evaluation.satisfied_now:
            counts.append(clear_count)
    return sorted(counts)


def _spotting_block_clear_counts(
    *,
    work_block: list[str],
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[int]:
    counts: list[int] = []
    for clear_count in range(len(target_seq) + 1):
        clear_block = target_seq[:clear_count]
        preview = preview_work_positions_after_prepend(
            target_track=target_track,
            incoming_vehicle_nos=[*clear_block, *work_block],
            existing_vehicle_nos=target_seq[clear_count:],
            vehicle_by_no=vehicle_by_no,
        )
        if preview.valid and all(
            (evaluation := preview.evaluations.get(vehicle_no)) is not None
            and evaluation.satisfied_now
            for vehicle_no in work_block
        ):
            counts.append(clear_count)
    return sorted(counts)


def _clear_source_prefix_before_work_block(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    source_blockers: list[str],
    target_track: str,
) -> list[tuple[ReplayState, list[HookAction], list[str], str | None]]:
    if not source_blockers:
        return [(state, [], [], None)]
    attached_state, attach_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=source_blockers,
    )
    if attached_state is None:
        return []
    current_state = attached_state
    steps = list(attach_steps)
    order_buffer_block = _target_track_tail_block(
        carry=list(current_state.loco_carry),
        target_track=target_track,
        vehicle_by_no=vehicle_by_no,
    )
    order_buffer_track: str | None = None
    if order_buffer_block:
        staged_options = _stage_carried_tail_options(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=order_buffer_block,
            source_track=source_track,
            target_hints=(target_track,),
        )
        if not staged_options:
            return []
        current_state, staged_steps, order_buffer_track = staged_options[0]
        steps.extend(staged_steps)
    current_state, dispatch_steps = _dispatch_carried_goal_blocks(
        plan_input=plan_input,
        state=current_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
    )
    if current_state is None:
        return []
    steps.extend(dispatch_steps)
    return [(current_state, steps, order_buffer_block, order_buffer_track)]


def _target_track_tail_block(
    *,
    carry: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    block: list[str] = []
    for vehicle_no in reversed(carry):
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            vehicle is None
            or vehicle.goal.target_track != target_track
            or vehicle.goal.work_position_kind != "FREE"
        ):
            break
        block.append(vehicle_no)
    block.reverse()
    return block


def _stage_prefix_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_hints: tuple[str, ...],
) -> list[tuple[ReplayState, list[HookAction]]]:
    if not block:
        return [(state, [])]
    attached_state, attach_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=block,
    )
    if attached_state is None:
        return []
    options = []
    for staged_state, staging_steps, _staging_track in _stage_carried_tail_options(
        plan_input=plan_input,
        state=attached_state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=block,
        source_track=source_track,
        target_hints=target_hints,
    ):
        options.append((staged_state, [*attach_steps, *staging_steps]))
    return options


def _attach_prefix(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    prefix_block: list[str],
) -> tuple[ReplayState | None, list[HookAction]]:
    if state.loco_carry:
        return None, []
    if not prefix_block:
        return state, []
    if list(state.track_sequences.get(source_track, []))[: len(prefix_block)] != list(prefix_block):
        return None, []
    if validate_hook_vehicle_group([vehicle_by_no[vno] for vno in prefix_block]):
        return None, []
    access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    )
    if not access.is_valid:
        return None, []
    move = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(prefix_block),
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    try:
        next_state = _apply_move(
            state=state,
            move=move,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None, []
    return next_state, [move]


def _stage_carried_tail_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    block: list[str],
    source_track: str,
    target_hints: tuple[str, ...],
) -> list[tuple[ReplayState, list[HookAction], str]]:
    if not block or tuple(state.loco_carry[-len(block):]) != tuple(block):
        return []
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    capacity_by_track = _effective_capacity_by_track(plan_input)
    targets = _staging_targets(
        state=state,
        plan_input=plan_input,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        block=block,
        target_hints=target_hints,
    )
    options: list[tuple[tuple[int, int, int, str], ReplayState, list[HookAction], str]] = []
    for target_track in targets:
        move = _build_detach_move(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            length_by_vehicle=length_by_vehicle,
            capacity_by_track=capacity_by_track,
            vehicle_nos=block,
            target_track=target_track,
            allow_goal_overflow=False,
        )
        if move is None:
            continue
        try:
            next_state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            continue
        if _state_key(next_state, plan_input) == _state_key(state, plan_input):
            continue
        options.append(
            (
                (
                    0 if target_track in STAGING_TRACKS else 1,
                    compute_structural_metrics(plan_input, next_state).target_sequence_defect_count,
                    len(move.path_tracks),
                    target_track,
                ),
                next_state,
                [move],
                target_track,
            )
        )
    options.sort(key=lambda item: item[0])
    return [(state, steps, target_track) for _score, state, steps, target_track in options[:3]]


def _restore_staged_block(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_track: str,
) -> tuple[ReplayState | None, list[HookAction]]:
    attached_state, attach_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=block,
    )
    if attached_state is None:
        return None, []
    target_state, detach_steps = _detach_carried_tail_to_track(
        plan_input=plan_input,
        state=attached_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        vehicle_nos=block,
        target_track=target_track,
    )
    if target_state is None:
        return None, []
    return target_state, [*attach_steps, *detach_steps]


def _dispatch_staged_block_to_goals(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_track: str,
) -> tuple[ReplayState | None, list[HookAction]]:
    attached_state, attach_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=block,
    )
    if attached_state is None:
        return None, []
    dispatched_state, dispatch_steps = _dispatch_carried_goal_blocks(
        plan_input=plan_input,
        state=attached_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        preferred_target_track=target_track,
    )
    if dispatched_state is None:
        return None, []
    return dispatched_state, [*attach_steps, *dispatch_steps]


def _dispatch_carried_goal_blocks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    preferred_target_track: str | None = None,
) -> tuple[ReplayState | None, list[HookAction]]:
    current_state = state
    steps: list[HookAction] = []
    while current_state.loco_carry:
        dispatch = _best_carried_tail_goal_dispatch(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            preferred_target_track=preferred_target_track,
        )
        if dispatch is None:
            return None, []
        current_state, dispatch_steps = dispatch
        steps.extend(dispatch_steps)
    return current_state, steps


def _best_carried_tail_goal_dispatch(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    preferred_target_track: str | None,
) -> tuple[ReplayState, list[HookAction]] | None:
    carry = list(state.loco_carry)
    tail_target = _dispatch_target_for_vehicle(
        carry[-1],
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        preferred_target_track=preferred_target_track,
    )
    if tail_target is None:
        return None
    start = len(carry) - 1
    while start > 0:
        previous_target = _dispatch_target_for_vehicle(
            carry[start - 1],
            state=state,
            plan_input=plan_input,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            preferred_target_track=preferred_target_track,
        )
        if previous_target != tail_target:
            break
        start -= 1
    for block_start in range(start, len(carry)):
        block = carry[block_start:]
        next_state, steps = _detach_carried_tail_to_track(
            plan_input=plan_input,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=block,
            target_track=tail_target,
        )
        if next_state is not None:
            return next_state, steps
    return None


def _dispatch_target_for_vehicle(
    vehicle_no: str,
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    preferred_target_track: str | None,
) -> str | None:
    vehicle = vehicle_by_no.get(vehicle_no)
    if vehicle is None:
        return None
    if vehicle.need_weigh and vehicle_no not in state.weighed_vehicle_nos:
        return "机库"
    allowed_tracks = goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
    )
    if preferred_target_track is not None and preferred_target_track in allowed_tracks:
        return preferred_target_track
    if vehicle.goal.target_track in allowed_tracks:
        return vehicle.goal.target_track
    return allowed_tracks[0] if allowed_tracks else None


def _detach_carried_tail_to_track(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    vehicle_nos: list[str],
    target_track: str,
) -> tuple[ReplayState | None, list[HookAction]]:
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in plan_input.vehicles}
    move = _build_detach_move(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        length_by_vehicle=length_by_vehicle,
        capacity_by_track=_effective_capacity_by_track(plan_input),
        vehicle_nos=vehicle_nos,
        target_track=target_track,
        allow_goal_overflow=True,
    )
    if move is None:
        return None, []
    try:
        next_state = _apply_move(
            state=state,
            move=move,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None, []
    return next_state, [move]


def _build_detach_move(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    length_by_vehicle: dict[str, float],
    capacity_by_track: dict[str, float],
    vehicle_nos: list[str],
    target_track: str,
    allow_goal_overflow: bool,
) -> HookAction | None:
    if not vehicle_nos:
        return None
    if tuple(state.loco_carry[-len(vehicle_nos):]) != tuple(vehicle_nos):
        return None
    vehicles = [vehicle_by_no.get(vehicle_no) for vehicle_no in vehicle_nos]
    if any(vehicle is None for vehicle in vehicles):
        return None
    concrete_vehicles = [vehicle for vehicle in vehicles if vehicle is not None]
    if validate_hook_vehicle_group(concrete_vehicles):
        return None
    if not allow_goal_overflow and not _fits_capacity(
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        state=state,
        capacity_by_track=capacity_by_track,
        length_by_vehicle=length_by_vehicle,
    ):
        return None
    if allow_goal_overflow and not _is_goal_detach(
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        state=state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ) and not _fits_capacity(
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        state=state,
        capacity_by_track=capacity_by_track,
        length_by_vehicle=length_by_vehicle,
    ):
        return None
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=list(vehicle_nos),
        existing_vehicle_nos=list(state.track_sequences.get(target_track, [])),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return None
    route_train_length_m = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in state.loco_carry)
    source_node = (
        state.loco_node
        if len(state.track_sequences.get(state.loco_track_name, [])) > 0
        else None
    )
    target_node = route_oracle.order_end_node(target_track)
    path_tracks = route_oracle.resolve_clear_path_tracks(
        state.loco_track_name,
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
    route_result = route_oracle.validate_path(
        source_track=state.loco_track_name,
        target_track=target_track,
        path_tracks=path_tracks,
        train_length_m=route_train_length_m,
        occupied_track_sequences=state.track_sequences,
        expected_path_tracks=path_tracks,
        route=route,
        source_node=source_node,
        target_node=target_node,
    )
    if not route_result.is_valid:
        return None
    return HookAction(
        source_track=state.loco_track_name,
        target_track=target_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=path_tracks,
        action_type="DETACH",
    )


def _staging_targets(
    *,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_hints: tuple[str, ...],
) -> list[str]:
    candidates: list[tuple[tuple[int, int, int, float, str], str]] = []
    for info in plan_input.track_info:
        target_track = info.track_name
        if target_track == source_track:
            continue
        track = master.tracks.get(target_track)
        if track is None or not track.allow_parking:
            continue
        type_priority = 0 if target_track in STAGING_TRACKS else 1
        if type_priority > 0 and track.track_type != "STORAGE":
            continue
        if any(_is_hard_goal_track(target_track, vehicle_by_no[vehicle_no]) for vehicle_no in block):
            continue
        distance = _route_distance(route_oracle, source_track, target_track)
        if distance is None:
            continue
        follow_distances = [
            follow
            for hint in target_hints
            if hint != target_track
            for follow in [_route_distance(route_oracle, target_track, hint)]
            if follow is not None
        ]
        combined_distance = distance + (min(follow_distances) if follow_distances else 0.0)
        candidates.append(
            (
                (
                    type_priority,
                    _corridor_penalty(
                        route_oracle=route_oracle,
                        source_track=source_track,
                        target_track=target_track,
                        target_hints=target_hints,
                    ),
                    len(state.track_sequences.get(target_track, [])),
                    combined_distance,
                    target_track,
                ),
                target_track,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return [target for _score, target in candidates]


def _sequence_candidate_improves(*, before_metrics: Any, after_metrics: Any) -> bool:
    return (
        after_metrics.target_sequence_defect_count < before_metrics.target_sequence_defect_count
        or after_metrics.work_position_unfinished_count < before_metrics.work_position_unfinished_count
    )


def _effective_capacity_by_track(plan_input: NormalizedPlanInput) -> dict[str, float]:
    capacity_by_track = {info.track_name: info.track_distance for info in plan_input.track_info}
    initial_occupation: dict[str, float] = {}
    for vehicle in plan_input.vehicles:
        initial_occupation[vehicle.current_track] = (
            initial_occupation.get(vehicle.current_track, 0.0)
            + vehicle.vehicle_length
        )
    return {
        track: max(capacity, initial_occupation.get(track, 0.0))
        for track, capacity in capacity_by_track.items()
    }


def _fits_capacity(
    *,
    target_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    capacity_by_track: dict[str, float],
    length_by_vehicle: dict[str, float],
) -> bool:
    capacity = capacity_by_track.get(target_track)
    if capacity is None:
        return False
    current_length = sum(
        length_by_vehicle.get(vehicle_no, 0.0)
        for vehicle_no in state.track_sequences.get(target_track, [])
    )
    block_length = sum(length_by_vehicle.get(vehicle_no, 0.0) for vehicle_no in vehicle_nos)
    return current_length + block_length <= capacity + 1e-9


def _is_goal_detach(
    *,
    target_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    projected_state = state.model_copy(
        update={
            "track_sequences": {
                **state.track_sequences,
                target_track: list(vehicle_nos) + list(state.track_sequences.get(target_track, [])),
            }
        }
    )
    for vehicle_no in vehicle_nos:
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            return False
        if target_track not in goal_effective_allowed_tracks(
            vehicle,
            state=state,
            plan_input=plan_input,
        ):
            return False
        if vehicle.goal.work_position_kind is not None:
            if not goal_is_satisfied(
                vehicle,
                track_name=target_track,
                state=projected_state,
                plan_input=plan_input,
            ):
                return False
    return True


def _is_hard_goal_track(track_name: str, vehicle: NormalizedVehicle) -> bool:
    goal = vehicle.goal
    if track_name in goal.preferred_target_tracks:
        return True
    if goal.target_mode == "SNAPSHOT" and track_name in goal.fallback_target_tracks:
        return False
    return track_name in goal.allowed_target_tracks


def _route_distance(route_oracle: RouteOracle, source_track: str, target_track: str) -> float | None:
    route = route_oracle.resolve_route(source_track, target_track)
    return route.total_length_m if route is not None else None


def _corridor_penalty(
    *,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
    target_hints: tuple[str, ...],
) -> int:
    for hint in target_hints:
        if hint == target_track:
            continue
        path_tracks = route_oracle.resolve_path_tracks(source_track, hint)
        if path_tracks is not None and target_track in path_tracks[1:-1]:
            return 1
    return 0


def _dedup_candidates(candidates: list[MoveCandidate]) -> list[MoveCandidate]:
    seen: set[tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], str], ...]] = set()
    result: list[MoveCandidate] = []
    for candidate in candidates:
        key = tuple(
            (
                step.source_track,
                step.target_track,
                tuple(step.vehicle_nos),
                tuple(step.path_tracks),
                step.action_type,
            )
            for step in candidate.steps
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _candidate_steps_by_kind(candidates: list[MoveCandidate]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for candidate in candidates:
        totals[candidate.kind] = totals.get(candidate.kind, 0) + len(candidate.steps)
    return totals
