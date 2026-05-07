from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import (
    allowed_spotting_south_ranks,
    preview_work_positions_after_prepend,
)
from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import generate_real_hook_moves
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.state import _apply_move, _state_key, _vehicle_track_lookup
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState

WORK_POSITION_SEQUENCE_CANDIDATE_LIMIT = 8
SOURCE_ACCESS_RELEASE_MAX_ROUNDS = 6
SOURCE_ACCESS_OPTION_LIMIT = 8
SOURCE_ACCESS_FRONTIER_LIMIT = 12


@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]
    kind: str = "primitive"
    reason: str = ""
    focus_tracks: tuple[str, ...] = ()
    buffer_blocks: tuple[tuple[str, ...], ...] = ()
    structural_reserve: bool = False


@dataclass(frozen=True)
class WorkPositionDebtPlan:
    target_track: str
    source_track: str


@dataclass(frozen=True)
class _PreparedSourceAccess:
    state: ReplayState
    steps: tuple[HookAction, ...]


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
        debug_stats["candidate_focus_tracks_by_kind"] = _candidate_focus_tracks_by_kind(all_candidates)
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
    has_internal_spotting_debt = _has_internal_spotting_sequence_debt(
        plan_input=plan_input,
        state=state,
        track_by_vehicle=track_by_vehicle,
    )
    if not spotting_jobs and not has_internal_spotting_debt:
        return []
    before_metrics = compute_structural_metrics(plan_input, state)
    if (
        before_metrics.target_sequence_defect_count <= 0
        and before_metrics.work_position_unfinished_count <= 0
    ):
        return []

    candidates: list[MoveCandidate] = []
    debt_plans = _work_position_debt_plans(
        plan_input=plan_input,
        state=state,
        vehicle_by_no=vehicle_by_no,
        track_by_vehicle=track_by_vehicle,
        before_metrics=before_metrics,
    )
    for debt_plan in debt_plans:
        for candidate in _build_rank_window_sequence_candidates(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            debt_plan=debt_plan,
            before_metrics=before_metrics,
        ):
            candidates.append(candidate)
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


def _has_internal_spotting_sequence_debt(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_by_vehicle: dict[str, str],
) -> bool:
    for vehicle in plan_input.vehicles:
        if vehicle.goal.work_position_kind != "SPOTTING":
            continue
        current_track = track_by_vehicle.get(vehicle.vehicle_no)
        if current_track not in vehicle.goal.allowed_target_tracks:
            continue
        if not goal_is_satisfied(
            vehicle,
            track_name=current_track,
            state=state,
            plan_input=plan_input,
        ):
            return True
    return False


def _work_position_debt_plans(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_by_vehicle: dict[str, str],
    before_metrics: Any,
) -> list[WorkPositionDebtPlan]:
    plans: list[WorkPositionDebtPlan] = []
    for target_track, defect_count in before_metrics.target_sequence_defect_by_track.items():
        if defect_count <= 0 or not allowed_spotting_south_ranks(target_track):
            continue
        external_source_tracks = _spotting_source_tracks_for_target(
            plan_input=plan_input,
            state=state,
            track_by_vehicle=track_by_vehicle,
            target_track=target_track,
        )
        for source_track in external_source_tracks:
            plans.append(
                WorkPositionDebtPlan(
                    target_track=target_track,
                    source_track=source_track,
                )
            )
        if not external_source_tracks and any(
            _is_spotting_goal(
                vehicle_no,
                target_track=target_track,
                vehicle_by_no=vehicle_by_no,
            )
            and not _is_satisfied_spotting_on_track(
                vehicle_no=vehicle_no,
                target_track=target_track,
                state=state,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            for vehicle_no in state.track_sequences.get(target_track, [])
        ):
            plans.append(
                WorkPositionDebtPlan(
                    target_track=target_track,
                    source_track=target_track,
                )
            )
    return plans


def _spotting_source_tracks_for_target(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    track_by_vehicle: dict[str, str],
    target_track: str,
) -> list[str]:
    tracks: list[str] = []
    seen: set[str] = set()
    for vehicle in plan_input.vehicles:
        if (
            vehicle.goal.work_position_kind != "SPOTTING"
            or target_track not in vehicle.goal.allowed_target_tracks
        ):
            continue
        source_track = track_by_vehicle.get(vehicle.vehicle_no)
        if (
            source_track is None
            or source_track == target_track
            or source_track in seen
            or vehicle.vehicle_no not in state.track_sequences.get(source_track, [])
        ):
            continue
        tracks.append(source_track)
        seen.add(source_track)
    return tracks


def _is_satisfied_spotting_on_track(
    *,
    vehicle_no: str,
    target_track: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    vehicle = vehicle_by_no.get(vehicle_no)
    return (
        vehicle is not None
        and vehicle.goal.work_position_kind == "SPOTTING"
        and target_track in vehicle.goal.allowed_target_tracks
        and goal_is_satisfied(
            vehicle,
            track_name=target_track,
            state=state,
            plan_input=plan_input,
        )
    )


def _rank_work_position_sequence_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    candidates: list[MoveCandidate],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[MoveCandidate]:
    ranked: list[tuple[tuple[Any, ...], MoveCandidate]] = []
    before_staging_debt = compute_structural_metrics(plan_input, state).staging_debt_count
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
        dependency_blockers = _candidate_dependency_blocker_count(candidate, vehicle_by_no)
        buffer_debt_budget = _candidate_buffer_vehicle_count(candidate)
        reserve_candidate = (
            candidate.kind == "work_position_sequence"
            and dependency_blockers == 0
            and metrics.staging_debt_count <= before_staging_debt + buffer_debt_budget
        )
        if candidate.structural_reserve != reserve_candidate:
            candidate = MoveCandidate(
                steps=candidate.steps,
                kind=candidate.kind,
                reason=candidate.reason,
                focus_tracks=candidate.focus_tracks,
                buffer_blocks=candidate.buffer_blocks,
                structural_reserve=reserve_candidate,
            )
        ranked.append(
            (
                (
                    metrics.target_sequence_defect_count,
                    dependency_blockers,
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


def _candidate_dependency_blocker_count(
    candidate: MoveCandidate,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    blocked_tracks = set(candidate.focus_tracks)
    count = 0
    for block in candidate.buffer_blocks:
        for vehicle_no in block:
            vehicle = vehicle_by_no.get(vehicle_no)
            if vehicle is None or vehicle.goal.work_position_kind != "SPOTTING":
                continue
            if vehicle.goal.target_track not in blocked_tracks:
                count += 1
    return count


def _candidate_buffer_vehicle_count(candidate: MoveCandidate) -> int:
    return len(
        {
            vehicle_no
            for block in candidate.buffer_blocks
            for vehicle_no in block
        }
    )


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


def _build_rank_window_sequence_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt_plan: WorkPositionDebtPlan,
    before_metrics: Any,
) -> list[MoveCandidate]:
    target_track = debt_plan.target_track
    source_track = debt_plan.source_track
    if source_track == target_track:
        return _build_internal_rank_window_sequence_candidates(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=target_track,
            before_metrics=before_metrics,
        )
    target_seq = list(state.track_sequences.get(target_track, []))
    source_seq = list(state.track_sequences.get(source_track, []))
    if not source_seq:
        return []
    groups = _rank_window_source_groups(
        source_seq=source_seq,
        target_track=target_track,
        target_seq=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    candidates: list[MoveCandidate] = []
    for group in groups:
        for clear_count in _rank_window_clear_counts(
            target_track=target_track,
            target_seq=target_seq,
            work_block=group,
            vehicle_by_no=vehicle_by_no,
        ):
            candidate = _build_rank_window_sequence_candidate(
                plan_input=plan_input,
                state=state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                debt_plan=debt_plan,
                source_seq=source_seq,
                work_block=group,
                clear_count=clear_count,
                before_metrics=before_metrics,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _build_internal_rank_window_sequence_candidates(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    target_track: str,
    before_metrics: Any,
) -> list[MoveCandidate]:
    target_seq = list(state.track_sequences.get(target_track, []))
    clear_counts = _internal_rank_window_clear_counts(
        target_track=target_track,
        target_seq=target_seq,
        vehicle_by_no=vehicle_by_no,
    )
    candidates: list[MoveCandidate] = []
    for clear_count in clear_counts:
        clear_block = target_seq[:clear_count]
        staged_options = _stage_prefix_options(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=target_track,
            block=clear_block,
            target_hints=(target_track,),
        )
        for staged_state, clear_steps in staged_options:
            if not clear_steps:
                continue
            staging_track = clear_steps[-1].target_track
            settlement_options = _target_prefix_settlement_options(
                plan_input=plan_input,
                state=staged_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=staging_track,
                block=clear_block,
                target_track=target_track,
            )
            for settled_state, settle_steps, clear_buffer_block in settlement_options:
                if settled_state.loco_carry:
                    continue
                after_metrics = compute_structural_metrics(plan_input, settled_state)
                candidate = MoveCandidate(
                    steps=tuple([*clear_steps, *settle_steps]),
                    kind="work_position_sequence",
                    reason=f"{target_track} internal rank-window sequence repair",
                    focus_tracks=(target_track,),
                    buffer_blocks=((clear_buffer_block,) if clear_buffer_block else ()),
                )
                if _candidate_dependency_blocker_count(candidate, vehicle_by_no) > 0:
                    continue
                if after_metrics.staging_debt_count > before_metrics.staging_debt_count:
                    continue
                if not _rank_window_candidate_improves(
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    target_track=target_track,
                ):
                    continue
                if not _work_block_satisfied(
                    work_block=_spotting_goals_in_block(
                        block=clear_block,
                        target_track=target_track,
                        vehicle_by_no=vehicle_by_no,
                    ),
                    target_track=target_track,
                    state=settled_state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                ):
                    continue
                candidates.append(candidate)
    return candidates


def _internal_rank_window_clear_counts(
    *,
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[int]:
    counts: list[int] = []
    for clear_count in range(1, len(target_seq) + 1):
        clear_block = target_seq[:clear_count]
        spotting_block = _spotting_goals_in_block(
            block=clear_block,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
        if not spotting_block:
            continue
        if all(
            _can_satisfy_rank_window_after_prepend(
                target_track=target_track,
                target_seq=target_seq[clear_count:],
                work_block=[vehicle_no],
                vehicle_by_no=vehicle_by_no,
            )
            for vehicle_no in spotting_block
        ):
            counts.append(clear_count)
    return sorted(counts, key=lambda count: (count, target_seq[:count]))


def _spotting_goals_in_block(
    *,
    block: list[str],
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    return [
        vehicle_no
        for vehicle_no in block
        if _is_spotting_goal(
            vehicle_no,
            target_track=target_track,
            vehicle_by_no=vehicle_by_no,
        )
    ]


def _rank_window_source_groups(
    *,
    source_seq: list[str],
    target_track: str,
    target_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[list[str]]:
    groups: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    spotting_indices = [
        index
        for index, vehicle_no in enumerate(source_seq)
        if _is_spotting_goal(vehicle_no, target_track=target_track, vehicle_by_no=vehicle_by_no)
    ]
    if not spotting_indices:
        return []
    max_window_size = max(allowed_spotting_south_ranks(target_track), default=0)
    for start_pos, start_index in enumerate(spotting_indices):
        group: list[str] = []
        for index in spotting_indices[start_pos:]:
            group.append(source_seq[index])
            if len(group) > max_window_size:
                break
            key = tuple(group)
            if key in seen:
                continue
            groups.append(list(group))
            seen.add(key)
    groups.sort(
        key=lambda group: (
            -len(group),
            source_seq.index(group[0]),
            group,
        )
    )
    return groups[:4]


def _rank_window_clear_counts(
    *,
    target_track: str,
    target_seq: list[str],
    work_block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[int]:
    counts = [
        clear_count
        for clear_count in range(len(target_seq) + 1)
        if _can_satisfy_rank_window_after_prepend(
            target_track=target_track,
            target_seq=target_seq[clear_count:],
            work_block=work_block,
            vehicle_by_no=vehicle_by_no,
        )
    ]
    counts.sort(key=lambda clear_count: (abs(clear_count - len(work_block)), clear_count))
    return counts


def _can_satisfy_rank_window_after_prepend(
    *,
    target_track: str,
    target_seq: list[str],
    work_block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    preview = preview_work_positions_after_prepend(
        target_track=target_track,
        incoming_vehicle_nos=list(work_block),
        existing_vehicle_nos=list(target_seq),
        vehicle_by_no=vehicle_by_no,
    )
    if not preview.valid:
        return False
    return all(
        (evaluation := preview.evaluations.get(vehicle_no)) is not None
        and evaluation.satisfied_now
        for vehicle_no in work_block
    )


def _is_spotting_goal(
    vehicle_no: str,
    *,
    target_track: str,
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> bool:
    vehicle = vehicle_by_no.get(vehicle_no)
    return (
        vehicle is not None
        and vehicle.goal.work_position_kind == "SPOTTING"
        and target_track in vehicle.goal.allowed_target_tracks
    )


def _build_rank_window_sequence_candidate(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    debt_plan: WorkPositionDebtPlan,
    source_seq: list[str],
    work_block: list[str],
    clear_count: int,
    before_metrics: Any,
) -> MoveCandidate | None:
    target_track = debt_plan.target_track
    source_track = debt_plan.source_track
    target_seq = list(state.track_sequences.get(target_track, []))
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
    if not staged_options:
        return None
    first_work_index = source_seq.index(work_block[0])
    for staged_state, clear_steps in staged_options:
        source_ready_options = _clear_source_prefix_before_work_block(
            plan_input=plan_input,
            state=staged_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            source_blockers=source_seq[:first_work_index],
            target_track=target_track,
        )
        for (
            source_ready_state,
            source_ready_steps,
            order_buffer_block,
            order_buffer_track,
        ) in source_ready_options:
            current_state = source_ready_state
            steps = [*clear_steps, *source_ready_steps]
            order_buffers: list[tuple[list[str], str]] = []
            if order_buffer_block and order_buffer_track is not None:
                order_buffers.append((list(order_buffer_block), order_buffer_track))
            inserted: list[str] = []
            remaining_work = list(work_block)
            while remaining_work:
                try:
                    current_source_seq = list(current_state.track_sequences[source_track])
                    next_vehicle = remaining_work[0]
                    next_index = current_source_seq.index(next_vehicle)
                except (KeyError, ValueError):
                    break
                if current_source_seq[:next_index]:
                    followup_source_options = _clear_source_prefix_before_work_block(
                        plan_input=plan_input,
                        state=current_state,
                        master=master,
                        route_oracle=route_oracle,
                        vehicle_by_no=vehicle_by_no,
                        source_track=source_track,
                        source_blockers=current_source_seq[:next_index],
                        target_track=target_track,
                    )
                    if not followup_source_options:
                        break
                    current_state, source_steps, extra_buffer, extra_buffer_track = (
                        followup_source_options[0]
                    )
                    steps.extend(source_steps)
                    if extra_buffer and extra_buffer_track is not None:
                        order_buffers.append((list(extra_buffer), extra_buffer_track))
                attach_state, attach_steps = _attach_prefix(
                    plan_input=plan_input,
                    state=current_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    prefix_block=[next_vehicle],
                )
                if attach_state is None:
                    break
                target_state, target_steps = _detach_carried_tail_to_track(
                    plan_input=plan_input,
                    state=attach_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    vehicle_nos=[next_vehicle],
                    target_track=target_track,
                )
                if target_state is None:
                    break
                current_state = target_state
                steps.extend([*attach_steps, *target_steps])
                inserted.append(next_vehicle)
                remaining_work = remaining_work[1:]
            if remaining_work:
                continue
            for buffer_block, buffer_track in reversed(order_buffers):
                current_state, order_restore_steps = _restore_staged_block(
                    plan_input=plan_input,
                    state=current_state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=buffer_track,
                    block=buffer_block,
                    target_track=target_track,
                )
                if current_state is None:
                    break
                steps.extend(order_restore_steps)
            if current_state is None:
                continue
            clear_settlement_options = _target_prefix_settlement_options(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=clear_steps[-1].target_track if clear_block else target_track,
                block=clear_block,
                target_track=target_track,
            )
            for settled_state, clear_restore_steps, clear_buffer_block in clear_settlement_options:
                if settled_state.loco_carry:
                    continue
                after_metrics = compute_structural_metrics(plan_input, settled_state)
                candidate = MoveCandidate(
                    steps=tuple([*steps, *clear_restore_steps]),
                    kind="work_position_sequence",
                    reason=f"{target_track} rank-window sequence repair",
                    focus_tracks=(target_track,),
                    buffer_blocks=tuple(
                        block
                        for block in (
                            *(tuple(block) for block, _track in order_buffers),
                            clear_buffer_block,
                        )
                        if block
                    ),
                )
                if _candidate_dependency_blocker_count(candidate, vehicle_by_no) > 0:
                    continue
                allowed_buffer_debt = len(clear_buffer_block)
                if after_metrics.staging_debt_count > before_metrics.staging_debt_count + allowed_buffer_debt:
                    continue
                if not _rank_window_candidate_improves(
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    target_track=target_track,
                ):
                    continue
                if not _work_block_satisfied(
                    work_block=inserted,
                    target_track=target_track,
                    state=settled_state,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                ):
                    continue
                return candidate
    return None


def _rank_window_candidate_improves(
    *,
    before_metrics: Any,
    after_metrics: Any,
    target_track: str,
) -> bool:
    before_track_defect = before_metrics.target_sequence_defect_by_track.get(target_track, 0)
    after_track_defect = after_metrics.target_sequence_defect_by_track.get(target_track, 0)
    return (
        after_track_defect <= before_track_defect
        and (
            after_track_defect < before_track_defect
            or after_metrics.work_position_unfinished_count < before_metrics.work_position_unfinished_count
        )
    )


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
        source_access_options = _prepare_source_access_options(
            plan_input=plan_input,
            state=state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
        )
        for source_access in source_access_options:
            prepared_source_seq = list(source_access.state.track_sequences.get(source_track, []))
            if vehicle_no not in prepared_source_seq:
                continue
            prepared_target_index = prepared_source_seq.index(vehicle_no)
            staged_options = _stage_prefix_options(
                plan_input=plan_input,
                state=source_access.state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=target_track,
                block=clear_block,
                target_hints=(target_track, source_track),
            )
            for staged_state, clear_steps in staged_options:
                candidate = _build_single_spotting_sequence_candidate_from_prepared_access(
                    plan_input=plan_input,
                    state=staged_state,
                    master=master,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    source_track=source_track,
                    target_track=target_track,
                    before_metrics=before_metrics,
                    source_seq=prepared_source_seq,
                    target_index=prepared_target_index,
                    vehicle_no=vehicle_no,
                    clear_block=clear_block,
                    clear_steps=[
                        *source_access.steps,
                        *clear_steps,
                    ],
                )
                if candidate is not None:
                    return candidate
    return None


def _build_single_spotting_sequence_candidate_from_prepared_access(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    before_metrics: Any,
    source_seq: list[str],
    target_index: int,
    vehicle_no: str,
    clear_block: list[str],
    clear_steps: list[HookAction],
) -> MoveCandidate | None:
    source_prefix = source_seq[: target_index + 1]
    prefix_state, prefix_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=source_prefix,
    )
    if prefix_state is None:
        return None
    target_state, target_steps = _detach_carried_tail_to_track(
        plan_input=plan_input,
        state=prefix_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        vehicle_nos=[vehicle_no],
        target_track=target_track,
    )
    if target_state is None:
        return None
    restored_state, source_dispatch_steps = _dispatch_carried_goal_blocks(
        plan_input=plan_input,
        state=target_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
    )
    if restored_state is None:
        return None
    restore_steps: list[HookAction] = []
    if clear_block:
        clear_staging_track = clear_steps[-1].target_track
        clear_settlement_options = _target_prefix_settlement_options(
            plan_input=plan_input,
            state=restored_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=clear_staging_track,
            block=clear_block,
            target_track=target_track,
        )
        if not clear_settlement_options:
            return None
        restored_state, clear_restore_steps, _clear_buffer_block = clear_settlement_options[0]
        restore_steps.extend(clear_restore_steps)
    steps = [
        *clear_steps,
        *prefix_steps,
        *target_steps,
        *source_dispatch_steps,
        *restore_steps,
    ]
    if not steps or restored_state.loco_carry:
        return None
    after_metrics = compute_structural_metrics(plan_input, restored_state)
    if not _sequence_candidate_improves(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        target_track=target_track,
    ):
        return None
    if not _work_block_satisfied(
        work_block=[vehicle_no],
        target_track=target_track,
        state=restored_state,
        plan_input=plan_input,
        vehicle_by_no=vehicle_by_no,
    ):
        return None
    return MoveCandidate(
        steps=tuple(steps),
        kind="work_position_sequence",
        reason=f"{target_track} SPOTTING sequence repair",
        focus_tracks=(target_track,),
    )


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
                clear_buffer_block: tuple[str, ...] = ()
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
                elif clear_block:
                    clear_buffer_block = tuple(clear_block)
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
                    target_track=target_track,
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
                    focus_tracks=(target_track,),
                    buffer_blocks=tuple(
                        block
                        for block in (
                            tuple(order_buffer_block),
                            clear_buffer_block,
                        )
                        if block
                    ),
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


def _prepare_source_access_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
) -> list[_PreparedSourceAccess]:
    if _access_is_clear(state=state, route_oracle=route_oracle, track=source_track) and _access_is_clear(
        state=state,
        route_oracle=route_oracle,
        track=target_track,
    ):
        return [_PreparedSourceAccess(state=state, steps=())]
    return _prepare_source_access_variants_by_releasing_blockers(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        target_track=target_track,
    )


def _prepare_source_access_variants_by_releasing_blockers(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
) -> list[_PreparedSourceAccess]:
    frontier: list[tuple[ReplayState, tuple[HookAction, ...]]] = [(state, ())]
    completed: list[_PreparedSourceAccess] = []
    seen_keys = {_state_key(state, plan_input)}
    for _round in range(SOURCE_ACCESS_RELEASE_MAX_ROUNDS):
        next_frontier: list[tuple[ReplayState, tuple[HookAction, ...]]] = []
        for current_state, steps in frontier:
            source_access = route_oracle.validate_loco_access(
                loco_track=current_state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=current_state.track_sequences,
                loco_node=current_state.loco_node,
            )
            target_access = route_oracle.validate_loco_access(
                loco_track=current_state.loco_track_name,
                target_track=target_track,
                occupied_track_sequences=current_state.track_sequences,
                loco_node=current_state.loco_node,
            )
            if source_access.is_valid and target_access.is_valid:
                completed.append(_PreparedSourceAccess(state=current_state, steps=steps))
                continue
            for next_state, option_steps in _source_access_release_options(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                target_track=target_track,
                blocking_tracks=(
                    source_access.blocking_tracks
                    if not source_access.is_valid
                    else target_access.blocking_tracks
                ),
            ):
                next_key = _state_key(next_state, plan_input)
                if next_key in seen_keys:
                    continue
                seen_keys.add(next_key)
                next_frontier.append((next_state, (*steps, *option_steps)))
        if completed:
            return _rank_prepared_source_access_options(
                completed,
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=target_track,
            )[:SOURCE_ACCESS_OPTION_LIMIT]
        if not next_frontier:
            return []
        next_frontier.sort(
            key=lambda item: _source_access_frontier_score(
                item[0],
                item[1],
                route_oracle=route_oracle,
                source_track=source_track,
                target_track=target_track,
            )
        )
        frontier = next_frontier[:SOURCE_ACCESS_FRONTIER_LIMIT]
    return []


def _source_access_release_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    blocking_tracks: list[str],
) -> list[tuple[ReplayState, list[HookAction]]]:
    ranked: list[tuple[tuple[int, int, int, int, int, tuple], ReplayState, list[HookAction]]] = []
    for blocking_track in blocking_tracks:
        if blocking_track == source_track:
            continue
        source_seq = list(state.track_sequences.get(blocking_track, []))
        if not source_seq:
            continue
        for prefix_block in _source_access_release_prefix_blocks(
            source_seq=source_seq,
            vehicle_by_no=vehicle_by_no,
        ):
            attached_state, attach_steps = _attach_prefix(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=blocking_track,
                prefix_block=prefix_block,
            )
            if attached_state is None:
                continue
            candidate = _dispatch_source_access_release_block(
                plan_input=plan_input,
                state=attached_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=source_track,
                target_track=target_track,
                blocking_track=blocking_track,
                prefix_block=prefix_block,
            )
            if candidate is None:
                continue
            next_state, dispatch_steps = candidate
            after_access = route_oracle.validate_loco_access(
                loco_track=next_state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=next_state.track_sequences,
                loco_node=next_state.loco_node,
            )
            after_target_access = route_oracle.validate_loco_access(
                loco_track=next_state.loco_track_name,
                target_track=target_track,
                occupied_track_sequences=next_state.track_sequences,
                loco_node=next_state.loco_node,
            )
            after_blockers = set(after_access.blocking_tracks)
            if (
                not after_access.is_valid
                and len(next_state.track_sequences.get(blocking_track, []))
                >= len(state.track_sequences.get(blocking_track, []))
            ):
                continue
            ranked.append(
                (
                    (
                        0
                        if after_access.is_valid and after_target_access.is_valid
                        else 1
                        if after_access.is_valid
                        else 2,
                        len(after_blockers),
                        len(after_target_access.blocking_tracks),
                        len(next_state.track_sequences.get(blocking_track, [])),
                        len(attach_steps) + len(dispatch_steps),
                        tuple(prefix_block),
                    ),
                    next_state,
                    [*attach_steps, *dispatch_steps],
                )
            )
    ranked.sort(key=lambda item: item[0])
    return [
        (next_state, steps)
        for _score, next_state, steps in ranked[:SOURCE_ACCESS_OPTION_LIMIT]
    ]


def _source_access_frontier_score(
    state: ReplayState,
    steps: tuple[HookAction, ...],
    *,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
) -> tuple[int, int, int, int, str]:
    source_access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    )
    target_access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=target_track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    )
    return (
        0 if source_access.is_valid and target_access.is_valid else 1,
        len(source_access.blocking_tracks),
        len(target_access.blocking_tracks),
        len(steps),
        state.loco_track_name,
    )


def _rank_prepared_source_access_options(
    options: list[_PreparedSourceAccess],
    *,
    route_oracle: RouteOracle,
    source_track: str,
    target_track: str,
) -> list[_PreparedSourceAccess]:
    deduped: dict[tuple, _PreparedSourceAccess] = {}
    for option in options:
        key = (
            option.state.loco_track_name,
            tuple((track, tuple(seq)) for track, seq in sorted(option.state.track_sequences.items())),
        )
        current = deduped.get(key)
        if current is None or len(option.steps) < len(current.steps):
            deduped[key] = option
    return sorted(
        deduped.values(),
        key=lambda option: _source_access_frontier_score(
            option.state,
            option.steps,
            route_oracle=route_oracle,
            source_track=source_track,
            target_track=target_track,
        ),
    )


def _source_access_release_prefix_blocks(
    *,
    source_seq: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[list[str]]:
    valid: list[list[str]] = []
    for count in range(1, min(len(source_seq), 20) + 1):
        block = source_seq[:count]
        if validate_hook_vehicle_group([vehicle_by_no[vno] for vno in block]):
            continue
        valid.append(list(block))
    valid.sort(
        key=lambda block: (
            -len(block),
            _goal_run_count(block, vehicle_by_no),
            block,
        )
    )
    return valid[:8]


def _goal_run_count(
    block: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> int:
    runs = 0
    previous_target: str | None = None
    for vehicle_no in block:
        vehicle = vehicle_by_no.get(vehicle_no)
        target = vehicle.goal.target_track if vehicle is not None else None
        if target != previous_target:
            runs += 1
            previous_target = target
    return runs


def _dispatch_source_access_release_block(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    blocking_track: str,
    prefix_block: list[str],
) -> tuple[ReplayState, list[HookAction]] | None:
    current_state = state
    steps: list[HookAction] = []
    while current_state.loco_carry:
        carry = list(current_state.loco_carry)
        dispatch = _best_carried_tail_goal_dispatch_avoiding_source_access(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            target_track=target_track,
            blocking_track=blocking_track,
        )
        if dispatch is not None:
            current_state, dispatch_steps = dispatch
            steps.extend(dispatch_steps)
            continue
        tail_block = _largest_carried_tail_block(carry, vehicle_by_no)
        staged_options = _stage_carried_tail_options(
            plan_input=plan_input,
            state=current_state,
            master=master,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            block=tail_block,
            source_track=blocking_track,
            target_hints=(target_track, source_track),
        )
        staged_options = [
            option
            for option in staged_options
            if not _track_blocks_access(
                state=option[0],
                route_oracle=route_oracle,
                access_track=source_track,
                candidate_track=option[2],
            )
            and not _track_blocks_access(
                state=option[0],
                route_oracle=route_oracle,
                access_track=target_track,
                candidate_track=option[2],
            )
        ]
        if not staged_options:
            return None
        current_state, stage_steps, staging_track = staged_options[0]
        steps.extend(stage_steps)
    if list(current_state.track_sequences.get(blocking_track, []))[: len(prefix_block)] == prefix_block:
        return None
    return current_state, steps


def _best_carried_tail_goal_dispatch_avoiding_source_access(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    target_track: str,
    blocking_track: str,
) -> tuple[ReplayState, list[HookAction]] | None:
    carry = list(state.loco_carry)
    tail_target = _dispatch_target_for_vehicle(
        carry[-1],
        state=state,
        plan_input=plan_input,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        preferred_target_track=None,
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
            preferred_target_track=None,
        )
        if previous_target != tail_target:
            break
        start -= 1
    ranked: list[tuple[tuple[int, int, int, int], ReplayState, list[HookAction]]] = []
    for block_start in range(start, len(carry)):
        block = carry[block_start:]
        target_candidates = _dispatch_target_candidates_for_source_release(
            vehicle_no=block[-1],
            state=state,
            plan_input=plan_input,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            preferred_target=tail_target,
            work_target=target_track,
        )
        for candidate_target in target_candidates:
            if candidate_target in {source_track, target_track, blocking_track}:
                continue
            if _track_blocks_source_access(
                state=state,
                route_oracle=route_oracle,
                source_track=source_track,
                candidate_track=candidate_target,
            ):
                continue
            if candidate_target == blocking_track:
                continue
            next_state, steps = _detach_carried_tail_to_track(
                plan_input=plan_input,
                state=state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle_nos=block,
                target_track=candidate_target,
            )
            if next_state is None:
                continue
            access = route_oracle.validate_loco_access(
                loco_track=next_state.loco_track_name,
                target_track=source_track,
                occupied_track_sequences=next_state.track_sequences,
                loco_node=next_state.loco_node,
            )
            target_access = route_oracle.validate_loco_access(
                loco_track=next_state.loco_track_name,
                target_track=target_track,
                occupied_track_sequences=next_state.track_sequences,
                loco_node=next_state.loco_node,
            )
            if candidate_target in target_access.blocking_tracks:
                continue
            ranked.append(
                (
                    (
                        0 if access.is_valid else 1,
                        len(access.blocking_tracks),
                        0 if candidate_target == tail_target else 1,
                        len(steps),
                    ),
                    next_state,
                    steps,
                )
            )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1], ranked[0][2]


def _dispatch_target_candidates_for_source_release(
    *,
    vehicle_no: str,
    state: ReplayState,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    preferred_target: str,
    work_target: str,
) -> list[str]:
    vehicle = vehicle_by_no.get(vehicle_no)
    if vehicle is None:
        return []
    targets: list[str] = []
    for target in [
        preferred_target,
        vehicle.goal.target_track,
        *goal_effective_allowed_tracks(
            vehicle,
            state=state,
            plan_input=plan_input,
            route_oracle=route_oracle,
        ),
    ]:
        if target and target not in targets:
            targets.append(target)
    return targets


def _largest_carried_tail_block(
    carry: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> list[str]:
    if not carry:
        return []
    tail_target = vehicle_by_no[carry[-1]].goal.target_track
    start = len(carry) - 1
    while start > 0:
        vehicle = vehicle_by_no.get(carry[start - 1])
        if vehicle is None or vehicle.goal.target_track != tail_target:
            break
        start -= 1
    return carry[start:]


def _access_is_clear(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    track: str,
) -> bool:
    return route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=track,
        occupied_track_sequences=state.track_sequences,
        loco_node=state.loco_node,
    ).is_valid


def _track_blocks_source_access(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    source_track: str,
    candidate_track: str,
) -> bool:
    return _track_blocks_access(
        state=state,
        route_oracle=route_oracle,
        access_track=source_track,
        candidate_track=candidate_track,
    )


def _track_blocks_access(
    *,
    state: ReplayState,
    route_oracle: RouteOracle,
    access_track: str,
    candidate_track: str,
) -> bool:
    access = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=access_track,
        occupied_track_sequences={
            **state.track_sequences,
            candidate_track: ["__candidate__"],
        },
        loco_node=state.loco_node,
    )
    return candidate_track in access.blocking_tracks


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
    prefer_larger_blocks: bool = False,
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
        prefer_larger_blocks=prefer_larger_blocks,
    )
    if dispatched_state is None:
        return None, []
    return dispatched_state, [*attach_steps, *dispatch_steps]


def _target_prefix_settlement_options(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_track: str,
) -> list[tuple[ReplayState, list[HookAction], tuple[str, ...]]]:
    if not block:
        return [(state, [], ())]
    options: list[tuple[int, ReplayState, list[HookAction], tuple[str, ...]]] = []
    rebuilt_state, rebuild_steps = _rebuild_target_window_from_staged_prefix(
        plan_input=plan_input,
        state=state,
        master=master,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        block=block,
        target_track=target_track,
    )
    if rebuilt_state is not None:
        options.append((0, rebuilt_state, rebuild_steps, ()))
    dispatched_state, dispatch_steps = _dispatch_staged_block_to_goals(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        block=block,
        target_track=target_track,
    )
    if dispatched_state is not None:
        options.append((1, dispatched_state, dispatch_steps, ()))
    restored_state, restore_steps = _restore_staged_block(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        block=block,
        target_track=target_track,
    )
    if restored_state is not None:
        options.append((2, restored_state, restore_steps, tuple(block)))
    if list(state.track_sequences.get(source_track, []))[: len(block)] == list(block):
        options.append((3, state, [], tuple(block)))
    options.sort(key=lambda item: (item[0], len(item[2])))
    return [
        (settled_state, steps, buffer_block)
        for _priority, settled_state, steps, buffer_block in options
    ]


def _rebuild_target_window_from_staged_prefix(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    master: MasterData,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    source_track: str,
    block: list[str],
    target_track: str,
) -> tuple[ReplayState | None, list[HookAction]]:
    if not block:
        return state, []
    if not any(_is_spotting_goal(vehicle_no, target_track=target_track, vehicle_by_no=vehicle_by_no) for vehicle_no in block):
        return None, []
    if any(
        (vehicle := vehicle_by_no.get(vehicle_no)) is None
        or target_track not in vehicle.goal.allowed_target_tracks
        for vehicle_no in block
    ):
        return None, []
    current_state, attach_steps = _attach_prefix(
        plan_input=plan_input,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        source_track=source_track,
        prefix_block=block,
    )
    if current_state is None:
        return None, []
    steps = list(attach_steps)
    order_buffers: list[tuple[list[str], str]] = []
    while current_state.loco_carry:
        carry = list(current_state.loco_carry)
        if _is_spotting_goal(carry[-1], target_track=target_track, vehicle_by_no=vehicle_by_no):
            start = len(carry) - 1
            while start > 0 and _is_spotting_goal(
                carry[start - 1],
                target_track=target_track,
                vehicle_by_no=vehicle_by_no,
            ):
                start -= 1
            spot_block = carry[start:]
            next_state, detach_steps = _detach_carried_tail_to_track(
                plan_input=plan_input,
                state=current_state,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                vehicle_nos=spot_block,
                target_track=target_track,
            )
            if next_state is None:
                return None, []
            current_state = next_state
            steps.extend(detach_steps)
            continue

        earlier_spot_exists = any(
            _is_spotting_goal(vehicle_no, target_track=target_track, vehicle_by_no=vehicle_by_no)
            for vehicle_no in carry[:-1]
        )
        if earlier_spot_exists:
            start = len(carry) - 1
            while start > 0 and not _is_spotting_goal(
                carry[start - 1],
                target_track=target_track,
                vehicle_by_no=vehicle_by_no,
            ):
                start -= 1
            buffer_block = carry[start:]
            staged_options = _stage_carried_tail_options(
                plan_input=plan_input,
                state=current_state,
                master=master,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                block=buffer_block,
                source_track=current_state.loco_track_name,
                target_hints=(target_track,),
            )
            if not staged_options:
                return None, []
            current_state, stage_steps, buffer_track = staged_options[0]
            steps.extend(stage_steps)
            order_buffers.append((list(buffer_block), buffer_track))
            continue

        remaining_block = list(current_state.loco_carry)
        next_state, detach_steps = _detach_carried_tail_to_track(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            vehicle_nos=remaining_block,
            target_track=target_track,
        )
        if next_state is None:
            return None, []
        current_state = next_state
        steps.extend(detach_steps)

    for buffer_block, buffer_track in reversed(order_buffers):
        current_state, restore_steps = _restore_staged_block(
            plan_input=plan_input,
            state=current_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=buffer_track,
            block=buffer_block,
            target_track=target_track,
        )
        if current_state is None:
            return None, []
        steps.extend(restore_steps)

    return current_state, steps


def _dispatch_carried_goal_blocks(
    *,
    plan_input: NormalizedPlanInput,
    state: ReplayState,
    route_oracle: RouteOracle,
    vehicle_by_no: dict[str, NormalizedVehicle],
    preferred_target_track: str | None = None,
    prefer_larger_blocks: bool = False,
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
            prefer_larger_blocks=prefer_larger_blocks,
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
    prefer_larger_blocks: bool,
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
    block_starts = range(start, -1, -1) if prefer_larger_blocks else range(start, len(carry))
    for block_start in block_starts:
        block = carry[block_start:]
        if prefer_larger_blocks and any(
            _dispatch_target_for_vehicle(
                vehicle_no,
                state=state,
                plan_input=plan_input,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                preferred_target_track=preferred_target_track,
            )
            != tail_target
            for vehicle_no in block
        ):
            continue
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


def _sequence_candidate_improves(
    *,
    before_metrics: Any,
    after_metrics: Any,
    target_track: str,
) -> bool:
    before_track_defect = before_metrics.target_sequence_defect_by_track.get(target_track, 0)
    after_track_defect = after_metrics.target_sequence_defect_by_track.get(target_track, 0)
    return (
        after_track_defect <= before_track_defect
        and after_metrics.target_sequence_defect_count <= before_metrics.target_sequence_defect_count
        and (
            after_track_defect < before_track_defect
            or after_metrics.work_position_unfinished_count < before_metrics.work_position_unfinished_count
        )
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


def _candidate_focus_tracks_by_kind(candidates: list[MoveCandidate]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        if not candidate.focus_tracks:
            continue
        by_track = totals.setdefault(candidate.kind, {})
        for track in candidate.focus_tracks:
            by_track[track] = by_track.get(track, 0) + 1
    return totals
