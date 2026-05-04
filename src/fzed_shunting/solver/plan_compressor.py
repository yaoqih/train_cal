from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.state import _apply_move
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import ReplayState


@dataclass(frozen=True)
class CompressionResult:
    compressed_plan: list[HookAction]
    accepted_rewrite_count: int


def compress_plan(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    plan: list[HookAction],
    *,
    master: MasterData | None,
    max_window_size: int = 10,
    max_passes: int = 16,
    time_budget_ms: float | None = None,
) -> CompressionResult:
    """Verifier-guarded local plan compression.

    This pass is intentionally conservative: it only accepts a shorter plan
    when replay reaches the same terminal state and the full verifier accepts
    the candidate. Failed rewrites leave the original plan unchanged.
    """
    if not plan or master is None or (time_budget_ms is not None and time_budget_ms <= 0):
        return CompressionResult(compressed_plan=list(plan), accepted_rewrite_count=0)
    started_at = perf_counter()
    current = list(plan)
    accepted = 0
    for _ in range(max_passes):
        if _compression_budget_exhausted(started_at, time_budget_ms):
            break
        changed = False
        current_terminal_state = _simulate(plan_input, initial_state, current)
        if current_terminal_state is None:
            break
        current, accepted_rebuild = _try_rebuild_single_source_window(
            master=master,
            plan_input=plan_input,
            initial_state=initial_state,
            current=current,
            current_terminal_state=current_terminal_state,
            max_window_size=max_window_size,
        )
        if accepted_rebuild:
            accepted += 1
            changed = True
            continue
        current, accepted_same_source_merge = _try_merge_adjacent_same_source_same_target_pairs(
            master=master,
            plan_input=plan_input,
            initial_state=initial_state,
            current=current,
            current_terminal_state=current_terminal_state,
        )
        if accepted_same_source_merge:
            accepted += 1
            changed = True
            continue
        current, accepted_merge = _try_merge_adjacent_same_target_pairs(
            master=master,
            plan_input=plan_input,
            initial_state=initial_state,
            current=current,
            current_terminal_state=current_terminal_state,
        )
        if accepted_merge:
            accepted += 1
            changed = True
            continue
        for window_size in range(2, min(max_window_size, len(current)) + 1):
            index = 0
            while index + window_size <= len(current):
                if _compression_budget_exhausted(started_at, time_budget_ms):
                    break
                candidate = current[:index] + current[index + window_size:]
                if len(candidate) >= len(current):
                    index += 1
                    continue
                candidate_state = _simulate(plan_input, initial_state, candidate)
                if candidate_state is None:
                    index += 1
                    continue
                if not _verify_candidate(master, plan_input, initial_state, candidate):
                    index += 1
                    continue
                current = candidate
                accepted += 1
                changed = True
            if _compression_budget_exhausted(started_at, time_budget_ms):
                break
            if changed:
                break
        if not changed:
            break
    return CompressionResult(compressed_plan=current, accepted_rewrite_count=accepted)


def _compression_budget_exhausted(
    started_at: float,
    time_budget_ms: float | None,
) -> bool:
    if time_budget_ms is None:
        return False
    return (perf_counter() - started_at) * 1000 >= time_budget_ms


def _try_merge_adjacent_same_source_same_target_pairs(
    *,
    master: MasterData,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    current: list[HookAction],
    current_terminal_state: ReplayState,
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    prefix_states = _prefix_states(plan_input, initial_state, current)
    for index in range(0, len(current) - 3):
        pre_state = prefix_states[index]
        if pre_state is None:
            continue
        replacement = _adjacent_same_source_same_target_pair_replacement(
            current=current,
            index=index,
            plan_input=plan_input,
            route_oracle=route_oracle,
            pre_state=pre_state,
        )
        if replacement is None:
            continue
        candidate = current[:index] + replacement + current[index + 4:]
        candidate_state = _simulate(plan_input, initial_state, candidate)
        if candidate_state is None:
            continue
        if not _preserves_terminal_track_sequences(candidate_state, current_terminal_state):
            continue
        if not _verify_candidate(master, plan_input, initial_state, candidate):
            continue
        return candidate, True
    return current, False


def _adjacent_same_source_same_target_pair_replacement(
    *,
    current: list[HookAction],
    index: int,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    pre_state: ReplayState,
) -> list[HookAction] | None:
    first_attach, first_detach, second_attach, second_detach = current[index : index + 4]
    if first_attach.action_type != "ATTACH" or second_attach.action_type != "ATTACH":
        return None
    if first_detach.action_type != "DETACH" or second_detach.action_type != "DETACH":
        return None
    if first_attach.source_track != first_detach.source_track:
        return None
    if second_attach.source_track != second_detach.source_track:
        return None
    if first_attach.source_track != second_attach.source_track:
        return None
    if first_detach.target_track != second_detach.target_track:
        return None
    if not first_attach.vehicle_nos or not second_attach.vehicle_nos:
        return None
    if first_attach.vehicle_nos != first_detach.vehicle_nos:
        return None
    if second_attach.vehicle_nos != second_detach.vehicle_nos:
        return None

    source_track = first_attach.source_track
    target_track = first_detach.target_track
    combined = list(first_attach.vehicle_nos) + list(second_attach.vehicle_nos)
    attach_move = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=combined,
        path_tracks=[source_track],
        action_type="ATTACH",
    )
    try:
        after_attach = _apply_move(
            state=ReplayState.model_validate(pre_state.model_dump()),
            move=attach_move,
            plan_input=plan_input,
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles},
        )
    except Exception:  # noqa: BLE001
        return None
    path_tracks = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=after_attach.track_sequences,
        source_node=after_attach.loco_node,
        target_node=route_oracle.order_end_node(target_track),
    )
    if path_tracks is None:
        return None
    return [
        attach_move,
        HookAction(
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=combined,
            path_tracks=path_tracks,
            action_type="DETACH",
        ),
    ]


def _try_merge_adjacent_same_target_pairs(
    *,
    master: MasterData,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    current: list[HookAction],
    current_terminal_state: ReplayState,
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    prefix_states = _prefix_states(plan_input, initial_state, current)
    for index in range(0, len(current) - 3):
        pre_state = prefix_states[index]
        if pre_state is None:
            continue
        replacement = _adjacent_same_target_pair_replacement(
            current=current,
            index=index,
            plan_input=plan_input,
            route_oracle=route_oracle,
            pre_state=pre_state,
        )
        if replacement is None:
            continue
        candidate = current[:index] + replacement + current[index + 4:]
        candidate_state = _simulate(plan_input, initial_state, candidate)
        if candidate_state is None:
            continue
        if not _preserves_terminal_track_sequences(candidate_state, current_terminal_state):
            continue
        if not _verify_candidate(master, plan_input, initial_state, candidate):
            continue
        return candidate, True
    return current, False


def _adjacent_same_target_pair_replacement(
    *,
    current: list[HookAction],
    index: int,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    pre_state: ReplayState,
) -> list[HookAction] | None:
    first_attach, first_detach, second_attach, second_detach = current[index : index + 4]
    if first_attach.action_type != "ATTACH" or second_attach.action_type != "ATTACH":
        return None
    if first_detach.action_type != "DETACH" or second_detach.action_type != "DETACH":
        return None
    if first_attach.source_track != first_detach.source_track:
        return None
    if second_attach.source_track != second_detach.source_track:
        return None
    if first_detach.target_track != second_detach.target_track:
        return None
    if not first_attach.vehicle_nos or not second_attach.vehicle_nos:
        return None
    if first_attach.vehicle_nos != first_detach.vehicle_nos:
        return None
    if second_attach.vehicle_nos != second_detach.vehicle_nos:
        return None

    target_track = first_detach.target_track
    combined = list(second_attach.vehicle_nos) + list(first_attach.vehicle_nos)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    try:
        after_attach = _apply_move(
            state=ReplayState.model_validate(pre_state.model_dump()),
            move=second_attach,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
        after_attach = _apply_move(
            state=after_attach,
            move=first_attach,
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    path_tracks = route_oracle.resolve_clear_path_tracks(
        first_attach.source_track,
        target_track,
        occupied_track_sequences=after_attach.track_sequences,
        source_node=after_attach.loco_node,
        target_node=route_oracle.order_end_node(target_track),
    )
    if path_tracks is None:
        return None
    return [
        second_attach,
        first_attach,
        HookAction(
            source_track=first_attach.source_track,
            target_track=target_track,
            vehicle_nos=combined,
            path_tracks=path_tracks,
            action_type="DETACH",
        ),
    ]


def _try_rebuild_single_source_window(
    *,
    master: MasterData,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    current: list[HookAction],
    current_terminal_state: ReplayState,
    max_window_size: int,
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    prefix_states = _prefix_states(plan_input, initial_state, current)
    for window_size in range(4, min(max_window_size, len(current)) + 1):
        for index in range(0, len(current) - window_size + 1):
            pre_state = prefix_states[index]
            post_state = prefix_states[index + window_size]
            if pre_state is None or post_state is None:
                continue
            replacement = _single_source_window_replacement(
                plan_input=plan_input,
                current=current,
                index=index,
                window_size=window_size,
                route_oracle=route_oracle,
                pre_state=pre_state,
                post_state=post_state,
            )
            if replacement is None or len(replacement) >= window_size:
                continue
            candidate = current[:index] + replacement + current[index + window_size:]
            candidate_state = _simulate(plan_input, initial_state, candidate)
            if candidate_state is None:
                continue
            if not _preserves_terminal_track_sequences(candidate_state, current_terminal_state):
                continue
            if not _verify_candidate(master, plan_input, initial_state, candidate):
                continue
            return candidate, True
    return current, False


def _single_source_window_replacement(
    *,
    plan_input: NormalizedPlanInput,
    current: list[HookAction],
    index: int,
    window_size: int,
    route_oracle: RouteOracle,
    pre_state: ReplayState,
    post_state: ReplayState,
) -> list[HookAction] | None:
    window = current[index : index + window_size]
    first = window[0]
    if first.action_type != "ATTACH" or not first.vehicle_nos:
        return None
    source_seq = pre_state.track_sequences.get(first.source_track, [])
    if source_seq[: len(first.vehicle_nos)] != list(first.vehicle_nos):
        return None
    changed = _changed_vehicle_set(pre_state, post_state)
    for moved in _single_source_moved_candidates(
        source_seq=source_seq,
        first=first,
        window=window,
        changed_vehicle_nos=changed,
        plan_input=plan_input,
        post_state=post_state,
    ):
        replacement = _single_source_replacement_for_moved(
            moved=moved,
            first=first,
            plan_input=plan_input,
            route_oracle=route_oracle,
            pre_state=pre_state,
            post_state=post_state,
        )
        if replacement is not None:
            return replacement
    return None


def _single_source_moved_candidates(
    *,
    source_seq: list[str],
    first: HookAction,
    window: list[HookAction],
    changed_vehicle_nos: set[str],
    plan_input: NormalizedPlanInput,
    post_state: ReplayState,
) -> list[list[str]]:
    candidates: list[list[str]] = []
    first_moved = list(first.vehicle_nos)
    has_later_same_source_attach = any(
        move.source_track == first.source_track and move.action_type == "ATTACH"
        for move in window[1:]
    )
    if not has_later_same_source_attach and set(first_moved) == changed_vehicle_nos:
        candidates.append(first_moved)

    if _is_promising_single_source_window(window):
        prefix_moved = _changed_source_prefix(
            source_seq=source_seq,
            changed_vehicle_nos=changed_vehicle_nos,
        )
        if prefix_moved is not None and prefix_moved not in candidates:
            candidates.append(prefix_moved)

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    return [
        moved
        for moved in candidates
        if _moved_vehicles_have_final_non_staging_or_goal(
            moved,
            post_state=post_state,
            vehicle_by_no=vehicle_by_no,
            plan_input=plan_input,
        )
    ]


def _moved_vehicles_have_final_non_staging_or_goal(
    moved: list[str],
    *,
    post_state: ReplayState,
    vehicle_by_no: dict[str, object],
    plan_input: NormalizedPlanInput,
) -> bool:
    final_track_by_vehicle = _vehicle_track_lookup(post_state)
    if any(vehicle_no not in final_track_by_vehicle for vehicle_no in moved):
        return False
    for vehicle_no in moved:
        track_name = final_track_by_vehicle[vehicle_no]
        vehicle = vehicle_by_no.get(vehicle_no)
        if (
            track_name in STAGING_TRACKS
            and (
                vehicle is None
                or not goal_is_satisfied(
                    vehicle,
                    track_name=track_name,
                    state=post_state,
                    plan_input=plan_input,
                )
            )
        ):
            return False
    return True


def _single_source_replacement_for_moved(
    *,
    moved: list[str],
    first: HookAction,
    plan_input: NormalizedPlanInput,
    route_oracle: RouteOracle,
    pre_state: ReplayState,
    post_state: ReplayState,
) -> list[HookAction] | None:
    final_track_by_vehicle = _vehicle_track_lookup(post_state)
    detach_groups = _final_detach_groups(moved, final_track_by_vehicle)
    if detach_groups is None or not detach_groups:
        return None
    replacement = [
        HookAction(
            source_track=first.source_track,
            target_track=first.source_track,
            vehicle_nos=moved,
            path_tracks=[first.source_track],
            action_type="ATTACH",
        )
    ]
    current_loco_track = first.source_track
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    replayed = ReplayState.model_validate(pre_state.model_dump())
    try:
        replayed = _apply_move(
            state=replayed,
            move=replacement[0],
            plan_input=plan_input,
            vehicle_by_no=vehicle_by_no,
        )
    except Exception:  # noqa: BLE001
        return None
    for target_track, vehicle_nos in detach_groups:
        path_tracks = route_oracle.resolve_clear_path_tracks(
            current_loco_track,
            target_track,
            occupied_track_sequences=replayed.track_sequences,
            source_node=replayed.loco_node,
            target_node=route_oracle.order_end_node(target_track),
        )
        if path_tracks is None:
            return None
        move = HookAction(
            source_track=current_loco_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            path_tracks=path_tracks,
            action_type="DETACH",
        )
        replacement.append(move)
        try:
            replayed = _apply_move(
                state=replayed,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
        current_loco_track = target_track
    if _state_equivalence_key(replayed) != _state_equivalence_key(post_state):
        return None
    return replacement


def _is_promising_single_source_window(window: list[HookAction]) -> bool:
    first = window[0]
    stages_then_regrabs = any(
        move.action_type == "DETACH" and move.target_track in STAGING_TRACKS
        for move in window
    ) and any(
        move.action_type == "ATTACH" and move.source_track in STAGING_TRACKS
        for move in window[1:]
    )
    splits_same_source = any(
        move.action_type == "ATTACH" and move.source_track == first.source_track
        for move in window[1:]
    )
    return stages_then_regrabs or splits_same_source


def _changed_source_prefix(
    *,
    source_seq: list[str],
    changed_vehicle_nos: set[str],
) -> list[str] | None:
    if not changed_vehicle_nos:
        return None
    positions = [
        index
        for index, vehicle_no in enumerate(source_seq)
        if vehicle_no in changed_vehicle_nos
    ]
    if len(positions) != len(changed_vehicle_nos):
        return None
    prefix_size = max(positions) + 1
    moved = source_seq[:prefix_size]
    if set(moved) != changed_vehicle_nos:
        return None
    return list(moved)


def _changed_vehicle_set(pre_state: ReplayState, post_state: ReplayState) -> set[str]:
    pre_locations = _vehicle_track_lookup(pre_state)
    post_locations = _vehicle_track_lookup(post_state)
    changed = {
        vehicle_no
        for vehicle_no, track in pre_locations.items()
        if post_locations.get(vehicle_no) != track
    }
    changed.update(vehicle_no for vehicle_no in post_locations if vehicle_no not in pre_locations)
    if pre_state.loco_carry != post_state.loco_carry:
        changed.update(pre_state.loco_carry)
        changed.update(post_state.loco_carry)
    return changed


def _vehicle_track_lookup(state: ReplayState) -> dict[str, str]:
    return {
        vehicle_no: track_name
        for track_name, seq in state.track_sequences.items()
        for vehicle_no in seq
    }


def _final_detach_groups(
    moved: list[str],
    final_track_by_vehicle: dict[str, str],
) -> list[tuple[str, list[str]]] | None:
    groups: list[tuple[str, list[str]]] = []
    index = len(moved)
    while index > 0:
        target = final_track_by_vehicle[moved[index - 1]]
        start = index - 1
        while start > 0 and final_track_by_vehicle[moved[start - 1]] == target:
            start -= 1
        groups.append((target, list(moved[start:index])))
        index = start
    return groups


def _prefix_states(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    plan: list[HookAction],
) -> list[ReplayState | None]:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    state: ReplayState | None = ReplayState.model_validate(initial_state.model_dump())
    states: list[ReplayState | None] = [state]
    for move in plan:
        if state is None:
            states.append(None)
            continue
        if move.action_type == "DETACH" and move.source_track != state.loco_track_name:
            state = None
            states.append(None)
            continue
        try:
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            state = None
        states.append(state)
    return states


def _simulate(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    plan: list[HookAction],
) -> ReplayState | None:
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    state = ReplayState.model_validate(initial_state.model_dump())
    try:
        for move in plan:
            if move.action_type == "DETACH" and move.source_track != state.loco_track_name:
                return None
            state = _apply_move(
                state=state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
    except Exception:  # noqa: BLE001
        return None
    return state


def _state_equivalence_key(state: ReplayState) -> tuple:
    return (
        tuple(
            (track, tuple(seq))
            for track, seq in sorted(state.track_sequences.items())
            if seq
        ),
        tuple(sorted(state.weighed_vehicle_nos)),
        tuple(sorted(state.spot_assignments.items())),
        state.loco_carry,
    )


def _preserves_terminal_track_sequences(
    candidate_state: ReplayState,
    current_terminal_state: ReplayState,
) -> bool:
    return _track_sequence_key(candidate_state) == _track_sequence_key(current_terminal_state)


def _track_sequence_key(state: ReplayState) -> tuple:
    return tuple(
        (track, tuple(seq))
        for track, seq in sorted(state.track_sequences.items())
        if seq
    )


def _verify_candidate(
    master: MasterData,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    plan: list[HookAction],
) -> bool:
    hook_plan = [
        {
            "hookNo": index,
            "actionType": move.action_type,
            "sourceTrack": move.source_track,
            "targetTrack": move.target_track,
            "vehicleNos": list(move.vehicle_nos),
            "pathTracks": list(move.path_tracks),
        }
        for index, move in enumerate(plan, start=1)
    ]
    try:
        report = verify_plan(
            master,
            plan_input,
            hook_plan,
            initial_state_override=initial_state,
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(report.is_valid)
