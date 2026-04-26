from __future__ import annotations

from dataclasses import dataclass

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput
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
    max_window_size: int = 8,
    max_passes: int = 16,
) -> CompressionResult:
    """Verifier-guarded local plan compression.

    This pass is intentionally conservative: it only accepts a shorter plan
    when replay reaches the same terminal state and the full verifier accepts
    the candidate. Failed rewrites leave the original plan unchanged.
    """
    if not plan or master is None:
        return CompressionResult(compressed_plan=list(plan), accepted_rewrite_count=0)
    current = list(plan)
    accepted = 0
    for _ in range(max_passes):
        changed = False
        baseline_state = _simulate(plan_input, initial_state, current)
        if baseline_state is None:
            break
        baseline_key = _state_equivalence_key(baseline_state)
        current, accepted_rebuild = _try_rebuild_single_source_window(
            master=master,
            plan_input=plan_input,
            initial_state=initial_state,
            current=current,
            baseline_key=baseline_key,
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
            baseline_key=baseline_key,
        )
        if accepted_merge:
            accepted += 1
            changed = True
            continue
        for window_size in range(2, min(max_window_size, len(current)) + 1):
            index = 0
            while index + window_size <= len(current):
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
                if _state_equivalence_key(candidate_state) != baseline_key:
                    if not _is_order_only_goal_state_change(
                        plan_input=plan_input,
                        baseline_state=baseline_state,
                        candidate_state=candidate_state,
                    ):
                        index += 1
                        continue
                current = candidate
                accepted += 1
                changed = True
                baseline_state = candidate_state
                baseline_key = _state_equivalence_key(candidate_state)
            if changed:
                break
        if not changed:
            break
    return CompressionResult(compressed_plan=current, accepted_rewrite_count=accepted)


def _is_order_only_goal_state_change(
    *,
    plan_input: NormalizedPlanInput,
    baseline_state: ReplayState,
    candidate_state: ReplayState,
) -> bool:
    if baseline_state.loco_carry != candidate_state.loco_carry:
        return False
    if baseline_state.loco_track_name != candidate_state.loco_track_name:
        return False
    if baseline_state.weighed_vehicle_nos != candidate_state.weighed_vehicle_nos:
        return False
    if baseline_state.spot_assignments != candidate_state.spot_assignments:
        return False
    baseline_tracks = {
        track: sorted(seq)
        for track, seq in baseline_state.track_sequences.items()
        if seq
    }
    candidate_tracks = {
        track: sorted(seq)
        for track, seq in candidate_state.track_sequences.items()
        if seq
    }
    if baseline_tracks != candidate_tracks:
        return False
    spot_vehicle_nos = {
        vehicle.vehicle_no
        for vehicle in plan_input.vehicles
        if vehicle.goal.target_mode == "SPOT" or vehicle.goal.target_spot_code
    }
    changed_tracks = set(baseline_state.track_sequences) | set(candidate_state.track_sequences)
    for track in changed_tracks:
        baseline_seq = baseline_state.track_sequences.get(track, [])
        candidate_seq = candidate_state.track_sequences.get(track, [])
        if baseline_seq == candidate_seq:
            continue
        if any(vehicle_no in spot_vehicle_nos for vehicle_no in baseline_seq + candidate_seq):
            return False
    return True


def _try_merge_adjacent_same_source_same_target_pairs(
    *,
    master: MasterData,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    current: list[HookAction],
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    for index in range(0, len(current) - 3):
        replacement = _adjacent_same_source_same_target_pair_replacement(
            current=current,
            index=index,
            route_oracle=route_oracle,
        )
        if replacement is None:
            continue
        candidate = current[:index] + replacement + current[index + 4:]
        candidate_state = _simulate(plan_input, initial_state, candidate)
        if candidate_state is None:
            continue
        if not _verify_candidate(master, plan_input, initial_state, candidate):
            continue
        return candidate, True
    return current, False


def _adjacent_same_source_same_target_pair_replacement(
    *,
    current: list[HookAction],
    index: int,
    route_oracle: RouteOracle,
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
    path_tracks = route_oracle.resolve_path_tracks(source_track, target_track)
    if path_tracks is None:
        return None
    return [
        HookAction(
            source_track=source_track,
            target_track=source_track,
            vehicle_nos=combined,
            path_tracks=[source_track],
            action_type="ATTACH",
        ),
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
    baseline_key: tuple,
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    for index in range(0, len(current) - 3):
        replacement = _adjacent_same_target_pair_replacement(
            current=current,
            index=index,
            route_oracle=route_oracle,
        )
        if replacement is None:
            continue
        candidate = current[:index] + replacement + current[index + 4:]
        candidate_state = _simulate(plan_input, initial_state, candidate)
        if candidate_state is None:
            continue
        if _state_equivalence_key(candidate_state) != baseline_key:
            continue
        if not _verify_candidate(master, plan_input, initial_state, candidate):
            continue
        return candidate, True
    return current, False


def _adjacent_same_target_pair_replacement(
    *,
    current: list[HookAction],
    index: int,
    route_oracle: RouteOracle,
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
    path_tracks = route_oracle.resolve_path_tracks(first_attach.source_track, target_track)
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
    baseline_key: tuple,
    max_window_size: int,
) -> tuple[list[HookAction], bool]:
    route_oracle = RouteOracle(master)
    for window_size in range(4, min(max_window_size, len(current)) + 1):
        for index in range(0, len(current) - window_size + 1):
            replacement = _single_source_window_replacement(
                plan_input=plan_input,
                initial_state=initial_state,
                current=current,
                index=index,
                window_size=window_size,
                route_oracle=route_oracle,
            )
            if replacement is None or len(replacement) >= window_size:
                continue
            candidate = current[:index] + replacement + current[index + window_size:]
            candidate_state = _simulate(plan_input, initial_state, candidate)
            if candidate_state is None:
                continue
            if _state_equivalence_key(candidate_state) != baseline_key:
                continue
            if not _verify_candidate(master, plan_input, initial_state, candidate):
                continue
            return candidate, True
    return current, False


def _single_source_window_replacement(
    *,
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    current: list[HookAction],
    index: int,
    window_size: int,
    route_oracle: RouteOracle,
) -> list[HookAction] | None:
    window = current[index : index + window_size]
    first = window[0]
    if first.action_type != "ATTACH" or not first.vehicle_nos:
        return None
    pre_state = _simulate(plan_input, initial_state, current[:index])
    post_state = _simulate(plan_input, initial_state, current[: index + window_size])
    if pre_state is None or post_state is None:
        return None
    source_seq = pre_state.track_sequences.get(first.source_track, [])
    moved = list(first.vehicle_nos)
    if source_seq[: len(moved)] != moved:
        return None
    if any(move.source_track == first.source_track and move.action_type == "ATTACH" for move in window[1:]):
        return None

    final_track_by_vehicle = _vehicle_track_lookup(post_state)
    if any(vehicle_no not in final_track_by_vehicle for vehicle_no in moved):
        return None
    if set(moved) != _changed_vehicle_set(pre_state, post_state):
        return None

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
    for target_track, vehicle_nos in detach_groups:
        path_tracks = route_oracle.resolve_path_tracks(current_loco_track, target_track)
        if path_tracks is None:
            return None
        replacement.append(
            HookAction(
                source_track=current_loco_track,
                target_track=target_track,
                vehicle_nos=vehicle_nos,
                path_tracks=path_tracks,
                action_type="DETACH",
            )
        )
        current_loco_track = target_track
    replayed = pre_state
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in plan_input.vehicles}
    for move in replacement:
        try:
            replayed = _apply_move(
                state=replayed,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
        except Exception:  # noqa: BLE001
            return None
    if _state_equivalence_key(replayed) != _state_equivalence_key(post_state):
        return None
    return replacement


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
    index = 0
    while index < len(moved):
        target = final_track_by_vehicle[moved[index]]
        group: list[str] = []
        while index < len(moved) and final_track_by_vehicle[moved[index]] == target:
            group.append(moved[index])
            index += 1
        groups.append((target, group))
    return groups


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
