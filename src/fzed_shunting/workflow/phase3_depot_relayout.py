from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.depot_spots import list_track_spots, realign_spots_for_track_order
from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle


DEPOT_TRACKS = ("修1", "修2", "修3", "修4")
LONG_DEPOT_TRACKS = frozenset({"修3", "修4"})


@dataclass(frozen=True)
class Phase3DepotRelayoutPlan:
    feasible: bool
    resolved_track_by_vehicle: dict[str, str]
    track_sequences: dict[str, list[str]]
    spot_assignments: dict[str, str]
    diagnostics: dict[str, Any]


def score_phase3_depot_execution_fitness(
    *,
    resolved_track_by_vehicle: dict[str, str],
    current_by_vehicle: dict[str, dict[str, Any]],
    vehicle_by_no: dict[str, NormalizedVehicle],
    track_sequences: dict[str, list[str]],
) -> tuple[Any, ...]:
    source_targets: dict[str, set[str]] = {}
    preferred_penalty = 0
    factory_short_penalty = 0
    for vehicle_no, target_track in resolved_track_by_vehicle.items():
        source_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
        if source_track:
            source_targets.setdefault(source_track, set()).add(target_track)
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        preferred = set(vehicle.goal.preferred_target_tracks)
        fallback = set(vehicle.goal.fallback_target_tracks)
        if preferred and target_track not in preferred:
            preferred_penalty += 2
        elif fallback and target_track in fallback:
            preferred_penalty += 1
        if vehicle.repair_process == "厂修" and target_track in {"修1", "修2"}:
            factory_short_penalty += 1
    source_fragment_penalty = sum(max(0, len(targets) - 1) for targets in source_targets.values())
    load_penalty = sum(len(track_sequences.get(track, ())) * len(track_sequences.get(track, ())) for track in DEPOT_TRACKS)
    return (
        source_fragment_penalty,
        preferred_penalty,
        factory_short_penalty,
        load_penalty,
        tuple((track, tuple(track_sequences.get(track, ()))) for track in DEPOT_TRACKS),
    )


def search_phase3_depot_relayout(
    *,
    goals: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict[str, Any]],
    base_track_sequences: dict[str, list[str]],
    base_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
    max_search_nodes: int = 5_000,
) -> Phase3DepotRelayoutPlan:
    pending_goals = [
        dict(goal)
        for goal in goals
        if _needs_random_depot_relayout(goal=goal, current_by_vehicle=current_by_vehicle)
    ]
    pending_vehicle_nos = [str(goal["vehicleNo"]) for goal in pending_goals]
    if not pending_goals:
        return Phase3DepotRelayoutPlan(
            feasible=True,
            resolved_track_by_vehicle={},
            track_sequences={track: list(base_track_sequences.get(track, ())) for track in DEPOT_TRACKS},
            spot_assignments=dict(base_spot_assignments),
            diagnostics={
                "enabled": False,
                "pendingVehicleNos": [],
                "reason": "no_random_depot_goals",
            },
        )

    capacities = {track: len(list_track_spots(track, yard_mode)) for track in DEPOT_TRACKS}
    base_sequences = {track: list(base_track_sequences.get(track, ())) for track in DEPOT_TRACKS}
    pending_by_no = {str(goal["vehicleNo"]): dict(goal) for goal in pending_goals}
    candidate_tracks_by_vehicle = {
        vehicle_no: _candidate_tracks(goal=goal, vehicle=current_by_vehicle[vehicle_no])
        for vehicle_no, goal in pending_by_no.items()
    }
    ordered_vehicle_nos = sorted(
        pending_by_no,
        key=lambda vehicle_no: _search_order_key(
            vehicle_no=vehicle_no,
            candidate_tracks=candidate_tracks_by_vehicle[vehicle_no],
            current_by_vehicle=current_by_vehicle,
        ),
    )
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "pendingVehicleNos": pending_vehicle_nos,
        "searchOrderVehicleNos": ordered_vehicle_nos,
        "candidateTracksByVehicle": candidate_tracks_by_vehicle,
        "baseCounts": {track: len(base_sequences.get(track, ())) for track in DEPOT_TRACKS},
        "capacities": capacities,
        "rejectedCandidates": [],
        "maxSearchNodes": max_search_nodes,
        "visitedSearchNodes": 0,
        "searchBudgetExhausted": False,
    }
    best = _search(
        ordered_vehicle_nos=ordered_vehicle_nos,
        index=0,
        candidate_tracks_by_vehicle=candidate_tracks_by_vehicle,
        base_sequences=base_sequences,
        assigned_track_by_vehicle={},
        vehicle_by_no=vehicle_by_no,
        yard_mode=yard_mode,
        base_spot_assignments=base_spot_assignments,
        reserved_spot_codes=reserved_spot_codes,
        diagnostics=diagnostics,
        search_state={"visited": 0, "max": max_search_nodes},
        best=None,
    )
    if best is None:
        return Phase3DepotRelayoutPlan(
            feasible=False,
            resolved_track_by_vehicle={},
            track_sequences=base_sequences,
            spot_assignments=dict(base_spot_assignments),
            diagnostics={
                **diagnostics,
                "feasible": False,
                "reason": "no_valid_depot_relayout",
            },
        )
    sequences, spot_assignments, assigned, score = best
    return Phase3DepotRelayoutPlan(
        feasible=True,
        resolved_track_by_vehicle=assigned,
        track_sequences=sequences,
        spot_assignments=spot_assignments,
        diagnostics={
            **diagnostics,
            "feasible": True,
            "score": score,
            "resolvedTrackByVehicle": assigned,
            "trackCounts": {track: len(sequences.get(track, ())) for track in DEPOT_TRACKS},
            "targetSequences": sequences,
        },
    )


def _search(
    *,
    ordered_vehicle_nos: list[str],
    index: int,
    candidate_tracks_by_vehicle: dict[str, list[str]],
    base_sequences: dict[str, list[str]],
    assigned_track_by_vehicle: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    base_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str],
    diagnostics: dict[str, Any],
    search_state: dict[str, int],
    best: tuple[dict[str, list[str]], dict[str, str], dict[str, str], tuple[Any, ...]] | None,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], tuple[Any, ...]] | None:
    search_state["visited"] = search_state.get("visited", 0) + 1
    diagnostics["visitedSearchNodes"] = search_state["visited"]
    if search_state["visited"] > search_state.get("max", 0):
        diagnostics["searchBudgetExhausted"] = True
        return best

    if index >= len(ordered_vehicle_nos):
        spot_assignments = _realign_all_tracks(
            sequences=base_sequences,
            vehicle_by_no=vehicle_by_no,
            yard_mode=yard_mode,
            base_spot_assignments=base_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if spot_assignments is None:
            return best
        score = _score_solution(
            sequences=base_sequences,
            assigned_track_by_vehicle=assigned_track_by_vehicle,
            vehicle_by_no=vehicle_by_no,
        )
        if best is None or score < best[3]:
            return (
                {track: list(seq) for track, seq in base_sequences.items()},
                spot_assignments,
                dict(assigned_track_by_vehicle),
                score,
            )
        return best

    vehicle_no = ordered_vehicle_nos[index]
    for target_track in candidate_tracks_by_vehicle[vehicle_no]:
        next_sequences = {track: list(seq) for track, seq in base_sequences.items()}
        next_sequences[target_track] = [vehicle_no] + next_sequences.get(target_track, [])
        if len(next_sequences[target_track]) > len(list_track_spots(target_track, yard_mode)):
            diagnostics["rejectedCandidates"].append(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": target_track,
                    "reason": "capacity_exceeded",
                }
            )
            continue
        if (
            _realign_track(
                target_track=target_track,
                sequence=next_sequences[target_track],
                vehicle_by_no=vehicle_by_no,
                yard_mode=yard_mode,
                base_spot_assignments=base_spot_assignments,
                reserved_spot_codes=reserved_spot_codes,
            )
            is None
        ):
            diagnostics["rejectedCandidates"].append(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": target_track,
                    "reason": "spot_realign_failed",
                }
            )
            continue
        next_assigned = dict(assigned_track_by_vehicle)
        next_assigned[vehicle_no] = target_track
        best = _search(
            ordered_vehicle_nos=ordered_vehicle_nos,
            index=index + 1,
            candidate_tracks_by_vehicle=candidate_tracks_by_vehicle,
            base_sequences=next_sequences,
            assigned_track_by_vehicle=next_assigned,
            vehicle_by_no=vehicle_by_no,
            yard_mode=yard_mode,
            base_spot_assignments=base_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
            diagnostics=diagnostics,
            search_state=search_state,
            best=best,
        )
        if diagnostics.get("searchBudgetExhausted"):
            break
    return best


def _realign_all_tracks(
    *,
    sequences: dict[str, list[str]],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    base_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str],
) -> dict[str, str] | None:
    spot_assignments = dict(base_spot_assignments)
    for track in DEPOT_TRACKS:
        realigned = _realign_track(
            target_track=track,
            sequence=sequences.get(track, []),
            vehicle_by_no=vehicle_by_no,
            yard_mode=yard_mode,
            base_spot_assignments=spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if realigned is None:
            return None
        spot_assignments = realigned
    return spot_assignments


def _realign_track(
    *,
    target_track: str,
    sequence: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    base_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str],
) -> dict[str, str] | None:
    adjusted_vehicle_by_no = dict(vehicle_by_no)
    for vehicle_no in sequence:
        vehicle = adjusted_vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        if vehicle.goal.target_area_code != "大库:RANDOM":
            continue
        adjusted_vehicle_by_no[vehicle_no] = vehicle.model_copy(
            update={
                "goal": GoalSpec(
                    target_mode="TRACK",
                    target_track=target_track,
                    allowed_target_tracks=[target_track],
                    target_area_code="大库:RANDOM",
                    target_source=vehicle.goal.target_source,
                )
            }
        )
    return realign_spots_for_track_order(
        vehicle_nos_in_order=sequence,
        vehicle_by_no=adjusted_vehicle_by_no,
        target_track=target_track,
        yard_mode=yard_mode,
        current_spot_assignments=base_spot_assignments,
        reserved_spot_codes=reserved_spot_codes,
    )


def _needs_random_depot_relayout(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict[str, Any]],
) -> bool:
    if str(goal.get("targetSource") or "") == "PHASE3_DYNAMIC_CURRENT_HOLD":
        return False
    if str(goal.get("targetAreaCode") or "") != "大库:RANDOM" and str(goal.get("targetTrack") or "") != "大库":
        return False
    vehicle_no = str(goal["vehicleNo"])
    current_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
    return current_track not in DEPOT_TRACKS


def _candidate_tracks(*, goal: dict[str, Any], vehicle: dict[str, Any]) -> list[str]:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    fallback = [str(item) for item in goal.get("fallbackTargetTracks") or ()]
    allowed = [str(item) for item in goal.get("allowedTargetTracks") or ()]
    candidates = [track for track in [*preferred, *fallback, *allowed] if track in DEPOT_TRACKS]
    if not candidates:
        candidates = list(DEPOT_TRACKS)
    if float(vehicle.get("vehicleLength") or 0.0) >= 17.6:
        candidates = [track for track in candidates if track in LONG_DEPOT_TRACKS]
    return list(dict.fromkeys(candidates))


def _search_order_key(
    *,
    vehicle_no: str,
    candidate_tracks: list[str],
    current_by_vehicle: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    vehicle = current_by_vehicle[vehicle_no]
    return (
        len(candidate_tracks),
        0 if float(vehicle.get("vehicleLength") or 0.0) >= 17.6 else 1,
        0 if str(vehicle.get("repairProcess") or "") == "厂修" else 1,
        str(vehicle.get("trackName") or ""),
        _parse_order(vehicle.get("order")),
        vehicle_no,
    )


def _parse_order(value: Any) -> int:
    try:
        return int(float(str(value)))
    except ValueError:
        return 10**9


def _score_solution(
    *,
    sequences: dict[str, list[str]],
    assigned_track_by_vehicle: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> tuple[Any, ...]:
    preferred_penalty = 0
    load_penalty = 0
    factory_short_penalty = 0
    for vehicle_no, target_track in assigned_track_by_vehicle.items():
        vehicle = vehicle_by_no[vehicle_no]
        preferred = set(vehicle.goal.preferred_target_tracks)
        fallback = set(vehicle.goal.fallback_target_tracks)
        if preferred and target_track not in preferred:
            preferred_penalty += 2
        elif fallback and target_track in fallback:
            preferred_penalty += 1
        if vehicle.repair_process == "厂修" and target_track in {"修1", "修2"}:
            factory_short_penalty += 1
    for track, sequence in sequences.items():
        load_penalty += len(sequence) * len(sequence)
    return (
        preferred_penalty,
        factory_short_penalty,
        load_penalty,
        tuple((track, tuple(sequences.get(track, ()))) for track in DEPOT_TRACKS),
    )
