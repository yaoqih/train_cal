from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.depot_spots import realign_spots_for_track_order
from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle


DEPOT_INNER_TRACKS = ("修1", "修2", "修3", "修4")
LONG_DEPOT_TRACKS = frozenset({"修3", "修4"})


@dataclass(frozen=True)
class Phase3DepotInsertionPlan:
    resolved_track_by_vehicle: dict[str, str]
    track_sequences: dict[str, list[str]]
    spot_assignments: dict[str, str]
    diagnostics: dict[str, Any]


def plan_phase3_depot_insertion(
    *,
    goals: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict],
    current_track_sequences: dict[str, list[str]],
    current_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> Phase3DepotInsertionPlan:
    """Assign concrete 修N tracks for Phase3 random depot arrivals.

    Phase3 must hand the solver executable, concrete targets. An abstract
    ``大库`` target is a business area, not a physical track, so this planner
    resolves each pending depot arrival against the current depot order and
    spot-allocation rules before the stage payload is normalized.
    """
    track_sequences = {
        track: list(current_track_sequences.get(track, ()))
        for track in DEPOT_INNER_TRACKS
    }
    spot_assignments = dict(current_spot_assignments)
    pending_goals = [
        dict(goal)
        for goal in goals
        if _needs_random_depot_assignment(
            goal=goal,
            current_by_vehicle=current_by_vehicle,
        )
    ]
    pending_goals.sort(
        key=lambda goal: _pending_goal_key(
            goal=goal,
            current_by_vehicle=current_by_vehicle,
        )
    )

    resolved_track_by_vehicle: dict[str, str] = {}
    rejected_by_vehicle: dict[str, dict[str, str]] = {}
    for goal in pending_goals:
        vehicle_no = str(goal["vehicleNo"])
        vehicle = current_by_vehicle[vehicle_no]
        track, rejected = _choose_track(
            goal=goal,
            vehicle=vehicle,
            track_sequences=track_sequences,
            spot_assignments=spot_assignments,
            vehicle_by_no=vehicle_by_no,
            yard_mode=yard_mode,
            reserved_spot_codes=reserved_spot_codes,
        )
        rejected_by_vehicle[vehicle_no] = rejected
        if track is None:
            raise ValueError(
                "phase3 depot insertion failed: "
                f"{vehicle_no} cannot be assigned to 修1/修2/修3/修4; "
                f"rejected={rejected}"
            )

        next_sequence = [vehicle_no] + list(track_sequences.get(track, ()))
        next_spots = _realign_with_concrete_random_goal(
            vehicle_nos_in_order=next_sequence,
            vehicle_by_no=vehicle_by_no,
            target_track=track,
            yard_mode=yard_mode,
            current_spot_assignments=spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if next_spots is None:
            raise ValueError(
                "phase3 depot insertion failed after selection: "
                f"{vehicle_no} -> {track}"
            )
        track_sequences[track] = next_sequence
        spot_assignments = next_spots
        resolved_track_by_vehicle[vehicle_no] = track

    diagnostics = {
        "pendingVehicleNos": [str(goal["vehicleNo"]) for goal in pending_goals],
        "resolvedTrackByVehicle": dict(resolved_track_by_vehicle),
        "rejectedByVehicle": rejected_by_vehicle,
        "trackCounts": {
            track: len(track_sequences.get(track, ()))
            for track in DEPOT_INNER_TRACKS
        },
    }
    return Phase3DepotInsertionPlan(
        resolved_track_by_vehicle=resolved_track_by_vehicle,
        track_sequences=track_sequences,
        spot_assignments=spot_assignments,
        diagnostics=diagnostics,
    )


def _needs_random_depot_assignment(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> bool:
    if str(goal.get("targetSource") or "") == "PHASE3_DYNAMIC_CURRENT_HOLD":
        return False
    if (
        str(goal.get("targetAreaCode") or "") != "大库:RANDOM"
        and str(goal.get("targetTrack") or "") != "大库"
    ):
        return False
    vehicle_no = str(goal["vehicleNo"])
    current_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
    return current_track not in DEPOT_INNER_TRACKS


def _pending_goal_key(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> tuple[Any, ...]:
    vehicle = current_by_vehicle[str(goal["vehicleNo"])]
    candidates = _candidate_tracks(goal=goal, vehicle=vehicle)
    source_track = str(vehicle.get("trackName") or "")
    order_text = str(vehicle.get("order") or "")
    try:
        source_order = int(float(order_text))
    except ValueError:
        source_order = 10**9
    return (
        len(candidates),
        0 if float(vehicle.get("vehicleLength") or 0.0) >= 17.6 else 1,
        0 if str(vehicle.get("repairProcess") or "") == "厂修" else 1,
        source_track,
        source_order,
        str(goal["vehicleNo"]),
    )


def _choose_track(
    *,
    goal: dict[str, Any],
    vehicle: dict,
    track_sequences: dict[str, list[str]],
    spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> tuple[str | None, dict[str, str]]:
    vehicle_no = str(goal["vehicleNo"])
    candidates = _candidate_tracks(goal=goal, vehicle=vehicle)
    rejected: dict[str, str] = {}
    usable: list[str] = []
    for track in candidates:
        if float(vehicle.get("vehicleLength") or 0.0) >= 17.6 and track not in LONG_DEPOT_TRACKS:
            rejected[track] = "long_vehicle_requires_修3_or_修4"
            continue
        next_sequence = [vehicle_no] + list(track_sequences.get(track, ()))
        next_spots = _realign_with_concrete_random_goal(
            vehicle_nos_in_order=next_sequence,
            vehicle_by_no=vehicle_by_no,
            target_track=track,
            yard_mode=yard_mode,
            current_spot_assignments=spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if next_spots is None:
            rejected[track] = "spot_realign_failed"
            continue
        usable.append(track)
    if not usable:
        return None, rejected

    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    return (
        sorted(
            usable,
            key=lambda track: (
                0 if track in preferred else 1,
                len(track_sequences.get(track, ())),
                track,
            ),
        )[0],
        rejected,
    )


def _candidate_tracks(*, goal: dict[str, Any], vehicle: dict) -> list[str]:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    fallback = [str(item) for item in goal.get("fallbackTargetTracks") or ()]
    allowed = [str(item) for item in goal.get("allowedTargetTracks") or ()]
    candidates = [
        track
        for track in [*preferred, *fallback, *allowed]
        if track in DEPOT_INNER_TRACKS
    ]
    if not candidates:
        candidates = list(DEPOT_INNER_TRACKS)
    if float(vehicle.get("vehicleLength") or 0.0) >= 17.6:
        candidates = [track for track in candidates if track in LONG_DEPOT_TRACKS]
    return list(dict.fromkeys(candidates))


def _realign_with_concrete_random_goal(
    *,
    vehicle_nos_in_order: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    target_track: str,
    yard_mode: str,
    current_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str],
) -> dict[str, str] | None:
    adjusted_vehicle_by_no = dict(vehicle_by_no)
    for vehicle_no in vehicle_nos_in_order:
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
        vehicle_nos_in_order=vehicle_nos_in_order,
        vehicle_by_no=adjusted_vehicle_by_no,
        target_track=target_track,
        yard_mode=yard_mode,
        current_spot_assignments=current_spot_assignments,
        reserved_spot_codes=reserved_spot_codes,
    )
