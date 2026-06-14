from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.depot_spots import list_track_spots, realign_spots_for_track_order
from fzed_shunting.io.normalize_input import NormalizedVehicle


PHASE3_BLOCK_TARGET_TRACKS = frozenset({"修1", "修2", "修3", "修4", "轮"})
PHASE3_DEPOT_TARGET_TRACKS = frozenset({"修1", "修2", "修3", "修4"})


@dataclass(frozen=True)
class Phase3BlockPlan:
    wave_plans: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def build_phase3_depot_block_plan(
    *,
    goals: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict[str, Any]],
    track_sequences: dict[str, list[str] | tuple[str, ...]],
    depot_track_sequences: dict[str, list[str]] | None = None,
    current_spot_assignments: dict[str, str] | None = None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None = None,
    yard_mode: str = "NORMAL",
    reserved_spot_codes: set[str] | frozenset[str] | None = None,
    min_active_vehicle_count: int = 10**9,
    min_source_track_count: int = 4,
) -> Phase3BlockPlan:
    goal_by_vehicle = {str(goal["vehicleNo"]): dict(goal) for goal in goals}
    active_by_vehicle: dict[str, dict[str, Any]] = {}
    source_tracks: set[str] = set()
    target_tracks: set[str] = set()
    for vehicle_no, goal in goal_by_vehicle.items():
        target_track = str(goal.get("targetTrack") or "")
        if target_track not in PHASE3_BLOCK_TARGET_TRACKS:
            continue
        current_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
        if not current_track or current_track == target_track:
            continue
        active_by_vehicle[vehicle_no] = goal
        source_tracks.add(current_track)
        target_tracks.add(target_track)

    source_block_result = _build_source_blocks(
        active_by_vehicle=active_by_vehicle,
        track_sequences=track_sequences,
        current_by_vehicle=current_by_vehicle,
    )
    source_blocks = source_block_result["blocks"]
    hidden_active_vehicle_nos = source_block_result["hiddenActiveVehicleNos"]
    missing_vehicle_nos = source_block_result["missingVehicleNos"]
    covered_vehicle_nos = source_block_result["coveredVehicleNos"]
    target_counts = {
        track: sum(1 for goal in active_by_vehicle.values() if str(goal.get("targetTrack") or "") == track)
        for track in sorted(PHASE3_BLOCK_TARGET_TRACKS)
    }
    risk = _score_phase3_risk(
        active_vehicle_count=len(active_by_vehicle),
        source_track_count=len(source_tracks),
        target_counts=target_counts,
        source_blocks=source_blocks,
        active_by_vehicle=active_by_vehicle,
        current_by_vehicle=current_by_vehicle,
    )
    sequence_plan = _build_target_sequence_plan(
        active_by_vehicle=active_by_vehicle,
        source_blocks=source_blocks,
        depot_track_sequences=depot_track_sequences or {},
        current_spot_assignments=current_spot_assignments or {},
        vehicle_by_no=vehicle_by_no,
        yard_mode=yard_mode,
        reserved_spot_codes=reserved_spot_codes or frozenset(),
    )
    risk["shouldUseExecutionPlan"] = bool(
        risk["score"] >= 6
        and sequence_plan.get("available") is True
        and sequence_plan.get("valid") is False
    )
    should_enable = (
        len(active_by_vehicle) >= min_active_vehicle_count
        or len(source_tracks) >= min_source_track_count
    )
    wave_plans = _build_wave_plans(source_blocks) if should_enable else []
    diagnostics = {
        "enabled": bool(wave_plans),
        "activeVehicleCount": len(active_by_vehicle),
        "sourceTrackCount": len(source_tracks),
        "targetTrackCount": len(target_tracks),
        "sourceTracks": sorted(source_tracks),
        "targetTracks": sorted(target_tracks),
        "targetCounts": target_counts,
        "sourceBlocks": source_blocks,
        "sourceBlockCount": len(source_blocks),
        "activeVehicleNos": sorted(active_by_vehicle),
        "coveredVehicleNos": covered_vehicle_nos,
        "coveredVehicleCount": len(covered_vehicle_nos),
        "hiddenActiveVehicleNos": hidden_active_vehicle_nos,
        "hiddenActiveVehicleCount": len(hidden_active_vehicle_nos),
        "missingVehicleNos": missing_vehicle_nos,
        "missingVehicleCount": len(missing_vehicle_nos),
        "allActiveCoveredByFrontier": set(covered_vehicle_nos) == set(active_by_vehicle),
        "waveCount": len(wave_plans),
        "risk": risk,
        "targetSequencePlan": sequence_plan,
        "minActiveVehicleCount": min_active_vehicle_count,
        "minSourceTrackCount": min_source_track_count,
    }
    return Phase3BlockPlan(wave_plans=wave_plans, diagnostics=diagnostics)


def _build_source_blocks(
    *,
    active_by_vehicle: dict[str, dict[str, Any]],
    track_sequences: dict[str, list[str] | tuple[str, ...]],
    current_by_vehicle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    hidden: set[str] = set()
    for source_track in sorted(track_sequences):
        sequence = [str(vehicle_no) for vehicle_no in track_sequences.get(source_track, ())]
        current_vehicle_nos: list[str] = []
        current_target = ""
        block_index = 1
        frontier_closed = False

        def flush() -> None:
            nonlocal current_vehicle_nos, current_target, block_index
            if not current_vehicle_nos:
                return
            blocks.append(
                {
                    "blockId": f"PHASE3::{source_track}::{block_index}",
                    "sourceTrack": source_track,
                    "targetTrack": current_target,
                    "vehicleNos": list(current_vehicle_nos),
                    "vehicleCount": len(current_vehicle_nos),
                    "sourceOrders": [
                        str(current_by_vehicle.get(vehicle_no, {}).get("order") or "")
                        for vehicle_no in current_vehicle_nos
                    ],
                }
            )
            seen.update(current_vehicle_nos)
            block_index += 1
            current_vehicle_nos = []
            current_target = ""

        for vehicle_no in sequence:
            goal = active_by_vehicle.get(vehicle_no)
            if goal is None:
                flush()
                frontier_closed = True
                continue
            if frontier_closed:
                hidden.add(vehicle_no)
                continue
            target_track = str(goal.get("targetTrack") or "")
            if current_vehicle_nos and target_track != current_target:
                flush()
            if not current_vehicle_nos:
                current_target = target_track
            current_vehicle_nos.append(vehicle_no)
        flush()

    missing_vehicle_nos = [
        vehicle_no for vehicle_no in active_by_vehicle
        if vehicle_no not in seen and vehicle_no not in hidden
    ]
    return {
        "blocks": blocks,
        "coveredVehicleNos": sorted(seen),
        "hiddenActiveVehicleNos": sorted(hidden),
        "missingVehicleNos": sorted(missing_vehicle_nos),
    }


def _score_phase3_risk(
    *,
    active_vehicle_count: int,
    source_track_count: int,
    target_counts: dict[str, int],
    source_blocks: list[dict[str, Any]],
    active_by_vehicle: dict[str, dict[str, Any]],
    current_by_vehicle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    if active_vehicle_count >= 14:
        score += 3
        reasons.append("active_vehicle_count_ge_14")
    elif active_vehicle_count >= 10:
        score += 1
        reasons.append("active_vehicle_count_ge_10")
    if source_track_count >= 4:
        score += 2
        reasons.append("source_track_count_ge_4")
    short_full_tracks = [
        track for track in ("修1", "修2")
        if target_counts.get(track, 0) >= 5
    ]
    if short_full_tracks:
        score += 2 * len(short_full_tracks)
        reasons.append("short_depot_track_full:" + ",".join(short_full_tracks))
    depot_target_count = sum(
        1 for track in ("修1", "修2", "修3", "修4")
        if target_counts.get(track, 0) > 0
    )
    if depot_target_count >= 3:
        score += 1
        reasons.append("multiple_depot_targets")
    split_source_tracks = {
        str(block.get("sourceTrack") or "")
        for block in source_blocks
        if sum(1 for item in source_blocks if item.get("sourceTrack") == block.get("sourceTrack")) > 1
    }
    if split_source_tracks:
        score += len(split_source_tracks)
        reasons.append("source_target_interleaving:" + ",".join(sorted(split_source_tracks)))
    factory_targets = [
        str(goal.get("targetTrack") or "")
        for vehicle_no, goal in active_by_vehicle.items()
        if str(current_by_vehicle.get(vehicle_no, {}).get("repairProcess") or "") == "厂修"
    ]
    if any(track in {"修1", "修2"} for track in factory_targets):
        score += 1
        reasons.append("factory_repair_on_short_depot_track")
    return {
        "score": score,
        "reasons": reasons,
        "shouldUseExecutionPlan": score >= 6,
    }


def _build_target_sequence_plan(
    *,
    active_by_vehicle: dict[str, dict[str, Any]],
    source_blocks: list[dict[str, Any]],
    depot_track_sequences: dict[str, list[str]],
    current_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, NormalizedVehicle] | None,
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> dict[str, Any]:
    if vehicle_by_no is None:
        return {
            "available": False,
            "reason": "vehicle_by_no_missing",
        }
    inbound_by_target: dict[str, list[str]] = {track: [] for track in sorted(PHASE3_DEPOT_TARGET_TRACKS)}
    for block in source_blocks:
        target_track = str(block.get("targetTrack") or "")
        if target_track not in PHASE3_DEPOT_TARGET_TRACKS:
            continue
        inbound_by_target.setdefault(target_track, []).extend(
            str(vehicle_no) for vehicle_no in block.get("vehicleNos") or ()
        )
    track_plans: dict[str, Any] = {}
    all_valid = True
    for target_track in sorted(PHASE3_DEPOT_TARGET_TRACKS):
        existing_sequence = [
            vehicle_no
            for vehicle_no in depot_track_sequences.get(target_track, ())
            if vehicle_no in vehicle_by_no
        ]
        inbound_sequence = [
            vehicle_no
            for vehicle_no in inbound_by_target.get(target_track, ())
            if vehicle_no in active_by_vehicle
        ]
        final_sequence = list(inbound_sequence) + list(existing_sequence)
        realigned = realign_spots_for_track_order(
            vehicle_nos_in_order=final_sequence,
            vehicle_by_no=vehicle_by_no,
            target_track=target_track,
            yard_mode=yard_mode,
            current_spot_assignments=current_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        capacity = len(list_track_spots(target_track, yard_mode))
        capacity_valid = len(final_sequence) <= capacity
        valid = realigned is not None and capacity_valid
        all_valid = all_valid and valid
        track_plans[target_track] = {
            "existingVehicleNos": existing_sequence,
            "inboundVehicleNos": inbound_sequence,
            "finalVehicleNos": final_sequence,
            "valid": valid,
            "capacity": capacity,
            "capacityValid": capacity_valid,
            "assignedSpots": {
                vehicle_no: realigned[vehicle_no]
                for vehicle_no in final_sequence
                if realigned is not None and vehicle_no in realigned
            },
            "inboundVehicleCount": len(inbound_sequence),
            "finalVehicleCount": len(final_sequence),
        }
    return {
        "available": True,
        "valid": all_valid,
        "tracks": track_plans,
    }


def _build_wave_plans(source_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wave_plans: list[dict[str, Any]] = []
    for index, block in enumerate(source_blocks, start=1):
        target_track = str(block.get("targetTrack") or "")
        source_track = str(block.get("sourceTrack") or "")
        active_goals = {
            str(vehicle_no): {
                "vehicleNo": str(vehicle_no),
                "targetTrack": target_track,
                "targetMode": "TRACK",
                "targetSource": "PHASE3_BLOCK_PLAN",
                "isSpotting": "",
            }
            for vehicle_no in block.get("vehicleNos") or ()
        }
        if not active_goals:
            continue
        wave_plans.append(
            {
                "waveName": f"phase3_block_{index:02d}_{source_track}_{target_track}",
                "waveRole": "PHASE3_SOURCE_BLOCK_TO_DEPOT",
                "waveSourceTrack": source_track,
                "waveTargetTrack": target_track,
                "waveWeight": max(0.05, float(block.get("vehicleCount") or 1)),
                "sourceBlock": dict(block),
                "activeGoalsByVehicle": active_goals,
            }
        )
    total_weight = sum(float(item.get("waveWeight") or 0.0) for item in wave_plans)
    if total_weight > 0:
        for item in wave_plans:
            item["waveWeight"] = float(item["waveWeight"]) / total_weight
    return wave_plans
