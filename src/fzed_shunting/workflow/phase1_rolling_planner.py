from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fzed_shunting.workflow.l7_closed_topology_mode import (
    JI_BUFFER_TRACKS,
    PHASE1_BLOCKER_BUCKET_WORK,
    PHASE1_BLOCKER_BUCKET_YARD,
    PHASE1_TEMP_PARKING_TRACKS,
)

PHASE1_ROLLING_MAX_ACTIVE_VEHICLES = 12
PHASE1_ROLLING_MAX_CANDIDATES = 8


@dataclass(frozen=True)
class Phase1RollingCandidate:
    wave: dict[str, Any]
    score: tuple[Any, ...]


def build_phase1_rolling_candidates(
    *,
    runtime_stage: dict[str, Any],
    selected_block_ids: frozenset[str] | None = None,
) -> tuple[Phase1RollingCandidate, ...]:
    stage_policy = dict(runtime_stage.get("stagePolicy") or {})
    diagnostics = dict(stage_policy.get("phase1Diagnostics") or {})
    phase1_blocks = {
        str(block.get("blockId") or ""): dict(block)
        for block in list(diagnostics.get("phase1Blocks") or [])
        if str(block.get("blockId") or "")
    }
    if selected_block_ids is None:
        selected_block_ids = frozenset(
            block_id
            for block_id, block in phase1_blocks.items()
            if bool(block.get("selectedBackbone"))
        )
    if not selected_block_ids:
        return tuple()
    goal_by_vehicle = {
        str(goal.get("vehicleNo") or ""): dict(goal)
        for goal in list(runtime_stage.get("vehicleGoals") or [])
        if str(goal.get("vehicleNo") or "")
    }
    macro_candidates = _build_macro_task_candidates(
        diagnostics=diagnostics,
        phase1_blocks=phase1_blocks,
        selected_block_ids=selected_block_ids,
        goal_by_vehicle=goal_by_vehicle,
    )
    if macro_candidates:
        return macro_candidates
    return _build_source_frontier_candidates(
        diagnostics=diagnostics,
        phase1_blocks=phase1_blocks,
        selected_block_ids=selected_block_ids,
        goal_by_vehicle=goal_by_vehicle,
    )


def _build_source_frontier_candidates(
    *,
    diagnostics: dict[str, Any],
    phase1_blocks: dict[str, dict[str, Any]],
    selected_block_ids: frozenset[str],
    goal_by_vehicle: dict[str, dict[str, Any]],
) -> tuple[Phase1RollingCandidate, ...]:
    candidates: list[Phase1RollingCandidate] = []
    for source_plan in list(diagnostics.get("sourceTrackPlans") or []):
        source_track = str(dict(source_plan).get("sourceTrack") or "")
        ordered_block_ids = [
            str(block_id)
            for block_id in list(dict(source_plan).get("blockIds") or [])
            if str(block_id) in selected_block_ids
        ]
        if not source_track or not ordered_block_ids:
            continue
        frontier_block = _first_ready_block(
            ordered_block_ids=ordered_block_ids,
            phase1_blocks=phase1_blocks,
            selected_block_ids=selected_block_ids,
        )
        if frontier_block is None:
            continue
        block_vehicle_nos = tuple(
            str(vehicle_no)
            for vehicle_no in list(frontier_block.get("vehicleNos") or [])
            if str(vehicle_no)
        )
        if not block_vehicle_nos or len(block_vehicle_nos) > PHASE1_ROLLING_MAX_ACTIVE_VEHICLES:
            continue
        target_tracks = _candidate_target_tracks(
            block=frontier_block,
            source_track=source_track,
            goal_by_vehicle=goal_by_vehicle,
        )
        for target_track in target_tracks:
            wave = _build_candidate_wave(
                block=frontier_block,
                source_track=source_track,
                vehicle_nos=block_vehicle_nos,
                target_track=target_track,
                goal_by_vehicle=goal_by_vehicle,
            )
            candidates.append(
                Phase1RollingCandidate(
                    wave=wave,
                    score=_candidate_score(
                        block=frontier_block,
                        source_track=source_track,
                        target_track=target_track,
                        selected_source_block_count=len(ordered_block_ids),
                    ),
                )
            )
    candidates.sort(key=lambda candidate: candidate.score)
    return tuple(candidates[:PHASE1_ROLLING_MAX_CANDIDATES])


def _build_macro_task_candidates(
    *,
    diagnostics: dict[str, Any],
    phase1_blocks: dict[str, dict[str, Any]],
    selected_block_ids: frozenset[str],
    goal_by_vehicle: dict[str, dict[str, Any]],
) -> tuple[Phase1RollingCandidate, ...]:
    macro_tasks = [
        dict(item)
        for item in list(diagnostics.get("phase1MacroTasks") or [])
        if dict(item)
    ]
    if not macro_tasks:
        return tuple()
    candidates: list[Phase1RollingCandidate] = []
    for task_index, macro_task in enumerate(macro_tasks):
        ordered_block_ids = [
            str(block_id)
            for block_id in list(macro_task.get("blockIds") or [])
            if str(block_id) in selected_block_ids
        ]
        if not ordered_block_ids:
            continue
        frontier_block = _first_ready_block(
            ordered_block_ids=ordered_block_ids,
            phase1_blocks=phase1_blocks,
            selected_block_ids=selected_block_ids,
        )
        if frontier_block is None:
            continue
        source_track = str(macro_task.get("sourceTrack") or frontier_block.get("sourceTrack") or "")
        block_vehicle_nos = tuple(
            str(vehicle_no)
            for vehicle_no in list(frontier_block.get("vehicleNos") or [])
            if str(vehicle_no)
        )
        if not source_track or not block_vehicle_nos or len(block_vehicle_nos) > PHASE1_ROLLING_MAX_ACTIVE_VEHICLES:
            continue
        target_tracks = _candidate_target_tracks(
            block=frontier_block,
            source_track=source_track,
            goal_by_vehicle=goal_by_vehicle,
        )
        for target_track in target_tracks:
            wave = _build_candidate_wave(
                block=frontier_block,
                source_track=source_track,
                vehicle_nos=block_vehicle_nos,
                target_track=target_track,
                goal_by_vehicle=goal_by_vehicle,
                macro_task=macro_task,
            )
            candidates.append(
                Phase1RollingCandidate(
                    wave=wave,
                    score=_macro_candidate_score(
                        macro_task=macro_task,
                        task_index=task_index,
                        block=frontier_block,
                        target_track=target_track,
                        remaining_task_block_count=len(ordered_block_ids),
                    ),
                )
            )
    candidates.sort(key=lambda candidate: candidate.score)
    return tuple(candidates[:PHASE1_ROLLING_MAX_CANDIDATES])


def phase1_rolling_selected_block_ids(stage: dict[str, Any]) -> frozenset[str]:
    diagnostics = dict(((stage.get("stagePolicy") or {}).get("phase1Diagnostics")) or {})
    return frozenset(
        str(block.get("blockId") or "")
        for block in list(diagnostics.get("phase1Blocks") or [])
        if str(block.get("blockId") or "")
        and (bool(block.get("selectedBackbone")) or bool(block.get("selectedFinish")))
    )


def _first_ready_block(
    *,
    ordered_block_ids: list[str],
    phase1_blocks: dict[str, dict[str, Any]],
    selected_block_ids: frozenset[str],
) -> dict[str, Any] | None:
    for block_id in ordered_block_ids:
        block = phase1_blocks.get(block_id)
        if block is None:
            continue
        if any(str(predecessor_id) in selected_block_ids for predecessor_id in list(block.get("requiredPredecessorIds") or [])):
            continue
        return block
    return None


def _candidate_target_tracks(
    *,
    block: dict[str, Any],
    source_track: str,
    goal_by_vehicle: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    vehicle_nos = [str(vehicle_no) for vehicle_no in list(block.get("vehicleNos") or [])]
    if bool(block.get("usesBuffer")):
        target_track = str(goal_by_vehicle.get(vehicle_nos[0], {}).get("targetTrack") or "")
        preferred_tracks = [
            str(track)
            for track in list(block.get("bufferPreference") or [])
            if str(track) and str(track) != source_track and str(track) in JI_BUFFER_TRACKS
        ]
        if target_track:
            preferred_tracks.insert(0, target_track)
        return tuple(dict.fromkeys(track for track in preferred_tracks if track))
    target_source = str(block.get("targetSource") or "")
    if target_source == "PHASE1_LOCAL_FINISH":
        target_track = str(goal_by_vehicle.get(vehicle_nos[0], {}).get("targetTrack") or block.get("targetTrack") or "")
        return (target_track,) if target_track else tuple()
    if target_source == PHASE1_BLOCKER_BUCKET_WORK:
        preferred = ("调棚", "预修", "调北", "存5南", "存3", "存2", "存1")
    elif target_source == PHASE1_BLOCKER_BUCKET_YARD:
        preferred = _storage_bucket_order(source_track)
    else:
        target_track = str(goal_by_vehicle.get(vehicle_nos[0], {}).get("targetTrack") or block.get("targetTrack") or "")
        return (target_track,) if target_track else tuple()
    return tuple(track for track in preferred if track and track != source_track)


def _storage_bucket_order(source_track: str) -> tuple[str, ...]:
    if source_track == "存5北":
        return ("存5南", "调北", "存3", "存2", "存1", "预修", "调棚")
    if source_track == "存5南":
        return ("存5北", "调北", "存3", "存2", "存1", "预修", "调棚")
    return tuple(PHASE1_TEMP_PARKING_TRACKS)


def _build_candidate_wave(
    *,
    block: dict[str, Any],
    source_track: str,
    vehicle_nos: tuple[str, ...],
    target_track: str,
    goal_by_vehicle: dict[str, dict[str, Any]],
    macro_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block_id = str(block.get("blockId") or "")
    target_source = str(block.get("targetSource") or "")
    active_goals = []
    for vehicle_no in vehicle_nos:
        base_goal = dict(goal_by_vehicle.get(vehicle_no) or {})
        if target_track and not bool(block.get("usesBuffer")):
            base_goal.update(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": target_track,
                    "targetMode": "AREA",
                    "targetAreaCode": f"STAGE::{target_source or 'PHASE1_ROLLING_TEMP'}",
                    "targetSource": target_source or "PHASE1_ROLLING_TEMP",
                    "isSpotting": "",
                }
            )
        active_goals.append(base_goal)
    buffer_assignment = {
        vehicle_no: target_track
        for vehicle_no in vehicle_nos
        if target_track in JI_BUFFER_TRACKS and bool(block.get("usesBuffer"))
    }
    macro_task = dict(macro_task or {})
    macro_task_id = str(macro_task.get("taskId") or "")
    macro_task_block_ids = [
        str(block_id)
        for block_id in list(macro_task.get("blockIds") or [])
        if str(block_id)
    ]
    macro_task_wave_chunks = [
        dict(chunk)
        for chunk in list(macro_task.get("waveChunks") or [])
        if dict(chunk)
    ]
    wave = {
        "waveName": f"phase1_roll_{source_track}_{block_id}_{target_track}",
        "waveRole": "phase1_rolling",
        "waveType": "phase1_rolling",
        "selectedSourceTrack": source_track,
        "selectedBlockIds": [block_id],
        "selectedVehicleNos": list(vehicle_nos),
        "packageAssignments": dict(buffer_assignment),
        "layoutAssignments": dict(buffer_assignment),
        "packageTargetRanks": {},
        "layoutTargetRanks": {},
        "vehicleGoals": active_goals,
        "waveDiagnostics": {
            "waveName": "phase1_rolling",
            "waveRole": "phase1_rolling",
            "waveType": "phase1_rolling",
            "selectedSourceTrack": source_track,
            "selectedBlockIds": [block_id],
            "selectedVehicleCount": len(vehicle_nos),
            "targetTrack": target_track,
            "runtimeFrontierStrategy": "rolling_multi_candidate",
        },
    }
    if macro_task_id:
        diagnostics = dict(wave["waveDiagnostics"])
        diagnostics.update(
            {
                "macroTaskId": macro_task_id,
                "macroTaskBlockIds": macro_task_block_ids,
                "macroTaskWaveChunks": macro_task_wave_chunks,
                "macroTaskSourceRole": str(macro_task.get("sourceRole") or ""),
                "macroTaskScoreKey": list(macro_task.get("scoreKey") or []),
                "remainingMacroTaskBlockCount": len(macro_task_block_ids),
                "runtimeFrontierStrategy": "macro_task_frontier",
            }
        )
        wave["waveDiagnostics"] = diagnostics
    return wave


def _macro_candidate_score(
    *,
    macro_task: dict[str, Any],
    task_index: int,
    block: dict[str, Any],
    target_track: str,
    remaining_task_block_count: int,
) -> tuple[Any, ...]:
    block_score = _candidate_score(
        block=block,
        source_track=str(macro_task.get("sourceTrack") or block.get("sourceTrack") or ""),
        target_track=target_track,
        selected_source_block_count=remaining_task_block_count,
    )
    return (
        task_index,
        *_flat_sort_key(macro_task.get("scoreKey") or ()),
        -remaining_task_block_count,
        *block_score,
    )


def _flat_sort_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(item for nested in value for item in _flat_sort_key(nested))
    return (value,)


def _candidate_score(
    *,
    block: dict[str, Any],
    source_track: str,
    target_track: str,
    selected_source_block_count: int,
) -> tuple[Any, ...]:
    block_type = str(block.get("blockType") or "")
    vehicle_count = len(list(block.get("vehicleNos") or []))
    released_depot = int(block.get("releasedDepotVehicleCount") or 0)
    uses_buffer = bool(block.get("usesBuffer"))
    source_priority = {
        "存5北": 0,
        "存5南": 1,
        "存3": 2,
        "存2": 3,
        "存1": 4,
        "调棚": 5,
        "预修": 5,
        "调北": 6,
        "洗南": 7,
        "洗北": 7,
        "油": 7,
        "抛": 8,
    }.get(source_track, 9)
    bucket_risk = {
        "存5南": 0,
        "调北": 1,
        "存3": 2,
        "存2": 3,
        "存1": 4,
        "预修": 5,
        "调棚": 6,
    }.get(target_track, 3)
    return (
        0 if block_type in {"prefix_clear", "clear_cun4"} else 1,
        0 if source_track in {"存5北", "存5南"} and block_type == "prefix_clear" else 1,
        source_priority,
        -selected_source_block_count,
        -released_depot,
        0 if uses_buffer else 1,
        bucket_risk,
        vehicle_count,
        source_track,
        str(block.get("blockId") or ""),
        target_track,
    )
