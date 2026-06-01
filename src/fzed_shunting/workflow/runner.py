from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData, clone_master_with_blocked_branches
from fzed_shunting.io.normalize_input import SOURCE_TRACK_ALIASES, normalize_plan_input
from fzed_shunting.verify.replay import ReplayState, build_initial_state
from fzed_shunting.workflow.l7_closed_topology_mode import (
    PHASE4_DYNAMIC_CURRENT_HOLD,
    PHASE4_RESIDUAL_CLEANUP,
    build_l7_closed_topology_workflow_payload,
    is_l7_closed_topology_mode,
)

if TYPE_CHECKING:
    from fzed_shunting.demo.view_model import DemoViewModel


class WorkflowStageResult(BaseModel):
    name: str
    description: str = ""
    input_payload: dict
    view: Any | None = None


class WorkflowResult(BaseModel):
    stage_count: int
    stages: list[WorkflowStageResult] = Field(default_factory=list)


class WorkflowStageFailure(RuntimeError):
    def __init__(
        self,
        *,
        failed_stage_name: str,
        failed_stage_index: int,
        total_stage_count: int,
        completed_stage_names: list[str],
        cause_message: str,
        stage_input_summary: dict[str, Any] | None = None,
    ) -> None:
        self.failed_stage_name = failed_stage_name
        self.failed_stage_index = failed_stage_index
        self.total_stage_count = total_stage_count
        self.completed_stage_names = list(completed_stage_names)
        self.cause_message = cause_message
        self.stage_input_summary = dict(stage_input_summary or {})
        super().__init__(
            f"Workflow failed at stage {failed_stage_index}/{total_stage_count} "
            f"({failed_stage_name}): {cause_message}"
        )


def solve_workflow(
    master: MasterData,
    payload: dict,
    *,
    solver: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    time_budget_ms: float | None = None,
    use_validation_recovery: bool = True,
    diagnose_front_search_only: bool = False,
) -> WorkflowResult:
    from fzed_shunting.demo.view_model import build_demo_view_model

    if is_l7_closed_topology_mode(payload) and "workflowStages" not in payload:
        payload = build_l7_closed_topology_workflow_payload(master, payload)

    workflow_stages = payload.get("workflowStages")
    if not isinstance(workflow_stages, list) or not workflow_stages:
        raise ValueError("workflowStages must be a non-empty list")

    track_info = payload.get("trackInfo")
    if not isinstance(track_info, list) or not track_info:
        raise ValueError("trackInfo must be a non-empty list")

    initial_vehicle_info = payload.get("initialVehicleInfo")
    if not isinstance(initial_vehicle_info, list) or not initial_vehicle_info:
        raise ValueError("initialVehicleInfo must be a non-empty list")

    input_loco_track_name = payload.get("locoTrackName") or "机库"
    current_vehicle_info = [
        {
            "trackName": str(item["trackName"]),
            "order": str(item["order"]),
            "vehicleModel": str(item["vehicleModel"]),
            "vehicleNo": str(item["vehicleNo"]),
            "repairProcess": str(item["repairProcess"]),
            "vehicleLength": item["vehicleLength"],
            "vehicleAttributes": str(item.get("vehicleAttributes", "")),
        }
        for item in initial_vehicle_info
    ]
    vehicle_meta = {item["vehicleNo"]: dict(item) for item in current_vehicle_info}
    current_state = ReplayState(
        track_sequences={},
        loco_track_name=input_loco_track_name,
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )

    stages: list[WorkflowStageResult] = []
    for index, stage in enumerate(workflow_stages, start=1):
        stage_name = str(stage.get("name", f"stage_{index}"))
        stage_payload: dict[str, Any] | None = None
        stage_master = master
        try:
            stage = _resolve_dynamic_stage(
                stage=stage,
                current_vehicle_info=current_vehicle_info,
                current_state=current_state,
            )
            route_policy = stage.get("routePolicy") if isinstance(stage, dict) else None
            stage_master = _master_for_stage(master, route_policy)
            stage_payload = _build_stage_payload(
                track_info=track_info,
                current_vehicle_info=current_vehicle_info,
                vehicle_meta=vehicle_meta,
                stage=stage,
                loco_track_name=current_state.loco_track_name,
            )
            if index == 1:
                current_state = build_initial_state(
                    normalize_plan_input(
                        stage_payload,
                        master,
                        allow_internal_loco_tracks=True,
                    )
                )
            view = build_demo_view_model(
                stage_master,
                stage_payload,
                solver=solver,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
                time_budget_ms=time_budget_ms,
                initial_state_override=current_state,
                use_validation_recovery=use_validation_recovery,
                diagnose_front_search_only=diagnose_front_search_only,
            )
            stages.append(
                WorkflowStageResult(
                    name=stage_name,
                    description=str(stage.get("description", "")),
                    input_payload=stage_payload,
                    view=view,
                )
            )
            current_vehicle_info = _next_vehicle_info(
                stage_payload=stage_payload,
                stage_view=view,
            )
            current_state = ReplayState.model_validate({
                "track_sequences": view.steps[-1].track_sequences,
                "loco_track_name": view.steps[-1].loco_track_name,
                "weighed_vehicle_nos": set(view.steps[-1].weighed_vehicle_nos),
                "spot_assignments": view.steps[-1].spot_assignments,
            })
            for item in current_vehicle_info:
                vehicle_meta[item["vehicleNo"]]["trackName"] = item["trackName"]
                vehicle_meta[item["vehicleNo"]]["order"] = item["order"]
        except WorkflowStageFailure:
            raise
        except Exception as exc:
            raise WorkflowStageFailure(
                failed_stage_name=stage_name,
                failed_stage_index=index,
                total_stage_count=len(workflow_stages),
                completed_stage_names=[result.name for result in stages],
                cause_message=str(exc),
                stage_input_summary=_summarize_stage_payload(
                    master=stage_master,
                    stage_payload=stage_payload,
                ),
            ) from exc

    return WorkflowResult(
        stage_count=len(stages),
        stages=stages,
    )
def _build_stage_payload(
    *,
    track_info: list[dict],
    current_vehicle_info: list[dict],
    vehicle_meta: dict[str, dict],
    stage: dict,
    loco_track_name: str,
) -> dict:
    vehicle_goals = stage.get("vehicleGoals")
    if not isinstance(vehicle_goals, list) or not vehicle_goals:
        raise ValueError("each workflow stage must contain non-empty vehicleGoals")

    goal_by_vehicle = {
        str(item["vehicleNo"]): item
        for item in vehicle_goals
    }
    if set(goal_by_vehicle) != {item["vehicleNo"] for item in current_vehicle_info}:
        raise ValueError("each workflow stage must define a goal for every current vehicle")

    stage_vehicle_info: list[dict] = []
    for current in current_vehicle_info:
        goal = goal_by_vehicle[current["vehicleNo"]]
        base = dict(vehicle_meta[current["vehicleNo"]])
        base["trackName"] = current["trackName"]
        base["order"] = current["order"]
        base["targetTrack"] = str(goal["targetTrack"])
        base["isSpotting"] = str(goal.get("isSpotting", ""))
        for key in ("targetMode", "targetAreaCode", "targetSpotCode", "targetSource"):
            if key in goal:
                base[key] = goal[key]
        stage_vehicle_info.append(base)

    stage_payload = {
        "trackInfo": [dict(item) for item in track_info],
        "vehicleInfo": stage_vehicle_info,
        "locoTrackName": loco_track_name,
        "workflowStageName": str(stage.get("name", "")),
    }
    if "routePolicy" in stage:
        stage_payload["routePolicy"] = dict(stage.get("routePolicy") or {})
    if "stagePolicy" in stage:
        stage_payload["stagePolicy"] = dict(stage.get("stagePolicy") or {})
    return stage_payload


def _resolve_dynamic_stage(
    *,
    stage: dict,
    current_vehicle_info: list[dict],
    current_state: ReplayState,
) -> dict:
    stage_policy = stage.get("stagePolicy") if isinstance(stage, dict) else None
    stage_mode = str((stage_policy or {}).get("stageMode") or "")
    if stage_mode != PHASE4_RESIDUAL_CLEANUP:
        return stage

    current_by_vehicle = {
        str(item["vehicleNo"]): dict(item)
        for item in current_vehicle_info
    }
    resolved_goals: list[dict[str, Any]] = []
    for goal in list(stage.get("vehicleGoals") or []):
        if goal.get("targetSource") != PHASE4_DYNAMIC_CURRENT_HOLD:
            resolved_goals.append(dict(goal))
            continue
        vehicle_no = str(goal["vehicleNo"])
        current = current_by_vehicle[vehicle_no]
        resolved_goals.append(
            _build_dynamic_current_hold_goal(
                vehicle_no=vehicle_no,
                current_track=str(current["trackName"]),
                current_state=current_state,
            )
        )
    resolved_stage = dict(stage)
    resolved_stage["vehicleGoals"] = resolved_goals
    return resolved_stage


def _build_dynamic_current_hold_goal(
    *,
    vehicle_no: str,
    current_track: str,
    current_state: ReplayState,
) -> dict[str, Any]:
    current_spot = current_state.spot_assignments.get(vehicle_no)
    if current_spot:
        current_spot_str = str(current_spot)
        return {
            "vehicleNo": vehicle_no,
            "targetTrack": current_track,
            "targetMode": "SPOT",
            "targetSpotCode": current_spot_str,
            "targetSource": PHASE4_DYNAMIC_CURRENT_HOLD,
            "isSpotting": "迎检" if current_spot_str[1:] in {"06", "07"} else current_spot_str,
        }
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": current_track,
        "targetMode": "SNAPSHOT",
        "targetSource": PHASE4_DYNAMIC_CURRENT_HOLD,
        "isSpotting": "",
    }


def _summarize_stage_payload(
    *,
    master: MasterData,
    stage_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not stage_payload:
        return {}
    vehicle_info = list(stage_payload.get("vehicleInfo") or [])
    if not vehicle_info:
        return {}
    stage_policy = dict(stage_payload.get("stagePolicy") or {})
    moved = [
        item
        for item in vehicle_info
        if _normalize_stage_track_name(str(item["trackName"])) != str(item["targetTrack"])
    ]
    source_counts = Counter(_normalize_stage_track_name(str(item["trackName"])) for item in moved)
    target_counts = Counter(str(item["targetTrack"]) for item in moved)
    target_lengths: dict[str, float] = {}
    for item in moved:
        target_track = str(item["targetTrack"])
        target_lengths[target_track] = round(
            target_lengths.get(target_track, 0.0) + float(item["vehicleLength"]),
            1,
        )
    summary: dict[str, Any] = {
        "workflow_stage_name": str(stage_payload.get("workflowStageName") or ""),
        "stage_mode": str(stage_policy.get("stageMode") or ""),
        "vehicle_count": len(vehicle_info),
        "active_move_count": len(moved),
        "active_move_total_length_m": round(
            sum(float(item["vehicleLength"]) for item in moved),
            1,
        ),
        "active_move_source_counts": dict(sorted(source_counts.items())),
        "active_move_target_counts": dict(sorted(target_counts.items())),
        "active_move_target_lengths_m": target_lengths,
    }
    if summary["stage_mode"] == "PHASE1_PRE_REPAIR_BUFFERING":
        summary.update(_summarize_phase1_stage_payload(master=master, vehicle_info=vehicle_info))
        phase1_diagnostics = stage_policy.get("phase1Diagnostics")
        if isinstance(phase1_diagnostics, dict):
            summary["phase1_backbone_diagnostics"] = dict(phase1_diagnostics)
            summary["phase1_selected_package_count"] = len(phase1_diagnostics.get("selectedPackageIds") or [])
            summary["phase1_selected_vehicle_count"] = phase1_diagnostics.get("selectedVehicleCount")
            summary["phase1_selected_total_length_m"] = phase1_diagnostics.get("selectedTotalLengthM")
            summary["phase1_deferred_vehicle_count"] = len(phase1_diagnostics.get("deferredVehicleNos") or [])
            summary["phase1_deferred_vehicle_nos"] = list(phase1_diagnostics.get("deferredVehicleNos") or [])
            summary["phase1_buffer_lengths_m"] = dict(phase1_diagnostics.get("bufferLengthsM") or {})
    return summary


def _normalize_stage_track_name(track_name: str) -> str:
    return SOURCE_TRACK_ALIASES.get(track_name, track_name)


def _summarize_phase1_stage_payload(
    *,
    master: MasterData,
    vehicle_info: list[dict[str, Any]],
) -> dict[str, Any]:
    buffer_tracks = {"机南", "机棚", "机北1", "机北2", "机北3"}
    buffer_goals = [
        item
        for item in vehicle_info
        if str(item.get("targetSource") or "") == "PHASE1_BACKBONE_PLACE"
    ]
    clear_ji_goals = [
        item
        for item in vehicle_info
        if str(item.get("targetSource") or "") == "PHASE1_CLEAR_JI"
    ]
    clear_cun4bei_goals = [
        item
        for item in vehicle_info
        if str(item.get("targetSource") or "") == "PHASE1_CLEAR_CUN4"
    ]
    temp_repark_goals = [
        item
        for item in vehicle_info
        if str(item.get("targetSource") or "") == "PHASE1_DEPOT_BLOCKER_CLEAR"
    ]
    local_finish_goals = [
        item
        for item in vehicle_info
        if str(item.get("targetSource") or "") == "PHASE1_LOCAL_FINISH"
    ]
    source_counts = Counter(str(item["trackName"]) for item in buffer_goals)
    source_lengths: dict[str, float] = {}
    target_lengths: dict[str, float] = {}
    for item in buffer_goals:
        source_track = str(item["trackName"])
        target_track = str(item["targetTrack"])
        source_lengths[source_track] = round(
            source_lengths.get(source_track, 0.0) + float(item["vehicleLength"]),
            1,
        )
        target_lengths[target_track] = round(
            target_lengths.get(target_track, 0.0) + float(item["vehicleLength"]),
            1,
        )
    buffer_capacity = {
        track: round(float(master.tracks[track].effective_length_m), 1)
        for track in sorted(buffer_tracks)
    }
    overflow_by_track = {
        track: round(max(0.0, target_lengths.get(track, 0.0) - buffer_capacity[track]), 1)
        for track in sorted(buffer_tracks)
    }
    return {
        "phase1_buffer_vehicle_count": len(buffer_goals),
        "phase1_buffer_total_length_m": round(
            sum(float(item["vehicleLength"]) for item in buffer_goals),
            1,
        ),
        "phase1_backbone_vehicle_count": len(buffer_goals),
        "phase1_buffer_source_counts": dict(sorted(source_counts.items())),
        "phase1_buffer_source_lengths_m": source_lengths,
        "phase1_buffer_target_lengths_m": target_lengths,
        "phase1_buffer_total_capacity_m": round(sum(buffer_capacity.values()), 1),
        "phase1_buffer_capacity_by_track_m": buffer_capacity,
        "phase1_buffer_overflow_by_track_m": overflow_by_track,
        "phase1_buffer_total_overflow_m": round(sum(overflow_by_track.values()), 1),
        "phase1_clear_ji_vehicle_count": len(clear_ji_goals),
        "phase1_clear_ji_source_counts": dict(
            sorted(Counter(str(item["trackName"]) for item in clear_ji_goals).items())
        ),
        "phase1_clear_cun4bei_vehicle_count": len(clear_cun4bei_goals),
        "phase1_temp_repark_vehicle_count": len(temp_repark_goals),
        "phase1_local_finish_vehicle_count": len(local_finish_goals),
        "phase1_existing_ji_vehicle_count": sum(
            1 for item in vehicle_info if str(item["trackName"]) in buffer_tracks
        ),
        "phase1_existing_ji_non_depot_vehicle_count": sum(
            1
            for item in vehicle_info
            if str(item["trackName"]) in buffer_tracks
            and str(item.get("targetSource") or "") == "PHASE1_CLEAR_JI"
        ),
    }


def _next_vehicle_info(
    *,
    stage_payload: dict,
    stage_view: DemoViewModel,
) -> list[dict]:
    final_sequences = stage_view.steps[-1].track_sequences
    previous_info = {
        item["vehicleNo"]: item
        for item in stage_payload["vehicleInfo"]
    }
    next_vehicle_info: list[dict] = []
    for track_name in sorted(final_sequences):
        for index, vehicle_no in enumerate(final_sequences[track_name], start=1):
            base = dict(previous_info[vehicle_no])
            base["trackName"] = track_name
            base["order"] = str(index)
            next_vehicle_info.append(base)
    return next_vehicle_info


def _master_for_stage(master: MasterData, route_policy: dict[str, Any] | None) -> MasterData:
    blocked_branches = ()
    if isinstance(route_policy, dict):
        blocked_branches = tuple(route_policy.get("blockedBranches") or ())
    return clone_master_with_blocked_branches(master, blocked_branches)
