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
            if _is_phase1_wave_stage(stage):
                view, current_vehicle_info, current_state = _solve_phase1_wave_stage(
                    stage_master=stage_master,
                    track_info=track_info,
                    vehicle_meta=vehicle_meta,
                    stage=stage,
                    stage_payload=stage_payload,
                    current_vehicle_info=current_vehicle_info,
                    current_state=current_state,
                    solver=solver,
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    time_budget_ms=time_budget_ms,
                    use_validation_recovery=use_validation_recovery,
                    diagnose_front_search_only=diagnose_front_search_only,
                )
            else:
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
            if not _is_phase1_wave_stage(stage):
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


def _is_phase1_wave_stage(stage: dict[str, Any]) -> bool:
    stage_policy = dict(stage.get("stagePolicy") or {})
    return (
        str(stage_policy.get("stageMode") or "") == "PHASE1_PRE_REPAIR_BUFFERING"
        and bool(stage_policy.get("phase1WavePlans"))
    )


def _solve_phase1_wave_stage(
    *,
    stage_master: MasterData,
    track_info: list[dict],
    vehicle_meta: dict[str, dict],
    stage: dict[str, Any],
    stage_payload: dict[str, Any],
    current_vehicle_info: list[dict[str, Any]],
    current_state: ReplayState,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None,
    use_validation_recovery: bool,
    diagnose_front_search_only: bool,
):
    from fzed_shunting.demo.view_model import (
        DemoHook,
        DemoSummary,
        DemoViewModel,
        build_demo_view_model,
    )

    wave_plans = list((stage.get("stagePolicy") or {}).get("phase1WavePlans") or [])
    if not wave_plans:
        raise ValueError("phase1WavePlans must be non-empty for wave stage solve")
    wave_weights = [0.52, 0.30, 0.18]
    working_vehicle_info = [dict(item) for item in current_vehicle_info]
    working_state = current_state
    wave_views: list[DemoViewModel] = []
    for index, wave_plan in enumerate(wave_plans):
        sub_stage_policy = dict(stage.get("stagePolicy") or {})
        sub_stage_policy.update({
            "packageAssignments": dict(wave_plan.get("packageAssignments") or {}),
            "layoutAssignments": dict(wave_plan.get("layoutAssignments") or {}),
            "packageTargetRanks": dict(wave_plan.get("packageTargetRanks") or {}),
            "layoutTargetRanks": dict(wave_plan.get("layoutTargetRanks") or {}),
            "phase1WaveActiveName": str(wave_plan.get("waveName") or f"wave_{index + 1}"),
            "phase1WaveDiagnostics": dict(wave_plan.get("waveDiagnostics") or {}),
        })
        sub_stage = {
            "name": str(stage.get("name", "phase1")),
            "description": str(stage.get("description", "")),
            "routePolicy": dict(stage.get("routePolicy") or {}),
            "stagePolicy": sub_stage_policy,
            "vehicleGoals": list(wave_plan.get("vehicleGoals") or []),
        }
        sub_stage_payload = _build_stage_payload(
            track_info=track_info,
            current_vehicle_info=working_vehicle_info,
            vehicle_meta=vehicle_meta,
            stage=sub_stage,
            loco_track_name=working_state.loco_track_name,
        )
        sub_time_budget_ms = None
        if time_budget_ms is not None:
            weight = wave_weights[index] if index < len(wave_weights) else wave_weights[-1]
            sub_time_budget_ms = max(5_000.0, float(time_budget_ms) * weight)
        view = build_demo_view_model(
            stage_master,
            sub_stage_payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            time_budget_ms=sub_time_budget_ms,
            initial_state_override=working_state,
            use_validation_recovery=use_validation_recovery,
            diagnose_front_search_only=diagnose_front_search_only,
        )
        wave_views.append(view)
        working_vehicle_info = _next_vehicle_info(
            stage_payload=sub_stage_payload,
            stage_view=view,
        )
        working_state = ReplayState.model_validate({
            "track_sequences": view.steps[-1].track_sequences,
            "loco_track_name": view.steps[-1].loco_track_name,
            "weighed_vehicle_nos": set(view.steps[-1].weighed_vehicle_nos),
            "spot_assignments": view.steps[-1].spot_assignments,
        })
    final_view = wave_views[-1]
    merged_hooks: list[DemoHook] = []
    merged_steps = []
    merged_verifier_errors: list[str] = []
    merged_failed_hook_nos: list[int] = []
    hook_offset = 0
    step_index = 0
    is_valid = True
    for wave_view in wave_views:
        renumbered_hook_nos: dict[int, int] = {}
        for hook in list(getattr(wave_view, "hook_plan", []) or []):
            new_hook_no = hook_offset + hook.hook_no
            renumbered_hook_nos[hook.hook_no] = new_hook_no
            if hasattr(hook, "model_copy"):
                merged_hooks.append(hook.model_copy(update={"hook_no": new_hook_no}))
            else:
                merged_hooks.append(DemoHook.model_validate({
                    "hook_no": new_hook_no,
                    "action_type": hook.action_type,
                    "source_track": hook.source_track,
                    "target_track": hook.target_track,
                    "vehicle_count": hook.vehicle_count,
                    "vehicle_nos": list(getattr(hook, "vehicle_nos", []) or []),
                    "path_tracks": list(getattr(hook, "path_tracks", []) or []),
                    "branch_codes": list(getattr(hook, "branch_codes", []) or []),
                    "route_length_m": getattr(hook, "route_length_m", None),
                    "reverse_branch_codes": list(getattr(hook, "reverse_branch_codes", []) or []),
                    "required_reverse_clearance_m": getattr(hook, "required_reverse_clearance_m", None),
                    "remark": getattr(hook, "remark", ""),
                }))
        hook_offset = len(merged_hooks)
        step_start = 0 if step_index == 0 else 1
        for raw_step in list(getattr(wave_view, "steps", []) or [])[step_start:]:
            hook = getattr(raw_step, "hook", None)
            remapped_hook = (
                None
                if hook is None
                else hook.model_copy(update={"hook_no": renumbered_hook_nos.get(hook.hook_no, hook.hook_no)})
                if hasattr(hook, "model_copy")
                else DemoHook.model_validate({
                    "hook_no": renumbered_hook_nos.get(getattr(hook, "hook_no", 0), getattr(hook, "hook_no", 0)),
                    "action_type": getattr(hook, "action_type", ""),
                    "source_track": getattr(hook, "source_track", ""),
                    "target_track": getattr(hook, "target_track", ""),
                    "vehicle_count": getattr(hook, "vehicle_count", 0),
                    "vehicle_nos": list(getattr(hook, "vehicle_nos", []) or []),
                    "path_tracks": list(getattr(hook, "path_tracks", []) or []),
                    "branch_codes": list(getattr(hook, "branch_codes", []) or []),
                    "route_length_m": getattr(hook, "route_length_m", None),
                    "reverse_branch_codes": list(getattr(hook, "reverse_branch_codes", []) or []),
                    "required_reverse_clearance_m": getattr(hook, "required_reverse_clearance_m", None),
                    "remark": getattr(hook, "remark", ""),
                })
            )
            if hasattr(raw_step, "model_copy"):
                merged_steps.append(
                    raw_step.model_copy(
                        update={
                            "step_index": step_index,
                            "hook": remapped_hook,
                        }
                    )
                )
            else:
                merged_steps.append(
                    {
                        "step_index": step_index,
                        "hook": remapped_hook,
                        "loco_track_name": getattr(raw_step, "loco_track_name", ""),
                        "loco_carry_vehicle_nos": list(getattr(raw_step, "loco_carry_vehicle_nos", []) or []),
                        "changed_tracks": list(getattr(raw_step, "changed_tracks", []) or []),
                        "track_sequences": dict(getattr(raw_step, "track_sequences", {}) or {}),
                        "weighed_vehicle_nos": list(getattr(raw_step, "weighed_vehicle_nos", []) or []),
                        "spot_assignments": dict(getattr(raw_step, "spot_assignments", {}) or {}),
                        "work_position_assignments": dict(getattr(raw_step, "work_position_assignments", {}) or {}),
                        "verifier_errors": list(getattr(raw_step, "verifier_errors", []) or []),
                        "track_map": getattr(raw_step, "track_map", {}),
                        "topology_graph": getattr(raw_step, "topology_graph", {}),
                        "transition_frames": list(getattr(raw_step, "transition_frames", []) or []),
                    }
                )
            step_index += 1
        merged_verifier_errors.extend(list(getattr(wave_view, "verifier_errors", []) or []))
        merged_failed_hook_nos.extend(
            renumbered_hook_nos.get(hook_no, hook_no)
            for hook_no in list(getattr(wave_view, "failed_hook_nos", []) or [])
        )
        is_valid = is_valid and bool(getattr(wave_view.summary, "is_valid", True))
    merged_summary = DemoSummary(
        hook_count=len(merged_hooks),
        vehicle_count=int(getattr(final_view.summary, "vehicle_count", len(working_vehicle_info))),
        is_valid=is_valid,
        error_count=len(merged_verifier_errors),
        final_tracks=list(getattr(final_view.summary, "final_tracks", [])),
        weighed_vehicle_count=int(getattr(final_view.summary, "weighed_vehicle_count", 0)),
        assigned_spot_count=int(getattr(final_view.summary, "assigned_spot_count", 0)),
        assigned_work_position_count=int(getattr(final_view.summary, "assigned_work_position_count", 0)),
    )
    return (
        DemoViewModel(
            summary=merged_summary,
            verifier_errors=merged_verifier_errors,
            hook_plan=merged_hooks,
            steps=merged_steps,
            final_spot_assignments=dict(getattr(final_view, "final_spot_assignments", {}) or {}),
            final_work_position_assignments=dict(getattr(final_view, "final_work_position_assignments", {}) or {}),
            failed_hook_nos=sorted(dict.fromkeys(merged_failed_hook_nos)),
            track_map=getattr(final_view, "track_map", {}),
            topology_graph=getattr(final_view, "topology_graph", {}),
            comparison_summary=getattr(final_view, "comparison_summary", None),
            vehicle_target_tracks=dict(getattr(final_view, "vehicle_target_tracks", {}) or {}),
        ),
        working_vehicle_info,
        working_state,
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
