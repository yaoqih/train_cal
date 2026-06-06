from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData, clone_master_with_blocked_branches
from fzed_shunting.domain.depot_spots import exact_spot_reservations, realign_spots_for_track_order
from fzed_shunting.io.normalize_input import SOURCE_TRACK_ALIASES, normalize_plan_input
from fzed_shunting.verify.replay import ReplayState, build_initial_state
from fzed_shunting.workflow.l7_closed_topology_mode import (
    PHASE3_DYNAMIC_CURRENT_HOLD,
    PHASE4_DYNAMIC_CURRENT_HOLD,
    PHASE4_RESIDUAL_CLEANUP,
    PHASE2_L1_TRANSFER_MAX_LENGTH_M,
    build_l7_closed_topology_workflow_payload,
    is_l7_closed_topology_mode,
    rebuild_phase2_execution_policy_for_runtime,
)

PHASE2_RELEASE_BUFFER_TRACKS = (
    "存4北",
    "存5北",
    "存5南",
    "存1",
    "存2",
    "存3",
    "调北",
    "机北3",
    "预修",
    "机棚",
)
@dataclass(frozen=True)
class Phase2ReleaseTask:
    blocked_source_track: str
    blocker_track: str
    blocker_vehicle_nos: tuple[str, ...]
    release_target_track: str
    reason: str

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
        completed_stages: list[WorkflowStageResult] | None = None,
        cause_message: str,
        stage_input_summary: dict[str, Any] | None = None,
    ) -> None:
        self.failed_stage_name = failed_stage_name
        self.failed_stage_index = failed_stage_index
        self.total_stage_count = total_stage_count
        self.completed_stage_names = list(completed_stage_names)
        self.completed_stages = list(completed_stages or [])
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
                master=master,
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
            wave_plan_key = _stage_wave_plan_key(stage)
            if wave_plan_key is not None:
                view, current_vehicle_info, current_state = _solve_wave_stage(
                    stage_master=stage_master,
                    track_info=track_info,
                    vehicle_meta=vehicle_meta,
                    stage=stage,
                    wave_plan_key=wave_plan_key,
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
            elif str((stage.get("stagePolicy") or {}).get("stageMode") or "") == "PHASE2_DEPOT_AREA_MARSHALLING":
                view = _solve_phase2_execution_stage(
                    stage_master=stage_master,
                    stage_payload=stage_payload,
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
            if wave_plan_key is None:
                current_vehicle_info = _next_vehicle_info(
                    stage_payload=stage_payload,
                    stage_view=view,
                )
                current_state = ReplayState.model_validate({
                    "track_sequences": view.steps[-1].track_sequences,
                    "loco_track_name": view.steps[-1].loco_track_name,
                    "loco_node": getattr(view.steps[-1], "loco_node", None),
                    "weighed_vehicle_nos": set(view.steps[-1].weighed_vehicle_nos),
                    "spot_assignments": view.steps[-1].spot_assignments,
                    "loco_carry": tuple(getattr(view.steps[-1], "loco_carry_vehicle_nos", []) or []),
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
                completed_stages=list(stages),
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


def _stage_wave_plan_key(stage: dict[str, Any]) -> str | None:
    stage_policy = dict(stage.get("stagePolicy") or {})
    if str(stage_policy.get("stageMode") or "") == "PHASE1_PRE_REPAIR_BUFFERING" and bool(stage_policy.get("phase1WavePlans")):
        return "phase1WavePlans"
    if str(stage_policy.get("stageMode") or "") == "PHASE3_JI_TO_DEPOT_ALLOCATION" and bool(stage_policy.get("phase3WavePlans")):
        return "phase3WavePlans"
    return None


def _solve_wave_stage(
    *,
    stage_master: MasterData,
    track_info: list[dict],
    vehicle_meta: dict[str, dict],
    stage: dict[str, Any],
    wave_plan_key: str,
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
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    wave_plans = list((stage.get("stagePolicy") or {}).get(wave_plan_key) or [])
    if not wave_plans:
        raise ValueError(f"{wave_plan_key} must be non-empty for wave stage solve")
    default_weights = [0.52, 0.30, 0.18] if wave_plan_key == "phase1WavePlans" else None
    working_vehicle_info = [dict(item) for item in current_vehicle_info]
    working_state = current_state
    wave_views: list[DemoViewModel] = []
    for index, wave_plan in enumerate(wave_plans):
        sub_stage_policy = dict(stage.get("stagePolicy") or {})
        for key, value in wave_plan.items():
            if key == "vehicleGoals":
                continue
            if isinstance(value, dict):
                sub_stage_policy[key] = dict(value)
            elif isinstance(value, list):
                sub_stage_policy[key] = list(value)
            else:
                sub_stage_policy[key] = value
        sub_vehicle_goals = list(wave_plan.get("vehicleGoals") or [])
        if wave_plan_key == "phase2WavePlans":
            collected_vehicle_nos = {
                str(item) for item in sub_stage_policy.get("phase2CollectedVehicleNos") or ()
            }
            exchange_track = str(sub_stage_policy.get("exchangeTrack") or "存4北")
            sub_vehicle_goals = []
            for item in working_vehicle_info:
                vehicle_no = str(item["vehicleNo"])
                current_track = str(item["trackName"])
                if vehicle_no in collected_vehicle_nos:
                    sub_vehicle_goals.append(
                        {
                            "vehicleNo": vehicle_no,
                            "targetTrack": exchange_track,
                            "targetMode": "TRACK",
                            "targetSource": "PHASE2_DYNAMIC_TRANSFER",
                            "isSpotting": "",
                        }
                    )
                    continue
                sub_vehicle_goals.append(
                    {
                        "vehicleNo": vehicle_no,
                        "targetTrack": current_track,
                        "targetMode": "TRACK",
                        "targetSource": "PHASE2_DYNAMIC_HOLD",
                        "isSpotting": "",
                    }
                )
        elif wave_plan_key == "phase3WavePlans":
            active_goals = {
                str(vehicle_no): dict(goal)
                for vehicle_no, goal in dict(wave_plan.get("activeGoalsByVehicle") or {}).items()
            }
            sub_vehicle_goals = []
            for item in working_vehicle_info:
                vehicle_no = str(item["vehicleNo"])
                current_track = str(item["trackName"])
                if vehicle_no in active_goals:
                    sub_vehicle_goals.append(active_goals[vehicle_no])
                    continue
                sub_vehicle_goals.append(
                    {
                        "vehicleNo": vehicle_no,
                        "targetTrack": current_track,
                        "targetMode": "TRACK",
                        "targetSource": "PHASE3_DYNAMIC_HOLD",
                        "isSpotting": "",
                    }
                )
        sub_stage = {
            "name": str(stage.get("name", "phase1")),
            "description": str(stage.get("description", "")),
            "routePolicy": dict(stage.get("routePolicy") or {}),
            "stagePolicy": sub_stage_policy,
            "vehicleGoals": sub_vehicle_goals,
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
            if wave_plan.get("waveWeight") is not None:
                weight = float(wave_plan.get("waveWeight"))
            elif default_weights is not None:
                weight = default_weights[index] if index < len(default_weights) else default_weights[-1]
            else:
                weight = 1.0 / max(len(wave_plans), 1)
            sub_time_budget_ms = max(5_000.0, float(time_budget_ms) * weight)
        plan_payload = None
        if wave_plan_key == "phase2WavePlans":
            normalized = normalize_plan_input(
                sub_stage_payload,
                stage_master,
                allow_internal_loco_tracks=True,
            )
            route_oracle = RouteOracle(stage_master)
            stage_policy = sub_stage_payload.get("stagePolicy") or {}
            exchange_track = str(stage_policy.get("exchangeTrack") or "存4北")
            vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
            attach_units = [
                (
                    str(item.get("sourceTrack") or ""),
                    tuple(str(vehicle_no) for vehicle_no in item.get("vehicleNos") or ()),
                )
                for item in stage_policy.get("phase2AttachUnits") or ()
            ]
            if not attach_units:
                raise ValueError("phase2 wave is missing attach units")
            plan_payload = []
            attached_state = working_state
            hook_no = 1
            pending_attach_units = list(attach_units)
            while pending_attach_units:
                selected_index = None
                selected_path = None
                for index, (source_track, vehicle_nos) in enumerate(pending_attach_units):
                    if not source_track or not vehicle_nos:
                        raise ValueError("phase2 attach unit is incomplete")
                    source_seq = list(attached_state.track_sequences.get(source_track, []))
                    if source_seq[: len(vehicle_nos)] != list(vehicle_nos):
                        continue
                    attach_path = route_oracle.resolve_clear_path_tracks(
                        attached_state.loco_track_name,
                        source_track,
                        occupied_track_sequences=attached_state.track_sequences,
                        source_node=attached_state.loco_node,
                        target_node=route_oracle.order_end_node(source_track),
                    )
                    if attach_path is None:
                        continue
                    selected_index = index
                    selected_path = attach_path
                    break
                if selected_index is None or selected_path is None:
                    blocked_units = [
                        {
                            "sourceTrack": source_track,
                            "vehicleNos": list(vehicle_nos),
                        }
                        for source_track, vehicle_nos in pending_attach_units
                    ]
                    raise ValueError(f"no clear attach path for pending units: {blocked_units}")
                source_track, vehicle_nos = pending_attach_units.pop(selected_index)
                if not source_track or not vehicle_nos:
                    raise ValueError("phase2 attach unit is incomplete")
                attach_move = HookAction(
                    source_track=source_track,
                    target_track=source_track,
                    vehicle_nos=list(vehicle_nos),
                    path_tracks=list(selected_path),
                    action_type="ATTACH",
                )
                plan_payload.append(
                    {
                        "hookNo": hook_no,
                        "actionType": "ATTACH",
                        "sourceTrack": source_track,
                        "targetTrack": source_track,
                        "vehicleNos": list(vehicle_nos),
                        "pathTracks": list(selected_path),
                    }
                )
                hook_no += 1
                attached_state = _apply_move(
                    state=attached_state,
                    move=attach_move,
                    plan_input=normalized,
                    vehicle_by_no=vehicle_by_no,
                )
            detach_path = route_oracle.resolve_clear_path_tracks(
                attached_state.loco_track_name,
                exchange_track,
                occupied_track_sequences=attached_state.track_sequences,
                source_node=attached_state.loco_node,
                target_node=None,
            )
            if detach_path is None:
                raise ValueError(f"no clear detach path {attached_state.loco_track_name} -> {exchange_track}")
            plan_payload.append(
                {
                    "hookNo": hook_no,
                    "actionType": "DETACH",
                    "sourceTrack": attached_state.loco_track_name,
                    "targetTrack": exchange_track,
                    "vehicleNos": list(attached_state.loco_carry),
                    "pathTracks": list(detach_path),
                },
            )
        view = build_demo_view_model(
            stage_master,
            sub_stage_payload,
            plan_payload=plan_payload,
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
            "loco_node": getattr(view.steps[-1], "loco_node", None),
            "weighed_vehicle_nos": set(view.steps[-1].weighed_vehicle_nos),
            "spot_assignments": view.steps[-1].spot_assignments,
            "loco_carry": tuple(getattr(view.steps[-1], "loco_carry_vehicle_nos", []) or []),
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
                        "loco_node": getattr(raw_step, "loco_node", None),
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
        if wave_plan_key != "phase2WavePlans":
            merged_verifier_errors.extend(list(getattr(wave_view, "verifier_errors", []) or []))
            merged_failed_hook_nos.extend(
                renumbered_hook_nos.get(hook_no, hook_no)
                for hook_no in list(getattr(wave_view, "failed_hook_nos", []) or [])
            )
            is_valid = is_valid and bool(getattr(wave_view.summary, "is_valid", True))
    if wave_plan_key == "phase2WavePlans":
        merged_verifier_errors = list(getattr(final_view, "verifier_errors", []) or [])
        merged_failed_hook_nos = list(getattr(final_view, "failed_hook_nos", []) or [])
        is_valid = bool(getattr(final_view.summary, "is_valid", True))
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
            diagnostics=dict(getattr(final_view, "diagnostics", {}) or {}),
            vehicle_target_tracks=dict(getattr(final_view, "vehicle_target_tracks", {}) or {}),
        ),
        working_vehicle_info,
        working_state,
    )


def _solve_phase2_execution_stage(
    *,
    stage_master: MasterData,
    stage_payload: dict[str, Any],
    current_state: ReplayState,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None,
    use_validation_recovery: bool,
    diagnose_front_search_only: bool,
):
    from fzed_shunting.demo.view_model import build_demo_view_model
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    stage_policy = dict(stage_payload.get("stagePolicy") or {})
    runtime_execution_plan = rebuild_phase2_execution_policy_for_runtime(
        stage_payload=stage_payload,
        track_sequences=current_state.track_sequences,
    )
    if runtime_execution_plan is not None:
        stage_policy["phase2ExecutionPlan"] = runtime_execution_plan
        transfer_vehicle_nos = {
            str(item)
            for item in runtime_execution_plan.get("transferVehicleNos") or ()
        }
        deferred_tail_vehicle_nos = {
            str(item)
            for item in runtime_execution_plan.get("deferredTailVehicleNos") or ()
        }
        original_vehicle_info = list(stage_payload.get("vehicleInfo") or [])
        stage_payload = {
            **stage_payload,
            "vehicleInfo": [],
            "stagePolicy": stage_policy,
        }
        for item in original_vehicle_info:
            row = dict(item)
            vehicle_no = str(row["vehicleNo"])
            if vehicle_no in transfer_vehicle_nos:
                row["targetTrack"] = str(stage_policy.get("exchangeTrack") or "存4北")
                row["targetMode"] = "AREA"
                row["targetAreaCode"] = "STAGE::PHASE2_TRANSFER_TO_CUN4"
                row["targetSource"] = "PHASE2_TRANSFER_TO_CUN4"
                row["isSpotting"] = ""
            elif vehicle_no in deferred_tail_vehicle_nos:
                row["targetTrack"] = str(row["trackName"])
                row["targetMode"] = "AREA"
                row["targetAreaCode"] = "STAGE::PHASE2_DEFERRED_TAIL"
                row["targetSource"] = "PHASE2_DEFERRED_TAIL"
                row["isSpotting"] = ""
            stage_payload["vehicleInfo"].append(row)
    execution_plan = dict((stage_payload.get("stagePolicy") or {}).get("phase2ExecutionPlan") or {})
    track_layers = list(execution_plan.get("trackLayers") or [])
    transfer_vehicle_nos = {str(item) for item in execution_plan.get("transferVehicleNos") or ()}
    deferred_tail_vehicle_nos = {str(item) for item in execution_plan.get("deferredTailVehicleNos") or ()}
    collection_batches = [
        tuple(str(vehicle_no) for vehicle_no in batch)
        for batch in execution_plan.get("collectionBatches") or ()
    ]
    predecessor_unlock_vehicle_nos = {str(item) for item in execution_plan.get("predecessorUnlockVehicleNos") or ()}
    must_pull_vehicle_nos = {str(item) for item in execution_plan.get("mustPullVehicleNos") or ()}
    phase3_clearance_vehicle_nos = {str(item) for item in execution_plan.get("phase3ClearanceVehicleNos") or ()}
    hard_required_vehicle_nos = predecessor_unlock_vehicle_nos | must_pull_vehicle_nos | phase3_clearance_vehicle_nos
    if not track_layers:
        view = build_demo_view_model(
            stage_master,
            stage_payload,
            plan_payload=[],
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            time_budget_ms=time_budget_ms,
            initial_state_override=current_state,
            use_validation_recovery=use_validation_recovery,
            diagnose_front_search_only=diagnose_front_search_only,
        )
        view.diagnostics = {
            **dict(getattr(view, "diagnostics", {}) or {}),
            "phase2Runtime": {
                "plannedTransferVehicleNos": sorted(transfer_vehicle_nos),
                "plannedTransferVehicleCount": len(transfer_vehicle_nos),
                "plannedCollectionBatchCount": len(collection_batches),
                "plannedTrackLayerCount": len(track_layers),
                "executedHookCount": 0,
                "executedAttachCount": 0,
                "executedDetachCount": 0,
                "executedVehicleNos": [],
                "runtimeDeferredLayers": [],
                "runtimeDeferredVehicleNos": sorted(deferred_tail_vehicle_nos),
                "runtimeDeferredVehicleCount": len(deferred_tail_vehicle_nos),
                "runtimeDeferredReasons": (
                    {"planned_deferred_tail": len(deferred_tail_vehicle_nos)}
                    if deferred_tail_vehicle_nos
                    else {}
                ),
                "runtimeDeferredBlockingTracks": {},
                "releasedVehicleTargets": {},
                "releasedVehicleCount": 0,
                "effectiveTargetSources": dict(
                    sorted(
                        Counter(
                            str(item.get("targetSource") or "")
                            for item in stage_payload.get("vehicleInfo", [])
                        ).items()
                    )
                ),
            },
        }
        return view

    normalized = normalize_plan_input(
        stage_payload,
        stage_master,
        allow_internal_loco_tracks=True,
    )
    route_oracle = RouteOracle(stage_master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    capacity_by_track = {info.track_name: float(info.track_distance) for info in normalized.track_info}
    exchange_track = str((stage_payload.get("stagePolicy") or {}).get("exchangeTrack") or "存4北")
    phase2_hold_vehicle_nos = {
        str(item["vehicleNo"])
        for item in stage_payload["vehicleInfo"]
        if str(item.get("targetSource") or "") not in {"PHASE2_TRANSFER_TO_CUN4"}
    }
    layer_by_track: dict[str, list[dict[str, Any]]] = {}
    for item in track_layers:
        layer_vehicle_nos = tuple(str(vehicle_no) for vehicle_no in item.get("vehicleNos") or ())
        if transfer_vehicle_nos and not any(vehicle_no in transfer_vehicle_nos for vehicle_no in layer_vehicle_nos):
            continue
        source_track = str(item.get("sourceTrack") or "")
        layer_by_track.setdefault(source_track, []).append(
            {
                "sourceTrack": source_track,
                "layerIndex": int(item.get("layerIndex") or 0),
                "vehicleNos": layer_vehicle_nos,
                "exposedPrefixVehicleNos": tuple(str(vehicle_no) for vehicle_no in item.get("exposedPrefixVehicleNos") or ()),
            }
        )
    pending_by_track = {
        track: sorted(layers, key=lambda item: item["layerIndex"])
        for track, layers in layer_by_track.items()
    }
    working_state = current_state
    plan_payload: list[dict[str, Any]] = []
    hook_no = 1
    runtime_deferred_optional_layers: list[dict[str, Any]] = []
    released_vehicle_targets: dict[str, str] = {}
    phase2_deferred_source_tracks: set[str] = set()
    batch_queue = list(collection_batches) if collection_batches else [
        tuple(vehicle_no for layer in track_layers for vehicle_no in layer.get("vehicleNos") or ())
    ]
    for batch_vehicle_nos in batch_queue:
        batch_vehicle_set = set(batch_vehicle_nos)
        batch_pending_by_track = {
            track: [
                layer
                for layer in layers
                if any(vehicle_no in batch_vehicle_set for vehicle_no in layer["vehicleNos"])
            ]
            for track, layers in pending_by_track.items()
        }
        while True:
            if not any(batch_pending_by_track.values()):
                break
            reachable_candidates: list[tuple[tuple[int, int, int, str], str, tuple[str, ...], list[str]]] = []
            blocked_candidates: list[dict[str, Any]] = []
            for source_track, layers in batch_pending_by_track.items():
                if not layers:
                    continue
                layer = layers[0]
                vehicle_nos = tuple(layer["vehicleNos"])
                layer_must_pull = any(vehicle_no in hard_required_vehicle_nos for vehicle_no in vehicle_nos)
                source_seq = list(working_state.track_sequences.get(source_track, []))
                if source_seq[: len(vehicle_nos)] != list(vehicle_nos):
                    blocked_candidates.append(
                        {
                            "sourceTrack": source_track,
                            "layerIndex": layer["layerIndex"],
                            "vehicleNos": list(vehicle_nos),
                            "reason": "prefix_not_exposed",
                            "mustPull": layer_must_pull,
                        }
                    )
                    continue
                carried_train_length_m = sum(
                    float(vehicle_by_no[vehicle_no].vehicle_length)
                    for vehicle_no in working_state.loco_carry
                )
                access_result = route_oracle.validate_loco_access(
                    loco_track=working_state.loco_track_name,
                    target_track=source_track,
                    occupied_track_sequences=working_state.track_sequences,
                    loco_node=working_state.loco_node,
                    carried_train_length_m=carried_train_length_m,
                )
                if not access_result.is_valid:
                    blocked_candidates.append(
                        {
                            "sourceTrack": source_track,
                            "layerIndex": layer["layerIndex"],
                            "vehicleNos": list(vehicle_nos),
                            "reason": "attach_path_blocked",
                            "mustPull": layer_must_pull,
                            "blockingTracks": list(access_result.blocking_tracks),
                            "errors": list(access_result.errors),
                        }
                    )
                    continue
                attach_path = route_oracle.resolve_clear_path_tracks(
                    working_state.loco_track_name,
                    source_track,
                    occupied_track_sequences=working_state.track_sequences,
                    source_node=working_state.loco_node,
                    target_node=route_oracle.order_end_node(source_track),
                )
                if attach_path is None:
                    continue
                score = (
                    0 if layer_must_pull else 1,
                    _phase2_attach_access_priority(source_track),
                    layer["layerIndex"],
                    -len(vehicle_nos),
                    source_track,
                )
                reachable_candidates.append((score, source_track, vehicle_nos, list(attach_path)))
            if not reachable_candidates:
                remaining = []
                deferred_optional_tracks: list[str] = []
                for source_track, layers in batch_pending_by_track.items():
                    if not layers:
                        continue
                    layer = layers[0]
                    vehicle_nos = tuple(layer["vehicleNos"])
                    layer_must_pull = any(vehicle_no in hard_required_vehicle_nos for vehicle_no in vehicle_nos)
                    if layer_must_pull:
                        remaining.append(layer)
                        continue
                    batch_pending_by_track[source_track].pop(0)
                    runtime_deferred_optional_layers.append(
                        {
                            "sourceTrack": source_track,
                            "layerIndex": layer["layerIndex"],
                            "vehicleNos": list(vehicle_nos),
                            "reason": "runtime_unreachable_optional",
                        }
                    )
                    deferred_optional_tracks.append(source_track)
                if deferred_optional_tracks:
                    continue
                if remaining:
                    pending_vehicle_nos = {
                        vehicle_no
                        for layers in batch_pending_by_track.values()
                        for layer in layers
                        for vehicle_no in layer["vehicleNos"]
                    }
                    release_task = None
                    for blocked_candidate in blocked_candidates:
                        release_task = _phase2_build_release_task(
                            blocked_candidate=blocked_candidate,
                            working_state=working_state,
                            route_oracle=route_oracle,
                            vehicle_by_no=vehicle_by_no,
                            capacity_by_track=capacity_by_track,
                            exchange_track=exchange_track,
                            phase2_transfer_vehicle_nos=transfer_vehicle_nos,
                            phase2_hold_vehicle_nos=phase2_hold_vehicle_nos,
                            pending_vehicle_nos=pending_vehicle_nos,
                        )
                        if release_task is not None:
                            break
                    if release_task is not None:
                        hook_no, working_state = _phase2_apply_release_task(
                            release_task=release_task,
                            hook_no=hook_no,
                            plan_payload=plan_payload,
                            working_state=working_state,
                            normalized=normalized,
                            route_oracle=route_oracle,
                            vehicle_by_no=vehicle_by_no,
                        )
                        for vehicle_no in release_task.blocker_vehicle_nos:
                            released_vehicle_targets[vehicle_no] = release_task.release_target_track
                        continue
                    blocked_by_layer = {
                        (str(item.get("sourceTrack") or ""), int(item.get("layerIndex") or 0)): item
                        for item in blocked_candidates
                    }
                    path_blocked_remaining = []
                    for layer in remaining:
                        blocked = blocked_by_layer.get((layer["sourceTrack"], layer["layerIndex"]))
                        if (
                            layer["sourceTrack"] in phase2_deferred_source_tracks
                            or (
                                blocked is not None
                                and blocked.get("reason") == "attach_path_blocked"
                                and blocked.get("blockingTracks")
                            )
                        ):
                            path_blocked_remaining.append(layer)
                            continue
                        break
                    if len(path_blocked_remaining) == len(remaining):
                        for layer in path_blocked_remaining:
                            batch_pending_by_track[layer["sourceTrack"]].pop(0)
                            blocked = blocked_by_layer.get((layer["sourceTrack"], layer["layerIndex"]), {})
                            blocking_tracks = list(blocked.get("blockingTracks") or [])
                            runtime_deferred_optional_layers.append(
                                {
                                    "sourceTrack": layer["sourceTrack"],
                                    "layerIndex": layer["layerIndex"],
                                    "vehicleNos": list(layer["vehicleNos"]),
                                    "reason": (
                                        "path_blocked_deferred"
                                        if blocking_tracks
                                        else "prefix_after_path_blocked_deferred"
                                    ),
                                    "blockingTracks": blocking_tracks,
                                }
                            )
                            phase2_deferred_source_tracks.add(layer["sourceTrack"])
                        continue
                    raise ValueError(
                        f"phase2 depot collect stalled: {blocked_candidates or remaining}"
                    )
                break
            _, source_track, vehicle_nos, attach_path = sorted(reachable_candidates, key=lambda item: item[0])[0]
            plan_payload.append(
                {
                    "hookNo": hook_no,
                    "actionType": "ATTACH",
                    "sourceTrack": source_track,
                    "targetTrack": source_track,
                    "vehicleNos": list(vehicle_nos),
                    "pathTracks": [source_track],
                }
            )
            hook_no += 1
            attach_move = HookAction(
                source_track=source_track,
                target_track=source_track,
                vehicle_nos=list(vehicle_nos),
                path_tracks=attach_path,
                action_type="ATTACH",
            )
            working_state = _apply_move(
                state=working_state,
                move=attach_move,
                plan_input=normalized,
                vehicle_by_no=vehicle_by_no,
            )
            batch_pending_by_track[source_track].pop(0)
        if working_state.loco_carry:
            source_track = working_state.loco_track_name
            detach_path = route_oracle.resolve_clear_path_tracks(
                source_track,
                exchange_track,
                occupied_track_sequences=working_state.track_sequences,
                source_node=working_state.loco_node,
                target_node=route_oracle.order_end_node(exchange_track),
            )
            if detach_path is None:
                raise ValueError(f"no clear detach path {source_track} -> {exchange_track}")
            batch_to_detach = tuple(working_state.loco_carry)
            plan_payload.append(
                {
                    "hookNo": hook_no,
                    "actionType": "DETACH",
                    "sourceTrack": source_track,
                    "targetTrack": exchange_track,
                    "vehicleNos": list(batch_to_detach),
                    "pathTracks": list(detach_path),
                }
            )
            hook_no += 1
            detach_move = HookAction(
                source_track=source_track,
                target_track=exchange_track,
                vehicle_nos=list(batch_to_detach),
                path_tracks=list(detach_path),
                action_type="DETACH",
            )
            working_state = _apply_move(
                state=working_state,
                move=detach_move,
                plan_input=normalized,
                vehicle_by_no=vehicle_by_no,
            )

    deferred_runtime_vehicle_nos = {
        vehicle_no
        for row in runtime_deferred_optional_layers
        for vehicle_no in row["vehicleNos"]
    }
    effective_stage_payload = stage_payload
    if deferred_runtime_vehicle_nos or released_vehicle_targets:
        effective_stage_payload = {
            **stage_payload,
            "vehicleInfo": [],
        }
        for item in stage_payload["vehicleInfo"]:
            row = dict(item)
            vehicle_no = str(row["vehicleNo"])
            if vehicle_no in released_vehicle_targets:
                row["targetTrack"] = released_vehicle_targets[vehicle_no]
                row["targetMode"] = "TRACK"
                row["targetSource"] = "PHASE2_RELEASE_BLOCKER"
                row["isSpotting"] = ""
            elif vehicle_no in deferred_runtime_vehicle_nos:
                row["targetTrack"] = str(row["trackName"])
                row["targetMode"] = "AREA"
                row["targetAreaCode"] = "STAGE::PHASE2_RUNTIME_DEFERRED"
                row["targetSource"] = "PHASE2_RUNTIME_DEFERRED"
                row["isSpotting"] = ""
            effective_stage_payload["vehicleInfo"].append(row)

    view = build_demo_view_model(
        stage_master,
        effective_stage_payload,
        plan_payload=plan_payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
        initial_state_override=current_state,
        use_validation_recovery=use_validation_recovery,
        diagnose_front_search_only=diagnose_front_search_only,
    )
    view.diagnostics = {
        **dict(getattr(view, "diagnostics", {}) or {}),
        "phase2Runtime": {
            "plannedTransferVehicleNos": sorted(transfer_vehicle_nos),
            "plannedTransferVehicleCount": len(transfer_vehicle_nos),
            "plannedCollectionBatchCount": len(collection_batches),
            "plannedTrackLayerCount": len(track_layers),
            "executedHookCount": len(plan_payload),
            "executedAttachCount": sum(1 for hook in plan_payload if hook.get("actionType") == "ATTACH"),
            "executedDetachCount": sum(1 for hook in plan_payload if hook.get("actionType") == "DETACH"),
            "executedVehicleNos": sorted({
                vehicle_no
                for hook in plan_payload
                for vehicle_no in hook.get("vehicleNos", [])
                if hook.get("actionType") == "ATTACH"
            }),
            "runtimeDeferredLayers": list(runtime_deferred_optional_layers),
            "runtimeDeferredVehicleNos": sorted(deferred_runtime_vehicle_nos),
            "runtimeDeferredVehicleCount": len(deferred_runtime_vehicle_nos),
            "runtimeDeferredReasons": dict(Counter(str(row.get("reason") or "") for row in runtime_deferred_optional_layers)),
            "runtimeDeferredBlockingTracks": dict(
                sorted(
                    Counter(
                        str(track)
                        for row in runtime_deferred_optional_layers
                        for track in row.get("blockingTracks", [])
                    ).items()
                )
            ),
            "releasedVehicleTargets": dict(sorted(released_vehicle_targets.items())),
            "releasedVehicleCount": len(released_vehicle_targets),
            "effectiveTargetSources": dict(
                sorted(
                    Counter(
                        str(item.get("targetSource") or "")
                        for item in effective_stage_payload.get("vehicleInfo", [])
                    ).items()
                )
            ),
        },
    }
    return view


def _phase2_attach_access_priority(source_track: str) -> int:
    return {
        "轮": 0,
        "修4库外": 1,
        "修3库外": 2,
        "修2库外": 3,
        "修1库外": 4,
        "修4": 5,
        "修3": 6,
        "修2": 7,
        "修1": 8,
    }.get(source_track, 99)


def _phase2_build_release_task(
    *,
    blocked_candidate: dict[str, Any],
    working_state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
    exchange_track: str,
    phase2_transfer_vehicle_nos: set[str],
    phase2_hold_vehicle_nos: set[str],
    pending_vehicle_nos: set[str],
) -> Phase2ReleaseTask | None:
    if blocked_candidate.get("reason") != "attach_path_blocked":
        return None
    blocked_source_track = str(blocked_candidate.get("sourceTrack") or "")
    blocking_tracks = tuple(str(item) for item in blocked_candidate.get("blockingTracks") or ())
    for blocker_track in blocking_tracks:
        blocker_seq = tuple(str(vehicle_no) for vehicle_no in working_state.track_sequences.get(blocker_track, ()))
        if not blocker_seq:
            continue
        blocker_prefix = _phase2_release_prefix(
            blocker_seq=blocker_seq,
            phase2_transfer_vehicle_nos=phase2_transfer_vehicle_nos,
            phase2_hold_vehicle_nos=phase2_hold_vehicle_nos,
            pending_vehicle_nos=pending_vehicle_nos,
            vehicle_by_no=vehicle_by_no,
        )
        if not blocker_prefix:
            continue
        target_track = _phase2_release_target_track(
            blocker_vehicle_nos=blocker_prefix,
            blocker_track=blocker_track,
            working_state=working_state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
            exchange_track=exchange_track,
            phase2_transfer_vehicle_nos=phase2_transfer_vehicle_nos,
        )
        if target_track is None:
            continue
        return Phase2ReleaseTask(
            blocked_source_track=blocked_source_track,
            blocker_track=blocker_track,
            blocker_vehicle_nos=blocker_prefix,
            release_target_track=target_track,
            reason="attach_path_blocked",
        )
    return None


def _phase2_release_prefix(
    *,
    blocker_seq: tuple[str, ...],
    phase2_transfer_vehicle_nos: set[str],
    phase2_hold_vehicle_nos: set[str],
    pending_vehicle_nos: set[str],
    vehicle_by_no: dict[str, Any],
) -> tuple[str, ...]:
    prefix: list[str] = []
    for vehicle_no in blocker_seq:
        if vehicle_no in pending_vehicle_nos:
            break
        candidate = tuple([*prefix, vehicle_no])
        if prefix and not _phase2_release_group_is_valid(candidate, vehicle_by_no=vehicle_by_no):
            break
        if not _phase2_release_group_is_valid(candidate, vehicle_by_no=vehicle_by_no):
            return ()
        prefix.append(vehicle_no)
        if vehicle_no in phase2_transfer_vehicle_nos:
            break
    return tuple(prefix)


def _phase2_release_group_is_valid(
    vehicle_nos: tuple[str, ...],
    *,
    vehicle_by_no: dict[str, Any],
) -> bool:
    vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in vehicle_nos]
    if validate_hook_vehicle_group(vehicles):
        return False
    total_length_m = sum(float(vehicle.vehicle_length) for vehicle in vehicles)
    return total_length_m <= PHASE2_L1_TRANSFER_MAX_LENGTH_M + 1e-9


def _phase2_release_target_track(
    *,
    blocker_vehicle_nos: tuple[str, ...],
    blocker_track: str,
    working_state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
    exchange_track: str,
    phase2_transfer_vehicle_nos: set[str],
) -> str | None:
    preferred_tracks = (
        (exchange_track,)
        if all(vehicle_no in phase2_transfer_vehicle_nos for vehicle_no in blocker_vehicle_nos)
        else tuple(track for track in PHASE2_RELEASE_BUFFER_TRACKS if track != exchange_track)
    )
    for target_track in preferred_tracks:
        if target_track == blocker_track:
            continue
        if not _phase2_track_has_capacity(
            target_track=target_track,
            adding_vehicle_nos=blocker_vehicle_nos,
            working_state=working_state,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
        ):
            continue
        detach_path = route_oracle.resolve_clear_path_tracks(
            blocker_track,
            target_track,
            occupied_track_sequences=working_state.track_sequences,
            source_node=route_oracle.order_end_node(blocker_track),
            target_node=route_oracle.order_end_node(target_track),
        )
        if detach_path is not None:
            return target_track
    return None


def _phase2_apply_release_task(
    *,
    release_task: Phase2ReleaseTask,
    hook_no: int,
    plan_payload: list[dict[str, Any]],
    working_state: ReplayState,
    normalized: Any,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
) -> tuple[int, ReplayState]:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    attach_path = route_oracle.resolve_clear_path_tracks(
        working_state.loco_track_name,
        release_task.blocker_track,
        occupied_track_sequences=working_state.track_sequences,
        source_node=working_state.loco_node,
        target_node=route_oracle.order_end_node(release_task.blocker_track),
    )
    if attach_path is None:
        raise ValueError(f"phase2 release attach path blocked: {release_task}")
    plan_payload.append(
        {
            "hookNo": hook_no,
            "actionType": "ATTACH",
            "sourceTrack": release_task.blocker_track,
            "targetTrack": release_task.blocker_track,
            "vehicleNos": list(release_task.blocker_vehicle_nos),
            "pathTracks": [release_task.blocker_track],
            "reason": "PHASE2_RELEASE_BLOCKER",
            "blockedSourceTrack": release_task.blocked_source_track,
        }
    )
    attach_move = HookAction(
        source_track=release_task.blocker_track,
        target_track=release_task.blocker_track,
        vehicle_nos=list(release_task.blocker_vehicle_nos),
        path_tracks=attach_path,
        action_type="ATTACH",
    )
    next_state = _apply_move(
        state=working_state,
        move=attach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    hook_no += 1

    detach_path = route_oracle.resolve_clear_path_tracks(
        next_state.loco_track_name,
        release_task.release_target_track,
        occupied_track_sequences=next_state.track_sequences,
        source_node=next_state.loco_node,
        target_node=route_oracle.order_end_node(release_task.release_target_track),
    )
    if detach_path is None:
        raise ValueError(f"phase2 release detach path blocked: {release_task}")
    plan_payload.append(
        {
            "hookNo": hook_no,
            "actionType": "DETACH",
            "sourceTrack": next_state.loco_track_name,
            "targetTrack": release_task.release_target_track,
            "vehicleNos": list(release_task.blocker_vehicle_nos),
            "pathTracks": list(detach_path),
            "reason": "PHASE2_RELEASE_BLOCKER",
            "blockedSourceTrack": release_task.blocked_source_track,
        }
    )
    detach_move = HookAction(
        source_track=next_state.loco_track_name,
        target_track=release_task.release_target_track,
        vehicle_nos=list(release_task.blocker_vehicle_nos),
        path_tracks=list(detach_path),
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=next_state,
        move=detach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    return hook_no + 1, next_state


def _phase2_track_has_capacity(
    *,
    target_track: str,
    adding_vehicle_nos: tuple[str, ...],
    working_state: ReplayState,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
) -> bool:
    capacity_m = capacity_by_track.get(target_track)
    if capacity_m is None:
        return False
    occupied_m = sum(
        float(vehicle_by_no[vehicle_no].vehicle_length)
        for vehicle_no in working_state.track_sequences.get(target_track, ())
    )
    adding_m = sum(float(vehicle_by_no[vehicle_no].vehicle_length) for vehicle_no in adding_vehicle_nos)
    return occupied_m + adding_m <= capacity_m + 1e-9


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
    master: MasterData,
) -> dict:
    stage_policy = stage.get("stagePolicy") if isinstance(stage, dict) else None
    stage_mode = str((stage_policy or {}).get("stageMode") or "")
    if stage_mode == "PHASE2_DEPOT_AREA_MARSHALLING":
        current_by_vehicle = {
            str(item["vehicleNo"]): dict(item)
            for item in current_vehicle_info
        }
        resolved_goals: list[dict[str, Any]] = []
        for goal in list(stage.get("vehicleGoals") or []):
            target_source = str(goal.get("targetSource") or "")
            if target_source == "PHASE2_TRANSFER_TO_CUN4":
                resolved_goals.append(dict(goal))
                continue
            vehicle_no = str(goal["vehicleNo"])
            current = current_by_vehicle[vehicle_no]
            resolved_goals.append(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": str(current["trackName"]),
                    "targetMode": "AREA",
                    "targetAreaCode": f"STAGE::{target_source or 'PHASE2_DYNAMIC_HOLD'}",
                    "targetSource": target_source or "PHASE2_DYNAMIC_HOLD",
                    "isSpotting": "",
                }
            )
        resolved_stage = dict(stage)
        resolved_stage["vehicleGoals"] = resolved_goals
        return resolved_stage
    if stage_mode == "PHASE3_JI_TO_DEPOT_ALLOCATION":
        return _resolve_phase3_depot_targets(
            stage=stage,
            current_vehicle_info=current_vehicle_info,
            current_state=current_state,
            master=master,
        )
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


def _resolve_phase3_depot_targets(
    *,
    stage: dict,
    current_vehicle_info: list[dict],
    current_state: ReplayState,
    master: MasterData,
) -> dict:
    current_by_vehicle = {
        str(item["vehicleNo"]): dict(item)
        for item in current_vehicle_info
    }
    stage_probe = dict(stage)
    stage_probe["vehicleGoals"] = [dict(goal) for goal in stage.get("vehicleGoals") or []]
    probe_payload = _build_stage_payload(
        track_info=[],
        current_vehicle_info=current_vehicle_info,
        vehicle_meta=current_by_vehicle,
        stage=stage_probe,
        loco_track_name=current_state.loco_track_name,
    )
    normalized = normalize_plan_input(
        probe_payload,
        master,
        allow_internal_loco_tracks=True,
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    reserved_spot_codes = exact_spot_reservations(normalized)
    goal_by_vehicle = {
        str(goal["vehicleNo"]): dict(goal)
        for goal in stage.get("vehicleGoals") or []
    }
    planned_track_sequences = {
        track: [
            vehicle_no
            for vehicle_no in current_state.track_sequences.get(track, ())
            if _phase3_planned_to_stay_in_depot_track(
                vehicle_no=vehicle_no,
                current_track=track,
                goal=goal_by_vehicle.get(vehicle_no),
            )
        ]
        for track in ("修1", "修2", "修3", "修4")
    }
    planned_spot_assignments = {
        vehicle_no: spot_code
        for vehicle_no, spot_code in current_state.spot_assignments.items()
        if any(vehicle_no in seq for seq in planned_track_sequences.values())
    }
    for track, vehicle_nos in planned_track_sequences.items():
        realigned = realign_spots_for_track_order(
            vehicle_nos_in_order=vehicle_nos,
            vehicle_by_no=vehicle_by_no,
            target_track=track,
            yard_mode=normalized.yard_mode,
            current_spot_assignments=planned_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if realigned is not None:
            planned_spot_assignments = realigned
    resolved_goals: list[dict[str, Any]] = []
    for goal in list(stage.get("vehicleGoals") or []):
        if str(goal.get("targetSource") or "") == PHASE3_DYNAMIC_CURRENT_HOLD:
            vehicle_no = str(goal["vehicleNo"])
            current = current_by_vehicle[vehicle_no]
            resolved_goals.append(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": str(current["trackName"]),
                    "targetMode": "TRACK",
                    "targetSource": PHASE3_DYNAMIC_CURRENT_HOLD,
                    "isSpotting": "",
                }
            )
            continue
        if (
            str(goal.get("targetAreaCode") or "") != "大库:RANDOM"
            and str(goal.get("targetTrack") or "") != "大库"
        ):
            resolved_goals.append(dict(goal))
            continue
        vehicle_no = str(goal["vehicleNo"])
        vehicle = current_by_vehicle[vehicle_no]
        current_track = str(vehicle.get("trackName") or "")
        if current_track in {"修1", "修2", "修3", "修4"}:
            resolved_goals.append({
                "vehicleNo": vehicle_no,
                "targetTrack": current_track,
                "targetMode": "TRACK",
                "targetSource": "PHASE3_CONCRETE_DEPOT_TRACK",
                "isSpotting": "",
            })
            continue
        target_track = _choose_phase3_depot_track(
            goal=goal,
            vehicle=vehicle,
            planned_track_sequences=planned_track_sequences,
            planned_spot_assignments=planned_spot_assignments,
            vehicle_by_no=vehicle_by_no,
            yard_mode=normalized.yard_mode,
            reserved_spot_codes=reserved_spot_codes,
        )
        if target_track is None:
            target_track = "大库"
        else:
            next_sequence = [vehicle_no] + list(planned_track_sequences.get(target_track, ()))
            next_spot_assignments = realign_spots_for_track_order(
                vehicle_nos_in_order=next_sequence,
                vehicle_by_no=vehicle_by_no,
                target_track=target_track,
                yard_mode=normalized.yard_mode,
                current_spot_assignments=planned_spot_assignments,
                reserved_spot_codes=reserved_spot_codes,
            )
            if next_spot_assignments is not None:
                planned_track_sequences[target_track] = next_sequence
                planned_spot_assignments = next_spot_assignments
        resolved_goals.append({
            "vehicleNo": vehicle_no,
            "targetTrack": target_track,
            "targetMode": "TRACK",
            "targetSource": "PHASE3_CONCRETE_DEPOT_TRACK",
            "isSpotting": "",
        })
    resolved_stage = dict(stage)
    resolved_stage["vehicleGoals"] = resolved_goals
    resolved_policy = dict(stage.get("stagePolicy") or {})
    resolved_policy["phase3ConcreteDepotCounts"] = {
        track: len(planned_track_sequences.get(track, ()))
        for track in sorted(planned_track_sequences)
    }
    resolved_stage["stagePolicy"] = resolved_policy
    return resolved_stage


def _phase3_planned_to_stay_in_depot_track(
    *,
    vehicle_no: str,
    current_track: str,
    goal: dict[str, Any] | None,
) -> bool:
    if current_track not in {"修1", "修2", "修3", "修4"}:
        return False
    if not goal:
        return True
    target_track = str(goal.get("targetTrack") or "")
    target_area_code = str(goal.get("targetAreaCode") or "")
    if target_track == current_track:
        return True
    if target_track == "大库" or target_area_code == "大库:RANDOM":
        return True
    return False


def _assign_phase3_random_depot_targets(
    *,
    goals: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict],
    planned_track_sequences: dict[str, list[str]],
    planned_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, Any],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    pending_goals = [
        dict(goal)
        for goal in goals
        if _phase3_goal_needs_random_depot_assignment(
            goal=goal,
            current_by_vehicle=current_by_vehicle,
        )
    ]
    pending_goals.sort(
        key=lambda goal: _phase3_random_depot_assignment_key(
            goal=goal,
            current_by_vehicle=current_by_vehicle,
        )
    )
    result = _search_phase3_random_depot_assignment(
        pending_goals=pending_goals,
        index=0,
        planned_track_sequences={
            track: list(planned_track_sequences.get(track, ()))
            for track in ("修1", "修2", "修3", "修4")
        },
        planned_spot_assignments=dict(planned_spot_assignments),
        assigned_track_by_vehicle={},
        current_by_vehicle=current_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        yard_mode=yard_mode,
        reserved_spot_codes=reserved_spot_codes,
    )
    if result is None:
        vehicle_nos = [str(goal["vehicleNo"]) for goal in pending_goals]
        raise ValueError(f"phase3 random depot targets are infeasible: {vehicle_nos}")
    return result


def _phase3_goal_needs_random_depot_assignment(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> bool:
    if str(goal.get("targetSource") or "") == PHASE3_DYNAMIC_CURRENT_HOLD:
        return False
    if (
        str(goal.get("targetAreaCode") or "") != "大库:RANDOM"
        and str(goal.get("targetTrack") or "") != "大库"
    ):
        return False
    vehicle_no = str(goal["vehicleNo"])
    current_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
    return current_track not in {"修1", "修2", "修3", "修4"}


def _phase3_random_depot_assignment_key(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> tuple[Any, ...]:
    vehicle = current_by_vehicle[str(goal["vehicleNo"])]
    candidates = _phase3_random_depot_candidate_tracks(goal=goal, vehicle=vehicle)
    return (
        len(candidates),
        0 if str(vehicle.get("repairProcess") or "") == "厂修" else 1,
        0 if float(vehicle.get("vehicleLength") or 0.0) >= 17.6 else 1,
        str(goal["vehicleNo"]),
    )


def _search_phase3_random_depot_assignment(
    *,
    pending_goals: list[dict[str, Any]],
    index: int,
    planned_track_sequences: dict[str, list[str]],
    planned_spot_assignments: dict[str, str],
    assigned_track_by_vehicle: dict[str, str],
    current_by_vehicle: dict[str, dict],
    vehicle_by_no: dict[str, Any],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]] | None:
    if index >= len(pending_goals):
        return planned_track_sequences, planned_spot_assignments, assigned_track_by_vehicle
    goal = pending_goals[index]
    vehicle_no = str(goal["vehicleNo"])
    vehicle = current_by_vehicle[vehicle_no]
    for target_track in _ordered_phase3_random_depot_candidate_tracks(
        goal=goal,
        vehicle=vehicle,
        planned_track_sequences=planned_track_sequences,
    ):
        next_track_seq = [vehicle_no] + list(planned_track_sequences.get(target_track, ()))
        next_spot_assignments = realign_spots_for_track_order(
            vehicle_nos_in_order=next_track_seq,
            vehicle_by_no=vehicle_by_no,
            target_track=target_track,
            yard_mode=yard_mode,
            current_spot_assignments=planned_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        )
        if next_spot_assignments is None:
            continue
        next_sequences = {
            track: list(sequence)
            for track, sequence in planned_track_sequences.items()
        }
        next_sequences[target_track] = next_track_seq
        next_assigned = dict(assigned_track_by_vehicle)
        next_assigned[vehicle_no] = target_track
        result = _search_phase3_random_depot_assignment(
            pending_goals=pending_goals,
            index=index + 1,
            planned_track_sequences=next_sequences,
            planned_spot_assignments=next_spot_assignments,
            assigned_track_by_vehicle=next_assigned,
            current_by_vehicle=current_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            yard_mode=yard_mode,
            reserved_spot_codes=reserved_spot_codes,
        )
        if result is not None:
            return result
    return None


def _ordered_phase3_random_depot_candidate_tracks(
    *,
    goal: dict[str, Any],
    vehicle: dict,
    planned_track_sequences: dict[str, list[str]],
) -> list[str]:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    candidates = _phase3_random_depot_candidate_tracks(goal=goal, vehicle=vehicle)
    return sorted(
        candidates,
        key=lambda track: (
            0 if track in preferred else 1,
            len(planned_track_sequences.get(track, ())),
            track,
        ),
    )


def _phase3_random_depot_candidate_tracks(
    *,
    goal: dict[str, Any],
    vehicle: dict,
) -> list[str]:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    fallback = [str(item) for item in goal.get("fallbackTargetTracks") or ()]
    allowed = [str(item) for item in goal.get("allowedTargetTracks") or ()]
    candidates = [
        track
        for track in [*preferred, *fallback, *allowed]
        if track in {"修1", "修2", "修3", "修4"}
    ]
    if not candidates:
        candidates = ["修1", "修2", "修3", "修4"]
    if float(vehicle.get("vehicleLength") or 0.0) >= 17.6:
        candidates = [track for track in candidates if track in {"修3", "修4"}]
    return list(dict.fromkeys(candidates))


def _build_phase3_wave_plans(
    goals: list[dict[str, Any]],
    *,
    current_by_vehicle: dict[str, dict],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for goal in goals:
        target_track = str(goal.get("targetTrack") or "")
        if target_track == "大库":
            continue
        vehicle_no = str(goal["vehicleNo"])
        source_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
        if target_track == source_track:
            continue
        if target_track not in {"修1", "修2", "修3", "修4", "轮"}:
            continue
        group_key = target_track
        grouped.setdefault(group_key, {})[vehicle_no] = dict(goal)
    wave_order = ("轮", "修4", "修3", "修2", "修1")
    wave_plans: list[dict[str, Any]] = []
    for group_key in wave_order:
        active_goals = grouped.get(group_key)
        if not active_goals:
            continue
        wave_plans.append(
                {
                    "waveName": f"phase3_{group_key}",
                    "waveTargetTrack": group_key,
                    "waveWeight": 1.0,
                    "activeGoalsByVehicle": active_goals,
                }
        )
    return wave_plans


def _choose_phase3_depot_track(
    *,
    goal: dict[str, Any],
    vehicle: dict,
    planned_track_sequences: dict[str, list[str]],
    planned_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, Any],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> str | None:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    fallback = [str(item) for item in goal.get("fallbackTargetTracks") or ()]
    allowed = [str(item) for item in goal.get("allowedTargetTracks") or ()]
    candidates = [
        track
        for track in [*preferred, *fallback, *allowed]
        if track in {"修1", "修2", "修3", "修4"}
    ]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates = ["修1", "修2", "修3", "修4"]

    vehicle_length = float(vehicle.get("vehicleLength") or 0.0)
    repair_process = str(vehicle.get("repairProcess") or "")

    def usable(track: str) -> bool:
        if vehicle_length >= 17.6 and track not in {"修3", "修4"}:
            return False
        next_track_seq = [str(vehicle["vehicleNo"])] + list(planned_track_sequences.get(track, ()))
        return realign_spots_for_track_order(
            vehicle_nos_in_order=next_track_seq,
            vehicle_by_no=vehicle_by_no,
            target_track=track,
            yard_mode=yard_mode,
            current_spot_assignments=planned_spot_assignments,
            reserved_spot_codes=reserved_spot_codes,
        ) is not None

    usable_candidates = [track for track in candidates if usable(track)]
    if not usable_candidates:
        return None

    def key(track: str) -> tuple[int, int, int, str]:
        preferred_rank = 0 if track in preferred else 1
        if repair_process == "厂修":
            load_rank = len(planned_track_sequences.get(track, ()))
        else:
            load_rank = len(planned_track_sequences.get(track, ()))
        long_rank = 0 if (vehicle_length >= 17.6 and track in {"修3", "修4"}) else 1
        return (preferred_rank, load_rank, long_rank, track)

    return sorted(usable_candidates, key=key)[0]


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
