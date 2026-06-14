from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from fzed_shunting.domain.carry_order import is_carried_tail_block
from fzed_shunting.domain.hook_constraints import validate_hook_vehicle_group
from fzed_shunting.domain.master_data import MasterData, clone_master_with_blocked_branches
from fzed_shunting.domain.depot_spots import exact_spot_reservations, list_track_spots, realign_spots_for_track_order
from fzed_shunting.io.normalize_input import SOURCE_TRACK_ALIASES, normalize_plan_input
from fzed_shunting.verify.replay import ReplayState, build_initial_state, replay_plan
from fzed_shunting.workflow.phase3_depot_block_planner import build_phase3_depot_block_plan
from fzed_shunting.workflow.phase3_depot_insertion import plan_phase3_depot_insertion
from fzed_shunting.workflow.phase3_depot_relayout import (
    score_phase3_depot_execution_fitness,
    search_phase3_depot_relayout,
)
from fzed_shunting.workflow.phase1_rolling_planner import (
    Phase1RollingCandidate,
    build_phase1_rolling_candidates,
    phase1_rolling_selected_block_ids,
)
from fzed_shunting.workflow.l7_closed_topology_mode import (
    JI_BUFFER_TRACKS,
    PHASE1_TEMP_PARKING_TRACKS,
    PHASE3_DYNAMIC_CURRENT_HOLD,
    PHASE4_DYNAMIC_CURRENT_HOLD,
    PHASE4_RESIDUAL_CLEANUP,
    PHASE2_L1_TRANSFER_MAX_LENGTH_M,
    build_l7_closed_topology_workflow_payload,
    is_l7_closed_topology_mode,
    rebuild_phase1_stage_for_runtime,
    rebuild_phase2_execution_policy_for_runtime,
)

PHASE1_RUNTIME_WAVE_MAX_ACTIVE_VEHICLES = 12


@dataclass(frozen=True)
class Phase1ExecutableCheck:
    ok: bool
    reason: str
    source_track: str
    target_track: str
    vehicle_nos: tuple[str, ...]
    source_prefix: tuple[str, ...]
    loco_track: str
    blocking_tracks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def diagnostic(self, candidate_wave: dict[str, Any]) -> dict[str, Any]:
        return {
            "waveName": str(candidate_wave.get("waveName") or ""),
            "selectedSourceTrack": self.source_track,
            "selectedBlockIds": list(candidate_wave.get("selectedBlockIds") or []),
            "selectedVehicleNos": list(self.vehicle_nos),
            "targetTrack": self.target_track,
            "error": self.reason,
            "sourcePrefix": list(self.source_prefix),
            "locoTrack": self.loco_track,
            "blockingTracks": list(self.blocking_tracks),
            "routeErrors": list(self.errors),
        }

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
PHASE2_REORDER_BUFFER_TRACKS = (
    "修4库外",
    "修3库外",
    "修2库外",
    "修1库外",
    "存5北",
    "存5南",
    "存1",
    "存2",
    "存3",
    "存4南",
)
PHASE3_ADMISSION_MAX_RANDOM_DEPOT_VEHICLES = 10

@dataclass(frozen=True)
class Phase2ReleaseTask:
    blocked_source_track: str
    blocker_track: str
    blocker_vehicle_nos: tuple[str, ...]
    release_target_track: str
    reason: str


@dataclass(frozen=True)
class Phase2ReorderTask:
    source_track: str
    current_prefix: tuple[str, ...]
    desired_prefix: tuple[str, ...]
    outbound_vehicle_nos: tuple[str, ...]
    cun4_vehicle_nos: tuple[str, ...]


@dataclass(frozen=True)
class Phase2ReorderBufferCandidate:
    task: Phase2ReorderTask
    target_track: str
    score: tuple[Any, ...]

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
                track_info=track_info,
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
            stage_mode = str((stage.get("stagePolicy") or {}).get("stageMode") or "")
            stage_policy = dict(stage.get("stagePolicy") or {})
            if (
                stage_mode == "PHASE1_PRE_REPAIR_BUFFERING"
                and bool(stage_policy.get("phase1OriginalGoalRows"))
            ):
                view, current_vehicle_info, current_state = _solve_phase1_runtime_wave_stage(
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
            elif stage_mode == "PHASE2_DEPOT_AREA_MARSHALLING":
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
            elif wave_plan_key is not None:
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
        if wave_plan_key == "phase1WavePlans":
            active_goals = {
                str(goal.get("vehicleNo")): dict(goal)
                for goal in sub_vehicle_goals
                if str(goal.get("targetSource") or "") not in {"HOLD_CURRENT", "STAGE_HOLD", "FIXED_DEPOT_RESIDENT"}
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
                        "targetMode": "SNAPSHOT",
                        "targetSource": "HOLD_CURRENT",
                        "isSpotting": "",
                    }
                )
        elif wave_plan_key == "phase2WavePlans":
            active_goals = {
                str(vehicle_no): dict(goal)
                for vehicle_no, goal in dict(wave_plan.get("activeGoalsByVehicle") or {}).items()
            }
            reorder_outbound_vehicle_nos = {
                str(vehicle_no)
                for vehicle_no in dict(wave_plan.get("waveDiagnostics") or {}).get("outboundVehicleNos") or ()
            }
            if active_goals:
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
                            "targetMode": "SNAPSHOT",
                            "targetSource": "PHASE2_DYNAMIC_HOLD",
                            "isSpotting": "",
                        }
                    )
            else:
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
                            "targetMode": "SNAPSHOT",
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
                        "targetMode": "SNAPSHOT",
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
        if (
            wave_plan_key == "phase1WavePlans"
            and str(wave_plan.get("waveType") or "") == "phase1_rolling"
        ):
            normalized = normalize_plan_input(
                sub_stage_payload,
                stage_master,
                allow_internal_loco_tracks=True,
            )
            route_oracle = RouteOracle(stage_master)
            vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
            plan_payload = _build_phase1_rolling_plan_payload(
                candidate_wave=wave_plan,
                working_state=working_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
            )
        elif wave_plan_key == "phase2WavePlans":
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
            if attach_units:
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
            elif active_goals:
                plan_payload = _build_phase2_reorder_plan_payload(
                    normalized=normalized,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    working_state=working_state,
                    active_goals=active_goals,
                    outbound_vehicle_nos=reorder_outbound_vehicle_nos,
                    capacity_by_track={
                        info.track_name: float(info.track_distance)
                        for info in normalized.track_info
                    },
                )
                if plan_payload is None:
                    raise ValueError("phase2 reorder wave has no executable explicit plan")
        elif wave_plan_key == "phase3WavePlans":
            normalized = normalize_plan_input(
                sub_stage_payload,
                stage_master,
                allow_internal_loco_tracks=True,
            )
            route_oracle = RouteOracle(stage_master)
            vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
            try:
                plan_payload = _build_phase3_wave_plan_payload(
                    wave_plan=wave_plan,
                    working_state=working_state,
                    normalized=normalized,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                )
            except ValueError:
                if bool(wave_plan.get("requiresExplicitPlan")):
                    raise
                plan_payload = None
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
            diagnostics=dict(getattr(final_view, "diagnostics", {}) or {}),
            vehicle_target_tracks=dict(getattr(final_view, "vehicle_target_tracks", {}) or {}),
        ),
        working_vehicle_info,
        working_state,
    )


def _solve_phase1_runtime_wave_stage(
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
    from fzed_shunting.demo.view_model import DemoHook, DemoSummary, DemoViewModel

    working_vehicle_info = [dict(item) for item in current_vehicle_info]
    working_state = current_state
    wave_views: list[DemoViewModel] = []
    stage_policy = dict(stage.get("stagePolicy") or {})
    original_goal_rows = list(stage_policy.get("phase1OriginalGoalRows") or [])
    initial_buffer_vehicle_nos = frozenset(
        str(item)
        for item in list(stage_policy.get("phase1InitialBufferVehicleNos") or [])
        if str(item)
    )
    if not original_goal_rows:
        raise ValueError("phase1 runtime solve requires phase1OriginalGoalRows")

    runtime_stage = rebuild_phase1_stage_for_runtime(
        master=stage_master,
        track_info=track_info,
        current_vehicle_info=working_vehicle_info,
        loco_track_name=working_state.loco_track_name,
        original_goal_rows=original_goal_rows,
        initial_buffer_vehicle_nos=initial_buffer_vehicle_nos,
    )
    remaining_selected_block_ids = phase1_rolling_selected_block_ids(runtime_stage)
    if not remaining_selected_block_ids:
        return (
            _empty_phase1_runtime_view(
                state=working_state,
                vehicle_count=len(working_vehicle_info),
            ),
            working_vehicle_info,
            working_state,
        )
    completed_selected_block_ids: set[str] = set()
    completed_selected_vehicle_nos: set[str] = set()
    active_macro_task_id = ""
    max_iterations = max(1, len(remaining_selected_block_ids) + 12)
    rolling_failures: list[dict[str, Any]] = []
    rolling_selected_history: list[dict[str, Any]] = []
    for _ in range(max_iterations):
        previous_vehicle_layout = tuple(
            (str(item["vehicleNo"]), str(item["trackName"]), str(item["order"]))
            for item in working_vehicle_info
        )
        current_selected_block_ids = frozenset(remaining_selected_block_ids)
        if not current_selected_block_ids:
            break
        candidates = build_phase1_rolling_candidates(
            runtime_stage=runtime_stage,
            selected_block_ids=current_selected_block_ids,
        )
        candidates = _phase1_rolling_candidates_with_route_clearance(
            runtime_stage=runtime_stage,
            base_candidates=candidates,
            state=working_state,
            master=stage_master,
            vehicle_meta=vehicle_meta,
            active_macro_task_id=active_macro_task_id,
        )
        candidates, prefiltered_candidate_errors = _phase1_filter_executable_rolling_candidates(
            candidates=candidates,
            state=working_state,
            master=stage_master,
        )
        candidates = _rank_phase1_rolling_candidates_for_runtime(
            runtime_stage=runtime_stage,
            candidates=candidates,
            vehicle_meta=vehicle_meta,
            state=working_state,
            master=stage_master,
            selected_block_ids=current_selected_block_ids,
            active_macro_task_id=active_macro_task_id,
        )
        if not candidates:
            raise ValueError(
                "phase1 rolling planner produced no executable candidate: "
                + str(prefiltered_candidate_errors[:12])
            )
        selected_view = None
        selected_vehicle_info = None
        selected_state = None
        selected_candidate = None
        candidate_errors: list[dict[str, Any]] = []
        for candidate in candidates:
            is_route_clearance = (
                str((candidate.wave.get("waveDiagnostics") or {}).get("runtimeFrontierStrategy") or "")
                == "rolling_route_clearance"
            )
            executable_check = _phase1_check_rolling_candidate_executable(
                candidate_wave=candidate.wave,
                state=working_state,
                master=stage_master,
            )
            if not is_route_clearance and not executable_check.ok:
                candidate_errors.append(executable_check.diagnostic(candidate.wave))
                continue
            candidate_stage = dict(runtime_stage)
            runtime_policy = dict(runtime_stage.get("stagePolicy") or {})
            runtime_policy["phase1WavePlans"] = [candidate.wave]
            candidate_stage["stagePolicy"] = runtime_policy
            try:
                view, next_vehicle_info, next_state = _solve_wave_stage(
                    stage_master=stage_master,
                    track_info=track_info,
                    vehicle_meta=vehicle_meta,
                    stage=candidate_stage,
                    wave_plan_key="phase1WavePlans",
                    stage_payload=stage_payload,
                    current_vehicle_info=working_vehicle_info,
                    current_state=working_state,
                    solver=solver,
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width,
                    time_budget_ms=_phase1_rolling_candidate_time_budget_ms(time_budget_ms),
                    use_validation_recovery=use_validation_recovery,
                    diagnose_front_search_only=diagnose_front_search_only,
                )
                next_state = _phase1_state_with_actual_detach_entry_node(
                    view=view,
                    state=next_state,
                    master=stage_master,
                )
            except Exception as exc:
                candidate_errors.append(
                    {
                        "waveName": str(candidate.wave.get("waveName") or ""),
                        "selectedSourceTrack": str(candidate.wave.get("selectedSourceTrack") or ""),
                        "selectedBlockIds": list(candidate.wave.get("selectedBlockIds") or []),
                        "selectedVehicleNos": list(candidate.wave.get("selectedVehicleNos") or []),
                        "targetTrack": str((candidate.wave.get("waveDiagnostics") or {}).get("targetTrack") or ""),
                        "error": str(exc),
                    }
                )
                continue
            next_vehicle_layout = tuple(
                (str(item["vehicleNo"]), str(item["trackName"]), str(item["order"]))
                for item in next_vehicle_info
            )
            if next_vehicle_layout == previous_vehicle_layout:
                candidate_errors.append(
                    {
                        "waveName": str(candidate.wave.get("waveName") or ""),
                        "selectedSourceTrack": str(candidate.wave.get("selectedSourceTrack") or ""),
                        "selectedBlockIds": list(candidate.wave.get("selectedBlockIds") or []),
                        "selectedVehicleNos": list(candidate.wave.get("selectedVehicleNos") or []),
                        "targetTrack": str((candidate.wave.get("waveDiagnostics") or {}).get("targetTrack") or ""),
                        "error": "candidate made no vehicle-layout progress",
                    }
                )
                continue
            selected_view = view
            selected_vehicle_info = next_vehicle_info
            selected_state = next_state
            selected_candidate = candidate
            break
        rolling_failures.extend((prefiltered_candidate_errors + candidate_errors)[:6])
        if selected_view is None or selected_vehicle_info is None or selected_state is None:
            raise ValueError(
                "phase1 rolling planner could not solve any candidate: "
                + str(candidate_errors[:12])
                + "; selected history: "
                + str(rolling_selected_history[-12:])
            )
        view = selected_view
        working_vehicle_info = selected_vehicle_info
        working_state = selected_state
        is_selected_route_clearance = (
            str((selected_candidate.wave.get("waveDiagnostics") or {}).get("runtimeFrontierStrategy") or "")
            == "rolling_route_clearance"
        )
        if is_selected_route_clearance:
            runtime_stage = rebuild_phase1_stage_for_runtime(
                master=stage_master,
                track_info=track_info,
                current_vehicle_info=working_vehicle_info,
                loco_track_name=working_state.loco_track_name,
                original_goal_rows=original_goal_rows,
                initial_buffer_vehicle_nos=initial_buffer_vehicle_nos,
            )
            remaining_selected_block_ids = _phase1_selected_block_ids_excluding_completed_vehicles(
                runtime_stage,
                completed_vehicle_nos=frozenset(completed_selected_vehicle_nos),
            )
        else:
            completed_block_ids = frozenset(
                str(block_id)
                for block_id in list(selected_candidate.wave.get("selectedBlockIds") or [])
                if selected_candidate and str(block_id) in remaining_selected_block_ids
            )
            completed_vehicle_nos = frozenset(
                str(vehicle_no)
                for vehicle_no in list(selected_candidate.wave.get("selectedVehicleNos") or [])
                if str(vehicle_no)
            )
            completed_selected_block_ids.update(completed_block_ids)
            completed_selected_vehicle_nos.update(completed_vehicle_nos)
            remaining_selected_block_ids = frozenset(remaining_selected_block_ids - completed_block_ids)
        selected_macro_task_id = str(
            (selected_candidate.wave.get("waveDiagnostics") or {}).get("macroTaskId") or ""
        ) if selected_candidate else ""
        if selected_macro_task_id:
            selected_macro_block_ids = {
                str(block_id)
                for block_id in list(
                    (selected_candidate.wave.get("waveDiagnostics") or {}).get("macroTaskBlockIds") or []
                )
                if str(block_id)
            }
            if selected_macro_block_ids & set(remaining_selected_block_ids):
                active_macro_task_id = selected_macro_task_id
            else:
                active_macro_task_id = ""
        diagnostics = dict(getattr(view, "diagnostics", {}) or {})
        diagnostics["phase1RollingSelectedCandidate"] = {
            "waveName": str(selected_candidate.wave.get("waveName") or "") if selected_candidate else "",
            "selectedSourceTrack": str(selected_candidate.wave.get("selectedSourceTrack") or "") if selected_candidate else "",
            "selectedBlockIds": list(selected_candidate.wave.get("selectedBlockIds") or []) if selected_candidate else [],
            "selectedVehicleNos": list(selected_candidate.wave.get("selectedVehicleNos") or []) if selected_candidate else [],
            "targetTrack": str((selected_candidate.wave.get("waveDiagnostics") or {}).get("targetTrack") or "") if selected_candidate else "",
            "macroTaskId": str((selected_candidate.wave.get("waveDiagnostics") or {}).get("macroTaskId") or "") if selected_candidate else "",
            "score": list(selected_candidate.score) if selected_candidate else [],
        }
        rolling_selected_history.append(dict(diagnostics["phase1RollingSelectedCandidate"]))
        view.diagnostics = diagnostics
        wave_views.append(view)
        if not remaining_selected_block_ids:
            break
        next_vehicle_layout = tuple(
            (str(item["vehicleNo"]), str(item["trackName"]), str(item["order"]))
            for item in working_vehicle_info
        )
        if (
            remaining_selected_block_ids == current_selected_block_ids
            and next_vehicle_layout == previous_vehicle_layout
        ):
            raise ValueError(
                "phase1 runtime frontier wave made no structural progress: "
                + ",".join(sorted(current_selected_block_ids))
            )
    if not wave_views:
        raise ValueError("phase1 runtime solve produced no executable wave")

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
            merged_hooks.append(hook.model_copy(update={"hook_no": new_hook_no}))
        hook_offset = len(merged_hooks)
        step_start = 0 if step_index == 0 else 1
        for raw_step in list(getattr(wave_view, "steps", []) or [])[step_start:]:
            hook = getattr(raw_step, "hook", None)
            remapped_hook = None if hook is None else hook.model_copy(
                update={"hook_no": renumbered_hook_nos.get(hook.hook_no, hook.hook_no)}
            )
            merged_steps.append(
                raw_step.model_copy(
                    update={
                        "step_index": step_index,
                        "hook": remapped_hook,
                    }
                )
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
            diagnostics={
                **dict(getattr(final_view, "diagnostics", {}) or {}),
                "phase1RollingCandidateFailureSamples": rolling_failures[:20],
                "phase1RollingSelectedHistory": list(rolling_selected_history),
                "phase1RollingWaveCount": len(wave_views),
            },
            vehicle_target_tracks=dict(getattr(final_view, "vehicle_target_tracks", {}) or {}),
        ),
        working_vehicle_info,
        working_state,
    )


def _empty_phase1_runtime_view(
    *,
    state: ReplayState,
    vehicle_count: int,
) -> Any:
    from fzed_shunting.demo.view_model import DemoStep, DemoSummary, DemoViewModel

    return DemoViewModel(
        summary=DemoSummary(
            hook_count=0,
            vehicle_count=vehicle_count,
            is_valid=True,
            error_count=0,
            final_tracks=sorted(track for track, seq in state.track_sequences.items() if seq),
            weighed_vehicle_count=len(state.weighed_vehicle_nos),
            assigned_spot_count=len(state.spot_assignments),
            assigned_work_position_count=0,
        ),
        verifier_errors=[],
        hook_plan=[],
        steps=[
            DemoStep(
                step_index=0,
                hook=None,
                loco_track_name=state.loco_track_name,
                loco_node=state.loco_node,
                loco_carry_vehicle_nos=list(state.loco_carry),
                changed_tracks=[],
                track_sequences={track: list(seq) for track, seq in state.track_sequences.items()},
                weighed_vehicle_nos=sorted(state.weighed_vehicle_nos),
                spot_assignments=dict(state.spot_assignments),
                work_position_assignments={},
                verifier_errors=[],
            )
        ],
        final_spot_assignments=dict(state.spot_assignments),
        final_work_position_assignments={},
        failed_hook_nos=[],
        diagnostics={
            "phase1RollingSelectedHistory": [],
            "phase1RollingWaveCount": 0,
        },
        vehicle_target_tracks={},
    )


def _phase1_runtime_selected_block_ids(stage: dict[str, Any]) -> frozenset[str]:
    diagnostics = dict(((stage.get("stagePolicy") or {}).get("phase1Diagnostics")) or {})
    return frozenset(
        str(block.get("blockId") or "")
        for block in list(diagnostics.get("phase1Blocks") or [])
        if str(block.get("blockId") or "")
        and (bool(block.get("selectedBackbone")) or bool(block.get("selectedFinish")))
    )


def _phase1_selected_block_ids_excluding_completed_vehicles(
    stage: dict[str, Any],
    *,
    completed_vehicle_nos: frozenset[str],
) -> frozenset[str]:
    diagnostics = dict(((stage.get("stagePolicy") or {}).get("phase1Diagnostics")) or {})
    return frozenset(
        str(block.get("blockId") or "")
        for block in list(diagnostics.get("phase1Blocks") or [])
        if str(block.get("blockId") or "")
        and bool(block.get("selectedBackbone"))
        and not set(str(vehicle_no) for vehicle_no in list(block.get("vehicleNos") or [])) <= completed_vehicle_nos
    )


def _phase1_rolling_candidate_time_budget_ms(time_budget_ms: float | None) -> float | None:
    if time_budget_ms is None:
        return 8_000.0
    return max(2_000.0, min(8_000.0, float(time_budget_ms) * 0.12))


def _phase1_state_with_actual_detach_entry_node(
    *,
    view: Any,
    state: ReplayState,
    master: MasterData,
) -> ReplayState:
    from fzed_shunting.domain.route_oracle import RouteOracle

    hook_plan = list(getattr(view, "hook_plan", []) or [])
    if not hook_plan:
        return state
    last_hook = hook_plan[-1]
    if str(getattr(last_hook, "action_type", "") or "") != "DETACH":
        return state
    path_tracks = list(getattr(last_hook, "path_tracks", []) or [])
    entry_node = RouteOracle(master).path_entry_node(path_tracks)
    if entry_node is None:
        return state
    return state.model_copy(update={"loco_node": entry_node})


def _phase1_filter_executable_rolling_candidates(
    *,
    candidates: tuple[Phase1RollingCandidate, ...],
    state: ReplayState,
    master: MasterData,
) -> tuple[tuple[Phase1RollingCandidate, ...], list[dict[str, Any]]]:
    executable: list[Phase1RollingCandidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        strategy = str((candidate.wave.get("waveDiagnostics") or {}).get("runtimeFrontierStrategy") or "")
        if strategy == "rolling_route_clearance":
            executable.append(candidate)
            continue
        check = _phase1_check_rolling_candidate_executable(
            candidate_wave=candidate.wave,
            state=state,
            master=master,
        )
        if check.ok:
            executable.append(candidate)
        else:
            rejected.append(check.diagnostic(candidate.wave))
    return tuple(executable), rejected


def _rank_phase1_rolling_candidates_for_runtime(
    *,
    runtime_stage: dict[str, Any],
    candidates: tuple[Phase1RollingCandidate, ...],
    vehicle_meta: dict[str, dict],
    state: ReplayState,
    master: MasterData,
    selected_block_ids: frozenset[str],
    active_macro_task_id: str = "",
) -> tuple[Phase1RollingCandidate, ...]:
    if len(candidates) <= 1:
        return candidates

    scored: list[tuple[tuple[Any, ...], Phase1RollingCandidate]] = []
    score_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for candidate in candidates:
        diagnostics = dict(candidate.wave.get("waveDiagnostics") or {})
        candidate_macro_task_id = str(diagnostics.get("macroTaskId") or "")
        score = _phase1_rolling_candidate_future_score(
            runtime_stage=runtime_stage,
            candidate=candidate,
            selected_block_ids=selected_block_ids,
            vehicle_meta=vehicle_meta,
            state=state,
            master=master,
            depth=2,
            visited=frozenset(),
            score_cache=score_cache,
        )
        macro_continuation_key = (
            0
            if active_macro_task_id and candidate_macro_task_id == active_macro_task_id
            else 1
            if active_macro_task_id
            else 0
        )
        scored.append((
            _phase1_flat_runtime_score((macro_continuation_key,), candidate.score, score),
            candidate,
        ))
    scored.sort(key=lambda item: item[0])
    return tuple(candidate for _, candidate in scored)


def _phase1_rolling_candidate_future_score(
    *,
    runtime_stage: dict[str, Any],
    candidate: Phase1RollingCandidate,
    selected_block_ids: frozenset[str],
    vehicle_meta: dict[str, dict],
    state: ReplayState,
    master: MasterData,
    depth: int,
    visited: frozenset[tuple[Any, ...]],
    score_cache: dict[tuple[Any, ...], tuple[Any, ...]],
) -> tuple[Any, ...]:
    cache_key = (
        _phase1_state_key(state),
        str(candidate.wave.get("waveName") or ""),
        tuple(str(vehicle_no) for vehicle_no in candidate.wave.get("selectedVehicleNos") or ()),
        str((candidate.wave.get("waveDiagnostics") or {}).get("targetTrack") or ""),
        tuple(sorted(selected_block_ids)),
        depth,
    )
    cached = score_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        next_state = _preview_phase1_rolling_candidate_state(
            candidate_wave=candidate.wave,
            vehicle_meta=vehicle_meta,
            state=state,
            master=master,
        )
    except Exception:
        score = (1, 1, len(selected_block_ids), 99, 99, 99)
        score_cache[cache_key] = score
        return score

    completed_block_ids = frozenset(
        str(block_id)
        for block_id in list(candidate.wave.get("selectedBlockIds") or [])
        if str(block_id) in selected_block_ids
    )
    next_selected_block_ids = selected_block_ids - completed_block_ids
    next_selected_count = len(next_selected_block_ids)
    removed_count = len(completed_block_ids)
    if next_selected_count == 0:
        score = (0, 0, 0, -removed_count, 0, 0)
        score_cache[cache_key] = score
        return score

    state_key = _phase1_state_key(next_state)
    if state_key in visited:
        score = (1, 1, next_selected_count, -removed_count, 99, 99)
        score_cache[cache_key] = score
        return score

    next_candidates = build_phase1_rolling_candidates(
        runtime_stage=runtime_stage,
        selected_block_ids=next_selected_block_ids,
    )
    next_candidates = _phase1_rolling_candidates_with_route_clearance(
        runtime_stage=runtime_stage,
        base_candidates=next_candidates,
        state=next_state,
        master=master,
        vehicle_meta=vehicle_meta,
        active_macro_task_id=str((candidate.wave.get("waveDiagnostics") or {}).get("macroTaskId") or ""),
    )
    executable_next_candidates = tuple(
        item
        for item in next_candidates
        if _phase1_candidate_is_preview_executable(
            candidate=item,
            state=next_state,
            master=master,
            vehicle_meta=vehicle_meta,
        )
    )
    executable_frontier_count = len(executable_next_candidates)
    direct_frontier_count = sum(
        1
        for item in executable_next_candidates
        if str((item.wave.get("waveDiagnostics") or {}).get("runtimeFrontierStrategy") or "")
        != "rolling_route_clearance"
    )
    if executable_frontier_count == 0:
        score = (1, 1, next_selected_count, -removed_count, 99, 99)
        score_cache[cache_key] = score
        return score
    if depth <= 0:
        score = (
            0,
            1,
            next_selected_count,
            -removed_count,
            -direct_frontier_count,
            -executable_frontier_count,
            0,
        )
        score_cache[cache_key] = score
        return score

    child_scores = [
        _phase1_rolling_candidate_future_score(
            runtime_stage=runtime_stage,
            candidate=item,
            selected_block_ids=next_selected_block_ids,
            vehicle_meta=vehicle_meta,
            state=next_state,
            master=master,
            depth=depth - 1,
            visited=visited | {state_key},
            score_cache=score_cache,
        )
        for item in executable_next_candidates[:8]
    ]
    best_child = min(child_scores) if child_scores else (1, 1, next_selected_count, 99, 99, 99)
    score = (
        best_child[0],
        best_child[1],
        best_child[2],
        -removed_count,
        -direct_frontier_count,
        -executable_frontier_count,
        best_child[3:],
    )
    score_cache[cache_key] = score
    return score


def _phase1_candidate_is_frontier_or_clearance(
    *,
    candidate: Phase1RollingCandidate,
    state: ReplayState,
    master: MasterData,
) -> bool:
    is_route_clearance = (
        str((candidate.wave.get("waveDiagnostics") or {}).get("runtimeFrontierStrategy") or "")
        == "rolling_route_clearance"
    )
    if is_route_clearance:
        return True
    return _phase1_rolling_candidate_is_physically_frontier(
        candidate_wave=candidate.wave,
        state=state,
        master=master,
    )


def _phase1_candidate_is_preview_executable(
    *,
    candidate: Phase1RollingCandidate,
    state: ReplayState,
    master: MasterData,
    vehicle_meta: dict[str, dict],
) -> bool:
    try:
        _preview_phase1_rolling_candidate_state(
            candidate_wave=candidate.wave,
            vehicle_meta=vehicle_meta,
            state=state,
            master=master,
        )
    except Exception:
        return False
    return True


def _preview_phase1_rolling_candidate_state(
    *,
    candidate_wave: dict[str, Any],
    vehicle_meta: dict[str, dict],
    state: ReplayState,
    master: MasterData,
) -> ReplayState:
    from fzed_shunting.domain.route_oracle import RouteOracle

    route_oracle = RouteOracle(master)
    source_track = str(candidate_wave.get("selectedSourceTrack") or "")
    target_track = str((candidate_wave.get("waveDiagnostics") or {}).get("targetTrack") or "")
    vehicle_nos = [
        str(vehicle_no)
        for vehicle_no in list(candidate_wave.get("selectedVehicleNos") or [])
        if str(vehicle_no)
    ]
    if not source_track or not target_track or not vehicle_nos:
        raise ValueError("phase1 preview wave is incomplete")
    source_seq = list(state.track_sequences.get(source_track, []))
    if source_seq[: len(vehicle_nos)] != vehicle_nos:
        raise ValueError("phase1 preview source prefix mismatch")
    train_length_m = sum(
        float(dict(vehicle_meta.get(vehicle_no) or {}).get("vehicleLength") or 0.0)
        for vehicle_no in vehicle_nos
    )
    target_capacity_m = float(master.tracks[target_track].effective_length_m) if target_track in master.tracks else 0.0
    existing_target_length_m = sum(
        float(dict(vehicle_meta.get(vehicle_no) or {}).get("vehicleLength") or 0.0)
        for vehicle_no in list(state.track_sequences.get(target_track, []))
    )
    if target_capacity_m and existing_target_length_m + train_length_m > target_capacity_m + 1e-9:
        raise ValueError("phase1 preview target capacity exceeded")
    attach_path = route_oracle.resolve_clear_path_tracks(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        raise ValueError("phase1 preview no clear attach path")
    after_attach_sequences = dict(state.track_sequences)
    after_attach_sequences[source_track] = list(source_seq[len(vehicle_nos):])
    detach_path = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=after_attach_sequences,
        source_node=route_oracle.order_end_node(source_track),
        target_node=None,
    )
    if detach_path is None:
        raise ValueError("phase1 preview no clear detach path")
    next_sequences = dict(after_attach_sequences)
    next_sequences[target_track] = list(vehicle_nos) + list(next_sequences.get(target_track, []))
    entry_node = route_oracle.path_entry_node(detach_path) or route_oracle.order_end_node(target_track)
    return state.model_copy(
        update={
            "track_sequences": next_sequences,
            "loco_track_name": target_track,
            "loco_node": entry_node,
            "loco_carry": (),
        }
    )


def _phase1_vehicle_info_from_state(
    *,
    current_vehicle_info: list[dict[str, Any]],
    state: ReplayState,
) -> list[dict[str, Any]]:
    row_by_vehicle = {str(item["vehicleNo"]): dict(item) for item in current_vehicle_info}
    next_rows: list[dict[str, Any]] = []
    for track_name, sequence in state.track_sequences.items():
        for order, vehicle_no in enumerate(sequence, start=1):
            row = dict(row_by_vehicle[str(vehicle_no)])
            row["trackName"] = track_name
            row["order"] = str(order)
            next_rows.append(row)
    return next_rows


def _phase1_state_key(state: ReplayState) -> tuple[Any, ...]:
    return (
        state.loco_track_name,
        state.loco_node,
        tuple(
            (track, tuple(sequence))
            for track, sequence in sorted(state.track_sequences.items())
            if sequence
        ),
        tuple(state.loco_carry),
    )


def _phase1_rolling_candidates_with_route_clearance(
    *,
    runtime_stage: dict[str, Any],
    base_candidates: tuple[Phase1RollingCandidate, ...],
    state: ReplayState,
    master: MasterData,
    vehicle_meta: dict[str, dict],
    active_macro_task_id: str = "",
) -> tuple[Phase1RollingCandidate, ...]:
    from fzed_shunting.domain.route_oracle import RouteOracle

    if not base_candidates:
        return tuple()
    route_oracle = RouteOracle(master)
    goal_by_vehicle = {
        str(goal.get("vehicleNo") or ""): dict(goal)
        for goal in list(runtime_stage.get("vehicleGoals") or [])
        if str(goal.get("vehicleNo") or "")
    }
    generated: list[Phase1RollingCandidate] = list(base_candidates)
    seen = {
        (
            str(candidate.wave.get("selectedSourceTrack") or ""),
            tuple(str(vehicle_no) for vehicle_no in candidate.wave.get("selectedVehicleNos") or ()),
            str((candidate.wave.get("waveDiagnostics") or {}).get("targetTrack") or ""),
        )
        for candidate in generated
    }
    pending_clearance_sources: list[tuple[str, tuple[str, ...] | None, dict[str, Any], int]] = []
    for candidate in base_candidates:
        source_prefix_blockers = _phase1_source_prefix_blocking_vehicle_nos(
            candidate_wave=candidate.wave,
            state=state,
        )
        if source_prefix_blockers:
            pending_clearance_sources.append(
                (
                    str(candidate.wave.get("selectedSourceTrack") or ""),
                    source_prefix_blockers,
                    candidate.wave,
                    0,
                )
            )
        executable_check = _phase1_check_rolling_candidate_executable(
            candidate_wave=candidate.wave,
            state=state,
            master=master,
        )
        check_blocking_tracks = tuple(
            track
            for track in executable_check.blocking_tracks
            if track
        )
        pending_clearance_sources.extend(
            (blocking_track, None, candidate.wave, 0)
            for blocking_track in dict.fromkeys(
                (
                    *check_blocking_tracks,
                    *_phase1_candidate_blocking_tracks(
                        candidate_wave=candidate.wave,
                        state=state,
                        route_oracle=route_oracle,
                    ),
                )
            )
        )
    processed_clearance_sources: set[tuple[str, tuple[str, ...] | None, int]] = set()
    while pending_clearance_sources:
        blocking_track, blocking_vehicle_nos, reason_wave, depth = pending_clearance_sources.pop(0)
        if (blocking_track, blocking_vehicle_nos, depth) in processed_clearance_sources:
            continue
        processed_clearance_sources.add((blocking_track, blocking_vehicle_nos, depth))
        if depth > 1:
            continue
        blocking_seq = list(blocking_vehicle_nos or state.track_sequences.get(blocking_track, []))
        if not blocking_seq:
            continue
        if len(blocking_seq) > PHASE1_RUNTIME_WAVE_MAX_ACTIVE_VEHICLES:
            continue
        generated_for_blocker = False
        for target_track in _phase1_clearance_target_tracks(blocking_track):
            key = (blocking_track, tuple(blocking_seq), target_track)
            if key in seen:
                continue
            if not _phase1_clearance_target_is_usable(
                source_track=blocking_track,
                target_track=target_track,
                vehicle_nos=blocking_seq,
                state=state,
                route_oracle=route_oracle,
                vehicle_meta=vehicle_meta,
                master=master,
            ):
                if depth == 0:
                    pending_clearance_sources.extend(
                        (secondary_blocker, None, reason_wave, depth + 1)
                        for secondary_blocker in _phase1_clearance_candidate_blocking_tracks(
                            source_track=blocking_track,
                            target_track=target_track,
                            vehicle_nos=blocking_seq,
                            state=state,
                            route_oracle=route_oracle,
                        )
                        if secondary_blocker != blocking_track
                    )
                continue
            wave = _build_phase1_clearance_wave(
                source_track=blocking_track,
                target_track=target_track,
                vehicle_nos=blocking_seq,
                goal_by_vehicle=goal_by_vehicle,
                reason_wave=reason_wave,
            )
            reason_macro_task_id = str((reason_wave.get("waveDiagnostics") or {}).get("macroTaskId") or "")
            macro_continuation_rank = 0 if active_macro_task_id and reason_macro_task_id == active_macro_task_id else 1
            clearance_blocker_count = len(
                _phase1_candidate_blocking_tracks(
                    candidate_wave=wave,
                    state=state,
                    route_oracle=route_oracle,
                )
            )
            generated.append(
                Phase1RollingCandidate(
                    wave=wave,
                    score=(
                        2,
                        macro_continuation_rank,
                        clearance_blocker_count,
                        len(blocking_seq),
                        _phase1_clearance_target_rank(target_track),
                        blocking_track,
                        target_track,
                        tuple(blocking_seq),
                    ),
                )
            )
            seen.add(key)
            generated_for_blocker = True
            break
        if generated_for_blocker:
            continue
    generated.sort(key=lambda item: _phase1_flat_runtime_score(item.score))
    return tuple(generated[: max(16, len(base_candidates) + 8)])


def _phase1_source_prefix_blocking_vehicle_nos(
    *,
    candidate_wave: dict[str, Any],
    state: ReplayState,
) -> tuple[str, ...]:
    source_track = str(candidate_wave.get("selectedSourceTrack") or "")
    vehicle_nos = tuple(
        str(vehicle_no)
        for vehicle_no in list(candidate_wave.get("selectedVehicleNos") or [])
        if str(vehicle_no)
    )
    source_seq = tuple(str(vehicle_no) for vehicle_no in list(state.track_sequences.get(source_track, [])))
    if not source_track or not vehicle_nos or source_seq[: len(vehicle_nos)] == vehicle_nos:
        return tuple()
    contiguous_start = _phase1_contiguous_subsequence_start(source_seq, vehicle_nos)
    if contiguous_start is not None:
        blocking_prefix = source_seq[:contiguous_start]
    else:
        selected_positions = [
            index
            for index, vehicle_no in enumerate(source_seq)
            if vehicle_no in set(vehicle_nos)
        ]
        if not selected_positions:
            return tuple()
        blocking_prefix = source_seq[: min(selected_positions)]
    return tuple(blocking_prefix[:PHASE1_RUNTIME_WAVE_MAX_ACTIVE_VEHICLES])


def _phase1_contiguous_subsequence_start(
    source_seq: tuple[str, ...],
    vehicle_nos: tuple[str, ...],
) -> int | None:
    if not vehicle_nos or len(vehicle_nos) > len(source_seq):
        return None
    for index in range(0, len(source_seq) - len(vehicle_nos) + 1):
        if source_seq[index : index + len(vehicle_nos)] == vehicle_nos:
            return index
    return None


def _phase1_flat_runtime_score(*parts: Any) -> tuple[Any, ...]:
    flattened: list[Any] = []
    for part in parts:
        if isinstance(part, (tuple, list)):
            flattened.extend(_phase1_flat_runtime_score(*part))
        else:
            flattened.append(_phase1_sortable_score_atom(part))
    return tuple(flattened)


def _phase1_sortable_score_atom(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    if value is None:
        return (2, "")
    if isinstance(value, str):
        return (1, value)
    return (3, repr(value))


def _phase1_candidate_blocking_tracks(
    *,
    candidate_wave: dict[str, Any],
    state: ReplayState,
    route_oracle,
) -> tuple[str, ...]:
    source_track = str(candidate_wave.get("selectedSourceTrack") or "")
    target_track = str((candidate_wave.get("waveDiagnostics") or {}).get("targetTrack") or "")
    if not source_track or not target_track:
        return tuple()
    blockers: list[str] = []
    attach_path = route_oracle.resolve_path_tracks_for_endpoint_constraints(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    ) or route_oracle.resolve_path_tracks(state.loco_track_name, source_track)
    if attach_path is not None:
        blockers.extend(
            route_oracle._blocking_tracks_for_path(
                attach_path,
                occupied_track_sequences=state.track_sequences,
                source_node=state.loco_node,
                target_node=route_oracle.order_end_node(source_track),
            )
        )
    source_node = route_oracle.order_end_node(source_track)
    detach_path = route_oracle.resolve_path_tracks_for_endpoint_constraints(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=None,
    ) or route_oracle.resolve_path_tracks(source_track, target_track)
    if detach_path is not None:
        blockers.extend(
            route_oracle._blocking_tracks_for_path(
                detach_path,
                occupied_track_sequences=state.track_sequences,
                source_node=source_node,
                target_node=None,
            )
        )
    return tuple(
        dict.fromkeys(
            track
            for track in blockers
            if track
            and track not in {source_track, target_track, state.loco_track_name}
            and track not in JI_BUFFER_TRACKS
        )
    )


def _phase1_clearance_candidate_blocking_tracks(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    route_oracle,
) -> tuple[str, ...]:
    if not source_track or not target_track or not vehicle_nos:
        return tuple()
    blockers: list[str] = []
    attach_path = route_oracle.resolve_path_tracks_for_endpoint_constraints(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    ) or route_oracle.resolve_path_tracks(state.loco_track_name, source_track)
    if attach_path is not None:
        blockers.extend(
            route_oracle._blocking_tracks_for_path(
                attach_path,
                occupied_track_sequences=state.track_sequences,
                source_node=state.loco_node,
                target_node=route_oracle.order_end_node(source_track),
            )
        )
    after_attach_sequences = dict(state.track_sequences)
    after_attach_sequences[source_track] = list(after_attach_sequences.get(source_track, []))[len(vehicle_nos):]
    detach_path = route_oracle.resolve_path_tracks_for_endpoint_constraints(
        source_track,
        target_track,
        occupied_track_sequences=after_attach_sequences,
        source_node=route_oracle.order_end_node(source_track),
        target_node=None,
    ) or route_oracle.resolve_path_tracks(source_track, target_track)
    if detach_path is not None:
        blockers.extend(
            route_oracle._blocking_tracks_for_path(
                detach_path,
                occupied_track_sequences=after_attach_sequences,
                source_node=route_oracle.order_end_node(source_track),
                target_node=None,
            )
        )
    return tuple(
        dict.fromkeys(
            track
            for track in blockers
            if track
            and track not in {source_track, target_track, state.loco_track_name}
            and track not in JI_BUFFER_TRACKS
        )
    )


def _phase1_clearance_target_tracks(source_track: str) -> tuple[str, ...]:
    preferred = (
        "存5南",
        "存5北",
        "存3",
        "存2",
        "存1",
        "调北",
        "预修",
        "调棚",
    )
    return tuple(track for track in preferred if track != source_track and track in PHASE1_TEMP_PARKING_TRACKS)


def _phase1_clearance_target_rank(track: str) -> int:
    order = {track: index for index, track in enumerate(_phase1_clearance_target_tracks(""))}
    return order.get(track, 99)


def _phase1_clearance_target_is_usable(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    state: ReplayState,
    route_oracle,
    vehicle_meta: dict[str, dict],
    master: MasterData,
) -> bool:
    if not source_track or not target_track or not vehicle_nos:
        return False
    if list(state.track_sequences.get(source_track, []))[: len(vehicle_nos)] != vehicle_nos:
        return False
    train_length_m = sum(
        float(dict(vehicle_meta.get(vehicle_no) or {}).get("vehicleLength") or 0.0)
        for vehicle_no in vehicle_nos
    )
    target_capacity_m = float(master.tracks[target_track].effective_length_m) if target_track in master.tracks else 0.0
    existing_length_m = sum(
        float(dict(vehicle_meta.get(vehicle_no) or {}).get("vehicleLength") or 0.0)
        for vehicle_no in list(state.track_sequences.get(target_track, []))
    )
    if target_capacity_m and existing_length_m + train_length_m > target_capacity_m + 1e-9:
        return False
    if not route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=state.track_sequences,
        carried_train_length_m=0.0,
        loco_node=state.loco_node,
    ).is_valid:
        return False
    attach_path = route_oracle.resolve_clear_path_tracks(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        return False
    after_attach_sequences = dict(state.track_sequences)
    after_attach_sequences[source_track] = list(after_attach_sequences.get(source_track, []))[len(vehicle_nos):]
    detach_path = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=after_attach_sequences,
        source_node=route_oracle.order_end_node(source_track),
        target_node=None,
    )
    return detach_path is not None


def _build_phase1_clearance_wave(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    goal_by_vehicle: dict[str, dict[str, Any]],
    reason_wave: dict[str, Any],
) -> dict[str, Any]:
    active_goals = []
    for vehicle_no in vehicle_nos:
        base_goal = dict(goal_by_vehicle.get(vehicle_no) or {})
        base_goal.update(
            {
                "vehicleNo": vehicle_no,
                "targetTrack": target_track,
                "targetMode": "AREA",
                "targetAreaCode": "STAGE::PHASE1_ROUTE_CLEARANCE",
                "targetSource": "PHASE1_ROUTE_CLEARANCE",
                "isSpotting": "",
            }
        )
        active_goals.append(base_goal)
    reason_diagnostics = dict(reason_wave.get("waveDiagnostics") or {})
    macro_task_id = str(reason_diagnostics.get("macroTaskId") or "")
    wave_diagnostics = {
        "waveName": "phase1_route_clearance",
        "waveRole": "phase1_rolling",
        "waveType": "phase1_rolling",
        "selectedSourceTrack": source_track,
        "selectedBlockIds": [f"ROUTE_CLEAR::{source_track}"],
        "selectedVehicleCount": len(vehicle_nos),
        "targetTrack": target_track,
        "runtimeFrontierStrategy": "rolling_route_clearance",
        "reasonWaveName": str(reason_wave.get("waveName") or ""),
        "reasonSourceTrack": str(reason_wave.get("selectedSourceTrack") or ""),
        "reasonTargetTrack": str(reason_diagnostics.get("targetTrack") or ""),
    }
    if macro_task_id:
        wave_diagnostics.update(
            {
                "macroTaskId": macro_task_id,
                "macroTaskBlockIds": list(reason_diagnostics.get("macroTaskBlockIds") or []),
                "macroTaskWaveChunks": list(reason_diagnostics.get("macroTaskWaveChunks") or []),
                "macroTaskSourceRole": str(reason_diagnostics.get("macroTaskSourceRole") or ""),
                "macroTaskScoreKey": list(reason_diagnostics.get("macroTaskScoreKey") or []),
                "routeClearanceForMacroTask": True,
            }
        )
    return {
        "waveName": f"phase1_clear_route_{source_track}_{target_track}",
        "waveRole": "phase1_rolling",
        "waveType": "phase1_rolling",
        "selectedSourceTrack": source_track,
        "selectedBlockIds": [f"ROUTE_CLEAR::{source_track}"],
        "selectedVehicleNos": list(vehicle_nos),
        "packageAssignments": {},
        "layoutAssignments": {},
        "packageTargetRanks": {},
        "layoutTargetRanks": {},
        "vehicleGoals": active_goals,
        "waveDiagnostics": wave_diagnostics,
    }


def _phase1_rolling_candidate_is_physically_frontier(
    *,
    candidate_wave: dict[str, Any],
    state: ReplayState,
    master: MasterData,
) -> bool:
    return _phase1_check_rolling_candidate_executable(
        candidate_wave=candidate_wave,
        state=state,
        master=master,
    ).ok


def _phase1_check_rolling_candidate_executable(
    *,
    candidate_wave: dict[str, Any],
    state: ReplayState,
    master: MasterData,
) -> Phase1ExecutableCheck:
    from fzed_shunting.domain.route_oracle import RouteOracle

    source_track = str(candidate_wave.get("selectedSourceTrack") or "")
    vehicle_nos = tuple(str(vehicle_no) for vehicle_no in list(candidate_wave.get("selectedVehicleNos") or []))
    target_track = str((candidate_wave.get("waveDiagnostics") or {}).get("targetTrack") or "")
    source_seq = tuple(str(vehicle_no) for vehicle_no in list(state.track_sequences.get(source_track, [])))
    source_prefix = source_seq[: max(len(vehicle_nos), 1) + 8]
    if not source_track or not target_track or not vehicle_nos:
        return Phase1ExecutableCheck(
            ok=False,
            reason="missing_source_target_or_vehicle_group",
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            source_prefix=source_prefix,
            loco_track=state.loco_track_name,
        )
    if source_seq[: len(vehicle_nos)] != vehicle_nos:
        return Phase1ExecutableCheck(
            ok=False,
            reason="source_prefix_mismatch",
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            source_prefix=source_prefix,
            loco_track=state.loco_track_name,
        )
    route_oracle = RouteOracle(master)
    attach_result = route_oracle.validate_loco_access(
        loco_track=state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=state.track_sequences,
        carried_train_length_m=0.0,
        loco_node=state.loco_node,
    )
    if not attach_result.is_valid:
        return Phase1ExecutableCheck(
            ok=False,
            reason="attach_path_blocked",
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            source_prefix=source_prefix,
            loco_track=state.loco_track_name,
            blocking_tracks=tuple(str(track) for track in list(getattr(attach_result, "blocking_tracks", []) or [])),
            errors=tuple(str(error) for error in list(getattr(attach_result, "errors", []) or [])),
        )
    attach_path = route_oracle.resolve_clear_path_tracks(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        return Phase1ExecutableCheck(
            ok=False,
            reason="attach_path_blocked",
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            source_prefix=source_prefix,
            loco_track=state.loco_track_name,
        )
    after_attach_sequences = dict(state.track_sequences)
    after_attach_sequences[source_track] = list(after_attach_sequences.get(source_track, []))[len(vehicle_nos):]
    source_node = route_oracle.order_end_node(source_track)
    detach_path = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=after_attach_sequences,
        source_node=source_node,
        target_node=None,
    )
    if detach_path is None:
        blocking_tracks = _phase1_clearance_candidate_blocking_tracks(
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=list(vehicle_nos),
            state=state,
            route_oracle=route_oracle,
        )
        return Phase1ExecutableCheck(
            ok=False,
            reason="detach_path_blocked",
            source_track=source_track,
            target_track=target_track,
            vehicle_nos=vehicle_nos,
            source_prefix=source_prefix,
            loco_track=state.loco_track_name,
            blocking_tracks=blocking_tracks,
        )
    return Phase1ExecutableCheck(
        ok=True,
        reason="ok",
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        source_prefix=source_prefix,
        loco_track=state.loco_track_name,
    )


def _build_phase1_rolling_plan_payload(
    *,
    candidate_wave: dict[str, Any],
    working_state: ReplayState,
    normalized,
    route_oracle,
    vehicle_by_no: dict[str, Any],
) -> list[dict[str, Any]]:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    source_track = str(candidate_wave.get("selectedSourceTrack") or "")
    vehicle_nos = [
        str(vehicle_no)
        for vehicle_no in list(candidate_wave.get("selectedVehicleNos") or [])
        if str(vehicle_no)
    ]
    target_track = str((candidate_wave.get("waveDiagnostics") or {}).get("targetTrack") or "")
    if not source_track or not target_track or not vehicle_nos:
        raise ValueError("phase1 rolling wave is incomplete")
    source_seq = list(working_state.track_sequences.get(source_track, []))
    if source_seq[: len(vehicle_nos)] != vehicle_nos:
        raise ValueError(
            f"phase1 rolling source prefix mismatch on {source_track}: "
            f"need {vehicle_nos}, got {source_seq[: len(vehicle_nos)]}"
        )
    carried_train_length_m = sum(
        float(vehicle_by_no[vehicle_no].vehicle_length)
        for vehicle_no in working_state.loco_carry
    )
    access_result = route_oracle.validate_loco_access(
        loco_track=working_state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=working_state.track_sequences,
        carried_train_length_m=carried_train_length_m,
        loco_node=working_state.loco_node,
    )
    if not access_result.is_valid:
        raise ValueError(
            f"phase1 rolling no clear attach path {working_state.loco_track_name} -> {source_track}: "
            + "; ".join(access_result.errors)
        )
    attach_path = route_oracle.resolve_clear_path_tracks(
        working_state.loco_track_name,
        source_track,
        occupied_track_sequences=working_state.track_sequences,
        source_node=working_state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        raise ValueError(f"phase1 rolling no clear attach path {working_state.loco_track_name} -> {source_track}")
    attach_move = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=list(attach_path),
        action_type="ATTACH",
    )
    attached_state = _apply_move(
        state=working_state,
        move=attach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    detach_path = route_oracle.resolve_clear_path_tracks(
        attached_state.loco_track_name,
        target_track,
        occupied_track_sequences=attached_state.track_sequences,
        source_node=attached_state.loco_node,
        target_node=None,
    )
    if detach_path is None:
        raise ValueError(f"phase1 rolling no clear detach path {attached_state.loco_track_name} -> {target_track}")
    return [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": list(vehicle_nos),
            "pathTracks": [source_track],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": attached_state.loco_track_name,
            "targetTrack": target_track,
            "vehicleNos": list(vehicle_nos),
            "pathTracks": list(detach_path),
        },
    ]


def _build_phase3_wave_plan_payload(
    *,
    wave_plan: dict[str, Any],
    working_state: ReplayState,
    normalized,
    route_oracle,
    vehicle_by_no: dict[str, Any],
) -> list[dict[str, Any]]:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    source_track = str(wave_plan.get("waveSourceTrack") or "")
    target_track = str(wave_plan.get("waveTargetTrack") or "")
    source_block = dict(wave_plan.get("sourceBlock") or {})
    target_runs = [
        {
            "targetTrack": str(item.get("targetTrack") or ""),
            "vehicleNos": [str(vehicle_no) for vehicle_no in item.get("vehicleNos") or ()],
        }
        for item in list(wave_plan.get("waveTargetRuns") or source_block.get("targetRuns") or [])
        if dict(item)
    ]
    vehicle_nos = [
        str(vehicle_no)
        for vehicle_no in list(source_block.get("vehicleNos") or [])
        if str(vehicle_no)
    ]
    if not vehicle_nos:
        vehicle_nos = [
            str(vehicle_no)
            for vehicle_no in dict(wave_plan.get("activeGoalsByVehicle") or {}).keys()
            if str(vehicle_no)
        ]
    if not source_track or not target_track or not vehicle_nos:
        raise ValueError("phase3 wave is incomplete")
    if not target_runs:
        target_runs = [{"targetTrack": target_track, "vehicleNos": list(vehicle_nos)}]
    target_runs = [
        {
            "targetTrack": str(item["targetTrack"]),
            "vehicleNos": [str(vehicle_no) for vehicle_no in item["vehicleNos"] if str(vehicle_no)],
        }
        for item in target_runs
        if str(item.get("targetTrack") or "") and list(item.get("vehicleNos") or [])
    ]
    if [vehicle_no for run in target_runs for vehicle_no in run["vehicleNos"]] != vehicle_nos:
        raise ValueError("phase3 wave target runs do not match source prefix")
    source_seq = list(working_state.track_sequences.get(source_track, []))
    if source_seq[: len(vehicle_nos)] != vehicle_nos:
        raise ValueError(
            f"phase3 wave source prefix mismatch on {source_track}: "
            f"need {vehicle_nos}, got {source_seq[: len(vehicle_nos)]}"
        )
    carried_train_length_m = sum(
        float(vehicle_by_no[vehicle_no].vehicle_length)
        for vehicle_no in working_state.loco_carry
    )
    access_result = route_oracle.validate_loco_access(
        loco_track=working_state.loco_track_name,
        target_track=source_track,
        occupied_track_sequences=working_state.track_sequences,
        carried_train_length_m=carried_train_length_m,
        loco_node=working_state.loco_node,
    )
    if not access_result.is_valid:
        raise ValueError(
            f"phase3 wave no clear attach path {working_state.loco_track_name} -> {source_track}: "
            + "; ".join(access_result.errors)
        )
    attach_path = route_oracle.resolve_clear_path_tracks(
        working_state.loco_track_name,
        source_track,
        occupied_track_sequences=working_state.track_sequences,
        source_node=working_state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        raise ValueError(f"phase3 wave no clear attach path {working_state.loco_track_name} -> {source_track}")
    attach_move = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=list(attach_path),
        action_type="ATTACH",
    )
    attached_state = _apply_move(
        state=working_state,
        move=attach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    plan_payload = [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": list(vehicle_nos),
            "pathTracks": [source_track],
        },
    ]
    hook_no = 2
    for run in reversed(target_runs):
        run_target_track = str(run["targetTrack"])
        run_vehicle_nos = [str(vehicle_no) for vehicle_no in run["vehicleNos"]]
        if not is_carried_tail_block(attached_state.loco_carry, run_vehicle_nos):
            raise ValueError(
                "phase3 wave target run is not carried tail block: "
                f"{run_vehicle_nos} from carry {list(attached_state.loco_carry)}"
            )
        detach_path = route_oracle.resolve_clear_path_tracks(
            attached_state.loco_track_name,
            run_target_track,
            occupied_track_sequences=attached_state.track_sequences,
            source_node=attached_state.loco_node,
            target_node=route_oracle.order_end_node(run_target_track),
        )
        if detach_path is None:
            raise ValueError(
                f"phase3 wave no clear detach path {attached_state.loco_track_name} -> {run_target_track}"
            )
        detach_move = HookAction(
            source_track=attached_state.loco_track_name,
            target_track=run_target_track,
            vehicle_nos=list(run_vehicle_nos),
            path_tracks=list(detach_path),
            action_type="DETACH",
        )
        plan_payload.append(
            {
                "hookNo": hook_no,
                "actionType": "DETACH",
                "sourceTrack": attached_state.loco_track_name,
                "targetTrack": run_target_track,
                "vehicleNos": list(run_vehicle_nos),
                "pathTracks": list(detach_path),
            }
        )
        hook_no += 1
        attached_state = _apply_move(
            state=attached_state,
            move=detach_move,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    return plan_payload


def _build_preflighted_phase3_wave_plans(
    *,
    stage: dict[str, Any],
    current_vehicle_info: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict[str, Any]],
    current_state: ReplayState,
    master: MasterData,
    source_wave_plans: list[dict[str, Any]],
    all_active_covered: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from fzed_shunting.domain.route_oracle import RouteOracle

    if not all_active_covered:
        return [], {
            "enabled": False,
            "reason": "active_vehicle_hidden_behind_hold",
        }
    if not source_wave_plans:
        return [], {
            "enabled": False,
            "reason": "no_source_wave_plans",
        }

    candidate_wave_plans = _build_phase3_tail_run_candidates(source_wave_plans)
    route_oracle = RouteOracle(master)
    working_state = ReplayState.model_validate(current_state.model_dump())
    working_vehicle_info = [dict(item) for item in current_vehicle_info]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for wave_plan in candidate_wave_plans:
        sub_stage = _build_phase3_wave_sub_stage(
            stage=stage,
            wave_plan=wave_plan,
            working_vehicle_info=working_vehicle_info,
        )
        stage_payload = _build_stage_payload(
            track_info=[],
            current_vehicle_info=working_vehicle_info,
            vehicle_meta=current_by_vehicle,
            stage=sub_stage,
            loco_track_name=working_state.loco_track_name,
        )
        normalized = normalize_plan_input(
            stage_payload,
            master,
            allow_internal_loco_tracks=True,
        )
        vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
        try:
            plan_payload = _build_phase3_wave_plan_payload(
                wave_plan=wave_plan,
                working_state=working_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
            )
            working_state = replay_plan(
                initial_state=working_state,
                hook_plan=plan_payload,
                plan_input=normalized,
            ).final_state
            working_vehicle_info = _vehicle_info_from_replay_state(
                state=working_state,
                previous_vehicle_info=working_vehicle_info,
            )
        except ValueError as exc:
            rejected.append(
                {
                    "waveName": str(wave_plan.get("waveName") or ""),
                    "sourceTrack": str(wave_plan.get("waveSourceTrack") or ""),
                    "targetRuns": list(wave_plan.get("waveTargetRuns") or []),
                    "reason": str(exc),
                }
            )
            return [], {
                "enabled": False,
                "reason": "preflight_failed",
                "candidateWaveCount": len(candidate_wave_plans),
                "acceptedWaveCount": len(accepted),
                "rejectedWaves": rejected,
            }
        accepted.append({**dict(wave_plan), "requiresExplicitPlan": True})

    return accepted, {
        "enabled": True,
        "reason": "preflight_passed",
        "candidateWaveCount": len(candidate_wave_plans),
        "acceptedWaveCount": len(accepted),
        "rejectedWaves": [],
        "plannedHookCount": sum(
            1 + len(list(wave.get("waveTargetRuns") or []))
            for wave in accepted
        ),
    }


def _build_phase3_tail_run_candidates(
    source_wave_plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    source_order: list[str] = []
    for wave_plan in source_wave_plans:
        source_track = str(wave_plan.get("waveSourceTrack") or "")
        if not source_track:
            continue
        if source_track not in by_source:
            source_order.append(source_track)
            by_source[source_track] = []
        by_source[source_track].append(dict(wave_plan))

    candidates: list[dict[str, Any]] = []
    for source_track in source_order:
        source_waves = by_source[source_track]
        if len(source_waves) == 1:
            wave = dict(source_waves[0])
            block = dict(wave.get("sourceBlock") or {})
            wave["waveTargetRuns"] = [
                {
                    "targetTrack": str(wave.get("waveTargetTrack") or block.get("targetTrack") or ""),
                    "vehicleNos": [
                        str(vehicle_no)
                        for vehicle_no in list(block.get("vehicleNos") or [])
                    ],
                }
            ]
            candidates.append(wave)
            continue

        target_runs: list[dict[str, Any]] = []
        vehicle_nos: list[str] = []
        active_goals: dict[str, dict[str, Any]] = {}
        block_ids: list[str] = []
        total_weight = 0.0
        for wave in source_waves:
            block = dict(wave.get("sourceBlock") or {})
            run_vehicle_nos = [
                str(vehicle_no)
                for vehicle_no in list(block.get("vehicleNos") or [])
                if str(vehicle_no)
            ]
            target_track = str(wave.get("waveTargetTrack") or block.get("targetTrack") or "")
            if not run_vehicle_nos or not target_track:
                continue
            target_runs.append(
                {
                    "targetTrack": target_track,
                    "vehicleNos": run_vehicle_nos,
                }
            )
            vehicle_nos.extend(run_vehicle_nos)
            active_goals.update(
                {
                    str(vehicle_no): dict(goal)
                    for vehicle_no, goal in dict(wave.get("activeGoalsByVehicle") or {}).items()
                }
            )
            block_ids.append(str(block.get("blockId") or ""))
            total_weight += float(wave.get("waveWeight") or 0.0)
        if not target_runs or not vehicle_nos:
            continue
        first_target = str(target_runs[0]["targetTrack"])
        candidates.append(
            {
                "waveName": f"phase3_tail_run_{len(candidates) + 1:02d}_{source_track}",
                "waveRole": "PHASE3_SOURCE_TAIL_RUN_TO_DEPOT",
                "waveSourceTrack": source_track,
                "waveTargetTrack": first_target,
                "waveWeight": total_weight,
                "sourceBlock": {
                    "blockId": "PHASE3_TAIL_RUN::" + source_track,
                    "sourceTrack": source_track,
                    "targetTrack": first_target,
                    "vehicleNos": vehicle_nos,
                    "vehicleCount": len(vehicle_nos),
                    "targetRuns": target_runs,
                    "sourceBlockIds": [block_id for block_id in block_ids if block_id],
                },
                "waveTargetRuns": target_runs,
                "activeGoalsByVehicle": active_goals,
            }
        )
    total_weight = sum(float(item.get("waveWeight") or 0.0) for item in candidates)
    if total_weight > 0:
        for item in candidates:
            item["waveWeight"] = float(item["waveWeight"]) / total_weight
    return candidates


def _build_phase3_wave_sub_stage(
    *,
    stage: dict[str, Any],
    wave_plan: dict[str, Any],
    working_vehicle_info: list[dict[str, Any]],
) -> dict[str, Any]:
    active_goals = {
        str(vehicle_no): dict(goal)
        for vehicle_no, goal in dict(wave_plan.get("activeGoalsByVehicle") or {}).items()
    }
    sub_vehicle_goals: list[dict[str, Any]] = []
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
                "targetMode": "SNAPSHOT",
                "targetSource": "PHASE3_DYNAMIC_HOLD",
                "isSpotting": "",
            }
        )
    return {
        "name": str(stage.get("name", "phase3")),
        "description": str(stage.get("description", "")),
        "routePolicy": dict(stage.get("routePolicy") or {}),
        "stagePolicy": dict(stage.get("stagePolicy") or {}),
        "vehicleGoals": sub_vehicle_goals,
    }


def _vehicle_info_from_replay_state(
    *,
    state: ReplayState,
    previous_vehicle_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_by_vehicle = {
        str(item["vehicleNo"]): dict(item)
        for item in previous_vehicle_info
    }
    next_vehicle_info: list[dict[str, Any]] = []
    for track_name in sorted(state.track_sequences):
        for index, vehicle_no in enumerate(state.track_sequences[track_name], start=1):
            base = dict(previous_by_vehicle[str(vehicle_no)])
            base["trackName"] = track_name
            base["order"] = str(index)
            next_vehicle_info.append(base)
    return next_vehicle_info


def _build_phase1_runtime_frontier_wave(
    *,
    runtime_stage: dict[str, Any],
) -> dict[str, Any] | None:
    stage_policy = dict(runtime_stage.get("stagePolicy") or {})
    diagnostics = dict(stage_policy.get("phase1Diagnostics") or {})
    phase1_blocks = {
        str(block.get("blockId") or ""): dict(block)
        for block in list(diagnostics.get("phase1Blocks") or [])
        if str(block.get("blockId") or "")
    }
    if not phase1_blocks:
        return _build_phase1_static_frontier_wave(stage_policy)
    selected_block_ids = _phase1_runtime_selected_block_ids(runtime_stage)
    if not selected_block_ids:
        return None
    goal_by_vehicle = {
        str(goal.get("vehicleNo") or ""): dict(goal)
        for goal in list(runtime_stage.get("vehicleGoals") or [])
        if str(goal.get("vehicleNo") or "")
    }
    raw_source_plans = [
        dict(item)
        for item in list(diagnostics.get("sourceTrackPlans") or [])
        if dict(item)
    ]
    if not raw_source_plans:
        return _build_phase1_static_frontier_wave(stage_policy, phase1_blocks=phase1_blocks)
    source_plans = sorted(
        raw_source_plans,
        key=lambda item: _phase1_runtime_source_frontier_score(
            source_plan=item,
            phase1_blocks=phase1_blocks,
            selected_block_ids=selected_block_ids,
        ),
    )
    chosen_blocks: list[dict[str, Any]] = []
    chosen_block_ids: list[str] = []
    chosen_vehicle_nos: list[str] = []
    chosen_source_tracks: list[str] = []
    for source_plan in source_plans:
        source_track = str(source_plan.get("sourceTrack") or "")
        frontier_blocks = _phase1_runtime_source_frontier_blocks(
            ordered_block_ids=[
                str(block_id)
                for block_id in list(source_plan.get("blockIds") or [])
                if str(block_id) in selected_block_ids
            ],
            phase1_blocks=phase1_blocks,
            selected_block_ids=selected_block_ids,
        )
        if not frontier_blocks:
            continue
        frontier_vehicle_count = sum(len(list(block.get("vehicleNos") or [])) for block in frontier_blocks)
        if frontier_vehicle_count > PHASE1_RUNTIME_WAVE_MAX_ACTIVE_VEHICLES:
            continue
        chosen_source_tracks.append(source_track)
        chosen_blocks.extend(frontier_blocks)
        for block in frontier_blocks:
            block_id = str(block.get("blockId") or "")
            if block_id:
                chosen_block_ids.append(block_id)
            for vehicle_no in list(block.get("vehicleNos") or []):
                vehicle_no = str(vehicle_no)
                if vehicle_no:
                    chosen_vehicle_nos.append(vehicle_no)
        break
    if not chosen_blocks or not chosen_vehicle_nos:
        return None
    chosen_vehicle_nos = list(dict.fromkeys(chosen_vehicle_nos))
    active_goals = [
        goal_by_vehicle[vehicle_no]
        for vehicle_no in chosen_vehicle_nos
        if vehicle_no in goal_by_vehicle
    ]
    wave_buffer_assignment = {
        vehicle_no: str(goal_by_vehicle[vehicle_no].get("targetTrack") or "")
        for vehicle_no in chosen_vehicle_nos
        if str(goal_by_vehicle[vehicle_no].get("targetTrack") or "") in JI_BUFFER_TRACKS
    }
    return {
        "waveName": f"runtime_frontier_{'_'.join(chosen_source_tracks[:3])}",
        "waveRole": "runtime_frontier",
        "waveType": "runtime_frontier",
        "selectedSourceTrack": chosen_source_tracks[0] if chosen_source_tracks else "",
        "selectedBlockIds": list(dict.fromkeys(chosen_block_ids)),
        "selectedVehicleNos": list(chosen_vehicle_nos),
        "packageAssignments": dict(wave_buffer_assignment),
        "layoutAssignments": dict(wave_buffer_assignment),
        "packageTargetRanks": {},
        "layoutTargetRanks": {},
        "vehicleGoals": active_goals,
        "waveDiagnostics": {
            "waveName": "runtime_frontier",
            "waveRole": "runtime_frontier",
            "waveType": "runtime_frontier",
            "selectedSourceTrack": chosen_source_tracks[0] if chosen_source_tracks else "",
            "selectedBlockIds": list(dict.fromkeys(chosen_block_ids)),
            "selectedSourceTracks": list(chosen_source_tracks),
            "selectedVehicleCount": len(chosen_vehicle_nos),
            "selectedBufferVehicleCount": len(wave_buffer_assignment),
            "runtimeFrontierStrategy": "single_source_dynamic",
        },
    }


def _phase1_runtime_source_frontier_blocks(
    *,
    ordered_block_ids: list[str],
    phase1_blocks: dict[str, dict[str, Any]],
    selected_block_ids: frozenset[str],
) -> list[dict[str, Any]]:
    for block_id in ordered_block_ids:
        block = phase1_blocks.get(block_id)
        if block is None:
            continue
        unfinished_predecessors = [
            str(predecessor_id)
            for predecessor_id in list(block.get("requiredPredecessorIds") or [])
            if str(predecessor_id) in selected_block_ids
        ]
        if unfinished_predecessors:
            continue
        return [block]
    return []


def _phase1_runtime_source_frontier_score(
    *,
    source_plan: dict[str, Any],
    phase1_blocks: dict[str, dict[str, Any]],
    selected_block_ids: frozenset[str],
) -> tuple[Any, ...]:
    source_track = str(source_plan.get("sourceTrack") or "")
    ordered_block_ids = [
        str(block_id)
        for block_id in list(source_plan.get("blockIds") or [])
        if str(block_id) in selected_block_ids
    ]
    frontier_blocks = _phase1_runtime_source_frontier_blocks(
        ordered_block_ids=ordered_block_ids,
        phase1_blocks=phase1_blocks,
        selected_block_ids=selected_block_ids,
    )
    if not frontier_blocks:
        return (99, source_track)
    block = frontier_blocks[0]
    block_type = str(block.get("blockType") or "")
    vehicle_count = len(list(block.get("vehicleNos") or []))
    released_depot_count = int(block.get("releasedDepotVehicleCount") or 0)
    selected_total = len(ordered_block_ids)
    successor_count = max(0, selected_total - 1)
    is_unlocking_clear = block_type in {"clear_cun4", "prefix_clear"}
    is_storage_unlock = source_track == "存5北" and is_unlocking_clear
    is_direct_depot = bool(block.get("usesBuffer"))
    source_priority = {
        "存5北": 0,
        "存5南": 1,
        "存3": 2,
        "存2": 3,
        "存1": 4,
        "调棚": 5,
        "预修": 5,
        "抛": 6,
        "洗南": 7,
        "洗北": 7,
        "油": 7,
        "机南": 8,
        "机棚": 8,
        "机北1": 8,
        "机北2": 8,
        "机北3": 8,
    }.get(source_track, 9)
    return (
        0 if is_unlocking_clear else 1,
        0 if is_storage_unlock else 1,
        -successor_count,
        -released_depot_count,
        0 if is_direct_depot else 1,
        vehicle_count,
        source_priority,
        tuple(source_plan.get("sourcePriorityScore") or ()),
        block_type,
        source_track,
    )


def _build_phase1_static_frontier_wave(
    stage_policy: dict[str, Any],
    *,
    phase1_blocks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    static_waves = [
        dict(item)
        for item in list(stage_policy.get("phase1WavePlans") or [])
        if dict(item)
    ]
    clearance_waves = [
        wave for wave in static_waves
        if str(wave.get("waveType") or wave.get("waveRole") or "") == "source_clearance"
    ]
    ordered_waves = sorted(
        clearance_waves or static_waves,
        key=_phase1_runtime_static_wave_order,
    )
    shallow_storage_waves: list[dict[str, Any]] = []
    deferred_storage_waves: list[dict[str, Any]] = []
    fallback_waves: list[dict[str, Any]] = []
    for wave in ordered_waves:
        diagnostics = dict(wave.get("waveDiagnostics") or {})
        wave_type = str(wave.get("waveType") or wave.get("waveRole") or "")
        source_role = str(diagnostics.get("selectedSourceRole") or "")
        predecessor_depth = int(diagnostics.get("maxRequiredPredecessorDepth") or 0)
        if wave_type == "source_clearance" and source_role in {"receiving_storage", "yard_storage"}:
            if predecessor_depth <= 1:
                shallow_storage_waves.append(wave)
            else:
                deferred_storage_waves.append(wave)
            continue
        fallback_waves.append(wave)
    for wave in shallow_storage_waves + fallback_waves + deferred_storage_waves:
        wave_block_ids = [
            str(block_id)
            for block_id in list(wave.get("selectedBlockIds") or [])
            if str(block_id)
        ]
        required_predecessor_ids = [
            str(block_id)
            for block_id in list(wave.get("requiredPredecessorIds") or [])
            if str(block_id)
        ]
        if phase1_blocks is not None and any(
            predecessor_id in phase1_blocks
            and (
                bool(phase1_blocks[predecessor_id].get("selectedBackbone"))
                or bool(phase1_blocks[predecessor_id].get("selectedFinish"))
            )
            for predecessor_id in required_predecessor_ids
        ):
            continue
        vehicle_nos = [
            str(vehicle_no)
            for vehicle_no in list(wave.get("selectedVehicleNos") or [])
            if str(vehicle_no)
        ]
        if not vehicle_nos:
            continue
        wave_copy = dict(wave)
        wave_copy["selectedVehicleNos"] = list(dict.fromkeys(vehicle_nos))
        diagnostics = dict(wave_copy.get("waveDiagnostics") or {})
        diagnostics.setdefault("runtimeSelected", True)
        diagnostics.setdefault("runtimeFrontierStrategy", "static_fallback")
        diagnostics.setdefault("selectedBlockIds", list(dict.fromkeys(wave_block_ids)))
        diagnostics.setdefault("requiredPredecessorIds", list(dict.fromkeys(required_predecessor_ids)))
        wave_copy["waveDiagnostics"] = diagnostics
        return wave_copy
    return None


def _phase1_runtime_static_wave_order(wave: dict[str, Any]) -> tuple[Any, ...]:
    diagnostics = dict(wave.get("waveDiagnostics") or {})
    pressure_cut_counts = dict(diagnostics.get("pressureCutCounts") or {})
    wave_type = str(wave.get("waveType") or wave.get("waveRole") or "")
    selected_source_track = str(wave.get("selectedSourceTrack") or "")
    selected_source_role = str(diagnostics.get("selectedSourceRole") or "")
    selected_vehicle_count = len(list(wave.get("selectedVehicleNos") or ()))
    predecessor_depth = int(diagnostics.get("maxRequiredPredecessorDepth") or 0)
    released_depot_vehicle_count = int(diagnostics.get("releasedDepotVehicleCount") or 0)
    pressure_gain = int(diagnostics.get("pressureGain") or 0)
    is_storage_clearance = wave_type == "source_clearance" and selected_source_role in {
        "receiving_storage",
        "yard_storage",
    }
    is_hot_clearance = wave_type == "source_clearance" and selected_source_role in {
        "wash_gate",
        "work_gate",
        "work_support",
    }
    is_marshalling = wave_type == "source_marshalling"
    return (
        0 if is_hot_clearance else 1 if is_storage_clearance else 2 if is_marshalling else 3,
        predecessor_depth,
        selected_vehicle_count,
        0 if released_depot_vehicle_count > 0 else 1,
        -released_depot_vehicle_count,
        -pressure_gain,
        -int(pressure_cut_counts.get("opening_release_to_ji") or 0),
        -int(pressure_cut_counts.get("work_to_ji") or 0),
        -int(pressure_cut_counts.get("wash_to_ji") or 0),
        selected_source_track,
        str(wave.get("waveName") or ""),
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
    working_state = current_state
    plan_payload: list[dict[str, Any]] = []
    hook_no = 1
    reorder_plan_payload = _build_phase2_reorder_plan_payload(
        normalized=normalized,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        working_state=working_state,
        active_goals={vehicle_no: {} for vehicle_no in transfer_vehicle_nos},
        outbound_vehicle_nos=set(must_pull_vehicle_nos),
        capacity_by_track=capacity_by_track,
    )
    if reorder_plan_payload:
        for raw_hook in reorder_plan_payload:
            hook = dict(raw_hook)
            hook["hookNo"] = hook_no
            plan_payload.append(hook)
            move = HookAction(
                source_track=str(hook["sourceTrack"]),
                target_track=str(hook["targetTrack"]),
                vehicle_nos=[str(vehicle_no) for vehicle_no in hook.get("vehicleNos") or ()],
                path_tracks=[str(track) for track in hook.get("pathTracks") or ()],
                action_type=str(hook["actionType"]),
            )
            working_state = _apply_move(
                state=working_state,
                move=move,
                plan_input=normalized,
                vehicle_by_no=vehicle_by_no,
            )
            hook_no += 1
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
    def layer_runtime_order_key(layer: dict[str, Any]) -> tuple[int, int, str]:
        source_seq = list(working_state.track_sequences.get(str(layer["sourceTrack"]), ()))
        positions = [
            source_seq.index(vehicle_no)
            for vehicle_no in layer["vehicleNos"]
            if vehicle_no in source_seq
        ]
        return (
            min(positions) if positions else 9999,
            int(layer["layerIndex"]),
            str(layer["sourceTrack"]),
        )

    pending_by_track = {
        track: sorted(layers, key=layer_runtime_order_key)
        for track, layers in layer_by_track.items()
    }
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
                layer_has_outbound = any(vehicle_no in must_pull_vehicle_nos for vehicle_no in vehicle_nos)
                layer_has_predecessor_unlock = any(
                    vehicle_no in predecessor_unlock_vehicle_nos for vehicle_no in vehicle_nos
                )
                layer_has_phase3_clearance = any(
                    vehicle_no in phase3_clearance_vehicle_nos for vehicle_no in vehicle_nos
                )
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
                    0
                    if layer_has_outbound
                    else 1
                    if layer_has_predecessor_unlock
                    else 2
                    if layer_has_phase3_clearance
                    else 3,
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


def _build_phase2_reorder_plan_payload(
    *,
    normalized: Any,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    working_state: ReplayState,
    active_goals: dict[str, dict[str, Any]],
    outbound_vehicle_nos: set[str],
    capacity_by_track: dict[str, float],
) -> list[dict[str, Any]] | None:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    active_vehicle_nos = set(active_goals)
    track_to_active: dict[str, list[str]] = {}
    for vehicle_no in active_vehicle_nos:
        current_track = _locate_vehicle_track(working_state, vehicle_no)
        track_to_active.setdefault(current_track, []).append(vehicle_no)

    reorder_tasks = _phase2_build_reorder_tasks(
        track_to_active=track_to_active,
        state=working_state,
        outbound_vehicle_nos=outbound_vehicle_nos,
    )
    if reorder_tasks is None:
        return None
    buffer_layout = _phase2_select_reorder_buffer_layout(
        tasks=reorder_tasks,
        normalized=normalized,
        state=working_state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        capacity_by_track=capacity_by_track,
    )
    if buffer_layout is None:
        return None

    hook_no = 1
    plan_payload: list[dict[str, Any]] = []
    state = working_state
    for task in reorder_tasks:
        source_track = task.source_track
        staging_track = buffer_layout.get(source_track)
        if staging_track is None:
            return None

        state, hook_no = _phase2_append_attach_plan(
            plan_payload=plan_payload,
            hook_no=hook_no,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=source_track,
            vehicle_nos=list(task.current_prefix),
        )
        state, hook_no = _phase2_append_detach_plan(
            plan_payload=plan_payload,
            hook_no=hook_no,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=staging_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
        state, hook_no = _phase2_append_detach_plan(
            plan_payload=plan_payload,
            hook_no=hook_no,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=source_track,
            vehicle_nos=list(task.cun4_vehicle_nos),
        )
    for task in reorder_tasks:
        source_track = task.source_track
        staging_track = buffer_layout.get(source_track)
        if staging_track is None:
            return None
        state, hook_no = _phase2_append_attach_plan(
            plan_payload=plan_payload,
            hook_no=hook_no,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=staging_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
        state, hook_no = _phase2_append_detach_plan(
            plan_payload=plan_payload,
            hook_no=hook_no,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=source_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
    return plan_payload


def _phase2_build_reorder_tasks(
    *,
    track_to_active: dict[str, list[str]],
    state: ReplayState,
    outbound_vehicle_nos: set[str],
) -> tuple[Phase2ReorderTask, ...] | None:
    tasks: list[Phase2ReorderTask] = []
    for source_track in sorted(track_to_active):
        source_seq = list(state.track_sequences.get(source_track, ()))
        current_prefix = source_seq[: len(track_to_active[source_track])]
        if set(current_prefix) != set(track_to_active[source_track]):
            return None
        desired_prefix = _phase2_desired_prefix_order(
            current_prefix=current_prefix,
            outbound_vehicle_nos=outbound_vehicle_nos,
        )
        if current_prefix == desired_prefix:
            continue
        outbound_prefix_vehicle_nos = tuple(
            vehicle_no
            for vehicle_no in desired_prefix
            if vehicle_no in outbound_vehicle_nos
        )
        cun4_vehicle_nos = tuple(
            vehicle_no
            for vehicle_no in desired_prefix
            if vehicle_no not in outbound_vehicle_nos
        )
        if not outbound_prefix_vehicle_nos or not cun4_vehicle_nos:
            return None
        if current_prefix != [*cun4_vehicle_nos, *outbound_prefix_vehicle_nos]:
            return None
        tasks.append(
            Phase2ReorderTask(
                source_track=source_track,
                current_prefix=tuple(current_prefix),
                desired_prefix=tuple(desired_prefix),
                outbound_vehicle_nos=outbound_prefix_vehicle_nos,
                cun4_vehicle_nos=cun4_vehicle_nos,
            )
        )
    return tuple(tasks)


def _phase2_select_reorder_buffer_layout(
    *,
    tasks: tuple[Phase2ReorderTask, ...],
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
) -> dict[str, str] | None:
    candidates_by_task = {
        task.source_track: _phase2_reorder_buffer_candidates(
            task=task,
            normalized=normalized,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
        )
        for task in tasks
    }
    if any(not candidates for candidates in candidates_by_task.values()):
        return None
    best = _phase2_search_reorder_buffer_layout(
        tasks=tasks,
        candidates_by_task=candidates_by_task,
        normalized=normalized,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
    )
    if best is None:
        return None
    layout = {task.source_track: target_track for task, target_track in best}
    return layout


def _phase2_search_reorder_buffer_layout(
    *,
    tasks: tuple[Phase2ReorderTask, ...],
    candidates_by_task: dict[str, list[Phase2ReorderBufferCandidate]],
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
) -> tuple[tuple[Phase2ReorderTask, str], ...] | None:
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (len(candidates_by_task.get(task.source_track, ())), task.source_track),
    )
    complete_layouts: list[tuple[tuple[Any, ...], tuple[tuple[Phase2ReorderTask, str], ...]]] = []

    def dfs(index: int, used_tracks: set[str], selected: list[tuple[Phase2ReorderTask, str, tuple[Any, ...]]]) -> None:
        if index >= len(ordered_tasks):
            score = tuple(item[2] for item in selected)
            layout = tuple((item[0], item[1]) for item in selected)
            complete_layouts.append((score, layout))
            return
        task = ordered_tasks[index]
        for candidate in candidates_by_task[task.source_track]:
            if candidate.target_track in used_tracks:
                continue
            selected.append((task, candidate.target_track, candidate.score))
            used_tracks.add(candidate.target_track)
            dfs(index + 1, used_tracks, selected)
            used_tracks.remove(candidate.target_track)
            selected.pop()

    dfs(0, set(), [])
    for _score, layout_items in sorted(complete_layouts, key=lambda item: item[0]):
        layout = {task.source_track: target_track for task, target_track in layout_items}
        if _phase2_reorder_layout_is_executable(
            tasks=tasks,
            layout=layout,
            normalized=normalized,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        ):
            return layout_items
    return None


def _phase2_reorder_layout_is_executable(
    *,
    tasks: tuple[Phase2ReorderTask, ...],
    layout: dict[str, str],
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
) -> bool:
    try:
        scratch_plan: list[dict[str, Any]] = []
        scratch_state = state
        hook_no = 1
        for task in tasks:
            target_track = layout.get(task.source_track)
            if target_track is None:
                return False
            scratch_state, hook_no = _phase2_append_attach_plan(
                plan_payload=scratch_plan,
                hook_no=hook_no,
                state=scratch_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=task.source_track,
                vehicle_nos=list(task.current_prefix),
            )
            scratch_state, hook_no = _phase2_append_detach_plan(
                plan_payload=scratch_plan,
                hook_no=hook_no,
                state=scratch_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                target_track=target_track,
                vehicle_nos=list(task.outbound_vehicle_nos),
            )
            scratch_state, hook_no = _phase2_append_detach_plan(
                plan_payload=scratch_plan,
                hook_no=hook_no,
                state=scratch_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                target_track=task.source_track,
                vehicle_nos=list(task.cun4_vehicle_nos),
            )
        for task in tasks:
            target_track = layout.get(task.source_track)
            if target_track is None:
                return False
            scratch_state, hook_no = _phase2_append_attach_plan(
                plan_payload=scratch_plan,
                hook_no=hook_no,
                state=scratch_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                source_track=target_track,
                vehicle_nos=list(task.outbound_vehicle_nos),
            )
            scratch_state, hook_no = _phase2_append_detach_plan(
                plan_payload=scratch_plan,
                hook_no=hook_no,
                state=scratch_state,
                normalized=normalized,
                route_oracle=route_oracle,
                vehicle_by_no=vehicle_by_no,
                target_track=task.source_track,
                vehicle_nos=list(task.outbound_vehicle_nos),
            )
        return all(
            list(scratch_state.track_sequences.get(task.source_track, ()))[: len(task.desired_prefix)] == list(task.desired_prefix)
            for task in tasks
        )
    except Exception:
        return False


def _phase2_select_single_reorder_buffer(
    *,
    task: Phase2ReorderTask,
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
) -> str | None:
    candidates = _phase2_reorder_buffer_candidates(
        task=task,
        normalized=normalized,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        capacity_by_track=capacity_by_track,
    )
    if not candidates:
            return None
    return candidates[0].target_track


def _phase2_reorder_buffer_candidates(
    *,
    task: Phase2ReorderTask,
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
) -> list[Phase2ReorderBufferCandidate]:
    candidates: list[Phase2ReorderBufferCandidate] = []
    for target_track in PHASE2_REORDER_BUFFER_TRACKS:
        if target_track == task.source_track:
            continue
        if state.track_sequences.get(target_track):
            continue
        if not _phase2_track_has_capacity(
            target_track=target_track,
            adding_vehicle_nos=task.outbound_vehicle_nos,
            working_state=state,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
        ):
            continue
        detach_path = route_oracle.resolve_clear_path_tracks(
            task.source_track,
            target_track,
            occupied_track_sequences=state.track_sequences,
            source_node=route_oracle.order_end_node(task.source_track),
            target_node=route_oracle.order_end_node(target_track),
        )
        restore_path = route_oracle.resolve_clear_path_tracks(
            target_track,
            task.source_track,
            occupied_track_sequences=state.track_sequences,
            source_node=route_oracle.order_end_node(target_track),
            target_node=route_oracle.order_end_node(task.source_track),
        )
        if detach_path is None or restore_path is None:
            continue
        if not _phase2_reorder_buffer_is_executable(
            task=task,
            target_track=target_track,
            normalized=normalized,
            state=state,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
        ):
            continue
        candidates.append(
            Phase2ReorderBufferCandidate(
                task=task,
                target_track=target_track,
                score=_phase2_reorder_buffer_score(
                    task=task,
                    target_track=target_track,
                    state=state,
                    route_oracle=route_oracle,
                    vehicle_by_no=vehicle_by_no,
                    detach_path=detach_path,
                    restore_path=restore_path,
                    capacity_by_track=capacity_by_track,
                ),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score)


def _phase2_reorder_buffer_is_executable(
    *,
    task: Phase2ReorderTask,
    target_track: str,
    normalized: Any,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
) -> bool:
    try:
        scratch_plan: list[dict[str, Any]] = []
        scratch_state, hook_no = _phase2_append_attach_plan(
            plan_payload=scratch_plan,
            hook_no=1,
            state=state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=task.source_track,
            vehicle_nos=list(task.current_prefix),
        )
        scratch_state, hook_no = _phase2_append_detach_plan(
            plan_payload=scratch_plan,
            hook_no=hook_no,
            state=scratch_state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=target_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
        scratch_state, hook_no = _phase2_append_detach_plan(
            plan_payload=scratch_plan,
            hook_no=hook_no,
            state=scratch_state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=task.source_track,
            vehicle_nos=list(task.cun4_vehicle_nos),
        )
        scratch_state, hook_no = _phase2_append_attach_plan(
            plan_payload=scratch_plan,
            hook_no=hook_no,
            state=scratch_state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            source_track=target_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
        scratch_state, _hook_no = _phase2_append_detach_plan(
            plan_payload=scratch_plan,
            hook_no=hook_no,
            state=scratch_state,
            normalized=normalized,
            route_oracle=route_oracle,
            vehicle_by_no=vehicle_by_no,
            target_track=task.source_track,
            vehicle_nos=list(task.outbound_vehicle_nos),
        )
        return list(scratch_state.track_sequences.get(task.source_track, ()))[: len(task.desired_prefix)] == list(task.desired_prefix)
    except Exception:
        return False


def _phase2_reorder_buffer_score(
    *,
    task: Phase2ReorderTask,
    target_track: str,
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    detach_path: list[str],
    restore_path: list[str],
    capacity_by_track: dict[str, float],
) -> tuple[Any, ...]:
    adding_m = sum(float(vehicle_by_no[vehicle_no].vehicle_length) for vehicle_no in task.outbound_vehicle_nos)
    capacity_m = capacity_by_track.get(target_track, 0.0)
    spare_m = max(0.0, capacity_m - adding_m)
    route_m = _phase2_path_track_cost_m(
        path_tracks=detach_path,
        capacity_by_track=capacity_by_track,
    ) + _phase2_path_track_cost_m(
        path_tracks=restore_path,
        capacity_by_track=capacity_by_track,
    )
    return (
        round(route_m, 3),
        len(detach_path) + len(restore_path),
        _phase2_reorder_buffer_role_rank(target_track),
        round(spare_m, 3),
        target_track,
    )


def _phase2_reorder_buffer_role_rank(target_track: str) -> int:
    if target_track in {"修1库外", "修2库外", "修3库外", "修4库外"}:
        return 0
    if target_track in {"存5北", "存5南"}:
        return 1
    if target_track in {"存1", "存2", "存3"}:
        return 2
    return 3


def _phase2_path_track_cost_m(
    *,
    path_tracks: list[str],
    capacity_by_track: dict[str, float],
) -> float:
    return sum(float(capacity_by_track.get(track, 0.0) or 0.0) for track in path_tracks)


def _phase2_desired_prefix_order(
    *,
    current_prefix: list[str],
    outbound_vehicle_nos: set[str],
) -> list[str]:
    return sorted(
        current_prefix,
        key=lambda vehicle_no: (
            0 if vehicle_no in outbound_vehicle_nos else 1,
            current_prefix.index(vehicle_no),
            vehicle_no,
        ),
    )


def _phase2_choose_reorder_staging_track(
    *,
    source_track: str,
    staging_vehicle_nos: tuple[str, ...],
    state: ReplayState,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    capacity_by_track: dict[str, float],
) -> str | None:
    for target_track in PHASE2_REORDER_BUFFER_TRACKS:
        if target_track == source_track:
            continue
        if state.track_sequences.get(target_track):
            continue
        if not _phase2_track_has_capacity(
            target_track=target_track,
            adding_vehicle_nos=staging_vehicle_nos,
            working_state=state,
            vehicle_by_no=vehicle_by_no,
            capacity_by_track=capacity_by_track,
        ):
            continue
        detach_path = route_oracle.resolve_clear_path_tracks(
            source_track,
            target_track,
            occupied_track_sequences=state.track_sequences,
            source_node=route_oracle.order_end_node(source_track),
            target_node=route_oracle.order_end_node(target_track),
        )
        if detach_path is not None:
            return target_track
    return None


def _phase2_append_attach_plan(
    *,
    plan_payload: list[dict[str, Any]],
    hook_no: int,
    state: ReplayState,
    normalized: Any,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    source_track: str,
    vehicle_nos: list[str],
) -> tuple[ReplayState, int]:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in vehicle_nos]
    if validate_hook_vehicle_group(vehicles):
        raise ValueError(f"phase2 reorder attach invalid hook group: {vehicle_nos}")
    attach_path = route_oracle.resolve_clear_path_tracks(
        state.loco_track_name,
        source_track,
        occupied_track_sequences=state.track_sequences,
        source_node=state.loco_node,
        target_node=route_oracle.order_end_node(source_track),
    )
    if attach_path is None:
        raise ValueError(f"phase2 reorder attach path blocked: {state.loco_track_name} -> {source_track}")
    move = HookAction(
        source_track=source_track,
        target_track=source_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=list(attach_path),
        action_type="ATTACH",
    )
    next_state = _apply_move(
        state=state,
        move=move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    plan_payload.append(
        {
            "hookNo": hook_no,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": list(vehicle_nos),
            "pathTracks": list(attach_path),
            "reason": "PHASE2_REORDER_PREFIX",
        }
    )
    return next_state, hook_no + 1


def _phase2_append_detach_plan(
    *,
    plan_payload: list[dict[str, Any]],
    hook_no: int,
    state: ReplayState,
    normalized: Any,
    route_oracle: Any,
    vehicle_by_no: dict[str, Any],
    target_track: str,
    vehicle_nos: list[str],
) -> tuple[ReplayState, int]:
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    if not is_carried_tail_block(state.loco_carry, vehicle_nos):
        raise ValueError(f"phase2 reorder detach not at carry tail: {vehicle_nos}")
    vehicles = [vehicle_by_no[vehicle_no] for vehicle_no in vehicle_nos]
    if validate_hook_vehicle_group(vehicles):
        raise ValueError(f"phase2 reorder detach invalid hook group: {vehicle_nos}")
    source_track = state.loco_track_name
    source_node = state.loco_node if _phase2_remaining_source_vehicle_count_after_detach(
        state=state,
        source_track=source_track,
        vehicle_nos=vehicle_nos,
    ) > 0 else None
    target_node = route_oracle.order_end_node(target_track)
    detach_path = route_oracle.resolve_clear_path_tracks(
        source_track,
        target_track,
        occupied_track_sequences=state.track_sequences,
        source_node=source_node,
        target_node=target_node,
    )
    if detach_path is None:
        raise ValueError(f"phase2 reorder detach path blocked: {source_track} -> {target_track}")
    move = HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=list(vehicle_nos),
        path_tracks=list(detach_path),
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    plan_payload.append(
        {
            "hookNo": hook_no,
            "actionType": "DETACH",
            "sourceTrack": source_track,
            "targetTrack": target_track,
            "vehicleNos": list(vehicle_nos),
            "pathTracks": list(detach_path),
            "reason": "PHASE2_REORDER_PREFIX",
        }
    )
    return next_state, hook_no + 1


def _phase2_remaining_source_vehicle_count_after_detach(
    *,
    state: ReplayState,
    source_track: str,
    vehicle_nos: list[str] | tuple[str, ...],
) -> int:
    source_seq = list(state.track_sequences.get(source_track, ()))
    vehicle_list = list(vehicle_nos)
    if vehicle_list and source_seq[: len(vehicle_list)] == vehicle_list:
        return max(0, len(source_seq) - len(vehicle_list))
    return len(source_seq)


def _locate_vehicle_track(state: ReplayState, vehicle_no: str) -> str:
    for track_name, seq in state.track_sequences.items():
        if vehicle_no in seq:
            return track_name
    raise ValueError(f"Vehicle not found in replay state: {vehicle_no}")


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
    track_info: list[dict],
    current_vehicle_info: list[dict],
    current_state: ReplayState,
    master: MasterData,
) -> dict:
    stage_policy = stage.get("stagePolicy") if isinstance(stage, dict) else None
    stage_mode = str((stage_policy or {}).get("stageMode") or "")
    if stage_mode == "PHASE1_PRE_REPAIR_BUFFERING":
        original_goal_rows = list((stage_policy or {}).get("phase1OriginalGoalRows") or [])
        if not original_goal_rows:
            return stage
        return rebuild_phase1_stage_for_runtime(
            master=master,
            track_info=track_info,
            current_vehicle_info=current_vehicle_info,
            loco_track_name=current_state.loco_track_name,
            original_goal_rows=original_goal_rows,
        )
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
    base_depot_track_sequences = {
        track: list(vehicle_nos)
        for track, vehicle_nos in planned_track_sequences.items()
    }
    base_depot_spot_assignments = dict(planned_spot_assignments)
    admitted_goals, admission_diagnostics = _apply_phase3_admission_budget(
        goals=[dict(goal) for goal in stage.get("vehicleGoals") or []],
        current_by_vehicle=current_by_vehicle,
        base_track_sequences=base_depot_track_sequences,
        base_spot_assignments=base_depot_spot_assignments,
        vehicle_by_no=vehicle_by_no,
        yard_mode=normalized.yard_mode,
        reserved_spot_codes=reserved_spot_codes,
    )

    insertion_plan = None
    insertion_score = None
    insertion_error = ""
    try:
        insertion_plan = plan_phase3_depot_insertion(
            goals=admitted_goals,
            current_by_vehicle=current_by_vehicle,
            current_track_sequences=planned_track_sequences,
            current_spot_assignments=planned_spot_assignments,
            vehicle_by_no=vehicle_by_no,
            yard_mode=normalized.yard_mode,
            reserved_spot_codes=reserved_spot_codes,
        )
        insertion_score = score_phase3_depot_execution_fitness(
            resolved_track_by_vehicle=insertion_plan.resolved_track_by_vehicle,
            current_by_vehicle=current_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            track_sequences=insertion_plan.track_sequences,
        )
    except ValueError as exc:
        insertion_error = str(exc)
    relayout_plan = search_phase3_depot_relayout(
        goals=admitted_goals,
        current_by_vehicle=current_by_vehicle,
        base_track_sequences=base_depot_track_sequences,
        base_spot_assignments=base_depot_spot_assignments,
        vehicle_by_no=vehicle_by_no,
        yard_mode=normalized.yard_mode,
        reserved_spot_codes=reserved_spot_codes,
        max_search_nodes=20_000,
    )
    relayout_score = (
        score_phase3_depot_execution_fitness(
            resolved_track_by_vehicle=relayout_plan.resolved_track_by_vehicle,
            current_by_vehicle=current_by_vehicle,
            vehicle_by_no=vehicle_by_no,
            track_sequences=relayout_plan.track_sequences,
        )
        if relayout_plan.feasible
        else None
    )
    if insertion_plan is None and not relayout_plan.feasible:
        raise ValueError(
            "phase3 depot target resolution failed: "
            f"insertion_error={insertion_error}; "
            f"relayout={relayout_plan.diagnostics}"
        )
    if relayout_plan.feasible and (insertion_score is None or relayout_score < insertion_score):
        active_depot_plan = relayout_plan
        active_resolver = "relayout"
    else:
        active_depot_plan = insertion_plan
        active_resolver = "insertion"
    if active_depot_plan is None:
        raise ValueError(
            "phase3 depot target resolution failed: no active depot plan"
        )
    planned_track_sequences = active_depot_plan.track_sequences
    planned_spot_assignments = active_depot_plan.spot_assignments

    resolved_goals: list[dict[str, Any]] = []
    for goal in admitted_goals:
        if str(goal.get("targetSource") or "") == PHASE3_DYNAMIC_CURRENT_HOLD:
            vehicle_no = str(goal["vehicleNo"])
            current = current_by_vehicle[vehicle_no]
            resolved_goals.append(
                {
                    "vehicleNo": vehicle_no,
                    "targetTrack": str(current["trackName"]),
                    "targetMode": "SNAPSHOT",
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
        target_track = active_depot_plan.resolved_track_by_vehicle[vehicle_no]
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
    resolved_policy["phase3DepotResolverScores"] = {
        "insertion": insertion_score,
        "relayout": relayout_score,
    }
    resolved_policy["phase3AdmissionDiagnostics"] = admission_diagnostics
    resolved_policy["phase3DepotRelayoutDiagnostics"] = relayout_plan.diagnostics
    resolved_policy["phase3DepotTargetResolver"] = active_resolver
    resolved_policy["phase3DepotInsertionDiagnostics"] = (
        insertion_plan.diagnostics
        if insertion_plan is not None
        else {
            "enabled": False,
            "error": insertion_error,
        }
    )
    block_plan = build_phase3_depot_block_plan(
        goals=resolved_goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences=current_state.track_sequences,
        depot_track_sequences=base_depot_track_sequences,
        current_spot_assignments=base_depot_spot_assignments,
        vehicle_by_no=vehicle_by_no,
        yard_mode=normalized.yard_mode,
        reserved_spot_codes=reserved_spot_codes,
        min_active_vehicle_count=1,
        min_source_track_count=1,
    )
    resolved_policy["phase3BlockPlanDiagnostics"] = block_plan.diagnostics
    preflighted_wave_plans, preflight_diagnostics = _build_preflighted_phase3_wave_plans(
        stage=resolved_stage,
        current_vehicle_info=current_vehicle_info,
        current_by_vehicle=current_by_vehicle,
        current_state=current_state,
        master=master,
        source_wave_plans=block_plan.wave_plans,
        all_active_covered=block_plan.diagnostics.get("allActiveCoveredByFrontier") is True,
    )
    resolved_policy["phase3ExecutionPlanDiagnostics"] = preflight_diagnostics
    if preflighted_wave_plans:
        resolved_policy["phase3WavePlans"] = preflighted_wave_plans
    else:
        resolved_policy.pop("phase3WavePlans", None)
    resolved_stage["stagePolicy"] = resolved_policy
    return resolved_stage


def _apply_phase3_admission_budget(
    *,
    goals: list[dict[str, Any]],
    current_by_vehicle: dict[str, dict],
    base_track_sequences: dict[str, list[str]],
    base_spot_assignments: dict[str, str],
    vehicle_by_no: dict[str, Any],
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    depot_tracks = ("修1", "修2", "修3", "修4")
    capacity = sum(len(list_track_spots(track, yard_mode)) for track in depot_tracks)
    working_track_sequences = {
        track: list(base_track_sequences.get(track, ()))
        for track in depot_tracks
    }
    working_spot_assignments = dict(base_spot_assignments)
    base_vehicle_nos = tuple(
        vehicle_no
        for track in depot_tracks
        for vehicle_no in list(working_track_sequences.get(track, ()))
    )
    pending_goals = [
        dict(goal)
        for goal in goals
        if _phase3_goal_needs_depot_admission(
            goal=goal,
            current_by_vehicle=current_by_vehicle,
        )
    ]
    base_count = len(base_vehicle_nos)
    available_slots = max(0, capacity - base_count)
    active_budget = min(available_slots, PHASE3_ADMISSION_MAX_RANDOM_DEPOT_VEHICLES)
    admitted_vehicle_nos: set[str] = set()
    deferred_vehicle_nos: set[str] = set()
    assigned_track_by_vehicle: dict[str, str] = {}
    rejected_by_vehicle: dict[str, dict[str, str]] = {}
    for goal in sorted(
        pending_goals,
        key=lambda item: _phase3_admission_goal_priority(
            goal=item,
            current_by_vehicle=current_by_vehicle,
        ),
    ):
        vehicle_no = str(goal["vehicleNo"])
        if len(admitted_vehicle_nos) >= active_budget:
            deferred_vehicle_nos.add(vehicle_no)
            rejected_by_vehicle[vehicle_no] = {"*": "phase3_active_vehicle_budget"}
            continue
        vehicle = current_by_vehicle[vehicle_no]
        rejected: dict[str, str] = {}
        admitted_track = ""
        for target_track in _ordered_phase3_admission_candidate_tracks(
            goal=goal,
            vehicle=vehicle,
            planned_track_sequences=working_track_sequences,
        ):
            next_track_seq = [vehicle_no] + list(working_track_sequences.get(target_track, ()))
            next_spot_assignments = realign_spots_for_track_order(
                vehicle_nos_in_order=next_track_seq,
                vehicle_by_no=vehicle_by_no,
                target_track=target_track,
                yard_mode=yard_mode,
                current_spot_assignments=working_spot_assignments,
                reserved_spot_codes=reserved_spot_codes,
            )
            if next_spot_assignments is None:
                rejected[target_track] = "spot_realign_failed"
                continue
            admitted_track = target_track
            working_track_sequences[target_track] = next_track_seq
            working_spot_assignments = next_spot_assignments
            admitted_vehicle_nos.add(vehicle_no)
            assigned_track_by_vehicle[vehicle_no] = target_track
            break
        if not admitted_track:
            deferred_vehicle_nos.add(vehicle_no)
            rejected_by_vehicle[vehicle_no] = rejected or {"*": "no_candidate_depot_track"}
    admitted_goals: list[dict[str, Any]] = []
    for goal in goals:
        vehicle_no = str(goal.get("vehicleNo") or "")
        if vehicle_no not in deferred_vehicle_nos:
            admitted_goals.append(dict(goal))
            continue
        current = current_by_vehicle[vehicle_no]
        admitted_goals.append(
            {
                "vehicleNo": vehicle_no,
                "targetTrack": str(current["trackName"]),
                "targetMode": "SNAPSHOT",
                "targetSource": PHASE3_DYNAMIC_CURRENT_HOLD,
                "isSpotting": "",
            }
        )
    return (
        admitted_goals,
        {
            "enabled": True,
            "capacity": capacity,
            "baseVehicleCount": base_count,
            "baseVehicleNos": list(base_vehicle_nos),
            "baseTrackCounts": {
                track: len(base_track_sequences.get(track, ()))
                for track in depot_tracks
            },
            "pendingVehicleCount": len(pending_goals),
            "availableSlots": available_slots,
            "activeVehicleBudget": active_budget,
            "admittedVehicleNos": sorted(admitted_vehicle_nos),
            "admittedVehicleCount": len(admitted_vehicle_nos),
            "assignedTrackByVehicle": dict(sorted(assigned_track_by_vehicle.items())),
            "deferredVehicleNos": sorted(deferred_vehicle_nos),
            "deferredVehicleCount": len(deferred_vehicle_nos),
            "deferredReason": (
                ""
                if not deferred_vehicle_nos
                else "phase3_depot_track_admission_budget"
            ),
            "rejectedByVehicle": rejected_by_vehicle,
            "projectedTrackCounts": {
                track: len(working_track_sequences.get(track, ()))
                for track in depot_tracks
            },
            "projectedVehicleCount": base_count + len(admitted_vehicle_nos),
        },
    )


def _phase3_admission_goal_priority(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> tuple[Any, ...]:
    vehicle_no = str(goal["vehicleNo"])
    vehicle = current_by_vehicle[vehicle_no]
    candidates = _phase3_admission_candidate_tracks(goal=goal, vehicle=vehicle)
    vehicle_length = float(vehicle.get("vehicleLength") or 0.0)
    repair_process = str(vehicle.get("repairProcess") or "")
    return (
        len(candidates),
        0 if repair_process == "厂修" else 1,
        0 if vehicle_length >= 17.6 else 1,
        str(vehicle.get("trackName") or ""),
        str(vehicle.get("order") or ""),
        vehicle_no,
    )


def _phase3_goal_needs_depot_admission(
    *,
    goal: dict[str, Any],
    current_by_vehicle: dict[str, dict],
) -> bool:
    if str(goal.get("targetSource") or "") == PHASE3_DYNAMIC_CURRENT_HOLD:
        return False
    vehicle_no = str(goal["vehicleNo"])
    current_track = str(current_by_vehicle.get(vehicle_no, {}).get("trackName") or "")
    if current_track in {"修1", "修2", "修3", "修4"}:
        return False
    target_track = str(goal.get("targetTrack") or "")
    target_area_code = str(goal.get("targetAreaCode") or "")
    return (
        target_area_code == "大库:RANDOM"
        or target_track == "大库"
        or target_track in {"修1", "修2", "修3", "修4"}
    )


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


def _ordered_phase3_admission_candidate_tracks(
    *,
    goal: dict[str, Any],
    vehicle: dict,
    planned_track_sequences: dict[str, list[str]],
) -> list[str]:
    preferred = [str(item) for item in goal.get("preferredTargetTracks") or ()]
    candidates = _phase3_admission_candidate_tracks(goal=goal, vehicle=vehicle)
    return sorted(
        candidates,
        key=lambda track: (
            0 if track in preferred else 1,
            len(planned_track_sequences.get(track, ())),
            track,
        ),
    )


def _phase3_admission_candidate_tracks(
    *,
    goal: dict[str, Any],
    vehicle: dict,
) -> list[str]:
    target_track = str(goal.get("targetTrack") or "")
    target_area_code = str(goal.get("targetAreaCode") or "")
    if target_track in {"修1", "修2", "修3", "修4"} and target_area_code != "大库:RANDOM":
        candidates = [target_track]
    else:
        candidates = _phase3_random_depot_candidate_tracks(goal=goal, vehicle=vehicle)
    if float(vehicle.get("vehicleLength") or 0.0) >= 17.6:
        candidates = [track for track in candidates if track in {"修3", "修4"}]
    return list(dict.fromkeys(candidates))


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
