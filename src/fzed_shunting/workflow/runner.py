from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.verify.replay import ReplayState

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


def solve_workflow(
    master: MasterData,
    payload: dict,
    *,
    solver: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    time_budget_ms: float | None = None,
) -> WorkflowResult:
    from fzed_shunting.demo.view_model import build_demo_view_model

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
        track_sequences={
            item["trackName"]: []
            for item in current_vehicle_info
        },
        loco_track_name=input_loco_track_name,
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    for item in current_vehicle_info:
        current_state.track_sequences.setdefault(item["trackName"], []).append(item["vehicleNo"])

    stages: list[WorkflowStageResult] = []
    for index, stage in enumerate(workflow_stages, start=1):
        stage_payload = _build_stage_payload(
            track_info=track_info,
            current_vehicle_info=current_vehicle_info,
            vehicle_meta=vehicle_meta,
            stage=stage,
            loco_track_name=current_state.loco_track_name,
        )
        view = build_demo_view_model(
            master,
            stage_payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            time_budget_ms=time_budget_ms,
            initial_state_override=current_state,
        )
        stages.append(
            WorkflowStageResult(
                name=str(stage.get("name", f"stage_{index}")),
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
        stage_vehicle_info.append(base)

    return {
        "trackInfo": [dict(item) for item in track_info],
        "vehicleInfo": stage_vehicle_info,
        "locoTrackName": loco_track_name,
        "workflowStageName": str(stage.get("name", "")),
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
