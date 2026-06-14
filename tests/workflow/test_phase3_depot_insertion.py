from __future__ import annotations

from pathlib import Path

import pytest

from fzed_shunting.domain.depot_spots import exact_spot_reservations
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.replay import ReplayState
from fzed_shunting.workflow.phase3_depot_insertion import plan_phase3_depot_insertion
from fzed_shunting.workflow.runner import _resolve_phase3_depot_targets


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(
    *,
    vehicle_no: str,
    track_name: str,
    order: str = "1",
    target_track: str = "大库",
    vehicle_length: float = 14.3,
    repair_process: str = "段修",
    is_spotting: str = "",
) -> dict:
    return {
        "trackName": track_name,
        "order": order,
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": repair_process,
        "vehicleLength": vehicle_length,
        "vehicleAttributes": "",
        "targetTrack": target_track,
        "isSpotting": is_spotting,
    }


def _stage_goal(vehicle_no: str, *, vehicle_length: float = 14.3) -> dict:
    if vehicle_length >= 17.6:
        preferred = ["修3", "修4"]
        fallback = []
    else:
        preferred = ["修1", "修2"]
        fallback = ["修3", "修4"]
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": "修1",
        "targetMode": "AREA",
        "targetAreaCode": "大库:RANDOM",
        "allowedTargetTracks": ["修1", "修2", "修3", "修4"],
        "preferredTargetTracks": preferred,
        "fallbackTargetTracks": fallback,
        "isSpotting": "",
    }


def _normalized(vehicle_info: list[dict], goals: list[dict]):
    master = load_master_data(DATA_DIR)
    goal_by_vehicle = {goal["vehicleNo"]: goal for goal in goals}
    payload = {
        "trackInfo": [{"trackName": item["trackName"], "trackDistance": 120.0} for item in vehicle_info],
        "vehicleInfo": [
            {
                **item,
                **goal_by_vehicle.get(item["vehicleNo"], {"targetTrack": item["targetTrack"]}),
            }
            for item in vehicle_info
        ],
        "locoTrackName": "机库",
    }
    return normalize_plan_input(payload, master, allow_internal_loco_tracks=True)


def test_plan_phase3_depot_insertion_resolves_random_depot_to_concrete_track():
    goals = [_stage_goal("A")]
    current_by_vehicle = {"A": _vehicle(vehicle_no="A", track_name="存4北")}
    normalized = _normalized(list(current_by_vehicle.values()), goals)

    plan = plan_phase3_depot_insertion(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        current_track_sequences={track: [] for track in ("修1", "修2", "修3", "修4")},
        current_spot_assignments={},
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        yard_mode=normalized.yard_mode,
        reserved_spot_codes=exact_spot_reservations(normalized),
    )

    assert plan.resolved_track_by_vehicle["A"] in {"修1", "修2", "修3", "修4"}
    assert plan.resolved_track_by_vehicle["A"] != "大库"


def test_plan_phase3_depot_insertion_puts_long_vehicle_only_on_long_depot_tracks():
    goals = [_stage_goal("LONG", vehicle_length=17.6)]
    current_by_vehicle = {
        "LONG": _vehicle(vehicle_no="LONG", track_name="存4北", vehicle_length=17.6),
    }
    normalized = _normalized(list(current_by_vehicle.values()), goals)

    plan = plan_phase3_depot_insertion(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        current_track_sequences={track: [] for track in ("修1", "修2", "修3", "修4")},
        current_spot_assignments={},
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        yard_mode=normalized.yard_mode,
        reserved_spot_codes=exact_spot_reservations(normalized),
    )

    assert plan.resolved_track_by_vehicle["LONG"] in {"修3", "修4"}


def test_plan_phase3_depot_insertion_fails_explicitly_when_no_depot_track_fits():
    goals = [_stage_goal("EXTRA")]
    current_by_vehicle = {"EXTRA": _vehicle(vehicle_no="EXTRA", track_name="存4北")}
    for track in ("修1", "修2", "修3", "修4"):
        for slot in range(1, 6):
            vehicle_no = f"{track}_{slot}"
            current_by_vehicle[vehicle_no] = _vehicle(
                vehicle_no=vehicle_no,
                track_name=track,
                order=str(slot),
                target_track=track,
            )
    normalized = _normalized(list(current_by_vehicle.values()), goals)

    with pytest.raises(ValueError, match="phase3 depot insertion failed"):
        plan_phase3_depot_insertion(
            goals=goals,
            current_by_vehicle=current_by_vehicle,
            current_track_sequences={
                track: [f"{track}_{slot}" for slot in range(1, 6)]
                for track in ("修1", "修2", "修3", "修4")
            },
            current_spot_assignments={},
            vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
            yard_mode=normalized.yard_mode,
            reserved_spot_codes=exact_spot_reservations(normalized),
        )


def test_resolve_phase3_depot_targets_never_emits_track_mode_abstract_depot():
    master = load_master_data(DATA_DIR)
    current_vehicle_info = [_vehicle(vehicle_no="A", track_name="存4北")]
    stage = {
        "name": "phase3_ji_to_depot_allocation",
        "vehicleGoals": [_stage_goal("A")],
        "stagePolicy": {"stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION"},
    }

    resolved = _resolve_phase3_depot_targets(
        stage=stage,
        current_vehicle_info=current_vehicle_info,
        current_state=ReplayState(
            track_sequences={"存4北": ["A"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )

    goal = resolved["vehicleGoals"][0]
    assert goal["targetMode"] == "TRACK"
    assert goal["targetTrack"] in {"修1", "修2", "修3", "修4"}


def test_resolve_phase3_depot_targets_defers_vehicle_when_only_allowed_track_is_full():
    master = load_master_data(DATA_DIR)
    current_vehicle_info = [_vehicle(vehicle_no="EXTRA", track_name="机北1")]
    goals = [
        {
            "vehicleNo": "EXTRA",
            "targetTrack": "大库",
            "targetMode": "AREA",
            "targetAreaCode": "大库:RANDOM",
            "allowedTargetTracks": ["修1"],
            "preferredTargetTracks": ["修1"],
            "fallbackTargetTracks": [],
            "isSpotting": "",
        }
    ]
    track_sequences = {"机北1": ["EXTRA"], "修1": []}
    for slot in range(1, 6):
        vehicle_no = f"R{slot}"
        current_vehicle_info.append(
            _vehicle(
                vehicle_no=vehicle_no,
                track_name="修1",
                order=str(slot),
                target_track="修1",
            )
        )
        goals.append(
            {
                "vehicleNo": vehicle_no,
                "targetTrack": "修1",
                "targetMode": "TRACK",
                "isSpotting": "",
            }
        )
        track_sequences["修1"].append(vehicle_no)
    stage = {
        "name": "phase3_ji_to_depot_allocation",
        "vehicleGoals": goals,
        "stagePolicy": {"stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION"},
    }

    resolved = _resolve_phase3_depot_targets(
        stage=stage,
        current_vehicle_info=current_vehicle_info,
        current_state=ReplayState(
            track_sequences=track_sequences,
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )

    goal_by_vehicle = {goal["vehicleNo"]: goal for goal in resolved["vehicleGoals"]}
    assert goal_by_vehicle["EXTRA"]["targetTrack"] == "机北1"
    assert goal_by_vehicle["EXTRA"]["targetMode"] == "SNAPSHOT"
    assert goal_by_vehicle["EXTRA"]["targetSource"] == "PHASE3_DYNAMIC_CURRENT_HOLD"
    diagnostics = resolved["stagePolicy"]["phase3AdmissionDiagnostics"]
    assert diagnostics["deferredVehicleNos"] == ["EXTRA"]
    assert diagnostics["projectedTrackCounts"]["修1"] == 5


def test_resolve_phase3_depot_targets_defers_concrete_depot_target_when_track_is_full():
    master = load_master_data(DATA_DIR)
    current_vehicle_info = [_vehicle(vehicle_no="EXTRA", track_name="机北1", target_track="修1")]
    goals = [
        {
            "vehicleNo": "EXTRA",
            "targetTrack": "修1",
            "targetMode": "TRACK",
            "isSpotting": "",
        }
    ]
    track_sequences = {"机北1": ["EXTRA"], "修1": []}
    for slot in range(1, 6):
        vehicle_no = f"R{slot}"
        current_vehicle_info.append(
            _vehicle(
                vehicle_no=vehicle_no,
                track_name="修1",
                order=str(slot),
                target_track="修1",
            )
        )
        goals.append(
            {
                "vehicleNo": vehicle_no,
                "targetTrack": "修1",
                "targetMode": "TRACK",
                "isSpotting": "",
            }
        )
        track_sequences["修1"].append(vehicle_no)
    stage = {
        "name": "phase3_ji_to_depot_allocation",
        "vehicleGoals": goals,
        "stagePolicy": {"stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION"},
    }

    resolved = _resolve_phase3_depot_targets(
        stage=stage,
        current_vehicle_info=current_vehicle_info,
        current_state=ReplayState(
            track_sequences=track_sequences,
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )

    goal_by_vehicle = {goal["vehicleNo"]: goal for goal in resolved["vehicleGoals"]}
    assert goal_by_vehicle["EXTRA"]["targetTrack"] == "机北1"
    assert goal_by_vehicle["EXTRA"]["targetSource"] == "PHASE3_DYNAMIC_CURRENT_HOLD"
    diagnostics = resolved["stagePolicy"]["phase3AdmissionDiagnostics"]
    assert diagnostics["deferredVehicleNos"] == ["EXTRA"]
    assert diagnostics["rejectedByVehicle"]["EXTRA"] == {"修1": "spot_realign_failed"}


def test_resolve_phase3_dynamic_hold_uses_snapshot_for_non_final_buffer_track():
    master = load_master_data(DATA_DIR)
    current_vehicle_info = [_vehicle(vehicle_no="HOLD", track_name="机北1")]
    stage = {
        "name": "phase3_ji_to_depot_allocation",
        "vehicleGoals": [
            {
                "vehicleNo": "HOLD",
                "targetTrack": "机北1",
                "targetMode": "SNAPSHOT",
                "targetSource": "PHASE3_DYNAMIC_CURRENT_HOLD",
                "isSpotting": "",
            }
        ],
        "stagePolicy": {"stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION"},
    }

    resolved = _resolve_phase3_depot_targets(
        stage=stage,
        current_vehicle_info=current_vehicle_info,
        current_state=ReplayState(
            track_sequences={"机北1": ["HOLD"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )

    goal = resolved["vehicleGoals"][0]
    assert goal["targetTrack"] == "机北1"
    assert goal["targetMode"] == "SNAPSHOT"


def test_resolve_phase3_depot_targets_can_use_relayout_when_insertion_fails(monkeypatch):
    import fzed_shunting.workflow.runner as runner

    master = load_master_data(DATA_DIR)
    current_vehicle_info = [_vehicle(vehicle_no="A", track_name="存4北")]
    stage = {
        "name": "phase3_ji_to_depot_allocation",
        "vehicleGoals": [_stage_goal("A")],
        "stagePolicy": {"stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION"},
    }

    def fail_insertion(**kwargs):
        raise ValueError("forced insertion failure")

    monkeypatch.setattr(runner, "plan_phase3_depot_insertion", fail_insertion)

    resolved = _resolve_phase3_depot_targets(
        stage=stage,
        current_vehicle_info=current_vehicle_info,
        current_state=ReplayState(
            track_sequences={"存4北": ["A"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )

    assert resolved["stagePolicy"]["phase3DepotTargetResolver"] == "relayout"
    assert resolved["stagePolicy"]["phase3DepotInsertionDiagnostics"]["error"] == "forced insertion failure"
    assert resolved["vehicleGoals"][0]["targetTrack"] in {"修1", "修2", "修3", "修4"}
