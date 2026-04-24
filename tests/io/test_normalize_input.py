from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import (
    InputValidationError,
    normalize_plan_input,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _base_payload():
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修1库外", "trackDistance": 49.3},
        ],
        "vehicleInfo": [],
        "locoTrackName": "",
    }


def test_default_loco_track_name_is_jiku():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()

    result = normalize_plan_input(payload, master)

    assert result.loco_track_name == "机库"


def test_workflow_internal_normalization_accepts_non_contract_loco_track():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["locoTrackName"] = "调棚"
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "WFIN1",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)

    assert result.loco_track_name == "调棚"


def test_track_target_stays_track_for_regular_storage_line():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A1",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "TRACK"
    assert vehicle.goal.target_track == "机库"
    assert vehicle.goal.allowed_target_tracks == ["机库"]


def test_dispatch_track_becomes_area_for_work_track():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A2",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "调棚",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "调棚:PRE_REPAIR"


def test_spotting_yes_maps_to_dispatch_work_area():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A3",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "调棚",
            "isSpotting": "是",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "调棚:WORK"


def test_wash_track_defaults_to_work_area_when_spotting_empty():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "罐车",
            "vehicleNo": "A3W",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "洗南",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "洗南:WORK"
    assert vehicle.goal.allowed_target_tracks == ["洗南"]


def test_wheel_track_defaults_to_operate_area_when_spotting_empty():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].append({"trackName": "轮", "trackDistance": 118.2})
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "敞车",
            "vehicleNo": "A3L",
            "repairProcess": "临修",
            "vehicleLength": 14.3,
            "targetTrack": "轮",
            "isSpotting": "",
            "vehicleAttributes": "重车",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "轮:OPERATE"
    assert vehicle.goal.allowed_target_tracks == ["轮"]


def test_outer_depot_aggregate_defaults_to_random_area():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A3O",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "大库外",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "大库外:RANDOM"
    assert set(vehicle.goal.allowed_target_tracks) == {
        "修1库外",
        "修2库外",
        "修3库外",
        "修4库外",
    }


def test_specific_outer_depot_track_stays_track_goal():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A3T",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "修1库外",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "TRACK"
    assert vehicle.goal.target_track == "修1库外"
    assert vehicle.goal.allowed_target_tracks == ["修1库外"]


def test_paint_track_defaults_to_work_area_when_spotting_empty():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].append({"trackName": "油", "trackDistance": 124})
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A3P",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "油",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "油:WORK"
    assert vehicle.goal.allowed_target_tracks == ["油"]


def test_shot_track_defaults_to_work_area_when_spotting_empty():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].append({"trackName": "抛", "trackDistance": 131.8})
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A3S",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "抛",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "抛:WORK"
    assert vehicle.goal.allowed_target_tracks == ["抛"]


def test_inspection_promotes_plan_mode_and_random_depot_area():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A4",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "迎检",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert result.yard_mode == "INSPECTION"
    assert vehicle.goal.target_mode == "AREA"
    assert vehicle.goal.target_area_code == "大库:RANDOM"
    assert set(vehicle.goal.allowed_target_tracks) == {
        "修1库内",
        "修2库内",
        "修3库内",
        "修4库内",
    }


def test_random_depot_short_vehicle_prefers_inner_tracks_1_2_with_3_4_as_fallback():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].extend(
        [
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ]
    )
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "SHORT_DEPOT",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]

    assert vehicle.goal.target_area_code == "大库:RANDOM"
    assert vehicle.goal.preferred_target_tracks == ["修1库内", "修2库内"]
    assert vehicle.goal.fallback_target_tracks == ["修3库内", "修4库内"]


def test_random_depot_long_vehicle_prefers_only_inner_tracks_3_4():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].extend(
        [
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ]
    )
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "LONG_DEPOT",
            "repairProcess": "厂修",
            "vehicleLength": 17.6,
            "targetTrack": "大库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]

    assert vehicle.goal.target_area_code == "大库:RANDOM"
    assert vehicle.goal.preferred_target_tracks == ["修3库内", "修4库内"]
    assert vehicle.goal.fallback_target_tracks == []


def test_numeric_spotting_maps_to_spot_goal():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "A5",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "101",
            "vehicleAttributes": "",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "SPOT"
    assert vehicle.goal.target_track == "修1库内"
    assert vehicle.goal.target_spot_code == "101"


def test_vehicle_attribute_parsing():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "敞车",
            "vehicleNo": "A6",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "称重",
        }
    ]

    result = normalize_plan_input(payload, master)
    vehicle = result.vehicles[0]
    assert vehicle.need_weigh is True
    assert vehicle.is_heavy is False
    assert vehicle.is_close_door is False


def test_invalid_area_request_on_storage_track_raises():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "敞车",
            "vehicleNo": "A7",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "是",
            "vehicleAttributes": "",
        }
    ]

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_temporary_track_cannot_be_final_target():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "敞车",
            "vehicleNo": "A8",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "存4南",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_running_track_cannot_be_final_target():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].append({"trackName": "联6", "trackDistance": 53})
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "敞车",
            "vehicleNo": "A9",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "联6",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_mixed_mode_allows_inspection_with_normal_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["trackInfo"].extend(
        [
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ]
    )
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "INSPECT_A",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "迎检",
            "vehicleAttributes": "",
        },
        {
            "trackName": "存5北",
            "order": "2",
            "vehicleModel": "棚车",
            "vehicleNo": "NORMAL_ONLY_205",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "205",
            "vehicleAttributes": "",
        },
    ]

    result = normalize_plan_input(payload, master)

    assert result.yard_mode == "INSPECTION"
    goals_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in result.vehicles}
    assert goals_by_vehicle["INSPECT_A"].target_mode == "AREA"
    assert goals_by_vehicle["NORMAL_ONLY_205"].target_mode == "SPOT"
    assert goals_by_vehicle["NORMAL_ONLY_205"].target_spot_code == "205"
