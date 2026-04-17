from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import InputValidationError, normalize_plan_input


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_normal_mode_rejects_06_07_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "S1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "106",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_inspection_mode_allows_06_07_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "S2",
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
                "vehicleNo": "S3",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "106",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }

    result = normalize_plan_input(payload, master)

    assert result.yard_mode == "INSPECTION"
    exact_spot_vehicle = next(vehicle for vehicle in result.vehicles if vehicle.vehicle_no == "S3")
    assert exact_spot_vehicle.goal.target_mode == "SPOT"
    assert exact_spot_vehicle.goal.target_spot_code == "106"
