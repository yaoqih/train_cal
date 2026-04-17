from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import InputValidationError, normalize_plan_input


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _base_payload():
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "联6", "trackDistance": 53},
        ],
        "vehicleInfo": [],
        "locoTrackName": "",
    }


def test_duplicate_orders_on_same_track_raise():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "F1",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        },
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "F2",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        },
    ]

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_running_track_cannot_be_source_track():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["vehicleInfo"] = [
        {
            "trackName": "联6",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": "F3",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
    ]

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)


def test_invalid_loco_track_raises():
    master = load_master_data(DATA_DIR)
    payload = _base_payload()
    payload["locoTrackName"] = "不存在"

    with pytest.raises(InputValidationError):
        normalize_plan_input(payload, master)
