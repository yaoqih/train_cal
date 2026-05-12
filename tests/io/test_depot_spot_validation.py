from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.depot_spots import (
    build_initial_spot_assignments,
    exact_spot_reservations,
    allocate_spots_for_block,
)
from fzed_shunting.io.normalize_input import InputValidationError, normalize_plan_input
from fzed_shunting.solver.goal_logic import goal_is_satisfied
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_normal_mode_rejects_06_07_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
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
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
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


def test_track_mode_depot_vehicles_consume_initial_depot_spots():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TD1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetMode": "TRACK",
                "targetTrack": "修1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TD2",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetMode": "TRACK",
                "targetTrack": "修1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }

    normalized = normalize_plan_input(payload, master)
    assignments = build_initial_spot_assignments(normalized)

    assert assignments == {"TD1": "101", "TD2": "102"}


def test_random_depot_allocation_reserves_unsatisfied_exact_depot_spots():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RANDOM_DEPOT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT106",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "修1",
                "targetSpotCode": "106",
                "isSpotting": "迎检",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicles = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    occupied = {f"OCC{i}": f"10{i}" for i in range(1, 6)}
    reservations = exact_spot_reservations(normalized)

    random_allocated = allocate_spots_for_block(
        vehicles=[vehicles["RANDOM_DEPOT"]],
        target_track="修1",
        yard_mode=normalized.yard_mode,
        occupied_spot_assignments=occupied,
        reserved_spot_codes=reservations,
    )
    exact_allocated = allocate_spots_for_block(
        vehicles=[vehicles["SPOT106"]],
        target_track="修1",
        yard_mode=normalized.yard_mode,
        occupied_spot_assignments=occupied,
        reserved_spot_codes=reservations,
    )

    assert random_allocated == {"RANDOM_DEPOT": "107"}
    assert exact_allocated == {"SPOT106": "106"}


def test_initial_spot_assignments_preserve_real_occupants_of_reserved_exact_spots():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "修1",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"OCC{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 6)
            ],
            {
                "trackName": "修1",
                "order": "6",
                "vehicleModel": "棚车",
                "vehicleNo": "REAL_106_OCCUPANT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT106",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "修1",
                "targetSpotCode": "106",
                "isSpotting": "迎检",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    assignments = build_initial_spot_assignments(normalized)

    assert assignments["REAL_106_OCCUPANT"] == "106"


def test_random_depot_vehicle_on_initial_reserved_exact_spot_is_still_satisfied():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "修1",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"OCC{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 6)
            ],
            {
                "trackName": "修1",
                "order": "6",
                "vehicleModel": "棚车",
                "vehicleNo": "REAL_106_OCCUPANT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT106",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "修1",
                "targetSpotCode": "106",
                "isSpotting": "迎检",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    occupant = next(
        vehicle for vehicle in normalized.vehicles
        if vehicle.vehicle_no == "REAL_106_OCCUPANT"
    )

    assert state.spot_assignments["REAL_106_OCCUPANT"] == "106"
    assert goal_is_satisfied(
        occupant,
        track_name="修1",
        state=state,
        plan_input=normalized,
    )
