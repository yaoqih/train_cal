from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(vehicle_no, track, target, *, order=1, length=14.3):
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": length,
        "targetTrack": target,
        "isSpotting": "",
        "vehicleAttributes": "",
    }


def test_capacity_release_plan_reports_fixed_inbound_and_releasable_occupants():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 30.0},
            {"trackName": "存2", "trackDistance": 120.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("KEEP", "存1", "存1", order=1, length=10.0),
            _vehicle("RELEASE", "存1", "存2", order=2, length=12.0),
            _vehicle("INBOUND", "存5北", "存1", order=1, length=15.0),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    plan = compute_capacity_release_plan(normalized, state)
    fact = plan.facts_by_track["存1"]

    assert fact.capacity_length == 30.0
    assert fact.current_length == 22.0
    assert fact.keepable_current_length == 10.0
    assert fact.non_goal_current_length == 12.0
    assert fact.fixed_inbound_length == 15.0
    assert fact.release_pressure_length == 7.0
    assert fact.front_release_vehicle_nos == ["KEEP", "RELEASE"]
    assert fact.front_release_length == 22.0


def test_capacity_release_plan_does_not_force_random_area_vehicles_into_fixed_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("RANDOM_DEPOT", "存5北", "大库", order=1, length=14.3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    plan = compute_capacity_release_plan(normalized, state)

    assert plan.facts_by_track["修1库内"].fixed_inbound_vehicle_count == 0
    assert plan.facts_by_track["修2库内"].fixed_inbound_vehicle_count == 0
