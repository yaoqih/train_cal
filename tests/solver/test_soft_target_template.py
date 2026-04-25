from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.soft_target_template import compute_soft_target_template
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(vehicle_no, track, target, *, order=1, length=14.3, attrs=""):
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": length,
        "targetTrack": target,
        "isSpotting": "",
        "vehicleAttributes": attrs,
    }


def test_soft_target_template_prefers_low_pressure_random_depot_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 30.0},
            {"trackName": "修2库内", "trackDistance": 60.0},
            {"trackName": "修3库内", "trackDistance": 60.0},
            {"trackName": "修4库内", "trackDistance": 60.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("R", "存5北", "大库", order=1, length=14.3),
            _vehicle("OCC1", "修1库内", "修1库内", order=1, length=14.3),
            _vehicle("OCC2", "修1库内", "修1库内", order=2, length=14.3),
            _vehicle("OCC3", "修1库内", "修1库内", order=3, length=14.3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    template = compute_soft_target_template(normalized, state)
    candidate_tracks = [
        candidate.track_name
        for candidate in template.vehicle_templates["R"].candidate_tracks
    ]

    assert candidate_tracks.index("修2库内") < candidate_tracks.index("修1库内")
    assert template.vehicle_templates["R"].preferred_track == "修2库内"


def test_soft_target_template_reports_cun4bei_close_door_demand():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存1", "trackDistance": 120.0},
        ],
        "vehicleInfo": [
            _vehicle("CD", "存5北", "存4北", order=1, attrs="关门车"),
            _vehicle("N1", "存1", "存4北", order=1),
            _vehicle("N2", "存1", "存4北", order=2),
            _vehicle("N3", "存1", "存4北", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    template = compute_soft_target_template(normalized, state)

    assert template.cun4bei_template.unresolved_close_door_vehicle_nos == ["CD"]
    assert template.cun4bei_template.required_plain_prefix_count == 3
    assert template.cun4bei_template.available_plain_target_vehicle_count == 3
