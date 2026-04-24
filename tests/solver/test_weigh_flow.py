from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_solver_inserts_jiku_step_for_weigh_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "W1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(normalized, initial, master=master)
    hook_plan = [
        {
            "hookNo": idx,
            "actionType": item.action_type,
            "sourceTrack": item.source_track,
            "targetTrack": item.target_track,
            "vehicleNos": item.vehicle_nos,
            "pathTracks": item.path_tracks,
        }
        for idx, item in enumerate(plan, start=1)
    ]
    report = verify_plan(master, normalized, hook_plan)

    assert [item["actionType"] for item in hook_plan] == ["ATTACH", "DETACH", "ATTACH", "DETACH"]
    assert [item["targetTrack"] for item in hook_plan if item["actionType"] == "DETACH"] == ["机库", "存4北"]
    assert report.is_valid is True


def test_verifier_rejects_direct_move_for_weigh_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "W2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["W2"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "存4北",
                "vehicleNos": ["W2"],
                "pathTracks": ["存5北", "存4北"],
            },
        ],
    )

    assert report.is_valid is False
    assert any("weigh" in error.lower() for error in report.errors)
