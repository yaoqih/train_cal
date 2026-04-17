from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.replay import build_initial_state, replay_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_build_initial_state_orders_vehicles_north_to_south():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "B2",
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
                "vehicleNo": "B1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "",
    }
    normalized = normalize_plan_input(payload, master)

    state = build_initial_state(normalized)

    assert state.track_sequences["存5北"] == ["B1", "B2"]


def test_replay_put_action_moves_front_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["C1"],
                "pathTracks": ["存5北", "机库"],
            }
        ],
    )

    assert result.final_state.track_sequences["存5北"] == []
    assert result.final_state.track_sequences["机库"] == ["C1"]
    assert len(result.snapshots) == 2


def test_replay_assigns_exact_depot_spot_when_moving_into_depot():
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
                "vehicleNo": "C2",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "修1库内",
                "vehicleNos": ["C2"],
                "pathTracks": ["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["修1库内"] == ["C2"]
    assert result.final_state.spot_assignments["C2"] == "101"


def test_replay_assigns_dispatch_work_spot_when_moving_into_work_area():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "是",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "调棚",
                "vehicleNos": ["C3"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["调棚"] == ["C3"]
    assert result.final_state.spot_assignments["C3"] == "调棚:1"


def test_replay_assigns_dispatch_pre_repair_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C4",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "调棚",
                "vehicleNos": ["C4"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["调棚"] == ["C4"]
    assert result.final_state.spot_assignments["C4"] == "调棚:PRE_REPAIR"
