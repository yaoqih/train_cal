from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.replay import build_initial_state, replay_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _native_direct_plan(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    detach_path_tracks: list[str],
) -> list[dict]:
    return [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": [source_track],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": source_track,
            "targetTrack": target_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": detach_path_tracks,
        },
    ]


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


def test_replay_native_attach_and_detach_move_front_block():
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
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["C1"],
            detach_path_tracks=["存5北", "机库"],
        ),
    )

    assert result.final_state.track_sequences["存5北"] == []
    assert result.final_state.track_sequences["机库"] == ["C1"]
    assert len(result.snapshots) == 3


def test_replay_attach_releases_source_spot_assignment():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C_ATTACH",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    assert initial.spot_assignments == {"C_ATTACH": "101"}

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "修1库内",
                "targetTrack": "修1库内",
                "vehicleNos": ["C_ATTACH"],
                "pathTracks": ["修1库内"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["修1库内"] == []
    assert result.final_state.loco_carry == ("C_ATTACH",)
    assert result.final_state.spot_assignments == {}


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
        _native_direct_plan(
            source_track="存5北",
            target_track="修1库内",
            vehicle_nos=["C2"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        ),
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
        _native_direct_plan(
            source_track="存5北",
            target_track="调棚",
            vehicle_nos=["C3"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
        ),
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
        _native_direct_plan(
            source_track="存5北",
            target_track="调棚",
            vehicle_nos=["C4"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
        ),
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["调棚"] == ["C4"]
    assert result.final_state.spot_assignments["C4"] == "调棚:PRE_REPAIR"
