from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.move_generator import generate_goal_moves
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_long_depot_random_vehicle_prefers_3_and_4_depots():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "平车",
                "vehicleNo": "R1",
                "repairProcess": "厂修",
                "vehicleLength": 18.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert {move.target_track for move in moves} == {"修3库内", "修4库内"}


def test_verifier_rejects_attach_to_wash_south_when_wash_north_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "洗北", "trackDistance": 71.6},
            {"trackName": "洗南", "trackDistance": 90.0},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "洗南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WASH_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "洗北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WASH_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "洗北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临3",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "洗南",
                "targetTrack": "洗南",
                "vehicleNos": ["WASH_TARGET"],
                "pathTracks": ["洗南"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "洗南",
                "targetTrack": "存1",
                "vehicleNos": ["WASH_TARGET"],
                "pathTracks": ["洗南", "洗北", "临3", "机棚", "机北", "渡6", "存1"],
            },
        ],
        initial_state_override=initial,
    )

    assert report.is_valid is False
    hook1 = next(item for item in report.hook_reports if item.hook_no == 1)
    assert any("洗北" in error for error in hook1.errors)


def test_verifier_rejects_attach_to_depot_inner_when_depot_outer_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临4", "trackDistance": 81.4},
            {"trackName": "修3库外", "trackDistance": 80.0},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临4",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "修3库内",
                "targetTrack": "修3库内",
                "vehicleNos": ["DEPOT_TARGET"],
                "pathTracks": ["修3库内"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "修3库内",
                "targetTrack": "存1",
                "vehicleNos": ["DEPOT_TARGET"],
                "pathTracks": ["修3库内", "修3库外", "渡13", "渡12", "联7", "渡10", "渡9", "预修", "渡7", "存1"],
            },
        ],
        initial_state_override=initial,
    )

    assert report.is_valid is False
    hook1 = next(item for item in report.hook_reports if item.hook_no == 1)
    assert any("修3库外" in error for error in hook1.errors)


def test_verifier_rejects_attach_when_any_intermediate_track_blocks_loco_access():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临4", "trackDistance": 81.4},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存4南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临4",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存4北",
                "targetTrack": "存4北",
                "vehicleNos": ["TARGET"],
                "pathTracks": ["存4北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存4北",
                "targetTrack": "存1",
                "vehicleNos": ["TARGET"],
                "pathTracks": ["存4北", "渡1", "渡2", "临1", "存1"],
            },
        ],
        initial_state_override=initial,
    )

    assert report.is_valid is False
    hook1 = next(item for item in report.hook_reports if item.hook_no == 1)
    hook2 = next(item for item in report.hook_reports if item.hook_no == 2)
    assert hook1.blocking_tracks == []
    assert hook2.blocking_tracks == []
    assert report.global_errors == ["Vehicle BLOCK final track/spot/weigh state does not satisfy goal"]


def test_work_area_capacity_blocks_extra_move():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "调棚",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"T{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "调棚",
                    "isSpotting": "是",
                    "vehicleAttributes": "",
                }
                for i in range(1, 5)
            ],
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "T5",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert moves == []


def test_verifier_rejects_close_door_vehicle_at_front_of_cun4bei():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CLOSE1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
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
                "vehicleNos": ["CLOSE1"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "存4北",
                "vehicleNos": ["CLOSE1"],
                "pathTracks": ["存5北", "存4北"],
            },
        ],
    )

    assert report.is_valid is False
    assert any("close-door" in error.lower() for error in report.errors)


def test_move_generator_limits_empty_vehicle_block_size_to_20():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 500},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"EMPTY{i}",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for i in range(1, 22)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert max(len(move.vehicle_nos) for move in moves) == 20


def test_move_generator_limits_weigh_hook_to_one_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 200},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WEIGH1",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "WEIGH2",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    move_blocks = [tuple(move.vehicle_nos) for move in moves]

    assert ("WEIGH1",) in move_blocks
    assert ("WEIGH1", "WEIGH2") not in move_blocks


def test_move_generator_skips_occupied_exact_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_OCC",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_NEW",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert moves == []


def test_move_generator_respects_random_depot_spot_availability():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "修1库内",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"FULL{i}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": str(100 + i),
                    "vehicleAttributes": "",
                }
                for i in range(1, 6)
            ],
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RAND1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert "修1库内" not in {move.target_track for move in moves}
    assert "修2库内" in {move.target_track for move in moves}
