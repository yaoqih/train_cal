from pathlib import Path
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.move_generator import generate_goal_moves
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_generate_goal_moves_from_north_prefix_only():
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
                "vehicleNo": "D1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "D2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    move_blocks = [tuple(move.vehicle_nos) for move in moves]
    assert ("D1",) in move_blocks
    assert ("D2",) not in move_blocks


def test_generate_goal_moves_keeps_only_longest_feasible_prefix_for_single_target_goal():
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
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "P2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert [tuple(move.vehicle_nos) for move in moves] == [("P1", "P2")]


def test_generate_goal_moves_keeps_shorter_prefix_when_longest_single_target_prefix_does_not_fit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 20.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "Q1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "Q2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert [tuple(move.vehicle_nos) for move in moves] == [("Q1",)]


def test_generate_goal_moves_keeps_shorter_prefix_for_single_target_goal_when_target_track_is_not_empty():
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
                "vehicleNo": "R1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "R2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "R_OCC",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state)
        if move.source_track == "存5北"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("R1", "R2"), ("R1",)]


def test_generate_goal_moves_keeps_shorter_prefix_for_single_target_goal_from_temporary_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 120.0},
            {"trackName": "修3库外", "trackDistance": 120.0},
            {"trackName": "机库", "trackDistance": 120.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临3",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master, route_oracle=RouteOracle(master))
        if move.source_track == "临3" and move.target_track == "修3库外"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("TMP1", "TMP2"), ("TMP1",)]


def test_generate_goal_moves_keeps_only_longest_prefix_for_three_car_temporary_same_goal_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 120.0},
            {"trackName": "修3库外", "trackDistance": 120.0},
            {"trackName": "机库", "trackDistance": 120.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临3",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master, route_oracle=RouteOracle(master))
        if move.source_track == "临3" and move.target_track == "修3库外"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("TMP3_1", "TMP3_2", "TMP3_3")]


def test_generate_goal_moves_keeps_shorter_prefixes_when_same_goal_front_block_does_not_consume_source_track():
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
                "vehicleNo": "M1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "M2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "M3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "M4",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state)
        if move.source_track == "存5北" and move.target_track == "机库"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [
        ("M1", "M2", "M3"),
        ("M1", "M2"),
        ("M1",),
    ]


def test_generate_goal_moves_respects_allowed_target_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修2库外", "trackDistance": 49.3},
            {"trackName": "修3库外", "trackDistance": 49.3},
            {"trackName": "修4库外", "trackDistance": 49.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    targets = {move.target_track for move in moves}
    assert targets == {"修1库外", "修2库外", "修3库外", "修4库外"}


def test_generate_goal_moves_keeps_all_random_depot_targets_for_direct_rebalancing():
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
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_PREF",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert {move.target_track for move in moves} == {"修1库内", "修2库内", "修3库内", "修4库内"}


def test_generate_goal_moves_orders_random_depot_targets_by_current_occupancy_desc():
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
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_ORDER",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_A",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_B",
                "repairProcess": "厂修",
                "vehicleLength": 19.0,
                "targetTrack": "修3库内",
                "isSpotting": "301",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_C",
                "repairProcess": "厂修",
                "vehicleLength": 24.0,
                "targetTrack": "修4库内",
                "isSpotting": "401",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master)
        if move.source_track == "存5北"
    ]

    assert [move.target_track for move in moves] == ["修4库内", "修3库内", "修1库内", "修2库内"]


def test_generate_goal_moves_supports_oil_and_shot_targets_when_missing_segment_is_40m_placeholder():
    master = load_master_data(DATA_DIR)
    shot_payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "抛", "trackDistance": 131.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D4",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "抛",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    oil_payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "油", "trackDistance": 124},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D4O",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "油",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    shot_normalized = normalize_plan_input(shot_payload, master)
    shot_state = build_initial_state(shot_normalized)
    oil_normalized = normalize_plan_input(oil_payload, master)
    oil_state = build_initial_state(oil_normalized)

    shot_moves = generate_goal_moves(shot_normalized, shot_state, master=master)
    oil_moves = generate_goal_moves(oil_normalized, oil_state, master=master)

    assert len(shot_moves) == 1
    assert shot_moves[0].target_track == "抛"
    assert len(oil_moves) == 1
    assert oil_moves[0].target_track == "油"


def test_generate_goal_moves_filters_l1_overflow():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D5",
                "repairProcess": "段修",
                "vehicleLength": 200.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert moves == []


def test_generate_goal_moves_filters_interference_on_intermediate_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D6",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D7",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert moves == []


def test_generate_goal_moves_can_emit_temporary_clear_move_for_front_blocker():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCK_FRONT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "NEED_OUT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5北"
        and move.target_track == "临1"
        and move.vehicle_nos == ["BLOCK_FRONT"]
        for move in moves
    )


def test_generate_goal_moves_skips_pure_random_depot_rebalancing_when_block_is_already_satisfied():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RANDOM_OK",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OTHER_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert not any(move.source_track == "修3库内" for move in moves)


def test_generate_goal_moves_keeps_random_depot_front_blocker_movable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RAND_FRONT",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "NEED_C4B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "修3库内"
        and move.vehicle_nos == ["RAND_FRONT"]
        and move.target_track in {"修1库内", "修2库内", "修4库内"}
        for move in moves
    )


def test_generate_goal_moves_can_emit_temporary_clear_move_to_cun4nan():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C4N_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "C4N_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5北"
        and move.target_track == "存4南"
        and move.vehicle_nos == ["C4N_BLOCK"]
        for move in moves
    )


def test_generate_goal_moves_can_emit_temporary_clear_move_for_interfering_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PATH_NEED",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PATH_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5南"
        and move.target_track == "临1"
        and move.vehicle_nos == ["PATH_BLOCK"]
        for move in moves
    )


def test_generate_goal_moves_records_debug_stats_for_direct_and_staging_moves():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DBG_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "DBG_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    debug_stats: dict = {}

    moves = generate_goal_moves(normalized, state, master=master, debug_stats=debug_stats)

    assert moves
    assert debug_stats["total_moves"] == len(moves)
    assert debug_stats["staging_moves"] >= 1
    assert debug_stats["moves_by_target"]["临1"] >= 1
    assert debug_stats["moves_by_source"]["存5北"] == len(moves)
    assert debug_stats["moves_by_block_size"][1] >= 1


def test_generate_goal_moves_limits_front_blocker_staging_targets_to_nearest_temporaries():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LIMIT_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "LIMIT_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert {move.target_track for move in staging_moves} == {"临1", "临2"}

def test_generate_goal_moves_prunes_heavy_equivalent_block_before_staging_without_l1():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MIX_PATH_NEED",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "存5南",
                    "order": str(i),
                    "vehicleModel": "敞车",
                    "vehicleNo": f"MIX_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 4.0,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "重车" if i <= 2 else "",
                }
                for i in range(1, 16)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5南"]

    assert any(move.target_track == "临1" for move in staging_moves)
    assert {len(move.vehicle_nos) for move in staging_moves} == {14}


def test_generate_goal_moves_falls_back_to_smaller_interfering_prefix_when_largest_block_cannot_stage():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 171.6},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PATH_NEED_FALLBACK",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "存5南",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"FALLBACK_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 13)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5南"]

    assert any(
        move.target_track == "临4" and len(move.vehicle_nos) == 6
        for move in staging_moves
    )


def test_generate_goal_moves_can_fallback_to_storage_cache_when_no_temporary_track_fits():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "存5北",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"STORAGE_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 12)
            ],
            {
                "trackName": "存5北",
                "order": "12",
                "vehicleModel": "棚车",
                "vehicleNo": "STORAGE_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert any(
        move.target_track == "存2" and len(move.vehicle_nos) == 11
        for move in staging_moves
    )


def test_generate_goal_moves_does_not_add_storage_cache_when_temporary_stage_is_already_feasible():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TEMP_FIRST_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TEMP_FIRST_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert {move.target_track for move in staging_moves} == {"临1"}

def test_generate_goal_moves_reuses_injected_route_oracle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "INJECTED_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "INJECTED_GO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)

    with patch("fzed_shunting.solver.move_generator.RouteOracle", side_effect=AssertionError("unexpected constructor call")):
        moves = generate_goal_moves(
            normalized,
            state,
            master=master,
            route_oracle=route_oracle,
        )

    assert any(move.target_track == "临1" for move in moves)
