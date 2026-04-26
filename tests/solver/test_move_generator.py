from pathlib import Path
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import (
    _collect_real_hook_access_blocker_attach_requests,
    generate_goal_moves,
    generate_real_hook_moves,
)
from fzed_shunting.verify.replay import ReplayState, build_initial_state


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


def test_generate_goal_moves_prefers_1_2_for_short_random_depot_vehicle():
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

    assert [move.target_track for move in moves] == ["修1库内", "修2库内"]


def test_generate_goal_moves_allows_3_4_fallback_only_when_1_2_are_full_for_short_random_depot_vehicle():
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
                "vehicleNo": "OCC_101",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_102",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "102",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_103",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "103",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_104",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "104",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "5",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_105",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "105",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_201",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "201",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_202",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "202",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_203",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "203",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_204",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "204",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "5",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_205",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "205",
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

    assert [move.target_track for move in moves] == ["修3库内", "修4库内"]


def test_generate_goal_moves_keeps_short_random_depot_on_preferred_when_1_2_have_releaseable_occupants():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
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
            *[
                {
                    "trackName": "修1库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"R1_{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
            *[
                {
                    "trackName": "修2库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"R2_{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
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

    assert [move.target_track for move in moves] == ["修1库内", "修2库内"]


def test_generate_goal_moves_counts_track_mode_depot_occupants_against_random_depot_capacity():
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
                "vehicleNo": "DEPOT_TRACK_FULL",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "修1库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"TRACK1_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetMode": "TRACK",
                    "targetTrack": "修1库内",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
            *[
                {
                    "trackName": "修2库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"TRACK2_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetMode": "TRACK",
                    "targetTrack": "修2库内",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
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

    assert [move.target_track for move in moves] == ["修3库内", "修4库内"]


def test_random_depot_effective_targets_are_not_rewritten_by_overflow_template():
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
            *[
                {
                    "trackName": "修1库内" if idx <= 5 else "修2库内",
                    "order": str(idx if idx <= 5 else idx - 5),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"PREF_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 11)
            ],
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OVERFLOW_STABLE",
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
                "vehicleNo": "OVERFLOW_PENDING",
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
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}

    stable_vehicle = vehicle_by_no["OVERFLOW_STABLE"]
    pending_vehicle = vehicle_by_no["OVERFLOW_PENDING"]

    assert goal_is_satisfied(
        stable_vehicle,
        track_name="修3库内",
        state=state,
        plan_input=normalized,
    )
    assert goal_effective_allowed_tracks(
        stable_vehicle,
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]
    assert goal_effective_allowed_tracks(
        pending_vehicle,
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]
    assert goal_effective_allowed_tracks(
        vehicle_by_no["PREF_1"],
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]


def test_generate_real_hook_moves_keeps_dirty_goal_target_as_soft_candidate():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LONG_INBOUND",
                "repairProcess": "厂修",
                "vehicleLength": 17.6,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL_MUST_LEAVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": [], "修3库内": ["TAIL_MUST_LEAVE"]},
            "loco_track_name": "存5北",
            "loco_carry": ("LONG_INBOUND",),
            "spot_assignments": {},
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(move.target_track == "修4库内" for move in moves)
    assert any(
        move.action_type == "DETACH"
        and move.target_track == "修3库内"
        and tuple(move.vehicle_nos) == ("LONG_INBOUND",)
        for move in moves
    )

    tail_vehicle = vehicle_by_no["TAIL_MUST_LEAVE"]
    assert not goal_is_satisfied(
        tail_vehicle,
        track_name="修3库内",
        state=state,
        plan_input=normalized,
    )


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
                "trackName": "修1库内",
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


def test_generate_real_hook_moves_keeps_interfering_identity_track_attachable():
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
                "vehicleNo": "PATH_NEED_NATIVE",
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
                "vehicleNo": "PATH_BLOCK_NATIVE",
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

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存5南"
        and move.vehicle_nos == ["PATH_BLOCK_NATIVE"]
        for move in moves
    )


def test_generate_real_hook_moves_keeps_capacity_eviction_attachable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 28.6},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_OK_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_OK_2",
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
                "vehicleNo": "DEPOT_NEED_NATIVE",
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

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "机库"
        and move.vehicle_nos == ["DEPOT_OK_1"]
        for move in moves
    )


def test_generate_real_hook_moves_keeps_spot_eviction_attachable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_OCC_NATIVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_NEED_NATIVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修1库内"
        and move.vehicle_nos == ["SPOT_OCC_NATIVE"]
        for move in moves
    )


def test_generate_real_hook_moves_skips_extra_attach_without_detach_group_synergy():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_B",
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
                "vehicleNo": "BLOCK_A",
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
    state = ReplayState(
        track_sequences={"存5北": ["BLOCK_A"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY_A", "CARRY_B"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    attach_moves = [move for move in moves if move.action_type == "ATTACH"]
    assert ("BLOCK_A",) not in [tuple(move.vehicle_nos) for move in attach_moves]


def test_generate_real_hook_moves_keeps_extra_attach_when_it_reduces_detach_groups():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_C",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_D",
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
                "vehicleNo": "BLOCK_B",
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
    state = ReplayState(
        track_sequences={"存5北": ["BLOCK_B"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY_C", "CARRY_D"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    attach_moves = [move for move in moves if move.action_type == "ATTACH"]
    assert ("BLOCK_B",) in [tuple(move.vehicle_nos) for move in attach_moves]


def test_generate_real_hook_moves_keeps_attach_even_when_combined_carry_exceeds_equivalent_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": str(index),
                "vehicleModel": "C70",
                "vehicleNo": f"EQC{index:02d}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车" if index <= 2 else "",
            }
            for index in range(1, 15)
        ]
        + [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "EQ_OVER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["EQ_OVER"]},
        loco_track_name="修1库内",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=tuple(f"EQC{index:02d}" for index in range(1, 15)),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("EQ_OVER",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_attach_even_when_combined_carry_has_three_heavy_cars():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "C70E",
                "vehicleNo": "HC1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
            {
                "trackName": "修1库内",
                "order": "2",
                "vehicleModel": "C70E",
                "vehicleNo": "HC2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70E",
                "vehicleNo": "HC3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["HC3"]},
        loco_track_name="修1库内",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("HC1", "HC2"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("HC3",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_extra_attach_with_multiple_unweighed_needweigh_when_detach_remains_feasible():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "GQ70",
                "vehicleNo": "WC1",
                "repairProcess": "段修",
                "vehicleLength": 13.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "GQ70",
                "vehicleNo": "WC2",
                "repairProcess": "段修",
                "vehicleLength": 13.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["WC2"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("WC1",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("WC2",)
        for move in moves
    )


def test_generate_real_hook_moves_compresses_empty_carry_attach_prefixes_to_detach_group_frontier():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存1", "trackDistance": 113},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
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
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "S1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "targetMode": "SPOT",
                "targetSpotCode": "1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "Y1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)
    attach_prefixes = sorted(
        [
            tuple(move.vehicle_nos)
            for move in moves
            if move.action_type == "ATTACH" and move.source_track == "存5北"
        ],
        key=len,
    )

    assert attach_prefixes == [
        ("P1", "P2"),
        ("P1", "P2", "S1"),
        ("P1", "P2", "S1", "Y1"),
    ]


def test_generate_real_hook_moves_keeps_short_random_depot_vehicle_attachable_from_fallback_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "SHORT_FALLBACK",
                "repairProcess": "段修",
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

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修4库内"
        and tuple(move.vehicle_nos) == ("SHORT_FALLBACK",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_wash_north_blocks_access_to_wash_south():
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "洗南"
        and tuple(move.vehicle_nos) == ("WASH_TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_depot_outer_blocks_access_to_inner():
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "修3库内"
        and tuple(move.vehicle_nos) == ("DEPOT_TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_generic_intermediate_track_blocks_access():
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "存4北"
        and tuple(move.vehicle_nos) == ("TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_access_blocker_attachable_when_target_source_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "存5北"
        and tuple(move.vehicle_nos) == ("ACCESS_TARGET",)
        for move in moves
    )
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存5南"
        and tuple(move.vehicle_nos) == ("ACCESS_BLOCKER",)
        for move in moves
    )


def test_collect_real_hook_access_blocker_requests_is_limited_to_blocking_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )

    assert requests == {"存5南": {1}}


def test_collect_real_hook_access_blocker_requests_clears_whole_blocking_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "机棚", "trackDistance": 105.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "机棚",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"ACCESS_BLOCK_{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "机棚",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 4)
            ],
        ],
        "locoTrackName": "修4库内",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )
    moves = generate_real_hook_moves(normalized, state, master=master)

    assert requests == {"机棚": {3}}
    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "机棚"
        and tuple(move.vehicle_nos) == ("ACCESS_BLOCK_1",)
        for move in moves
    )
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "机棚"
        and tuple(move.vehicle_nos)
        == ("ACCESS_BLOCK_1", "ACCESS_BLOCK_2", "ACCESS_BLOCK_3")
        for move in moves
    )


def test_collect_real_hook_access_blocker_requests_waits_while_normal_attach_exists():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "READY_SOURCE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )

    assert requests == {}


def test_generate_real_hook_moves_skips_loaded_attach_when_access_length_exceeds_l1_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "存5北",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"LOAD_A_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.0,
                    "targetTrack": "存2",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 16)
            ],
            {
                "trackName": "存2",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LOAD_B",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {
                "存5北": [],
                "存2": ["LOAD_B"],
            },
            "loco_track_name": "存5北",
            "loco_carry": tuple(f"LOAD_A_{i}" for i in range(1, 16)),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "存2"
        and tuple(move.vehicle_nos) == ("LOAD_B",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_full_same_track_detach_when_it_can_finish_carry():
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
                "vehicleNo": "SAME_FULL_A",
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
                "vehicleNo": "SAME_FULL_B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": ("SAME_FULL_A", "SAME_FULL_B"),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.source_track == "存5北"
        and move.target_track == "存5北"
        and tuple(move.vehicle_nos) == ("SAME_FULL_A", "SAME_FULL_B")
        for move in moves
    )


def test_generate_real_hook_moves_skips_detach_when_full_carry_exceeds_route_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"DETACH_FULL_{i}",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for i in range(1, 16)
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": tuple(f"DETACH_FULL_{i}" for i in range(1, 16)),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "存2"
        and tuple(move.vehicle_nos) == ("DETACH_FULL_1",)
        for move in moves
    )
