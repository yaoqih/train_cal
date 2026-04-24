from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.move_generator import (
    _violates_close_door_hook_rule,
    generate_goal_moves,
)
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _normalize(vehicles: list[dict]):
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    return master, normalized


def test_close_door_vehicle_cannot_land_in_existing_4bei_first_three_via_single_hook():
    master, normalized = _normalize(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            }
        ]
    )
    initial = build_initial_state(normalized)
    moves = generate_goal_moves(normalized, initial, master=master)
    target_4bei = [move for move in moves if move.target_track == "存4北"]
    assert not target_4bei, (
        "solitary close-door vehicle going to 存4北 would land at position 1 — must be pruned"
    )


def test_close_door_vehicle_allowed_in_position_4_and_beyond():
    # PREPEND model: CD1 lands at index 0 initially, but 3 pending vehicles
    # (NORMAL1-3, not yet on 存4北) will each prepend after CD1, pushing it to
    # index 3 (position 4). So pending=3 ≥ 3 → NOT pruned at move-gen time.
    master, normalized = _normalize(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "NORMAL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "NORMAL2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "NORMAL3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    initial = build_initial_state(normalized)
    vehicle_by_no = {v.vehicle_no: v for v in normalized.vehicles}
    # 3 pending vehicles → CD1 will reach index 3 (position 4) → allowed
    assert not _violates_close_door_hook_rule(
        ["CD1"], "存4北", vehicle_by_no, initial
    ), "close-door landing on 存4北 must not be pruned when 3 pending vehicles exist"


def test_close_door_not_going_to_4bei_big_block_still_rejected():
    from unittest.mock import MagicMock

    vehicle = MagicMock()
    vehicle.is_close_door = True
    vehicle.goal = MagicMock(target_mode="TRACK", target_track="存1", allowed_target_tracks=["存1"])
    vehicle_by_no = {"CD1": vehicle}
    block = ["CD1"] + [f"X{i}" for i in range(11)]
    for ident in block[1:]:
        vehicle_by_no[ident] = MagicMock(is_close_door=False, goal=vehicle.goal)
    state = ReplayState(
        track_sequences={}, loco_track_name="机库", weighed_vehicle_nos=set(), spot_assignments={}
    )
    assert _violates_close_door_hook_rule(block, "存1", vehicle_by_no, state) is True


def test_close_door_not_going_to_4bei_small_block_not_rejected():
    from unittest.mock import MagicMock

    vehicle = MagicMock(is_close_door=True)
    vehicle.goal = MagicMock(target_mode="TRACK", target_track="存1", allowed_target_tracks=["存1"])
    vehicle_by_no = {"CD1": vehicle}
    state = ReplayState(
        track_sequences={}, loco_track_name="机库", weighed_vehicle_nos=set(), spot_assignments={}
    )
    assert _violates_close_door_hook_rule(["CD1"], "存1", vehicle_by_no, state) is False
