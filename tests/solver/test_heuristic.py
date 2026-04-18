from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.heuristic import (
    compute_admissible_heuristic,
    compute_heuristic_breakdown,
    make_state_heuristic,
)
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _base_payload(vehicles: list[dict]) -> dict:
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "洗北", "trackDistance": 100},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def test_heuristic_zero_when_all_vehicles_at_goal():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    assert compute_admissible_heuristic(normalized, initial) == 0


def test_heuristic_counts_misplaced_vehicle():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_misplaced == 1
    assert breakdown.value >= 1


def test_heuristic_lower_bound_respects_optimal_single_hook():
    from fzed_shunting.solver.astar_solver import solve_with_simple_astar

    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    h0 = compute_admissible_heuristic(normalized, initial)
    plan = solve_with_simple_astar(normalized, initial, master=master)
    assert h0 <= len(plan), f"heuristic {h0} must be <= optimal hooks {len(plan)}"


def test_h_weigh_strengthens_when_weigh_outstanding():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "H1",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_weigh == 1


def test_h_weigh_zero_when_weigh_already_done():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "H1",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    state = ReplayState(
        track_sequences=dict(initial.track_sequences),
        loco_track_name=initial.loco_track_name,
        weighed_vehicle_nos={"H1"},
        spot_assignments=dict(initial.spot_assignments),
    )
    breakdown = compute_heuristic_breakdown(normalized, state)
    assert breakdown.h_weigh == 0


def test_h_blocking_strengthens_when_target_has_blocker():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "GOAL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_blocking >= 1


def test_make_state_heuristic_matches_compute_admissible_heuristic():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "E2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    stateful = make_state_heuristic(normalized)
    assert stateful(initial) == compute_admissible_heuristic(normalized, initial)
