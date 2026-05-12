from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.goal_logic import (
    goal_can_use_fallback_now,
    goal_effective_allowed_tracks,
    goal_is_satisfied,
)
from fzed_shunting.solver import goal_logic
from fzed_shunting.solver.purity import compute_state_purity
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_goal_effective_allowed_tracks_reuses_fallback_decision_for_same_state(monkeypatch):
    master = load_master_data(DATA_DIR)
    vehicles = [
        {
            "trackName": "存5北",
            "order": str(index),
            "vehicleModel": "棚车",
            "vehicleNo": f"DEPOT_{index}",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
        for index in range(1, 4)
    ]
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = normalized.vehicles[0]
    calls = 0
    real = goal_logic._preferred_tracks_lack_group_spot_capacity

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(goal_logic, "_preferred_tracks_lack_group_spot_capacity", wrapped)

    for _ in range(5):
        assert goal_effective_allowed_tracks(
            vehicle,
            state=state,
            plan_input=normalized,
        )

    assert calls == 1


def test_snapshot_goal_can_use_fallback_when_preferred_track_has_no_length_capacity():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 28.6},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_INBOUND",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_OCC1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_OCC2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "SNAP_INBOUND")

    assert goal_can_use_fallback_now(vehicle, state=state, plan_input=normalized)
    assert goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=normalized,
    ) == ["存5南", "存5北"]


def test_snapshot_goal_keeps_preferred_track_when_length_capacity_remains():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 42.9},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_INBOUND",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_OCC1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "SNAP_INBOUND")

    assert goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=normalized,
    ) == ["存5南"]


def test_random_depot_can_use_fallback_when_preferred_route_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修2库外", "trackDistance": 49.3},
            {"trackName": "修3库外", "trackDistance": 49.3},
            {"trackName": "修4库外", "trackDistance": 49.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_PENDING",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCK_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCK_2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "DEPOT_PENDING")

    assert goal_can_use_fallback_now(
        vehicle,
        state=state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    )
    assert goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    ) == ["修1", "修2", "修3", "修4"]


def test_random_depot_can_use_fallback_when_group_demand_exceeds_preferred_spots():
    master = load_master_data(DATA_DIR)
    vehicles = [
        {
            "trackName": "存5北",
            "order": str(index),
            "vehicleModel": "棚车",
            "vehicleNo": f"DEPOT_{index}",
            "repairProcess": "厂修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
        for index in range(1, 4)
    ]
    vehicles.extend(
        [
            {
                "trackName": "修1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_2",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1",
                "isSpotting": "102",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_3",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1",
                "isSpotting": "103",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_4",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1",
                "isSpotting": "104",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_5",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2",
                "isSpotting": "201",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_6",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2",
                "isSpotting": "202",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_7",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2",
                "isSpotting": "203",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC_8",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2",
                "isSpotting": "204",
                "vehicleAttributes": "",
            },
        ]
    )
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修3", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "DEPOT_1")

    assert goal_can_use_fallback_now(vehicle, state=state, plan_input=normalized)
    assert goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=normalized,
    ) == ["修1", "修2", "修3", "修4"]


def test_end_snapshot_fallback_track_is_complete_even_when_preferred_has_capacity():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_SOFT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle = normalized.vehicles[0]

    assert goal_effective_allowed_tracks(
        vehicle,
        state=state,
        plan_input=normalized,
    ) == ["存5南"]
    assert goal_is_satisfied(
        vehicle,
        track_name="存5北",
        state=state,
        plan_input=normalized,
    )
    purity = compute_state_purity(normalized, state)
    assert purity.unfinished_count == 0
    assert purity.preferred_violation_count == 1


def test_random_depot_fallback_track_is_complete_even_when_preferred_can_rebalance():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1", "trackDistance": 151.7},
            {"trackName": "修2", "trackDistance": 151.7},
            {"trackName": "修4", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PREF_OCC",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修4",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "FALLBACK_OK",
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
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "FALLBACK_OK")

    assert not goal_can_use_fallback_now(vehicle, state=state, plan_input=normalized)
    assert goal_is_satisfied(
        vehicle,
        track_name="修4",
        state=state,
        plan_input=normalized,
    )
    purity = compute_state_purity(normalized, state)
    assert purity.unfinished_count == 0
    assert purity.preferred_violation_count == 1


def test_work_spotting_goal_uses_south_rank_not_spot_assignment():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "WORK_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "洗南",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
            {
                "trackName": "洗南",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "SOUTH_PAD",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "洗南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "WORK_TARGET")
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"洗南": ["WORK_TARGET", "SOUTH_PAD"]},
            "spot_assignments": {},
        }
    )

    assert goal_is_satisfied(
        vehicle,
        track_name="洗南",
        state=state,
        plan_input=normalized,
    )

    wrong_state = state.model_copy(update={"track_sequences": {"洗南": ["WORK_TARGET"]}})

    assert not goal_is_satisfied(
        vehicle,
        track_name="洗南",
        state=wrong_state,
        plan_input=normalized,
    )


def test_work_exact_position_goal_uses_final_north_rank():
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
                "vehicleNo": "NORTH_PAD",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "WORK_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "2",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle = next(item for item in normalized.vehicles if item.vehicle_no == "WORK_TARGET")
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"调棚": ["NORTH_PAD", "WORK_TARGET"]},
            "spot_assignments": {},
        }
    )

    assert goal_is_satisfied(
        vehicle,
        track_name="调棚",
        state=state,
        plan_input=normalized,
    )


def test_explicit_work_spot_goal_uses_slot_capacity_not_required_north_rank():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "WORK_SLOT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "洗南",
                "targetSpotCode": "3",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle = normalized.vehicles[0]
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"洗南": ["WORK_SLOT"]},
            "spot_assignments": {},
        }
    )

    assert goal_is_satisfied(
        vehicle,
        track_name="洗南",
        state=state,
        plan_input=normalized,
    )

    blocked_slot_state = state.model_copy(
        update={"track_sequences": {"洗南": ["A", "B", "C", "WORK_SLOT"]}}
    )

    assert not goal_is_satisfied(
        vehicle,
        track_name="洗南",
        state=blocked_slot_state,
        plan_input=normalized,
    )


def test_explicit_work_spot_goals_conflict_when_same_slot_is_reused():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "SLOT_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "洗南",
                "targetSpotCode": "2",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "罐车",
                "vehicleNo": "SLOT_B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "洗南",
                "targetSpotCode": "2",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"洗南": ["SLOT_A", "SLOT_B"]},
            "spot_assignments": {},
        }
    )

    assert not any(
        goal_is_satisfied(
            vehicle,
            track_name="洗南",
            state=state,
            plan_input=normalized,
        )
        for vehicle in normalized.vehicles
    )
