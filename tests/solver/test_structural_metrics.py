from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.structural_metrics import (
    summarize_plan_shape,
    compute_structural_metrics,
)
from fzed_shunting.solver.capacity_release import compute_capacity_release_plan
from fzed_shunting.solver.constructive import _score_native_move
from fzed_shunting.solver.move_generator import _candidate_targets
from fzed_shunting.solver.state import _apply_move
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _payload(vehicles: list[dict], track_distances: dict[str, float] | None = None) -> dict:
    distances = {
        "存5北": 367,
        "存4北": 317.8,
        "存1": 113,
        "临1": 81.4,
        "洗南": 88.7,
        "修1库内": 151.7,
        "修2库内": 151.7,
        "修3库内": 151.7,
        "修4库内": 151.7,
        "机库": 71.6,
    }
    if track_distances:
        distances.update(track_distances)
    return {
        "trackInfo": [
            {"trackName": track_name, "trackDistance": distance}
            for track_name, distance in distances.items()
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def _vehicle(
    vehicle_no: str,
    source: str,
    target: str,
    *,
    order: int = 1,
    length: float = 14.3,
    spotting: str = "",
) -> dict:
    return {
        "trackName": source,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": length,
        "targetTrack": target,
        "isSpotting": spotting,
        "vehicleAttributes": "",
    }


def _normalize(payload: dict):
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(payload, master)
    return normalized, build_initial_state(normalized)


def test_structural_metrics_zero_for_finished_state():
    normalized, initial = _normalize(_payload([_vehicle("A", "存4北", "存4北")]))

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.unfinished_count == 0
    assert metrics.staging_debt_count == 0
    assert metrics.front_blocker_count == 0
    assert metrics.goal_track_blocker_count == 0
    assert metrics.capacity_overflow_track_count == 0


def test_structural_metrics_counts_staging_debt_by_track():
    normalized, _ = _normalize(_payload([_vehicle("A", "临1", "存4北")]))
    state = ReplayState(
        track_sequences={"临1": ["A"], "存4北": [], "机库": []},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )

    metrics = compute_structural_metrics(normalized, state)

    assert metrics.unfinished_count == 1
    assert metrics.staging_debt_count == 1
    assert metrics.staging_debt_by_track == {"临1": 1}


def test_structural_metrics_counts_random_area_unfinished():
    normalized, initial = _normalize(_payload([_vehicle("A", "存5北", "大库")]))

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.unfinished_count == 1
    assert metrics.area_random_unfinished_count == 1
    assert metrics.work_position_unfinished_count == 0


def test_structural_metrics_counts_work_position_unfinished_separately_from_area():
    normalized, initial = _normalize(
        _payload([_vehicle("WASH", "存5北", "洗南", spotting="是")])
    )

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.unfinished_count == 1
    assert metrics.area_random_unfinished_count == 0
    assert metrics.work_position_unfinished_count == 1


def test_structural_metrics_counts_front_blocker_pressure():
    normalized, initial = _normalize(
        _payload(
            [
                _vehicle("BLOCK", "存5北", "存4北", order=1),
                _vehicle("SEEK", "存5北", "机库", order=2),
            ]
        )
    )

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.front_blocker_count == 1
    assert metrics.front_blocker_by_track == {"存5北": 1}


def test_structural_metrics_counts_satisfied_front_vehicle_blocking_unfinished_tail():
    normalized, initial = _normalize(
        _payload(
            [
                _vehicle("DONE", "存5北", "存5北", order=1),
                _vehicle("SEEK", "存5北", "机库", order=2),
            ]
        )
    )

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.front_blocker_count == 1
    assert metrics.front_blocker_by_track == {"存5北": 1}


def test_structural_metrics_counts_goal_track_blockers_and_capacity_debt():
    normalized, initial = _normalize(
        _payload(
            [
                _vehicle("ARRIVE", "存5北", "存1", order=1, length=15),
                _vehicle("BLOCK", "存1", "存4北", order=1, length=20),
            ],
            track_distances={"存1": 10},
        )
    )

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.goal_track_blocker_count == 1
    assert metrics.goal_track_blocker_by_track == {"存1": 1}
    assert metrics.capacity_overflow_track_count == 1
    assert metrics.capacity_debt_by_track == {"存1": 15.0}


def test_structural_metrics_does_not_double_count_vehicle_already_on_required_track():
    normalized, initial = _normalize(
        _payload(
            [
                _vehicle("DONE", "存1", "存1", order=1, length=8),
                _vehicle("ARRIVE", "存5北", "存1", order=1, length=5),
            ],
            track_distances={"存1": 10},
        )
    )

    metrics = compute_structural_metrics(normalized, initial)

    assert metrics.capacity_overflow_track_count == 1
    assert metrics.capacity_debt_by_track == {"存1": 3.0}


def test_structural_capacity_uses_initial_overlength_as_effective_capacity():
    normalized, initial = _normalize(
        _payload(
            [
                _vehicle("DONE1", "存1", "存1", order=1, length=8),
                _vehicle("DONE2", "存1", "存1", order=2, length=8),
            ],
            track_distances={"存1": 10},
        )
    )

    metrics = compute_structural_metrics(normalized, initial)
    release = compute_capacity_release_plan(normalized, initial)

    assert metrics.capacity_overflow_track_count == 0
    assert metrics.capacity_debt_by_track == {}
    assert release.facts_by_track["存1"].release_pressure_length == 0


def test_plan_shape_counts_staging_hooks_and_rehandles():
    moves = [
        HookAction(source_track="存5北", target_track="临1", vehicle_nos=["A", "B"], action_type="DETACH"),
        HookAction(source_track="临1", target_track="临2", vehicle_nos=["A"], action_type="DETACH"),
        HookAction(source_track="临2", target_track="存4北", vehicle_nos=["A"], action_type="ATTACH"),
        HookAction(source_track="存4北", target_track="存1", vehicle_nos=["A"], action_type="DETACH"),
    ]

    shape = summarize_plan_shape(moves)

    assert shape["staging_hook_count"] == 3
    assert shape["staging_to_staging_hook_count"] == 1
    assert shape["rehandled_vehicle_count"] == 1
    assert shape["max_vehicle_touch_count"] == 4


def test_native_move_scoring_avoids_polluting_needed_goal_track_within_same_tier():
    normalized, _ = _normalize(
        _payload(
            [
                _vehicle("A", "存5北", "存4北", order=1),
                _vehicle("B", "存5北", "存1", order=2),
            ]
        )
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = ReplayState(
        track_sequences={"存5北": ["B"], "临1": [], "存1": [], "机库": [], "存4北": []},
        loco_track_name="临1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("A",),
    )
    staging = HookAction(
        source_track="临1",
        target_track="临1",
        vehicle_nos=["A"],
        path_tracks=[],
        action_type="DETACH",
    )
    polluting_goal_track = HookAction(
        source_track="临1",
        target_track="存1",
        vehicle_nos=["A"],
        path_tracks=[],
        action_type="DETACH",
    )
    staging_next = _apply_move(
        state=state,
        move=staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    polluting_next = _apply_move(
        state=state,
        move=polluting_goal_track,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    staging_score, staging_tier = _score_native_move(
        move=staging,
        state=state,
        next_state=staging_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北", "存1"},
    )
    polluting_score, polluting_tier = _score_native_move(
        move=polluting_goal_track,
        state=state,
        next_state=polluting_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北", "存1"},
    )

    assert staging_tier == polluting_tier
    assert staging_score < polluting_score


def test_native_move_scoring_delays_close_door_pushers_until_close_door_is_placed():
    normalized, _ = _normalize(
        _payload(
            [
                _vehicle("N1", "存1", "存4北", order=1),
                _vehicle("N2", "存1", "存4北", order=2),
                _vehicle("N3", "存1", "存4北", order=3),
                {
                    **_vehicle("CD", "存5北", "存4北", order=1),
                    "vehicleAttributes": "关门车",
                },
            ]
        )
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = ReplayState(
        track_sequences={"存1": [], "存5北": ["CD"], "临1": [], "存4北": [], "机库": []},
        loco_track_name="存1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("N1", "N2", "N3"),
    )
    early_final = HookAction(
        source_track="存1",
        target_track="存4北",
        vehicle_nos=["N1", "N2", "N3"],
        path_tracks=[],
        action_type="DETACH",
    )
    hold_as_pushers = HookAction(
        source_track="存1",
        target_track="临1",
        vehicle_nos=["N1", "N2", "N3"],
        path_tracks=[],
        action_type="DETACH",
    )
    early_next = _apply_move(
        state=state,
        move=early_final,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    hold_next = _apply_move(
        state=state,
        move=hold_as_pushers,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    early_score, early_tier = _score_native_move(
        move=early_final,
        state=state,
        next_state=early_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北"},
    )
    hold_score, hold_tier = _score_native_move(
        move=hold_as_pushers,
        state=state,
        next_state=hold_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北"},
    )

    assert hold_score < early_score


def test_native_move_scoring_drops_close_door_pushers_before_attaching_close_door():
    normalized, _ = _normalize(
        _payload(
            [
                _vehicle("N1", "存1", "存4北", order=1),
                _vehicle("N2", "存1", "存4北", order=2),
                _vehicle("N3", "存1", "存4北", order=3),
                {
                    **_vehicle("CD", "存5北", "存4北", order=1),
                    "vehicleAttributes": "关门车",
                },
            ]
        )
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = ReplayState(
        track_sequences={"存1": [], "存5北": ["CD"], "临1": [], "存4北": [], "机库": []},
        loco_track_name="存1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("N1", "N2", "N3"),
    )
    attach_close_door = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["CD"],
        path_tracks=[],
        action_type="ATTACH",
    )
    hold_as_pushers = HookAction(
        source_track="存1",
        target_track="临1",
        vehicle_nos=["N1", "N2", "N3"],
        path_tracks=[],
        action_type="DETACH",
    )
    attach_next = _apply_move(
        state=state,
        move=attach_close_door,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    hold_next = _apply_move(
        state=state,
        move=hold_as_pushers,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    attach_score, _ = _score_native_move(
        move=attach_close_door,
        state=state,
        next_state=attach_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=1,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北"},
    )
    hold_score, _ = _score_native_move(
        move=hold_as_pushers,
        state=state,
        next_state=hold_next,
        plan_input=normalized,
        current_heuristic=2,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存4北"},
    )

    assert hold_score < attach_score


def test_random_depot_candidate_targets_prefer_less_loaded_same_preference_track():
    normalized, _ = _normalize(
        _payload(
            [
                _vehicle("R", "存5北", "大库", order=1),
                _vehicle("OCC1", "修1库内", "修1库内", order=1),
                _vehicle("OCC2", "修1库内", "修1库内", order=2),
            ]
        )
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = ReplayState(
        track_sequences={
            "存5北": ["R"],
            "修1库内": ["OCC1", "OCC2"],
            "修2库内": [],
            "修3库内": [],
            "修4库内": [],
            "机库": [],
        },
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={"OCC1": "101", "OCC2": "102"},
    )

    targets = _candidate_targets(["R"], normalized, state, vehicle_by_no)

    assert targets.index("修2库内") < targets.index("修1库内")
