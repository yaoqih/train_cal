from pathlib import Path
import inspect

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.route_blockage import (
    compute_route_blockage_plan,
    route_blockage_release_score,
)
from fzed_shunting.solver.state import _apply_move
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import build_initial_state
from fzed_shunting.solver.move_generator import generate_real_hook_moves
from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
from fzed_shunting.solver.heuristic import make_state_heuristic_real_hook


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(vehicle_no: str, track: str, target: str, *, order: int = 1) -> dict:
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": 14.3,
        "targetTrack": target,
        "isSpotting": "",
        "vehicleAttributes": "",
    }


def _corridor_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "临3", "trackDistance": 77.9},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("SEEK", "临4", "存5北"),
            _vehicle("BLOCK", "存5南", "存5南"),
        ],
        "locoTrackName": "机库",
    }


def test_route_blockage_plan_reports_occupied_intermediate_goal_corridor():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized)

    plan = compute_route_blockage_plan(normalized, state, RouteOracle(master))

    assert plan.total_blockage_pressure == 1
    assert plan.blocked_vehicle_nos == ["SEEK"]
    fact = plan.facts_by_blocking_track["存5南"]
    assert fact.blocking_vehicle_nos == ["BLOCK"]
    assert fact.blocked_vehicle_nos == ["SEEK"]
    assert fact.target_tracks == ["存5北"]
    assert fact.source_tracks == ["临4"]


def test_route_blockage_plan_preserves_blocking_track_physical_order():
    master = load_master_data(DATA_DIR)
    payload = {
        **_corridor_payload(),
        "vehicleInfo": [
            _vehicle("SEEK", "临4", "存5北"),
            _vehicle("Z_FRONT", "存5南", "存5南", order=1),
            _vehicle("A_BACK", "存5南", "存5南", order=2),
        ],
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    plan = compute_route_blockage_plan(normalized, state, RouteOracle(master))

    assert plan.facts_by_blocking_track["存5南"].blocking_vehicle_nos == [
        "Z_FRONT",
        "A_BACK",
    ]


def test_route_blockage_plan_reports_blocked_loco_access_to_unfinished_source():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5南", "trackDistance": 156.0},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "修4库内", "trackDistance": 151.7},
                {"trackName": "临1", "trackDistance": 81.4},
                {"trackName": "临2", "trackDistance": 62.9},
            ],
            "vehicleInfo": [
                {
                    **_vehicle("SEEK", "存5南", "修4库内"),
                    "isSpotting": "405",
                },
                _vehicle("BLOCK", "存5北", "存5北"),
            ],
            "locoTrackName": "机库",
        },
        master,
    )
    state = build_initial_state(normalized)

    plan = compute_route_blockage_plan(normalized, state, RouteOracle(master))

    fact = plan.facts_by_blocking_track["存5北"]
    assert fact.blocking_vehicle_nos == ["BLOCK"]
    assert fact.blocked_vehicle_nos == ["SEEK"]
    assert fact.source_tracks == ["存5南"]
    assert fact.target_tracks == ["修4库内"]


def test_route_blockage_plan_pressure_drops_when_blocking_track_is_attached():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized).model_copy(update={"loco_track_name": "存5南"})
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    before = compute_route_blockage_plan(normalized, state, RouteOracle(master))

    next_state = _apply_move(
        state=state,
        move=HookAction(
            source_track="存5南",
            target_track="存5南",
            vehicle_nos=["BLOCK"],
            path_tracks=["存5南"],
            action_type="ATTACH",
        ),
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    after = compute_route_blockage_plan(normalized, next_state, RouteOracle(master))

    assert before.total_blockage_pressure == 2
    assert after.total_blockage_pressure == 0


def test_route_blockage_plan_can_filter_to_staging_debt_sources():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized)

    staging_plan = compute_route_blockage_plan(
        normalized,
        state,
        RouteOracle(master),
        blocked_source_tracks={"临4"},
    )
    non_staging_plan = compute_route_blockage_plan(
        normalized,
        state,
        RouteOracle(master),
        blocked_source_tracks={"存5北"},
    )

    assert staging_plan.total_blockage_pressure == 1
    assert non_staging_plan.total_blockage_pressure == 0


def test_route_blockage_plan_reuses_oracle_cache_by_state_and_source_filter(monkeypatch):
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)
    calls = 0
    original = route_oracle.validate_loco_access

    def wrapped_validate_loco_access(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(route_oracle, "validate_loco_access", wrapped_validate_loco_access)

    first = compute_route_blockage_plan(
        normalized,
        state,
        route_oracle,
        blocked_source_tracks={"临4"},
    )
    calls_after_first = calls
    second = compute_route_blockage_plan(
        normalized,
        state,
        route_oracle,
        blocked_source_tracks=frozenset({"临4"}),
    )
    filtered = compute_route_blockage_plan(
        normalized,
        state,
        route_oracle,
        blocked_source_tracks={"存5北"},
    )

    assert first is second
    assert first.total_blockage_pressure == 1
    assert filtered is not first
    assert filtered.total_blockage_pressure == 0
    assert calls_after_first > 0
    assert calls == calls_after_first


def test_real_hook_generator_can_attach_satisfied_goal_vehicle_to_release_corridor():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized).model_copy(update={"loco_track_name": "存5南"})

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存5南"
        and move.vehicle_nos == ["BLOCK"]
        for move in moves
    )


def test_route_blockage_release_score_reports_fact_pressure_only():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized).model_copy(update={"loco_track_name": "存5南"})
    route_oracle = RouteOracle(master)
    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)

    assert route_blockage_release_score(
        source_track="存5南",
        vehicle_nos=["BLOCK"],
        route_blockage_plan=route_blockage_plan,
    ) == 2
    assert route_blockage_release_score(
        source_track="临4",
        vehicle_nos=["SEEK"],
        route_blockage_plan=route_blockage_plan,
    ) == 0


def test_native_scoring_does_not_use_route_blockage_as_default_bias():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_corridor_payload(), master)
    state = build_initial_state(normalized).model_copy(update={"loco_track_name": "存5南"})
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    attach_blocker = HookAction(
        source_track="存5南",
        target_track="存5南",
        vehicle_nos=["BLOCK"],
        path_tracks=["存5南"],
        action_type="ATTACH",
    )
    attach_blocked_vehicle = HookAction(
        source_track="临4",
        target_track="临4",
        vehicle_nos=["SEEK"],
        path_tracks=["临4"],
        action_type="ATTACH",
    )
    blocker_next = _apply_move(
        state=state,
        move=attach_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    blocked_next = _apply_move(
        state=state,
        move=attach_blocked_vehicle,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    assert "route_blockage_plan" in inspect.signature(_score_native_move).parameters

    blocker_default_score, _ = _score_native_move(
        move=attach_blocker,
        state=state,
        next_state=blocker_next,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(blocker_next),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
    )
    blocked_default_score, _ = _score_native_move(
        move=attach_blocked_vehicle,
        state=state,
        next_state=blocked_next,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(blocked_next),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
    )
    route_blockage_plan = compute_route_blockage_plan(normalized, state, RouteOracle(master))
    blocker_route_release_score, _ = _score_native_move(
        move=attach_blocker,
        state=state,
        next_state=blocker_next,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(blocker_next),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        route_blockage_plan=route_blockage_plan,
    )

    assert blocker_default_score >= blocked_default_score
    assert blocker_route_release_score < blocker_default_score
