from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.candidate_compiler import replay_candidate_steps
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(
    vehicle_no: str,
    track: str,
    target: str,
    *,
    order: int = 1,
    attributes: str = "",
) -> dict:
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": 14.3,
        "targetTrack": target,
        "isSpotting": "",
        "vehicleAttributes": attributes,
    }


def test_replay_candidate_steps_rejects_unreachable_locomotive_between_steps():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "调棚", "trackDistance": 134.0},
                {"trackName": "调北", "trackDistance": 43.0},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存3", "trackDistance": 367.0},
            ],
            "vehicleInfo": [
                _vehicle("A", "调棚", "存3", order=1),
                _vehicle("BLOCK", "调北", "调北", order=1),
                _vehicle("B", "存5北", "存3", order=1),
            ],
            "locoTrackName": "调棚",
        },
        master,
        allow_internal_loco_tracks=True,
    )
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    steps = [
        HookAction(
            source_track="调棚",
            target_track="调棚",
            vehicle_nos=["A"],
            path_tracks=["调棚"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="调棚",
            target_track="存3",
            vehicle_nos=["A"],
            path_tracks=["调棚", "调北", "渡4", "机北2", "机北1", "渡2", "渡3", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存5北",
            target_track="存5北",
            vehicle_nos=["B"],
            path_tracks=["存5北"],
            action_type="ATTACH",
        ),
    ]

    assert replay_candidate_steps(
        plan_input=normalized,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=RouteOracle(master),
    ) is None


def test_replay_candidate_steps_rejects_hook_vehicle_group_violation():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存1", "trackDistance": 113.0},
            ],
            "vehicleInfo": [
                _vehicle(f"H{index}", "存5北", "存1", order=index, attributes="重车")
                for index in range(1, 4)
            ],
            "locoTrackName": "机库",
        },
        master,
    )
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    steps = [
        HookAction(
            source_track="存5北",
            target_track="存5北",
            vehicle_nos=["H1", "H2", "H3"],
            path_tracks=["存5北"],
            action_type="ATTACH",
        ),
    ]

    assert replay_candidate_steps(
        plan_input=normalized,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=RouteOracle(master),
    ) is None


def test_replay_candidate_steps_rejects_large_non_cun4bei_close_door_first_hook():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存3", "trackDistance": 258.5},
            ],
            "vehicleInfo": [
                _vehicle(
                    f"LCD{index:02d}",
                    "存5北",
                    "存3",
                    order=index,
                    attributes="关门车" if index == 1 else "",
                )
                for index in range(1, 12)
            ],
            "locoTrackName": "机库",
        },
        master,
    )
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    vehicle_nos = [f"LCD{index:02d}" for index in range(1, 12)]
    steps = [
        HookAction(
            source_track="存5北",
            target_track="存5北",
            vehicle_nos=vehicle_nos,
            path_tracks=["存5北"],
            action_type="ATTACH",
        ),
    ]

    assert replay_candidate_steps(
        plan_input=normalized,
        state=state,
        vehicle_by_no=vehicle_by_no,
        steps=steps,
        route_oracle=RouteOracle(master),
    ) is None
