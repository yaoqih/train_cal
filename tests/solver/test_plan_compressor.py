from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.plan_compressor import compress_plan
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _payload(target_track: str = "存1") -> dict:
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": target_track,
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }


def _three_vehicle_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "存2", "trackDistance": 85},
            {"trackName": "存3", "trackDistance": 85},
            {"trackName": "临1", "trackDistance": 89},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": target_track,
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, (vehicle_no, target_track) in enumerate(
                [("A", "存2"), ("B", "存3"), ("C", "存3")],
                start=1,
            )
        ],
        "locoTrackName": "机库",
    }


def _nine_vehicle_payload() -> dict:
    vehicle_specs = []
    for offset, group in enumerate((("A", "B", "C"), ("D", "E", "F"), ("G", "H", "I"))):
        vehicle_specs.extend(
            [
                (group[0], "存2"),
                (group[1], "存3"),
                (group[2], "存3"),
            ]
        )
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 160},
            {"trackName": "存2", "trackDistance": 160},
            {"trackName": "存3", "trackDistance": 160},
            {"trackName": "临1", "trackDistance": 160},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": target_track,
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, (vehicle_no, target_track) in enumerate(vehicle_specs, start=1)
        ],
        "locoTrackName": "机库",
    }


def _many_rebuild_vehicle_payload(group_count: int) -> dict:
    vehicle_specs = []
    for group_index in range(group_count):
        first = group_index * 3
        vehicle_specs.extend(
            [
                (f"V{first + 1}", "存2"),
                (f"V{first + 2}", "存3"),
                (f"V{first + 3}", "存3"),
            ]
        )
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 600},
            {"trackName": "存2", "trackDistance": 600},
            {"trackName": "存3", "trackDistance": 600},
            {"trackName": "临1", "trackDistance": 600},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": target_track,
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, (vehicle_no, target_track) in enumerate(vehicle_specs, start=1)
        ],
        "locoTrackName": "机库",
    }


def test_compressor_removes_redundant_round_trip_window():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload("存1"), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert result.compressed_plan == []


def test_compressor_continues_until_multiple_independent_windows_removed():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload("存1"), master)
    initial = build_initial_state(normalized)
    round_trip = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="DETACH",
        ),
    ]
    plan = round_trip * 3

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 3
    assert result.compressed_plan == []


def test_compressor_rejects_window_that_changes_final_state():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_payload("存4北"), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存4北",
            vehicle_nos=["A"],
            path_tracks=["存1", "临1", "渡2", "渡1", "存4北"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 0
    assert result.compressed_plan == plan


def test_compressor_rebuilds_single_source_window_to_final_prefixes():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_three_vehicle_payload(), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A", "B", "C"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="临1",
            vehicle_nos=["A", "B", "C"],
            path_tracks=["存1", "临1"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="临1",
            target_track="临1",
            vehicle_nos=["A", "B", "C"],
            path_tracks=["临1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="临1",
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["临1", "渡3", "存2"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存3",
            vehicle_nos=["B", "C"],
            path_tracks=["存2", "存3"],
            action_type="DETACH",
        ),
    ]


def test_compressor_continues_until_multiple_rebuild_windows_removed():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_nine_vehicle_payload(), master)
    initial = build_initial_state(normalized)
    plan = []
    for group in (["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]):
        plan.extend(
            [
                HookAction(
                    source_track="存1",
                    target_track="存1",
                    vehicle_nos=group,
                    path_tracks=["存1"],
                    action_type="ATTACH",
                ),
                HookAction(
                    source_track="存1",
                    target_track="临1",
                    vehicle_nos=group,
                    path_tracks=["存1", "临1"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="临1",
                    target_track="临1",
                    vehicle_nos=group,
                    path_tracks=["临1"],
                    action_type="ATTACH",
                ),
                HookAction(
                    source_track="临1",
                    target_track="存2",
                    vehicle_nos=[group[0]],
                    path_tracks=["临1", "渡3", "存2"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="存2",
                    target_track="存3",
                    vehicle_nos=group[1:],
                    path_tracks=["存2", "存3"],
                    action_type="DETACH",
                ),
            ]
        )

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 3
    assert len(result.compressed_plan) == 9


def test_compressor_convergence_limit_covers_many_rebuild_windows():
    group_count = 10
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_many_rebuild_vehicle_payload(group_count), master)
    initial = build_initial_state(normalized)
    plan = []
    for group_index in range(group_count):
        first = group_index * 3
        group = [f"V{first + 1}", f"V{first + 2}", f"V{first + 3}"]
        plan.extend(
            [
                HookAction(
                    source_track="存1",
                    target_track="存1",
                    vehicle_nos=group,
                    path_tracks=["存1"],
                    action_type="ATTACH",
                ),
                HookAction(
                    source_track="存1",
                    target_track="临1",
                    vehicle_nos=group,
                    path_tracks=["存1", "临1"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="临1",
                    target_track="临1",
                    vehicle_nos=group,
                    path_tracks=["临1"],
                    action_type="ATTACH",
                ),
                HookAction(
                    source_track="临1",
                    target_track="存2",
                    vehicle_nos=[group[0]],
                    path_tracks=["临1", "渡3", "存2"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="存2",
                    target_track="存3",
                    vehicle_nos=group[1:],
                    path_tracks=["存2", "存3"],
                    action_type="DETACH",
                ),
            ]
        )

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == group_count
    assert len(result.compressed_plan) == group_count * 3


def test_compressor_rejects_source_discontinuous_candidate():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_three_vehicle_payload(), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="临1",
            vehicle_nos=["A"],
            path_tracks=["存1", "临1"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="临1",
            target_track="临1",
            vehicle_nos=["A"],
            path_tracks=["临1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="临1",
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["临1", "渡3", "存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 0
    assert result.compressed_plan == plan
