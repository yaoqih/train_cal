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


def _three_vehicle_non_staging_buffer_payload() -> dict:
    payload = _three_vehicle_payload()
    payload["trackInfo"].append({"trackName": "存5北", "trackDistance": 367})
    return payload


def _split_source_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "存2", "trackDistance": 113},
            {"trackName": "存3", "trackDistance": 113},
            {"trackName": "临1", "trackDistance": 113},
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
                [("A", "存2"), ("B", "存2"), ("C", "存3"), ("D", "存3")],
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


def _multi_source_same_target_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 160},
            {"trackName": "存2", "trackDistance": 160},
            {"trackName": "存3", "trackDistance": 160},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": track_name,
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存3",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for track_name, vehicle_no in (("存1", "A"), ("存2", "B"))
        ],
        "locoTrackName": "机库",
    }


def _same_source_same_target_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 160},
            {"trackName": "存2", "trackDistance": 160},
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
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, vehicle_no in enumerate(("A", "B"), start=1)
        ],
        "locoTrackName": "机库",
    }


def _same_target_order_free_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "存2", "trackDistance": 160},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存2",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, vehicle_no in enumerate(("A", "B"), start=1)
        ],
        "locoTrackName": "存2",
    }


def _depot_area_equivalent_payload() -> dict:
    return {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 89.4},
            {"trackName": "修2库内", "trackDistance": 91.1},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "AREA",
                "targetTrack": "大库",
                "targetAreaCode": "大库:RANDOM",
                "targetSpotCode": "",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
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


def test_compressor_removes_goal_equivalent_window_when_verifier_accepts_candidate():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_depot_area_equivalent_payload(), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="修1库内",
            target_track="修1库内",
            vehicle_nos=["A"],
            path_tracks=["修1库内"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="修1库内",
            target_track="修2库内",
            vehicle_nos=["A"],
            path_tracks=["修1库内", "修1库外", "渡11", "渡12", "修2库外", "修2库内"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert result.compressed_plan == []


def test_compressor_removes_order_only_rewrite_when_verifier_accepts_final_state():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        _same_target_order_free_payload(),
        master,
        allow_internal_loco_tracks=True,
    )
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存2",
            target_track="存2",
            vehicle_nos=["A", "B"],
            path_tracks=["存2"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存2",
            vehicle_nos=["B"],
            path_tracks=["存2"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
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
            target_track="存3",
            vehicle_nos=["B", "C"],
            path_tracks=["临1", "渡3", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存3",
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["存3", "存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert [
        (move.action_type, move.target_track, move.vehicle_nos)
        for move in result.compressed_plan
    ] == [
        ("ATTACH", "存1", ["A", "B", "C"]),
        ("DETACH", "存3", ["B", "C"]),
        ("DETACH", "存2", ["A"]),
    ]


def test_compressor_keeps_legacy_single_source_rebuild_without_staging():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_three_vehicle_non_staging_buffer_payload(), master)
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
            target_track="存5北",
            vehicle_nos=["A", "B", "C"],
            path_tracks=["存1", "存5北"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存5北",
            target_track="存5北",
            vehicle_nos=["A", "B", "C"],
            path_tracks=["存5北"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存5北",
            target_track="存3",
            vehicle_nos=["B", "C"],
            path_tracks=["存5北", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存3",
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["存3", "存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert [
        (move.action_type, move.target_track, move.vehicle_nos)
        for move in result.compressed_plan
    ] == [
        ("ATTACH", "存1", ["A", "B", "C"]),
        ("DETACH", "存3", ["B", "C"]),
        ("DETACH", "存2", ["A"]),
    ]


def test_compressor_rebuilds_split_same_source_prefix_window():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_split_source_payload(), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A", "B"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="临1",
            vehicle_nos=["A", "B"],
            path_tracks=["存1", "临1"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["C", "D"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存3",
            vehicle_nos=["C", "D"],
            path_tracks=["存1", "临1", "渡3", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="临1",
            target_track="临1",
            vehicle_nos=["A", "B"],
            path_tracks=["临1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="临1",
            target_track="存2",
            vehicle_nos=["A", "B"],
            path_tracks=["临1", "渡3", "存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert [
        (move.action_type, move.target_track, move.vehicle_nos)
        for move in result.compressed_plan
    ] == [
        ("ATTACH", "存1", ["A", "B", "C", "D"]),
        ("DETACH", "存3", ["C", "D"]),
        ("DETACH", "存2", ["A", "B"]),
    ]


def test_compressor_rebuilds_wider_single_source_window_with_interleaved_detour():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 160},
            {"trackName": "存2", "trackDistance": 160},
            {"trackName": "存3", "trackDistance": 160},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 160},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北" if vehicle_no == "X" else "存1",
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
                [
                    ("A", "存3"),
                    ("B", "存3"),
                    ("C", "存2"),
                    ("D", "存4北"),
                    ("E", "存4北"),
                    ("X", "存5北"),
                ],
                start=1,
            )
        ],
        "locoTrackName": "存1",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A", "B"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存3",
            vehicle_nos=["A", "B"],
            path_tracks=["存1", "临1", "渡3", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存5北",
            target_track="存5北",
            vehicle_nos=["X"],
            path_tracks=["存5北"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存5北",
            target_track="临1",
            vehicle_nos=["X"],
            path_tracks=["存5北", "渡1", "渡2", "临1"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="临1",
            target_track="临1",
            vehicle_nos=["X"],
            path_tracks=["临1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="临1",
            target_track="存5北",
            vehicle_nos=["X"],
            path_tracks=["临1", "渡2", "渡1", "存5北"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["C"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存2",
            vehicle_nos=["C"],
            path_tracks=["存1", "渡7", "存2"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["D", "E"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存4北",
            vehicle_nos=["D", "E"],
            path_tracks=["存1", "临1", "渡2", "渡1", "存4北"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count >= 1
    assert [
        (move.action_type, move.target_track, move.vehicle_nos)
        for move in result.compressed_plan
    ] == [
        ("ATTACH", "存1", ["A", "B", "C", "D", "E"]),
        ("DETACH", "存4北", ["D", "E"]),
        ("DETACH", "存2", ["C"]),
        ("DETACH", "存3", ["A", "B"]),
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
                    target_track="存3",
                    vehicle_nos=group[1:],
                    path_tracks=["临1", "渡3", "存3"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="存3",
                    target_track="存2",
                    vehicle_nos=[group[0]],
                    path_tracks=["存3", "存2"],
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
                    target_track="存3",
                    vehicle_nos=group[1:],
                    path_tracks=["临1", "渡3", "存3"],
                    action_type="DETACH",
                ),
                HookAction(
                    source_track="存3",
                    target_track="存2",
                    vehicle_nos=[group[0]],
                    path_tracks=["存3", "存2"],
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


def test_compressor_merges_adjacent_attach_detach_pairs_when_final_order_is_preserved():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_multi_source_same_target_payload(), master)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="存2",
            target_track="存2",
            vehicle_nos=["B"],
            path_tracks=["存2"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存3",
            vehicle_nos=["B"],
            path_tracks=["存2", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存3",
            vehicle_nos=["A"],
            path_tracks=["存1", "临1", "渡3", "存3"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert result.compressed_plan == [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存2",
            vehicle_nos=["B"],
            path_tracks=["存2"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存2",
            target_track="存3",
            vehicle_nos=["A", "B"],
            path_tracks=["存2", "存3"],
            action_type="DETACH",
        ),
    ]


def test_compressor_merges_adjacent_same_source_same_target_pairs():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(_same_source_same_target_payload(), master)
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
            target_track="存2",
            vehicle_nos=["A"],
            path_tracks=["存1", "渡7", "存2"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["B"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存2",
            vehicle_nos=["B"],
            path_tracks=["存1", "渡7", "存2"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 1
    assert result.compressed_plan == [
        HookAction(
            source_track="存1",
            target_track="存1",
            vehicle_nos=["A", "B"],
            path_tracks=["存1"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="存1",
            target_track="存2",
            vehicle_nos=["A", "B"],
            path_tracks=["存1", "渡7", "存2"],
            action_type="DETACH",
        ),
    ]


def test_compressor_rejects_adjacent_merge_when_reordered_access_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "洗北", "trackDistance": 71.6},
            {"trackName": "洗南", "trackDistance": 90.0},
            {"trackName": "存3", "trackDistance": 85},
        ],
        "vehicleInfo": [
            {
                "trackName": "洗北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存3",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "洗南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存3",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临3",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)
    plan = [
        HookAction(
            source_track="洗北",
            target_track="洗北",
            vehicle_nos=["B"],
            path_tracks=["洗北"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="洗北",
            target_track="存3",
            vehicle_nos=["B"],
            path_tracks=["洗北", "临3", "机棚", "机北", "渡6", "渡7", "存3"],
            action_type="DETACH",
        ),
        HookAction(
            source_track="洗南",
            target_track="洗南",
            vehicle_nos=["A"],
            path_tracks=["洗南"],
            action_type="ATTACH",
        ),
        HookAction(
            source_track="洗南",
            target_track="存3",
            vehicle_nos=["A"],
            path_tracks=["洗南", "洗北", "临3", "机棚", "机北", "渡6", "渡7", "存3"],
            action_type="DETACH",
        ),
    ]

    result = compress_plan(normalized, initial, plan, master=master)

    assert result.accepted_rewrite_count == 0
    assert result.compressed_plan == plan
