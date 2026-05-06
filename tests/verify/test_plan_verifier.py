from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.plan_verifier import verify_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _native_direct_plan(
    *,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    detach_path_tracks: list[str],
) -> list[dict]:
    return [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": [source_track],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": source_track,
            "targetTrack": target_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": detach_path_tracks,
        },
    ]


def _native_staging_round_trip(
    *,
    source_track: str,
    staging_track: str,
    final_target_track: str,
    vehicle_nos: list[str],
    stage_path_tracks: list[str],
    return_path_tracks: list[str],
) -> list[dict]:
    return [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": source_track,
            "targetTrack": source_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": [source_track],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": source_track,
            "targetTrack": staging_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": stage_path_tracks,
        },
        {
            "hookNo": 3,
            "actionType": "ATTACH",
            "sourceTrack": staging_track,
            "targetTrack": staging_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": [staging_track],
        },
        {
            "hookNo": 4,
            "actionType": "DETACH",
            "sourceTrack": staging_track,
            "targetTrack": final_target_track,
            "vehicleNos": vehicle_nos,
            "pathTracks": return_path_tracks,
        },
    ]


def test_plan_verifier_accepts_valid_direct_move():
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
                "vehicleNo": "H1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["H1"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        ),
    )

    assert report.is_valid is True
    assert report.errors == []


def test_plan_verifier_rejects_non_sequential_hook_numbers():
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
                "vehicleNo": "HOOK_GAP",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    hook_plan = _native_direct_plan(
        source_track="存5北",
        target_track="机库",
        vehicle_nos=["HOOK_GAP"],
        detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
    )
    hook_plan[1]["hookNo"] = 3

    report = verify_plan(master, normalized, hook_plan)

    assert report.is_valid is False
    assert any("hookNo" in error and "sequential" in error for error in report.errors)


def test_plan_verifier_rejects_wrong_final_track():
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
                "vehicleNo": "H2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="存4北",
            vehicle_nos=["H2"],
            detach_path_tracks=["存5北", "存4北"],
        ),
    )

    assert report.is_valid is False
    assert any("final track" in error.lower() for error in report.errors)


def test_plan_verifier_reports_work_position_rank_mismatch():
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
                "vehicleNo": "HW1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "是",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="调棚",
            vehicle_nos=["HW1"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
        ),
    )

    assert report.is_valid is False
    assert any("south_rank=1" in error and "expected one of" in error for error in report.errors)


def test_plan_verifier_reports_duplicate_explicit_work_slot():
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
    plan = [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": "存5北",
            "targetTrack": "存5北",
            "vehicleNos": ["SLOT_A", "SLOT_B"],
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": "存5北",
            "targetTrack": "洗南",
            "vehicleNos": ["SLOT_A", "SLOT_B"],
            "pathTracks": ["存5北", "存5南", "渡8", "渡9", "临4", "临3", "洗北", "洗南"],
        },
    ]

    report = verify_plan(master, normalized, plan)

    assert report.is_valid is False
    assert any("duplicate explicit work slot 2 on 洗南" in error for error in report.errors)


def test_plan_verifier_rejects_detach_source_that_does_not_match_loco_track():
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
                "vehicleNo": "HV_SRC",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["HV_SRC"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "临1",
                "targetTrack": "机库",
                "vehicleNos": ["HV_SRC"],
                "pathTracks": ["临1", "临2", "渡4", "机库"],
            },
        ],
    )

    assert report.is_valid is False
    assert any("DETACH sourceTrack" in error for error in report.errors)


def test_plan_verifier_warns_capacity_overflow_without_rejecting_plan():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 20},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "H3",
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
                "vehicleNo": "H4",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["H3"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        ),
    )

    assert report.is_valid is True
    assert report.errors == []
    assert any("capacity" in warning.lower() for warning in report.capacity_warnings)


def test_plan_verifier_rejects_close_door_front_position_non_cun4bei_when_gt10():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 200},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"H5_{i}",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "关门车" if i == 1 else "",
            }
            for i in range(1, 12)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle_nos = [f"H5_{i}" for i in range(1, 12)]

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=vehicle_nos,
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        ),
    )

    assert report.is_valid is False
    assert any("close-door" in error.lower() for error in report.errors)


def test_plan_verifier_rejects_hook_exceeding_empty_vehicle_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 500},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"H6_{i}",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for i in range(1, 22)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle_nos = [f"H6_{i}" for i in range(1, 22)]

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="存5南",
            vehicle_nos=vehicle_nos,
            detach_path_tracks=["存5北", "存5南"],
        ),
    )

    assert report.is_valid is False
    assert any("20" in error for error in report.errors)


def test_plan_verifier_rejects_hook_with_three_heavy_vehicles():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 500},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "敞车",
                "vehicleNo": f"H7_{i}",
                "repairProcess": "段修",
                "vehicleLength": 12.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            }
            for i in range(1, 4)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="存5南",
            vehicle_nos=["H7_1", "H7_2", "H7_3"],
            detach_path_tracks=["存5北", "存5南"],
        ),
    )

    assert report.is_valid is False
    assert any("重车" in error for error in report.errors)


def test_plan_verifier_rejects_cun4nan_staging_when_source_remainder_blocks_route():
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
                "vehicleNo": "H7A",
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
                "vehicleNo": "H7B",
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

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "存4南",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存5北", "存5南", "存4南"],
            },
            {
                "hookNo": 3,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["H7B"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 4,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["H7B"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            },
            {
                "hookNo": 5,
                "actionType": "ATTACH",
                "sourceTrack": "存4南",
                "targetTrack": "存4南",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存4南"],
            },
            {
                "hookNo": 6,
                "actionType": "DETACH",
                "sourceTrack": "存4南",
                "targetTrack": "存5北",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存4南", "存5南", "存5北"],
            },
        ],
    )

    assert report.is_valid is False
    hook2 = next(item for item in report.hook_reports if item.hook_no == 2)
    assert hook2.blocking_tracks == ["存5北"]
    assert any("interference" in error.lower() for error in hook2.errors)


def test_plan_verifier_rejects_heavy_equivalent_and_l1_overflow_together():
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
                    "vehicleModel": "敞车",
                    "vehicleNo": f"H7L_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.0,
                    "targetTrack": "存2",
                    "isSpotting": "",
                    "vehicleAttributes": "重车" if i <= 2 else "",
                }
                for i in range(1, 16)
            ]
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle_nos = [f"H7L_{i}" for i in range(1, 16)]

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="存2",
            vehicle_nos=vehicle_nos,
            detach_path_tracks=["存5北", "渡1", "渡2", "渡3", "存2"],
        ),
    )

    assert report.is_valid is False
    assert any("折算" in error for error in report.errors)
    assert any("190m" in error or "190" in error for error in report.errors)


def test_plan_verifier_rejects_loaded_loco_access_length_overflow_before_second_attach():
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
    first_block = [f"LOAD_A_{i}" for i in range(1, 16)]
    plan = [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": "存5北",
            "targetTrack": "存5北",
            "vehicleNos": first_block,
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "ATTACH",
            "sourceTrack": "存2",
            "targetTrack": "存2",
            "vehicleNos": ["LOAD_B"],
            "pathTracks": ["存2"],
        },
        {
            "hookNo": 3,
            "actionType": "DETACH",
            "sourceTrack": "存2",
            "targetTrack": "存2",
            "vehicleNos": first_block + ["LOAD_B"],
            "pathTracks": ["存2"],
        },
    ]

    report = verify_plan(master, normalized, plan)

    assert report.is_valid is False
    assert any("Locomotive access before ATTACH" in error for error in report.errors)
    assert any("190m" in error or "190" in error for error in report.errors)


def test_plan_verifier_rejects_detach_prefix_when_full_carry_exceeds_route_length():
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
    vehicle_nos = [f"DETACH_FULL_{i}" for i in range(1, 16)]
    plan = [
        {
            "hookNo": 1,
            "actionType": "ATTACH",
            "sourceTrack": "存5北",
            "targetTrack": "存5北",
            "vehicleNos": vehicle_nos,
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": "存5北",
            "targetTrack": "存2",
            "vehicleNos": ["DETACH_FULL_15"],
            "pathTracks": ["存5北", "渡1", "渡2", "渡3", "存2"],
        },
        {
            "hookNo": 3,
            "actionType": "DETACH",
            "sourceTrack": "存2",
            "targetTrack": "存2",
            "vehicleNos": vehicle_nos[:14],
            "pathTracks": ["存2"],
        },
    ]

    report = verify_plan(master, normalized, plan)

    assert report.is_valid is False
    assert any("190m" in error or "190" in error for error in report.errors)


def test_plan_verifier_rejects_more_than_one_weigh_vehicle_per_hook():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 200},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"H8_{i}",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
            for i in range(1, 3)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["H8_1", "H8_2"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        ),
    )

    assert report.is_valid is False
    assert any("spot" in error.lower() or "机库" in error for error in report.errors)


def test_plan_verifier_rejects_occupied_exact_depot_spot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "H9_OCC",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "H9_NEW",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="修1库内",
            vehicle_nos=["H9_NEW"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        ),
    )

    assert report.is_valid is False
    assert any("spot" in error.lower() or "台位" in error for error in report.errors)


def test_plan_verifier_returns_hook_level_diagnostics():
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
                "vehicleNo": "H10",
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
                "vehicleNo": "H10_BLOCK",
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

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="修1库内",
            vehicle_nos=["H10"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        ),
    )

    assert report.is_valid is False
    assert len(report.hook_reports) == 2
    hook_report = report.hook_reports[1]
    assert hook_report.hook_no == 2
    assert hook_report.is_valid is False
    assert "存5南" in hook_report.blocking_tracks
    assert any("interference" in error.lower() for error in hook_report.errors)
    assert hook_report.route_length_m and hook_report.route_length_m > 0


def test_plan_verifier_rejects_short_random_depot_vehicle_on_3_4_when_1_2_still_available():
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
                "vehicleNo": "VSHORT34",
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

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="修3库内",
            vehicle_nos=["VSHORT34"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修3库外", "修3库内"],
        ),
    )

    assert report.is_valid is False
    assert any("preferred" in error.lower() or "fallback" in error.lower() for error in report.errors)


def test_plan_verifier_accepts_snapshot_fallback_as_soft_preference():
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
                "vehicleNo": "SNAPSHOT_SOFT_FALLBACK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(master, normalized, [])

    assert report.is_valid is True
    assert report.errors == []


def test_plan_verifier_accepts_short_random_depot_vehicle_on_3_4_when_1_2_are_full():
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
                "vehicleNo": "VSHORTFB",
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
                    "vehicleNo": f"F1_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "修1库内",
                    "isSpotting": f"10{idx}",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
            *[
                {
                    "trackName": "修2库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"F2_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "修2库内",
                    "isSpotting": f"20{idx}",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)

    report = verify_plan(
        master,
        normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="修3库内",
            vehicle_nos=["VSHORTFB"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡12", "渡13", "修3库外", "修3库内"],
        ),
    )

    assert report.is_valid is True


def test_plan_verifier_accepts_close_door_vehicle_reaching_fourth_position_on_cun4bei():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "EN320",
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
                "vehicleNo": "EN321",
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
                "vehicleNo": "CDN303",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "EN322",
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

    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["EN320", "EN321", "CDN303", "EN322"],
                "pathTracks": ["存5北"],
            },
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "临1",
                "vehicleNos": ["EN322"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1"],
            },
            {
                "hookNo": 3,
                "actionType": "DETACH",
                "sourceTrack": "临1",
                "targetTrack": "存4北",
                "vehicleNos": ["CDN303"],
                "pathTracks": ["临1", "渡2", "渡1", "存4北"],
            },
            {
                "hookNo": 4,
                "actionType": "DETACH",
                "sourceTrack": "存4北",
                "targetTrack": "存4北",
                "vehicleNos": ["EN320", "EN321"],
                "pathTracks": ["存4北"],
            },
            {
                "hookNo": 5,
                "actionType": "ATTACH",
                "sourceTrack": "临1",
                "targetTrack": "临1",
                "vehicleNos": ["EN322"],
                "pathTracks": ["临1"],
            },
            {
                "hookNo": 6,
                "actionType": "DETACH",
                "sourceTrack": "临1",
                "targetTrack": "存4北",
                "vehicleNos": ["EN322"],
                "pathTracks": ["临1", "渡2", "渡1", "存4北"],
            },
        ],
    )

    assert report.is_valid is True
    assert report.errors == []
