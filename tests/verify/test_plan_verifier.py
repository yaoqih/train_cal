from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.plan_verifier import verify_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["H1"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            }
        ],
    )

    assert report.is_valid is True
    assert report.errors == []


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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "存4北",
                "vehicleNos": ["H2"],
                "pathTracks": ["存5北", "存4北"],
            }
        ],
    )

    assert report.is_valid is False
    assert any("final track" in error.lower() for error in report.errors)


def test_plan_verifier_requires_work_area_spot_assignment():
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "调棚",
                "vehicleNos": ["HW1"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
            }
        ],
    )

    assert report.is_valid is True


def test_plan_verifier_rejects_capacity_overflow():
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["H3"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            }
        ],
    )

    assert report.is_valid is False
    assert any("capacity" in error.lower() for error in report.errors)


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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": vehicle_nos,
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            }
        ],
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "存5南",
                "vehicleNos": vehicle_nos,
                "pathTracks": ["存5北", "存5南"],
            }
        ],
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "存5南",
                "vehicleNos": ["H7_1", "H7_2", "H7_3"],
                "pathTracks": ["存5北", "存5南"],
            }
        ],
    )

    assert report.is_valid is False
    assert any("重车" in error for error in report.errors)


def test_plan_verifier_accepts_cun4nan_as_temporary_staging_not_final_goal():
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
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "存4南",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存5北", "存5南", "存4南"],
            },
            {
                "hookNo": 2,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["H7B"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            },
            {
                "hookNo": 3,
                "actionType": "PUT",
                "sourceTrack": "存4南",
                "targetTrack": "存5北",
                "vehicleNos": ["H7A"],
                "pathTracks": ["存4南", "存5南", "存5北"],
            },
        ],
    )

    assert report.is_valid is True
    assert report.errors == []


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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "存2",
                "vehicleNos": vehicle_nos,
                "pathTracks": ["存5北", "渡1", "渡2", "渡3", "存2"],
            }
        ],
    )

    assert report.is_valid is False
    assert any("折算" in error for error in report.errors)
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "机库",
                "vehicleNos": ["H8_1", "H8_2"],
                "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
            }
        ],
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "修1库内",
                "vehicleNos": ["H9_NEW"],
                "pathTracks": ["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
            }
        ],
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
        [
            {
                "hookNo": 1,
                "actionType": "PUT",
                "sourceTrack": "存5北",
                "targetTrack": "修1库内",
                "vehicleNos": ["H10"],
                "pathTracks": ["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
            }
        ],
    )

    assert report.is_valid is False
    assert len(report.hook_reports) == 1
    hook_report = report.hook_reports[0]
    assert hook_report.hook_no == 1
    assert hook_report.is_valid is False
    assert "存5南" in hook_report.blocking_tracks
    assert any("interference" in error.lower() for error in hook_report.errors)
    assert hook_report.route_length_m and hook_report.route_length_m > 0
