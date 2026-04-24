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


def _native_attach(
    *,
    hook_no: int,
    source_track: str,
    vehicle_nos: list[str],
) -> dict:
    return {
        "hookNo": hook_no,
        "actionType": "ATTACH",
        "sourceTrack": source_track,
        "targetTrack": source_track,
        "vehicleNos": vehicle_nos,
        "pathTracks": [source_track],
    }


def _native_detach(
    *,
    hook_no: int,
    source_track: str,
    target_track: str,
    vehicle_nos: list[str],
    path_tracks: list[str],
) -> dict:
    return {
        "hookNo": hook_no,
        "actionType": "DETACH",
        "sourceTrack": source_track,
        "targetTrack": target_track,
        "vehicleNos": vehicle_nos,
        "pathTracks": path_tracks,
    }



def test_verifier_rejects_missing_path_tracks():
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
                "vehicleNo": "P1",
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
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["P1"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="机库",
                vehicle_nos=["P1"],
                path_tracks=[],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("path" in error.lower() for error in report.errors)


def test_verifier_rejects_path_not_starting_and_ending_correctly():
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
                "vehicleNo": "P2",
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
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["P2"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="机库",
                vehicle_nos=["P2"],
                path_tracks=["存4北", "机库"],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("path" in error.lower() for error in report.errors)


def test_verifier_rejects_non_complete_path_tracks_for_jiku_route():
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
                "vehicleNo": "P2B",
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
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["P2B"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="机库",
                vehicle_nos=["P2B"],
                path_tracks=["存5北", "机库"],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("complete" in error.lower() for error in report.errors)


def test_verifier_accepts_oil_and_shot_physical_route_branch_when_missing_segment_is_40m_placeholder():
    master = load_master_data(DATA_DIR)
    shot_payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "抛", "trackDistance": 131.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "抛",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    oil_payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "油", "trackDistance": 124.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P3O",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "油",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    shot_normalized = normalize_plan_input(shot_payload, master)
    oil_normalized = normalize_plan_input(oil_payload, master)

    shot_report = verify_plan(
        master,
        shot_normalized,
        _native_direct_plan(
            source_track="存5北",
            target_track="抛",
            vehicle_nos=["P3"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "抛"],
        ),
    )
    oil_report = verify_plan(
        master,
        oil_normalized,
        _native_direct_plan(
            source_track="机库",
            target_track="油",
            vehicle_nos=["P3O"],
            detach_path_tracks=["机库", "渡4", "渡5", "机北", "机棚", "临3", "油"],
        ),
    )

    assert shot_report.is_valid is True
    assert oil_report.is_valid is True


def test_verifier_rejects_l1_overflow_route():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P4",
                "repairProcess": "段修",
                "vehicleLength": 200.0,
                "targetTrack": "存2",
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
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["P4"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="存2",
                vehicle_nos=["P4"],
                path_tracks=["存5北", "渡1", "联6", "存2"],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("190" in error for error in report.errors)


def test_verifier_rejects_interference_when_intermediate_track_is_occupied():
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
                "vehicleNo": "P5",
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
                "vehicleNo": "BLOCK1",
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
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["P5"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="修1库内",
                vehicle_nos=["P5"],
                path_tracks=[
                    "存5北",
                    "存5南",
                    "渡8",
                    "渡9",
                    "渡10",
                    "联7",
                    "渡11",
                    "修1库外",
                    "修1库内",
                ],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("interference" in error.lower() for error in report.errors)
    assert any("存5南" in error for error in report.errors)
