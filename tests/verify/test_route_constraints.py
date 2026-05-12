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
            detach_path_tracks=["机库", "渡4", "渡5", "机北3", "机棚", "洗油北", "油"],
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
            {"trackName": "修1", "trackDistance": 151.7},
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
                target_track="修1",
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
                    "修1",
                ],
            ),
        ],
    )

    assert report.is_valid is False
    assert any("interference" in error.lower() for error in report.errors)
    assert any("存5南" in error for error in report.errors)


def test_verifier_rejects_attach_after_detach_leaves_loco_behind_source_track_cars():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "机南", "trackDistance": 90.1},
            {"trackName": "机棚", "trackDistance": 105.8},
            {"trackName": "机北3", "trackDistance": 69.1},
            {"trackName": "调北", "trackDistance": 70.1},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MOVE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PARKED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": track,
                    "order": "2" if track == "存5北" else "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": f"BLOCK-{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx, track in enumerate(
                    ["存5北", "机南", "机棚", "机北3", "调北"],
                    start=1,
                )
            ],
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)

    report = verify_plan(
        master,
        normalized,
        [
            _native_attach(hook_no=1, source_track="存5北", vehicle_nos=["MOVE"]),
            _native_detach(
                hook_no=2,
                source_track="存5北",
                target_track="存5南",
                vehicle_nos=["MOVE"],
                path_tracks=["存5北", "存5南"],
            ),
            {
                "hookNo": 3,
                "actionType": "ATTACH",
                "sourceTrack": "调棚",
                "targetTrack": "调棚",
                "vehicleNos": [],
                "pathTracks": ["调棚"],
            },
        ],
        require_complete_goals=False,
    )

    assert report.is_valid is False
    hook3 = next(item for item in report.hook_reports if item.hook_no == 3)
    assert "调北" in hook3.blocking_tracks
    assert "存5南" not in hook3.blocking_tracks


def test_verifier_rejects_loco_access_from_cun5nan_north_end_when_all_exits_are_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MOVE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PARKED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": track,
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": f"ROUTE_BLOCK_{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": track,
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx, track in enumerate(["存5北"], start=1)
            ],
        ],
        "locoTrackName": "存4北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)

    report = verify_plan(
        master,
        normalized,
        [
            _native_attach(hook_no=1, source_track="存4北", vehicle_nos=["MOVE"]),
            _native_detach(
                hook_no=2,
                source_track="存4北",
                target_track="存5南",
                vehicle_nos=["MOVE"],
                path_tracks=["存4北", "存4南", "存5南"],
            ),
            _native_attach(hook_no=3, source_track="调棚", vehicle_nos=[]),
        ],
        require_complete_goals=False,
    )

    assert report.is_valid is False
    hook3 = next(item for item in report.hook_reports if item.hook_no == 3)
    assert hook3.blocking_tracks == ["存5北"]


def test_verifier_rejects_detach_to_empty_target_from_wrong_entry_end():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存5南", "trackDistance": 156.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MOVE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)

    report = verify_plan(
        master,
        normalized,
        [
            _native_attach(hook_no=1, source_track="存4北", vehicle_nos=["MOVE"]),
            _native_detach(
                hook_no=2,
                source_track="存4北",
                target_track="存5南",
                vehicle_nos=["MOVE"],
                path_tracks=["存4北", "存4南", "存5南"],
            ),
        ],
    )

    assert report.is_valid is False
    hook2 = next(item for item in report.hook_reports if item.hook_no == 2)
    assert any("complete route path" in error for error in hook2.errors)
