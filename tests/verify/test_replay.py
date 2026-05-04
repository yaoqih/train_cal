from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import ReplayState, build_initial_state, replay_plan


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


def test_build_initial_state_orders_vehicles_north_to_south():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "B2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "B1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "",
    }
    normalized = normalize_plan_input(payload, master)

    state = build_initial_state(normalized)

    assert state.track_sequences["存5北"] == ["B1", "B2"]


def test_replay_multi_vehicle_attach_and_detach_keep_physical_sequence_order():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, vehicle_no in enumerate(("NORTH", "MIDDLE", "SOUTH"), start=1)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    after_attach = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "存5北",
                "targetTrack": "存5北",
                "vehicleNos": ["NORTH", "MIDDLE", "SOUTH"],
                "pathTracks": ["存5北"],
            }
        ],
        plan_input=normalized,
    ).final_state

    assert after_attach.track_sequences["存5北"] == []
    assert after_attach.loco_carry == ("NORTH", "MIDDLE", "SOUTH")

    with pytest.raises(ValueError, match="tail of loco_carry"):
        replay_plan(
            after_attach,
            [
                {
                    "hookNo": 2,
                    "actionType": "DETACH",
                    "sourceTrack": "存5北",
                    "targetTrack": "修2库内",
                    "vehicleNos": ["SOUTH", "MIDDLE"],
                    "pathTracks": ["存5北", "修2库内"],
                }
            ],
            plan_input=normalized,
        )

    after_detach = replay_plan(
        after_attach,
        [
            {
                "hookNo": 2,
                "actionType": "DETACH",
                "sourceTrack": "存5北",
                "targetTrack": "修2库内",
                "vehicleNos": ["MIDDLE", "SOUTH"],
                "pathTracks": ["存5北", "修2库内"],
            }
        ],
        plan_input=normalized,
    ).final_state

    assert after_detach.loco_carry == ("NORTH",)
    assert after_detach.track_sequences["修2库内"] == ["MIDDLE", "SOUTH"]


def test_replay_native_attach_and_detach_move_front_block():
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
                "vehicleNo": "C1",
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
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        _native_direct_plan(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["C1"],
            detach_path_tracks=["存5北", "机库"],
        ),
    )

    assert result.final_state.track_sequences["存5北"] == []
    assert result.final_state.track_sequences["机库"] == ["C1"]
    assert len(result.snapshots) == 3


def test_replay_attach_releases_source_spot_assignment():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C_ATTACH",
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
    initial = build_initial_state(normalized)

    assert initial.spot_assignments == {"C_ATTACH": "101"}

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "修1库内",
                "targetTrack": "修1库内",
                "vehicleNos": ["C_ATTACH"],
                "pathTracks": ["修1库内"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["修1库内"] == []
    assert result.final_state.loco_carry == ("C_ATTACH",)
    assert result.final_state.spot_assignments == {}


def test_replay_detach_removes_tail_block_from_loco_carry_and_prepends_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修2库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "调棚",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, vehicle_no in enumerate(["HEAD1", "HEAD2", "TAIL1", "TAIL2"], start=1)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = ReplayState(
        track_sequences={"修2库内": ["OLD_NORTH", "OLD_SOUTH"]},
        loco_track_name="调棚",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("HEAD1", "HEAD2", "TAIL1", "TAIL2"),
    )

    result = replay_plan(
        initial,
        [
            {
                "hookNo": 1,
                "actionType": "DETACH",
                "sourceTrack": "调棚",
                "targetTrack": "修2库内",
                "vehicleNos": ["TAIL1", "TAIL2"],
                "pathTracks": ["调棚", "修2库内"],
            }
        ],
        plan_input=normalized,
    )

    assert result.final_state.loco_carry == ("HEAD1", "HEAD2")
    assert result.final_state.track_sequences["修2库内"] == [
        "TAIL1",
        "TAIL2",
        "OLD_NORTH",
        "OLD_SOUTH",
    ]


def test_plan_verifier_rejects_detach_from_loco_carry_head():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修2库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "HEAD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "HEAD2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL2",
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
    report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": 1,
                "actionType": "ATTACH",
                "sourceTrack": "机库",
                "targetTrack": "机库",
                "vehicleNos": ["HEAD1", "HEAD2"],
                "pathTracks": ["机库"],
            },
            {
                "hookNo": 2,
                "actionType": "ATTACH",
                "sourceTrack": "调棚",
                "targetTrack": "调棚",
                "vehicleNos": ["TAIL1", "TAIL2"],
                "pathTracks": ["调棚"],
            },
            {
                "hookNo": 3,
                "actionType": "DETACH",
                "sourceTrack": "调棚",
                "targetTrack": "修2库内",
                "vehicleNos": ["HEAD1", "HEAD2"],
                "pathTracks": ["调棚", "修2库内"],
            },
        ],
    )

    assert not report.is_valid
    assert "tail of loco_carry" in report.errors[0]


def test_replay_assigns_exact_depot_spot_when_moving_into_depot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C2",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        _native_direct_plan(
            source_track="存5北",
            target_track="修1库内",
            vehicle_nos=["C2"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        ),
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["修1库内"] == ["C2"]
    assert result.final_state.spot_assignments["C2"] == "101"


def test_replay_realigns_depot_spots_to_full_track_order_after_prepend():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OLD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "OLD2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "NEW_SPOT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "修1库内",
                "targetSpotCode": "101",
                "isSpotting": "迎检",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        _native_direct_plan(
            source_track="存5北",
            target_track="修1库内",
            vehicle_nos=["NEW_SPOT"],
            detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        ),
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["修1库内"] == ["NEW_SPOT", "OLD1", "OLD2"]
    assert result.final_state.spot_assignments == {
        "NEW_SPOT": "101",
        "OLD1": "102",
        "OLD2": "103",
    }


def test_replay_does_not_assign_dispatch_work_spot_when_moving_into_work_area():
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
                "vehicleNo": "C3",
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
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        _native_direct_plan(
            source_track="存5北",
            target_track="调棚",
            vehicle_nos=["C3"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
        ),
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["调棚"] == ["C3"]
    assert "C3" not in result.final_state.spot_assignments


def test_replay_does_not_assign_dispatch_pre_repair_spot():
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
                "vehicleNo": "C4",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    result = replay_plan(
        initial,
        _native_direct_plan(
            source_track="存5北",
            target_track="调棚",
            vehicle_nos=["C4"],
            detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北", "调棚"],
        ),
        plan_input=normalized,
    )

    assert result.final_state.track_sequences["调棚"] == ["C4"]
    assert "C4" not in result.final_state.spot_assignments


def test_replay_rejects_detach_source_that_does_not_match_loco_track():
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
                "vehicleNo": "SRC1",
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
    initial = build_initial_state(normalized)

    with pytest.raises(ValueError, match="DETACH sourceTrack"):
        replay_plan(
            initial,
            [
                {
                    "hookNo": 1,
                    "actionType": "ATTACH",
                    "sourceTrack": "存5北",
                    "targetTrack": "存5北",
                    "vehicleNos": ["SRC1"],
                    "pathTracks": ["存5北"],
                },
                {
                    "hookNo": 2,
                    "actionType": "DETACH",
                    "sourceTrack": "临1",
                    "targetTrack": "机库",
                    "vehicleNos": ["SRC1"],
                    "pathTracks": ["临1", "临2", "渡4", "机库"],
                },
            ],
            plan_input=normalized,
        )
