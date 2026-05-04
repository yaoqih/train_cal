from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.runner import solve_workflow


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_solve_workflow_executes_explicit_stage_sequence():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFSEQ1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "dispatch_work",
                "description": "先送调棚作业区",
                "vehicleGoals": [
                    {"vehicleNo": "WFSEQ1", "targetTrack": "调棚", "isSpotting": ""}
                ],
            },
            {
                "name": "depot_spot",
                "description": "工序后送入大库精准台位",
                "vehicleGoals": [
                    {"vehicleNo": "WFSEQ1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
            {
                "name": "departure",
                "description": "修竣后进入存4北",
                "vehicleGoals": [
                    {"vehicleNo": "WFSEQ1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert [stage.name for stage in result.stages] == [
        "dispatch_work",
        "depot_spot",
        "departure",
    ]
    assert all(stage.view is not None for stage in result.stages)
    assert all(stage.view.summary.is_valid is True for stage in result.stages if stage.view)
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[0].input_payload["locoTrackName"] == "机库"
    assert result.stages[1].input_payload["locoTrackName"] == "调棚"
    assert result.stages[2].input_payload["locoTrackName"] == "修1库内"
    assert result.stages[1].view.final_spot_assignments == {"WFSEQ1": "101"}
    assert result.stages[2].view.steps[-1].track_sequences["存4北"] == ["WFSEQ1"]


def test_solve_workflow_supports_outer_depot_random_stage():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修2库外", "trackDistance": 49.3},
            {"trackName": "修3库外", "trackDistance": 49.3},
            {"trackName": "修4库外", "trackDistance": 49.3},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFOUT1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "outer_random",
                "vehicleGoals": [
                    {"vehicleNo": "WFOUT1", "targetTrack": "大库外", "isSpotting": ""}
                ],
            }
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 1
    assert result.stages[0].view is not None
    assert result.stages[0].view.summary.is_valid is True
    final_tracks = result.stages[0].view.summary.final_tracks
    assert len(final_tracks) == 1
    assert final_tracks[0] in {"修1库外", "修2库外", "修3库外", "修4库外"}


def test_solve_workflow_requires_goal_for_each_vehicle_in_stage():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFREQ1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "invalid_stage",
                "vehicleGoals": [],
            }
        ],
        "locoTrackName": "机库",
    }

    with pytest.raises(ValueError):
        solve_workflow(master, payload)


def test_solve_workflow_carries_weigh_state_across_stages():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFWEIGH1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "称重",
            }
        ],
        "workflowStages": [
            {
                "name": "weigh_stage",
                "vehicleGoals": [
                    {"vehicleNo": "WFWEIGH1", "targetTrack": "机库", "isSpotting": ""}
                ],
            },
            {
                "name": "store_stage",
                "vehicleGoals": [
                    {"vehicleNo": "WFWEIGH1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.steps[-1].weighed_vehicle_nos == ["WFWEIGH1"]
    assert result.stages[1].view.steps[0].weighed_vehicle_nos == ["WFWEIGH1"]
    assert [hook.target_track for hook in result.stages[1].view.hook_plan] == ["机库", "存4北"]
    assert [hook.action_type for hook in result.stages[1].view.hook_plan] == ["ATTACH", "DETACH"]
    assert result.stages[1].view.hook_plan[0].source_track == "机库"


def test_solve_workflow_supports_multi_vehicle_diverge_and_merge():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFMIX1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "WFMIX2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            },
        ],
        "workflowStages": [
            {
                "name": "dispatch_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFMIX1", "targetTrack": "调棚", "isSpotting": ""},
                    {"vehicleNo": "WFMIX2", "targetTrack": "存5北", "isSpotting": ""},
                ],
            },
            {
                "name": "depot_hold",
                "vehicleGoals": [
                    {"vehicleNo": "WFMIX1", "targetTrack": "大库", "isSpotting": "101"},
                    {"vehicleNo": "WFMIX2", "targetTrack": "存5北", "isSpotting": ""},
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFMIX1", "targetTrack": "存4北", "isSpotting": ""},
                    {"vehicleNo": "WFMIX2", "targetTrack": "存4北", "isSpotting": ""},
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[2].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.final_spot_assignments == {"WFMIX1": "101"}
    assert result.stages[2].view.summary.is_valid is True
    assert set(result.stages[2].view.steps[-1].track_sequences["存4北"]) == {"WFMIX2", "WFMIX1"}


def test_solve_workflow_supports_wash_depot_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "WFWASH1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "wash_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFWASH1", "targetTrack": "洗南", "isSpotting": ""}
                ],
            },
            {
                "name": "depot_spot",
                "vehicleGoals": [
                    {"vehicleNo": "WFWASH1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFWASH1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[2].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.final_spot_assignments == {"WFWASH1": "101"}
    assert result.stages[2].view.steps[-1].track_sequences["存4北"] == ["WFWASH1"]


def test_solve_workflow_supports_wheel_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "轮", "trackDistance": 118.2},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "WFWHEEL1",
                "repairProcess": "临修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "重车",
            }
        ],
        "workflowStages": [
            {
                "name": "wheel_operate",
                "vehicleGoals": [
                    {"vehicleNo": "WFWHEEL1", "targetTrack": "轮", "isSpotting": "是"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFWHEEL1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[1].view.summary.is_valid is True
    assert result.stages[0].view.steps[-1].track_sequences["轮"] == ["WFWHEEL1"]
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFWHEEL1"]


def test_solve_workflow_supports_paint_depot_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "油", "trackDistance": 124},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFPAINT1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "paint_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFPAINT1", "targetTrack": "油", "isSpotting": "是"}
                ],
            },
            {
                "name": "depot_spot",
                "vehicleGoals": [
                    {"vehicleNo": "WFPAINT1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFPAINT1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[2].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.final_spot_assignments == {"WFPAINT1": "101"}
    assert result.stages[2].view.steps[-1].track_sequences["存4北"] == ["WFPAINT1"]


def test_solve_workflow_supports_shot_depot_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "抛", "trackDistance": 131.8},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFSHOT1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "shot_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFSHOT1", "targetTrack": "抛", "isSpotting": "是"}
                ],
            },
            {
                "name": "depot_spot",
                "vehicleGoals": [
                    {"vehicleNo": "WFSHOT1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFSHOT1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[2].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.final_spot_assignments == {"WFSHOT1": "101"}
    assert result.stages[2].view.steps[-1].track_sequences["存4北"] == ["WFSHOT1"]


def test_solve_workflow_supports_outer_depot_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修2库外", "trackDistance": 49.3},
            {"trackName": "修3库外", "trackDistance": 49.3},
            {"trackName": "修4库外", "trackDistance": 49.3},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFOUTDEP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "outer_random",
                "vehicleGoals": [
                    {"vehicleNo": "WFOUTDEP1", "targetTrack": "大库外", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFOUTDEP1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[1].view.summary.is_valid is True
    assert result.stages[0].view.summary.final_tracks[0] in {"修1库外", "修2库外", "修3库外", "修4库外"}
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFOUTDEP1"]


def test_solve_workflow_supports_dispatch_then_jiku_final_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFJIKU1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "dispatch_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFJIKU1", "targetTrack": "调棚", "isSpotting": ""}
                ],
            },
            {
                "name": "jiku_final",
                "vehicleGoals": [
                    {"vehicleNo": "WFJIKU1", "targetTrack": "机库", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.summary.final_tracks == ["机库"]
    assert result.stages[1].view.steps[-1].track_sequences["机库"] == ["WFJIKU1"]


def test_solve_workflow_supports_tank_wash_direct_departure_chain():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "WFTANK1",
                "repairProcess": "临修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "wash_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFTANK1", "targetTrack": "洗南", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFTANK1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFTANK1"]


def test_solve_workflow_rejects_cun4nan_as_stage_goal():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4南", "trackDistance": 154.5},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFBAD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "invalid_stage",
                "vehicleGoals": [
                    {"vehicleNo": "WFBAD1", "targetTrack": "存4南", "isSpotting": ""}
                ],
            }
        ],
        "locoTrackName": "机库",
    }

    with pytest.raises(Exception, match="存4南 cannot be a final target"):
        solve_workflow(master, payload)


def test_solve_workflow_supports_close_door_departure_ordering():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        # Under PREPEND, vehicles must be placed in the order they're accessible
        # (front first). WFCD4 (close-door) must be placed FIRST on 存4北 so that
        # WF1-3 subsequently push it to index 3 (position 4). For WFCD4 to be
        # accessible first, it must be at order=1 (north/front of 存5北).
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFCD4",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "关门车",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "WFCD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "WFCD2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "WFCD3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            },
        ],
        "workflowStages": [
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFCD1", "targetTrack": "存4北", "isSpotting": ""},
                    {"vehicleNo": "WFCD2", "targetTrack": "存4北", "isSpotting": ""},
                    {"vehicleNo": "WFCD3", "targetTrack": "存4北", "isSpotting": ""},
                    {"vehicleNo": "WFCD4", "targetTrack": "存4北", "isSpotting": ""},
                ],
            }
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 1
    assert result.stages[0].view is not None
    final_seq = result.stages[0].view.steps[-1].track_sequences["存4北"]
    assert result.stages[0].view.summary.is_valid is True
    assert len(final_seq) == 4
    assert final_seq[-1] == "WFCD4"


def test_solve_workflow_supports_inspection_depot_then_departure():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFINS1",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "inspection_depot",
                "vehicleGoals": [
                    {"vehicleNo": "WFINS1", "targetTrack": "大库", "isSpotting": "迎检"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFINS1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.final_spot_assignments == {"WFINS1": "101"}
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFINS1"]


def test_solve_workflow_supports_weigh_then_jiku_final():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFWJ1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "称重",
            }
        ],
        "workflowStages": [
            {
                "name": "jiku_final",
                "vehicleGoals": [
                    {"vehicleNo": "WFWJ1", "targetTrack": "机库", "isSpotting": ""}
                ],
            }
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 1
    assert result.stages[0].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.steps[-1].track_sequences["机库"] == ["WFWJ1"]
    assert result.stages[0].view.steps[-1].weighed_vehicle_nos == ["WFWJ1"]
    assert result.stages[0].view.final_spot_assignments == {"WFWJ1": "机库:WEIGH"}


def test_solve_workflow_supports_pre_repair_then_departure():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFPR1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "pre_repair",
                "vehicleGoals": [
                    {"vehicleNo": "WFPR1", "targetTrack": "调棚", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFPR1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.final_spot_assignments == {}
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFPR1"]


def test_solve_workflow_supports_main_pre_repair_then_departure():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFPM1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "main_pre_repair",
                "vehicleGoals": [
                    {"vehicleNo": "WFPM1", "targetTrack": "预修", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFPM1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.summary.final_tracks == ["预修"]
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFPM1"]


def test_solve_workflow_supports_jipeng_then_departure():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机棚", "trackDistance": 105.8},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFJP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "jipeng_hold",
                "vehicleGoals": [
                    {"vehicleNo": "WFJP1", "targetTrack": "机棚", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFJP1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 2
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.summary.final_tracks == ["机棚"]
    assert result.stages[1].view.steps[-1].track_sequences["存4北"] == ["WFJP1"]


def test_solve_workflow_supports_depot_then_wheel_then_departure():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "轮", "trackDistance": 118.2},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "WFDW1",
                "repairProcess": "临修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "重车",
            }
        ],
        "workflowStages": [
            {
                "name": "depot_spot",
                "vehicleGoals": [
                    {"vehicleNo": "WFDW1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
            {
                "name": "wheel_operate",
                "vehicleGoals": [
                    {"vehicleNo": "WFDW1", "targetTrack": "轮", "isSpotting": "是"}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFDW1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    result = solve_workflow(master, payload)

    assert result.stage_count == 3
    assert result.stages[0].view is not None
    assert result.stages[1].view is not None
    assert result.stages[2].view is not None
    assert result.stages[0].view.summary.is_valid is True
    assert result.stages[0].view.final_spot_assignments == {"WFDW1": "101"}
    assert result.stages[1].view.summary.final_tracks == ["轮"]
    assert result.stages[2].view.steps[-1].track_sequences["存4北"] == ["WFDW1"]
