from pathlib import Path

from fzed_shunting.demo.view_model import build_demo_view_model, select_demo_payload
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.sim.generator import generate_typical_suite


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_build_demo_view_model_for_single_hook_case():
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
                "vehicleNo": "V1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert view.summary.hook_count == 1
    assert view.summary.vehicle_count == 1
    assert view.summary.is_valid is True
    assert len(view.steps) == 2
    assert set(view.steps[1].changed_tracks) == {"存5北", "机库"}
    assert view.steps[1].hook.target_track == "机库"
    assert view.hook_plan[0].vehicle_count == 1
    assert view.hook_plan[0].route_length_m and view.hook_plan[0].route_length_m > 0
    assert view.hook_plan[0].remark
    assert view.track_map.track_nodes["存5北"].is_occupied is True
    assert view.track_map.track_nodes["机库"].is_occupied is False
    assert view.steps[1].track_map.active_path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert view.steps[1].track_map.changed_tracks == ["存5北", "机库"]
    assert view.steps[1].track_map.track_nodes["机库"].has_loco is True


def test_build_demo_view_model_keeps_verifier_errors():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "V2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert view.summary.is_valid is True
    assert [step.hook.target_track for step in view.steps[1:] if step.hook] == ["机库", "存4北"]


def test_build_demo_view_model_supports_weighted_solver_mode():
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
                "vehicleNo": "VW1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload, solver="weighted", heuristic_weight=2.0)

    assert view.summary.is_valid is True
    assert view.summary.hook_count == 1


def test_build_demo_view_model_supports_lns_solver_mode():
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
                "vehicleNo": "VL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload, solver="lns", beam_width=8)

    assert view.summary.is_valid is True
    assert view.summary.hook_count == 1


def test_build_demo_view_model_exposes_depot_spot_assignments():
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
                "vehicleNo": "V3",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert view.summary.assigned_spot_count == 1
    assert view.final_spot_assignments == {"V3": "101"}
    assert view.steps[1].spot_assignments == {"V3": "101"}
    assert "L19-修1尽头" in view.hook_plan[0].reverse_branch_codes


def test_build_demo_view_model_exposes_dispatch_work_spot_assignments():
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
                "vehicleNo": "V5",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "isSpotting": "是",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert view.summary.assigned_spot_count == 1
    assert view.final_spot_assignments == {"V5": "调棚:1"}
    assert view.steps[1].spot_assignments == {"V5": "调棚:1"}


def test_build_demo_view_model_accepts_external_plan_and_surfaces_step_errors():
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
                "vehicleNo": "V4",
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
                "vehicleNo": "V4_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    plan_payload = [
        {
            "hookNo": 1,
            "actionType": "PUT",
            "sourceTrack": "存5北",
            "targetTrack": "修1库内",
            "vehicleNos": ["V4"],
            "pathTracks": ["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        }
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)

    assert view.summary.is_valid is False
    assert view.failed_hook_nos == [1]
    assert any("interference" in error.lower() for error in view.steps[1].verifier_errors)
    assert view.comparison_summary is not None
    assert view.comparison_summary["externalHookCount"] == 1
    assert view.comparison_summary["externalIsValid"] is False
    assert view.comparison_summary["failedHookNos"] == [1]
    assert view.comparison_summary["solverHookCount"] is None
    assert "No solution found" in view.comparison_summary["solverError"]


def test_build_demo_view_model_computes_external_plan_comparison_summary():
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
                "vehicleNo": "CMP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    plan_payload = [
        {
            "hookNo": 1,
            "actionType": "PUT",
            "sourceTrack": "存5北",
            "targetTrack": "机库",
            "vehicleNos": ["CMP1"],
            "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        }
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)

    assert view.comparison_summary is not None
    assert view.comparison_summary["solverHookCount"] == 1
    assert view.comparison_summary["externalHookCount"] == 1
    assert view.comparison_summary["hookCountDelta"] == 0
    assert view.comparison_summary["externalIsValid"] is True
    assert view.comparison_summary["failedHookNos"] == []


def test_select_demo_payload_supports_typical_suite_scenarios():
    suite_payload = {
        "suite": "typical",
        "scenarioCount": 2,
        "scenarios": [
            {
                "name": "single_direct",
                "description": "single vehicle direct move",
                "payload": {
                    "trackInfo": [
                        {"trackName": "存5北", "trackDistance": 367},
                        {"trackName": "机库", "trackDistance": 71.6},
                    ],
                    "vehicleInfo": [
                        {
                            "trackName": "存5北",
                            "order": "1",
                            "vehicleModel": "棚车",
                            "vehicleNo": "TS1",
                            "repairProcess": "段修",
                            "vehicleLength": 14.3,
                            "targetTrack": "机库",
                            "isSpotting": "",
                            "vehicleAttributes": "",
                        }
                    ],
                    "locoTrackName": "机库",
                },
            },
            {
                "name": "weigh_then_store",
                "description": "weigh before store",
                "payload": {
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
                            "vehicleNo": "TS2",
                            "repairProcess": "段修",
                            "vehicleLength": 14.3,
                            "targetTrack": "存4北",
                            "isSpotting": "",
                            "vehicleAttributes": "称重",
                        }
                    ],
                    "locoTrackName": "机库",
                },
            },
        ],
    }

    selected, scenario_names, active_name = select_demo_payload(
        suite_payload,
        selected_name="weigh_then_store",
    )

    assert scenario_names == ["single_direct", "weigh_then_store"]
    assert active_name == "weigh_then_store"
    assert selected["vehicleInfo"][0]["vehicleNo"] == "TS2"


def test_build_demo_view_model_exposes_track_map_for_intermediate_step():
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
                "vehicleNo": "MAP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert len(view.steps) == 3
    assert view.steps[1].track_map.active_path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert view.steps[1].track_map.track_nodes["机库"].is_occupied is True
    assert view.steps[1].track_map.track_nodes["机库"].has_loco is True
    assert view.steps[2].track_map.active_path_tracks == ["机库", "渡4", "临2", "临1", "渡2", "渡1", "存4北"]
    assert view.steps[2].track_map.track_nodes["存4北"].is_occupied is True


def test_build_demo_view_model_exposes_topology_graph_edges():
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
                "vehicleNo": "TG1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)

    assert "存5北" in view.topology_graph.nodes
    assert "机库" in view.topology_graph.nodes
    assert ("存5北", "渡1") in view.topology_graph.edge_keys
    assert ("渡4", "机库") in view.topology_graph.edge_keys
    assert view.steps[1].topology_graph.active_edge_keys == [
        ("存5北", "渡1"),
        ("渡1", "渡2"),
        ("渡2", "临1"),
        ("临1", "临2"),
        ("临2", "渡4"),
        ("渡4", "机库"),
    ]


def test_build_demo_view_model_exposes_transition_frames_for_hook_motion():
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
                "vehicleNo": "ANIM1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload)
    frames = view.steps[1].transition_frames

    assert len(frames) > len(view.steps[1].hook.path_tracks)
    assert frames[0].frame_index == 0
    assert frames[0].progress == 0.0
    assert frames[0].current_track == "存5北"
    assert frames[-1].progress == 1.0
    assert frames[-1].current_track == "机库"
    assert frames[-1].passed_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]


def test_build_demo_view_model_supports_new_typical_suite_scenarios():
    master = load_master_data(DATA_DIR)
    suite = generate_typical_suite(master)

    inspection_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "inspection_depot")
    pre_repair_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "dispatch_pre_repair")
    wash_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "wash_work_area")
    wheel_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "wheel_operate")

    inspection_view = build_demo_view_model(master, inspection_payload)
    pre_repair_view = build_demo_view_model(master, pre_repair_payload)
    wash_view = build_demo_view_model(master, wash_payload)
    wheel_view = build_demo_view_model(master, wheel_payload)

    assert inspection_view.summary.vehicle_count == 1
    assert inspection_view.summary.is_valid is True
    assert pre_repair_view.summary.vehicle_count == 1
    assert pre_repair_view.summary.is_valid is True
    assert wash_view.summary.vehicle_count == 1
    assert wash_view.summary.is_valid is True
    assert wash_view.final_spot_assignments == {"TYP009": "洗南:1"}
    assert wheel_view.summary.vehicle_count == 1
    assert wheel_view.summary.is_valid is True
    assert wheel_view.steps[-1].hook.target_track == "轮"


def test_build_demo_view_model_supports_paint_and_shot_scenarios():
    master = load_master_data(DATA_DIR)
    suite = generate_typical_suite(master)

    paint_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "paint_work_area")
    shot_payload = next(item["payload"] for item in suite["scenarios"] if item["name"] == "shot_work_area")

    paint_view = build_demo_view_model(master, paint_payload)
    shot_view = build_demo_view_model(master, shot_payload)

    assert paint_view.summary.is_valid is True
    assert paint_view.steps[-1].hook.target_track == "油"
    assert shot_view.summary.is_valid is True
    assert shot_view.steps[-1].hook.target_track == "抛"
