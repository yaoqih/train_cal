from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from fzed_shunting.demo.view_model import build_demo_view_model, select_demo_payload
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.sim.generator import generate_typical_workflow_suite


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
APP_PATH = Path(__file__).resolve().parents[2] / "app.py"
APP_SPEC = spec_from_file_location("train_cal_app", APP_PATH)
assert APP_SPEC is not None
assert APP_SPEC.loader is not None
app = module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(app)


def test_build_topology_graph_dot_marks_active_and_inactive_edges():
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
                "vehicleNo": "DOT1",
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

    dot = app._build_topology_graph_dot(
        view.steps[2].topology_graph,
        view.steps[2].track_map,
    )

    assert dot.startswith("graph shunting_topology {")
    assert '"存5北" -- "渡1" [color="#1d6f6d"' in dot
    assert '"渡1" -- "存4北" [color="#c5c0b5"' in dot
    assert '"机库" [label="机库\\n机车 / 占用 1"' in dot


def test_build_topology_svg_contains_schematic_areas_and_motion_marker():
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
                "vehicleNo": "SVG1",
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

    svg = app._build_topology_svg(
        view.steps[2].topology_graph,
        view.steps[2].track_map,
        hook=view.steps[2].hook,
        transition_frame=view.steps[2].transition_frames[3],
    )

    assert svg.startswith("<svg")
    assert "schematic-track-mainline" in svg
    assert ">存5北<" in svg
    assert 'class="moving-block-marker"' in svg
    assert 'class="topology-background"' not in svg


def test_build_topology_svg_can_emit_animated_route():
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
                "vehicleNo": "SVG2",
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

    svg = app._build_topology_svg(
        view.steps[2].topology_graph,
        view.steps[2].track_map,
        hook=view.steps[2].hook,
        animate=True,
    )

    assert "<animateMotion" in svg
    assert "route-motion-path" in svg


def test_build_topology_svg_promotes_active_and_target_track_labels():
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
                "vehicleNo": "SVG3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    view = build_demo_view_model(master, payload)
    svg = app._build_topology_svg(
        view.steps[-1].topology_graph,
        view.steps[-1].track_map,
        hook=view.steps[-1].hook,
    )

    assert 'schematic-track-label-active' in svg
    assert ">机库<" in svg
    assert ">存4北<" in svg


def test_build_topology_svg_does_not_render_all_track_labels_by_default():
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
                "vehicleNo": "SVG4",
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

    svg = app._build_topology_svg(
        view.steps[2].topology_graph,
        view.steps[2].track_map,
        hook=view.steps[2].hook,
    )

    assert 'class="topology-background"' not in svg
    assert "schematic-track-active" in svg
    assert ">存3<" not in svg
    assert ">联6<" not in svg


def test_build_topology_svg_uses_schematic_points_for_current_track_and_loco_markers():
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
                "vehicleNo": "SVG5",
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
    frame = view.steps[2].transition_frames[2]

    svg = app._build_topology_svg(
        view.steps[2].topology_graph,
        view.steps[2].track_map,
        hook=view.steps[2].hook,
        transition_frame=frame,
    )

    assert 'class="moving-block-marker"' in svg
    assert 'class="loco-marker"' in svg


def test_app_module_exposes_workflow_renderer():
    assert hasattr(app, "_render_workflow_demo")


def test_app_defaults_match_validation_runner_solver_profile():
    assert app.VALIDATION_DEFAULT_SOLVER == "beam"
    assert app.VALIDATION_DEFAULT_BEAM_WIDTH == 8
    assert app.VALIDATION_DEFAULT_TIMEOUT_SECONDS == 60.0
    assert app._validation_time_budget_ms(60.0) == 50_000.0


def test_app_labels_external_plan_validation_source():
    assert "外部 Plan JSON" in app._plan_validation_source_label([{}])
    assert "当前 Solver" in app._plan_validation_source_label(None)


def test_app_module_exposes_hook_sidebar_and_distance_helpers():
    assert hasattr(app, "_build_hook_sidebar_rows")
    assert hasattr(app, "_build_distance_breakdown_rows")
    assert hasattr(app, "_build_distance_catalog_rows")


def test_format_hook_vehicle_text_appends_spotting_attributes_and_length():
    payload = {
        "vehicleInfo": [
            {
                "vehicleNo": "SHOW1",
                "vehicleLength": 14.3,
                "isSpotting": "",
                "targetTrack": "预修",
                "vehicleAttributes": "",
            },
            {
                "vehicleNo": "SHOW2",
                "vehicleLength": 16.9,
                "isSpotting": "101",
                "targetTrack": "大库",
                "vehicleAttributes": "称重",
            },
            {
                "vehicleNo": "SHOW3",
                "vehicleLength": 15.6,
                "isSpotting": "否",
                "targetTrack": "存4北",
                "vehicleAttributes": "重车",
            },
        ]
    }

    vehicle_display_metadata = app._build_vehicle_display_metadata(payload)
    vehicle_text = app._format_hook_vehicle_text(
        ["SHOW1", "SHOW2", "SHOW3"],
        vehicle_display_metadata,
    )

    assert vehicle_text == (
        "SHOW1(对位=预修，属性=无，长度=14.3m) "
        "SHOW2(对位=101，属性=称重，长度=16.9m) "
        "SHOW3(对位=存4北，属性=重车，长度=15.6m)"
    )


def test_format_pre_hook_loco_carry_text_uses_previous_step_carry():
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
                "vehicleNo": "CARRY_SHOW_1",
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
            "actionType": "ATTACH",
            "sourceTrack": "存5北",
            "targetTrack": "存5北",
            "vehicleNos": ["CARRY_SHOW_1"],
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": "存5北",
            "targetTrack": "机库",
            "vehicleNos": ["CARRY_SHOW_1"],
            "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        },
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)
    vehicle_display_metadata = app._build_vehicle_display_metadata(payload)

    assert app._format_pre_hook_loco_carry_text(
        view,
        1,
        vehicle_display_metadata,
    ) == "本钩前调车机后挂: 无"
    assert app._format_pre_hook_loco_carry_text(
        view,
        2,
        vehicle_display_metadata,
    ) == "本钩前调车机后挂: CARRY_SHOW_1(对位=机库，属性=无，长度=14.3m)"


def test_build_step_state_rows_uses_formatted_vehicle_text():
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
                "vehicleNo": "STEPFMT1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "称重",
            }
        ],
        "locoTrackName": "机库",
    }

    view = build_demo_view_model(master, payload, plan_payload=[])
    vehicle_display_metadata = app._build_vehicle_display_metadata(payload)
    rows = app._build_step_state_rows(view.steps[0].track_map, vehicle_display_metadata)

    assert rows[0]["vehicles"] == "STEPFMT1(对位=101，属性=称重，长度=14.3m)"


def test_build_vehicle_display_metadata_prefers_vehicle_info_over_initial_vehicle_info():
    payload = {
        "initialVehicleInfo": [
            {
                "vehicleNo": "META1",
                "vehicleLength": 12.0,
                "vehicleAttributes": "",
            }
        ],
        "vehicleInfo": [
            {
                "vehicleNo": "META1",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "vehicleAttributes": "称重",
                "isSpotting": "",
            }
        ],
    }

    vehicle_display_metadata = app._build_vehicle_display_metadata(payload)

    assert vehicle_display_metadata["META1"] == {
        "requirement": "预修",
        "attributes": "称重",
        "length": "14.3m",
    }


def test_select_demo_payload_supports_workflow_suite_payload():
    master = load_master_data(DATA_DIR)
    suite = generate_typical_workflow_suite(master)

    payload, scenario_names, active_name = select_demo_payload(suite)

    assert scenario_names
    assert active_name == "dispatch_depot_departure"
    assert payload["workflowStages"][0]["name"] == "dispatch_work"


def test_app_module_detects_workflow_payloads_and_suites():
    assert app._is_workflow_payload({"workflowStages": []}) is True
    assert app._is_workflow_payload({"workflowStages": [{"name": "stage_1"}]}) is True
    assert app._is_workflow_payload({"scenarios": []}) is False


def test_workflow_progress_fraction_uses_stage_index():
    assert app._workflow_progress_value(stage_index=0, stage_count=1) == 1.0
    assert app._workflow_progress_value(stage_index=0, stage_count=3) == 0.0
    assert app._workflow_progress_value(stage_index=1, stage_count=3) == 0.5
    assert app._workflow_progress_value(stage_index=2, stage_count=3) == 1.0


def test_build_workflow_stage_rows_summarizes_stage_status():
    master = load_master_data(DATA_DIR)
    suite = generate_typical_workflow_suite(master)
    payload, _, _ = select_demo_payload(suite)
    workflow = app.build_demo_workflow_view_model(master, payload).workflow

    rows = app._build_workflow_stage_rows(workflow)

    assert len(rows) == workflow.stage_count
    assert rows[0]["stageName"] == workflow.stages[0].name
    assert rows[0]["hookCount"] == workflow.stages[0].view.summary.hook_count
    assert rows[0]["isValid"] is True


def test_build_workflow_transition_rows_summarizes_track_loco_and_spot_changes():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "initialVehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WFTRANS1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "",
            }
        ],
        "workflowStages": [
            {
                "name": "dispatch_work",
                "vehicleGoals": [
                    {"vehicleNo": "WFTRANS1", "targetTrack": "调棚", "isSpotting": ""}
                ],
            },
            {
                "name": "depot_spot",
                "vehicleGoals": [
                    {"vehicleNo": "WFTRANS1", "targetTrack": "大库", "isSpotting": "101"}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    workflow = app.build_demo_workflow_view_model(master, payload).workflow
    rows = app._build_workflow_transition_rows(workflow)

    assert rows == [
        {
            "stageIndex": 1,
            "stageName": "dispatch_work",
            "locoTransition": "机库 -> 调棚",
            "movedVehicles": "WFTRANS1(存5北->调棚)",
            "newWeighedVehicles": "无",
            "spotChanges": "无",
        },
        {
            "stageIndex": 2,
            "stageName": "depot_spot",
            "locoTransition": "调棚 -> 修1库内",
            "movedVehicles": "WFTRANS1(调棚->修1库内)",
            "newWeighedVehicles": "无",
            "spotChanges": "WFTRANS1(无->101)",
        },
    ]


def test_build_workflow_transition_rows_marks_newly_weighed_vehicles():
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
                "vehicleNo": "WFTRANSW1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "vehicleAttributes": "称重",
            }
        ],
        "workflowStages": [
            {
                "name": "weigh_stage",
                "vehicleGoals": [
                    {"vehicleNo": "WFTRANSW1", "targetTrack": "机库", "isSpotting": ""}
                ],
            },
            {
                "name": "departure",
                "vehicleGoals": [
                    {"vehicleNo": "WFTRANSW1", "targetTrack": "存4北", "isSpotting": ""}
                ],
            },
        ],
        "locoTrackName": "机库",
    }

    workflow = app.build_demo_workflow_view_model(master, payload).workflow
    rows = app._build_workflow_transition_rows(workflow)

    assert rows[0]["newWeighedVehicles"] == "WFTRANSW1"
    assert rows[1]["newWeighedVehicles"] == "无"


def test_track_map_legend_contains_active_changed_and_loco_labels():
    legend = app._track_map_legend_markdown()

    assert "Active Path" in legend
    assert "Changed Track" in legend
    assert "Loco Track" in legend


def test_build_comparison_panel_formats_solver_failure_summary():
    panel = app._build_comparison_panel(
        {
            "solverHookCount": None,
            "externalHookCount": 1,
            "hookCountDelta": None,
            "externalIsValid": False,
            "failedHookNos": [1],
            "solverError": "No solution found",
        }
    )

    assert panel["metrics"] == [
        {"label": "外部钩数", "value": 1},
        {"label": "求解器钩数", "value": "N/A"},
        {"label": "钩数差值", "value": "N/A"},
        {"label": "外部计划校验", "value": "FAIL"},
    ]
    assert panel["captions"] == ["失败钩号: 1", "求解器对比结果: No solution found"]


def test_build_comparison_panel_uses_demo_view_model_summary():
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
                "vehicleNo": "APP_CMP_1",
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
            "actionType": "ATTACH",
            "sourceTrack": "存5北",
            "targetTrack": "存5北",
            "vehicleNos": ["APP_CMP_1"],
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": "存5北",
            "targetTrack": "机库",
            "vehicleNos": ["APP_CMP_1"],
            "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        }
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload, compare_external_plan=True)
    panel = app._build_comparison_panel(view.comparison_summary)

    assert panel["metrics"] == [
        {"label": "外部钩数", "value": 2},
        {"label": "求解器钩数", "value": 2},
        {"label": "钩数差值", "value": 0},
        {"label": "外部计划校验", "value": "PASS"},
    ]
    assert panel["captions"] == []
