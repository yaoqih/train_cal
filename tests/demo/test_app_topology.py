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
        view.steps[1].topology_graph,
        view.steps[1].track_map,
    )

    assert dot.startswith("graph shunting_topology {")
    assert '"存5北" -- "渡1" [color="#1d6f6d"' in dot
    assert '"渡1" -- "存4北" [color="#c5c0b5"' in dot
    assert '"机库" [label="机库\\n机车 / 占用 1"' in dot


def test_build_topology_svg_contains_geometry_tracks_and_motion_marker():
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
        view.steps[1].topology_graph,
        view.steps[1].track_map,
        hook=view.steps[1].hook,
        transition_frame=view.steps[1].transition_frames[3],
    )

    assert svg.startswith("<svg")
    assert 'class="track-path active-path"' in svg
    assert 'class="moving-block-marker"' in svg
    assert ">存5北<" in svg
    assert ">机库<" in svg


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
        view.steps[1].topology_graph,
        view.steps[1].track_map,
        hook=view.steps[1].hook,
        animate=True,
    )

    assert "<animateMotion" in svg
    assert "route-motion-path" in svg


def test_app_module_exposes_workflow_renderer():
    assert hasattr(app, "_render_workflow_demo")


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
                    {"vehicleNo": "WFTRANS1", "targetTrack": "调棚", "isSpotting": "是"}
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
            "spotChanges": "WFTRANS1(无->调棚:1)",
        },
        {
            "stageIndex": 2,
            "stageName": "depot_spot",
            "locoTransition": "调棚 -> 修1库内",
            "movedVehicles": "WFTRANS1(调棚->修1库内)",
            "newWeighedVehicles": "无",
            "spotChanges": "WFTRANS1(调棚:1->101)",
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
            "actionType": "PUT",
            "sourceTrack": "存5北",
            "targetTrack": "机库",
            "vehicleNos": ["APP_CMP_1"],
            "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        }
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)
    panel = app._build_comparison_panel(view.comparison_summary)

    assert panel["metrics"] == [
        {"label": "外部钩数", "value": 1},
        {"label": "求解器钩数", "value": 1},
        {"label": "钩数差值", "value": 0},
        {"label": "外部计划校验", "value": "PASS"},
    ]
    assert panel["captions"] == []
