from pathlib import Path

from fzed_shunting.demo import view_model as view_model_module
from fzed_shunting.demo.view_model import build_demo_view_model, select_demo_payload
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.solver.profile import (
    VALIDATION_DEFAULT_BEAM_WIDTH,
    VALIDATION_DEFAULT_SOLVER,
    VALIDATION_DEFAULT_TIMEOUT_SECONDS,
    validation_retry_time_budget_ms,
    validation_time_budget_ms,
)
from fzed_shunting.solver.astar_solver import (
    RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC,
)
from fzed_shunting.solver.result import SolverResult
from fzed_shunting.solver.types import HookAction
from fzed_shunting.sim.generator import generate_typical_suite


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _mock_hook(source_track: str, target_track: str, vehicle_nos: list[str]) -> HookAction:
    return HookAction(
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=vehicle_nos,
        path_tracks=[source_track, target_track],
        action_type="DETACH",
    )


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


def _hook_steps(view):
    return [step for step in view.steps if step.hook]


def _detach_steps(view):
    return [step for step in view.steps if step.hook and step.hook.action_type == "DETACH"]


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

    assert view.summary.hook_count == 2
    assert view.summary.vehicle_count == 1
    assert view.summary.is_valid is True
    assert len(view.steps) == 3
    assert [step.hook.action_type for step in view.steps[1:] if step.hook] == ["ATTACH", "DETACH"]
    assert set(view.steps[2].changed_tracks) == {"存5北", "机库"}
    assert view.steps[2].hook.target_track == "机库"
    assert view.hook_plan[1].vehicle_count == 1
    assert view.hook_plan[1].route_length_m and view.hook_plan[1].route_length_m > 0
    assert view.hook_plan[1].remark
    assert view.track_map.track_nodes["存5北"].is_occupied is True
    assert view.track_map.track_nodes["机库"].is_occupied is False
    assert view.steps[2].track_map.active_path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert view.steps[2].track_map.changed_tracks == ["存5北", "机库"]
    assert view.steps[2].track_map.track_nodes["机库"].has_loco is True


def test_build_demo_view_model_defaults_match_validation_runner_profile(monkeypatch):
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
                "vehicleNo": "VIEW_PROFILE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    captured = {}

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
        )

    monkeypatch.setattr(view_model_module, "solve_with_simple_astar_result", fake_solve)

    view = build_demo_view_model(master, payload)

    assert view.summary.hook_count == 0
    assert captured["solver_mode"] == VALIDATION_DEFAULT_SOLVER
    assert captured["beam_width"] == VALIDATION_DEFAULT_BEAM_WIDTH
    assert captured["time_budget_ms"] == validation_time_budget_ms(
        VALIDATION_DEFAULT_TIMEOUT_SECONDS
    )
    assert captured["enable_depot_late_scheduling"] is False


def test_build_demo_view_model_retries_beam_like_validation_runner(monkeypatch):
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
                "vehicleNo": "VIEW_RETRY",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    calls = []

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) < 3:
            return SolverResult(
                plan=[],
                partial_plan=[],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=0.0,
                is_complete=False,
                fallback_stage=kwargs.get("solver_mode"),
            )
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
        )

    monkeypatch.setattr(view_model_module, "solve_with_simple_astar_result", fake_solve)

    view = build_demo_view_model(master, payload)

    assert view.summary.is_valid is True
    assert [call["beam_width"] for call in calls] == [8, 8, 16]
    assert calls[1]["time_budget_ms"] == validation_retry_time_budget_ms(
        validation_time_budget_ms(VALIDATION_DEFAULT_TIMEOUT_SECONDS)
    )
    assert calls[1]["near_goal_partial_resume_max_final_heuristic"] == (
        RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )


def test_build_demo_view_model_retries_pathological_complete_result(monkeypatch):
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
                "vehicleNo": "VIEW_PATHO",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    calls = []

    def fake_solve(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        if len(calls) == 1:
            return SolverResult(
                plan=[
                    _mock_hook("存5北", "存5北", ["VIEW_PATHO"])
                    for _ in range(130)
                ],
                expanded_nodes=0,
                generated_nodes=0,
                closed_nodes=0,
                elapsed_ms=0.0,
                is_complete=True,
                fallback_stage=kwargs.get("solver_mode"),
                debug_stats={
                    "plan_shape_metrics": {
                        "max_vehicle_touch_count": 90,
                    }
                },
            )
        return SolverResult(
            plan=[],
            expanded_nodes=0,
            generated_nodes=0,
            closed_nodes=0,
            elapsed_ms=0.0,
            is_complete=True,
            fallback_stage=kwargs.get("solver_mode"),
            debug_stats={"plan_shape_metrics": {"max_vehicle_touch_count": 20}},
        )

    monkeypatch.setattr(view_model_module, "solve_with_simple_astar_result", fake_solve)

    view = build_demo_view_model(master, payload)

    assert view.summary.is_valid is True
    assert view.summary.hook_count == 0
    assert [call["beam_width"] for call in calls] == [8, 8]
    assert calls[1]["near_goal_partial_resume_max_final_heuristic"] == (
        RECOVERY_NEAR_GOAL_PARTIAL_RESUME_MAX_FINAL_HEURISTIC
    )


def test_build_demo_view_model_counts_atomic_attach_and_detach_hooks():
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
                "vehicleNo": "V_ATOMIC_1",
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
            "vehicleNos": ["V_ATOMIC_1"],
            "pathTracks": ["存5北"],
        },
        {
            "hookNo": 2,
            "actionType": "DETACH",
            "sourceTrack": "存5北",
            "targetTrack": "机库",
            "vehicleNos": ["V_ATOMIC_1"],
            "pathTracks": ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        },
    ]

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)

    assert view.summary.hook_count == 2
    assert [step.hook.action_type for step in view.steps[1:] if step.hook] == ["ATTACH", "DETACH"]
    assert view.steps[0].loco_carry_vehicle_nos == []
    assert view.steps[1].loco_carry_vehicle_nos == ["V_ATOMIC_1"]
    assert view.steps[2].loco_carry_vehicle_nos == []


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
    assert [step.hook.target_track for step in _detach_steps(view)] == ["机库", "存4北"]


def test_build_demo_view_model_skips_external_plan_comparison_by_default(monkeypatch):
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
                "vehicleNo": "CMP_SKIP_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    plan_payload = _native_direct_plan(
        source_track="存5北",
        target_track="机库",
        vehicle_nos=["CMP_SKIP_1"],
        detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
    )

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("solve_with_simple_astar_result should not run for external plan replay by default")

    monkeypatch.setattr(view_model_module, "solve_with_simple_astar_result", fail_if_called)

    view = build_demo_view_model(master, payload, plan_payload=plan_payload)

    assert view.summary.is_valid is True
    assert view.comparison_summary is None


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
    assert view.summary.hook_count == 2


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
    assert view.summary.hook_count == 2


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
    assert _detach_steps(view)[0].spot_assignments == {"V3": "101"}
    assert "L19-修1尽头" in view.hook_plan[-1].reverse_branch_codes


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
    assert _detach_steps(view)[0].spot_assignments == {"V5": "调棚:1"}


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
    plan_payload = _native_direct_plan(
        source_track="存5北",
        target_track="修1库内",
        vehicle_nos=["V4"],
        detach_path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
    )

    view = build_demo_view_model(master, payload, plan_payload=plan_payload, compare_external_plan=True)

    assert view.summary.is_valid is False
    assert view.failed_hook_nos == [2]
    assert any("interference" in error.lower() for error in view.steps[2].verifier_errors)
    assert view.comparison_summary is not None
    assert view.comparison_summary["externalHookCount"] == 2
    assert view.comparison_summary["externalIsValid"] is False
    assert view.comparison_summary["failedHookNos"] == [2]
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
    plan_payload = _native_direct_plan(
        source_track="存5北",
        target_track="机库",
        vehicle_nos=["CMP1"],
        detach_path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
    )

    view = build_demo_view_model(master, payload, plan_payload=plan_payload, compare_external_plan=True)

    assert view.comparison_summary is not None
    assert view.comparison_summary["solverHookCount"] == 2
    assert view.comparison_summary["externalHookCount"] == 2
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
    detach_steps = _detach_steps(view)

    assert len(view.steps) == 5
    assert detach_steps[0].track_map.active_path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert detach_steps[0].track_map.track_nodes["机库"].is_occupied is True
    assert detach_steps[0].track_map.track_nodes["机库"].has_loco is True
    assert detach_steps[1].track_map.active_path_tracks == ["机库", "渡4", "临2", "临1", "渡2", "渡1", "存4北"]
    assert detach_steps[1].track_map.track_nodes["存4北"].is_occupied is True


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
    assert _detach_steps(view)[0].topology_graph.active_edge_keys == [
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
    detach_step = _detach_steps(view)[0]
    frames = detach_step.transition_frames

    assert len(frames) > len(detach_step.hook.path_tracks)
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
