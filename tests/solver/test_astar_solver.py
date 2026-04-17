from pathlib import Path
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import (
    SolverResult,
    _build_repair_plan_input,
    _candidate_repair_cut_points,
    _is_better_plan,
    solve_with_simple_astar,
    solve_with_simple_astar_result,
)
from fzed_shunting.verify.replay import ReplayState, build_initial_state, replay_plan
from fzed_shunting.solver.astar_solver import _state_key
from fzed_shunting.solver.types import HookAction


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_simple_astar_solves_direct_single_vehicle_case():
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
                "vehicleNo": "E1",
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

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
    )

    assert len(plan) == 1
    assert plan[0].path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert replay.final_state.track_sequences["机库"] == ["E1"]


def test_simple_astar_solves_two_single_car_goals():
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
                "vehicleNo": "E2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "E3",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
    )

    assert len(plan) == 2
    assert replay.final_state.track_sequences["机库"] == ["E2"]
    assert replay.final_state.track_sequences["存4北"] == ["E3"]


def test_simple_astar_result_can_return_debug_stats():
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
                "vehicleNo": "ASTDBG_BLOCK",
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
                "vehicleNo": "ASTDBG_GO",
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
    initial = build_initial_state(normalized)
    debug_stats: dict = {}

    result = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        debug_stats=debug_stats,
    )

    assert result.plan
    assert result.debug_stats == debug_stats
    assert result.debug_stats["expanded_states"] >= 1
    assert result.debug_stats["generated_nodes"] == result.generated_nodes
    assert result.debug_stats["move_generation_calls"] >= 1
    assert result.debug_stats["candidate_moves_total"] >= len(result.plan)
    assert result.debug_stats["candidate_direct_moves"] >= 1
    assert result.debug_stats["candidate_staging_moves"] >= 1


def test_simple_astar_solves_oil_and_shot_targets_when_missing_segment_is_40m_placeholder():
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
                "vehicleNo": "E4",
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
            {"trackName": "油", "trackDistance": 124},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E4O",
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
    shot_initial = build_initial_state(shot_normalized)
    oil_normalized = normalize_plan_input(oil_payload, master)
    oil_initial = build_initial_state(oil_normalized)

    shot_plan = solve_with_simple_astar(shot_normalized, shot_initial, master=master)
    oil_plan = solve_with_simple_astar(oil_normalized, oil_initial, master=master)

    assert len(shot_plan) == 1
    assert shot_plan[0].target_track == "抛"
    assert shot_plan[0].path_tracks == ["存5北", "存5南", "渡8", "渡9", "渡10", "抛"]
    assert len(oil_plan) == 1
    assert oil_plan[0].target_track == "油"
    assert oil_plan[0].path_tracks == ["机库", "渡4", "渡5", "机北", "机棚", "临3", "油"]


def test_simple_astar_rejects_two_vehicles_competing_for_same_exact_spot():
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
                "vehicleNo": "E5",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "E6",
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
    initial = build_initial_state(normalized)

    try:
        solve_with_simple_astar(normalized, initial, master=master)
    except ValueError as exc:
        assert "No solution found" in str(exc)
    else:
        raise AssertionError("expected no solution for duplicate exact depot spot demand")


def test_simple_astar_rejects_close_door_vehicle_without_valid_cun4bei_position():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)

    try:
        solve_with_simple_astar(normalized, initial, master=master)
    except ValueError as exc:
        assert "No solution found" in str(exc)
    else:
        raise AssertionError("expected no solution for invalid close-door final placement")


def test_simple_astar_avoids_large_non_cun4bei_hook_with_close_door_first():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存3", "trackDistance": 258.5},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(idx + 1),
                "vehicleModel": "棚车",
                "vehicleNo": f"LCD{idx + 1:02d}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存3",
                "isSpotting": "",
                "vehicleAttributes": "关门车" if idx == 0 else "",
            }
            for idx in range(11)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    plan = solve_with_simple_astar(normalized, initial, master=master)

    assert len(plan) >= 2
    assert all(
        not (
            move.target_track != "存4北"
            and len(move.vehicle_nos) > 10
            and move.vehicle_nos[0] == "LCD01"
        )
        for move in plan
    )


def test_simple_astar_can_clear_front_blocker_via_temporary_track():
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
                "vehicleNo": "E7",
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
                "vehicleNo": "E8",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
    )

    assert len(plan) == 3
    assert (plan[0].source_track, plan[0].target_track, plan[0].vehicle_nos) == (
        "存5北",
        "临1",
        ["E7"],
    )
    assert replay.final_state.track_sequences["存5北"] == ["E7"]
    assert replay.final_state.track_sequences["机库"] == ["E8"]


def test_simple_astar_can_clear_front_blocker_via_cun4nan_staging():
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
                "vehicleNo": "E7A",
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
                "vehicleNo": "E8A",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
    )

    assert len(plan) == 3
    assert (plan[0].source_track, plan[0].target_track, plan[0].vehicle_nos) == (
        "存5北",
        "存4南",
        ["E7A"],
    )
    assert replay.final_state.track_sequences["存5北"] == ["E7A"]
    assert replay.final_state.track_sequences["机库"] == ["E8A"]


def test_simple_astar_can_clear_interfering_track_via_temporary_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E9",
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
                "vehicleNo": "E10",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
        plan_input=normalized,
    )

    assert len(plan) == 3
    assert (plan[0].source_track, plan[0].target_track, plan[0].vehicle_nos) == (
        "存5南",
        "临1",
        ["E10"],
    )
    assert replay.final_state.track_sequences["修1库内"] == ["E9"]
    assert replay.final_state.track_sequences["存5南"] == ["E10"]


def test_weighted_astar_solves_simple_case():
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
                "vehicleNo": "W1",
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

    plan = solve_with_simple_astar(
        normalized,
        initial,
        master=master,
        solver_mode="weighted",
        heuristic_weight=2.5,
    )

    assert len(plan) == 1
    assert plan[0].target_track == "机库"


def test_beam_search_solves_simple_case():
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
                "vehicleNo": "B1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "B2",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(
        normalized,
        initial,
        master=master,
        solver_mode="beam",
        beam_width=8,
    )

    assert len(plan) == 2


def test_lns_solver_mode_solves_simple_case():
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
                "vehicleNo": "L1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "L2",
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
    initial = build_initial_state(normalized)

    plan = solve_with_simple_astar(
        normalized,
        initial,
        master=master,
        solver_mode="lns",
        beam_width=8,
    )

    assert len(plan) == 2


def test_lns_recomputes_cut_points_after_plan_improvement():
    initial = ReplayState(
        track_sequences={"存5北": ["L1", "L2"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    incumbent_plan = [
        HookAction(
            source_track="存5北",
            target_track="临1",
            vehicle_nos=["L1"],
            path_tracks=["存5北", "临1"],
        ),
        HookAction(
            source_track="临1",
            target_track="机库",
            vehicle_nos=["L1"],
            path_tracks=["临1", "机库"],
        ),
    ]
    improved_suffix = [
        HookAction(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["L1", "L2"],
            path_tracks=["存5北", "机库"],
        ),
    ]
    cut_point_calls: list[list[str]] = []
    replay_call_count = 0
    search_call_count = 0

    def fake_replay_plan(*args, **kwargs):
        nonlocal replay_call_count
        replay_call_count += 1
        return type(
            "ReplayResult",
            (),
            {
                "snapshots": [
                    initial,
                    ReplayState(
                        track_sequences={"临1": ["L1"], "存5北": ["L2"]},
                        loco_track_name="临1",
                        weighed_vehicle_nos=set(),
                        spot_assignments={},
                    ),
                    ReplayState(
                        track_sequences={"机库": ["L1"], "存5北": ["L2"]},
                        loco_track_name="机库",
                        weighed_vehicle_nos={"L1"},
                        spot_assignments={},
                    ),
                ]
            },
        )()

    def fake_candidate_repair_cut_points(plan, repair_passes):
        cut_point_calls.append([f"{move.source_track}->{move.target_track}" for move in plan])
        if len(cut_point_calls) == 1:
            return [0]
        return []

    def fake_solve_search_result(*args, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return SolverResult(
                plan=incumbent_plan,
                expanded_nodes=1,
                generated_nodes=1,
                closed_nodes=1,
                elapsed_ms=1.0,
            )
        return SolverResult(
            plan=improved_suffix,
            expanded_nodes=2,
            generated_nodes=2,
            closed_nodes=2,
            elapsed_ms=1.0,
        )

    with patch("fzed_shunting.solver.astar_solver._solve_search_result", side_effect=fake_solve_search_result):
        with patch("fzed_shunting.solver.astar_solver._candidate_repair_cut_points", side_effect=fake_candidate_repair_cut_points):
            with patch("fzed_shunting.solver.astar_solver.replay_plan", side_effect=fake_replay_plan):
                result = solve_with_simple_astar_result(
                    plan_input=normalize_plan_input(
                        {
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
                                    "vehicleNo": "L1",
                                    "repairProcess": "段修",
                                    "vehicleLength": 14.3,
                                    "targetTrack": "机库",
                                    "isSpotting": "",
                                    "vehicleAttributes": "",
                                },
                                {
                                    "trackName": "存5北",
                                    "order": "2",
                                    "vehicleModel": "棚车",
                                    "vehicleNo": "L2",
                                    "repairProcess": "段修",
                                    "vehicleLength": 14.3,
                                    "targetTrack": "机库",
                                    "isSpotting": "",
                                    "vehicleAttributes": "",
                                },
                            ],
                            "locoTrackName": "机库",
                        },
                        load_master_data(DATA_DIR),
                    ),
                    initial_state=initial,
                    master=load_master_data(DATA_DIR),
                    solver_mode="lns",
                    beam_width=4,
                )

    assert len(result.plan) == 1
    assert cut_point_calls == [["存5北->临1", "临1->机库"], ["存5北->机库"]]
    assert replay_call_count == 2


def test_candidate_repair_cut_points_prioritize_repeated_touch_before_unrelated_long_path():
    plan = [
        HookAction(
            source_track="存5北",
            target_track="调北",
            vehicle_nos=["HOT1"],
            path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "调北"],
        ),
        HookAction(
            source_track="调北",
            target_track="临1",
            vehicle_nos=["HOT1"],
            path_tracks=["调北", "渡4", "临2", "临1"],
        ),
        HookAction(
            source_track="临1",
            target_track="存4北",
            vehicle_nos=["HOT1"],
            path_tracks=["临1", "渡2", "渡1", "存4北"],
        ),
        HookAction(
            source_track="存5南",
            target_track="抛",
            vehicle_nos=["LONG1"],
            path_tracks=["存5南", "渡8", "渡9", "渡10", "抛"],
        ),
    ]

    ranked = _candidate_repair_cut_points(plan, repair_passes=4)

    assert ranked[:4] == [1, 2, 0, 3]


def test_build_repair_plan_input_freezes_already_satisfied_vehicle_goal_at_snapshot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "FRZ1",
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
                "vehicleNo": "FRZ2",
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
    snapshot = ReplayState(
        track_sequences={"机库": ["FRZ1"], "存5北": ["FRZ2"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )

    localized = _build_repair_plan_input(normalized, snapshot)
    goals = {vehicle.vehicle_no: vehicle.goal for vehicle in localized.vehicles}

    assert goals["FRZ1"].target_mode == "TRACK"
    assert goals["FRZ1"].target_track == "机库"
    assert goals["FRZ1"].allowed_target_tracks == ["机库"]
    assert goals["FRZ2"].target_track == "存4北"


def test_is_better_plan_uses_branch_count_before_path_length():
    class FakeRoute:
        def __init__(self, branch_count: int, total_length_m: float):
            self.branch_codes = [f"B{idx}" for idx in range(branch_count)]
            self.total_length_m = total_length_m

    class FakeRouteOracle:
        def resolve_route(self, source_track: str, target_track: str):
            if source_track == "存5北" and target_track == "机库":
                return FakeRoute(branch_count=4, total_length_m=120.0)
            if source_track == "存5北" and target_track == "存4北":
                return FakeRoute(branch_count=2, total_length_m=150.0)
            return FakeRoute(branch_count=3, total_length_m=100.0)

    incumbent = [
        HookAction(
            source_track="存5北",
            target_track="机库",
            vehicle_nos=["Q1"],
            path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        )
    ]
    candidate = [
        HookAction(
            source_track="存5北",
            target_track="存4北",
            vehicle_nos=["Q1"],
            path_tracks=["存5北", "渡1", "存4北"],
        )
    ]

    assert _is_better_plan(candidate, incumbent, FakeRouteOracle()) is True


def test_simple_astar_assigns_work_area_spot_for_dispatch_goal():
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
                "vehicleNo": "WG1",
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

    plan = solve_with_simple_astar(normalized, initial, master=master)
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": item.action_type,
                "sourceTrack": item.source_track,
                "targetTrack": item.target_track,
                "vehicleNos": item.vehicle_nos,
                "pathTracks": item.path_tracks,
            }
            for idx, item in enumerate(plan, start=1)
        ],
        plan_input=normalized,
    )

    assert replay.final_state.track_sequences["调棚"] == ["WG1"]
    assert replay.final_state.spot_assignments["WG1"] == "调棚:1"


def test_state_key_distinguishes_weighed_state():
    base = ReplayState(
        track_sequences={"存5北": ["S1"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    weighed = ReplayState(
        track_sequences={"存5北": ["S1"]},
        loco_track_name="机库",
        weighed_vehicle_nos={"S1"},
        spot_assignments={},
    )

    assert _state_key(base) != _state_key(weighed)


def test_state_key_distinguishes_spot_assignments():
    first = ReplayState(
        track_sequences={"修1库内": ["S1"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={"S1": "101"},
    )
    second = ReplayState(
        track_sequences={"修1库内": ["S1"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={"S1": "102"},
    )

    assert _state_key(first) != _state_key(second)
