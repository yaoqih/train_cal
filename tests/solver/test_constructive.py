import json
from collections import deque
from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.constructive import solve_constructive, ConstructiveResult
from fzed_shunting.solver.heuristic import make_state_heuristic_real_hook
from fzed_shunting.solver.state import ReplayState
from fzed_shunting.verify.replay import build_initial_state, replay_plan
from fzed_shunting.verify.plan_verifier import verify_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _simple_payload(vehicles: list[dict]) -> dict:
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def _run(payload: dict) -> tuple[ConstructiveResult, object, object, object]:
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    result = solve_constructive(normalized, initial, master=master, max_iterations=500)
    return result, master, normalized, initial


def test_constructive_returns_plan_for_single_vehicle_track_goal():
    payload = _simple_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    result, master, normalized, initial = _run(payload)

    assert result.reached_goal is True
    assert len(result.plan) == 2
    assert [move.action_type for move in result.plan] == ["ATTACH", "DETACH"]
    assert result.plan[0].source_track == "存5北"
    assert result.plan[-1].target_track == "存4北"

    hook_plan = [
        {
            "hookNo": 1,
            "actionType": m.action_type,
            "sourceTrack": m.source_track,
            "targetTrack": m.target_track,
            "vehicleNos": list(m.vehicle_nos),
            "pathTracks": list(m.path_tracks),
        }
        for m in result.plan
    ]
    report = verify_plan(master, normalized, hook_plan, initial_state_override=initial)
    assert report.is_valid is True


def test_constructive_produces_plan_for_multi_vehicle_same_goal():
    payload = _simple_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "E2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "E3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    result, _, _, _ = _run(payload)

    assert result.reached_goal is True
    assert len(result.plan) == 2
    assert [move.action_type for move in result.plan] == ["ATTACH", "DETACH"]
    assert set(result.plan[-1].vehicle_nos) == {"E1", "E2", "E3"}


def test_constructive_handles_weigh_vehicle_via_jiku():
    payload = _simple_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "W1",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
        ]
    )
    result, master, normalized, initial = _run(payload)
    assert result.reached_goal is True
    # Weigh vehicles must pass through 机库 first
    assert [move.action_type for move in result.plan] == ["ATTACH", "DETACH", "ATTACH", "DETACH"]
    assert [move.target_track for move in result.plan if move.action_type == "DETACH"] == ["机库", "存4北"]

    hook_plan = [
        {
            "hookNo": idx,
            "actionType": m.action_type,
            "sourceTrack": m.source_track,
            "targetTrack": m.target_track,
            "vehicleNos": list(m.vehicle_nos),
            "pathTracks": list(m.path_tracks),
        }
        for idx, m in enumerate(result.plan, start=1)
    ]
    report = verify_plan(master, normalized, hook_plan, initial_state_override=initial)
    assert report.is_valid is True


def test_constructive_partial_rewinds_to_last_empty_carry_checkpoint():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PARTIAL_BLOCK",
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
                "vehicleNo": "PARTIAL_GO",
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

    result = solve_constructive(
        normalized,
        initial,
        master=master,
        max_iterations=3,
        max_backtracks=0,
    )

    assert result.reached_goal is False
    assert [move.action_type for move in result.plan] == ["ATTACH", "DETACH"]

    replay = replay_plan(
        initial,
        [
            {
                "hookNo": idx,
                "actionType": move.action_type,
                "sourceTrack": move.source_track,
                "targetTrack": move.target_track,
                "vehicleNos": list(move.vehicle_nos),
                "pathTracks": list(move.path_tracks),
            }
            for idx, move in enumerate(result.plan, start=1)
        ],
        plan_input=normalized,
    )
    assert replay.final_state.loco_carry == ()


def test_constructive_respects_close_door_four_north_constraint():
    payload = _simple_payload(
        [
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "N1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "N2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "N3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CD",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            },
        ]
    )
    result, master, normalized, initial = _run(payload)
    assert result.reached_goal is True

    hook_plan = [
        {
            "hookNo": idx,
            "actionType": m.action_type,
            "sourceTrack": m.source_track,
            "targetTrack": m.target_track,
            "vehicleNos": list(m.vehicle_nos),
            "pathTracks": list(m.path_tracks),
        }
        for idx, m in enumerate(result.plan, start=1)
    ]
    report = verify_plan(master, normalized, hook_plan, initial_state_override=initial)
    assert report.is_valid is True
    # final 存4北 sequence must not place CD in positions 1-3
    replay = replay_plan(initial, hook_plan, plan_input=normalized)
    final_4bei = replay.final_state.track_sequences.get("存4北", [])
    assert "CD" in final_4bei
    final_seq_pos = final_4bei.index("CD") + 1
    assert final_seq_pos >= 4, f"close-door landed at pos {final_seq_pos}, must be >=4"


def test_constructive_always_returns_plan_even_on_hard_case():
    """Regression: hard cases from external 109 should never return empty plan."""
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "external_validation_inputs"
            / "validation_20260104Z.json"
        ).read_text(encoding="utf-8")
    )
    result, _, _, _ = _run(payload)
    # May not reach goal, but plan must be non-empty
    assert len(result.plan) > 0, "constructive must always return a non-empty plan"


def test_constructive_reports_elapsed_and_iterations():
    payload = _simple_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    result, _, _, _ = _run(payload)
    assert result.iterations >= 0
    assert result.elapsed_ms >= 0
    assert result.debug_stats is not None


class TestScoreMoveSatisfiedProtection:
    """Tests for the identity-goal displacement protection in _score_move."""

    def test_score_move_penalizes_identity_goal_displacement(self):
        """A move that takes a satisfied vehicle OUT of its allowed target gets
        tier+100 so it's effectively last-resort."""
        from fzed_shunting.solver.constructive import _score_move
        from fzed_shunting.solver.types import HookAction
        from fzed_shunting.io.normalize_input import (
            GoalSpec, NormalizedVehicle,
        )
        from fzed_shunting.verify.replay import ReplayState

        v1 = NormalizedVehicle(
            current_track="存1", order=1, vehicle_model="敞车", vehicle_no="V1",
            repair_process="段修", vehicle_length=12.0,
            goal=GoalSpec(target_mode="TRACK", target_track="存1",
                          allowed_target_tracks=["存1"]),
        )
        state = ReplayState(
            track_sequences={"存1": ["V1"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        move = HookAction(
            source_track="存1", target_track="临2",
            vehicle_nos=["V1"], path_tracks=["存1", "临2"], action_type="PUT",
        )
        score, tier = _score_move(
            move=move, state=state, vehicle_by_no={"V1": v1},
            goal_tracks_needed={"存1"},
        )
        assert score[0] >= 100, f"Identity-goal displacement should have tier >= 100, got {score[0]}"

    def test_score_move_does_not_penalize_non_displacement(self):
        """A normal move not touching satisfied vehicles has tier < 100."""
        from fzed_shunting.solver.constructive import _score_move
        from fzed_shunting.solver.types import HookAction
        from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
        from fzed_shunting.verify.replay import ReplayState

        v1 = NormalizedVehicle(
            current_track="存5北", order=1, vehicle_model="敞车", vehicle_no="V1",
            repair_process="段修", vehicle_length=12.0,
            goal=GoalSpec(target_mode="TRACK", target_track="存4北",
                          allowed_target_tracks=["存4北"]),
        )
        state = ReplayState(
            track_sequences={"存5北": ["V1"]},
            loco_track_name="机库", weighed_vehicle_nos=set(), spot_assignments={},
        )
        move = HookAction(
            source_track="存5北", target_track="存4北",
            vehicle_nos=["V1"], path_tracks=["存5北", "存4北"], action_type="PUT",
        )
        score, tier = _score_move(
            move=move, state=state, vehicle_by_no={"V1": v1},
            goal_tracks_needed={"存4北"},
        )
        assert score[0] < 100, f"Non-displacement should have tier < 100, got {score[0]}"

    def test_score_move_does_not_penalize_unsatisfied_source(self):
        """A move from a non-target track (vehicle not satisfied at source) is
        NOT a 'displacement'. Should not get +100."""
        from fzed_shunting.solver.constructive import _score_move
        from fzed_shunting.solver.types import HookAction
        from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
        from fzed_shunting.verify.replay import ReplayState

        v1 = NormalizedVehicle(
            current_track="临1", order=1, vehicle_model="敞车", vehicle_no="V1",
            repair_process="段修", vehicle_length=12.0,
            goal=GoalSpec(target_mode="TRACK", target_track="存1",
                          allowed_target_tracks=["存1"]),
        )
        state = ReplayState(
            track_sequences={"临1": ["V1"]},
            loco_track_name="机库", weighed_vehicle_nos=set(), spot_assignments={},
        )
        move = HookAction(
            source_track="临1", target_track="临2",
            vehicle_nos=["V1"], path_tracks=["临1", "临2"], action_type="PUT",
        )
        score, tier = _score_move(
            move=move, state=state, vehicle_by_no={"V1": v1},
            goal_tracks_needed={"存1"},
        )
        assert score[0] < 100, f"Transit from non-target should not trigger protection, got {score[0]}"

    def test_score_move_penalizes_spot_matched_displacement(self):
        """A SPOT vehicle at its target with correct spot should be protected."""
        from fzed_shunting.solver.constructive import _score_move
        from fzed_shunting.solver.types import HookAction
        from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
        from fzed_shunting.verify.replay import ReplayState

        v1 = NormalizedVehicle(
            current_track="修3库内", order=1, vehicle_model="敞车", vehicle_no="S003",
            repair_process="段修", vehicle_length=17.0,
            goal=GoalSpec(
                target_mode="SPOT",
                target_track="修3库内",
                allowed_target_tracks=["修3库内"],
                target_spot_code="305",
            ),
        )
        state = ReplayState(
            track_sequences={"修3库内": ["S003"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={"S003": "305"},
        )
        move = HookAction(
            source_track="修3库内", target_track="临4",
            vehicle_nos=["S003"], path_tracks=["修3库内", "临4"], action_type="PUT",
        )
        score, tier = _score_move(
            move=move, state=state, vehicle_by_no={"S003": v1},
            goal_tracks_needed={"修3库内"},
        )
        assert score[0] >= 100, f"SPOT-matched displacement should have tier >= 100, got {score[0]}"

    def test_score_move_allows_spot_wrong_displacement(self):
        """A SPOT vehicle at target track but WRONG spot is NOT fully satisfied
        — moving it out is legitimate (e.g., to free the spot), no penalty."""
        from fzed_shunting.solver.constructive import _score_move
        from fzed_shunting.solver.types import HookAction
        from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
        from fzed_shunting.verify.replay import ReplayState

        v1 = NormalizedVehicle(
            current_track="修3库内", order=1, vehicle_model="敞车", vehicle_no="S003",
            repair_process="段修", vehicle_length=17.0,
            goal=GoalSpec(
                target_mode="SPOT",
                target_track="修3库内",
                allowed_target_tracks=["修3库内"],
                target_spot_code="305",
            ),
        )
        state = ReplayState(
            track_sequences={"修3库内": ["S003"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={"S003": "301"},  # wrong spot
        )
        move = HookAction(
            source_track="修3库内", target_track="临4",
            vehicle_nos=["S003"], path_tracks=["修3库内", "临4"], action_type="PUT",
        )
        score, tier = _score_move(
            move=move, state=state, vehicle_by_no={"S003": v1},
            goal_tracks_needed={"修3库内"},
        )
        assert score[0] < 100, f"SPOT wrong-spot displacement should not be protected, got {score[0]}"


class TestBacktrackingConstructive:
    """Tests for W3-N bounded-backtracking ``solve_constructive``."""

    def test_zero_backtrack_when_greedy_succeeds(self):
        """Easy scenario: greedy reaches goal in one sweep; no backtracks used."""
        payload = _simple_payload(
            [
                {
                    "trackName": "存5北",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": "E1",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                },
            ]
        )
        master = load_master_data(DATA_DIR)
        normalized = normalize_plan_input(payload, master)
        initial = build_initial_state(normalized)
        debug_stats: dict = {}
        result = solve_constructive(
            normalized,
            initial,
            master=master,
            max_iterations=500,
            debug_stats=debug_stats,
        )
        assert result.reached_goal is True
        assert debug_stats.get("constructive_backtrack_count") == 0

    def test_max_backtracks_limit(self):
        """Hard scenario: even backtracking fails. Returns best-effort partial
        plan, does not exceed ``max_backtracks=2``."""
        import json
        payload = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "data"
                / "validation_inputs"
                / "positive"
                / "case_3_3_spot_203_mid.json"
            ).read_text(encoding="utf-8")
        )
        master = load_master_data(DATA_DIR)
        normalized = normalize_plan_input(payload, master)
        initial = build_initial_state(normalized)
        debug_stats: dict = {}
        result = solve_constructive(
            normalized,
            initial,
            master=master,
            max_iterations=200,
            stuck_threshold=15,
            max_backtracks=2,
            time_budget_ms=20_000,
            debug_stats=debug_stats,
        )
        # Should not exceed max_backtracks=2
        assert debug_stats.get("constructive_backtrack_count", 0) <= 2
        # Best-effort partial plan is always returned (may or may not reach goal)
        assert len(result.plan) > 0

    def test_backtrack_structure_runs_multiple_attempts(self):
        """When the scenario isn't solvable by pure greedy, the backtracking
        loop should actually run more than 1 attempt (backtrack_count >= 1)."""
        import json
        payload = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "data"
                / "validation_inputs"
                / "positive"
                / "case_3_3_spot_203_mid.json"
            ).read_text(encoding="utf-8")
        )
        master = load_master_data(DATA_DIR)
        normalized = normalize_plan_input(payload, master)
        initial = build_initial_state(normalized)
        debug_stats: dict = {}
        result = solve_constructive(
            normalized,
            initial,
            master=master,
            max_iterations=500,
            stuck_threshold=30,
            max_backtracks=3,
            time_budget_ms=30_000,
            debug_stats=debug_stats,
        )
        # Greedy gets stuck here → backtracking should trigger at least once.
        # (If this case ever gets unstuck by greedy alone, the test should be
        # updated with a different scenario that still requires backtracking.)
        if not result.reached_goal:
            assert debug_stats.get("constructive_backtrack_count", 0) >= 1

    def test_result_contains_debug_backtrack_count(self):
        """``debug_stats`` should always include ``constructive_backtrack_count``."""
        payload = _simple_payload(
            [
                {
                    "trackName": "存5北",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": "E1",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                },
            ]
        )
        master = load_master_data(DATA_DIR)
        normalized = normalize_plan_input(payload, master)
        initial = build_initial_state(normalized)
        debug_stats: dict = {}
        result = solve_constructive(
            normalized, initial, master=master,
            max_iterations=500, debug_stats=debug_stats,
        )
        assert "constructive_backtrack_count" in debug_stats
        # Also surfaced on the returned result
        assert result.debug_stats is not None


def test_score_native_move_prefers_short_random_depot_preferred_track_over_fallback():
    from fzed_shunting.solver.constructive import _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "S1",
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
    carry_state = initial.model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": ("S1",),
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)

    move_pref = HookAction(
        source_track="存5北",
        target_track="修1库内",
        vehicle_nos=["S1"],
        path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        action_type="DETACH",
    )
    move_fallback = HookAction(
        source_track="存5北",
        target_track="修3库内",
        vehicle_nos=["S1"],
        path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡12", "渡13", "修3库外", "修3库内"],
        action_type="DETACH",
    )
    pref_state = _apply_move(state=carry_state, move=move_pref, plan_input=normalized, vehicle_by_no=vehicle_by_no)
    fallback_state = _apply_move(state=carry_state, move=move_fallback, plan_input=normalized, vehicle_by_no=vehicle_by_no)

    pref_score, _ = _score_native_move(
        move=move_pref,
        state=carry_state,
        next_state=pref_state,
        plan_input=normalized,
        current_heuristic=heuristic(carry_state),
        next_heuristic=heuristic(pref_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"修1库内", "修2库内"},
        satisfied_by_track={},
    )
    fallback_score, _ = _score_native_move(
        move=move_fallback,
        state=carry_state,
        next_state=fallback_state,
        plan_input=normalized,
        current_heuristic=heuristic(carry_state),
        next_heuristic=heuristic(fallback_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"修1库内", "修2库内"},
        satisfied_by_track={},
    )

    assert pref_score < fallback_score


def test_score_native_move_penalizes_fallback_detach_when_preferred_target_remains_feasible():
    from fzed_shunting.solver.constructive import _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SHORT_PREF",
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
    carry_state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": ("SHORT_PREF",),
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)

    preferred_move = HookAction(
        source_track="存5北",
        target_track="修1库内",
        vehicle_nos=["SHORT_PREF"],
        path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡11", "修1库外", "修1库内"],
        action_type="DETACH",
    )
    fallback_move = HookAction(
        source_track="存5北",
        target_track="修4库内",
        vehicle_nos=["SHORT_PREF"],
        path_tracks=["存5北", "存5南", "渡8", "渡9", "渡10", "联7", "渡12", "渡13", "修4库外", "修4库内"],
        action_type="DETACH",
    )

    preferred_state = _apply_move(
        state=carry_state,
        move=preferred_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    fallback_state = _apply_move(
        state=carry_state,
        move=fallback_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    preferred_score, _ = _score_native_move(
        move=preferred_move,
        state=carry_state,
        next_state=preferred_state,
        plan_input=normalized,
        current_heuristic=heuristic(carry_state),
        next_heuristic=heuristic(preferred_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"修1库内", "修2库内"},
        satisfied_by_track={},
    )
    fallback_score, _ = _score_native_move(
        move=fallback_move,
        state=carry_state,
        next_state=fallback_state,
        plan_input=normalized,
        current_heuristic=heuristic(carry_state),
        next_heuristic=heuristic(fallback_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"修1库内", "修2库内"},
        satisfied_by_track={},
    )

    assert preferred_score < fallback_score


def test_score_native_move_keeps_goal_detach_high_priority_even_when_heuristic_is_flat():
    from fzed_shunting.solver.constructive import _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "存1", "trackDistance": 113},
        ],
        "vehicleInfo": [
            {
                "trackName": "预修",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "GOAL_PRE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "预修",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_PEER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ATTACH_PEER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    carry_state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存1": ["ATTACH_PEER"]},
            "loco_track_name": "存5北",
            "loco_carry": ("CARRY_PEER", "GOAL_PRE"),
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)

    goal_move = HookAction(
        source_track="存5北",
        target_track="预修",
        vehicle_nos=["GOAL_PRE"],
        path_tracks=["存5北", "存5南", "渡8", "预修"],
        action_type="DETACH",
    )
    attach_move = HookAction(
        source_track="存1",
        target_track="存1",
        vehicle_nos=["ATTACH_PEER"],
        path_tracks=["存1"],
        action_type="ATTACH",
    )

    goal_state = _apply_move(
        state=carry_state,
        move=goal_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    attach_state = _apply_move(
        state=carry_state,
        move=attach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    current_h = heuristic(carry_state)
    assert heuristic(goal_state) == current_h

    goal_score, goal_tier = _score_native_move(
        move=goal_move,
        state=carry_state,
        next_state=goal_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(goal_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"预修"},
        satisfied_by_track={},
    )
    attach_score, attach_tier = _score_native_move(
        move=attach_move,
        state=carry_state,
        next_state=attach_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(attach_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"预修"},
        satisfied_by_track={},
    )

    assert goal_tier < attach_tier
    assert goal_score < attach_score


def test_move_exposes_buried_goal_seeker_recognizes_same_track_preferred_violation():
    from fzed_shunting.solver.constructive import _move_exposes_buried_goal_seeker
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修4库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修4库内",
                "order": "2",
                "vehicleModel": "C70",
                "vehicleNo": "SHORT_PREF",
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
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    move = HookAction(
        source_track="修4库内",
        target_track="临1",
        vehicle_nos=["BLOCKER"],
        path_tracks=["修4库内", "临1"],
        action_type="ATTACH",
    )

    assert _move_exposes_buried_goal_seeker(
        move=move,
        state=state,
        vehicle_by_no=vehicle_by_no,
        plan_input=normalized,
    ) is True


def test_move_clears_goal_blocker_does_not_treat_same_track_weigh_pending_vehicle_as_settled():
    from fzed_shunting.solver.constructive import _move_clears_goal_blocker
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "FRONT_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "2",
                "vehicleModel": "敞车",
                "vehicleNo": "WEIGH_PENDING",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    move = HookAction(
        source_track="存1",
        target_track="临1",
        vehicle_nos=["FRONT_BLOCK"],
        path_tracks=["存1", "临1"],
        action_type="ATTACH",
    )

    assert _move_clears_goal_blocker(
        move=move,
        state=state,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存1"},
        plan_input=normalized,
    ) is False


def test_score_native_move_prefers_staging_detach_that_exposes_committed_carry_vehicle_over_extra_attach():
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.move_generator import generate_real_hook_moves
    from fzed_shunting.solver.state import _apply_move

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "CARRY_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "C70",
                "vehicleNo": "CARRY_SPOT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "203",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存2",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "ATTACH_PEER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    carry_state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存2": ["ATTACH_PEER"]},
            "loco_track_name": "存5北",
            "loco_carry": ("CARRY_SPOT", "CARRY_BLOCKER"),
            "spot_assignments": {},
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    goal_tracks_needed = _collect_goal_tracks(normalized)
    moves = generate_real_hook_moves(normalized, carry_state, master=master)

    attach_move = next(
        move
        for move in moves
        if move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("ATTACH_PEER",)
    )
    staging_detach = next(
        move
        for move in moves
        if move.action_type == "DETACH"
        and move.target_track == "临1"
        and tuple(move.vehicle_nos) == ("CARRY_BLOCKER",)
    )

    current_h = heuristic(carry_state)
    attach_state = _apply_move(
        state=carry_state,
        move=attach_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    staging_state = _apply_move(
        state=carry_state,
        move=staging_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    attach_score, _ = _score_native_move(
        move=attach_move,
        state=carry_state,
        next_state=attach_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(attach_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=goal_tracks_needed,
        satisfied_by_track={},
    )
    staging_score, _ = _score_native_move(
        move=staging_detach,
        state=carry_state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=goal_tracks_needed,
        satisfied_by_track={},
    )

    assert staging_score < attach_score


def test_score_native_move_prefers_lower_future_detach_group_frontier():
    from fzed_shunting.solver.constructive import _score_native_move, _collect_goal_tracks
    from fzed_shunting.solver.move_generator import generate_real_hook_moves
    from fzed_shunting.solver.state import _apply_move

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存1", "trackDistance": 113},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "P2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "预修",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "S1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "调棚",
                "targetMode": "SPOT",
                "targetSpotCode": "1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "Y1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    goal_tracks_needed = _collect_goal_tracks(normalized)

    moves = {
        tuple(move.vehicle_nos): move
        for move in generate_real_hook_moves(normalized, initial, master=master)
        if move.action_type == "ATTACH" and move.source_track == "存5北"
    }
    two_group_attach = moves[("P1", "P2", "S1")]
    three_group_attach = moves[("P1", "P2", "S1", "Y1")]

    two_group_state = _apply_move(
        state=initial,
        move=two_group_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    three_group_state = _apply_move(
        state=initial,
        move=three_group_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    two_group_score, _ = _score_native_move(
        move=two_group_attach,
        state=initial,
        next_state=two_group_state,
        plan_input=normalized,
        current_heuristic=heuristic(initial),
        next_heuristic=heuristic(two_group_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=goal_tracks_needed,
        satisfied_by_track={track: 0 for track, seq in initial.track_sequences.items() if seq},
    )
    three_group_score, _ = _score_native_move(
        move=three_group_attach,
        state=initial,
        next_state=three_group_state,
        plan_input=normalized,
        current_heuristic=heuristic(initial),
        next_heuristic=heuristic(three_group_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=goal_tracks_needed,
        satisfied_by_track={track: 0 for track, seq in initial.track_sequences.items() if seq},
    )

    assert two_group_score < three_group_score


def test_score_native_move_penalizes_regrabbing_unfinished_staging_vehicle():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="存5北", track_distance=367),
            NormalizedTrackInfo(track_name="存4北", track_distance=317.8),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临1",
                order=1,
                vehicle_model="棚车",
                vehicle_no="STAGED",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="临1",
                    allowed_target_tracks=["临1"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="WORK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存4北",
                    allowed_target_tracks=["存4北"],
                ),
            ),
        ],
        loco_track_name="机库",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临1": ["STAGED"], "存5北": ["WORK"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    staging_attach = HookAction(
        source_track="临1",
        target_track="临1",
        vehicle_nos=["STAGED"],
        path_tracks=["临1"],
        action_type="ATTACH",
    )

    work_attach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["WORK"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    staging_state = _apply_move(
        state=state,
        move=staging_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    work_state = _apply_move(
        state=state,
        move=work_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    staging_score, _ = _score_native_move(
        move=staging_attach,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        recent_moves=deque([("抛", "临1", ("STAGED",))]),
    )
    work_score, _ = _score_native_move(
        move=work_attach,
        state=state,
        next_state=work_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(work_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )

    assert work_score < staging_score


def test_score_native_move_prioritizes_stale_staging_debt_recovery_over_new_work():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="存5北", track_distance=367),
            NormalizedTrackInfo(track_name="存4北", track_distance=317.8),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临1",
                order=1,
                vehicle_model="棚车",
                vehicle_no="STAGED_DEBT",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存4北",
                    allowed_target_tracks=["存4北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="NEW_WORK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存4北",
                    allowed_target_tracks=["存4北"],
                ),
            ),
        ],
        loco_track_name="机库",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临1": ["STAGED_DEBT"], "存5北": ["NEW_WORK"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    staging_attach = HookAction(
        source_track="临1",
        target_track="临1",
        vehicle_nos=["STAGED_DEBT"],
        path_tracks=["临1"],
        action_type="ATTACH",
    )
    work_attach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["NEW_WORK"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    staging_state = _apply_move(
        state=state,
        move=staging_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    work_state = _apply_move(
        state=state,
        move=work_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    staging_score, _ = _score_native_move(
        move=staging_attach,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        recent_moves=deque(),
    )
    work_score, _ = _score_native_move(
        move=work_attach,
        state=state,
        next_state=work_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(work_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        recent_moves=deque(),
    )

    assert staging_score < work_score


def test_score_native_move_penalizes_staging_detach_with_blocked_goal_corridor():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临4", track_distance=90.1),
            NormalizedTrackInfo(track_name="存4南", track_distance=154.5),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临4",
                order=1,
                vehicle_model="棚车",
                vehicle_no="CARRY",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5南",
                order=1,
                vehicle_model="棚车",
                vehicle_no="ROUTE_BLOCK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5南",
                    allowed_target_tracks=["存5南"],
                ),
            ),
        ],
        loco_track_name="临4",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"存5南": ["ROUTE_BLOCK"]},
        loco_track_name="临4",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    blocked_staging_move = HookAction(
        source_track="临4",
        target_track="存4南",
        vehicle_nos=["CARRY"],
        path_tracks=["临4", "渡9", "渡8", "存4南"],
        action_type="DETACH",
    )
    clear_staging_move = HookAction(
        source_track="临4",
        target_track="临4",
        vehicle_nos=["CARRY"],
        path_tracks=["临4"],
        action_type="DETACH",
    )
    blocked_state = _apply_move(
        state=state,
        move=blocked_staging_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    clear_state = _apply_move(
        state=state,
        move=clear_staging_move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    blocked_score, _ = _score_native_move(
        move=blocked_staging_move,
        state=state,
        next_state=blocked_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(blocked_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )
    clear_score, _ = _score_native_move(
        move=clear_staging_move,
        state=state,
        next_state=clear_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(clear_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )

    assert clear_score < blocked_score


def test_score_native_move_allows_pushers_to_complete_close_door_cun4bei_sequence():
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    payload = _simple_payload(
        [
            {
                "trackName": "存1",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": attrs,
            }
            for index, (vehicle_no, attrs) in enumerate(
                [("N1", ""), ("N2", ""), ("N3", ""), ("CD", "关门车")],
                start=1,
            )
        ]
    )
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(payload, master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = ReplayState(
        track_sequences={"存1": [], "存4北": ["CD"]},
        loco_track_name="存1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("N1", "N2", "N3"),
    )
    move = HookAction(
        source_track="存1",
        target_track="存4北",
        vehicle_nos=["N1", "N2", "N3"],
        path_tracks=["存1", "存4北"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=move,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    heuristic = make_state_heuristic_real_hook(normalized)

    score, tier = _score_native_move(
        move=move,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        current_heuristic=heuristic(state),
        next_heuristic=heuristic(next_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )

    assert tier == 1
    assert score[3] == 0
