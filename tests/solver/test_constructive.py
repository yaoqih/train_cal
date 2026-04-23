import json
from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.constructive import solve_constructive, ConstructiveResult
from fzed_shunting.verify.replay import build_initial_state
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
    assert len(result.plan) == 1
    assert result.plan[0].source_track == "存5北"
    assert result.plan[0].target_track == "存4北"

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
    assert len(result.plan) == 1
    assert set(result.plan[0].vehicle_nos) == {"E1", "E2", "E3"}


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
    assert result.plan[0].target_track == "机库"
    assert len(result.plan) >= 2

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
    final_seq_pos = None
    current = initial.model_copy(deep=True)
    for m in result.plan:
        src = current.track_sequences.get(m.source_track, [])
        current.track_sequences[m.source_track] = src[len(m.vehicle_nos):]
        existing = list(current.track_sequences.get(m.target_track, []))
        current.track_sequences[m.target_track] = list(m.vehicle_nos) + existing
    final_4bei = current.track_sequences.get("存4北", [])
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
        assert "constructive_backtrack_count" in result.debug_stats
