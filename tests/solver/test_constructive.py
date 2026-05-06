import json
from collections import Counter
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.constructive import (
    ConstructiveResult,
    _is_better_attempt,
    _score_native_move,
    solve_constructive,
)
from fzed_shunting.solver.heuristic import make_state_heuristic_real_hook
from fzed_shunting.solver.state import ReplayState, _apply_move, _state_key
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import build_initial_state, replay_plan
from fzed_shunting.verify.plan_verifier import verify_plan


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
ROOT_DIR = Path(__file__).resolve().parents[2]


class _SyntheticRouteBlockagePlan:
    total_blockage_pressure = 1
    facts_by_blocking_track = {}


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


def test_constructive_attempt_comparison_handles_progress_tuple_heuristics():
    shorter_partial = ConstructiveResult(
        plan=[],
        reached_goal=False,
        iterations=1,
        elapsed_ms=1.0,
        final_heuristic=12,
    )
    better_progress = ConstructiveResult(
        plan=[],
        reached_goal=False,
        iterations=2,
        elapsed_ms=1.0,
        final_heuristic=(10, 0, 0, 8),
    )

    assert _is_better_attempt(better_progress, shorter_partial)


def test_score_native_move_prefers_bounded_random_snapshot_attach_over_whole_prefix():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "存3", "trackDistance": 258.5},
            {"trackName": "机北", "trackDistance": 80.0},
            {"trackName": "调北", "trackDistance": 70.1},
            {"trackName": "洗北", "trackDistance": 100.0},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": f"SNAP_RANDOM_{index}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存3",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index in range(1, 17)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    chunk = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=[f"SNAP_RANDOM_{index}" for index in range(1, 11)],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    whole = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=[f"SNAP_RANDOM_{index}" for index in range(1, 17)],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    chunk_state = _apply_move(
        state=state,
        move=chunk,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    whole_state = _apply_move(
        state=state,
        move=whole,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    chunk_score, chunk_tier = _score_native_move(
        move=chunk,
        state=state,
        next_state=chunk_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(chunk_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存3"},
        satisfied_by_track={},
    )
    whole_score, whole_tier = _score_native_move(
        move=whole,
        state=state,
        next_state=whole_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(whole_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed={"存3"},
        satisfied_by_track={},
    )

    assert chunk_score < whole_score


def test_constructive_route_release_scoring_reuses_route_blockage_by_state(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (ROOT_DIR / "data/validation_inputs/truth/validation_20260206W.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    calls_by_state: Counter[tuple] = Counter()

    def fake_compute_route_blockage_plan(
        plan_input,
        candidate_state,
        route_oracle,
        *,
        blocked_source_tracks=None,
    ):
        calls_by_state[
            (
                _state_key(candidate_state, plan_input),
                tuple(sorted(blocked_source_tracks or ())),
            )
        ] += 1
        return _SyntheticRouteBlockagePlan()

    monkeypatch.setattr(
        "fzed_shunting.solver.constructive.compute_route_blockage_plan",
        fake_compute_route_blockage_plan,
    )

    solve_constructive(
        normalized,
        state,
        master=master,
        max_iterations=1,
        max_backtracks=0,
        route_release_bias=True,
    )

    assert max(calls_by_state.values(), default=0) == 1


def test_constructive_default_scoring_does_not_use_route_blockage_context(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (ROOT_DIR / "data/validation_inputs/truth/validation_20260206W.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    calls_by_state: Counter[tuple] = Counter()

    def fake_compute_route_blockage_plan(
        plan_input,
        candidate_state,
        route_oracle,
        *,
        blocked_source_tracks=None,
    ):
        calls_by_state[
            (
                _state_key(candidate_state, plan_input),
                tuple(sorted(blocked_source_tracks or ())),
            )
        ] += 1
        return _SyntheticRouteBlockagePlan()

    monkeypatch.setattr(
        "fzed_shunting.solver.constructive.compute_route_blockage_plan",
        fake_compute_route_blockage_plan,
    )

    solve_constructive(
        normalized,
        state,
        master=master,
        max_iterations=1,
        max_backtracks=0,
        route_release_bias=False,
    )

    assert calls_by_state == Counter()


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


def test_constructive_stale_guard_allows_forced_restore_after_front_blocker_staging():
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
                "vehicleNo": "RESTORE_FRONT",
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
                "vehicleNo": "NEED_DEPOT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    initial = build_initial_state(normalized)

    result = solve_constructive(
        normalized,
        initial,
        master=master,
        max_iterations=20,
        max_backtracks=0,
        stuck_threshold=1,
    )

    assert result.reached_goal is True
    assert result.stuck_reason is None


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


def test_score_native_move_prefers_snapshot_preferred_track_without_marking_fallback_unfinished():
    from fzed_shunting.solver.constructive import _score_native_move
    from fzed_shunting.solver.purity import compute_state_purity
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_SOFT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    carry_state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"机库": []},
            "loco_track_name": "机库",
            "loco_carry": ("SNAP_SOFT",),
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    preferred_move = HookAction(
        source_track="机库",
        target_track="存5南",
        vehicle_nos=["SNAP_SOFT"],
        path_tracks=["机库", "渡4", "临2", "临1", "渡2", "存5南"],
        action_type="DETACH",
    )
    fallback_move = HookAction(
        source_track="机库",
        target_track="存5北",
        vehicle_nos=["SNAP_SOFT"],
        path_tracks=["机库", "渡4", "临2", "临1", "渡2", "存5南", "存5北"],
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
        goal_tracks_needed={"存5南"},
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
        goal_tracks_needed={"存5南"},
        satisfied_by_track={},
    )

    assert compute_state_purity(normalized, fallback_state).unfinished_count == 0
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


def test_score_native_move_does_not_treat_same_track_detach_as_goal_blocker_clearance():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="预修", track_distance=208.5),
            NormalizedTrackInfo(track_name="存4北", track_distance=317.8),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="预修",
                order=1,
                vehicle_model="棚车",
                vehicle_no="RETURNED_PREFIX",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存4北",
                    allowed_target_tracks=["存4北"],
                ),
            ),
            NormalizedVehicle(
                current_track="预修",
                order=2,
                vehicle_model="棚车",
                vehicle_no="BURIED_TARGET",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="预修",
                    allowed_target_tracks=["预修"],
                ),
            ),
        ],
        loco_track_name="预修",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"预修": ["BURIED_TARGET"]},
        loco_track_name="预修",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("RETURNED_PREFIX",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    same_track_detach = HookAction(
        source_track="预修",
        target_track="预修",
        vehicle_nos=["RETURNED_PREFIX"],
        path_tracks=["预修"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=same_track_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    current_h = make_state_heuristic_real_hook(normalized)(state)

    _score, tier = _score_native_move(
        move=same_track_detach,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=make_state_heuristic_real_hook(normalized)(next_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={"预修": 1},
    )

    assert tier > 3


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


def test_score_native_move_prefers_parking_route_blocker_before_extra_attach():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="存2", track_distance=239.2),
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="ROUTE_BLOCK_A",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=2,
                vehicle_model="棚车",
                vehicle_no="ROUTE_BLOCK_B",
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
                vehicle_no="SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
            NormalizedVehicle(
                current_track="存2",
                order=1,
                vehicle_model="棚车",
                vehicle_no="EXTRA",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "存5北": [],
            "存5南": ["SEEK"],
            "修4库内": [],
            "存2": ["EXTRA"],
            "临1": [],
        },
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("ROUTE_BLOCK_A", "ROUTE_BLOCK_B"),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)
    extra_attach = HookAction(
        source_track="存2",
        target_track="存2",
        vehicle_nos=["EXTRA"],
        path_tracks=["存2"],
        action_type="ATTACH",
    )
    blocker_staging_detach = HookAction(
        source_track="存5北",
        target_track="临1",
        vehicle_nos=["ROUTE_BLOCK_A", "ROUTE_BLOCK_B"],
        path_tracks=["存5北", "渡1", "渡2", "临1"],
        action_type="DETACH",
    )
    extra_state = _apply_move(
        state=state,
        move=extra_attach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    detach_state = _apply_move(
        state=state,
        move=blocker_staging_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    extra_score, _ = _score_native_move(
        move=extra_attach,
        state=state,
        next_state=extra_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(extra_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        defer_extra_attach_for_satisfied_carry=True,
    )
    detach_score, _ = _score_native_move(
        move=blocker_staging_detach,
        state=state,
        next_state=detach_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(detach_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        defer_extra_attach_for_satisfied_carry=True,
    )

    assert detach_score < extra_score


def test_score_native_move_defers_goal_detach_onto_active_route_blocker_track():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="ROUTE_PARK_BLOCK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=2,
                vehicle_model="棚车",
                vehicle_no="ROUTE_PARK_CARRY",
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
                vehicle_no="ROUTE_PARK_SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "存5北": ["ROUTE_PARK_BLOCK"],
            "存5南": ["ROUTE_PARK_SEEK"],
            "修4库内": [],
            "临1": [],
        },
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("ROUTE_PARK_CARRY",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)
    goal_detach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["ROUTE_PARK_CARRY"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    staging_detach = HookAction(
        source_track="存5北",
        target_track="临1",
        vehicle_nos=["ROUTE_PARK_CARRY"],
        path_tracks=["存5北", "渡1", "渡2", "临1"],
        action_type="DETACH",
    )
    goal_state = _apply_move(
        state=state,
        move=goal_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    staging_state = _apply_move(
        state=state,
        move=staging_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    goal_score, _ = _score_native_move(
        move=goal_detach,
        state=state,
        next_state=goal_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(goal_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
    )
    staging_score, _ = _score_native_move(
        move=staging_detach,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
    )

    assert route_blockage_plan.facts_by_blocking_track["存5北"].blocked_vehicle_nos == ["ROUTE_PARK_SEEK"]
    assert staging_score < goal_score


def test_score_native_move_demotes_goal_detach_that_restores_route_blockage():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="临4", track_distance=42.9),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="ROUTE_RESTORE_BLOCKER",
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
                vehicle_no="ROUTE_RESTORE_SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "存5北": [],
            "存5南": ["ROUTE_RESTORE_SEEK"],
            "修4库内": [],
            "临4": [],
        },
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("ROUTE_RESTORE_BLOCKER",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)
    restore_blocker = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["ROUTE_RESTORE_BLOCKER"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    safe_staging = HookAction(
        source_track="存5北",
        target_track="临4",
        vehicle_nos=["ROUTE_RESTORE_BLOCKER"],
        path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡5", "机北", "机棚", "临4"],
        action_type="DETACH",
    )
    restore_state = _apply_move(
        state=state,
        move=restore_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    staging_state = _apply_move(
        state=state,
        move=safe_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    restore_score, restore_tier = _score_native_move(
        move=restore_blocker,
        state=state,
        next_state=restore_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(restore_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        route_oracle=route_oracle,
    )
    staging_score, staging_tier = _score_native_move(
        move=safe_staging,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        route_oracle=route_oracle,
    )

    assert compute_route_blockage_plan(normalized, restore_state, route_oracle).total_blockage_pressure > 0
    assert restore_tier >= 5
    assert staging_score < restore_score


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
            NormalizedTrackInfo(track_name="临4", track_distance=90.1),
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


def test_score_native_move_applies_route_pressure_before_staging_heuristic_noise():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="存4南", track_distance=154.5),
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临1",
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
            )
        ],
        loco_track_name="机库",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临1": [], "临2": [], "存4南": [], "存5北": []},
        loco_track_name="临1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    blocked_staging = HookAction(
        source_track="临1",
        target_track="存4南",
        vehicle_nos=["CARRY"],
        path_tracks=["临1", "存4南"],
        action_type="DETACH",
    )
    clear_staging = HookAction(
        source_track="临1",
        target_track="临2",
        vehicle_nos=["CARRY"],
        path_tracks=["临1", "临2"],
        action_type="DETACH",
    )
    blocked_state = _apply_move(
        state=state,
        move=blocked_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    clear_state = _apply_move(
        state=state,
        move=clear_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    blocked_score, blocked_tier = _score_native_move(
        move=blocked_staging,
        state=state,
        next_state=blocked_state,
        plan_input=normalized,
        current_heuristic=20,
        next_heuristic=10,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=1,
    )
    clear_score, clear_tier = _score_native_move(
        move=clear_staging,
        state=state,
        next_state=clear_state,
        plan_input=normalized,
        current_heuristic=20,
        next_heuristic=11,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=0,
    )

    assert clear_tier == blocked_tier
    assert clear_score < blocked_score


def test_score_native_move_keeps_goal_detach_in_goal_tier_when_it_parks_on_corridor():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="临4", track_distance=90.1),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临4",
                order=1,
                vehicle_model="棚车",
                vehicle_no="NORTH_GOAL",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5南",
                    allowed_target_tracks=["存5南"],
                ),
            ),
            NormalizedVehicle(
                current_track="临4",
                order=2,
                vehicle_model="棚车",
                vehicle_no="DEEP_GOAL",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
        ],
        loco_track_name="临4",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临4": ["DEEP_GOAL"], "临2": [], "存5南": [], "存5北": []},
        loco_track_name="临4",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("NORTH_GOAL",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    corridor_goal_detach = HookAction(
        source_track="临4",
        target_track="存5南",
        vehicle_nos=["NORTH_GOAL"],
        path_tracks=["临4", "存5南"],
        action_type="DETACH",
    )
    defer_to_staging = HookAction(
        source_track="临4",
        target_track="临2",
        vehicle_nos=["NORTH_GOAL"],
        path_tracks=["临4", "临2"],
        action_type="DETACH",
    )
    corridor_state = _apply_move(
        state=state,
        move=corridor_goal_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    staging_state = _apply_move(
        state=state,
        move=defer_to_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    corridor_score, corridor_tier = _score_native_move(
        move=corridor_goal_detach,
        state=state,
        next_state=corridor_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(corridor_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=1,
    )
    staging_score, staging_tier = _score_native_move(
        move=defer_to_staging,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=0,
    )

    assert corridor_tier < staging_tier
    assert corridor_score < staging_score


def test_score_native_move_avoids_reparking_current_route_blocker_during_release():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="BLOCK",
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
                vehicle_no="SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"存5北": [], "存5南": ["SEEK"], "修4库内": [], "临1": []},
        loco_track_name="存5北",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("BLOCK",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    repark_blocker = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["BLOCK"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    keep_released = HookAction(
        source_track="存5北",
        target_track="临1",
        vehicle_nos=["BLOCK"],
        path_tracks=["存5北", "临1"],
        action_type="DETACH",
    )
    repark_state = _apply_move(
        state=state,
        move=repark_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    released_state = _apply_move(
        state=state,
        move=keep_released,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    repark_score, repark_tier = _score_native_move(
        move=repark_blocker,
        state=state,
        next_state=repark_state,
        plan_input=normalized,
        current_heuristic=3,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=1,
    )
    released_score, released_tier = _score_native_move(
        move=keep_released,
        state=state,
        next_state=released_state,
        plan_input=normalized,
        current_heuristic=3,
        next_heuristic=4,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=0,
    )

    assert released_tier <= repark_tier
    assert released_score < repark_score


def test_score_native_move_keeps_goal_detach_when_staging_creates_more_route_pressure():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _route_blockage_parking_pressure, _score_native_move
    from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="机棚", track_distance=105.8),
            NormalizedTrackInfo(track_name="存2", track_distance=239.2),
            NormalizedTrackInfo(track_name="存4南", track_distance=154.5),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="RETURN_GOAL",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="机棚",
                order=1,
                vehicle_model="棚车",
                vehicle_no="SOURCE_SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存2",
                    allowed_target_tracks=["存2"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5南",
                order=1,
                vehicle_model="棚车",
                vehicle_no="SPOT_SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="SPOT",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                    target_spot_code="407",
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "存5北": ["BLOCK_ON_CORRIDOR"],
            "机棚": ["SOURCE_SEEK"],
            "存5南": ["SPOT_SEEK"],
            "修4库内": [],
            "存2": [],
            "临4": [],
        },
        loco_track_name="存5北",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("RETURN_GOAL",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)
    goal_detach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["RETURN_GOAL"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    staging_detach = HookAction(
        source_track="存5北",
        target_track="临4",
        vehicle_nos=["RETURN_GOAL"],
        path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡5", "渡6", "渡7", "预修", "渡9", "临4"],
        action_type="DETACH",
    )
    goal_state = _apply_move(
        state=state,
        move=goal_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    staging_state = _apply_move(
        state=state,
        move=staging_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    goal_score, goal_tier = _score_native_move(
        move=goal_detach,
        state=state,
        next_state=goal_state,
        plan_input=normalized,
        current_heuristic=5,
        next_heuristic=4,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        route_oracle=route_oracle,
        route_blockage_parking_pressure=_route_blockage_parking_pressure(
            move=goal_detach,
            state=state,
            next_state=goal_state,
            plan_input=normalized,
            route_oracle=route_oracle,
        ),
    )
    staging_score, staging_tier = _score_native_move(
        move=staging_detach,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=5,
        next_heuristic=6,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
        route_oracle=route_oracle,
        route_blockage_parking_pressure=_route_blockage_parking_pressure(
            move=staging_detach,
            state=state,
            next_state=staging_state,
            plan_input=normalized,
            route_oracle=route_oracle,
        ),
    )

    assert compute_route_blockage_plan(normalized, goal_state, route_oracle).total_blockage_pressure == 1
    assert compute_route_blockage_plan(normalized, staging_state, route_oracle).total_blockage_pressure > 1
    assert goal_tier == staging_tier
    assert goal_score < staging_score


def test_score_native_move_uses_route_pressure_after_progress_signals():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="临4", track_distance=90.1),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临4",
                order=1,
                vehicle_model="棚车",
                vehicle_no="A",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5南",
                    allowed_target_tracks=["存5南"],
                ),
            ),
            NormalizedVehicle(
                current_track="临4",
                order=2,
                vehicle_model="棚车",
                vehicle_no="B",
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
        track_sequences={"临4": [], "临2": [], "存5南": [], "存5北": []},
        loco_track_name="临4",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("A", "B"),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    larger_progress = HookAction(
        source_track="临4",
        target_track="存5南",
        vehicle_nos=["A", "B"],
        path_tracks=["临4", "存5南"],
        action_type="DETACH",
    )
    smaller_progress = HookAction(
        source_track="临4",
        target_track="存5南",
        vehicle_nos=["B"],
        path_tracks=["临4", "存5南"],
        action_type="DETACH",
    )
    larger_state = _apply_move(
        state=state,
        move=larger_progress,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    smaller_state = _apply_move(
        state=state,
        move=smaller_progress,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    larger_score, larger_tier = _score_native_move(
        move=larger_progress,
        state=state,
        next_state=larger_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(larger_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=1,
    )
    smaller_score, smaller_tier = _score_native_move(
        move=smaller_progress,
        state=state,
        next_state=smaller_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(smaller_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_parking_pressure=0,
    )

    assert larger_tier == smaller_tier
    assert larger_score < smaller_score


def test_route_blockage_parking_pressure_detects_newly_occupied_intermediate_track():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _route_blockage_parking_pressure
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="临4", track_distance=90.1),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临4",
                order=1,
                vehicle_model="棚车",
                vehicle_no="NORTH_GOAL",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5南",
                    allowed_target_tracks=["存5南"],
                ),
            ),
            NormalizedVehicle(
                current_track="临4",
                order=2,
                vehicle_model="棚车",
                vehicle_no="DEEP_GOAL",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
        ],
        loco_track_name="临4",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临4": ["DEEP_GOAL"], "临2": [], "存5南": [], "存5北": []},
        loco_track_name="临4",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("NORTH_GOAL",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    corridor_goal_detach = HookAction(
        source_track="临4",
        target_track="存5南",
        vehicle_nos=["NORTH_GOAL"],
        path_tracks=["临4", "存5南"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=corridor_goal_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    assert _route_blockage_parking_pressure(
        move=corridor_goal_detach,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    ) == 1


def test_route_blockage_parking_pressure_detects_newly_blocked_source_access():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _route_blockage_parking_pressure
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="BLOCK",
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
                vehicle_no="SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"存5北": [], "存5南": ["SEEK"], "修4库内": [], "机库": []},
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("BLOCK",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    repark_blocker = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["BLOCK"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=repark_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    assert _route_blockage_parking_pressure(
        move=repark_blocker,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    ) == 1


def test_route_blockage_parking_pressure_penalizes_reparking_onto_still_blocked_track():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _route_blockage_parking_pressure
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="REMAINING_BLOCK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=2,
                vehicle_model="棚车",
                vehicle_no="CARRIED_BLOCK",
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
                vehicle_no="SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"存5北": ["REMAINING_BLOCK"], "存5南": ["SEEK"], "修4库内": []},
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRIED_BLOCK",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    repark_blocker = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["CARRIED_BLOCK"],
        path_tracks=["存5北"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=repark_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    assert _route_blockage_parking_pressure(
        move=repark_blocker,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    ) == 1


def test_route_blockage_parking_pressure_penalizes_adding_to_active_blocking_staging_track():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _route_blockage_parking_pressure
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="机库", track_distance=71.6),
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="存3", track_distance=258.5),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="临1",
                order=1,
                vehicle_model="棚车",
                vehicle_no="EXISTING_BLOCKER",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="临1",
                    allowed_target_tracks=["临1"],
                ),
            ),
            NormalizedVehicle(
                current_track="临2",
                order=1,
                vehicle_model="棚车",
                vehicle_no="NEEDS_STORE3",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存3",
                    allowed_target_tracks=["存3"],
                ),
            ),
            NormalizedVehicle(
                current_track="机库",
                order=1,
                vehicle_model="棚车",
                vehicle_no="NEW_STAGING",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存3",
                    allowed_target_tracks=["存3"],
                ),
            ),
        ],
        loco_track_name="临1",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "临1": ["EXISTING_BLOCKER"],
            "临2": ["NEEDS_STORE3"],
            "存3": [],
            "机库": [],
        },
        loco_track_name="机库",
        loco_node="L7",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("NEW_STAGING",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    park_on_active_blocker = HookAction(
        source_track="机库",
        target_track="临1",
        vehicle_nos=["NEW_STAGING"],
        path_tracks=["机库", "渡4", "临2", "临1"],
        action_type="DETACH",
    )
    next_state = _apply_move(
        state=state,
        move=park_on_active_blocker,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    assert _route_blockage_parking_pressure(
        move=park_on_active_blocker,
        state=state,
        next_state=next_state,
        plan_input=normalized,
        route_oracle=RouteOracle(master),
    ) == 1


def test_score_native_move_does_not_force_primary_staging_without_new_route_blockage():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="存5北", track_distance=367.0),
            NormalizedTrackInfo(track_name="存5南", track_distance=156.0),
            NormalizedTrackInfo(track_name="修4库内", track_distance=151.7),
            NormalizedTrackInfo(track_name="存2", track_distance=239.2),
            NormalizedTrackInfo(track_name="存4南", track_distance=154.5),
        ],
        vehicles=[
            NormalizedVehicle(
                current_track="存5北",
                order=1,
                vehicle_model="棚车",
                vehicle_no="REMAINING_BLOCK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="存5北",
                    allowed_target_tracks=["存5北"],
                ),
            ),
            NormalizedVehicle(
                current_track="存5北",
                order=2,
                vehicle_model="棚车",
                vehicle_no="CARRIED_BLOCK",
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
                vehicle_no="SEEK",
                repair_process="段修",
                vehicle_length=14.3,
                goal=GoalSpec(
                    target_mode="TRACK",
                    target_track="修4库内",
                    allowed_target_tracks=["修4库内"],
                ),
            ),
        ],
        loco_track_name="存5北",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={
            "存5北": ["REMAINING_BLOCK"],
            "存5南": ["SEEK"],
            "修4库内": [],
            "存2": [],
            "存4南": [],
        },
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRIED_BLOCK",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    to_storage = HookAction(
        source_track="存5北",
        target_track="存2",
        vehicle_nos=["CARRIED_BLOCK"],
        path_tracks=["存5北", "存2"],
        action_type="DETACH",
    )
    to_primary_staging = HookAction(
        source_track="存5北",
        target_track="存4南",
        vehicle_nos=["CARRIED_BLOCK"],
        path_tracks=["存5北", "存4南"],
        action_type="DETACH",
    )
    storage_state = _apply_move(
        state=state,
        move=to_storage,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    primary_state = _apply_move(
        state=state,
        move=to_primary_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    route_blockage_plan = compute_route_blockage_plan(normalized, state, RouteOracle(master))

    storage_score, _ = _score_native_move(
        move=to_storage,
        state=state,
        next_state=storage_state,
        plan_input=normalized,
        current_heuristic=3,
        next_heuristic=2,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
    )
    primary_score, _ = _score_native_move(
        move=to_primary_staging,
        state=state,
        next_state=primary_state,
        plan_input=normalized,
        current_heuristic=3,
        next_heuristic=4,
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
        route_blockage_plan=route_blockage_plan,
    )

    assert storage_score < primary_score


def test_candidate_selection_pool_avoids_recent_state_repeats_before_inverse_fallback():
    from fzed_shunting.solver.constructive import _candidate_selection_pool
    from fzed_shunting.solver.types import HookAction

    repeat_non_inverse = HookAction(
        source_track="临1",
        target_track="临2",
        vehicle_nos=["A"],
        path_tracks=["临1", "临2"],
        action_type="DETACH",
    )
    inverse_non_repeat = HookAction(
        source_track="临2",
        target_track="临1",
        vehicle_nos=["B"],
        path_tracks=["临2", "临1"],
        action_type="DETACH",
    )
    clean = HookAction(
        source_track="临1",
        target_track="临3",
        vehicle_nos=["C"],
        path_tracks=["临1", "临3"],
        action_type="DETACH",
    )

    scored = [
        ((0,), 5, repeat_non_inverse, False, True),
        ((1,), 5, inverse_non_repeat, True, False),
        ((2,), 5, clean, False, False),
    ]
    assert _candidate_selection_pool(scored) == [scored[2]]
    assert _candidate_selection_pool(scored[:2]) == [scored[1]]


def test_candidate_selection_pool_keeps_route_pressure_release_before_clean_regression():
    from fzed_shunting.solver.constructive import _candidate_selection_pool
    from fzed_shunting.solver.types import HookAction

    pressure_release = HookAction(
        source_track="临3",
        target_track="临3",
        vehicle_nos=["A", "B"],
        path_tracks=["临3"],
        action_type="ATTACH",
    )
    clean_regression = HookAction(
        source_track="临3",
        target_track="临2",
        vehicle_nos=["C"],
        path_tracks=["临3", "临2"],
        action_type="DETACH",
    )

    release_entry = (
        (2, 0, 0, 0, 0, 1, 0, -21, -21, 2, 0, 20),
        2,
        pressure_release,
        True,
        False,
    )
    clean_entry = (
        (5, 0, 0, 0, 0, 1, 0, 0, 44, 0, 20, 24),
        5,
        clean_regression,
        False,
        False,
    )

    assert _candidate_selection_pool([release_entry, clean_entry]) == [release_entry]


def test_score_native_move_penalizes_staging_to_staging_detach_without_progress():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="存2", track_distance=239.2),
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
                    target_track="存2",
                    allowed_target_tracks=["存2"],
                ),
            )
        ],
        loco_track_name="临1",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临1": [], "临2": [], "存2": []},
        loco_track_name="临1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("STAGED",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    lateral_staging = HookAction(
        source_track="临1",
        target_track="临2",
        vehicle_nos=["STAGED"],
        path_tracks=["临1", "临2"],
        action_type="DETACH",
    )
    goal_detach = HookAction(
        source_track="临1",
        target_track="存2",
        vehicle_nos=["STAGED"],
        path_tracks=["临1", "渡3", "存2"],
        action_type="DETACH",
    )
    lateral_state = _apply_move(
        state=state,
        move=lateral_staging,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    goal_state = _apply_move(
        state=state,
        move=goal_detach,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    lateral_score, lateral_tier = _score_native_move(
        move=lateral_staging,
        state=state,
        next_state=lateral_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(lateral_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )
    goal_score, goal_tier = _score_native_move(
        move=goal_detach,
        state=state,
        next_state=goal_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(goal_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )

    assert goal_tier < lateral_tier
    assert goal_score < lateral_score


def test_score_native_move_prefers_non_staging_lateral_detach_over_staging_chain():
    from fzed_shunting.io.normalize_input import GoalSpec, NormalizedPlanInput, NormalizedTrackInfo, NormalizedVehicle
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    normalized = NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name="临1", track_distance=81.4),
            NormalizedTrackInfo(track_name="临2", track_distance=55.7),
            NormalizedTrackInfo(track_name="调北", track_distance=105.8),
            NormalizedTrackInfo(track_name="存2", track_distance=239.2),
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
                    target_track="存2",
                    allowed_target_tracks=["存2"],
                ),
            )
        ],
        loco_track_name="临1",
        yard_mode="NORMAL",
    )
    state = ReplayState(
        track_sequences={"临1": [], "临2": [], "调北": [], "存2": []},
        loco_track_name="临1",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("STAGED",),
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    heuristic = make_state_heuristic_real_hook(normalized)
    current_h = heuristic(state)
    staging_chain = HookAction(
        source_track="临1",
        target_track="临2",
        vehicle_nos=["STAGED"],
        path_tracks=["临1", "临2"],
        action_type="DETACH",
    )
    non_staging_lateral = HookAction(
        source_track="临1",
        target_track="调北",
        vehicle_nos=["STAGED"],
        path_tracks=["临1", "临2", "渡4", "调北"],
        action_type="DETACH",
    )
    staging_state = _apply_move(
        state=state,
        move=staging_chain,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )
    lateral_state = _apply_move(
        state=state,
        move=non_staging_lateral,
        plan_input=normalized,
        vehicle_by_no=vehicle_by_no,
    )

    staging_score, staging_tier = _score_native_move(
        move=staging_chain,
        state=state,
        next_state=staging_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(staging_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )
    lateral_score, lateral_tier = _score_native_move(
        move=non_staging_lateral,
        state=state,
        next_state=lateral_state,
        plan_input=normalized,
        current_heuristic=current_h,
        next_heuristic=heuristic(lateral_state),
        vehicle_by_no=vehicle_by_no,
        goal_tracks_needed=_collect_goal_tracks(normalized),
        satisfied_by_track={},
    )

    assert lateral_tier <= staging_tier
    assert lateral_score < staging_score


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


def test_score_native_move_prioritizes_work_position_progress_over_area_random():
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 80},
                {"trackName": "存5北", "trackDistance": 120},
                {"trackName": "调棚", "trackDistance": 120},
                {"trackName": "修1库内", "trackDistance": 120},
            ],
            "vehicleInfo": [
                {
                    "trackName": "存5北",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": "AREA_DONE",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "101",
                    "vehicleAttributes": "",
                },
                {
                    "trackName": "存5北",
                    "order": "2",
                    "vehicleModel": "棚车",
                    "vehicleNo": "WP_DONE",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "调棚",
                    "isSpotting": "是",
                    "vehicleAttributes": "",
                },
            ],
            "locoTrackName": "机库",
        },
        master,
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_tracks_needed = _collect_goal_tracks(normalized)
    area_move = HookAction(
        source_track="存5北",
        target_track="修1库内",
        vehicle_nos=["AREA_DONE"],
        path_tracks=["存5北", "修1库内"],
        action_type="DETACH",
    )
    work_position_move = HookAction(
        source_track="存5北",
        target_track="调棚",
        vehicle_nos=["WP_DONE"],
        path_tracks=["存5北", "调棚"],
        action_type="DETACH",
    )

    def score(move: HookAction):
        state = ReplayState(
            track_sequences={
                "存5北": [],
                "调棚": [],
                "修1库内": [],
                "机库": [],
            },
            loco_track_name="存5北",
            weighed_vehicle_nos=set(),
            spot_assignments={},
            loco_carry=tuple(move.vehicle_nos),
        )
        next_state = _apply_move(
            state=state,
            move=move,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
        return _score_native_move(
            move=move,
            state=state,
            next_state=next_state,
            plan_input=normalized,
            current_heuristic=2,
            next_heuristic=1,
            current_progress=(2, 0, 0, 2),
            next_progress=(1, 0, 0, 1),
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
        )[0]

    assert score(work_position_move) < score(area_move)


def test_score_native_move_prioritizes_frontier_debt_before_goal_track_blockers(monkeypatch):
    from fzed_shunting.solver.constructive import _collect_goal_tracks, _score_native_move
    from fzed_shunting.solver.state import _apply_move
    from fzed_shunting.solver.types import HookAction

    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 80},
                {"trackName": "存5北", "trackDistance": 120},
                {"trackName": "调棚", "trackDistance": 120},
                {"trackName": "存2", "trackDistance": 120},
            ],
            "vehicleInfo": [
                {
                    "trackName": "存5北",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": "WORK",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "调棚",
                    "isSpotting": "是",
                    "vehicleAttributes": "",
                },
                {
                    "trackName": "存5北",
                    "order": "2",
                    "vehicleModel": "棚车",
                    "vehicleNo": "TRACK",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存2",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                },
            ],
            "locoTrackName": "机库",
        },
        master,
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_tracks_needed = _collect_goal_tracks(normalized)
    work_move = HookAction(
        source_track="存5北",
        target_track="调棚",
        vehicle_nos=["WORK"],
        path_tracks=["存5北", "调棚"],
        action_type="DETACH",
    )
    track_move = HookAction(
        source_track="存5北",
        target_track="存2",
        vehicle_nos=["TRACK"],
        path_tracks=["存5北", "存2"],
        action_type="DETACH",
    )

    def fake_structural_metrics(_plan_input, next_state):
        if next_state.track_sequences.get("调棚"):
            return SimpleNamespace(
                goal_track_blocker_count=8,
                front_blocker_count=0,
                staging_debt_count=0,
                area_random_unfinished_count=0,
                work_position_unfinished_count=0,
            )
        return SimpleNamespace(
            goal_track_blocker_count=0,
            front_blocker_count=1,
            staging_debt_count=0,
            area_random_unfinished_count=0,
            work_position_unfinished_count=1,
        )

    monkeypatch.setattr(
        "fzed_shunting.solver.constructive.compute_structural_metrics",
        fake_structural_metrics,
    )

    def score(move: HookAction):
        state = ReplayState(
            track_sequences={
                "存5北": [],
                "调棚": [],
                "存2": [],
                "机库": [],
            },
            loco_track_name="存5北",
            weighed_vehicle_nos=set(),
            spot_assignments={},
            loco_carry=tuple(move.vehicle_nos),
        )
        next_state = _apply_move(
            state=state,
            move=move,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
        return _score_native_move(
            move=move,
            state=state,
            next_state=next_state,
            plan_input=normalized,
            current_heuristic=2,
            next_heuristic=1,
            current_progress=(2, 0, 0, 2),
            next_progress=(1, 0, 0, 1),
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
        )[0]

    assert score(work_move) < score(track_move)
