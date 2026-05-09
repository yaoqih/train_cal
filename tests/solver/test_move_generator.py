import json
from pathlib import Path
from collections import Counter
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver import move_generator
from fzed_shunting.domain.work_positions import allowed_spotting_south_ranks, north_rank, south_rank
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import (
    _collect_real_hook_access_blocker_attach_requests,
    _collect_real_hook_identity_attach_requests,
    _candidate_staging_targets,
    generate_goal_moves,
    generate_real_hook_moves,
)
from fzed_shunting.solver.move_candidates import generate_move_candidates
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.state import _state_key
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
ROOT_DIR = Path(__file__).resolve().parents[2]


def _vehicle(
    vehicle_no: str,
    track: str,
    target: str,
    *,
    order: int = 1,
    spotting: str = "",
) -> dict:
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": 14.3,
        "targetTrack": target,
        "isSpotting": spotting,
        "vehicleAttributes": "",
    }


def test_move_candidates_wrap_generated_real_hook_moves_without_structural_planning():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("A1", "存5北", "机库", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    primitive_moves = generate_real_hook_moves(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert candidates
    assert [candidate.steps[0] for candidate in candidates] == primitive_moves
    assert {candidate.kind for candidate in candidates} == {"primitive"}
    assert all(len(candidate.steps) == 1 for candidate in candidates)


def test_structural_candidate_limit_preserves_focus_track_diversity():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _select_structural_candidates,
    )

    candidates = [
        MoveCandidate(
            steps=(
                HookAction(
                    source_track=f"S{index}",
                    target_track=f"T{index}",
                    vehicle_nos=[f"V{index}"],
                    path_tracks=[f"S{index}", f"T{index}"],
                    action_type="DETACH",
                ),
            ),
            kind="structural",
            reason="resource_release",
            focus_tracks=(f"T{index}",),
            structural_reserve=True,
        )
        for index in range(5)
    ]

    selected = _select_structural_candidates(candidates, limit=4)

    assert len(selected) == 4
    assert {candidate.focus_tracks for candidate in selected} == {
        ("T0",),
        ("T1",),
        ("T2",),
        ("T3",),
    }


def test_structural_candidate_limit_does_not_let_one_track_crowd_out_others():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _select_structural_candidates,
    )

    def candidate(reason: str, focus_track: str, vehicle_no: str) -> MoveCandidate:
        return MoveCandidate(
            steps=(
                HookAction(
                    source_track=f"S{vehicle_no}",
                    target_track=focus_track,
                    vehicle_nos=[vehicle_no],
                    path_tracks=[f"S{vehicle_no}", focus_track],
                    action_type="DETACH",
                ),
            ),
            kind="structural",
            reason=reason,
            focus_tracks=(focus_track,),
            structural_reserve=True,
        )

    candidates = [
        candidate("work_position_source_opening", "调棚", "A"),
        candidate("work_position_window_repair", "调棚", "B"),
        candidate("work_position_free_fill", "调棚", "C"),
        candidate("resource_release", "预修", "D"),
        candidate("resource_release", "存1", "E"),
    ]

    selected = _select_structural_candidates(candidates, limit=3)

    assert {item.focus_tracks for item in selected} == {
        ("调棚",),
        ("预修",),
        ("存1",),
    }


def test_release_structural_candidate_ranking_prioritizes_route_and_resource_release():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _rank_release_structural_candidates,
    )

    def candidate(reason: str, focus_track: str, steps: int) -> MoveCandidate:
        return MoveCandidate(
            steps=tuple(
                HookAction(
                    source_track=f"S{index}",
                    target_track=focus_track,
                    vehicle_nos=[f"V{index}"],
                    path_tracks=[f"S{index}", focus_track],
                    action_type="DETACH",
                )
                for index in range(steps)
            ),
            kind="structural",
            reason=reason,
            focus_tracks=(focus_track,),
            structural_reserve=True,
        )

    candidates = [
        candidate("route_release_frontier", "存1", 4),
        candidate("resource_release", "预修", 2),
        candidate("resource_release", "存4南", 3),
        candidate("work_position_source_opening", "存5北", 1),
    ]

    ranked = _rank_release_structural_candidates(candidates)

    assert [item.reason for item in ranked] == [
        "route_release_frontier",
        "resource_release",
        "resource_release",
        "work_position_source_opening",
    ]
    assert [item.focus_tracks for item in ranked[:3]] == [
        ("存1",),
        ("预修",),
        ("存4南",),
    ]


def test_structural_generation_keeps_resource_debt_when_track_has_order_debt(monkeypatch):
    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import MoveCandidate
    from fzed_shunting.solver.structural_intent import (
        OrderDebt,
        ResourceDebt,
        StructuralIntent,
    )

    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "调棚", "trackDistance": 174.3},
            ],
            "vehicleInfo": [],
            "locoTrackName": "机库",
        },
        master,
        allow_internal_loco_tracks=True,
    )
    state = build_initial_state(normalized)

    def candidate(reason: str, focus_track: str) -> MoveCandidate:
        return MoveCandidate(
            steps=(
                HookAction(
                    source_track="机库",
                    target_track=focus_track,
                    vehicle_nos=["V"],
                    path_tracks=["机库", focus_track],
                    action_type="DETACH",
                ),
            ),
            kind="structural",
            reason=reason,
            focus_tracks=(focus_track,),
            structural_reserve=True,
        )

    monkeypatch.setattr(
        move_candidates,
        "_build_work_position_source_opening_candidate",
        lambda **kwargs: candidate("work_position_source_opening", kwargs["target_track"]),
    )
    monkeypatch.setattr(
        move_candidates,
        "_build_work_position_window_candidate",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        move_candidates,
        "_build_work_position_free_fill_candidate",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        move_candidates,
        "_build_resource_release_candidate",
        lambda **kwargs: candidate("resource_release", kwargs["debt"].track_name),
    )

    intent = StructuralIntent(
        committed_blocks_by_track={},
        order_debts_by_track={
            "调棚": OrderDebt(
                track_name="调棚",
                defect_count=1,
                pending_vehicle_nos=("SPOT",),
                blocking_prefix_vehicle_nos=(),
                kind_counts=(("SPOTTING", 1),),
            )
        },
        resource_debts=(
            ResourceDebt(
                kind="CAPACITY_RELEASE",
                track_name="调棚",
                vehicle_nos=("BLOCK",),
                pressure=10.0,
            ),
        ),
        staging_buffers=(),
        delayed_commitments=(),
    )

    candidates = move_candidates._generate_structural_candidates(
        plan_input=normalized,
        state=state,
        master=master,
        route_oracle=RouteOracle(master),
        intent=intent,
    )

    assert [candidate.reason for candidate in candidates] == [
        "work_position_source_opening",
        "resource_release",
    ]


def test_work_position_source_opening_does_not_commit_free_before_strict_window():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            _vehicle("FREE_A", "存5北", "调棚", order=1),
            _vehicle("SPOT_A", "存5南", "调棚", order=1, spotting="是"),
            _vehicle("PAD1", "调棚", "调棚", order=1),
            _vehicle("PAD2", "调棚", "调棚", order=2),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["SPOT_A"]
        for candidate in candidates
        for step in candidate.steps
    )
    assert not any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and "FREE_A" in step.vehicle_nos
        for candidate in candidates
        for step in candidate.steps
    )


def test_work_position_source_opening_keeps_free_group_buffered_until_strict_window():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "预修", "trackDistance": 68.7},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            _vehicle("TO_PRE_A", "存5北", "预修", order=1),
            _vehicle("TO_PRE_B", "存5北", "预修", order=2),
            _vehicle("FREE_A", "存5北", "调棚", order=3),
            _vehicle("FREE_B", "存5北", "调棚", order=4),
            _vehicle("SPOT_A", "存5南", "调棚", order=1, spotting="是"),
            _vehicle("BLOCK1", "调棚", "存4北", order=1),
            _vehicle("BLOCK2", "调棚", "存4北", order=2),
            _vehicle("BLOCK3", "调棚", "存4北", order=3),
            _vehicle("BLOCK4", "调棚", "存4北", order=4),
            _vehicle("BLOCK5", "调棚", "存4北", order=5),
            _vehicle("BLOCK6", "调棚", "存4北", order=6),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.reason == "work_position_source_opening"
        and any(
            step.action_type == "DETACH"
            and step.target_track == "调棚"
            and {"FREE_A", "FREE_B"}.intersection(step.vehicle_nos)
            for step in candidate.steps
        )
        for candidate in candidates
    )


def test_trim_candidate_holds_work_position_free_buffer_before_strict_window():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _trim_candidate_before_delayed_commitment,
    )
    from fzed_shunting.solver.structural_intent import (
        DelayedCommitment,
        StructuralIntent,
    )

    prefix_attach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["TO_PRE"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    prefix_detach = HookAction(
        source_track="存5北",
        target_track="预修",
        vehicle_nos=["TO_PRE"],
        path_tracks=["存5北", "预修"],
        action_type="DETACH",
    )
    free_attach = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["FREE_BEFORE_SPOT"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    free_detach = HookAction(
        source_track="存5北",
        target_track="调棚",
        vehicle_nos=["FREE_BEFORE_SPOT"],
        path_tracks=["存5北", "调棚"],
        action_type="DETACH",
    )
    candidate = MoveCandidate(
        steps=(prefix_attach, prefix_detach, free_attach, free_detach),
        kind="structural",
        reason="work_position_source_opening",
        focus_tracks=("调棚",),
        structural_reserve=True,
    )
    intent = StructuralIntent(
        committed_blocks_by_track={},
        order_debts_by_track={},
        resource_debts=(),
        staging_buffers=(),
        delayed_commitments=(
            DelayedCommitment(
                vehicle_no="FREE_BEFORE_SPOT",
                target_track="调棚",
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    trimmed = _trim_candidate_before_delayed_commitment(
        candidate,
        vehicle_by_no={},
        intent=intent,
    )

    assert trimmed is not None
    assert trimmed.steps == (prefix_attach, prefix_detach)


def test_move_candidates_protect_committed_goal_block_from_ordinary_churn(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("DONE", "存5北", "存5北", order=1),
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    debug_stats = {}
    churn_move = HookAction(
        source_track="存5北",
        target_track="存5北",
        vehicle_nos=["DONE"],
        path_tracks=["存5北"],
        action_type="ATTACH",
    )
    monkeypatch.setattr(
        "fzed_shunting.solver.move_candidates.generate_real_hook_moves",
        lambda *_args, **_kwargs: [churn_move],
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
        debug_stats=debug_stats,
    )

    assert all(
        not (
            candidate.steps[0].action_type == "ATTACH"
            and candidate.steps[0].source_track == "存5北"
            and candidate.steps[0].vehicle_nos == ["DONE"]
        )
        for candidate in candidates
    )
    assert debug_stats["protected_primitive_rejected_count"] >= 1


def test_move_candidates_allow_committed_block_when_resource_debt_requires_it(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("SEEK", "临4", "存5北", order=1),
            _vehicle("BLOCK", "存5南", "存5南", order=1),
        ],
        "locoTrackName": "存5南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    release_move = HookAction(
        source_track="存5南",
        target_track="存5南",
        vehicle_nos=["BLOCK"],
        path_tracks=["存5南"],
        action_type="ATTACH",
    )
    monkeypatch.setattr(
        "fzed_shunting.solver.move_candidates.generate_real_hook_moves",
        lambda *_args, **_kwargs: [release_move],
    )
    debug_stats = {}

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
        debug_stats=debug_stats,
    )

    assert any(
        candidate.steps == (release_move,)
        for candidate in candidates
    )
    assert debug_stats["protected_primitive_allowed_by_debt_count"] == 1
    assert debug_stats["protected_primitive_rejected_count"] == 0


def test_move_candidates_generate_work_position_window_candidate_with_order_buffer():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            *[
                _vehicle(f"PAD{index}", "洗南", "洗南", order=index)
                for index in range(1, 6)
            ],
            _vehicle("ORDER_BUFFER", "存5北", "洗南", order=1),
            _vehicle("SPOT1", "存5北", "洗南", order=2, spotting="是"),
            _vehicle("SPOT2", "存5北", "洗南", order=3, spotting="是"),
        ],
        "locoTrackName": "临2",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_source_opening"
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert candidate.structural_reserve
    assert candidate.focus_tracks == ("洗南",)
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "洗南"
        and step.vehicle_nos == ["SPOT1", "SPOT2"]
        for step in candidate.steps
    )
    assert not any(
        step.action_type == "DETACH"
        and step.target_track == "洗南"
        and "ORDER_BUFFER" in step.vehicle_nos
        for step in candidate.steps
    )

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert not next_state.loco_carry
    assert "ORDER_BUFFER" not in next_state.track_sequences["洗南"]
    assert south_rank(next_state.track_sequences["洗南"], "SPOT1") in allowed_spotting_south_ranks("洗南")
    assert south_rank(next_state.track_sequences["洗南"], "SPOT2") in allowed_spotting_south_ranks("洗南")


def test_move_candidates_open_source_prefix_by_dispatching_goal_blocks_before_work_window():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存3", "trackDistance": 367.0},
            {"trackName": "预修", "trackDistance": 109.6},
            {"trackName": "调棚", "trackDistance": 134.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
            _vehicle("TO_STORE_1", "存5北", "存3", order=1),
            _vehicle("TO_STORE_2", "存5北", "存3", order=2),
            _vehicle("TO_PRE", "存5北", "预修", order=3),
            _vehicle("FREE_BUFFER", "存5北", "调棚", order=4),
            _vehicle("SPOT1", "存5北", "调棚", order=5, spotting="是"),
            _vehicle("SPOT2", "存5北", "调棚", order=6, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_source_opening"
        and candidate.focus_tracks == ("调棚",)
    ]

    assert structural_candidates
    assert {candidate.kind for candidate in candidates} == {"primitive", "structural"}
    candidate = structural_candidates[0]
    detach_steps = [
        (step.action_type, step.source_track, step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert (
        "DETACH",
        "存5北",
        "存3",
        ["TO_STORE_1", "TO_STORE_2"],
    ) in detach_steps
    assert (
        "DETACH",
        "存5北",
        "预修",
        ["TO_PRE"],
    ) in detach_steps
    assert detach_steps.index(
        ("DETACH", "存5北", "存3", ["TO_STORE_1", "TO_STORE_2"]),
    ) < detach_steps.index(
        ("DETACH", "存5北", "调棚", ["SPOT1", "SPOT2"]),
    )
    assert detach_steps.index(
        ("DETACH", "存5北", "预修", ["TO_PRE"]),
    ) < detach_steps.index(
        ("DETACH", "存5北", "调棚", ["SPOT1", "SPOT2"]),
    )
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "存5北"
        and step.target_track in {"临1", "临2", "临3", "临4", "存4南"}
        and step.vehicle_nos == ["FREE_BUFFER"]
        for step in candidate.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "存5北"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["SPOT1", "SPOT2"]
        for step in candidate.steps
    )
    assert not any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and "FREE_BUFFER" in step.vehicle_nos
        for step in candidate.steps
    )

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert next_state.track_sequences["存3"][:2] == ["TO_STORE_1", "TO_STORE_2"]
    assert next_state.track_sequences["预修"][0] == "TO_PRE"
    assert "FREE_BUFFER" not in next_state.track_sequences["调棚"]
    assert south_rank(next_state.track_sequences["调棚"], "SPOT1") in allowed_spotting_south_ranks("调棚")
    assert south_rank(next_state.track_sequences["调棚"], "SPOT2") in allowed_spotting_south_ranks("调棚")


def test_move_candidates_insert_front_spotting_block_when_later_spotting_is_not_contiguous():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存2", "trackDistance": 113.0},
            {"trackName": "调棚", "trackDistance": 134.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
            _vehicle("SPOT_FRONT", "存5北", "调棚", order=1, spotting="是"),
            _vehicle("TO_STORE", "存5北", "存2", order=2),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=3, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_window_repair"
        and candidate.focus_tracks == ("调棚",)
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["SPOT_FRONT"]
        for step in candidate.steps
    )
    assert not any("SPOT_LATER" in step.vehicle_nos for step in candidate.steps)

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert south_rank(next_state.track_sequences["调棚"], "SPOT_FRONT") in allowed_spotting_south_ranks("调棚")


def test_move_candidates_fill_free_work_position_buffers_after_spotting_window_is_stable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "轮", "trackDistance": 118.2},
        ],
        "vehicleInfo": [
            _vehicle("SPOT1", "调棚", "调棚", order=1, spotting="是"),
            _vehicle("SPOT2", "调棚", "调棚", order=2, spotting="是"),
            _vehicle("SPOT3", "调棚", "调棚", order=3, spotting="是"),
            {**_vehicle("PAD1", "调棚", "调棚", order=4), "targetMode": "SNAPSHOT"},
            {**_vehicle("PAD2", "调棚", "调棚", order=5), "targetMode": "SNAPSHOT"},
            {**_vehicle("PAD3", "调棚", "调棚", order=6), "targetMode": "SNAPSHOT"},
            _vehicle("FREE_A", "存4南", "调棚", order=1),
            _vehicle("FREE_B", "存4南", "调棚", order=2),
            _vehicle("FREE_C", "存5北", "调棚", order=1),
            _vehicle("FREE_D", "存5北", "调棚", order=2),
            _vehicle("WHEEL", "存5北", "轮", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_free_fill"
        and candidate.focus_tracks == ("调棚",)
    ]

    assert structural_candidates
    assert {candidate.kind for candidate in candidates} == {"primitive", "structural"}
    candidate = structural_candidates[0]
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "存4南"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["FREE_A", "FREE_B"]
        for step in candidate.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "存5北"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["FREE_C", "FREE_D"]
        for step in candidate.steps
    )

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert not next_state.loco_carry
    assert all(vehicle_no in next_state.track_sequences["调棚"] for vehicle_no in ["FREE_A", "FREE_B", "FREE_C", "FREE_D"])
    assert south_rank(next_state.track_sequences["调棚"], "SPOT1") in allowed_spotting_south_ranks("调棚")
    assert south_rank(next_state.track_sequences["调棚"], "SPOT2") in allowed_spotting_south_ranks("调棚")
    assert south_rank(next_state.track_sequences["调棚"], "SPOT3") in allowed_spotting_south_ranks("调棚")


def test_move_candidates_fill_free_work_position_buffers_before_exact_slot_commitment():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("EXISTING_FREE", "调棚", "调棚", order=1),
            _vehicle("FREE_A", "存5北", "调棚", order=1),
            _vehicle("FREE_B", "存5北", "调棚", order=2),
            {
                **_vehicle("EXACT_SLOT", "存1", "调棚", order=1, spotting="是"),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_free_fill"
        and candidate.focus_tracks == ("调棚",)
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "存5北"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["FREE_A", "FREE_B"]
        for step in candidate.steps
    )
    assert not any("EXACT_SLOT" in step.vehicle_nos for step in candidate.steps)

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert next_state.track_sequences["调棚"][:2] == ["FREE_A", "FREE_B"]
    assert "EXACT_SLOT" not in next_state.track_sequences["调棚"]
    assert all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name="调棚",
            state=next_state,
            plan_input=normalized,
        )
        for vehicle_no in ["FREE_A", "FREE_B"]
    )


def test_move_candidates_open_protected_source_prefix_for_exact_work_slot():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            _vehicle("EXISTING_FREE", "调棚", "调棚", order=1),
            _vehicle("FREE_A", "调棚", "调棚", order=2),
            _vehicle("KEEP_A", "存1", "存1", order=1),
            _vehicle("KEEP_B", "存1", "存1", order=2),
            {
                **_vehicle("EXACT_SLOT", "存1", "调棚", order=3, spotting="是"),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "work_position_source_opening"
        and candidate.focus_tracks == ("调棚",)
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert [
        (step.action_type, step.source_track, step.target_track, step.vehicle_nos)
        for step in candidate.steps
    ] == [
        ("ATTACH", "存1", "存1", ["KEEP_A", "KEEP_B", "EXACT_SLOT"]),
        ("DETACH", "存1", "调棚", ["EXACT_SLOT"]),
        ("DETACH", "调棚", "存1", ["KEEP_A", "KEEP_B"]),
    ]

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert next_state.track_sequences["存1"] == ["KEEP_A", "KEEP_B"]
    assert next_state.track_sequences["调棚"][0] == "EXACT_SLOT"
    assert north_rank(next_state.track_sequences["调棚"], "EXACT_SLOT") == 1
    assert goal_is_satisfied(
        vehicle_by_no["EXACT_SLOT"],
        track_name="调棚",
        state=next_state,
        plan_input=normalized,
    )


def test_structural_candidates_do_not_clear_stable_spotting_window():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("FREE_A", "调棚", "调棚", order=1),
            _vehicle("SPOT1", "调棚", "调棚", order=2, spotting="是"),
            _vehicle("SPOT2", "调棚", "调棚", order=3, spotting="是"),
            _vehicle("SPOT3", "调棚", "调棚", order=4, spotting="是"),
            _vehicle("SPOT4", "调棚", "调棚", order=5, spotting="是"),
            {**_vehicle("PAD1", "调棚", "调棚", order=6), "targetMode": "SNAPSHOT"},
            {**_vehicle("PAD2", "调棚", "调棚", order=7), "targetMode": "SNAPSHOT"},
            _vehicle("SPOT0", "存5北", "调棚", order=1, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    stable_spotting = {"SPOT1", "SPOT2", "SPOT3", "SPOT4"}

    assert all(
        south_rank(state.track_sequences["调棚"], vehicle_no)
        in allowed_spotting_south_ranks("调棚")
        for vehicle_no in stable_spotting
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.kind == "structural"
        and any(
            step.action_type == "ATTACH"
            and step.source_track == "调棚"
            and stable_spotting.intersection(step.vehicle_nos)
            for step in candidate.steps
        )
        for candidate in candidates
    )


def test_move_candidates_generate_route_release_structural_candidate():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            _vehicle("SEEK", "临4", "存5北", order=1),
            _vehicle("BLOCK", "存5南", "存1", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "resource_release"
        and candidate.focus_tracks == ("存5南",)
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "存1"
        and step.vehicle_nos == ["BLOCK"]
        for step in candidate.steps
    )

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert next_state.track_sequences["存1"] == ["BLOCK"]
    assert not next_state.track_sequences.get("存5南")
    assert not next_state.loco_carry


def test_route_release_candidate_rejects_unrestorable_committed_blocker_clearance():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("SEEK", "临4", "存5北", order=1),
            _vehicle("COMMITTED_BLOCK", "存5南", "存5南", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    route_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "route_release_frontier"
        and candidate.focus_tracks == ("存5南", "临4", "存5北")
    ]

    assert not route_candidates


def test_route_release_frontier_candidate_does_not_clear_delayed_work_vehicle_to_goal():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "预修", "trackDistance": 109.6},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "存4南", "trackDistance": 154.5},
        ],
        "vehicleInfo": [
            _vehicle("TO_OIL", "存1", "油", order=1),
            _vehicle("KEEP_SOURCE", "存1", "存1", order=2),
            {
                **_vehicle("DELAYED_SLOT", "存1", "调棚", order=3),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
            _vehicle("SEEK", "临1", "预修", order=1),
            _vehicle("STRICT_SLOT", "调棚", "调棚", order=1, spotting="1"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    route_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "route_release_frontier"
        and candidate.focus_tracks == ("存1", "临1", "预修")
    ]

    assert not route_candidates


def test_route_release_frontier_candidate_keeps_clearance_when_delayed_trim_blocks_frontier():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _trim_candidate_before_delayed_commitment,
    )
    from fzed_shunting.solver.structural_intent import (
        DelayedCommitment,
        StructuralIntent,
    )

    candidate = MoveCandidate(
        steps=(
            HookAction(
                source_track="临1",
                target_track="临1",
                vehicle_nos=["BUFFER"],
                path_tracks=["临1"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="临1",
                target_track="临2",
                vehicle_nos=["BUFFER"],
                path_tracks=["临1", "临2"],
                action_type="DETACH",
            ),
            HookAction(
                source_track="存5北",
                target_track="存5北",
                vehicle_nos=["DELAYED_FRONTIER"],
                path_tracks=["存5北"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="存5北",
                target_track="调棚",
                vehicle_nos=["DELAYED_FRONTIER"],
                path_tracks=["存5北", "调棚"],
                action_type="DETACH",
            ),
        ),
        kind="structural",
        reason="route_release_frontier",
        focus_tracks=("临1", "存5北", "调棚"),
        structural_reserve=True,
    )
    intent = StructuralIntent(
        committed_blocks_by_track={},
        order_debts_by_track={},
        resource_debts=(),
        staging_buffers=(),
        delayed_commitments=(
            DelayedCommitment(
                vehicle_no="DELAYED_FRONTIER",
                target_track="调棚",
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    trimmed = _trim_candidate_before_delayed_commitment(
        candidate,
        vehicle_by_no={},
        intent=intent,
    )

    assert trimmed is not None
    assert trimmed.steps == candidate.steps[:2]


def test_best_staging_track_prefers_lower_route_blockage_pressure(monkeypatch):
    from types import SimpleNamespace

    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import (
        _best_staging_track,
        _ranked_staging_tracks,
    )

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("BUFFER", "存4南", "调棚", order=1),
            _vehicle("PAD", "调棚", "调棚", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fake_route_blockage_plan(plan_input, candidate_state, route_oracle, *, blocked_source_tracks=None):
        if "BUFFER" in candidate_state.track_sequences.get("临2", []):
            return SimpleNamespace(total_blockage_pressure=1)
        if "BUFFER" in candidate_state.track_sequences.get("临1", []):
            return SimpleNamespace(total_blockage_pressure=9)
        return SimpleNamespace(total_blockage_pressure=99)

    monkeypatch.setattr(
        move_candidates,
        "compute_route_blockage_plan",
        fake_route_blockage_plan,
    )

    stage_track = _best_staging_track(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="存4南",
        block=["BUFFER"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"存4南"},
        prefer_low_route_pressure=True,
    )

    assert stage_track == "临2"
    assert _ranked_staging_tracks(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="存4南",
        block=["BUFFER"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"存4南"},
        prefer_low_route_pressure=True,
    )[:2] == ["临2", "临1"]


def test_best_staging_track_uses_open_source_endpoint_when_releasing_whole_track(monkeypatch):
    from types import SimpleNamespace

    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import _best_staging_track

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("ROUTE_BLOCK", "临2", "调棚", order=1),
            {**_vehicle("OCCUPY_LIN1", "临1", "临1", order=1), "targetMode": "SNAPSHOT"},
            {**_vehicle("OCCUPY_CUN1", "存1", "存1", order=1), "targetMode": "SNAPSHOT"},
        ],
        "locoTrackName": "临2",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fake_route_blockage_plan(plan_input, candidate_state, route_oracle, *, blocked_source_tracks=None):
        if "ROUTE_BLOCK" in candidate_state.track_sequences.get("临3", []):
            return SimpleNamespace(total_blockage_pressure=1)
        if "ROUTE_BLOCK" in candidate_state.track_sequences.get("临1", []):
            return SimpleNamespace(total_blockage_pressure=9)
        return SimpleNamespace(total_blockage_pressure=99)

    monkeypatch.setattr(
        move_candidates,
        "compute_route_blockage_plan",
        fake_route_blockage_plan,
    )

    stage_track = _best_staging_track(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="临2",
        block=["ROUTE_BLOCK"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"临2"},
        prefer_low_route_pressure=True,
    )

    assert stage_track == "临3"


def test_best_staging_track_preserves_large_order_buffer_lease(monkeypatch):
    from types import SimpleNamespace

    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import _best_staging_track
    from fzed_shunting.solver.structural_intent import BufferLease

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {**_vehicle("ROUTE_BLOCK", "存5北", "存5北", order=1), "vehicleLength": 60.0},
            {**_vehicle("ORDER_BUFFER_A", "存5北", "调棚", order=2), "vehicleLength": 50.0},
            {**_vehicle("ORDER_BUFFER_B", "存5北", "调棚", order=3), "vehicleLength": 50.0},
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fake_route_blockage_plan(plan_input, candidate_state, route_oracle, *, blocked_source_tracks=None):
        if "ROUTE_BLOCK" in candidate_state.track_sequences.get("存4南", []):
            return SimpleNamespace(total_blockage_pressure=1)
        if "ROUTE_BLOCK" in candidate_state.track_sequences.get("临2", []):
            return SimpleNamespace(total_blockage_pressure=9)
        return SimpleNamespace(total_blockage_pressure=99)

    monkeypatch.setattr(
        move_candidates,
        "compute_route_blockage_plan",
        fake_route_blockage_plan,
    )

    stage_track = _best_staging_track(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="存5北",
        block=["ROUTE_BLOCK"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"存5北"},
        prefer_low_route_pressure=True,
        buffer_leases=(
            BufferLease(
                role="ORDER_BUFFER",
                vehicle_nos=("ORDER_BUFFER_A", "ORDER_BUFFER_B"),
                source_track="存5北",
                target_track="调棚",
                required_length=100.0,
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    assert stage_track == "临2"


def test_staging_track_prefers_lower_route_pressure_when_lease_cost_ties(monkeypatch):
    from types import SimpleNamespace

    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import _best_staging_track

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("BUFFER", "存5北", "存5北", order=1),
            _vehicle("SPOT", "存5北", "调棚", order=2, spotting="是"),
            _vehicle("PAD1", "调棚", "调棚", order=1),
            _vehicle("PAD2", "调棚", "调棚", order=2),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fake_route_blockage_plan(
        plan_input,
        candidate_state,
        route_oracle,
        *,
        blocked_source_tracks=None,
    ):
        if "BUFFER" in candidate_state.track_sequences.get("临2", []):
            return SimpleNamespace(total_blockage_pressure=1)
        if "BUFFER" in candidate_state.track_sequences.get("临4", []):
            return SimpleNamespace(total_blockage_pressure=3)
        return SimpleNamespace(total_blockage_pressure=99)

    monkeypatch.setattr(
        move_candidates,
        "compute_route_blockage_plan",
        fake_route_blockage_plan,
    )

    stage_track = _best_staging_track(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="存5北",
        block=["BUFFER"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"存5北"},
        prefer_low_route_pressure=True,
    )

    assert stage_track == "临2"


def test_staging_track_keeps_current_route_release_when_corridor_alternative_is_expensive(monkeypatch):
    from types import SimpleNamespace

    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import _best_staging_track

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("BUFFER", "存5北", "存5北", order=1),
            _vehicle("SPOT", "存5北", "调棚", order=2, spotting="是"),
            _vehicle("PAD1", "调棚", "调棚", order=1),
            _vehicle("PAD2", "调棚", "调棚", order=2),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fake_route_blockage_plan(
        plan_input,
        candidate_state,
        route_oracle,
        *,
        blocked_source_tracks=None,
    ):
        if "BUFFER" in candidate_state.track_sequences.get("临2", []):
            return SimpleNamespace(total_blockage_pressure=1)
        if "BUFFER" in candidate_state.track_sequences.get("临4", []):
            return SimpleNamespace(total_blockage_pressure=9)
        return SimpleNamespace(total_blockage_pressure=99)

    monkeypatch.setattr(
        move_candidates,
        "compute_route_blockage_plan",
        fake_route_blockage_plan,
    )

    stage_track = _best_staging_track(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        source_track="存5北",
        block=["BUFFER"],
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        forbidden_tracks={"存5北"},
        prefer_low_route_pressure=True,
    )

    assert stage_track == "临2"


def test_resource_release_candidate_dispatches_direct_goal_groups_before_staging():
    from fzed_shunting.solver.move_candidates import _build_resource_release_candidate
    from fzed_shunting.solver.structural_intent import ResourceDebt

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("TO_OIL_A", "存1", "油", order=1),
            _vehicle("TO_OIL_B", "存1", "油", order=2),
            _vehicle("KEEP_SOURCE", "存1", "存1", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidate = _build_resource_release_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        debt=ResourceDebt(
            kind="ROUTE_RELEASE",
            track_name="存1",
            vehicle_nos=("TO_OIL_A", "TO_OIL_B", "KEEP_SOURCE"),
            pressure=3.0,
        ),
    )

    assert candidate is not None
    detach_steps = [
        (step.source_track, step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert detach_steps[0] == ("存1", "存1", ["KEEP_SOURCE"])
    assert detach_steps[1][1] == "油"
    assert detach_steps[1][2] == ["TO_OIL_A", "TO_OIL_B"]


def test_front_clearance_candidate_dispatches_source_prefix_by_target_groups():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "轮", "trackDistance": 47.8},
            {"trackName": "存3", "trackDistance": 156.0},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            _vehicle("TO_STORE_A", "存5北", "存5南", order=1),
            _vehicle("TO_WHEEL_A", "存5北", "轮", order=2),
            _vehicle("TO_WHEEL_B", "存5北", "轮", order=3),
            _vehicle("TO_CUN3", "存5北", "存3", order=4),
            _vehicle("TO_STORE_B", "存5北", "存5南", order=5),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    candidate = next(
        (
            item
            for item in candidates
            if item.kind == "structural"
            and item.reason == "resource_release"
            and item.focus_tracks == ("存5北",)
            and len(item.steps) > 2
        ),
        None,
    )
    assert candidate is not None
    assert candidate.steps[0].action_type == "ATTACH"
    assert candidate.steps[0].source_track == "存5北"
    assert candidate.steps[0].vehicle_nos == [
        "TO_STORE_A",
        "TO_WHEEL_A",
        "TO_WHEEL_B",
        "TO_CUN3",
        "TO_STORE_B",
    ]
    detach_steps = [
        (step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert detach_steps == [
        ("存5南", ["TO_STORE_B"]),
        ("存3", ["TO_CUN3"]),
        ("轮", ["TO_WHEEL_A", "TO_WHEEL_B"]),
        ("存5南", ["TO_STORE_A"]),
    ]


def test_front_clearance_candidate_keeps_later_route_release_open_for_direct_goal_dispatch():
    from fzed_shunting.solver.candidate_compiler import replay_candidate_steps

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "轮", "trackDistance": 47.8},
            {"trackName": "存3", "trackDistance": 156.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临3", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("TO_STORE_A", "存5北", "存5南", order=1),
            _vehicle("TO_WHEEL_A", "存5北", "轮", order=2),
            _vehicle("TO_WHEEL_B", "存5北", "轮", order=3),
            _vehicle("TO_CUN3", "存5北", "存3", order=4),
            _vehicle("TO_DEPOT", "存5北", "大库", order=5),
            _vehicle("TO_STORE_B", "存5北", "存5南", order=6),
            _vehicle("ROUTE_BLOCK", "临1", "存5南", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={
            "存5北": [
                "TO_STORE_A",
                "TO_WHEEL_A",
                "TO_WHEEL_B",
                "TO_CUN3",
                "TO_DEPOT",
                "TO_STORE_B",
            ],
            "临1": ["ROUTE_BLOCK"],
        },
        loco_track_name="存4北",
        loco_node="L9",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    candidate = next(
        (
            item
            for item in candidates
            if item.kind == "structural"
            and item.reason == "resource_release"
            and item.focus_tracks == ("存5北",)
            and len(item.steps) > 2
        ),
        None,
    )
    assert candidate is not None
    detach_steps = [
        (step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert any(
        target_track in {"修1库内", "修2库内", "修3库内", "修4库内"}
        and vehicle_nos == ["TO_DEPOT"]
        for target_track, vehicle_nos in detach_steps
    )
    final = replay_candidate_steps(
        plan_input=normalized,
        state=state,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        steps=candidate.steps,
        route_oracle=RouteOracle(master),
    )
    assert final is not None
    assert RouteOracle(master).validate_loco_access(
        loco_track=final.final_state.loco_track_name,
        target_track="存5北",
        occupied_track_sequences=final.final_state.track_sequences,
        loco_node=final.final_state.loco_node,
    ).is_valid
    assert ("存5南", ["TO_STORE_A"]) in detach_steps


def test_resource_release_dispatch_preserves_source_reaccess_for_remaining_prefix():
    from fzed_shunting.solver.move_candidates import _build_resource_release_dispatch_candidate
    from fzed_shunting.solver.structural_intent import build_structural_intent

    master = load_master_data(DATA_DIR)
    scenario_path = (
        ROOT_DIR
        / "data"
        / "validation_inputs"
        / "positive"
        / "case_3_2_shed_work_gondola.json"
    )
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=route_oracle,
    )
    delayed_target_pairs = {
        (delayed.vehicle_no, delayed.target_track)
        for delayed in intent.delayed_commitments
    }

    candidate = _build_resource_release_dispatch_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no=vehicle_by_no,
        block=[
            "5265641",
            "1577445",
            "1663291",
            "1581012",
            "1574229",
            "4920676",
            "4921844",
            "1667888",
            "4906422",
            "1664387",
            "3810169",
            "1655769",
            "1662867",
        ],
        source_track="存5北",
        delayed_target_pairs=delayed_target_pairs,
        buffer_leases=intent.buffer_leases,
    )

    assert candidate is not None
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "存4南"
        and step.vehicle_nos == ["1655769", "1662867"]
        for step in candidate.steps
    )
    assert not any(
        step.action_type == "DETACH"
        and step.target_track in {"临1", "临4"}
        and step.vehicle_nos == ["1655769", "1662867"]
        for step in candidate.steps
    )


def test_carried_prefix_dispatch_stages_direct_group_when_preferred_route_is_blocked(monkeypatch):
    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.move_candidates import _carried_group_detach_step

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存3", "trackDistance": 258.5},
            {"trackName": "轮", "trackDistance": 118.2},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            _vehicle("TO_WHEEL_A", "存3", "轮", order=1),
            _vehicle("TO_WHEEL_B", "存3", "轮", order=2),
        ],
        "locoTrackName": "存3",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={"存3": []},
        loco_track_name="存3",
        loco_node="L4",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("TO_WHEEL_A", "TO_WHEEL_B"),
    )

    monkeypatch.setattr(
        move_candidates,
        "_ranked_staging_tracks",
        lambda **_kwargs: ["临1"],
    )
    monkeypatch.setattr(
        move_candidates,
        "_clear_path_tracks_for_detach",
        lambda **kwargs: None
        if kwargs["target_track"] == "轮"
        else ["存3", kwargs["target_track"]],
    )

    step = _carried_group_detach_step(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        group=["TO_WHEEL_A", "TO_WHEEL_B"],
        preferred_target="轮",
        used_staging_tracks={"存5北"},
        buffer_leases=(),
    )

    assert step is not None
    assert step.target_track == "临1"
    assert step.vehicle_nos == ["TO_WHEEL_A", "TO_WHEEL_B"]


def test_route_release_candidate_buffers_delayed_blocker_when_it_releases_route():
    from fzed_shunting.solver.move_candidates import _build_resource_release_candidate
    from fzed_shunting.solver.structural_intent import ResourceDebt

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            _vehicle("DELAYED_BLOCKER", "临2", "调棚", order=1),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=1, spotting="是"),
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidate = _build_resource_release_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        debt=ResourceDebt(
            kind="ROUTE_RELEASE",
            track_name="临2",
            vehicle_nos=("DELAYED_BLOCKER",),
            pressure=32.0,
        ),
        delayed_target_pairs={("DELAYED_BLOCKER", "调棚")},
    )

    assert candidate is not None
    assert any(
        step.action_type == "DETACH"
        and step.source_track == "临2"
        and step.target_track != "调棚"
        and step.vehicle_nos == ["DELAYED_BLOCKER"]
        for step in candidate.steps
    )


def test_move_candidates_allow_owned_resource_release_for_delayed_staging_blocker(monkeypatch):
    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.structural_intent import (
        DelayedCommitment,
        ResourceDebt,
        StructuralIntent,
    )

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("DELAYED_BLOCKER", "临2", "调棚", order=1),
            _vehicle("PAD1", "调棚", "调棚", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    monkeypatch.setattr(
        move_candidates,
        "build_structural_intent",
        lambda *_args, **_kwargs: StructuralIntent(
            committed_blocks_by_track={},
            order_debts_by_track={},
            resource_debts=(
                ResourceDebt(
                    kind="ROUTE_RELEASE",
                    track_name="临2",
                    vehicle_nos=("DELAYED_BLOCKER",),
                    pressure=32.0,
                ),
            ),
            staging_buffers=(),
            delayed_commitments=(
                DelayedCommitment(
                    vehicle_no="DELAYED_BLOCKER",
                    target_track="调棚",
                    reason="would_precede_unfinished_work_position_window",
                ),
            ),
        ),
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert any(
        candidate.kind == "structural"
        and candidate.reason == "resource_release"
        and any(
            step.action_type == "DETACH"
            and step.source_track == "临2"
            and step.target_track != "调棚"
            and step.vehicle_nos == ["DELAYED_BLOCKER"]
            for step in candidate.steps
        )
        for candidate in candidates
    )


def test_resource_release_groups_separate_committed_keep_from_order_buffer():
    from fzed_shunting.solver.move_candidates import _resource_release_prefix_groups

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("KEEP_A", "存1", "存1", order=1),
            _vehicle("KEEP_B", "存1", "存1", order=2),
            {
                **_vehicle("ORDER_BUFFER", "存1", "调棚", order=3, spotting="是"),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    groups = _resource_release_prefix_groups(
        block=["KEEP_A", "KEEP_B", "ORDER_BUFFER"],
        source_track="存1",
        state=state,
        plan_input=normalized,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        delayed_target_pairs={("ORDER_BUFFER", "调棚")},
    )

    assert groups == [
        ("存1", ["KEEP_A", "KEEP_B"]),
        (None, ["ORDER_BUFFER"]),
    ]


def test_resource_release_groups_defer_requested_route_release_target():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.solver.move_candidates import _resource_release_prefix_groups

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            _vehicle("TO_SHED", "存5北", "调棚", order=1),
            {**_vehicle("TO_REPAIR", "存5北", "存4北", order=2), "targetMode": "TRACK"},
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    groups = _resource_release_prefix_groups(
        block=["TO_SHED", "TO_REPAIR"],
        source_track="存5北",
        state=state,
        plan_input=normalized,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        deferred_target_tracks={"调棚"},
        route_oracle=RouteOracle(master),
    )

    assert groups == [
        (None, ["TO_SHED"]),
        ("存4北", ["TO_REPAIR"]),
    ]


def test_resource_release_groups_choose_effective_area_target_before_staging():
    from fzed_shunting.domain.route_oracle import RouteOracle
    from fzed_shunting.solver.move_candidates import _resource_release_prefix_groups

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            _vehicle("AREA_RANDOM", "存5北", "大库", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    groups = _resource_release_prefix_groups(
        block=["AREA_RANDOM"],
        source_track="存5北",
        state=state,
        plan_input=normalized,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        route_oracle=RouteOracle(master),
    )

    assert groups == [("修1库内", ["AREA_RANDOM"])]


def test_capacity_release_candidate_splits_carried_prefix_by_structural_role():
    from fzed_shunting.solver.move_candidates import _build_resource_release_candidate
    from fzed_shunting.solver.structural_intent import BufferLease, ResourceDebt

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("KEEP_A", "存1", "存1", order=1),
            _vehicle("TO_OIL", "存1", "油", order=2),
            {
                **_vehicle("ORDER_BUFFER", "存1", "调棚", order=3, spotting="是"),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidate = _build_resource_release_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        debt=ResourceDebt(
            kind="CAPACITY_RELEASE",
            track_name="存1",
            vehicle_nos=("KEEP_A", "TO_OIL", "ORDER_BUFFER"),
            pressure=3.0,
        ),
        delayed_target_pairs={("ORDER_BUFFER", "调棚")},
        buffer_leases=(
            BufferLease(
                role="ORDER_BUFFER",
                vehicle_nos=("ORDER_BUFFER",),
                source_track="存1",
                target_track="调棚",
                required_length=14.3,
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    assert candidate is not None
    detach_steps = [
        (step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert detach_steps[0][0] in {"临1", "临2", "临3", "临4", "存4南"}
    assert detach_steps[0][1] == ["ORDER_BUFFER"]
    assert detach_steps[1] == ("油", ["TO_OIL"])
    assert detach_steps[2] == ("存1", ["KEEP_A"])


def test_route_release_candidate_buffers_delayed_order_buffer_blocker():
    from fzed_shunting.solver.move_candidates import _build_resource_release_candidate
    from fzed_shunting.solver.structural_intent import BufferLease, ResourceDebt

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("TO_OIL", "存1", "油", order=1),
            {
                **_vehicle("ORDER_BUFFER", "存1", "调棚", order=2, spotting="是"),
                "targetMode": "SPOT",
                "targetSpotCode": "1",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidate = _build_resource_release_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        debt=ResourceDebt(
            kind="ROUTE_RELEASE",
            track_name="存1",
            vehicle_nos=("TO_OIL", "ORDER_BUFFER"),
            pressure=3.0,
        ),
        delayed_target_pairs={("ORDER_BUFFER", "调棚")},
        buffer_leases=(
            BufferLease(
                role="ORDER_BUFFER",
                vehicle_nos=("ORDER_BUFFER",),
                source_track="存1",
                target_track="调棚",
                required_length=14.3,
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    assert candidate is not None
    assert any(
        step.action_type == "DETACH"
        and step.target_track != "调棚"
        and step.vehicle_nos == ["ORDER_BUFFER"]
        for step in candidate.steps
    )


def test_route_release_frontier_dispatches_blocker_to_goal_before_staging():
    from fzed_shunting.solver.move_candidates import _build_route_release_frontier_candidate
    from fzed_shunting.solver.structural_intent import ResourceDebt

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            _vehicle("BUFFER_A", "临4", "大库", order=1),
            _vehicle("BUFFER_B", "临4", "大库", order=2),
            _vehicle("FRONTIER_A", "临3", "大库", order=1),
            _vehicle("FRONTIER_B", "临3", "大库", order=2),
        ],
        "locoTrackName": "临4",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidate = _build_route_release_frontier_candidate(
        plan_input=normalized,
        state=state,
        route_oracle=RouteOracle(master),
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        debt=ResourceDebt(
            kind="ROUTE_RELEASE",
            track_name="临4",
            vehicle_nos=("BUFFER_A", "BUFFER_B"),
            blocked_vehicle_nos=("FRONTIER_A", "FRONTIER_B"),
            source_tracks=("临3",),
            target_tracks=("修1库内",),
            pressure=2.0,
        ),
    )

    assert candidate is not None
    detach_steps = [
        (step.source_track, step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert detach_steps[0] == ("临4", "修1库内", ["BUFFER_A", "BUFFER_B"])
    assert detach_steps[1] == ("临3", "修1库内", ["FRONTIER_A", "FRONTIER_B"])


def test_route_release_frontier_defers_low_pressure_committed_blocker():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "预修", "trackDistance": 109.6},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "临3", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("KEEP_A", "存1", "存1", order=1),
            _vehicle("KEEP_B", "存1", "存1", order=2),
            _vehicle("SEEK", "存5北", "预修", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.kind == "structural"
        and candidate.reason in {"route_release_frontier", "resource_release"}
        and any(
            step.action_type == "ATTACH"
            and step.source_track == "存1"
            and set(step.vehicle_nos) == {"KEEP_A", "KEEP_B"}
            for step in candidate.steps
        )
        for candidate in candidates
    )


def test_move_candidates_reject_mixed_resource_debt_primitive_attach(monkeypatch):
    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.structural_intent import ResourceDebt, StructuralIntent

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("TO_OIL_A", "存1", "油", order=1),
            _vehicle("TO_OIL_B", "存1", "油", order=2),
            _vehicle("KEEP_SOURCE", "存1", "存1", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    mixed_attach = HookAction(
        source_track="存1",
        target_track="存1",
        vehicle_nos=["TO_OIL_A", "TO_OIL_B", "KEEP_SOURCE"],
        path_tracks=["存1"],
        action_type="ATTACH",
    )
    direct_group_attach = HookAction(
        source_track="存1",
        target_track="存1",
        vehicle_nos=["TO_OIL_A", "TO_OIL_B"],
        path_tracks=["存1"],
        action_type="ATTACH",
    )
    monkeypatch.setattr(
        move_candidates,
        "generate_real_hook_moves",
        lambda *_args, **_kwargs: [mixed_attach, direct_group_attach],
    )
    monkeypatch.setattr(
        move_candidates,
        "build_structural_intent",
        lambda *_args, **_kwargs: StructuralIntent(
            committed_blocks_by_track={},
            order_debts_by_track={},
            resource_debts=(
                ResourceDebt(
                    kind="ROUTE_RELEASE",
                    track_name="存1",
                    vehicle_nos=("TO_OIL_A", "TO_OIL_B", "KEEP_SOURCE"),
                    pressure=3.0,
                ),
            ),
            staging_buffers=(),
            delayed_commitments=(),
        ),
    )
    debug_stats = {}

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
        debug_stats=debug_stats,
    )

    primitive_steps = [
        candidate.steps[0]
        for candidate in candidates
        if candidate.kind == "primitive"
    ]
    assert mixed_attach not in primitive_steps
    assert direct_group_attach in primitive_steps
    assert debug_stats["mixed_resource_primitive_rejected_count"] == 1


def test_move_candidates_reject_primitive_attach_extending_resource_debt_across_roles(monkeypatch):
    from fzed_shunting.solver import move_candidates
    from fzed_shunting.solver.structural_intent import ResourceDebt, StructuralIntent

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "油", "trackDistance": 47.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("TO_OIL_A", "存1", "油", order=1),
            _vehicle("TO_OIL_B", "存1", "油", order=2),
            _vehicle("KEEP_SOURCE", "存1", "存1", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    over_attach = HookAction(
        source_track="存1",
        target_track="存1",
        vehicle_nos=["TO_OIL_A", "TO_OIL_B", "KEEP_SOURCE"],
        path_tracks=["存1"],
        action_type="ATTACH",
    )
    monkeypatch.setattr(
        move_candidates,
        "generate_real_hook_moves",
        lambda *_args, **_kwargs: [over_attach],
    )
    monkeypatch.setattr(
        move_candidates,
        "build_structural_intent",
        lambda *_args, **_kwargs: StructuralIntent(
            committed_blocks_by_track={},
            order_debts_by_track={},
            resource_debts=(
                ResourceDebt(
                    kind="CAPACITY_RELEASE",
                    track_name="存1",
                    vehicle_nos=("TO_OIL_A", "TO_OIL_B"),
                    pressure=20.0,
                ),
            ),
            staging_buffers=(),
            delayed_commitments=(),
        ),
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.kind == "primitive"
        and candidate.steps == (over_attach,)
        for candidate in candidates
    )


def test_move_candidates_stage_delayed_work_position_commitment_until_spotting_window_ready():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("FREE_FIRST", "存5北", "调棚", order=1),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=2, spotting="是"),
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and "FREE_FIRST" in step.vehicle_nos
        for candidate in candidates
        for step in candidate.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.target_track in {"临1", "临2", "临3", "临4", "存4南"}
        and step.vehicle_nos == ["FREE_FIRST"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_reject_staging_to_staging_churn_for_delayed_work_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("ORDER_BUFFER", "临1", "调棚", order=1),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=1, spotting="是"),
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
        ],
        "locoTrackName": "临1",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        deep=True,
        update={
            "track_sequences": {
                "临1": [],
                "存5北": ["SPOT_LATER"],
                "调棚": ["PAD1", "PAD2", "PAD3", "PAD4", "PAD5"],
            },
            "loco_track_name": "临1",
            "loco_carry": ("ORDER_BUFFER",),
        },
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        step.action_type == "DETACH"
        and step.target_track in {"临1", "临2", "临3", "临4", "存4南"}
        and step.vehicle_nos == ["ORDER_BUFFER"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_reject_staging_to_staging_churn_for_unfinished_buffer_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            _vehicle("CAPACITY_BUFFER", "临1", "大库", order=1),
        ],
        "locoTrackName": "临1",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        deep=True,
        update={
            "track_sequences": {
                "临1": [],
            },
            "loco_track_name": "临1",
            "loco_carry": ("CAPACITY_BUFFER",),
        },
    )

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        step.action_type == "DETACH"
        and step.target_track in {"临1", "临2", "临3", "临4", "存4南"}
        and step.vehicle_nos == ["CAPACITY_BUFFER"]
        for candidate in candidates
        for step in candidate.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.target_track in {"修1库内", "修2库内"}
        and step.vehicle_nos == ["CAPACITY_BUFFER"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_reject_unowned_delayed_staging_attach():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("ORDER_BUFFER", "临1", "调棚", order=1),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=1, spotting="是"),
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
        ],
        "locoTrackName": "临1",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.kind == "primitive"
        and step.action_type == "ATTACH"
        and step.source_track == "临1"
        and step.vehicle_nos == ["ORDER_BUFFER"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_reject_unowned_delayed_storage_buffer_attach():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("ORDER_BUFFER", "存4南", "调棚", order=1),
            _vehicle("SPOT_LATER", "存5北", "调棚", order=1, spotting="是"),
            *[
                _vehicle(f"PAD{index}", "调棚", "调棚", order=index)
                for index in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.kind == "primitive"
        and step.action_type == "ATTACH"
        and step.source_track == "存4南"
        and step.vehicle_nos == ["ORDER_BUFFER"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_emit_work_position_structural_candidate_when_sequence_defect_is_zero():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("FREE_FIRST", "存5北", "调棚", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    metrics = compute_structural_metrics(normalized, state)
    assert metrics.work_position_unfinished_count == 1
    assert metrics.target_sequence_defect_count == 0

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert any(
        candidate.kind == "structural"
        and candidate.reason == "work_position_free_fill"
        and candidate.focus_tracks == ("调棚",)
        for candidate in candidates
    )
    assert not any(
        candidate.kind == "primitive"
        and step.action_type == "DETACH"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["FREE_FIRST"]
        for candidate in candidates
        for step in candidate.steps
    )


def test_move_candidates_open_buried_free_work_position_vehicle_with_prefix_dispatch():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "存1", "trackDistance": 20.0},
            {"trackName": "抛", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            _vehicle("BLOCK1", "预修", "存1", order=1),
            _vehicle("BLOCK2", "预修", "存1", order=2),
            _vehicle("FREE_FIRST", "预修", "抛", order=3),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert any(
        candidate.kind == "structural"
        and candidate.reason == "work_position_source_opening"
        and candidate.focus_tracks == ("抛",)
        for candidate in candidates
    )


def test_work_position_prefix_staging_keeps_source_to_target_route_open():
    from fzed_shunting.solver.move_candidates import (
        _next_prefix_staging_chunk,
        _source_to_target_route_is_open,
    )

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "抛", "trackDistance": 131.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
        ],
        "vehicleInfo": [
            _vehicle("KEEP1", "预修", "预修", order=1),
            _vehicle("KEEP2", "预修", "预修", order=2),
            _vehicle("KEEP3", "预修", "预修", order=3),
            _vehicle("KEEP4", "预修", "预修", order=4),
            _vehicle("FREE1", "预修", "抛", order=5),
            _vehicle("FREE2", "预修", "抛", order=6),
            _vehicle("PAD", "抛", "抛", order=1),
        ],
        "locoTrackName": "预修",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)
    occupied_route_state = ReplayState(
        track_sequences={
            **state.track_sequences,
            "临4": ["KEEP1", "KEEP2"],
            "临3": ["ROUTE_BLOCK_1"],
            "临2": ["ROUTE_BLOCK_2"],
            "临1": ["ROUTE_BLOCK_3"],
            "存4南": ["ROUTE_BLOCK_4"],
            "存5北": ["ROUTE_BLOCK_5"],
            "调棚": ["ROUTE_BLOCK_6"],
            "油": ["ROUTE_BLOCK_7"],
            "轮": ["ROUTE_BLOCK_8"],
            "存4北": ["ROUTE_BLOCK_9"],
            "存1": ["ROUTE_BLOCK_10"],
            "存2": ["ROUTE_BLOCK_11"],
            "存3": ["ROUTE_BLOCK_12"],
            "机库": ["ROUTE_BLOCK_13"],
            "修1库内": ["ROUTE_BLOCK_14"],
            "修2库内": ["ROUTE_BLOCK_15"],
            "修3库内": ["ROUTE_BLOCK_16"],
            "修4库内": ["ROUTE_BLOCK_17"],
        },
        loco_track_name="临4",
        loco_node="L8",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )

    assert not _source_to_target_route_is_open(
        state=occupied_route_state,
        route_oracle=route_oracle,
        source_track="预修",
        target_track="抛",
        transfer_block=["FREE1", "FREE2"],
    )

    chunk_plan = _next_prefix_staging_chunk(
        plan_input=normalized,
        state=state,
        route_oracle=route_oracle,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        source_track="预修",
        target_track="抛",
        transfer_block=["FREE1", "FREE2"],
        remaining_prefix=["KEEP1", "KEEP2", "KEEP3", "KEEP4"],
        forbidden_tracks={"预修", "抛"},
        allow_split_prefix=True,
    )

    assert chunk_plan is not None
    stage_track, _chunk, _steps, next_state = chunk_plan
    assert stage_track != "临4"
    assert _source_to_target_route_is_open(
        state=next_state,
        route_oracle=route_oracle,
        source_track="预修",
        target_track="抛",
        transfer_block=["FREE1", "FREE2"],
    )


def test_trim_candidate_rejects_structural_delayed_buffer_regrab_from_staging():
    from fzed_shunting.solver.move_candidates import (
        MoveCandidate,
        _trim_candidate_before_delayed_commitment,
    )
    from fzed_shunting.solver.structural_intent import (
        DelayedCommitment,
        StructuralIntent,
    )

    candidate = MoveCandidate(
        steps=(
            HookAction(
                source_track="存4南",
                target_track="存4南",
                vehicle_nos=["ORDER_BUFFER"],
                path_tracks=["存4南"],
                action_type="ATTACH",
            ),
            HookAction(
                source_track="存4南",
                target_track="临2",
                vehicle_nos=["ORDER_BUFFER"],
                path_tracks=["存4南", "临2"],
                action_type="DETACH",
            ),
        ),
        kind="structural",
        reason="resource_release",
        focus_tracks=("存4南",),
        structural_reserve=True,
    )
    intent = StructuralIntent(
        committed_blocks_by_track={},
        order_debts_by_track={},
        resource_debts=(),
        staging_buffers=(),
        delayed_commitments=(
            DelayedCommitment(
                vehicle_no="ORDER_BUFFER",
                target_track="调棚",
                reason="would_precede_unfinished_work_position_window",
            ),
        ),
    )

    assert _trim_candidate_before_delayed_commitment(
        candidate,
        vehicle_by_no={},
        intent=intent,
    ) is None


def test_move_candidates_open_goal_frontier_and_restore_committed_prefix():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("DONE_A", "修1库内", "修1库内", order=1),
            _vehicle("DONE_B", "修1库内", "修1库内", order=2),
            _vehicle("TODO_A", "修1库内", "存4北", order=3),
            _vehicle("TODO_B", "修1库内", "存4北", order=4),
            _vehicle("DONE_C", "修1库内", "修1库内", order=5),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    structural_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "structural"
        and candidate.reason == "goal_frontier_source_opening"
        and candidate.focus_tracks == ("修1库内", "存4北")
    ]

    assert structural_candidates
    candidate = structural_candidates[0]
    assert [step.action_type for step in candidate.steps] == ["ATTACH", "DETACH", "DETACH"]
    assert candidate.steps[0].source_track == "修1库内"
    assert candidate.steps[0].vehicle_nos == ["DONE_A", "DONE_B", "TODO_A", "TODO_B"]
    detach_steps = [
        (step.source_track, step.target_track, step.vehicle_nos)
        for step in candidate.steps
        if step.action_type == "DETACH"
    ]
    assert detach_steps[0] == ("修1库内", "存4北", ["TODO_A", "TODO_B"])
    assert detach_steps[1] == ("存4北", "修1库内", ["DONE_A", "DONE_B"])

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidate.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    assert next_state.track_sequences["存4北"][:2] == ["TODO_A", "TODO_B"]
    assert next_state.track_sequences["修1库内"][:3] == ["DONE_A", "DONE_B", "DONE_C"]
    assert not next_state.loco_carry
    assert all(
        goal_is_satisfied(
            vehicle_by_no[vehicle_no],
            track_name="修1库内",
            state=next_state,
            plan_input=normalized,
        )
        for vehicle_no in ["DONE_A", "DONE_B", "DONE_C"]
    )


def test_goal_frontier_candidate_leaves_work_position_blocks_to_work_position_generator():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 62.9},
        ],
        "vehicleInfo": [
            _vehicle("DONE", "存5北", "存5北", order=1),
            _vehicle("FREE_BUFFER", "存5北", "调棚", order=2),
            _vehicle("SPOT1", "存5北", "调棚", order=3, spotting="是"),
            _vehicle("SPOT2", "存5北", "调棚", order=4, spotting="是"),
            _vehicle("PAD1", "调棚", "调棚", order=1),
            _vehicle("PAD2", "调棚", "调棚", order=2),
            _vehicle("PAD3", "调棚", "调棚", order=3),
            _vehicle("PAD4", "调棚", "调棚", order=4),
            _vehicle("PAD5", "调棚", "调棚", order=5),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        candidate.reason == "goal_frontier_source_opening"
        and candidate.focus_tracks == ("存5北", "调棚")
        for candidate in candidates
    )
    assert any(
        candidate.reason in {
            "work_position_source_opening",
            "work_position_window_repair",
        }
        and candidate.focus_tracks == ("调棚",)
        for candidate in candidates
    )


class _SyntheticRouteBlockagePlan:
    total_blockage_pressure = 1
    facts_by_blocking_track = {}


def test_generate_real_hook_moves_reuses_route_blockage_pressure_by_state(monkeypatch):
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
        "fzed_shunting.solver.move_generator.compute_route_blockage_plan",
        fake_compute_route_blockage_plan,
    )

    moves = generate_real_hook_moves(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert moves
    assert max(calls_by_state.values(), default=0) == 1


def test_empty_carry_followup_probe_does_not_recurse_into_full_move_generation(monkeypatch):
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
                "vehicleNo": "A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": ["A"], "机库": []},
            "loco_track_name": "机库",
            "loco_carry": ("B",),
        }
    )

    def recurse(*_args, **_kwargs):
        raise AssertionError("followup probe should not call full move generation")

    monkeypatch.setattr(move_generator, "generate_real_hook_moves", recurse)

    assert move_generator._empty_carry_detach_has_followup_attach(
        detach_move=HookAction(
            source_track="机库",
            target_track="临1",
            vehicle_nos=["B"],
            path_tracks=["机库", "临1"],
            action_type="DETACH",
        ),
        drop_block=["B"],
        state=state,
        plan_input=normalized,
        vehicle_by_no={vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles},
        master=master,
        route_oracle=RouteOracle(master),
        followup_attach_cache={},
    )


def test_generate_real_hook_moves_returns_stable_semantic_order():
    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (ROOT_DIR / "data/validation_inputs/truth/validation_20260327Z.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    repeated = generate_real_hook_moves(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )
    assert moves == repeated


def test_candidate_staging_targets_can_use_snapshot_fallback_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SNAP_STAGE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存1",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "存1",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}

    targets = _candidate_staging_targets(
        source_track="存1",
        block=["SNAP_STAGE"],
        state=state,
        plan_input=normalized,
        master=master,
        vehicle_by_no=vehicle_by_no,
        goal_target_hints=("存1",),
        route_oracle=RouteOracle(master),
    )

    assert "存2" in targets


def test_candidate_staging_targets_prefers_depot_inner_for_depot_exact_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_EXACT",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "调棚",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"调棚": []},
            "loco_track_name": "调棚",
            "loco_carry": ("DEPOT_EXACT",),
        }
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}

    targets = _candidate_staging_targets(
        source_track="调棚",
        block=["DEPOT_EXACT"],
        state=state,
        plan_input=normalized,
        master=master,
        vehicle_by_no=vehicle_by_no,
        goal_target_hints=("修1库内",),
        route_oracle=RouteOracle(master),
        route_pressure_sort=False,
    )

    assert set(targets[:3]) == {"修2库内", "修3库内", "修4库内"}
    assert "修1库内" not in targets
    assert targets.index("临1") > 2


def test_generate_goal_moves_from_north_prefix_only():
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
                "vehicleNo": "D1",
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
                "vehicleNo": "D2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    move_blocks = [tuple(move.vehicle_nos) for move in moves]
    assert ("D1",) in move_blocks
    assert ("D2",) not in move_blocks


def test_generate_goal_moves_keeps_only_longest_feasible_prefix_for_single_target_goal():
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
                "vehicleNo": "P1",
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
                "vehicleNo": "P2",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert [tuple(move.vehicle_nos) for move in moves] == [("P1", "P2")]


def test_generate_goal_moves_keeps_shorter_prefix_when_longest_single_target_prefix_does_not_fit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 20.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "Q1",
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
                "vehicleNo": "Q2",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    assert [tuple(move.vehicle_nos) for move in moves] == [("Q1",)]


def test_generate_goal_moves_keeps_shorter_prefix_for_single_target_goal_when_target_track_is_not_empty():
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
                "vehicleNo": "R1",
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
                "vehicleNo": "R2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "R_OCC",
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
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state)
        if move.source_track == "存5北"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("R1", "R2"), ("R1",)]


def test_generate_goal_moves_keeps_shorter_prefix_for_single_target_goal_from_temporary_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 120.0},
            {"trackName": "修3库外", "trackDistance": 120.0},
            {"trackName": "机库", "trackDistance": 120.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临3",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master, route_oracle=RouteOracle(master))
        if move.source_track == "临3" and move.target_track == "修3库外"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("TMP1", "TMP2"), ("TMP1",)]


def test_generate_goal_moves_keeps_only_longest_prefix_for_three_car_temporary_same_goal_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 120.0},
            {"trackName": "修3库外", "trackDistance": 120.0},
            {"trackName": "机库", "trackDistance": 120.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临3",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "临3",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "TMP3_3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master, route_oracle=RouteOracle(master))
        if move.source_track == "临3" and move.target_track == "修3库外"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [("TMP3_1", "TMP3_2", "TMP3_3")]


def test_generate_goal_moves_keeps_shorter_prefixes_when_same_goal_front_block_does_not_consume_source_track():
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
                "vehicleNo": "M1",
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
                "vehicleNo": "M2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "M3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "M4",
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
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state)
        if move.source_track == "存5北" and move.target_track == "机库"
    ]

    assert [tuple(move.vehicle_nos) for move in moves] == [
        ("M1", "M2", "M3"),
        ("M1", "M2"),
        ("M1",),
    ]


def test_generate_goal_moves_rejects_single_wash_spotting_without_south_pad():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
        ],
        "vehicleInfo": [
            _vehicle("WASH_TARGET", "存5北", "洗南", spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert not any(
        move.source_track == "存5北"
        and move.target_track == "洗南"
        and tuple(move.vehicle_nos) == ("WASH_TARGET",)
        for move in moves
    )


def test_generate_goal_moves_allows_wash_spotting_with_same_hook_south_pad():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "洗南", "trackDistance": 88.7},
        ],
        "vehicleInfo": [
            _vehicle("WASH_TARGET", "存5北", "洗南", order=1, spotting="是"),
            _vehicle("SOUTH_PAD", "存5北", "洗南", order=2),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5北"
        and move.target_track == "洗南"
        and tuple(move.vehicle_nos) == ("WASH_TARGET", "SOUTH_PAD")
        for move in moves
    )


def test_generate_real_hook_moves_rejects_exact_rank_detach_past_target_rank():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            _vehicle("N1", "存5北", "调棚", order=1),
            _vehicle("N2", "存5北", "调棚", order=2),
            _vehicle("EXACT", "存5北", "调棚", order=3, spotting="2"),
        ],
        "locoTrackName": "调棚",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={},
        loco_track_name="调棚",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("N1", "N2", "EXACT"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "调棚"
        and tuple(move.vehicle_nos) == ("N1", "N2", "EXACT")
        for move in moves
    )


def test_generate_real_hook_moves_allows_exact_rank_detach_before_target_rank():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            _vehicle("EXACT", "存5北", "调棚", order=1, spotting="2"),
        ],
        "locoTrackName": "调棚",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={},
        loco_track_name="调棚",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("EXACT",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track == "调棚"
        and tuple(move.vehicle_nos) == ("EXACT",)
        for move in moves
    )


def test_generate_real_hook_moves_rejects_duplicate_explicit_work_slot_detach():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "存5北", "trackDistance": 367},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "罐车",
                "vehicleNo": "SLOT_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "洗南",
                "targetSpotCode": "2",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "罐车",
                "vehicleNo": "SLOT_B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SPOT",
                "targetTrack": "洗南",
                "targetSpotCode": "2",
                "isSpotting": "是",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "洗南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={},
        loco_track_name="洗南",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("SLOT_A", "SLOT_B"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "洗南"
        and tuple(move.vehicle_nos) == ("SLOT_A", "SLOT_B")
        for move in moves
    )


def test_identity_attach_requests_include_unsatisfied_work_position_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("EXACT", "调棚", "调棚", order=1, spotting="2"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}
    length_by_vehicle = {
        vehicle.vehicle_no: vehicle.vehicle_length for vehicle in normalized.vehicles
    }
    capacity_by_track = {
        info.track_name: info.track_distance for info in normalized.track_info
    }

    requests = _collect_real_hook_identity_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        length_by_vehicle=length_by_vehicle,
        effective_capacity_by_track=capacity_by_track,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert requests["调棚"] == {1}


def test_generate_goal_moves_respects_allowed_target_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修2库外", "trackDistance": 49.3},
            {"trackName": "修3库外", "trackDistance": 49.3},
            {"trackName": "修4库外", "trackDistance": 49.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state)

    targets = {move.target_track for move in moves}
    assert targets == {"修1库外", "修2库外", "修3库外", "修4库外"}


def test_generate_goal_moves_prefers_1_2_for_short_random_depot_vehicle():
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
                "vehicleNo": "DEPOT_PREF",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert [move.target_track for move in moves] == ["修1库内", "修2库内"]


def test_generate_goal_moves_keeps_random_depot_capacity_balanced_across_preferred_tracks():
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
                "vehicleNo": "DEPOT_CLUSTER",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_201",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "201",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert [move.target_track for move in moves] == ["修1库内", "修2库内"]


def test_generate_goal_moves_allows_3_4_fallback_only_when_1_2_are_full_for_short_random_depot_vehicle():
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
                "vehicleNo": "DEPOT_ORDER",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_101",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_102",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "102",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_103",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "103",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_104",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "104",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库内",
                "order": "5",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_105",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "105",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_201",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "201",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_202",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "202",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_203",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "203",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_204",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "204",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修2库内",
                "order": "5",
                "vehicleModel": "棚车",
                "vehicleNo": "OCC_205",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "205",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master)
        if move.source_track == "存5北"
    ]

    assert [move.target_track for move in moves] == ["修3库内", "修4库内"]


def test_generate_goal_moves_keeps_short_random_depot_on_preferred_when_1_2_have_releaseable_occupants():
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
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_ORDER",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "修1库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"R1_{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
            *[
                {
                    "trackName": "修2库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"R2_{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存4北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master)
        if move.source_track == "存5北"
    ]

    assert [move.target_track for move in moves] == ["修1库内", "修2库内"]


def test_generate_goal_moves_counts_track_mode_depot_occupants_against_random_depot_capacity():
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
                "vehicleNo": "DEPOT_TRACK_FULL",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "修1库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"TRACK1_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetMode": "TRACK",
                    "targetTrack": "修1库内",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
            *[
                {
                    "trackName": "修2库内",
                    "order": str(idx),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"TRACK2_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetMode": "TRACK",
                    "targetTrack": "修2库内",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 6)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = [
        move
        for move in generate_goal_moves(normalized, state, master=master)
        if move.source_track == "存5北"
    ]

    assert [move.target_track for move in moves] == ["修3库内", "修4库内"]


def test_random_depot_effective_targets_are_not_rewritten_by_overflow_template():
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
            *[
                {
                    "trackName": "修1库内" if idx <= 5 else "修2库内",
                    "order": str(idx if idx <= 5 else idx - 5),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"PREF_{idx}",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx in range(1, 11)
            ],
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OVERFLOW_STABLE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OVERFLOW_PENDING",
                "repairProcess": "厂修",
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

    stable_vehicle = vehicle_by_no["OVERFLOW_STABLE"]
    pending_vehicle = vehicle_by_no["OVERFLOW_PENDING"]

    assert goal_is_satisfied(
        stable_vehicle,
        track_name="修3库内",
        state=state,
        plan_input=normalized,
    )
    assert goal_effective_allowed_tracks(
        stable_vehicle,
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]
    assert goal_effective_allowed_tracks(
        pending_vehicle,
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]
    assert goal_effective_allowed_tracks(
        vehicle_by_no["PREF_1"],
        state=state,
        plan_input=normalized,
    ) == ["修1库内", "修2库内", "修3库内", "修4库内"]


def test_generate_real_hook_moves_keeps_dirty_goal_target_as_soft_candidate():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LONG_INBOUND",
                "repairProcess": "厂修",
                "vehicleLength": 17.6,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL_MUST_LEAVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": [], "修3库内": ["TAIL_MUST_LEAVE"]},
            "loco_track_name": "存5北",
            "loco_carry": ("LONG_INBOUND",),
            "spot_assignments": {},
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(move.target_track == "修4库内" for move in moves)
    assert any(
        move.action_type == "DETACH"
        and move.target_track == "修3库内"
        and tuple(move.vehicle_nos) == ("LONG_INBOUND",)
        for move in moves
    )

    tail_vehicle = vehicle_by_no["TAIL_MUST_LEAVE"]
    assert not goal_is_satisfied(
        tail_vehicle,
        track_name="修3库内",
        state=state,
        plan_input=normalized,
    )


def test_generate_goal_moves_supports_oil_and_shot_targets_when_missing_segment_is_40m_placeholder():
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
                "vehicleNo": "D4",
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
                "vehicleNo": "D4O",
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
    shot_state = build_initial_state(shot_normalized)
    oil_normalized = normalize_plan_input(oil_payload, master)
    oil_state = build_initial_state(oil_normalized)

    shot_moves = generate_goal_moves(shot_normalized, shot_state, master=master)
    oil_moves = generate_goal_moves(oil_normalized, oil_state, master=master)

    assert len(shot_moves) == 1
    assert shot_moves[0].target_track == "抛"
    assert len(oil_moves) == 1
    assert oil_moves[0].target_track == "油"


def test_generate_goal_moves_filters_l1_overflow():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "D5",
                "repairProcess": "段修",
                "vehicleLength": 200.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert moves == []


def test_generate_goal_moves_uses_clear_alternate_route_when_default_path_is_blocked():
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
                "vehicleNo": "D6",
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
                "vehicleNo": "D7",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert [
        move
        for move in moves
        if move.source_track == "存5北" and move.target_track == "修1库内"
    ]


def test_generate_goal_moves_can_emit_temporary_clear_move_for_front_blocker():
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
                "vehicleNo": "BLOCK_FRONT",
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
                "vehicleNo": "NEED_OUT",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5北"
        and move.target_track == "临1"
        and move.vehicle_nos == ["BLOCK_FRONT"]
        for move in moves
    )


def test_generate_goal_moves_skips_pure_random_depot_rebalancing_when_block_is_already_satisfied():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RANDOM_OK",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "OTHER_GO",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert not any(move.source_track == "修3库内" for move in moves)


def test_generate_goal_moves_keeps_random_depot_front_blocker_movable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "RAND_FRONT",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "NEED_C4B",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "修3库内"
        and move.vehicle_nos == ["RAND_FRONT"]
        and move.target_track in {"修1库内", "修2库内", "修4库内"}
        for move in moves
    )


def test_generate_goal_moves_can_emit_temporary_clear_move_to_cun4nan():
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
                "vehicleNo": "C4N_BLOCK",
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
                "vehicleNo": "C4N_GO",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5北"
        and move.target_track == "存4南"
        and move.vehicle_nos == ["C4N_BLOCK"]
        for move in moves
    )


def test_generate_goal_moves_can_emit_temporary_clear_move_when_all_alternate_routes_are_blocked():
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
                "vehicleNo": "PATH_NEED",
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
                "vehicleNo": "PATH_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ALT_BLOCK",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)

    assert any(
        move.source_track == "存5南"
        and move.target_track == "临1"
        and move.vehicle_nos == ["PATH_BLOCK"]
        for move in moves
    )


def test_generate_goal_moves_records_debug_stats_for_direct_and_staging_moves():
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
                "vehicleNo": "DBG_BLOCK",
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
                "vehicleNo": "DBG_GO",
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
    state = build_initial_state(normalized)
    debug_stats: dict = {}

    moves = generate_goal_moves(normalized, state, master=master, debug_stats=debug_stats)

    assert moves
    assert debug_stats["total_moves"] == len(moves)
    assert debug_stats["staging_moves"] >= 1
    assert debug_stats["moves_by_target"]["临1"] >= 1
    assert debug_stats["moves_by_source"]["存5北"] == len(moves)
    assert debug_stats["moves_by_block_size"][1] >= 1


def test_generate_goal_moves_limits_front_blocker_staging_targets_to_nearest_temporaries():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LIMIT_BLOCK",
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
                "vehicleNo": "LIMIT_GO",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert {move.target_track for move in staging_moves} == {"临1", "临2"}


def test_route_release_staging_targets_include_lower_pressure_storage_before_cutoff():
    from fzed_shunting.solver.state import _apply_move

    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "data" / "validation_inputs" / "truth" / "validation_20260206W.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    prefix = [
        ("ATTACH", "机棚", "机棚", ["3451765", "3451387", "3825233"]),
        ("DETACH", "机棚", "油", ["3451765", "3451387", "3825233"]),
        ("ATTACH", "调棚", "调棚", ["5343660", "5740767", "5252529", "5743973", "5345070", "1760360", "4922071", "5280733"]),
        ("DETACH", "调棚", "修2库内", ["4922071", "5280733"]),
        ("ATTACH", "机库", "机库", ["5489133", "5320800", "5237784", "1676330", "1849948"]),
        ("DETACH", "机库", "修2库内", ["1849948"]),
        ("DETACH", "修2库内", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330"]),
        ("ATTACH", "修4库内", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330", "5342375", "5349272", "5346073", "4921581"]),
        ("DETACH", "修4库内", "存4北", ["5342375", "5349272", "5346073", "4921581"]),
        ("DETACH", "存4北", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330"]),
        ("DETACH", "修4库内", "临4", ["5343660", "5740767", "5252529", "5743973", "5345070"]),
        ("ATTACH", "预修", "预修", ["3406195", "1849573", "4904291", "1657219", "1517444", "4866019", "4887695", "4922413", "1663044", "1660229"]),
        ("DETACH", "预修", "洗北", ["1657219", "1517444", "4866019", "4887695", "4922413", "1663044", "1660229"]),
        ("DETACH", "洗北", "预修", ["3406195", "1849573", "4904291"]),
        ("ATTACH", "存2", "存2", ["5313847", "5243183"]),
    ]
    for action_type, source, target, vehicle_nos in prefix:
        state = _apply_move(
            state=state,
            move=HookAction(
                source_track=source,
                target_track=target,
                vehicle_nos=vehicle_nos,
                path_tracks=[source] if source == target else [source, target],
                action_type=action_type,
            ),
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    moves = generate_real_hook_moves(normalized, state, master=master, route_oracle=route_oracle)
    staging_targets = {
        move.target_track
        for move in moves
        if move.action_type == "DETACH"
        and move.source_track == "存2"
        and move.vehicle_nos == ["5313847", "5243183"]
    }

    assert staging_targets & {"存3", "存1", "洗北", "调北"}
    assert staging_targets != {"临1", "临2"}


def test_candidate_staging_targets_do_not_project_route_pressure_per_candidate(monkeypatch):
    from fzed_shunting.solver import move_generator
    from fzed_shunting.solver.move_generator import _candidate_staging_targets
    from fzed_shunting.solver.state import _apply_move

    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "data" / "validation_inputs" / "truth" / "validation_20260206W.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    prefix = [
        ("ATTACH", "机棚", "机棚", ["3451765", "3451387", "3825233"]),
        ("DETACH", "机棚", "油", ["3451765", "3451387", "3825233"]),
        ("ATTACH", "调棚", "调棚", ["5343660", "5740767", "5252529", "5743973", "5345070", "1760360", "4922071", "5280733"]),
        ("DETACH", "调棚", "修2库内", ["4922071", "5280733"]),
        ("ATTACH", "机库", "机库", ["5489133", "5320800", "5237784", "1676330", "1849948"]),
        ("DETACH", "机库", "修2库内", ["1849948"]),
        ("DETACH", "修2库内", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330"]),
        ("ATTACH", "修4库内", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330", "5342375", "5349272", "5346073", "4921581"]),
        ("DETACH", "修4库内", "存4北", ["5342375", "5349272", "5346073", "4921581"]),
        ("DETACH", "存4北", "修4库内", ["1760360", "5489133", "5320800", "5237784", "1676330"]),
        ("DETACH", "修4库内", "临4", ["5343660", "5740767", "5252529", "5743973", "5345070"]),
        ("ATTACH", "预修", "预修", ["3406195", "1849573", "4904291", "1657219", "1517444", "4866019", "4887695", "4922413", "1663044", "1660229"]),
        ("DETACH", "预修", "洗北", ["1657219", "1517444", "4866019", "4887695", "4922413", "1663044", "1660229"]),
        ("DETACH", "洗北", "预修", ["3406195", "1849573", "4904291"]),
        ("ATTACH", "存2", "存2", ["5313847", "5243183"]),
    ]
    for action_type, source, target, vehicle_nos in prefix:
        state = _apply_move(
            state=state,
            move=HookAction(
                source_track=source,
                target_track=target,
                vehicle_nos=vehicle_nos,
                path_tracks=[source] if source == target else [source, target],
                action_type=action_type,
            ),
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("staging target ordering must not recompute route pressure per candidate")

    monkeypatch.setattr(move_generator, "compute_route_blockage_plan", fail_compute)

    targets = _candidate_staging_targets(
        source_track="存2",
        block=["5313847", "5243183"],
        state=state,
        plan_input=normalized,
        master=master,
        vehicle_by_no=vehicle_by_no,
        goal_target_hints=("抛",),
        route_oracle=route_oracle,
    )

    assert targets
    assert set(targets) & {"存3", "存1", "洗北", "调北"}


def test_candidate_staging_targets_can_skip_route_pressure_for_feasibility_probe(monkeypatch):
    from fzed_shunting.solver import move_generator
    from fzed_shunting.solver.move_generator import _candidate_staging_targets

    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存2",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存2",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(update={"loco_carry": ("A",)})
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("route pressure projection is not needed for feasibility probes")

    monkeypatch.setattr(move_generator, "compute_route_blockage_plan", fail_compute)

    targets = _candidate_staging_targets(
        source_track="存2",
        block=["A"],
        state=state,
        plan_input=normalized,
        master=master,
        vehicle_by_no=vehicle_by_no,
        goal_target_hints=("修1库内",),
        route_oracle=RouteOracle(master),
        route_pressure_sort=False,
    )

    assert targets


def test_candidate_staging_targets_skips_projection_for_route_irrelevant_block(monkeypatch):
    from fzed_shunting.solver import move_generator
    from fzed_shunting.solver.move_generator import _candidate_staging_targets

    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (
            ROOT_DIR
            / "data"
            / "validation_inputs"
            / "truth"
            / "validation_20260206W.json"
        ).read_text(encoding="utf-8")
    )
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    route_oracle = RouteOracle(master)
    projection_calls = 0
    real_compute = move_generator.compute_route_blockage_plan

    def wrapped_compute(*args, **kwargs):
        nonlocal projection_calls
        candidate_state = args[1]
        if candidate_state is not state:
            projection_calls += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(move_generator, "compute_route_blockage_plan", wrapped_compute)

    targets = _candidate_staging_targets(
        source_track="存5北",
        block=["5487381", "1578911", "5343658"],
        state=state,
        plan_input=normalized,
        master=master,
        vehicle_by_no=vehicle_by_no,
        goal_target_hints=("存5南",),
        route_oracle=route_oracle,
    )

    assert targets
    assert projection_calls == 0


def test_generate_goal_moves_prunes_heavy_equivalent_block_before_staging_without_l1():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MIX_PATH_NEED",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "存5南",
                    "order": str(i),
                    "vehicleModel": "敞车",
                    "vehicleNo": f"MIX_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 4.0,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "重车" if i <= 2 else "",
                }
                for i in range(1, 16)
            ],
            {
                "trackName": "修1库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "MIX_ALT_BLOCK",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5南"]

    assert any(move.target_track == "临1" for move in staging_moves)
    assert {len(move.vehicle_nos) for move in staging_moves} == {14}


def test_generate_goal_moves_falls_back_to_smaller_interfering_prefix_when_largest_block_cannot_stage():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 171.6},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PATH_NEED_FALLBACK",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "存5南",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"FALLBACK_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 13)
            ],
            {
                "trackName": "修1库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "FALLBACK_ALT_BLOCK",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5南"]

    assert any(
        move.target_track == "临4" and len(move.vehicle_nos) == 6
        for move in staging_moves
    )


def test_generate_goal_moves_can_fallback_to_storage_cache_when_no_temporary_track_fits():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "存5北",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"STORAGE_BLOCK_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 12)
            ],
            {
                "trackName": "存5北",
                "order": "12",
                "vehicleModel": "棚车",
                "vehicleNo": "STORAGE_GO",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert any(
        move.target_track == "存2" and len(move.vehicle_nos) == 11
        for move in staging_moves
    )


def test_generate_goal_moves_does_not_add_storage_cache_when_temporary_stage_is_already_feasible():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TEMP_FIRST_BLOCK",
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
                "vehicleNo": "TEMP_FIRST_GO",
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
    state = build_initial_state(normalized)

    moves = generate_goal_moves(normalized, state, master=master)
    staging_moves = [move for move in moves if move.source_track == "存5北"]

    assert {move.target_track for move in staging_moves} == {"临1"}

def test_generate_goal_moves_reuses_injected_route_oracle():
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
                "vehicleNo": "INJECTED_BLOCK",
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
                "vehicleNo": "INJECTED_GO",
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
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)

    with patch("fzed_shunting.solver.move_generator.RouteOracle", side_effect=AssertionError("unexpected constructor call")):
        moves = generate_goal_moves(
            normalized,
            state,
            master=master,
            route_oracle=route_oracle,
        )

    assert any(move.target_track == "临1" for move in moves)


def test_generate_real_hook_moves_defers_satisfied_route_blocker_while_normal_attach_exists():
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
                "vehicleNo": "PATH_NEED_NATIVE",
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
                "vehicleNo": "PATH_BLOCK_NATIVE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修1库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "NATIVE_ALT_BLOCK",
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存5北"
        and move.vehicle_nos == ["PATH_NEED_NATIVE"]
        for move in moves
    )
    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "存5南"
        and move.vehicle_nos == ["PATH_BLOCK_NATIVE"]
        for move in moves
    )


def test_generate_real_hook_moves_keeps_capacity_eviction_attachable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 28.6},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_OK_1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_OK_2",
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
                "vehicleNo": "DEPOT_NEED_NATIVE",
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "机库"
        and move.vehicle_nos == ["DEPOT_OK_1"]
        for move in moves
    )


def test_generate_real_hook_moves_allows_final_goal_detach_over_capacity_warning():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 20},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CAP_NEED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CAP_DONE",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={"存5北": [], "存5南": ["CAP_DONE"]},
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CAP_NEED",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track == "存5南"
        and move.vehicle_nos == ["CAP_NEED"]
        for move in moves
    )


def test_generate_real_hook_moves_allows_final_snapshot_area_detach_over_capacity_warning():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 20},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CAP_AREA_NEED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CAP_AREA_DONE",
                "repairProcess": "段修",
                "vehicleLength": 10.0,
                "targetMode": "SNAPSHOT",
                "targetTrack": "存5南",
                "targetSource": "END_SNAPSHOT",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={"存5北": [], "存5南": ["CAP_AREA_DONE"]},
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CAP_AREA_NEED",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track == "存5南"
        and move.vehicle_nos == ["CAP_AREA_NEED"]
        for move in moves
    )


def test_generate_real_hook_moves_allows_final_random_depot_detach_over_length_capacity_warning():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修3库内", "trackDistance": 20},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_CAP_NEED",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_CAP_DONE",
                "repairProcess": "厂修",
                "vehicleLength": 10.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={"存5北": [], "修3库内": ["DEPOT_CAP_DONE"]},
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={"DEPOT_CAP_DONE": "301"},
        loco_carry=("DEPOT_CAP_NEED",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track == "修3库内"
        and move.vehicle_nos == ["DEPOT_CAP_NEED"]
        for move in moves
    )


def test_generate_real_hook_moves_can_split_carried_random_depot_tail_groups():
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
                "vehicleModel": "长大车",
                "vehicleNo": "LONG_DEPOT",
                "repairProcess": "厂修",
                "vehicleLength": 18.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "SHORT_DEPOT_A",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "SHORT_DEPOT_B",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={
            "存5北": [],
            "修1库内": [],
            "修2库内": [],
            "修3库内": [],
            "修4库内": [],
        },
        loco_track_name="存5北",
        loco_node="L2",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("LONG_DEPOT", "SHORT_DEPOT_A", "SHORT_DEPOT_B"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track in {"修1库内", "修2库内"}
        and move.vehicle_nos == ["SHORT_DEPOT_A", "SHORT_DEPOT_B"]
        for move in moves
    )


def test_generate_real_hook_moves_keeps_chunk_attach_for_large_random_snapshot_prefix():
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

    moves = [
        move
        for move in generate_real_hook_moves(normalized, state, master=master)
        if move.action_type == "ATTACH" and move.source_track == "存5北"
    ]
    block_sizes = {len(move.vehicle_nos) for move in moves}

    assert 16 in block_sizes
    assert any(size < 16 for size in block_sizes)


def test_generate_real_hook_moves_keeps_spot_eviction_attachable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_OCC_NATIVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "SPOT_NEED_NATIVE",
                "repairProcess": "厂修",
                "vehicleLength": 14.3,
                "targetTrack": "修1库内",
                "isSpotting": "101",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修1库内"
        and move.vehicle_nos == ["SPOT_OCC_NATIVE"]
        for move in moves
    )


def test_generate_real_hook_moves_skips_extra_attach_without_detach_group_synergy():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_B",
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
                "vehicleNo": "BLOCK_A",
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
    state = ReplayState(
        track_sequences={"存5北": ["BLOCK_A"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY_A", "CARRY_B"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    attach_moves = [move for move in moves if move.action_type == "ATTACH"]
    assert ("BLOCK_A",) not in [tuple(move.vehicle_nos) for move in attach_moves]


def test_generate_real_hook_moves_keeps_extra_attach_when_it_reduces_detach_groups():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_C",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "CARRY_D",
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
                "vehicleNo": "BLOCK_B",
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
    state = ReplayState(
        track_sequences={"存5北": ["BLOCK_B"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CARRY_C", "CARRY_D"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    attach_moves = [move for move in moves if move.action_type == "ATTACH"]
    assert ("BLOCK_B",) in [tuple(move.vehicle_nos) for move in attach_moves]


def test_generate_real_hook_moves_detaches_only_tail_blocks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修3库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "HEAD1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "HEAD2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修2库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "3",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "4",
                "vehicleModel": "棚车",
                "vehicleNo": "TAIL2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库内",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={},
        loco_track_name="调棚",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("HEAD1", "HEAD2", "TAIL1", "TAIL2"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)
    detach_blocks = {
        tuple(move.vehicle_nos)
        for move in moves
        if move.action_type == "DETACH"
    }

    assert ("TAIL2",) in detach_blocks
    assert ("TAIL1", "TAIL2") in detach_blocks
    assert ("HEAD1",) not in detach_blocks
    assert ("HEAD1", "HEAD2") not in detach_blocks


def test_generate_real_hook_moves_can_set_back_tail_pushers_to_expose_close_door():
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
                "vehicleNo": "CD_INNER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "关门车",
            },
        ]
        + [
            {
                "trackName": "存5北",
                "order": str(index),
                "vehicleModel": "棚车",
                "vehicleNo": vehicle_no,
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for index, vehicle_no in enumerate(["PUSH1", "PUSH2", "PUSH3"], start=2)
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": []},
        loco_track_name="存5北",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("CD_INNER", "PUSH1", "PUSH2", "PUSH3"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.source_track == "存5北"
        and move.target_track == "存5北"
        and tuple(move.vehicle_nos) == ("PUSH1", "PUSH2", "PUSH3")
        for move in moves
    )


def test_generate_real_hook_moves_skips_loaded_attach_when_access_route_exceeds_l1_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修1库内",
                "order": str(index),
                "vehicleModel": "C70",
                "vehicleNo": f"EQC{index:02d}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车" if index <= 2 else "",
            }
            for index in range(1, 15)
        ]
        + [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "EQ_OVER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["EQ_OVER"]},
        loco_track_name="修1库内",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=tuple(f"EQC{index:02d}" for index in range(1, 15)),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("EQ_OVER",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_attach_even_when_combined_carry_has_three_heavy_cars():
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
                "vehicleModel": "C70E",
                "vehicleNo": "HC1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
            {
                "trackName": "修1库内",
                "order": "2",
                "vehicleModel": "C70E",
                "vehicleNo": "HC2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70E",
                "vehicleNo": "HC3",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "重车",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["HC3"]},
        loco_track_name="修1库内",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("HC1", "HC2"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("HC3",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_extra_attach_with_multiple_unweighed_needweigh_when_detach_remains_feasible():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "GQ70",
                "vehicleNo": "WC1",
                "repairProcess": "段修",
                "vehicleLength": 13.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "GQ70",
                "vehicleNo": "WC2",
                "repairProcess": "段修",
                "vehicleLength": 13.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = ReplayState(
        track_sequences={"存5北": ["WC2"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("WC1",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH" and tuple(move.vehicle_nos) == ("WC2",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_extra_attach_when_combined_carry_cannot_detach_from_terminal_branch():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "油", "trackDistance": 124.0},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "临1", "trackDistance": 81.4},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "预修",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"CARRY_LONG_{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 16.5,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 11)
            ],
            *[
                {
                    "trackName": "油",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"OVER_ATTACH_{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 16.5,
                    "targetTrack": "大库",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 4)
            ],
        ],
        "locoTrackName": "预修",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={
            "油": ["OVER_ATTACH_1", "OVER_ATTACH_2", "OVER_ATTACH_3"],
            "预修": [],
            "存4北": [],
            "临1": [],
        },
        loco_track_name="预修",
        loco_node=None,
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=tuple(f"CARRY_LONG_{index}" for index in range(1, 11)),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and tuple(move.vehicle_nos)
        in {
            ("OVER_ATTACH_1",),
            ("OVER_ATTACH_1", "OVER_ATTACH_2"),
            ("OVER_ATTACH_1", "OVER_ATTACH_2", "OVER_ATTACH_3"),
        }
        for move in moves
    )


def test_generate_real_hook_moves_compresses_empty_carry_attach_prefixes_to_detach_group_frontier():
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)
    attach_prefixes = sorted(
        [
            tuple(move.vehicle_nos)
            for move in moves
            if move.action_type == "ATTACH" and move.source_track == "存5北"
        ],
        key=len,
    )

    assert attach_prefixes == [
        ("P1", "P2"),
        ("P1", "P2", "S1"),
        ("P1", "P2", "S1", "Y1"),
    ]


def test_generate_real_hook_moves_keeps_short_random_depot_vehicle_attachable_from_fallback_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": [
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "SHORT_FALLBACK",
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

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修4库内"
        and tuple(move.vehicle_nos) == ("SHORT_FALLBACK",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_random_depot_rebalance_available_when_pending_vehicle_can_use_preferred_slack():
    master = load_master_data(DATA_DIR)
    preferred_vehicles = [
        {
            "trackName": "修1库内" if index <= 5 else "修2库内",
            "order": str(index if index <= 5 else index - 5),
            "vehicleModel": "C70",
            "vehicleNo": f"PREF_{index}",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "大库",
            "isSpotting": "",
            "vehicleAttributes": "",
        }
        for index in range(1, 10)
    ]
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            *preferred_vehicles,
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "FALLBACK_SETTLED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "PENDING_RANDOM",
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

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修4库内"
        and tuple(move.vehicle_nos) == ("FALLBACK_SETTLED",)
        for move in moves
    )
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "调棚"
        and tuple(move.vehicle_nos) == ("PENDING_RANDOM",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_random_depot_front_blocker_attachable():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "存4北", "trackDistance": 317.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "修4库内",
                "order": "1",
                "vehicleModel": "C70",
                "vehicleNo": "SHORT_FALLBACK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修4库内",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "NEED_C4B",
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
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "修4库内"
        and tuple(move.vehicle_nos) == ("SHORT_FALLBACK",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_wash_north_blocks_access_to_wash_south():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "洗北", "trackDistance": 71.6},
            {"trackName": "洗南", "trackDistance": 90.0},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "洗南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WASH_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "洗北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "WASH_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "洗北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临3",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "洗南"
        and tuple(move.vehicle_nos) == ("WASH_TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_depot_outer_blocks_access_to_inner():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临4", "trackDistance": 81.4},
            {"trackName": "修3库外", "trackDistance": 80.0},
            {"trackName": "修3库内", "trackDistance": 151.7},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "修3库内",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "修3库外",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "DEPOT_BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "修3库外",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临4",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "修3库内"
        and tuple(move.vehicle_nos) == ("DEPOT_TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_uses_clear_alternate_route_when_default_intermediate_is_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "临4", "trackDistance": 81.4},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存4南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCK",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "临4",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存4北"
        and tuple(move.vehicle_nos) == ("TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_access_blocker_attachable_when_all_target_routes_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "机北", "trackDistance": 69.1},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临4",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "机北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "临4"
        and tuple(move.vehicle_nos) == ("ACCESS_TARGET",)
        for move in moves
    )
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "机北"
        and tuple(move.vehicle_nos) == ("ACCESS_BLOCKER",)
        for move in moves
    )


def test_generate_real_hook_moves_skips_attach_when_all_loco_exit_paths_are_blocked():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存5南", "trackDistance": 156},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "机棚", "trackDistance": 105.8},
            {"trackName": "机北", "trackDistance": 69.1},
            {"trackName": "调北", "trackDistance": 70.1},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "调棚",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "PARKED",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": track,
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": f"BLOCK-{idx}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for idx, track in enumerate(["临4", "机棚", "机北", "调北"], start=1)
            ],
        ],
        "locoTrackName": "存5南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "调棚"
        and tuple(move.vehicle_nos) == ("TARGET",)
        for move in moves
    )


def test_generate_real_hook_moves_rejects_detach_entering_empty_target_from_wrong_end():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存5南", "trackDistance": 156.0},
        ],
        "vehicleInfo": [
            _vehicle("MOVE", "存4北", "存5南"),
        ],
        "locoTrackName": "存4北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={"存4北": [], "存5南": []},
        loco_track_name="存4北",
        loco_node=RouteOracle(master).order_end_node("存4北"),
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("MOVE",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "存5南"
        and move.path_tracks == ["存4北", "存4南", "存5南"]
        for move in moves
    )


def test_generate_real_hook_moves_filters_empty_carry_detach_that_maroons_loco():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "预修", "trackDistance": 208.5},
            {"trackName": "抛", "trackDistance": 131.8},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "C1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
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
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "B2",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "预修",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "U1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "抛",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = ReplayState(
        track_sequences={
            "存5南": [],
            "存5北": ["B1"],
            "存1": ["B2"],
            "预修": ["U1"],
        },
        loco_track_name="存5南",
        loco_node=RouteOracle(master).order_end_node("存5南"),
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("C1",),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "存5南"
        and tuple(move.vehicle_nos) == ("C1",)
        for move in moves
    )


def test_collect_real_hook_access_blocker_requests_is_limited_to_blocking_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存3",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "REQ_ALT_BLOCK_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存4北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "REQ_ALT_BLOCK_B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )

    assert requests == {}


def test_collect_real_hook_access_blocker_requests_clears_whole_blocking_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "机北", "trackDistance": 69.1},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临4",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET_A",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "机北",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"ACCESS_BLOCK_{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "机北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 4)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )
    moves = generate_real_hook_moves(normalized, state, master=master)

    assert requests == {"机北": {3}}
    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "机北"
        and tuple(move.vehicle_nos) == ("ACCESS_BLOCK_1",)
        for move in moves
    )
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "机北"
        and tuple(move.vehicle_nos)
        == ("ACCESS_BLOCK_1", "ACCESS_BLOCK_2", "ACCESS_BLOCK_3")
        for move in moves
    )


def test_generate_real_hook_moves_splits_long_access_blocker_by_staging_feasibility():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "机北", "trackDistance": 69.1},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "存5北", "trackDistance": 367.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "临4",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LONG_ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            *[
                {
                    "trackName": "机北",
                    "order": str(index),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"LONG_ACCESS_BLOCK_{index}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "机北",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for index in range(1, 16)
            ],
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    moves = generate_real_hook_moves(normalized, state, master=master)

    attach_sizes = {
        len(move.vehicle_nos)
        for move in moves
        if move.action_type == "ATTACH" and move.source_track == "机北"
    }
    assert attach_sizes
    assert max(attach_sizes) < 15


def test_collect_real_hook_access_blocker_requests_waits_while_normal_attach_exists():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存4南", "trackDistance": 154.5},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "存1", "trackDistance": 113.0},
        ],
        "vehicleInfo": [
            {
                "trackName": "存4南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "READY_SOURCE",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_TARGET",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "ACCESS_BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存4南",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )

    assert requests == {}


def test_access_blocker_requests_include_route_blockage_even_with_normal_attach_available():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
        ],
        "vehicleInfo": [
            _vehicle("READY", "存1", "修4库内"),
            _vehicle("SEEK", "存5南", "修4库内", spotting="405"),
            _vehicle("BLOCK_A", "存5北", "存5北", order=1),
            _vehicle("BLOCK_B", "存5北", "存5北", order=2),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )
    moves = generate_real_hook_moves(normalized, state, master=master)

    assert requests == {"存5北": {2}}
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存5北"
        and tuple(move.vehicle_nos) == ("BLOCK_A", "BLOCK_B")
        for move in moves
    )


def test_generate_real_hook_moves_uses_route_blockage_facts_for_access_blockers():
    master = load_master_data(DATA_DIR)
    scenario = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "validation_inputs"
        / "truth"
        / "validation_20260327Z.json"
    )
    normalized = normalize_plan_input(
        json.loads(scenario.read_text(encoding="utf-8")),
        master,
    )
    state = build_initial_state(normalized)
    route_oracle = RouteOracle(master)

    route_blockage_plan = compute_route_blockage_plan(normalized, state, route_oracle)
    moves = generate_real_hook_moves(
        normalized,
        state,
        master=master,
        route_oracle=route_oracle,
    )

    assert route_blockage_plan.facts_by_blocking_track["存2"].blocked_vehicle_nos
    assert any(
        move.action_type == "ATTACH"
        and move.source_track == "存2"
        and "5310825" in move.vehicle_nos
        for move in moves
    )


def test_generate_real_hook_moves_includes_storage_parking_for_satisfied_route_blocker_carry():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存2", "trackDistance": 239.2},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "修4库内", "trackDistance": 151.7},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
        ],
        "vehicleInfo": [
            _vehicle("SEEK", "存5南", "修4库内", spotting="405"),
            _vehicle("BLOCK_A", "存5北", "存5北", order=1),
            _vehicle("BLOCK_B", "存5北", "存5北", order=2),
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    route_oracle = RouteOracle(master)
    state = ReplayState(
        track_sequences={
            "存5北": [],
            "存5南": ["SEEK"],
            "修4库内": [],
            "临1": [],
            "临2": [],
            "存2": [],
        },
        loco_track_name="存5北",
        loco_node=route_oracle.order_end_node("存5北"),
        weighed_vehicle_nos=set(),
        spot_assignments={},
        loco_carry=("BLOCK_A", "BLOCK_B"),
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.target_track == "存2"
        and tuple(move.vehicle_nos) == ("BLOCK_A", "BLOCK_B")
        for move in moves
    )


def test_access_blocker_requests_ignore_goal_corridor_blockage_with_normal_attach_available():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "存1", "trackDistance": 113.0},
            {"trackName": "存5南", "trackDistance": 156.0},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临4", "trackDistance": 90.1},
        ],
        "vehicleInfo": [
            _vehicle("READY", "存1", "修4库内"),
            _vehicle("SEEK", "临4", "存5北"),
            _vehicle("BLOCK", "存5南", "存5南"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master)
    state = build_initial_state(normalized)
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    goal_by_vehicle = {vehicle.vehicle_no: vehicle.goal for vehicle in normalized.vehicles}

    requests = _collect_real_hook_access_blocker_attach_requests(
        plan_input=normalized,
        state=state,
        goal_by_vehicle=goal_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        route_oracle=RouteOracle(master),
    )

    assert requests == {}


def test_generate_real_hook_moves_skips_loaded_attach_when_access_length_exceeds_l1_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            *[
                {
                    "trackName": "存5北",
                    "order": str(i),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"LOAD_A_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.0,
                    "targetTrack": "存2",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
                for i in range(1, 16)
            ],
            {
                "trackName": "存2",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "LOAD_B",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {
                "存5北": [],
                "存2": ["LOAD_B"],
            },
            "loco_track_name": "存5北",
            "loco_carry": tuple(f"LOAD_A_{i}" for i in range(1, 16)),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "ATTACH"
        and move.source_track == "存2"
        and tuple(move.vehicle_nos) == ("LOAD_B",)
        for move in moves
    )


def test_generate_real_hook_moves_keeps_full_same_track_detach_when_it_can_finish_carry():
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
                "vehicleNo": "SAME_FULL_A",
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
                "vehicleNo": "SAME_FULL_B",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存5北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": ("SAME_FULL_A", "SAME_FULL_B"),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert any(
        move.action_type == "DETACH"
        and move.source_track == "存5北"
        and move.target_track == "存5北"
        and tuple(move.vehicle_nos) == ("SAME_FULL_A", "SAME_FULL_B")
        for move in moves
    )


def test_generate_real_hook_moves_rejects_same_track_detach_when_it_only_enables_regrab():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            *[
                _vehicle(f"RETURN_{index:02d}", "存5北", "存5北", order=index)
                for index in range(1, 21)
            ],
            _vehicle("BURIED_TARGET", "存5北", "存2", order=21),
        ],
        "locoTrackName": "存5北",
    }
    return_block = tuple(f"RETURN_{index:02d}" for index in range(1, 21))
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {
                "存5北": ["BURIED_TARGET"],
                "存2": [],
            },
            "loco_track_name": "存5北",
            "loco_carry": return_block,
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.source_track == "存5北"
        and move.target_track == "存5北"
        and tuple(move.vehicle_nos) == return_block
        for move in moves
    )


def test_generate_real_hook_moves_skips_detach_when_full_carry_exceeds_route_limit():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存2", "trackDistance": 239.2},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": str(i),
                "vehicleModel": "棚车",
                "vehicleNo": f"DETACH_FULL_{i}",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存2",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
            for i in range(1, 16)
        ],
        "locoTrackName": "存5北",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized).model_copy(
        update={
            "track_sequences": {"存5北": []},
            "loco_track_name": "存5北",
            "loco_carry": tuple(f"DETACH_FULL_{i}" for i in range(1, 16)),
        }
    )

    moves = generate_real_hook_moves(normalized, state, master=master)

    assert not any(
        move.action_type == "DETACH"
        and move.target_track == "存2"
        and tuple(move.vehicle_nos) == ("DETACH_FULL_1",)
        for move in moves
    )
