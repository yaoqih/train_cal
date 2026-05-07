import json
from pathlib import Path
from collections import Counter
from unittest.mock import patch

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.work_positions import allowed_spotting_south_ranks, south_rank
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver import move_generator
from fzed_shunting.solver import move_candidates
from fzed_shunting.solver.goal_logic import goal_effective_allowed_tracks, goal_is_satisfied
from fzed_shunting.solver.move_generator import (
    _collect_real_hook_access_blocker_attach_requests,
    _collect_real_hook_identity_attach_requests,
    _candidate_staging_targets,
    generate_goal_moves,
    generate_real_hook_moves,
)
from fzed_shunting.solver.move_candidates import generate_move_candidates
from fzed_shunting.solver.purity import STAGING_TRACKS
from fzed_shunting.solver.route_blockage import compute_route_blockage_plan
from fzed_shunting.solver.state import _state_key
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
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


def test_work_position_sequence_candidate_repairs_spotting_order_debt():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "存3", "trackDistance": 258.5},
            {"trackName": "预修", "trackDistance": 208.5},
        ],
        "vehicleInfo": [
            _vehicle("R1", "调棚", "修1", order=1),
            _vehicle("R2", "调棚", "修1", order=2),
            _vehicle("R3", "调棚", "修2", order=3),
            _vehicle("KEEP1", "调棚", "调棚", order=4),
            _vehicle("KEEP2", "调棚", "调棚", order=5),
            _vehicle("KEEP3", "调棚", "调棚", order=6),
            _vehicle("KEEP4", "调棚", "调棚", order=7),
            _vehicle("SRC1", "存5北", "存3", order=1),
            _vehicle("SRC2", "存5北", "预修", order=2),
            _vehicle("ORD1", "存5北", "调棚", order=3),
            _vehicle("ORD2", "存5北", "调棚", order=4),
            _vehicle("SPOT1", "存5北", "调棚", order=5, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    candidates = generate_move_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    sequence_candidates = [
        candidate
        for candidate in candidates
        if candidate.kind == "work_position_sequence"
    ]
    assert sequence_candidates
    assert any(len(candidate.steps) >= 6 for candidate in sequence_candidates)

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    best = sequence_candidates[0]
    next_state = state
    for step in best.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert before.target_sequence_defect_count > 0
    assert after.target_sequence_defect_count < before.target_sequence_defect_count
    assert after.work_position_unfinished_count < before.work_position_unfinished_count
    assert south_rank(next_state.track_sequences["调棚"], "SPOT1") in allowed_spotting_south_ranks("调棚")
    assert next_state.track_sequences["存3"] == ["SRC1"]
    assert next_state.track_sequences["预修"] == ["SRC2"]
    assert next_state.track_sequences["修1库内"] == ["R1", "R2"]
    assert not next_state.loco_carry


def test_work_position_sequence_candidates_skip_structural_metrics_without_spotting_source(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("A1", "调棚", "调棚", order=1),
            _vehicle("A2", "存5北", "调棚", order=1),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("structural metrics should be skipped without SPOTTING source debt")

    monkeypatch.setattr(move_candidates, "compute_structural_metrics", fail_if_called)

    candidates = move_candidates.generate_work_position_sequence_candidates(
        normalized,
        state,
        master=master,
        route_oracle=RouteOracle(master),
    )

    assert candidates == []


def test_work_position_sequence_candidate_prefers_contiguous_spotting_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临4", "trackDistance": 90.1},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "修1库内", "trackDistance": 151.7},
            {"trackName": "修2库内", "trackDistance": 151.7},
            {"trackName": "存3", "trackDistance": 258.5},
            {"trackName": "预修", "trackDistance": 208.5},
        ],
        "vehicleInfo": [
            _vehicle("R1", "调棚", "修1", order=1),
            _vehicle("R2", "调棚", "修1", order=2),
            _vehicle("R3", "调棚", "修2", order=3),
            _vehicle("R4", "调棚", "修2", order=4),
            _vehicle("KEEP1", "调棚", "调棚", order=5),
            _vehicle("KEEP2", "调棚", "调棚", order=6),
            _vehicle("KEEP3", "调棚", "调棚", order=7),
            _vehicle("SRC1", "存5北", "存3", order=1),
            _vehicle("SRC2", "存5北", "预修", order=2),
            _vehicle("ORD1", "存5北", "调棚", order=3),
            _vehicle("ORD2", "存5北", "调棚", order=4),
            _vehicle("SPOT1", "存5北", "调棚", order=5, spotting="是"),
            _vehicle("SPOT2", "存5北", "调棚", order=6, spotting="是"),
            _vehicle("SPOT3", "存5北", "调棚", order=7, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
    ]

    assert candidates
    first_target_detach = next(
        step
        for step in candidates[0].steps
        if step.action_type == "DETACH" and step.target_track == "调棚"
    )
    assert first_target_detach.vehicle_nos == ["SPOT1", "SPOT2", "SPOT3"]

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidates[0].steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert before.target_sequence_defect_count > 0
    assert after.target_sequence_defect_count == 0
    for vehicle_no in ["SPOT1", "SPOT2", "SPOT3"]:
        assert (
            south_rank(next_state.track_sequences["调棚"], vehicle_no)
            in allowed_spotting_south_ranks("调棚")
        )


def test_work_position_sequence_candidate_repairs_internal_rank_window_debt():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("SPOT1", "调棚", "调棚", order=1, spotting="是"),
            _vehicle("FREE1", "调棚", "调棚", order=2),
            _vehicle("FREE2", "调棚", "调棚", order=3),
            _vehicle("FREE3", "调棚", "调棚", order=4),
            _vehicle("FREE4", "调棚", "调棚", order=5),
            _vehicle("FREE5", "调棚", "调棚", order=6),
            _vehicle("FREE6", "调棚", "调棚", order=7),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
    ]

    assert before.target_sequence_defect_by_track == {"调棚": 1}
    assert candidates

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidates[0].steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert after.target_sequence_defect_count == 0
    assert after.work_position_unfinished_count == 0
    assert south_rank(next_state.track_sequences["调棚"], "SPOT1") in allowed_spotting_south_ranks("调棚")
    assert next_state.track_sequences["调棚"].index("FREE1") < next_state.track_sequences["调棚"].index("SPOT1")
    assert not next_state.loco_carry


def test_work_position_sequence_candidate_repairs_rank_window_with_order_buffer():
    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (ROOT_DIR / "data/validation_inputs/online/validation_20260401Z.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    initial_candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
    ]

    assert initial_candidates[0].reason.startswith("油 ")

    oil_candidate = next(
        candidate
        for candidate in initial_candidates
        if candidate.reason.startswith("油 ")
    )
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    after_oil_state = state
    for step in oil_candidate.steps:
        after_oil_state = move_generator._apply_move(
            state=after_oil_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after_oil = compute_structural_metrics(normalized, after_oil_state)
    assert after_oil.target_sequence_defect_by_track.get("油", 0) == 0
    assert after_oil.target_sequence_defect_by_track.get("调棚", 0) > 0

    followup_candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            after_oil_state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
    ]
    shed_candidates = [
        candidate
        for candidate in followup_candidates
        if candidate.reason.startswith("调棚 ")
    ]
    assert shed_candidates

    best = shed_candidates[0]
    assert best.structural_reserve
    assert any(
        step.action_type == "DETACH"
        and step.target_track in STAGING_TRACKS
        and "5327856" in step.vehicle_nos
        for step in best.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["5324224"]
        for step in best.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.target_track in STAGING_TRACKS
        and {"4872364", "5270499", "5331124", "5334574"}.issubset(set(step.vehicle_nos))
        for step in best.steps
    )

    next_state = after_oil_state
    for step in best.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert after.target_sequence_defect_by_track.get("调棚", 0) == 0
    assert after.target_sequence_defect_count < after_oil.target_sequence_defect_count
    assert after.work_position_unfinished_count < after_oil.work_position_unfinished_count
    for vehicle_no in ["3834139", "5324224", "5321345", "5323182"]:
        assert (
            south_rank(next_state.track_sequences["调棚"], vehicle_no)
            in allowed_spotting_south_ranks("调棚")
        )
    assert "5327856" in next_state.track_sequences["调棚"]
    assert set(next_state.track_sequences["临3"]) >= {
        "4872364",
        "5270499",
        "5331124",
        "5334574",
    }
    assert not next_state.loco_carry


def test_work_position_sequence_candidate_releases_source_access_blocker_track():
    master = load_master_data(DATA_DIR)
    payload = json.loads(
        (ROOT_DIR / "data/validation_inputs/positive/case_3_2_shed_work_gondola.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
        and candidate.reason.startswith("调棚 ")
    ]

    assert candidates
    best = candidates[0]
    assert any(
        step.action_type == "ATTACH"
        and step.source_track == "存5北"
        and "1663291" in step.vehicle_nos
        for step in best.steps
    )
    assert any(
        step.action_type == "DETACH"
        and step.target_track == "调棚"
        and step.vehicle_nos == ["T003"]
        for step in best.steps
    )

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in best.steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert not next_state.track_sequences.get("存5北")
    assert goal_is_satisfied(
        vehicle_by_no["T003"],
        track_name="调棚",
        state=next_state,
        plan_input=normalized,
    )
    assert after.target_sequence_defect_count < before.target_sequence_defect_count
    assert after.work_position_unfinished_count < before.work_position_unfinished_count
    assert not next_state.loco_carry


def test_work_position_sequence_candidate_preserves_source_spotting_order_buffer():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "调棚", "trackDistance": 174.3},
            {"trackName": "存5北", "trackDistance": 367.0},
            {"trackName": "临1", "trackDistance": 81.4},
            {"trackName": "临2", "trackDistance": 55.7},
            {"trackName": "临3", "trackDistance": 62.9},
            {"trackName": "机库", "trackDistance": 71.6},
        ],
        "vehicleInfo": [
            _vehicle("KEEP1", "调棚", "调棚", order=1),
            _vehicle("KEEP2", "调棚", "调棚", order=2),
            _vehicle("KEEP3", "调棚", "调棚", order=3),
            _vehicle("KEEP4", "调棚", "调棚", order=4),
            _vehicle("KEEP5", "调棚", "调棚", order=5),
            _vehicle("KEEP6", "调棚", "调棚", order=6),
            _vehicle("SPOT_A", "存5北", "调棚", order=1, spotting="是"),
            _vehicle("MID_FREE", "存5北", "调棚", order=2),
            _vehicle("SPOT_B", "存5北", "调棚", order=3, spotting="是"),
            _vehicle("SPOT_C", "存5北", "调棚", order=4, spotting="是"),
        ],
        "locoTrackName": "机库",
    }
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    state = build_initial_state(normalized)
    before = compute_structural_metrics(normalized, state)

    candidates = [
        candidate
        for candidate in generate_move_candidates(
            normalized,
            state,
            master=master,
            route_oracle=RouteOracle(master),
        )
        if candidate.kind == "work_position_sequence"
    ]

    assert before.target_sequence_defect_by_track.get("调棚", 0) > 0
    assert candidates

    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in normalized.vehicles}
    next_state = state
    for step in candidates[0].steps:
        next_state = move_generator._apply_move(
            state=next_state,
            move=step,
            plan_input=normalized,
            vehicle_by_no=vehicle_by_no,
        )
    after = compute_structural_metrics(normalized, next_state)

    assert after.target_sequence_defect_by_track.get("调棚", 0) == 0
    assert after.work_position_unfinished_count == 0
    for vehicle_no in ["SPOT_A", "SPOT_B", "SPOT_C"]:
        assert (
            south_rank(next_state.track_sequences["调棚"], vehicle_no)
            in allowed_spotting_south_ranks("调棚")
        )
    assert not next_state.loco_carry


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
