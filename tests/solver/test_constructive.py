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
        current.track_sequences.setdefault(m.target_track, []).extend(m.vehicle_nos)
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
