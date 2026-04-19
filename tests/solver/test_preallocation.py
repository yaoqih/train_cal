"""Unit tests for 大库:RANDOM depot preallocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.preallocation import (
    DEPOT_INNER_TRACKS,
    preallocate_random_depot_targets,
)
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _payload_with_random_depot(
    vehicle_count: int,
    preexisting_per_depot: dict[str, int] | None = None,
) -> dict:
    """Build a synthetic payload with ``vehicle_count`` 大库:RANDOM seekers.

    Optionally pre-seeds each 修X库内 with ``preexisting_per_depot[track]`` TRACK-target
    vehicles (so preallocation has to respect capacity).
    """
    preexisting_per_depot = preexisting_per_depot or {}
    track_info = [
        {"trackName": "存5北", "trackDistance": 367},
        {"trackName": "存4北", "trackDistance": 317.8},
        {"trackName": "机库", "trackDistance": 71.6},
        {"trackName": "修1库内", "trackDistance": 151.7},
        {"trackName": "修2库内", "trackDistance": 151.7},
        {"trackName": "修3库内", "trackDistance": 151.7},
        {"trackName": "修4库内", "trackDistance": 151.7},
    ]
    vehicles = []
    next_order = 1
    # Pre-seed existing occupants with TRACK goals (stay put).
    for depot, n in preexisting_per_depot.items():
        for i in range(n):
            vehicles.append(
                {
                    "trackName": depot,
                    "order": str(next_order),
                    "vehicleModel": "棚车",
                    "vehicleNo": f"X_{depot}_{i}",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": depot,
                    "isSpotting": "",
                    "vehicleAttributes": "",
                }
            )
            next_order += 1
    # Now add random-depot seekers at 存5北.
    for i in range(vehicle_count):
        vehicles.append(
            {
                "trackName": "存5北",
                "order": str(next_order),
                "vehicleModel": "棚车",
                "vehicleNo": f"R{i + 1}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "大库",  # triggers 大库:RANDOM
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        )
        next_order += 1
    return {
        "trackInfo": track_info,
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def _run_preallocation(payload: dict):
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    result = preallocate_random_depot_targets(normalized, initial, master=master)
    return normalized, result


def test_preallocation_is_noop_when_no_random_depot_vehicles():
    """Inputs with no 大库:RANDOM vehicles come back unchanged."""
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
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        "locoTrackName": "机库",
    }
    before, after = _run_preallocation(payload)
    assert after.vehicles[0].goal.allowed_target_tracks == before.vehicles[0].goal.allowed_target_tracks


def test_preallocation_narrows_random_depot_to_single_track():
    """Each 大库:RANDOM vehicle ends up with allowed_target_tracks of length 1."""
    before, after = _run_preallocation(_payload_with_random_depot(vehicle_count=10))
    random_depot_vehicles = [v for v in after.vehicles if v.vehicle_no.startswith("R")]
    assert len(random_depot_vehicles) == 10
    for v in random_depot_vehicles:
        assert len(v.goal.allowed_target_tracks) == 1
        assert v.goal.target_track == v.goal.allowed_target_tracks[0]
        assert v.goal.target_track in DEPOT_INNER_TRACKS
        # target_area_code stays as 大库:RANDOM so spot allocator keeps working.
        assert v.goal.target_area_code == "大库:RANDOM"


def test_preallocation_load_balances_across_depots():
    """8 random-depot vehicles with all 4 depots empty → 2 per depot."""
    _, after = _run_preallocation(_payload_with_random_depot(vehicle_count=8))
    counts: dict[str, int] = {}
    for v in after.vehicles:
        if v.vehicle_no.startswith("R"):
            counts[v.goal.target_track] = counts.get(v.goal.target_track, 0) + 1
    assert len(counts) == 4, f"expected spread across all 4 depots, got {counts}"
    assert max(counts.values()) - min(counts.values()) <= 1


def test_preallocation_respects_preexisting_occupants():
    """Depots that already have 8/10 slots taken should receive fewer RANDOM seekers."""
    preexisting = {
        "修1库内": 8,  # Nearly full
        "修2库内": 0,
        "修3库内": 0,
        "修4库内": 0,
    }
    _, after = _run_preallocation(_payload_with_random_depot(vehicle_count=10, preexisting_per_depot=preexisting))
    counts: dict[str, int] = {}
    for v in after.vehicles:
        if v.vehicle_no.startswith("R"):
            counts[v.goal.target_track] = counts.get(v.goal.target_track, 0) + 1
    # 修1库内 should receive fewer seekers than emptier depots.
    assert counts.get("修1库内", 0) <= counts.get("修2库内", 0) + 1


def test_preallocation_skips_small_random_depot_counts():
    """Below the threshold (8), the flexibility of multi-track goals is retained."""
    before, after = _run_preallocation(_payload_with_random_depot(vehicle_count=3))
    r_vehicles = [v for v in after.vehicles if v.vehicle_no.startswith("R")]
    assert len(r_vehicles) == 3
    for v in r_vehicles:
        # 大库 expands to all four depots; preallocation should not narrow here.
        assert len(v.goal.allowed_target_tracks) > 1


def test_preallocation_bails_out_on_over_capacity_input():
    """When the combined 大库 demand exceeds all 4 depots' capacity, no narrowing."""
    # Seed every depot nearly full (each depot capacity ~10 cars = 143m vs 151.7m).
    preexisting = {
        "修1库内": 10,
        "修2库内": 10,
        "修3库内": 10,
        "修4库内": 10,
    }
    before, after = _run_preallocation(_payload_with_random_depot(vehicle_count=10, preexisting_per_depot=preexisting))
    random_vehicles_after = [v for v in after.vehicles if v.vehicle_no.startswith("R")]
    assert len(random_vehicles_after) == 10
    # Bail-out path: allowed_target_tracks remains multi-track.
    assert len(random_vehicles_after[0].goal.allowed_target_tracks) > 1
