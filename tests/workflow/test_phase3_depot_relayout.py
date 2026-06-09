from __future__ import annotations

from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
from fzed_shunting.workflow.phase3_depot_relayout import (
    score_phase3_depot_execution_fitness,
    search_phase3_depot_relayout,
)


def _vehicle(vehicle_no: str, *, length: float = 14.3, repair_process: str = "段修") -> NormalizedVehicle:
    return NormalizedVehicle(
        current_track="存4北",
        order=1,
        vehicle_model="棚车",
        vehicle_no=vehicle_no,
        repair_process=repair_process,
        vehicle_length=length,
        goal=GoalSpec(
            target_mode="AREA",
            target_track="修1",
            allowed_target_tracks=["修1", "修2", "修3", "修4"],
            preferred_target_tracks=["修1", "修2"] if length < 17.6 else ["修3", "修4"],
            fallback_target_tracks=["修3", "修4"] if length < 17.6 else [],
            target_area_code="大库:RANDOM",
        ),
    )


def _goal(vehicle_no: str, *, length: float = 14.3) -> dict:
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": "修1",
        "targetMode": "AREA",
        "targetAreaCode": "大库:RANDOM",
        "allowedTargetTracks": ["修1", "修2", "修3", "修4"],
        "preferredTargetTracks": ["修1", "修2"] if length < 17.6 else ["修3", "修4"],
        "fallbackTargetTracks": ["修3", "修4"] if length < 17.6 else [],
        "isSpotting": "",
    }


def _current(vehicle_no: str, *, length: float = 14.3, repair_process: str = "段修") -> dict:
    return {
        "vehicleNo": vehicle_no,
        "trackName": "存4北",
        "order": "1",
        "vehicleLength": length,
        "repairProcess": repair_process,
    }


def test_phase3_depot_relayout_spreads_random_depot_vehicles_across_capacity():
    vehicle_nos = [f"V{i}" for i in range(8)]
    plan = search_phase3_depot_relayout(
        goals=[_goal(vehicle_no) for vehicle_no in vehicle_nos],
        current_by_vehicle={vehicle_no: _current(vehicle_no) for vehicle_no in vehicle_nos},
        base_track_sequences={track: [] for track in ("修1", "修2", "修3", "修4")},
        base_spot_assignments={},
        vehicle_by_no={vehicle_no: _vehicle(vehicle_no) for vehicle_no in vehicle_nos},
        yard_mode="NORMAL",
        reserved_spot_codes=frozenset(),
    )

    assert plan.feasible is True
    assert set(plan.resolved_track_by_vehicle) == set(vehicle_nos)
    assert max(len(sequence) for sequence in plan.track_sequences.values()) <= 5


def test_phase3_depot_relayout_keeps_long_vehicle_off_short_depot_tracks():
    plan = search_phase3_depot_relayout(
        goals=[_goal("LONG", length=17.6)],
        current_by_vehicle={"LONG": _current("LONG", length=17.6)},
        base_track_sequences={track: [] for track in ("修1", "修2", "修3", "修4")},
        base_spot_assignments={},
        vehicle_by_no={"LONG": _vehicle("LONG", length=17.6)},
        yard_mode="NORMAL",
        reserved_spot_codes=frozenset(),
    )

    assert plan.feasible is True
    assert plan.resolved_track_by_vehicle["LONG"] in {"修3", "修4"}


def test_phase3_depot_relayout_reports_infeasible_when_capacity_is_full():
    vehicle_nos = [f"V{i}" for i in range(1)]
    base_sequences = {
        track: [f"{track}_{slot}" for slot in range(5)]
        for track in ("修1", "修2", "修3", "修4")
    }
    vehicle_by_no = {
        vehicle_no: _vehicle(vehicle_no)
        for vehicle_no in vehicle_nos
    }
    for sequence in base_sequences.values():
        for vehicle_no in sequence:
            vehicle_by_no[vehicle_no] = _vehicle(vehicle_no)

    plan = search_phase3_depot_relayout(
        goals=[_goal(vehicle_no) for vehicle_no in vehicle_nos],
        current_by_vehicle={vehicle_no: _current(vehicle_no) for vehicle_no in vehicle_nos},
        base_track_sequences=base_sequences,
        base_spot_assignments={},
        vehicle_by_no=vehicle_by_no,
        yard_mode="NORMAL",
        reserved_spot_codes=frozenset(),
    )

    assert plan.feasible is False
    assert plan.diagnostics["reason"] == "no_valid_depot_relayout"


def test_phase3_depot_execution_fitness_prefers_less_source_fragmentation():
    current_by_vehicle = {
        "A": {"trackName": "调棚"},
        "B": {"trackName": "调棚"},
        "C": {"trackName": "预修"},
        "D": {"trackName": "预修"},
    }
    vehicle_by_no = {
        vehicle_no: _vehicle(vehicle_no)
        for vehicle_no in ("A", "B", "C", "D")
    }
    compact_score = score_phase3_depot_execution_fitness(
        resolved_track_by_vehicle={
            "A": "修1",
            "B": "修1",
            "C": "修2",
            "D": "修2",
        },
        current_by_vehicle=current_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        track_sequences={
            "修1": ["A", "B"],
            "修2": ["C", "D"],
            "修3": [],
            "修4": [],
        },
    )
    fragmented_score = score_phase3_depot_execution_fitness(
        resolved_track_by_vehicle={
            "A": "修1",
            "B": "修2",
            "C": "修1",
            "D": "修2",
        },
        current_by_vehicle=current_by_vehicle,
        vehicle_by_no=vehicle_by_no,
        track_sequences={
            "修1": ["A", "C"],
            "修2": ["B", "D"],
            "修3": [],
            "修4": [],
        },
    )

    assert compact_score < fragmented_score
