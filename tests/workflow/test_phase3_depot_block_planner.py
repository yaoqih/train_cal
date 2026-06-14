from __future__ import annotations

from fzed_shunting.workflow.phase3_depot_block_planner import build_phase3_depot_block_plan


def _goal(vehicle_no: str, target_track: str) -> dict:
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": target_track,
        "targetMode": "TRACK",
        "isSpotting": "",
    }


def _current(vehicle_no: str, track_name: str, order: int) -> dict:
    return {
        "vehicleNo": vehicle_no,
        "trackName": track_name,
        "order": str(order),
    }


def _current_full(vehicle_no: str, track_name: str, order: int, repair_process: str = "段修") -> dict:
    row = _current(vehicle_no, track_name, order)
    row.update(
        {
            "repairProcess": repair_process,
            "vehicleLength": 14.3,
        }
    )
    return row


def test_phase3_block_plan_groups_contiguous_same_target_source_blocks():
    goals = [
        _goal("A", "修1"),
        _goal("B", "修1"),
        _goal("C", "修2"),
        _goal("D", "修2"),
        _goal("HOLD", "调棚"),
    ]
    current_by_vehicle = {
        "A": _current("A", "调棚", 1),
        "B": _current("B", "调棚", 2),
        "HOLD": _current("HOLD", "调棚", 3),
        "C": _current("C", "预修", 1),
        "D": _current("D", "预修", 2),
    }

    plan = build_phase3_depot_block_plan(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences={
            "调棚": ["A", "B", "HOLD"],
            "预修": ["C", "D"],
        },
        min_active_vehicle_count=1,
        min_source_track_count=1,
    )

    assert plan.diagnostics["enabled"] is True
    assert [
        (block["sourceTrack"], block["targetTrack"], block["vehicleNos"])
        for block in plan.diagnostics["sourceBlocks"]
    ] == [
        ("调棚", "修1", ["A", "B"]),
        ("预修", "修2", ["C", "D"]),
    ]
    assert len(plan.wave_plans) == 2
    assert set(plan.wave_plans[0]["activeGoalsByVehicle"]) == {"A", "B"}


def test_phase3_block_plan_keeps_simple_case_single_stage():
    goals = [_goal("A", "修1"), _goal("B", "修2")]
    current_by_vehicle = {
        "A": _current("A", "调棚", 1),
        "B": _current("B", "调棚", 2),
    }

    plan = build_phase3_depot_block_plan(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences={"调棚": ["A", "B"]},
    )

    assert plan.wave_plans == []
    assert plan.diagnostics["enabled"] is False


def test_phase3_block_plan_splits_when_target_changes_on_same_source_track():
    goals = [_goal("A", "修1"), _goal("B", "修2"), _goal("C", "修2")]
    current_by_vehicle = {
        "A": _current("A", "调棚", 1),
        "B": _current("B", "调棚", 2),
        "C": _current("C", "调棚", 3),
    }

    plan = build_phase3_depot_block_plan(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences={"调棚": ["A", "B", "C"]},
        min_active_vehicle_count=1,
        min_source_track_count=1,
    )

    assert [
        (block["targetTrack"], block["vehicleNos"])
        for block in plan.diagnostics["sourceBlocks"]
    ] == [
        ("修1", ["A"]),
        ("修2", ["B", "C"]),
    ]
    assert len(plan.wave_plans) == 2


def test_phase3_block_plan_stops_at_first_hold_vehicle_on_source_frontier():
    goals = [
        _goal("A", "修1"),
        _goal("HOLD", "调棚"),
        _goal("B", "修2"),
    ]
    current_by_vehicle = {
        "A": _current("A", "调棚", 1),
        "HOLD": _current("HOLD", "调棚", 2),
        "B": _current("B", "调棚", 3),
    }

    plan = build_phase3_depot_block_plan(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences={"调棚": ["A", "HOLD", "B"]},
        min_active_vehicle_count=1,
        min_source_track_count=1,
    )

    assert [
        (block["targetTrack"], block["vehicleNos"])
        for block in plan.diagnostics["sourceBlocks"]
    ] == [("修1", ["A"])]
    assert plan.diagnostics["coveredVehicleNos"] == ["A"]
    assert plan.diagnostics["hiddenActiveVehicleNos"] == ["B"]
    assert plan.diagnostics["allActiveCoveredByFrontier"] is False
    assert set(plan.wave_plans[0]["activeGoalsByVehicle"]) == {"A"}


def test_phase3_block_plan_scores_high_risk_for_full_short_depot_tracks_and_many_sources():
    goals = [
        *[_goal(f"A{i}", "修1") for i in range(5)],
        *[_goal(f"B{i}", "修2") for i in range(5)],
        _goal("C0", "修3"),
        _goal("D0", "修4"),
    ]
    current_by_vehicle = {
        **{f"A{i}": _current_full(f"A{i}", "调棚", i) for i in range(5)},
        **{f"B{i}": _current_full(f"B{i}", "预修", i) for i in range(5)},
        "C0": _current_full("C0", "油", 1),
        "D0": _current_full("D0", "洗南", 1),
    }

    plan = build_phase3_depot_block_plan(
        goals=goals,
        current_by_vehicle=current_by_vehicle,
        track_sequences={
            "调棚": [f"A{i}" for i in range(5)],
            "预修": [f"B{i}" for i in range(5)],
            "油": ["C0"],
            "洗南": ["D0"],
        },
    )

    risk = plan.diagnostics["risk"]
    assert risk["score"] >= 6
    assert risk["shouldUseExecutionPlan"] is False
    assert "source_track_count_ge_4" in risk["reasons"]
    assert "short_depot_track_full:修1,修2" in risk["reasons"]


def test_phase3_block_plan_reports_sequence_plan_unavailable_without_normalized_vehicles():
    plan = build_phase3_depot_block_plan(
        goals=[_goal("A", "修1")],
        current_by_vehicle={"A": _current_full("A", "调棚", 1)},
        track_sequences={"调棚": ["A"]},
        min_active_vehicle_count=1,
        min_source_track_count=1,
    )

    assert plan.diagnostics["targetSequencePlan"]["available"] is False
