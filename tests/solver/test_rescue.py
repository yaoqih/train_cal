"""Unit tests for goal-by-goal rescue.

Note: _misplaced_vehicles is a pure helper testable without master data.
attempt_goal_rescue requires master data because the sub-A* depends on
RouteOracle for move generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import (
    GoalSpec,
    NormalizedPlanInput,
    NormalizedTrackInfo,
    NormalizedVehicle,
    normalize_plan_input,
)
from fzed_shunting.solver.rescue import _misplaced_vehicles, attempt_goal_rescue
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(
    vno: str, current: str, targets: list[str], length: float = 12.0
) -> NormalizedVehicle:
    return NormalizedVehicle(
        current_track=current,
        order=1,
        vehicle_model="敞车",
        vehicle_no=vno,
        repair_process="段修",
        vehicle_length=length,
        goal=GoalSpec(
            target_mode="TRACK",
            target_track=targets[0],
            allowed_target_tracks=targets,
        ),
    )


def _plan_input(vehicles: list[NormalizedVehicle]) -> NormalizedPlanInput:
    track_names = {v.current_track for v in vehicles} | {
        t for v in vehicles for t in v.goal.allowed_target_tracks
    }
    return NormalizedPlanInput(
        track_info=[
            NormalizedTrackInfo(track_name=t, track_distance=200.0)
            for t in track_names
        ],
        vehicles=vehicles,
        loco_track_name="机库",
        yard_mode="NORMAL",
    )


class TestMisplacedVehicles:
    def test_no_misplaced_when_all_at_target(self) -> None:
        pi = _plan_input([_vehicle("V1", "存4北", ["存4北"])])
        state = ReplayState(
            track_sequences={"存4北": ["V1"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        assert _misplaced_vehicles(pi, state) == []

    def test_detects_misplaced(self) -> None:
        pi = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存1", ["存1"]),  # already at goal
        ])
        state = ReplayState(
            track_sequences={"存5北": ["V1"], "存1": ["V2"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        assert _misplaced_vehicles(pi, state) == ["V1"]

    def test_detects_multiple_misplaced(self) -> None:
        pi = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存3", ["预修"]),
            _vehicle("V3", "存1", ["存1"]),  # at goal
        ])
        state = ReplayState(
            track_sequences={"存5北": ["V1"], "存3": ["V2"], "存1": ["V3"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        assert _misplaced_vehicles(pi, state) == ["V1", "V2"]

    def test_allowed_target_tracks_list_all_accepted(self) -> None:
        # Goal specifies multiple allowed tracks; vehicle is on any of them.
        pi = _plan_input([_vehicle("V1", "修2库内", ["修1库内", "修2库内", "修3库内"])])
        state = ReplayState(
            track_sequences={"修2库内": ["V1"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        assert _misplaced_vehicles(pi, state) == []


class TestAttemptGoalRescueNoop:
    def test_noop_when_all_at_goal(self) -> None:
        # Even without master, this should short-circuit via _misplaced_vehicles.
        pi = _plan_input([_vehicle("V1", "存4北", ["存4北"])])
        state = ReplayState(
            track_sequences={"存4北": ["V1"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        )
        extended, new_state, still = attempt_goal_rescue(
            plan_input=pi,
            current_plan=[],
            terminal_state=state,
            initial_state=state,
            master=None,
            per_vehicle_budget_ms=500.0,
        )
        assert extended == []
        assert still == []
        assert new_state is state


class TestAttemptGoalRescueWithMaster:
    """End-to-end rescue tests requiring real master data for RouteOracle."""

    def _payload(self, vehicles: list[dict]) -> dict:
        return {
            "trackInfo": [
                {"trackName": "存5北", "trackDistance": 367},
                {"trackName": "存4北", "trackDistance": 317.8},
                {"trackName": "存1", "trackDistance": 113},
                {"trackName": "存3", "trackDistance": 155},
                {"trackName": "机库", "trackDistance": 71.6},
            ],
            "vehicleInfo": vehicles,
            "locoTrackName": "机库",
        }

    def test_rescues_single_displaced_vehicle(self) -> None:
        """A vehicle ended up off-target; rescue should append a single hook."""
        master = load_master_data(DATA_DIR)
        payload = self._payload([
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "V1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ])
        pi = normalize_plan_input(payload, master)
        # Simulate the main solver left V1 at its source track (misplaced).
        state = build_initial_state(pi)

        extended, new_state, still = attempt_goal_rescue(
            plan_input=pi,
            current_plan=[],
            terminal_state=state,
            initial_state=state,
            master=master,
            per_vehicle_budget_ms=5000.0,
        )

        assert still == [], f"Expected no unrescued vehicles, got {still}"
        assert len(extended) >= 1
        # V1 should now be at its goal.
        assert "V1" in new_state.track_sequences.get("存4北", [])


def test_rescue_module_exports() -> None:
    """Smoke test that the module's public API is importable."""
    from fzed_shunting.solver import rescue

    assert hasattr(rescue, "attempt_goal_rescue")
    assert hasattr(rescue, "_misplaced_vehicles")
