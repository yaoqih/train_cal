"""Property-based regression tests for the solver.

Verifies three invariants on randomly generated small yards:
1. SLA: any legal input returns a non-empty plan (within a generous budget).
2. Independence: the returned plan passes the independent verifier.
3. Monotonic anytime: more budget never makes the plan longer.

These are smoke tests — seeded for reproducibility and scoped to small yards
where search terminates within a couple seconds. Run count is kept modest
to keep CI tractable (~15 scenarios).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
SEEDS = list(range(15))  # 15 random scenarios per test — small enough for CI


# Tracks we allow in random yards — all have explicit capacity and are
# reachable by the master-data routing tables.
SAFE_TRACKS = ["存5北", "存4北", "机库", "机北1", "机北2", "洗油北", "存4南"]
# Tracks legal as final destinations (allows_final_destination=True).
FINAL_DEST_TRACKS = {"存5北", "存4北", "机库"}
# Capacity lookup by name (matches master_data).
TRACK_CAPACITY = {
    "存5北": 367.0,
    "存4北": 317.8,
    "存4南": 154.5,
    "机库": 71.6,
    "机北1": 81.4,
    "机北2": 55.7,
    "洗油北": 62.9,
}


def _generate_random_payload(rng: random.Random) -> dict:
    track_names = SAFE_TRACKS.copy()
    rng.shuffle(track_names)
    track_count = rng.randint(3, 5)
    chosen_tracks = track_names[:track_count]
    # Ensure 机库 is always present (loco track).
    if "机库" not in chosen_tracks:
        chosen_tracks[0] = "机库"
    # Ensure at least one final-destination track (besides 机库) is present.
    final_dest_in_yard = [t for t in chosen_tracks if t in FINAL_DEST_TRACKS and t != "机库"]
    if not final_dest_in_yard:
        chosen_tracks[1] = "存4北"

    track_info = [{"trackName": name, "trackDistance": TRACK_CAPACITY[name]} for name in chosen_tracks]
    # Valid vehicle placements: up to capacity of each track. Use conservative
    # 15m per car and cap vehicle count so moves are always feasible.
    vehicle_count = rng.randint(1, 3)
    vehicles = []
    # Non-机库 tracks are candidate current positions.
    candidate_source_tracks = [t for t in chosen_tracks if t != "机库"]
    # Only final-destination tracks are legal as targets.
    candidate_target_tracks = [t for t in chosen_tracks if t in FINAL_DEST_TRACKS and t != "机库"]
    if not candidate_target_tracks:
        candidate_target_tracks = ["存4北"]
    for i in range(vehicle_count):
        src = rng.choice(candidate_source_tracks)
        tgt = rng.choice(candidate_target_tracks)
        vehicles.append(
            {
                "trackName": src,
                "order": str(i + 1),
                "vehicleModel": "棚车",
                "vehicleNo": f"RND{i + 1}",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": tgt,
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        )
    return {
        "trackInfo": track_info,
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def _solve_payload(payload: dict, budget_ms: int):
    master = load_master_data(DATA_DIR)
    try:
        normalized = normalize_plan_input(payload, master)
    except (ValueError, KeyError):
        return None  # Skip inputs the normalizer rejects (expected for some random combos).
    try:
        initial = build_initial_state(normalized)
    except (ValueError, KeyError):
        return None
    try:
        return solve_with_simple_astar_result(
            normalized,
            initial,
            master=master,
            solver_mode="exact",
            time_budget_ms=budget_ms,
            enable_anytime_fallback=True,
            verify=True,
        )
    except ValueError:
        # Hard infeasibility (e.g. capacity overflow at pre-check) — not a solver bug.
        return None


@pytest.mark.parametrize("seed", SEEDS)
def test_property_sla_any_feasible_input_returns_nonempty_plan(seed):
    payload = _generate_random_payload(random.Random(seed))
    result = _solve_payload(payload, budget_ms=10_000)
    if result is None:
        pytest.skip("normalizer/pre-check rejected as infeasible")
    # A trivially-solved input (all vehicles already at goal) legitimately
    # returns plan=[] with is_proven_optimal=True. Only non-trivial inputs
    # must return a non-empty plan.
    if result.is_proven_optimal and not result.plan:
        return
    assert len(result.plan) > 0, f"seed={seed}: solver returned empty plan on non-trivial input"


@pytest.mark.parametrize("seed", SEEDS)
def test_property_returned_plans_pass_independent_verifier(seed):
    payload = _generate_random_payload(random.Random(seed))
    result = _solve_payload(payload, budget_ms=10_000)
    if result is None:
        pytest.skip("normalizer/pre-check rejected as infeasible")
    # Trivially solved (no hooks needed) — verification is not run.
    if not result.plan:
        return
    # Constructive_partial may be best-effort and fail verify — that's the
    # documented contract. All other stages must produce valid plans.
    if result.fallback_stage == "constructive_partial":
        return
    assert result.verification_report is not None, f"seed={seed}: no verifier report"
    assert result.verification_report.is_valid, (
        f"seed={seed}: plan verify failed with stage={result.fallback_stage}, "
        f"errors={result.verification_report.errors[:2]}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_property_anytime_monotonicity(seed):
    """Doubling the budget must not produce a longer plan."""
    payload = _generate_random_payload(random.Random(seed))
    short = _solve_payload(payload, budget_ms=3_000)
    long = _solve_payload(payload, budget_ms=12_000)
    if short is None or long is None:
        pytest.skip("normalizer/pre-check rejected")
    if not short.plan or not long.plan:
        pytest.skip("seed trivially solved or neither run succeeded")
    assert len(long.plan) <= len(short.plan), (
        f"seed={seed}: monotonicity violated "
        f"(3s: {len(short.plan)} hooks, 12s: {len(long.plan)} hooks)"
    )
