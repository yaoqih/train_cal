from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.heuristic import (
    compute_admissible_heuristic,
    compute_heuristic_breakdown,
    make_state_heuristic,
)
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _base_payload(vehicles: list[dict]) -> dict:
    return {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "存4北", "trackDistance": 317.8},
            {"trackName": "机库", "trackDistance": 71.6},
            {"trackName": "洗北", "trackDistance": 100},
            {"trackName": "洗南", "trackDistance": 88.7},
            {"trackName": "存1", "trackDistance": 113},
            {"trackName": "修1库外", "trackDistance": 49.3},
            {"trackName": "修1库内", "trackDistance": 151.7},
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def test_heuristic_zero_when_all_vehicles_at_goal():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "机库",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    assert compute_admissible_heuristic(normalized, initial) == 0


def test_heuristic_counts_misplaced_vehicle():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
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
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_misplaced == 1
    assert breakdown.value >= 1


def test_heuristic_lower_bound_respects_optimal_single_hook():
    from fzed_shunting.solver.astar_solver import solve_with_simple_astar

    master = load_master_data(DATA_DIR)
    payload = _base_payload(
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
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    h0 = compute_admissible_heuristic(normalized, initial)
    plan = solve_with_simple_astar(normalized, initial, master=master)
    assert h0 <= len(plan), f"heuristic {h0} must be <= optimal hooks {len(plan)}"


def test_h_weigh_strengthens_when_weigh_outstanding():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "H1",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_weigh == 1


def test_h_weigh_zero_when_weigh_already_done():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "敞车",
                "vehicleNo": "H1",
                "repairProcess": "段修",
                "vehicleLength": 14.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "称重",
            }
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    state = ReplayState(
        track_sequences=dict(initial.track_sequences),
        loco_track_name=initial.loco_track_name,
        weighed_vehicle_nos={"H1"},
        spot_assignments=dict(initial.spot_assignments),
    )
    breakdown = compute_heuristic_breakdown(normalized, state)
    assert breakdown.h_weigh == 0


def test_h_blocking_strengthens_when_target_has_blocker():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "GOAL1",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存1",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存1",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "BLOCKER",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_blocking >= 1


def test_make_state_heuristic_matches_compute_admissible_heuristic():
    master = load_master_data(DATA_DIR)
    payload = _base_payload(
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
        ]
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    stateful = make_state_heuristic(normalized)
    assert stateful(initial) == compute_admissible_heuristic(normalized, initial)


def _tight_capacity_payload(vehicles: list[dict], track_distances: dict[str, float]) -> dict:
    """Payload with custom track_distances (so we can force a tight declared cap)."""
    defaults = {
        "存5北": 367.0,
        "存5南": 156.0,
        "存4北": 317.8,
        "机库": 71.6,
        "存1": 113.0,
    }
    defaults.update(track_distances)
    return {
        "trackInfo": [
            {"trackName": name, "trackDistance": dist} for name, dist in defaults.items()
        ],
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }


def test_h_tight_capacity_eviction_zero_when_slack_available():
    """Track has plenty of capacity for all incoming — no eviction needed."""
    master = load_master_data(DATA_DIR)
    payload = _tight_capacity_payload(
        [
            # Two 15m cars targeting 存4北 (cap 317.8m). Plenty of slack.
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "A1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "A2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_tight_capacity == 0


def test_h_tight_capacity_eviction_counts_forced_evictions():
    """Track is over-cap initially; pending arrivals force evictions."""
    master = load_master_data(DATA_DIR)
    # 存5南 declared cap 156m. Put two 15m identity-goal cars there (30m),
    # plus a third 15m identity-goal car at 存5北 that needs to come in.
    # Also load 存5南 with non-identity cars to push initial mass ABOVE cap.
    # Initial mass at 存5南: 156 + 30 = 186m (over declared 156).
    # effective_cap = max(156, 186) = 186.
    # current_mass = 186. identity_goal_mass_at_T = 30.
    # demand (identity targets=存5南) = 30 (existing) + 15 (incoming) = 45.
    # pending_arrivals_mass = 45 - 30 = 15.
    # non-identity-mass = 186 - 30 = 156.
    # available_slack = 186 - 186 = 0.
    # free_slack = 0 + 156 = 156. 15 <= 156 → no eviction claimed.
    #
    # Instead: make the non-identity cars ALSO impossible to absorb the
    # overflow. We do this by adjusting so that existing non-identity cars
    # are smaller than pending arrivals AND effective_cap caps growth.
    #
    # Simpler construction:
    # 存5南: only identity-goal cars. Cap = 30m (override track_distance).
    # Initial: 2 identity-goal cars × 15m = 30m. effective_cap = max(30, 30) = 30.
    # Pending identity-goal arrival: 1 × 15m at 存5北.
    # demand = 30 + 15 = 45. current_identity_mass = 30. pending = 15.
    # current_mass = 30. non-identity = 0. available_slack = 0. free_slack = 0.
    # overflow = 15. max_ident_len = 15. evictions = 1. hooks = 2.
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "I1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "I2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={"存5南": 30.0},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_tight_capacity == 2
    # And the combined lower bound picks up the eviction contribution.
    assert breakdown.value >= breakdown.h_distinct_transfer_pairs + 2


def test_h_tight_capacity_eviction_multiple_overflow_ceils():
    """Overflow larger than one identity-goal car → ceil evictions."""
    master = load_master_data(DATA_DIR)
    # 存5南 cap 30m, 2 identity-goal cars (15m each) initially (30m).
    # Two pending identity-goal cars (15m each) elsewhere.
    # demand=60, current_ident=30, pending=30. non-id=0. slack=0.
    # overflow=30. max_ident_len=15. ceil(30/15)=2 → 4 hooks.
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "I1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "I2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "P2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={"存5南": 30.0},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_tight_capacity == 4


def test_h_tight_capacity_eviction_nonidentity_slack_absorbs_overflow():
    """Non-identity cars at T will leave, covering overflow — no eviction."""
    master = load_master_data(DATA_DIR)
    # 存5南 cap 45m. At 存5南: 1 identity-goal car (15m) + 1 non-identity car
    # (15m) targeting 存4北 → total 30m. effective_cap = max(45, 30) = 45.
    # Pending: 1 identity-goal car (15m) at 存5北.
    # demand=30, current_ident=15, pending=15.
    # current_mass=30, slack=15, non-id-mass=15, free_slack=30.
    # overflow = 15 - 30 = -15 → no eviction.
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "I1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "N1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存4北",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={"存5南": 45.0},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_tight_capacity == 0


def test_h_tight_capacity_eviction_ignores_multi_target_vehicles():
    """Multi-target (AREA/RANDOM) vehicles don't trigger single-track eviction."""
    master = load_master_data(DATA_DIR)
    # 大库 RANDOM target → allowed_target_tracks has 4 elements. Even if a
    # depot track is full, the RANDOM car doesn't count against any specific
    # identity-goal demand.
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "R1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "大库",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    assert breakdown.h_tight_capacity == 0


def test_h_tight_capacity_eviction_value_combines_with_transfer_pairs():
    """value property uses h_distinct_transfer_pairs + h_tight_capacity."""
    master = load_master_data(DATA_DIR)
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "I1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "I2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={"存5南": 30.0},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    breakdown = compute_heuristic_breakdown(normalized, initial)
    # h_distinct_transfer_pairs = 1 (one forced pair: 存5北 → 存5南 for P1).
    # h_tight_capacity = 2 (one evict-return roundtrip).
    # value should be at least 3.
    assert breakdown.h_distinct_transfer_pairs == 1
    assert breakdown.h_tight_capacity == 2
    assert breakdown.value >= 3


def test_h_tight_capacity_eviction_state_heuristic_matches_breakdown():
    """make_state_heuristic should include the tight-capacity contribution."""
    master = load_master_data(DATA_DIR)
    payload = _tight_capacity_payload(
        [
            {
                "trackName": "存5南",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "I1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5南",
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": "I2",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "棚车",
                "vehicleNo": "P1",
                "repairProcess": "段修",
                "vehicleLength": 15.0,
                "targetTrack": "存5南",
                "isSpotting": "",
                "vehicleAttributes": "",
            },
        ],
        track_distances={"存5南": 30.0},
    )
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    stateful = make_state_heuristic(normalized)
    assert stateful(initial) == compute_admissible_heuristic(normalized, initial)
