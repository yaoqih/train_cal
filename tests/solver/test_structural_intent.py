from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.structural_metrics import compute_structural_metrics
from fzed_shunting.solver.structural_intent import build_structural_intent
from fzed_shunting.verify.replay import ReplayState, build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(
    vehicle_no: str,
    source: str,
    target: str,
    *,
    order: int = 1,
    length: float = 14.3,
    spotting: str = "",
) -> dict:
    return {
        "trackName": source,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": length,
        "targetTrack": target,
        "isSpotting": spotting,
        "vehicleAttributes": "",
    }


def _normalize(payload: dict):
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)
    return master, normalized, build_initial_state(normalized)


def test_structural_intent_unifies_order_route_capacity_and_commitment_facts():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存5南", "trackDistance": 156.0},
                {"trackName": "临4", "trackDistance": 90.1},
                {"trackName": "洗南", "trackDistance": 88.7},
                {"trackName": "存1", "trackDistance": 20.0},
                {"trackName": "存4北", "trackDistance": 317.8},
            ],
            "vehicleInfo": [
                _vehicle("ROUTE_SEEK", "临4", "存5北", order=1),
                _vehicle("ROUTE_BLOCK", "存5南", "存5南", order=1),
                _vehicle("CAP_KEEP", "存1", "存1", order=1, length=15.0),
                _vehicle("CAP_RELEASE", "存1", "存4北", order=2, length=12.0),
                _vehicle("CAP_INBOUND", "存5北", "存1", order=1, length=10.0),
                _vehicle("WASH_PAD_1", "洗南", "洗南", order=1),
                _vehicle("WASH_PAD_2", "洗南", "洗南", order=2),
                _vehicle("WASH_PAD_3", "洗南", "洗南", order=3),
                _vehicle("WASH_PAD_4", "洗南", "洗南", order=4),
                _vehicle("WASH_PAD_5", "洗南", "洗南", order=5),
                _vehicle("WASH_SPOT", "存5北", "洗南", order=2, spotting="是"),
            ],
            "locoTrackName": "机库",
        }
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert "洗南" in intent.order_debts_by_track
    wash_debt = intent.order_debts_by_track["洗南"]
    assert wash_debt.defect_count > 0
    assert wash_debt.pending_vehicle_nos == ("WASH_SPOT",)
    assert wash_debt.blocking_prefix_vehicle_nos[:2] == ("WASH_PAD_1", "WASH_PAD_2")

    resource_debts = {(debt.kind, debt.track_name) for debt in intent.resource_debts}
    assert ("ROUTE_RELEASE", "存5南") in resource_debts
    assert ("CAPACITY_RELEASE", "存1") in resource_debts
    assert ("FRONT_CLEARANCE", "存5北") in resource_debts

    capacity_debt = next(
        debt
        for debt in intent.resource_debts
        if debt.kind == "CAPACITY_RELEASE" and debt.track_name == "存1"
    )
    assert capacity_debt.vehicle_nos == ("CAP_KEEP", "CAP_RELEASE")

    assert any(
        block.track_name == "存1" and block.vehicle_nos == ("CAP_KEEP",)
        for blocks in intent.committed_blocks_by_track.values()
        for block in blocks
    )

    assert any(buffer.role_scores["ORDER_BUFFER"] > 0 for buffer in intent.staging_buffers)


def test_structural_intent_front_clearance_debt_covers_dispatchable_prefix():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存5南", "trackDistance": 156.0},
                {"trackName": "轮", "trackDistance": 47.8},
                {"trackName": "存3", "trackDistance": 156.0},
            ],
            "vehicleInfo": [
                _vehicle("TO_STORE_A", "存5北", "存5南", order=1),
                _vehicle("TO_WHEEL_A", "存5北", "轮", order=2),
                _vehicle("TO_WHEEL_B", "存5北", "轮", order=3),
                _vehicle("TO_STORE_B", "存5北", "存5南", order=4),
                _vehicle("TO_CUN3", "存5北", "存3", order=5),
            ],
            "locoTrackName": "机库",
        }
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    front_debt = next(
        debt
        for debt in intent.resource_debts
        if debt.kind == "FRONT_CLEARANCE" and debt.track_name == "存5北"
    )
    assert front_debt.vehicle_nos == (
        "TO_STORE_A",
        "TO_WHEEL_A",
        "TO_WHEEL_B",
        "TO_STORE_B",
        "TO_CUN3",
    )


def test_structural_intent_marks_exact_spot_occupant_as_resource_debt():
    master, normalized, _state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "临1", "trackDistance": 81.4},
                {"trackName": "修1库内", "trackDistance": 151.7},
                {"trackName": "修2库内", "trackDistance": 151.7},
                {"trackName": "修3库内", "trackDistance": 151.7},
                {"trackName": "修4库内", "trackDistance": 151.7},
            ],
            "vehicleInfo": [
                _vehicle("DEPOT106", "修1库内", "大库", order=1),
                _vehicle("DEPOT107", "修1库内", "大库", order=2),
                {
                    **_vehicle("SPOT106", "临1", "修1库内", order=1, spotting="迎检"),
                    "targetMode": "SPOT",
                    "targetSpotCode": "106",
                },
            ],
            "locoTrackName": "机库",
        }
    )
    state = ReplayState(
        track_sequences={"修1库内": ["DEPOT106", "DEPOT107"], "临1": ["SPOT106"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={"DEPOT106": "106", "DEPOT107": "107"},
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert any(
        debt.kind == "EXACT_SPOT_RELEASE"
        and debt.track_name == "修1库内"
        and debt.vehicle_nos == ("DEPOT106",)
        and debt.blocked_vehicle_nos == ("SPOT106",)
        for debt in intent.resource_debts
    )


def test_structural_intent_delays_work_position_vehicle_that_would_break_unfinished_window():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "调棚", "trackDistance": 174.3},
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
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert any(
        delayed.vehicle_no == "FREE_FIRST"
        and delayed.target_track == "调棚"
        and delayed.reason == "would_precede_unfinished_work_position_window"
        for delayed in intent.delayed_commitments
    )
    assert any(
        lease.role == "ORDER_BUFFER"
        and lease.vehicle_nos == ("FREE_FIRST",)
        and lease.source_track == "存5北"
        and lease.target_track == "调棚"
        for lease in intent.buffer_leases
    )


def test_structural_intent_delays_spotting_vehicle_that_is_not_ready_for_window():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "调棚", "trackDistance": 174.3},
            ],
            "vehicleInfo": [
                _vehicle("SPOT_TOO_EARLY", "存5北", "调棚", order=1, spotting="是"),
                _vehicle("PAD1", "调棚", "调棚", order=1),
            ],
            "locoTrackName": "机库",
        }
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert any(
        delayed.vehicle_no == "SPOT_TOO_EARLY"
        and delayed.target_track == "调棚"
        and delayed.reason == "work_position_window_not_ready"
        for delayed in intent.delayed_commitments
    )


def test_structural_intent_does_not_delay_free_fill_when_spotting_window_is_already_stable():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "调棚", "trackDistance": 174.3},
                {"trackName": "存4南", "trackDistance": 154.5},
                {"trackName": "存5北", "trackDistance": 367.0},
            ],
            "vehicleInfo": [
                _vehicle("SPOT1", "调棚", "调棚", order=1, spotting="是"),
                _vehicle("SPOT2", "调棚", "调棚", order=2, spotting="是"),
                _vehicle("PAD1", "调棚", "调棚", order=3),
                _vehicle("PAD2", "调棚", "调棚", order=4),
                _vehicle("PAD3", "调棚", "调棚", order=5),
                _vehicle("FREE_A", "存4南", "调棚", order=1),
                _vehicle("FREE_B", "存5北", "调棚", order=2),
            ],
            "locoTrackName": "机库",
        }
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert not any(
        delayed.vehicle_no in {"FREE_A", "FREE_B"}
        and delayed.target_track == "调棚"
        for delayed in intent.delayed_commitments
    )


def test_structural_intent_delays_free_vehicle_that_would_precede_exact_work_slot():
    master, normalized, state = _normalize(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "调棚", "trackDistance": 174.3},
            ],
            "vehicleInfo": [
                _vehicle("FREE_FIRST", "存5北", "调棚", order=1),
                {
                    **_vehicle("EXACT_SLOT", "调棚", "调棚", order=1),
                    "targetMode": "SPOT",
                    "targetSpotCode": "1",
                },
                _vehicle("PAD1", "调棚", "调棚", order=2),
            ],
            "locoTrackName": "机库",
        }
    )

    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert any(
        delayed.vehicle_no == "FREE_FIRST"
        and delayed.target_track == "调棚"
        and delayed.reason == "work_position_window_not_ready"
        for delayed in intent.delayed_commitments
    )


def test_structural_intent_includes_work_position_tracks_even_when_sequence_defect_is_zero():
    master, normalized, state = _normalize(
        {
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
    )

    metrics = compute_structural_metrics(normalized, state)
    intent = build_structural_intent(
        normalized,
        state,
        route_oracle=RouteOracle(master),
    )

    assert metrics.work_position_unfinished_count == 1
    assert metrics.target_sequence_defect_count == 0
    assert "调棚" in intent.order_debts_by_track
    debt = intent.order_debts_by_track["调棚"]
    assert debt.defect_count == 0
    assert debt.pending_vehicle_nos == ("FREE_FIRST",)
    assert debt.blocking_prefix_vehicle_nos == ()
