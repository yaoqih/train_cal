from fzed_shunting.domain.work_positions import (
    allowed_spotting_south_ranks,
    north_rank,
    preview_work_positions_after_prepend,
    south_rank,
    work_slot_violations_by_vehicle,
    work_position_satisfied,
)
from fzed_shunting.io.normalize_input import GoalSpec, NormalizedVehicle
from fzed_shunting.verify.replay import ReplayState


def _vehicle(
    vehicle_no: str,
    *,
    target_track: str,
    work_position_kind: str,
    target_rank: int | None = None,
) -> NormalizedVehicle:
    return NormalizedVehicle(
        current_track="存5北",
        order=1,
        vehicle_model="棚车",
        vehicle_no=vehicle_no,
        repair_process="段修",
        vehicle_length=14.3,
        goal=GoalSpec(
            target_mode="WORK_POSITION",
            target_track=target_track,
            allowed_target_tracks=[target_track],
            work_position_kind=work_position_kind,
            target_rank=target_rank,
        ),
    )


def test_rank_helpers_read_north_to_south_sequences():
    seq = ["C", "B", "A"]

    assert north_rank(seq, "C") == 1
    assert north_rank(seq, "B") == 2
    assert north_rank(seq, "A") == 3
    assert south_rank(seq, "A") == 1
    assert south_rank(seq, "B") == 2
    assert south_rank(seq, "C") == 3
    assert north_rank(seq, "X") is None
    assert south_rank(seq, "X") is None


def test_allowed_spotting_south_ranks_match_work_track_business_windows():
    assert allowed_spotting_south_ranks("抛") == frozenset({1, 2})
    assert allowed_spotting_south_ranks("油") == frozenset({1, 2})
    assert allowed_spotting_south_ranks("洗南") == frozenset({2, 3, 4})
    assert allowed_spotting_south_ranks("调棚") == frozenset({3, 4, 5, 6})


def test_work_position_satisfied_uses_south_rank_for_spotting():
    vehicle = _vehicle("B", target_track="洗南", work_position_kind="SPOTTING")
    state = ReplayState(
        track_sequences={"洗南": ["C", "B", "A"]},
        loco_track_name="机库",
    )

    assert work_position_satisfied(vehicle, track_name="洗南", state=state)

    wrong_state = state.model_copy(update={"track_sequences": {"洗南": ["B"]}})

    assert not work_position_satisfied(vehicle, track_name="洗南", state=wrong_state)


def test_work_position_satisfied_free_only_requires_target_track():
    vehicle = _vehicle("B", target_track="洗南", work_position_kind="FREE")
    state = ReplayState(
        track_sequences={"洗南": ["B"]},
        loco_track_name="机库",
    )

    assert work_position_satisfied(vehicle, track_name="洗南", state=state)
    assert not work_position_satisfied(vehicle, track_name="调棚", state=state)


def test_work_position_satisfied_uses_north_rank_for_exact_position():
    vehicle = _vehicle(
        "B",
        target_track="调棚",
        work_position_kind="EXACT_NORTH_RANK",
        target_rank=2,
    )
    state = ReplayState(
        track_sequences={"调棚": ["A", "B", "C"]},
        loco_track_name="机库",
    )

    assert work_position_satisfied(vehicle, track_name="调棚", state=state)

    wrong_state = state.model_copy(update={"track_sequences": {"调棚": ["B", "C"]}})

    assert not work_position_satisfied(vehicle, track_name="调棚", state=wrong_state)


def test_work_position_satisfied_allows_explicit_slot_without_north_padding():
    vehicle = _vehicle(
        "B",
        target_track="洗南",
        work_position_kind="EXACT_WORK_SLOT",
        target_rank=3,
    )
    state = ReplayState(
        track_sequences={"洗南": ["B"]},
        loco_track_name="机库",
    )

    assert work_position_satisfied(vehicle, track_name="洗南", state=state)

    overfilled_state = state.model_copy(
        update={"track_sequences": {"洗南": ["A", "C", "D", "B"]}}
    )

    assert not work_position_satisfied(vehicle, track_name="洗南", state=overfilled_state)


def test_work_slot_violations_reject_duplicate_explicit_slot():
    first = _vehicle(
        "A",
        target_track="洗南",
        work_position_kind="EXACT_WORK_SLOT",
        target_rank=3,
    )
    second = _vehicle(
        "B",
        target_track="洗南",
        work_position_kind="EXACT_WORK_SLOT",
        target_rank=3,
    )
    state = ReplayState(
        track_sequences={"洗南": ["A", "B"]},
        loco_track_name="机库",
    )

    violations = work_slot_violations_by_vehicle(vehicles=[first, second], state=state)

    assert set(violations) == {"A", "B"}


def test_work_slot_violations_reject_reversed_explicit_slot_order():
    north_slot = _vehicle(
        "NORTH_SLOT",
        target_track="洗南",
        work_position_kind="EXACT_WORK_SLOT",
        target_rank=1,
    )
    south_slot = _vehicle(
        "SOUTH_SLOT",
        target_track="洗南",
        work_position_kind="EXACT_WORK_SLOT",
        target_rank=3,
    )
    state = ReplayState(
        track_sequences={"洗南": ["SOUTH_SLOT", "NORTH_SLOT"]},
        loco_track_name="机库",
    )

    violations = work_slot_violations_by_vehicle(
        vehicles=[north_slot, south_slot],
        state=state,
    )

    assert set(violations) == {"NORTH_SLOT", "SOUTH_SLOT"}


def test_preview_allows_exact_rank_before_target_and_rejects_past_target():
    vehicle = _vehicle(
        "B",
        target_track="调棚",
        work_position_kind="EXACT_NORTH_RANK",
        target_rank=2,
    )

    before_target = preview_work_positions_after_prepend(
        target_track="调棚",
        incoming_vehicle_nos=["B"],
        existing_vehicle_nos=[],
        vehicle_by_no={"B": vehicle},
    )
    assert before_target.valid
    assert before_target.evaluations["B"].north_rank == 1
    assert before_target.evaluations["B"].rank_gap == 1
    assert not before_target.evaluations["B"].satisfied_now

    past_target = preview_work_positions_after_prepend(
        target_track="调棚",
        incoming_vehicle_nos=["A", "C", "B"],
        existing_vehicle_nos=[],
        vehicle_by_no={"B": vehicle},
    )
    assert not past_target.valid
    assert past_target.hard_violations


def test_preview_rejects_existing_exact_rank_vehicle_pushed_past_target():
    vehicle = _vehicle(
        "B",
        target_track="调棚",
        work_position_kind="EXACT_NORTH_RANK",
        target_rank=2,
    )

    preview = preview_work_positions_after_prepend(
        target_track="调棚",
        incoming_vehicle_nos=["A"],
        existing_vehicle_nos=["C", "B"],
        vehicle_by_no={"B": vehicle},
    )

    assert not preview.valid
    assert preview.evaluations["B"].north_rank == 3
    assert preview.evaluations["B"].rank_gap == -1


def test_preview_uses_complete_new_sequence_for_spotting_block_with_pad():
    target = _vehicle("B", target_track="洗南", work_position_kind="SPOTTING")
    pad = _vehicle("A", target_track="洗南", work_position_kind="FREE")

    single_target = preview_work_positions_after_prepend(
        target_track="洗南",
        incoming_vehicle_nos=["B"],
        existing_vehicle_nos=[],
        vehicle_by_no={"B": target},
    )
    assert not single_target.valid

    with_south_pad = preview_work_positions_after_prepend(
        target_track="洗南",
        incoming_vehicle_nos=["B", "A"],
        existing_vehicle_nos=[],
        vehicle_by_no={"B": target, "A": pad},
    )
    assert with_south_pad.valid
    assert with_south_pad.evaluations["B"].south_rank == 2
    assert with_south_pad.evaluations["B"].satisfied_now
