from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


WORK_POSITION_TRACKS = frozenset({"调棚", "洗南", "油", "抛"})

SPOTTING_SOUTH_RANKS: dict[str, frozenset[int]] = {
    "抛": frozenset({1, 2}),
    "油": frozenset({1, 2}),
    "洗南": frozenset({2, 3, 4}),
    "调棚": frozenset({3, 4, 5, 6}),
}


class WorkPositionEvaluation(BaseModel):
    vehicle_no: str
    kind: str
    north_rank: int | None = None
    south_rank: int | None = None
    target_rank: int | None = None
    rank_gap: int | None = None
    satisfied_now: bool


class WorkPositionPreview(BaseModel):
    valid: bool
    hard_violations: list[str] = Field(default_factory=list)
    evaluations: dict[str, WorkPositionEvaluation] = Field(default_factory=dict)


def is_work_position_track(track_code: str) -> bool:
    return track_code in WORK_POSITION_TRACKS


def allowed_spotting_south_ranks(track_code: str) -> frozenset[int]:
    return SPOTTING_SOUTH_RANKS.get(track_code, frozenset())


def north_rank(seq: list[str], vehicle_no: str) -> int | None:
    try:
        return seq.index(vehicle_no) + 1
    except ValueError:
        return None


def south_rank(seq: list[str], vehicle_no: str) -> int | None:
    try:
        return len(seq) - seq.index(vehicle_no)
    except ValueError:
        return None


def work_position_satisfied(vehicle: Any, *, track_name: str, state: Any) -> bool:
    goal = vehicle.goal
    kind = goal.work_position_kind
    if kind is None:
        return False
    if track_name not in goal.allowed_target_tracks:
        return False
    seq = list(state.track_sequences.get(track_name, []))
    if vehicle.vehicle_no not in seq:
        return False
    if kind == "FREE":
        return True
    if kind == "SPOTTING":
        rank = south_rank(seq, vehicle.vehicle_no)
        return rank in allowed_spotting_south_ranks(track_name)
    if kind == "EXACT_NORTH_RANK":
        return north_rank(seq, vehicle.vehicle_no) == goal.target_rank
    if kind == "EXACT_WORK_SLOT":
        rank = north_rank(seq, vehicle.vehicle_no)
        return (
            rank is not None
            and goal.target_rank is not None
            and rank <= goal.target_rank
        )
    return False


def work_slot_violations_by_vehicle(
    *,
    vehicles: list[Any],
    state: Any,
) -> dict[str, str]:
    violations: dict[str, str] = {}
    vehicles_by_track: dict[str, list[Any]] = {}
    for vehicle in vehicles:
        goal = vehicle.goal
        if goal.work_position_kind != "EXACT_WORK_SLOT":
            continue
        current_track = _locate_track(state.track_sequences, vehicle.vehicle_no)
        if current_track is None or current_track not in goal.allowed_target_tracks:
            continue
        vehicles_by_track.setdefault(current_track, []).append(vehicle)

    for track_name, track_vehicles in vehicles_by_track.items():
        slot_counts = Counter(vehicle.goal.target_rank for vehicle in track_vehicles)
        for vehicle in track_vehicles:
            target_rank = vehicle.goal.target_rank
            if target_rank is None:
                violations[vehicle.vehicle_no] = "missing explicit work slot"
                continue
            if slot_counts[target_rank] > 1:
                violations[vehicle.vehicle_no] = (
                    f"duplicate explicit work slot {target_rank} on {track_name}"
                )

        ordered = sorted(
            track_vehicles,
            key=lambda vehicle: (
                north_rank(
                    list(state.track_sequences.get(track_name, [])),
                    vehicle.vehicle_no,
                )
                or 10**9,
                vehicle.vehicle_no,
            ),
        )
        max_seen_slot = 0
        for vehicle in ordered:
            target_rank = vehicle.goal.target_rank
            if target_rank is None:
                continue
            if target_rank < max_seen_slot:
                violations[vehicle.vehicle_no] = (
                    f"explicit work slot {target_rank} is north of an earlier larger slot on {track_name}"
                )
            max_seen_slot = max(max_seen_slot, target_rank)
        min_south_slot = 10**9
        for vehicle in reversed(ordered):
            target_rank = vehicle.goal.target_rank
            if target_rank is None:
                continue
            if target_rank > min_south_slot:
                violations[vehicle.vehicle_no] = (
                    f"explicit work slot {target_rank} is south of a later smaller slot on {track_name}"
                )
            min_south_slot = min(min_south_slot, target_rank)

    return violations


def preview_work_positions_after_prepend(
    *,
    target_track: str,
    incoming_vehicle_nos: list[str],
    existing_vehicle_nos: list[str],
    vehicle_by_no: dict[str, Any],
) -> WorkPositionPreview:
    if not is_work_position_track(target_track):
        return WorkPositionPreview(valid=True)

    new_seq = list(incoming_vehicle_nos) + list(existing_vehicle_nos)
    evaluations: dict[str, WorkPositionEvaluation] = {}
    hard_violations: list[str] = []
    seen: set[str] = set()
    for vehicle_no in new_seq:
        if vehicle_no in seen:
            continue
        seen.add(vehicle_no)
        vehicle = vehicle_by_no.get(vehicle_no)
        if vehicle is None:
            continue
        goal = vehicle.goal
        kind = goal.work_position_kind
        if kind is None or goal.target_track != target_track:
            continue
        evaluation, violation = _evaluate_vehicle(
            vehicle_no=vehicle_no,
            kind=kind,
            target_track=target_track,
            target_rank=goal.target_rank,
            new_seq=new_seq,
        )
        evaluations[vehicle_no] = evaluation
        if violation is not None:
            hard_violations.append(violation)
    return WorkPositionPreview(
        valid=not hard_violations,
        hard_violations=hard_violations,
        evaluations=evaluations,
    )


def build_work_position_assignments(
    *,
    vehicles: list[Any],
    state: Any,
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for vehicle in vehicles:
        goal = vehicle.goal
        if goal.work_position_kind is None:
            continue
        current_track = _locate_track(state.track_sequences, vehicle.vehicle_no)
        if current_track is None:
            continue
        seq = list(state.track_sequences.get(current_track, []))
        assignments[vehicle.vehicle_no] = {
            "track": current_track,
            "northRank": north_rank(seq, vehicle.vehicle_no),
            "southRank": south_rank(seq, vehicle.vehicle_no),
            "rule": goal.work_position_kind,
            "targetRank": goal.target_rank,
            "satisfied": work_position_satisfied(
                vehicle,
                track_name=current_track,
                state=state,
            ),
        }
    return dict(sorted(assignments.items()))


def _locate_track(track_sequences: dict[str, list[str]], vehicle_no: str) -> str | None:
    for track, seq in track_sequences.items():
        if vehicle_no in seq:
            return track
    return None


def _evaluate_vehicle(
    *,
    vehicle_no: str,
    kind: str,
    target_track: str,
    target_rank: int | None,
    new_seq: list[str],
) -> tuple[WorkPositionEvaluation, str | None]:
    n_rank = north_rank(new_seq, vehicle_no)
    s_rank = south_rank(new_seq, vehicle_no)
    if kind == "FREE":
        return (
            WorkPositionEvaluation(
                vehicle_no=vehicle_no,
                kind=kind,
                north_rank=n_rank,
                south_rank=s_rank,
                satisfied_now=True,
            ),
            None,
        )
    if kind == "SPOTTING":
        satisfied = s_rank in allowed_spotting_south_ranks(target_track)
        violation = None
        if not satisfied:
            violation = (
                f"vehicle {vehicle_no} on {target_track} has south_rank={s_rank}, "
                f"expected one of {sorted(allowed_spotting_south_ranks(target_track))}"
            )
        return (
            WorkPositionEvaluation(
                vehicle_no=vehicle_no,
                kind=kind,
                north_rank=n_rank,
                south_rank=s_rank,
                satisfied_now=satisfied,
            ),
            violation,
        )
    if kind == "EXACT_NORTH_RANK":
        rank_gap = None if n_rank is None or target_rank is None else target_rank - n_rank
        satisfied = rank_gap == 0
        violation = None
        if target_rank is None:
            violation = f"vehicle {vehicle_no} exact work position is missing target_rank"
        elif rank_gap is not None and rank_gap < 0:
            violation = (
                f"vehicle {vehicle_no} on {target_track} has north_rank={n_rank}, "
                f"expected no more than {target_rank}"
            )
        return (
            WorkPositionEvaluation(
                vehicle_no=vehicle_no,
                kind=kind,
                north_rank=n_rank,
                south_rank=s_rank,
                target_rank=target_rank,
                rank_gap=rank_gap,
                satisfied_now=satisfied,
            ),
            violation,
        )
    if kind == "EXACT_WORK_SLOT":
        rank_gap = None if n_rank is None or target_rank is None else target_rank - n_rank
        satisfied = rank_gap is not None and rank_gap >= 0
        violation = None
        if target_rank is None:
            violation = f"vehicle {vehicle_no} explicit work slot is missing target_rank"
        elif rank_gap is not None and rank_gap < 0:
            violation = (
                f"vehicle {vehicle_no} on {target_track} has north_rank={n_rank}, "
                f"expected no more than work slot {target_rank}"
            )
        return (
            WorkPositionEvaluation(
                vehicle_no=vehicle_no,
                kind=kind,
                north_rank=n_rank,
                south_rank=s_rank,
                target_rank=target_rank,
                rank_gap=rank_gap,
                satisfied_now=satisfied,
            ),
            violation,
        )
    return (
        WorkPositionEvaluation(
            vehicle_no=vehicle_no,
            kind=kind,
            north_rank=n_rank,
            south_rank=s_rank,
            satisfied_now=False,
        ),
        f"vehicle {vehicle_no} has unsupported work position kind {kind}",
    )
