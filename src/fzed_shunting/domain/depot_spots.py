from __future__ import annotations

from fzed_shunting.io.normalize_input import NormalizedPlanInput, NormalizedVehicle


DEPOT_TRACK_SPOTS: dict[str, dict[str, list[str]]] = {
    "修1": {
        "NORMAL": [f"10{i}" for i in range(1, 6)],
        "INSPECTION": [f"10{i}" for i in range(1, 8)],
    },
    "修2": {
        "NORMAL": [f"20{i}" for i in range(1, 6)],
        "INSPECTION": [f"20{i}" for i in range(1, 8)],
    },
    "修3": {
        "NORMAL": [f"30{i}" for i in range(1, 6)],
        "INSPECTION": [f"30{i}" for i in range(1, 8)],
    },
    "修4": {
        "NORMAL": [f"40{i}" for i in range(1, 6)],
        "INSPECTION": [f"40{i}" for i in range(1, 8)],
    },
}

SPECIAL_FIXED_SPOTS: dict[str, list[str]] = {
    "机库:WEIGH": ["机库:WEIGH"],
}

LONG_DEPOT_TRACKS = {"修3", "修4"}


def is_depot_inner_track(track_code: str) -> bool:
    return track_code in DEPOT_TRACK_SPOTS


def list_track_spots(track_code: str, yard_mode: str) -> list[str]:
    return list(DEPOT_TRACK_SPOTS.get(track_code, {}).get(yard_mode, []))


def build_initial_spot_assignments(plan_input: NormalizedPlanInput) -> dict[str, str]:
    assignments: dict[str, str] = {}
    grouped: dict[str, list[NormalizedVehicle]] = {}
    for vehicle in sorted(plan_input.vehicles, key=lambda item: (item.current_track, item.order)):
        if _requires_spot_assignment(vehicle, vehicle.current_track):
            grouped.setdefault(vehicle.current_track, []).append(vehicle)
    for track_code, vehicles in grouped.items():
        allocated = allocate_spots_for_block(
            vehicles=vehicles,
            target_track=track_code,
            yard_mode=plan_input.yard_mode,
            occupied_spot_assignments=assignments,
        )
        if allocated is None:
            continue
        assignments.update(allocated)
    return assignments


def exact_spot_reservations(plan_input: NormalizedPlanInput) -> frozenset[str]:
    return frozenset(
        vehicle.goal.target_spot_code
        for vehicle in plan_input.vehicles
        if vehicle.goal.target_mode == "SPOT" and vehicle.goal.target_spot_code
    )


def allocate_spots_for_block(
    vehicles: list[NormalizedVehicle],
    target_track: str,
    yard_mode: str,
    occupied_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str] | None = None,
) -> dict[str, str] | None:
    if not vehicles:
        return {}
    if not any(_requires_spot_assignment(vehicle, target_track) for vehicle in vehicles):
        return {}

    taken_spots = set(occupied_spot_assignments.values())
    allocations: dict[str, str] = {}
    vehicle_by_no = {vehicle.vehicle_no: vehicle for vehicle in vehicles}
    min_slot_no: int | None = None
    for vehicle in vehicles:
        # Only vehicles that actually need a spot participate in allocation.
        # Non-spot vehicles riding along (e.g. a blocker cadet coming to 机库
        # together with a weigh vehicle) simply occupy no spot.
        if not _requires_spot_assignment(vehicle, target_track):
            continue
        candidate_spots = spot_candidates_for_vehicle(
            vehicle,
            target_track,
            yard_mode,
            reserved_spot_codes=reserved_spot_codes,
            occupied_spot_assignments={**occupied_spot_assignments, **allocations},
            vehicle_by_no=vehicle_by_no,
        )
        if not candidate_spots:
            return None
        if is_depot_inner_track(target_track) and min_slot_no is not None:
            candidate_spots = [
                spot_code
                for spot_code in candidate_spots
                if (_spot_slot_no(spot_code) or 0) >= min_slot_no
            ]
        if not candidate_spots:
            return None
        chosen = next(
            (
                spot_code
                for spot_code in candidate_spots
                if spot_code not in taken_spots and spot_code not in allocations.values()
            ),
            None,
        )
        if chosen is None:
            return None
        allocations[vehicle.vehicle_no] = chosen
        if is_depot_inner_track(target_track):
            slot_no = _spot_slot_no(chosen)
            if slot_no is not None:
                min_slot_no = slot_no
    return allocations


def realign_spots_for_track_order(
    *,
    vehicle_nos_in_order: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
    target_track: str,
    yard_mode: str,
    current_spot_assignments: dict[str, str],
    reserved_spot_codes: set[str] | frozenset[str] | None = None,
) -> dict[str, str] | None:
    """Rebuild spot assignments for one physical track order.

    The sequence is the source of truth after a detach. Depot spots are
    positional, so inserting vehicles at the accessible end can shift existing
    occupants to later spot codes. Work-track positions are derived from the
    final track sequence and are not stored here.
    """
    target_vehicle_nos = set(vehicle_nos_in_order)
    base_assignments = {
        vehicle_no: spot_code
        for vehicle_no, spot_code in current_spot_assignments.items()
        if vehicle_no not in target_vehicle_nos
    }
    ordered_vehicles = [
        vehicle_by_no[vehicle_no]
        for vehicle_no in vehicle_nos_in_order
        if vehicle_no in vehicle_by_no
    ]
    realigned = allocate_spots_for_block(
        vehicles=ordered_vehicles,
        target_track=target_track,
        yard_mode=yard_mode,
        occupied_spot_assignments=base_assignments,
        reserved_spot_codes=reserved_spot_codes,
    )
    if realigned is None:
        return None
    next_assignments = dict(base_assignments)
    next_assignments.update(realigned)
    return next_assignments


def spot_candidates_for_vehicle(
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
    reserved_spot_codes: set[str] | frozenset[str] | None = None,
    occupied_spot_assignments: dict[str, str] | None = None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None = None,
) -> list[str]:
    if vehicle.need_weigh and target_track == "机库":
        candidates = list(SPECIAL_FIXED_SPOTS["机库:WEIGH"])
        return _without_reserved_foreign_spots(
            vehicle, target_track, yard_mode, candidates, reserved_spot_codes
        )
    if vehicle.goal.target_mode == "TRACK" and vehicle.goal.target_track == target_track and is_depot_inner_track(target_track):
        candidates = _compatible_depot_spots(
            vehicle=vehicle,
            target_track=target_track,
            yard_mode=yard_mode,
            candidate_spots=list_track_spots(target_track, yard_mode),
            occupied_spot_assignments=occupied_spot_assignments,
            vehicle_by_no=vehicle_by_no,
        )
        return _without_reserved_foreign_spots(
            vehicle, target_track, yard_mode, candidates, reserved_spot_codes
        )
    if vehicle.goal.target_mode == "SPOT":
        candidates = _exact_depot_spot_candidates(
            vehicle,
            target_track,
            yard_mode,
            occupied_spot_assignments=occupied_spot_assignments,
            vehicle_by_no=vehicle_by_no,
        )
        return _without_reserved_foreign_spots(
            vehicle, target_track, yard_mode, candidates, reserved_spot_codes
        )
    if vehicle.goal.target_area_code == "大库:RANDOM":
        candidates = _random_depot_spot_candidates(
            vehicle,
            target_track,
            yard_mode,
            occupied_spot_assignments=occupied_spot_assignments,
            vehicle_by_no=vehicle_by_no,
        )
        return _without_reserved_foreign_spots(
            vehicle, target_track, yard_mode, candidates, reserved_spot_codes
        )
    return []


def _without_reserved_foreign_spots(
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
    candidate_spots: list[str],
    reserved_spot_codes: set[str] | frozenset[str] | None,
) -> list[str]:
    if not reserved_spot_codes:
        return candidate_spots
    own_exact_spots = set(_own_exact_spot_candidates(vehicle, target_track, yard_mode))
    return [
        spot_code
        for spot_code in candidate_spots
        if spot_code not in reserved_spot_codes or spot_code in own_exact_spots
    ]


def _own_exact_spot_candidates(
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
) -> list[str]:
    if vehicle.goal.target_mode != "SPOT":
        return []
    return _exact_depot_spot_candidates(vehicle, target_track, yard_mode)


def _exact_depot_spot_candidates(
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
    occupied_spot_assignments: dict[str, str] | None = None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None = None,
) -> list[str]:
    if not is_depot_inner_track(target_track):
        return []
    available_spots = list_track_spots(target_track, yard_mode)
    if vehicle.goal.target_track != target_track or vehicle.goal.target_spot_code is None:
        return []
    if vehicle.goal.target_spot_code not in available_spots:
        return []
    return _compatible_depot_spots(
        vehicle=vehicle,
        target_track=target_track,
        yard_mode=yard_mode,
        candidate_spots=[vehicle.goal.target_spot_code],
        occupied_spot_assignments=occupied_spot_assignments,
        vehicle_by_no=vehicle_by_no,
    )


def _random_depot_spot_candidates(
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
    occupied_spot_assignments: dict[str, str] | None = None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None = None,
) -> list[str]:
    if not is_depot_inner_track(target_track):
        return []
    available_spots = _compatible_depot_spots(
        vehicle=vehicle,
        target_track=target_track,
        yard_mode=yard_mode,
        candidate_spots=list_track_spots(target_track, yard_mode),
        occupied_spot_assignments=occupied_spot_assignments,
        vehicle_by_no=vehicle_by_no,
    )
    if vehicle.goal.preferred_target_tracks or vehicle.goal.fallback_target_tracks:
        allowed_tracks = set(vehicle.goal.preferred_target_tracks) | set(vehicle.goal.fallback_target_tracks)
        if target_track not in allowed_tracks:
            return []
    return available_spots


def _compatible_depot_spots(
    *,
    vehicle: NormalizedVehicle,
    target_track: str,
    yard_mode: str,
    candidate_spots: list[str],
    occupied_spot_assignments: dict[str, str] | None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None,
) -> list[str]:
    if vehicle.vehicle_length >= 17.6 and target_track not in LONG_DEPOT_TRACKS:
        return []
    spots = list(candidate_spots)
    if vehicle.repair_process == "厂修":
        return [spot_code for spot_code in spots if _spot_slot_no(spot_code) in {4, 5}]
    if vehicle.repair_process != "段修":
        return spots
    max_slot_no = _max_segment_slot_no(
        target_track=target_track,
        yard_mode=yard_mode,
        occupied_spot_assignments=occupied_spot_assignments,
        vehicle_by_no=vehicle_by_no,
    )
    if max_slot_no is None:
        return spots
    return [
        spot_code
        for spot_code in spots
        if (_spot_slot_no(spot_code) or 10**9) <= max_slot_no
    ]


def _max_segment_slot_no(
    *,
    target_track: str,
    yard_mode: str,
    occupied_spot_assignments: dict[str, str] | None,
    vehicle_by_no: dict[str, NormalizedVehicle] | None,
) -> int | None:
    if not occupied_spot_assignments or not vehicle_by_no:
        return None
    track_spots = set(list_track_spots(target_track, yard_mode))
    factory_slots = [
        _spot_slot_no(spot_code)
        for vehicle_no, spot_code in occupied_spot_assignments.items()
        if spot_code in track_spots
        and (vehicle := vehicle_by_no.get(vehicle_no)) is not None
        and vehicle.repair_process == "厂修"
    ]
    factory_slots = [slot_no for slot_no in factory_slots if slot_no is not None]
    if not factory_slots:
        return None
    return min(factory_slots)


def _spot_slot_no(spot_code: str) -> int | None:
    suffix = spot_code[-1:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _requires_spot_assignment(vehicle: NormalizedVehicle, target_track: str) -> bool:
    """Whether a move of ``vehicle`` onto ``target_track`` should consume a spot.

    Spot assignment is a FINAL-placement concept: a vehicle claims its spot
    only when it actually arrives at a true fixed-spot resource such as
    修N库内 or 机库:WEIGH. Work-track ranks are derived from final track
    order, so transit through staging or work tracks must not trigger spot
    allocation.
    """
    if vehicle.need_weigh and target_track == "机库":
        return True
    if vehicle.goal.target_mode == "TRACK" and is_depot_inner_track(target_track):
        # Explicit TRACK goals on depot inner lines still consume physical
        # slots; otherwise random-depot preferred/fallback feasibility will
        # incorrectly treat occupied tracks as having spare capacity.
        return vehicle.goal.target_track == target_track
    if vehicle.goal.target_mode == "SPOT":
        # Only require spot at the declared target track. In transit through
        # staging/other tracks, the vehicle carries no spot.
        return vehicle.goal.target_track == target_track
    if vehicle.goal.target_area_code == "大库:RANDOM":
        # Random-depot vehicles only consume a slot once they actually enter
        # a 修N库内 track. Transit through anything else is spot-free.
        return is_depot_inner_track(target_track)
    return False
