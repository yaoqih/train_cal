from __future__ import annotations

from collections.abc import Sequence

from fzed_shunting.io.normalize_input import NormalizedVehicle


def _compute_hook_vehicle_group_counts(
    vehicles: Sequence[NormalizedVehicle],
) -> tuple[int, int, int, int]:
    heavy_count = sum(1 for vehicle in vehicles if vehicle.is_heavy)
    empty_count = len(vehicles) - heavy_count
    weigh_count = sum(1 for vehicle in vehicles if vehicle.need_weigh)
    equivalent_empty_count = empty_count + 4 * heavy_count
    return heavy_count, empty_count, weigh_count, equivalent_empty_count


def validate_hook_vehicle_group(vehicles: Sequence[NormalizedVehicle]) -> list[str]:
    heavy_count, empty_count, weigh_count, equivalent_empty_count = (
        _compute_hook_vehicle_group_counts(vehicles)
    )

    errors: list[str] = []
    if (
        len(vehicles) >= 2
        and vehicles[0].is_close_door
        and any(vehicle.is_heavy for vehicle in vehicles[1:])
    ):
        errors.append("有重车时关门车不能排在机后第一位")
    if heavy_count == 0 and empty_count > 20:
        errors.append("纯空车单钩最多 20 辆")
    if heavy_count > 2:
        errors.append("单钩重车最多 2 辆")
    if heavy_count > 0 and equivalent_empty_count > 20:
        errors.append("单钩重车折算后不得超过 20 辆空车")
    if weigh_count > 1:
        errors.append("单钩称重最多处理 1 辆称重车")
    if weigh_count == 1 and vehicles and not vehicles[-1].need_weigh:
        errors.append("称重车必须位于机后最后一位")
    return errors


def tail_unweighed_weigh_vehicle_no(
    vehicle_nos: Sequence[str],
    *,
    vehicle_by_no: dict,
    weighed_vehicle_nos: set[str],
) -> str | None:
    if not vehicle_nos:
        return None
    unweighed_need_weigh = [
        vehicle_no
        for vehicle_no in vehicle_nos
        if (
            (vehicle := vehicle_by_no.get(vehicle_no)) is not None
            and vehicle.need_weigh
            and vehicle_no not in weighed_vehicle_nos
        )
    ]
    if len(unweighed_need_weigh) != 1:
        return None
    tail_vehicle_no = vehicle_nos[-1]
    if tail_vehicle_no != unweighed_need_weigh[0]:
        return None
    return tail_vehicle_no


def close_door_first_for_large_rear_consist(
    vehicle_nos: Sequence[str],
    *,
    vehicle_by_no: dict,
    rear_vehicle_count: int,
) -> bool:
    if rear_vehicle_count <= 10 or not vehicle_nos:
        return False
    first_vehicle = vehicle_by_no.get(vehicle_nos[0])
    return bool(first_vehicle and first_vehicle.is_close_door)
