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
    if heavy_count == 0 and empty_count > 20:
        errors.append("纯空车单钩最多 20 辆")
    if heavy_count > 2:
        errors.append("单钩重车最多 2 辆")
    if heavy_count > 0 and equivalent_empty_count > 20:
        errors.append("单钩重车折算后不得超过 20 辆空车")
    if weigh_count > 1:
        errors.append("单钩称重最多处理 1 辆称重车")
    return errors
