from __future__ import annotations


def carried_tail_block(carry: tuple[str, ...] | list[str], size: int) -> tuple[str, ...]:
    if size < 0:
        raise ValueError("tail block size must be non-negative")
    if size == 0:
        return ()
    if size > len(carry):
        raise ValueError("tail block size exceeds carry length")
    return tuple(carry[-size:])


def is_carried_tail_block(
    carry: tuple[str, ...] | list[str],
    vehicle_nos: tuple[str, ...] | list[str],
) -> bool:
    return carried_tail_block(carry, len(vehicle_nos)) == tuple(vehicle_nos)


def remove_carried_tail_block(
    carry: tuple[str, ...] | list[str],
    vehicle_nos: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if not is_carried_tail_block(carry, vehicle_nos):
        raise ValueError("Vehicle block is not at the tail of loco_carry")
    if not vehicle_nos:
        return tuple(carry)
    return tuple(carry[: -len(vehicle_nos)])


def iter_carried_tail_blocks(carry: tuple[str, ...] | list[str]):
    for size in range(1, len(carry) + 1):
        yield list(carried_tail_block(carry, size))
