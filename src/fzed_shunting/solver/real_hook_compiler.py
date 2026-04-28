"""Disabled legacy PUT-to-real-hook compiler.

The project now uses the native ATTACH/DETACH state model directly. The old
PUT compiler grouped pickups by destination, which can violate tail-only
DETACH semantics after the physical-order cutover.
"""

from __future__ import annotations

from fzed_shunting.solver.types import HookAction


def compile_put_to_real_hook(put_plan: list[HookAction]) -> list[HookAction]:
    """Reject legacy conversion after the native real-hook cutover."""
    raise RuntimeError(
        "compile_put_to_real_hook is disabled; use the native real-hook solver "
        "so north-end attach, tail-only detach, and north-end placement are "
        "enforced by solver.state/replay."
    )
