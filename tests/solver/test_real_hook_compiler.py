import pytest

from fzed_shunting.solver.real_hook_compiler import compile_put_to_real_hook
from fzed_shunting.solver.types import HookAction


def test_put_to_real_hook_compiler_is_disabled_after_tail_only_cutover():
    put_plan = [
        HookAction(
            source_track="存1",
            target_track="存3",
            vehicle_nos=["A"],
            path_tracks=["存1", "存3"],
            action_type="PUT",
        ),
        HookAction(
            source_track="存2",
            target_track="存3",
            vehicle_nos=["B"],
            path_tracks=["存2", "存3"],
            action_type="PUT",
        ),
    ]

    with pytest.raises(RuntimeError, match="native real-hook solver"):
        compile_put_to_real_hook(put_plan)
