"""Compile a PUT-based hook plan into the real-hook ATTACH/DETACH representation.

Each PUT(A→B, vehicles) normally costs 2 real hooks (1 ATTACH + 1 DETACH).
This compiler applies a merge optimisation: when consecutive PUTs share the
same destination track, their ATTACHes can be combined into one loco "trip",
reducing total hooks.

Example:
    PUT(A→C, v1) + PUT(B→C, v2)  →  ATTACH(A,v1) + ATTACH(B,v2) + DETACH(C,v1+v2)
    = 3 hooks instead of 4.
"""

from __future__ import annotations

from fzed_shunting.solver.types import HookAction


def compile_put_to_real_hook(put_plan: list[HookAction]) -> list[HookAction]:
    """Convert a PUT-based plan to an optimised ATTACH/DETACH plan.

    Strategy: group consecutive PUTs that share the same target_track into a
    single "trip" (multiple ATTACHes, one DETACH per distinct group at that
    target), saving 1 hook per merged pair.

    Returns a new plan list with action_type in {"ATTACH", "DETACH"}.
    """
    if not put_plan:
        return []

    result: list[HookAction] = []

    # Walk the PUT plan and merge consecutive PUTs targeting the same track.
    i = 0
    while i < len(put_plan):
        current = put_plan[i]
        target = current.target_track

        # Collect a run of consecutive PUTs to the same target.
        run: list[HookAction] = [current]
        j = i + 1
        while j < len(put_plan) and put_plan[j].target_track == target:
            run.append(put_plan[j])
            j += 1

        if len(run) == 1:
            # No merge: 1 ATTACH + 1 DETACH = 2 hooks (standard trip)
            put = run[0]
            result.append(HookAction(
                source_track=put.source_track,
                target_track="LOCO",
                vehicle_nos=list(put.vehicle_nos),
                path_tracks=[put.source_track],
                action_type="ATTACH",
            ))
            result.append(HookAction(
                source_track="LOCO",
                target_track=put.target_track,
                vehicle_nos=list(put.vehicle_nos),
                path_tracks=put.path_tracks,
                action_type="DETACH",
            ))
        else:
            # Merge: N ATTACHes + 1 DETACH for all vehicles (or per sub-group)
            all_vehicles: list[str] = []
            for put in run:
                result.append(HookAction(
                    source_track=put.source_track,
                    target_track="LOCO",
                    vehicle_nos=list(put.vehicle_nos),
                    path_tracks=[put.source_track],
                    action_type="ATTACH",
                ))
                all_vehicles.extend(put.vehicle_nos)
            # Single DETACH for all accumulated vehicles.
            # Use path_tracks from the last PUT (close enough for routing).
            result.append(HookAction(
                source_track="LOCO",
                target_track=target,
                vehicle_nos=all_vehicles,
                path_tracks=run[-1].path_tracks,
                action_type="DETACH",
            ))

        i = j

    return result
