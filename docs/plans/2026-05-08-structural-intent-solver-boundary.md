# Structural Intent Solver Boundary

## Current Finding

The solver should not keep two independent local planners for the same physical problem.

The previous `work_position_sequence` candidate layer moved useful ideas earlier in the search, but it also grew into a second planner:

- it cleared target prefixes;
- released source access blockers;
- selected staging tracks;
- restored buffers;
- replayed candidate states;
- ranked candidates with work-position-specific reserve metadata.

That overlaps with the tail completion logic in `astar_solver.py`, while route blockage and capacity release remain separate fact models. The result is not a unified shunting model; it is multiple partial planners with different legality and scoring contracts.

The rollback keeps the valuable interface only:

- `MoveCandidate.steps` can still represent one or more real `HookAction`s;
- search still applies all steps and charges real hook cost;
- candidate generation currently wraps primitive real-hook moves only.

This preserves the extension point without keeping the work-position-specific second planner.

## Root Model

The next solver layer should model shunting as constrained vehicle-group rearrangement:

- **Order**: final tracks are ordered prepend structures, not unordered sets.
- **Commitment**: a vehicle group that is already in a valid final structure should be protected unless a declared resource debt requires moving it.
- **Resources**: route access, front clearance, capacity release, and staging tracks are shared resources, not separate rescue concepts.
- **Buffer role**: staging tracks should be assigned a role for a candidate, such as `ORDER_BUFFER`, `ROUTE_RELEASE`, or `CAPACITY_RELEASE`.
- **Group granularity**: actions should move compatible contiguous blocks, then compile to ordinary attach/detach hooks.

## Implementation Boundary

The next implementation should add one shared structural model, not another fallback:

1. Build `structural_intent.py`.
2. Convert current facts into one immutable intent object:
   - work-position sequence debt;
   - route blockage facts;
   - capacity release facts;
   - protected committed blocks;
   - available staging buffers with role scores.
3. Generate structural candidates from that object.
4. Compile candidates to normal `HookAction` steps.
5. Reuse the same legality checks for both search candidates and tail completion.

The old tail completion can remain as a temporary safety net, but new structural logic should enter through `generate_move_candidates()` and shared candidate compilers, not through new `astar_solver.py` tail branches.

## Tests To Add First

- A committed final block is protected from ordinary staging churn.
- A work-position same-target non-spotting vehicle is delayed when it would block a SPOTTING rank window.
- A route blocker and capacity blocker are represented as resource debts in the same intent object.
- A buffer track chosen for `ORDER_BUFFER` is not reused for unrelated staging inside the same structural candidate.
- A multi-step structural candidate compiles to replay-valid ordinary `HookAction` steps.
