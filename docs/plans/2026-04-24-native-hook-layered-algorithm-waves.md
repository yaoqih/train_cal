# Native Hook Layered Algorithm Waves

> **Status:** Living implementation plan. Update this document after each wave with validation results and the next decision.

## Goal

Improve the native atomic-hook solver without sample-shaped hacks. The solver must keep one hook equal to one `ATTACH` or one `DETACH`, preserve strict verifier correctness, improve general solvability, reduce hook count, and expose why hard cases are hard.

## Baseline

Current main includes the native atomic hook cutover:

- Native `ATTACH` / `DETACH` state transitions and verification.
- Constructive, beam, anytime, partial-resume, and runner-level no-solution recovery.
- Capacity-infeasible classification in external validation.
- Positive validation is fully solved; truth validation has remaining hard-capacity cases plus search-sensitive long tails.

## Prioritization Rule

Prioritize changes by:

1. Certainty of general benefit.
2. Low risk to correctness and solvability.
3. Low implementation complexity.
4. Measurable impact on hook distribution and debug explainability.

Avoid hard pruning until metrics prove it is safe. Prefer scoring/tiebreaking and verifier-guarded improvements.

## Wave 1: Structural Metrics And Diagnostics

### Purpose

Add observability without changing solver behavior.

### Deliverables

- `StructuralMetrics` computed from `NormalizedPlanInput` + `ReplayState`.
- Metrics exposed in solver `debug_stats` and validation summaries.
- Metrics for complete plans and partial plans where available.

### Metrics

- Staging debt:
  - Non-goal vehicles on staging tracks.
  - Staging pollution by track.
- Goal progress:
  - Unfinished vehicles.
  - Preferred target violations.
  - Area / random unfinished vehicles.
- Capacity debt:
  - Final fixed-track required length.
  - Current non-goal occupancy by target track.
  - Tracks already over current/final capacity.
- Blocker pressure:
  - Front blockers hiding unfinished vehicles.
  - Goal-track blockers occupying tracks needed by other vehicles.
- Plan shape:
  - Staging hook count.
  - Staging-to-staging hook count.
  - Repeated vehicle rehandles.

### Validation

- Unit tests for metrics on staged, blocked, capacity-tight, and random-area states.
- Core solver regression unchanged.
- Positive/truth validation summary includes structural metrics.

## Wave 2: Low-Risk Staging And Natural-Block Scoring

### Purpose

Reduce obvious high-hook patterns without sacrificing solvability.

### Deliverables

- Use structural metrics in constructive scoring.
- Prefer natural block boundaries in move ordering.
- Penalize staging-to-staging and repeated staging rehandles as scoring, not hard pruning.

### Natural Block Boundaries

Prefer prefix sizes that align with:

- Same effective target track.
- Same area/random target family.
- Blocker-exposure boundary.
- Traction/capacity feasible largest useful prefix.

Keep old prefix generation as fallback.

### Validation

- Positive remains fully solved.
- Truth non-hard-infeasible solved count does not drop.
- Staging rehandle metrics improve.
- p75/p90/max hook count improve or stay neutral.

## Wave 3 Decision Gate

After Wave 1/2 full validation, choose one:

### Option A: MoveIntent

Choose if metrics show candidate scoring confusion between goal delivery, blocker clearing, and staging.

### Option B: Native Lower Bound

Choose if expanded/generated nodes are high and partial states show low heuristic despite many remaining debts.

### Option C: Random/Area Assignment

Choose if high hooks or no-solution cases concentrate on `大库:RANDOM`, work-area, or spot allocation.

## Wave 4 Decision Gate

Pick after Wave 3 validation:

- Diversity beam if one intent dominates beam and prunes alternatives.
- Plan compressor if valid plans remain hook-heavy with staging chains.
- Macro compilation only after lower-level metrics and intents are stable.

## Running Validation

Core regression:

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/solver/test_constructive.py \
  tests/solver/test_astar_solver.py \
  tests/solver/test_external_validation_parallel_runs.py \
  tests/solver/test_depot_late.py \
  tests/solver/test_heuristic.py \
  tests/solver/test_move_generator.py \
  tests/verify/test_plan_verifier.py \
  tests/verify/test_replay.py \
  tests/verify/test_route_constraints.py
```

Positive validation:

```bash
rm -rf artifacts/validation_inputs_positive_native_current
PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py \
  --input-dir data/validation_inputs/positive \
  --output-dir artifacts/validation_inputs_positive_native_current \
  --solver beam --beam-width 8 --max-workers 8 \
  --timeout-seconds 120 --solver-time-budget-ms 30000
```

Truth validation:

```bash
rm -rf artifacts/validation_inputs_truth_native_current
PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py \
  --input-dir data/validation_inputs/truth \
  --output-dir artifacts/validation_inputs_truth_native_current \
  --solver beam --beam-width 8 --max-workers 8 \
  --timeout-seconds 120 --solver-time-budget-ms 30000
```

## Wave Log

### 2026-04-24

- Created plan after native hook cutover was merged to `main`.
- Wave 1 implemented:
  - Added `StructuralMetrics` for unfinished, preferred violation, staging debt, area/random unfinished, front blocker pressure, goal-track blocker pressure, capacity debt, and loco carry count.
  - Added plan-shape metrics for staging hooks, staging-to-staging hooks, repeated rehandles, and max vehicle touch count.
  - Solver `debug_stats` now includes `initial_structural_metrics`, `final_structural_metrics` or `partial_structural_metrics`, and `plan_shape_metrics`.
  - Focused tests: `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_structural_metrics.py tests/solver/test_astar_solver.py::test_simple_astar_result_can_return_debug_stats` passed.
- Wave 2 started:
  - Constructive native scoring now uses next-state structural metrics as low-risk within-tier tiebreakers.
  - The first applied signal is structural debt avoidance: prefer fewer goal-track blockers, front blockers, staging debt, and area/random unfinished vehicles before existing purity and block-size tie breakers.
  - Added close-door sequencing debt for `存4北`: when a close-door vehicle still needs to land behind at least three pusher vehicles, scoring preserves the pusher resource and avoids trapping the close-door vehicle behind an already-carried pusher block. This is a business-structural ordering rule, not sample-specific pruning.
  - This is scoring-only; no legal candidate is pruned.
- Wave 3 selected from positive distribution and implemented:
  - High-hook cases had large `area_random_unfinished_count`, high rehandle counts, and heavy depot random traffic.
  - `大库:RANDOM` candidate ordering now prefers less-loaded tracks within the same preference level instead of piling onto already occupied tracks.
  - Positive validation after Wave 3: `64/64` solved, hook distribution `min=2, p50=13, p75=26, p90=86, p95=86, max=150`.
  - Compared with the pre-Wave-3 positive run in this session (`min=2, p50=13, p75=42, p90=86, p95=98, max=125`), p75 and p95 improved substantially, while one constructive tail regressed to 150 hooks.
- Wave 4 investigated but not adopted:
  - The positive max tail is dominated by staging-to-staging chains (`staging_to_staging_hook_count=42` in `case_3_2_shed_pre_repair_from_pre_repair.json`).
  - A simple repeated-staging-block churn penalty was tested and rejected: it worsened the positive tail (`p95=121, max=203`) by steering necessary intermediate staging into worse constructive paths.
  - Next Wave 4 candidate should be a verifier-guarded plan compressor or local post-solve repair, not another greedy scoring penalty.
- Verification so far:
  - Core regression: `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_constructive.py tests/solver/test_astar_solver.py tests/solver/test_external_validation_parallel_runs.py tests/solver/test_depot_late.py tests/solver/test_heuristic.py tests/solver/test_move_generator.py tests/solver/test_structural_metrics.py tests/verify/test_plan_verifier.py tests/verify/test_replay.py tests/verify/test_route_constraints.py` passed with `226 passed, 5 skipped`.
  - Positive full validation artifact: `artifacts/validation_inputs_positive_native_wave_final2/summary.json`.
  - Truth full validation is running as `artifacts/validation_inputs_truth_native_wave_final_stable`.

### 2026-04-25

- Truth full validation completed:
  - Artifact: `artifacts/validation_inputs_truth_native_wave_final_stable/summary.json`.
  - Result: `117/127` solved.
  - Failures: `10/127`, all `capacity_infeasible`; no remaining `no_solution` in this run.
  - Hook distribution on solved truth scenarios: `min=5, p50=76, p75=122, p90=167, p95=192, max=443`.
  - High-tail cases remain dominated by staging churn and repeated rehandles:
    - `validation_20260310W.json`: 443 hooks, `staging_hook_count=302`, `staging_to_staging_hook_count=215`, `max_vehicle_touch_count=94`.
    - `validation_20260127W.json`: 438 hooks, `staging_hook_count=66`, `max_vehicle_touch_count=160`.
- Implementation boundary for this branch:
  - Adopted: Wave 1 metrics, Wave 2 structural scoring and close-door sequencing, Wave 3 random-depot load balancing.
  - Not adopted: Wave 4 greedy churn scoring, because positive validation worsened from `p95=86, max=150` to `p95=121, max=203`.
- Next recommended branch:
  - Build a verifier-guarded local plan compressor for staging-to-staging chains.
  - The compressor should work on complete valid plans, propose local rewrites, replay and verify before accepting, and compare `(hook_count, staging_to_staging_hook_count, max_vehicle_touch_count)`.
  - Avoid more greedy scoring penalties until the compressor provides counterexamples and accepted rewrite patterns.

### 2026-04-25 Continued

- Native hook semantic hardening:
  - Added a replay/verifier invariant for native `DETACH`: the hook `sourceTrack` must match the current loco track before detaching carried vehicles.
  - Rationale: an empty loco may move to any source track for `ATTACH`, but a loaded loco cannot detach from a different source than its actual current track. Without this invariant, post-solve compression can create physically discontinuous false-short plans.
  - Focused verification: `PYTHONPATH=src .venv/bin/pytest -q tests/verify/test_replay.py tests/verify/test_plan_verifier.py tests/verify/test_route_constraints.py tests/solver/test_plan_compressor.py tests/solver/test_astar_solver.py::test_simple_astar_can_clear_interfering_track_via_temporary_track tests/solver/test_depot_late.py` passed with `70 passed`.
  - Core regression after hardening: `234 passed, 5 skipped`.
- Wave 4A implemented: verifier-guarded plan compressor.
  - Delete-window compression removes redundant local windows when the terminal state is unchanged and full verification passes.
  - Single-source window rebuild replaces a local chain that starts by attaching a source prefix and eventually leaves only those vehicles on final prefixes, preserving source-order detach groups and continuous native hook sources.
  - Compression runs before depot-late sequencing so hook count remains the primary objective; depot-late remains a secondary reorder.
  - Compression now runs up to a conservative bounded convergence limit (`max_passes=8`) because truth validation showed many cases exactly hitting the previous 2-pass cap.
- Wave 4A validation before convergence-limit increase:
  - Truth artifact: `artifacts/validation_inputs_truth_compressor_wave3/summary.json`.
  - Truth result: `117/127` solved; failures remained `10 capacity_infeasible`.
  - Truth hook distribution improved from baseline `min=5, p50=76, p75=122, p90=167, p95=192, max=443` to `min=5, p50=74, p75=110, p90=163, p95=188, max=439`.
  - Compression accepted `154` rewrites across truth solved cases; `82` solved truth cases shortened.
  - Positive remained `64/64` solved, but p95 was unstable across runs (`102` parallel, `113` solo versus baseline `86`), while max improved from `150` to `147`. This indicates the next major gain should come from search/constructive stability and staging-chain prevention, not only post-solve compression.
- Wave 4B in progress:
  - Bounded convergence compression (`max_passes=8`) is implemented and under full validation as:
    - `artifacts/validation_inputs_positive_compressor_wave4`
    - `artifacts/validation_inputs_truth_compressor_wave4`
  - Decision gate after Wave 4B validation:
    - Keep convergence compression if truth p75/p90/p95 improves without positive solvability loss.
    - If positive p95 remains worse, do not add more broad post-processing; implement a lower-layer staging-chain prevention or stable candidate-selection improvement instead.
- Wave 4C adopted:
  - Increased verifier-guarded compression convergence to `max_passes=16`.
  - Focused tests cover redundant-window deletion, single-source rebuilds, repeated rebuild windows beyond the old two-pass cap, convergence over many rebuild windows, and rejection of source-discontinuous candidates.
  - Core regression after the convergence increase: `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_plan_compressor.py tests/solver/test_constructive.py tests/solver/test_astar_solver.py tests/solver/test_external_validation_parallel_runs.py tests/solver/test_depot_late.py tests/solver/test_heuristic.py tests/solver/test_move_generator.py tests/solver/test_structural_metrics.py tests/verify/test_plan_verifier.py tests/verify/test_replay.py tests/verify/test_route_constraints.py` passed with `235 passed, 5 skipped`.
  - Positive validation artifact: `artifacts/validation_inputs_positive_compressor_wave5/summary.json`.
  - Positive result: `64/64` solved; hook distribution `min=2, p50=13, p75=26, p90=85, p95=86, max=129`; accepted rewrites `45`.
  - Truth validation artifact: `artifacts/validation_inputs_truth_compressor_wave5/summary.json`.
  - Truth result: `117/127` solved; failures unchanged at `10`, all previously classified as capacity-infeasible; hook distribution `min=5, p50=60, p75=97, p90=129, p95=149, max=433`; accepted rewrites `645`.
  - Remaining long tails are not solved by larger local compression windows:
    - `validation_20260310W.json`: `433` hooks, only `4` accepted rewrites, `staging_hook_count=296`, `staging_to_staging_hook_count=209`, `max_vehicle_touch_count=94`.
    - `validation_20260127W.json`: `388` hooks, `24` accepted rewrites, `staging_hook_count=48`, `staging_to_staging_hook_count=21`, `max_vehicle_touch_count=154`.
  - Next algorithm layer should act before plan construction commits to poor temporary tracks: add route/contention-aware staging target ordering as a soft tiebreaker, not a hard prune, then compare against Wave 4C.

### 2026-04-25 Route-Contention Experiments

- Wave 6 tested route-contention-aware staging target ordering globally:
  - Mechanism: deprioritize temporary tracks that lie on the pending source-to-goal corridor, while keeping them as fallback candidates.
  - Positive artifact: `artifacts/validation_inputs_positive_route_contention_wave6/summary.json`.
  - Positive result: `64/64` solved, but distribution regressed from Wave 4C `min=2, p50=13, p75=26, p90=85, p95=86, max=129` to `min=2, p50=13, p75=24, p90=116, p95=123, max=154`.
  - Truth artifact: `artifacts/validation_inputs_truth_route_contention_wave6/summary.json`.
  - Truth result: solved count regressed from `117/127` to `115/127`; distribution regressed to `min=5, p50=60, p75=104, p90=160, p95=252, max=813`.
  - Positive signal: `validation_20260310W.json` improved from `433` to `252` hooks and `validation_20260127W.json` improved from `388` to `278`.
  - Rejection reason: the same global ordering created new extreme tails (`validation_20260212W.json` at `813`, `validation_2025_09_08_noon.json` at `663`) and reduced solvability. The signal is real but too blunt as a default candidate order.
- Wave 7 tested a narrower front-blocker-only variant:
  - Mechanism: only when detaching an already-satisfied front blocker to expose the next unfinished vehicle, use the next vehicle's route as an avoid-corridor hint.
  - Positive artifact: `artifacts/validation_inputs_positive_front_blocker_route_wave7/summary.json`.
  - Positive result: `64/64` solved, but distribution still regressed to `min=2, p50=13, p75=26, p90=86, p95=96, max=129`.
  - Truth artifact: `artifacts/validation_inputs_truth_front_blocker_route_wave7/summary.json`.
  - Truth result: solved count recovered to `117/127`, but distribution regressed to `min=5, p50=63, p75=101, p90=144, p95=162, max=1263`.
  - Positive signal: `validation_20260127W.json` improved from `388` to `117`.
  - Rejection reason: `validation_20260317Z.json` exploded from `78` to `1263` hooks. Even a narrow ordering change can create unstable constructive paths because early staging choices alter later block formation and compression opportunities.
- Decision:
  - Do not adopt route-contention ordering as a default move-generator rule.
  - Keep the artifacts as evidence that future-route awareness is useful but must be applied through a safer mechanism.
  - Next safer direction: candidate diversity or two-stage local selection. Instead of changing the single default order, preserve the existing distance/occupancy order and add limited alternative branches for low-contention temporary tracks, then let beam/score choose with structural metrics. This avoids forcing every scenario through the same route-avoidance bias.
- Wave 8 tested route-clear candidate diversity:
  - Mechanism: keep the existing default staging order, but when detaching a satisfied front blocker, add one extra route-clear temporary-track candidate instead of replacing the first two default candidates.
  - Positive artifact: `artifacts/validation_inputs_positive_route_diversity_wave8/summary.json`.
  - Positive result: `64/64` solved; distribution `min=2, p50=13, p75=26, p90=86, p95=86, max=144`.
  - Truth artifact: `artifacts/validation_inputs_truth_route_diversity_wave8/summary.json`.
  - Truth result: `117/127` solved; distribution `min=5, p50=63, p75=101, p90=140, p95=149, max=433`.
  - Positive signal: `case_3_2_shed_pre_repair_from_cun5bei.json` improved from `85` to `64`; `validation_20260127W.json` improved from `388` to `291`.
  - Rejection reason: despite no solvability loss and no new extreme max, net hooks increased on both positive and truth, positive max worsened from `129` to `144`, and truth p50/p75/p90 worsened. This remains a seesaw improvement, not a robust default.
- Updated decision:
  - Route-aware staging is not ready as a move-generation default, even as diversity.
  - The robust next layer should move route awareness into a score/selection layer that can compare complete partial states, not into candidate generation. Candidate generation should remain conservative and stable.
  - If revisited, route-clear alternatives should be behind a measurable state-quality selector: only accept the alternative when the next state improves structural metrics enough to offset extra search width and does not increase staging-to-staging debt.

### 2026-04-25 Lifecycle Scoring Experiment

- Wave 9 tested repeated-touch and fresh-staging-debt tie-breakers in constructive scoring:
  - Mechanism: keep move generation unchanged; when constructive scores same-tier moves, prefer lower repeated vehicle touch counts and fewer newly introduced unfinished vehicles on staging tracks.
  - Plan: `docs/plans/2026-04-25-native-hook-lifecycle-scoring.md`.
  - Positive artifact: `artifacts/validation_inputs_positive_lifecycle_wave9/summary.json`.
  - Positive result: `64/64` solved; distribution `min=2, p50=13, p75=26, p90=86, p95=86, max=129`.
  - Truth artifact: `artifacts/validation_inputs_truth_lifecycle_wave9/summary.json`.
  - Truth result: `117/127` solved; distribution `min=5, p50=63, p75=101, p90=135, p95=160, max=433`.
  - Positive signal: small positive net movement (`-3` hooks total) and `case_3_2_shed_work_*` improved from `105` to `101`.
  - Truth signal: some large improvements (`validation_20260318W.json` `199 -> 114`, `validation_20260304W.json` `142 -> 91`).
  - Rejection reason: truth p50/p75/p90/p95 all regressed, truth net hooks worsened by `505`, and `validation_2025_09_09_noon.json` regressed from `147` to `366`.
- Decision:
  - Do not adopt lifecycle tie-breakers as unconditional constructive ordering rules.
  - The same pattern as Wave 6-8 repeats: local signals are valid, but unconditional global insertion causes seesaw regressions.
  - Next implementation should use a bounded accept/reject layer: generate the default greedy choice and one lifecycle-favored alternative, simulate both one or more steps, and only switch when a composite state-quality predicate improves without increasing known risk metrics.
