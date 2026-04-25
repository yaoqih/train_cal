# Native Hook Lifecycle Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce truth long-tail hook counts by adding state-quality scoring for staging lifecycle and repeated vehicle touches without changing legal move generation or verifier semantics.

**Architecture:** Keep move generation stable. Add low-risk tie-breakers inside constructive native scoring, where complete next states are already available. The first wave only penalizes repeated vehicle touches within the current constructive plan and fresh staging debt for non-goal detaches; if full validation shows seesaw regressions, keep the evidence in this document and revert the behavior.

**Tech Stack:** Python, pytest, existing `fzed_shunting.solver.constructive`, `move_generator`, `structural_metrics`, and external validation runner.

---

### Task 1: Repeated Touch Scoring

**Files:**
- Modify: `src/fzed_shunting/solver/constructive.py`
- Test: `tests/solver/test_constructive.py`

**Step 1: Write failing test**

Add a test that builds two same-tier native moves with identical heuristic/structural outcome except one moves a vehicle already touched many times in the current constructive plan. Assert the lower-repeat move scores better.

**Step 2: Run failing test**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_constructive.py::test_score_native_move_prefers_less_rehandled_vehicle_within_same_tier
```

Expected: fail because `_score_native_move` has no `vehicle_touch_counts` argument or does not use it.

**Step 3: Implement minimal scoring input**

- Maintain `vehicle_touch_counts: Counter[str]` in `_greedy_forward`.
- Pass it into `_score_native_move`.
- Add optional parameter `vehicle_touch_counts: dict[str, int] | None = None` for tests/backward compatibility within internal call sites.
- Add a tie-breaker after structural debt metrics and before block size/path length:
  - `max(vehicle_touch_counts.get(vno, 0) for vno in move.vehicle_nos, default=0)`
  - `sum(vehicle_touch_counts.get(vno, 0) for vno in move.vehicle_nos)`
- Update counts only after chosen move is appended.

**Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_constructive.py::test_score_native_move_prefers_less_rehandled_vehicle_within_same_tier tests/solver/test_constructive.py tests/solver/test_structural_metrics.py
```

### Task 2: Fresh Staging Debt Tie-Breaker

**Files:**
- Modify: `src/fzed_shunting/solver/constructive.py`
- Test: `tests/solver/test_structural_metrics.py` or `tests/solver/test_constructive.py`

**Step 1: Write failing test**

Construct two staging detach moves in the same tier: one drops an unfinished vehicle to an empty staging track; the other drops a block that is already goal-satisfied or creates less new staging debt. Assert less fresh staging debt scores better.

**Step 2: Implement helper**

Add helper `_fresh_staging_debt(move, state, next_state, plan_input, vehicle_by_no)` returning count of vehicles newly placed on staging and not goal-satisfied there.

Insert after existing `next_structure.staging_debt_count` so it is a local lifecycle tiebreaker, not a hard priority.

**Step 3: Verify**

Run focused constructive/structural tests.

### Task 3: Full Regression And External Validation

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/solver/test_plan_compressor.py \
  tests/solver/test_constructive.py \
  tests/solver/test_astar_solver.py \
  tests/solver/test_external_validation_parallel_runs.py \
  tests/solver/test_depot_late.py \
  tests/solver/test_heuristic.py \
  tests/solver/test_move_generator.py \
  tests/solver/test_structural_metrics.py \
  tests/verify/test_plan_verifier.py \
  tests/verify/test_replay.py \
  tests/verify/test_route_constraints.py
```

Then run positive and truth validation into new artifact directories:

```bash
rm -rf artifacts/validation_inputs_positive_lifecycle_wave9
PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py \
  --input-dir data/validation_inputs/positive \
  --output-dir artifacts/validation_inputs_positive_lifecycle_wave9 \
  --solver beam --beam-width 8 --max-workers 8 \
  --timeout-seconds 120 --solver-time-budget-ms 30000

rm -rf artifacts/validation_inputs_truth_lifecycle_wave9
PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py \
  --input-dir data/validation_inputs/truth \
  --output-dir artifacts/validation_inputs_truth_lifecycle_wave9 \
  --solver beam --beam-width 8 --max-workers 8 \
  --timeout-seconds 120 --solver-time-budget-ms 30000
```

Adoption gate:
- Positive solved remains `64/64`.
- Truth solved remains at least `117/127`.
- Positive p95/max do not regress versus Wave 5 (`p95=86`, `max=129`).
- Truth p75/p90/p95 improve or stay neutral; max must not exceed `433`.
- If net hooks improve but one tail explodes, reject or narrow the scoring.

### Task 4: Documentation And Commit

Update `docs/plans/2026-04-24-native-hook-layered-algorithm-waves.md` with:
- Wave 9 mechanism.
- Positive/truth distributions.
- Top improved and worsened cases.
- Adopt/reject decision.

Commit only relevant files. Do not stage `.claude/scheduled_tasks.lock` or `AGENTS.md`.

## Wave 9 Result

Implemented and tested repeated-touch and fresh-staging-debt tie-breakers in constructive scoring, then ran full regression and positive/truth validation.

Artifacts:

- Positive: `artifacts/validation_inputs_positive_lifecycle_wave9/summary.json`
- Truth: `artifacts/validation_inputs_truth_lifecycle_wave9/summary.json`

Results versus Wave 5:

- Positive stayed solved at `64/64`; distribution changed from `min=2, p50=13, p75=26, p90=85, p95=86, max=129` to `min=2, p50=13, p75=26, p90=86, p95=86, max=129`.
- Truth stayed solved at `117/127`, but distribution regressed from `min=5, p50=60, p75=97, p90=129, p95=149, max=433` to `min=5, p50=63, p75=101, p90=135, p95=160, max=433`.
- Positive net hooks improved by only `3`; truth net hooks worsened by `505`.
- Best truth improvements included `validation_20260318W.json` `199 -> 114` and `validation_20260304W.json` `142 -> 91`.
- Worst truth regressions included `validation_2025_09_09_noon.json` `147 -> 366`, `validation_20260311Z.json` `76 -> 148`, and `validation_20260110Z.json` `55 -> 121`.

Decision:

- Reject Wave 9 behavior. The signals are real but not robust as global constructive tie-breakers.
- Reverted the implementation and tests after validation.
- Keep this document as evidence that lifecycle signals need a state-quality selector or bounded alternative acceptance, not unconditional insertion into the default greedy ordering.
