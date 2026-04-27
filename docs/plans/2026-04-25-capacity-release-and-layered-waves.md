# Capacity Release And Layered Waves Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Wave10 capacity-release facts first, then use validated facts to decide whether and how to proceed with Waves 11-14 without introducing seesaw solver behavior.

**Architecture:** Wave10 is an observability-only fact layer: compute target-line capacity pressure, keepable occupants, required inbound length, non-goal release demand, and front release prefixes. It must not alter move generation, scoring, or verification until the facts explain p75+ truth tails. Later waves may consume this fact layer as low-tier scoring or beam diversity signals, but only after full validation gates.

**Tech Stack:** Python, pytest, existing `NormalizedPlanInput`, `ReplayState`, `goal_is_satisfied`, solver `debug_stats`, validation runner artifacts.

---

### Task 1: CapacityReleasePlan Data Model And Initial Facts

**Files:**
- Create: `src/fzed_shunting/solver/capacity_release.py`
- Create: `tests/solver/test_capacity_release.py`

**Step 1: Write failing tests**

Tests must cover:

- A target track with pending inbound fixed-target vehicles and non-goal current occupants reports inbound length, non-goal occupant length, and release pressure.
- Satisfied vehicles already on the target are counted as keepable length.
- Multi-target/random-area vehicles are not forced into a fixed target in Wave10 facts.
- Front release prefix includes only the near-end prefix needed to expose non-goal releasable vehicles.

**Step 2: Implement facts**

Add dataclasses:

- `TrackCapacityReleaseFact`
- `CapacityReleasePlan`

Initial fields:

- `track_name`
- `capacity_length`
- `current_length`
- `fixed_inbound_length`
- `keepable_current_length`
- `non_goal_current_length`
- `release_pressure_length`
- `front_release_vehicle_nos`
- `front_release_length`
- `current_vehicle_count`
- `fixed_inbound_vehicle_count`
- `non_goal_current_vehicle_count`

Function:

```python
def compute_capacity_release_plan(plan_input: NormalizedPlanInput, state: ReplayState) -> CapacityReleasePlan:
    ...
```

Keep it deterministic and side-effect free.

### Task 2: Expose Wave10 Facts In Solver Debug Stats

**Files:**
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Test: `tests/solver/test_astar_solver.py`

**Step 1: Write failing test**

Extend existing debug stats test to assert `initial_capacity_release_plan` exists and contains per-track release facts.

**Step 2: Implement debug stat attachment**

Attach only initial facts first. Do not change behavior.

### Task 3: P75+ Truth Capacity Release Report

**Files:**
- Create: `scripts/analyze_capacity_release_truth_tails.py`
- Output: `artifacts/analysis/truth_p75_capacity_release_wave10.json`
- Output: `artifacts/analysis/truth_p75_capacity_release_wave10.md`

**Step 1: Implement report script**

Inputs:

- `--summary artifacts/validation_inputs_truth_compressor_wave5/summary.json`
- `--input-dir data/validation_inputs/truth`
- `--master-dir data/master`
- `--output-json artifacts/analysis/truth_p75_capacity_release_wave10.json`
- `--output-md artifacts/analysis/truth_p75_capacity_release_wave10.md`

Behavior:

- Determine p75 from solved hook counts.
- Select solved cases with hook count > p75.
- Compute initial `CapacityReleasePlan` for each selected case.
- Summarize top tracks by release pressure and front-release length.
- Include existing plan-shape metrics from summary for correlation.

### Task 4: Wave10 Validation Gate

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_capacity_release.py tests/solver/test_astar_solver.py tests/solver/test_structural_metrics.py
PYTHONPATH=src .venv/bin/python scripts/analyze_capacity_release_truth_tails.py \
  --summary artifacts/validation_inputs_truth_compressor_wave5/summary.json \
  --input-dir data/validation_inputs/truth \
  --master-dir data/master \
  --output-json artifacts/analysis/truth_p75_capacity_release_wave10.json \
  --output-md artifacts/analysis/truth_p75_capacity_release_wave10.md
```

Adopt Wave10 if facts are stable and explain the majority of p75+ tails. Since Wave10 is facts-only, no positive/truth rerun is required unless solver debug stats integration changes runtime significantly.

### Task 5: Wave11-14 Deferred Gates

Do not implement until Wave10 report is reviewed.

- Wave11 soft target template: consume fixed/random/存4北 pressure facts, still no hard assignment.
- Wave12 beam diversity: only bucket states using Wave10/11 facts.
- Wave13 natural block commitment: only score blocks whose target template and capacity release facts agree.
- Wave14 cross-window compressor: only after solver creates fewer but still locally redundant staging chains.

---

## 2026-04-25 Execution Update

Wave10 was committed as `e1db117 feat: add capacity release facts`.

Fresh focused verification:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_capacity_release.py tests/solver/test_astar_solver.py tests/solver/test_structural_metrics.py tests/solver/test_move_generator.py
```

Result: `115 passed, 5 skipped`.

### Wave11-14 Implementation Strategy

The earlier route-aware and lifecycle experiments showed that local signals are valid but unstable when injected as unconditional high-order sorting rules. The next implementation therefore uses a strict layering rule:

- Facts first: compute reusable state facts without changing behavior.
- Safe low-tier scoring second: consume facts only after hook count, heuristic, purity, and hard blocker signals have already decided the important ordering.
- Diversity only as a bounded reserve: add a small number of intent-preserving candidates instead of globally resorting the beam.
- Compressor last: only after generated plans show repeated cross-window staging chains that the solver itself cannot avoid.

### Wave11: Soft Target Template

Goal: give random/multi-target vehicles a deterministic but non-binding target layout hint.

Facts:

- For `大库:RANDOM`, rank each feasible depot inner track by business preference, currently allocatable spot count, current occupancy length, and capacity release pressure.
- For `存4北`, summarize unresolved close-door vehicles and the minimum ordinary vehicle prefix needed before close-door final placement.
- Do not hard-assign random vehicles and do not remove legal fallback tracks.

Adoption gate:

- Unit tests prove the template prefers feasible low-pressure tracks and emits `存4北` close-door demand.
- The template remains off the main solver debug path. Use `scripts/analyze_soft_target_truth_tails.py` for analysis so full validation artifacts and solve budgets are not affected.
- No behavior change until scoring tests and full validation are ready.

### Wave12: Capacity/Template Tie-Breaker And Beam Intent Diversity

Goal: reduce long-tail exploration churn without changing feasibility.

Tie-breaker:

- Add an action-level structural score that rewards moves that reduce capacity release pressure or place random/close-door vehicles according to the soft template.
- Insert it below primary search keys. In beam priority, keep the existing blocker reserve semantics intact; `_prune_queue` currently depends on `priority[-2]` being `-blocker_bonus`, so either insert the new key before that slot or make blocker detection explicit first.

Diversity:

- Keep a bounded reserve by intent only after the baseline priority order is computed.
- Candidate intents: direct goal delivery, capacity release, soft-template delivery, blocker clearance, low-conflict staging.
- Do not expand beam width globally unless validation shows it is necessary.

Adoption gate:

- Unit tests prove priority ordering is unchanged for primary keys and only differs on exact ties.
- Positive and truth full validation must show no solved-count regression and no p75/p90/p95/max regression large enough to offset wins.

Execution result:

- Implemented a low-tier state tie-breaker from capacity/template facts, then rejected it.
- Positive full validation regressed from Wave5 baseline `64/64 min=2 p50=10 p75=26 p90=82 p95=86 max=129` to `64/64 min=2 p50=10 p75=26 p90=103 p95=103 max=156`.
- The behavior layer was fully removed. Do not reintroduce a scalar structural tie-breaker without a stronger accept/reject guard.

### Wave13: Natural Block Commitment

Goal: stop rehandling already-formed target/region blocks.

Signal:

- Detect contiguous natural blocks with the same effective target family, already satisfied or close to a template target.
- Penalize repeated touches, splitting natural blocks, and removing fully or nearly satisfied blocks.
- Keep it as a soft cost; never freeze moves when capacity release facts say the block must be moved.

Adoption gate:

- Unit tests cover same-target block preservation, capacity-pressure override, and no penalty for necessary blocker clearance.
- Full validation must improve high-touch tail metrics without reducing solved count.

Execution decision:

- Not implemented as a behavior change in this pass. The Wave12 scalar tie-breaker regression shows that even low-tier global ordering can move truth long-tail cases between constructive and beam fallback paths.
- Next safe version should be verifier/runner guarded: generate incumbent and one block-commitment variant, compare full verified hook count and plan-shape metrics, and adopt only if verified shorter or equal-hook with lower max touch count.

### Wave14: Cross-Window Compressor

Goal: reduce staging-chain leftovers only after solver-side changes stabilize.

Scope:

- Identify repeated movement of the same vehicle set across staging tracks.
- Try verifier-guarded rewrites that collapse staging chains into one attach plus final detaches.
- Keep full verifier guard and reject any rewrite that changes final state or violates hook replay.

Adoption gate:

- Run only when Wave11-13 still leave high `staging_to_staging_hook_count` or high `max_vehicle_touch_count`.
- Positive/truth validation must be identical in solved cases and improve hook distribution.

Execution decision:

- Not implemented in this pass. Since Wave13 behavior was not adopted, Wave14 lacks a stable new solver output to compress.
- Keep Wave14 as the next candidate only for verified plans with high `staging_to_staging_hook_count` after a non-regressing solver-side change.

### Final State Of This Pass

Kept:

- Wave11 soft target template facts in `src/fzed_shunting/solver/soft_target_template.py`.
- Wave11 tests in `tests/solver/test_soft_target_template.py`.
- Tail analysis script `scripts/analyze_soft_target_truth_tails.py`.
- Analysis artifacts:
  - `artifacts/analysis/truth_p75_soft_target_wave11.json`
  - `artifacts/analysis/truth_p75_soft_target_wave11.md`

Rejected / reverted:

- Wave12 capacity/template scoring in search priority.
- Wave12 capacity/template scoring in constructive scoring.
- Automatic `initial_soft_target_template` injection into solver debug stats.
- Wave13/14 behavior implementation, because the validation gate did not justify further behavior changes.

Verification:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_soft_target_truth_tails.py \
  --summary artifacts/validation_inputs_truth_compressor_wave5/summary.json \
  --input-dir data/validation_inputs/truth \
  --master-dir data/master \
  --output-json artifacts/analysis/truth_p75_soft_target_wave11.json \
  --output-md artifacts/analysis/truth_p75_soft_target_wave11.md

PYTHONPATH=src .venv/bin/pytest -q \
  tests/solver/test_soft_target_template.py \
  tests/solver/test_astar_solver.py::test_simple_astar_result_can_return_debug_stats
```

Result: `3 passed`.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py \
  --input-dir data/validation_inputs/positive \
  --output-dir artifacts/validation_inputs_positive_wave11_facts_no_debug \
  --master-dir data/master \
  --solver beam \
  --beam-width 8 \
  --max-workers 8 \
  --timeout-seconds 60
```

Result: `64/64`, hook distribution `min=2 p50=10 p75=26 p90=86 p95=86 max=129`, no `initial_soft_target_template` in debug stats.
