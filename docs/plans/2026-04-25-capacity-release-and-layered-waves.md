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
