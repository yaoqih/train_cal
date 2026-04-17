# Search Space Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce solver search waste and false negatives without changing business rules or making the strategy overly complex.

**Architecture:** Keep the current solver structure and improve only three levers with clear evidence: deduplicate equivalent states, generate the largest feasible blocking-clear prefix instead of only the largest valid prefix, and reuse route-oracle state across node expansions. This keeps the search model simple while lowering duplicate states, dead-end misses, and per-node routing overhead.

**Tech Stack:** Python, pytest, existing `fzed_shunting` solver/domain modules.

---

### Task 1: Lock State-Key Dedup Behavior

**Files:**
- Modify: `tests/solver/test_astar_solver.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`

**Step 1: Write the failing test**

Add a test asserting two states with identical track layout / weigh / spot state but different `loco_track_name` produce the same `_state_key`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/solver/test_astar_solver.py::test_state_key_ignores_loco_track_position -q`

Expected: FAIL because `_state_key` currently includes `loco_track_name`.

**Step 3: Write minimal implementation**

Remove `loco_track_name` from `_state_key`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/solver/test_astar_solver.py::test_state_key_ignores_loco_track_position -q`

Expected: PASS.

### Task 2: Lock Blocking-Clear Fallback Behavior

**Files:**
- Modify: `tests/solver/test_move_generator.py`
- Modify: `src/fzed_shunting/solver/move_generator.py`

**Step 1: Write the failing test**

Add a test where the largest valid interfering prefix does not fit on any temporary track, but a smaller prefix does. Assert `generate_goal_moves()` emits a staging move for the smaller feasible prefix.

**Step 2: Run test to verify it fails**

Run: `pytest tests/solver/test_move_generator.py::test_generate_goal_moves_falls_back_to_smaller_interfering_prefix_when_largest_block_cannot_stage -q`

Expected: FAIL because the generator currently only asks for the largest valid prefix.

**Step 3: Write minimal implementation**

Change interfering-track staging to try descending valid prefix sizes and stop once at least one feasible prefix yields staging targets.

**Step 4: Run test to verify it passes**

Run: `pytest tests/solver/test_move_generator.py::test_generate_goal_moves_falls_back_to_smaller_interfering_prefix_when_largest_block_cannot_stage -q`

Expected: PASS.

### Task 3: Lock RouteOracle Reuse Path

**Files:**
- Modify: `tests/solver/test_move_generator.py`
- Modify: `src/fzed_shunting/solver/move_generator.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`

**Step 1: Write the failing test**

Add a test that passes an injected `RouteOracle` into `generate_goal_moves()` and asserts the function uses it instead of constructing a new one.

**Step 2: Run test to verify it fails**

Run: `pytest tests/solver/test_move_generator.py::test_generate_goal_moves_reuses_injected_route_oracle -q`

Expected: FAIL because the function currently always constructs a new oracle.

**Step 3: Write minimal implementation**

Accept an optional `route_oracle` parameter in `generate_goal_moves()` and pass one shared instance from the solver search loop.

**Step 4: Run test to verify it passes**

Run: `pytest tests/solver/test_move_generator.py::test_generate_goal_moves_reuses_injected_route_oracle -q`

Expected: PASS.

### Task 4: Run Regression and Batch Comparison

**Files:**
- No code changes required unless regressions appear

**Step 1: Run focused solver tests**

Run: `pytest tests/solver/test_move_generator.py tests/solver/test_astar_solver.py tests/solver/test_external_validation_parallel_runs.py -q`

Expected: PASS.

**Step 2: Run targeted scenario checks**

Run the previously failing sample scenarios (`validation_20260113W.json`, `validation_20260206W.json`, `validation_20260121Z.json`) with worker mode and compare status / expansions.

**Step 3: Run full 109-case batch**

Run: `python scripts/run_external_validation_parallel.py --solver beam --beam-width 8 --heuristic-weight 1.0 --max-workers 8 --timeout-seconds 60 --output-dir artifacts/external_validation_parallel_runs/<new_run_name>`

Expected: fresh summary for comparison against the baseline.

**Step 4: Evaluate next iteration**

Compare solved / timeout / no-solution counts and pick the next minimal change only if the evidence justifies it.
