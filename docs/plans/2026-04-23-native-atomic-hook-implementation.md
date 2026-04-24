# Native Atomic Hook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hard-cut the solver stack to native atomic-hook optimization so default solve modes optimize `ATTACH/DETACH` count directly instead of legacy PUT moves.

**Architecture:** Keep the existing solver layers (`search`, `constructive`, `anytime`, `lns`, `verify`, `demo`) but switch their optimization truth source to native state/action semantics. Preserve replay/verify support for imported PUT plans only as a compatibility reader, not as an optimization path.

**Tech Stack:** Python, pytest, Pydantic, repo-local solver/replay/verification modules

---

### Task 1: Unify solver modes onto native action semantics

**Files:**
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Modify: `src/fzed_shunting/solver/search.py`
- Modify: `src/fzed_shunting/solver/heuristic.py`
- Modify: `src/fzed_shunting/solver/types.py`
- Test: `tests/solver/test_astar_solver.py`
- Test: `tests/solver/test_heuristic.py`

**Step 1:** Add failing tests that assert default solver paths return native `ATTACH/DETACH` actions and atomic hook counts for simple cases.

**Step 2:** Route `exact/weighted/beam/lns` through native move generation, native cost accounting, and native heuristic selection.

**Step 3:** Collapse `real_hook` into an alias/entry for the same native stack instead of a special fallback branch.

**Step 4:** Tighten `HookAction` construction so optimization-generated actions are explicit native actions.

**Step 5:** Run the targeted solver/heuristic tests.

### Task 2: Native-first constructive and warm-start path

**Files:**
- Modify: `src/fzed_shunting/solver/constructive.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Test: `tests/solver/test_constructive.py`
- Test: `tests/solver/test_astar_solver.py`

**Step 1:** Add failing tests that constructive/warm-start return native `ATTACH/DETACH` plans.

**Step 2:** Make constructive use native move generation and native heuristic scoring.

**Step 3:** Make warm-start completion use native heuristic and native exact completion.

**Step 4:** Run the targeted constructive tests.

### Task 3: Native-first anytime fallback and LNS repair

**Files:**
- Modify: `src/fzed_shunting/solver/anytime.py`
- Modify: `src/fzed_shunting/solver/lns.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Test: `tests/solver/test_anytime_integration.py`
- Test: `tests/solver/test_budget_fallback.py`
- Test: `tests/solver/test_astar_solver.py`

**Step 1:** Add failing tests that fallback/repair stages stay in native action space.

**Step 2:** Make anytime stages invoke native search modes instead of re-entering legacy PUT semantics.

**Step 3:** Keep LNS destroy/repair generic but ensure seed/repair searches are native and compare native hook counts.

**Step 4:** Run the targeted anytime/LNS tests.

### Task 4: Sync replay/demo/CLI/workflow-facing semantics

**Files:**
- Modify: `src/fzed_shunting/demo/view_model.py`
- Modify: `src/fzed_shunting/cli.py`
- Modify: `src/fzed_shunting/workflow/runner.py`
- Test: `tests/demo/test_view_model.py`
- Test: `tests/workflow/test_runner.py`
- Test: `tests/io/test_cli_flow.py`

**Step 1:** Update failing assertions that still assume one PUT equals one hook.

**Step 2:** Ensure exported plans from default solve flows are native actions and hook counts reflect atomic operations.

**Step 3:** Run demo/workflow/CLI regression tests.

### Task 5: Clean residual legacy optimization assumptions

**Files:**
- Modify: `src/fzed_shunting/solver/real_hook_compiler.py`
- Modify: `docs/plans/2026-04-23-native-atomic-hook-optimization-design.md`
- Test: residual affected suites

**Step 1:** Remove or demote optimizer-only legacy artifacts that no longer belong in the native optimization path.

**Step 2:** Update the design doc to reflect what was actually implemented and what remains intentionally import-only.

**Step 3:** Run the consolidated regression command for touched areas.
