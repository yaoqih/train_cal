# Right-Middle SVG Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tighten the right-middle SVG cluster so the labels and short connectors feel cleaner without changing the network topology.

**Architecture:** Keep the current continuous network layout and adjust only local label anchor biases, baseline overrides, and, only if necessary, tiny connector presentation offsets. Protect the intended polish with focused regression tests on label gaps and label-to-track clearance.

**Tech Stack:** Python, pytest, SVG generation in `src/fzed_shunting/tools/segmented_routes_svg.py`

---

### Task 1: Add failing right-middle polish tests

**Files:**
- Modify: `tests/tools/test_segmented_routes_svg.py`
- Test: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Write the failing test**

Add focused assertions for:
- `存1` label box gap from `临2`
- `渡5` label clearance from `渡4`
- `机北` label clearance from `渡7`

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py -q`
Expected: FAIL on the new right-middle polish assertions.

### Task 2: Implement minimal local layout tweaks

**Files:**
- Modify: `src/fzed_shunting/tools/segmented_routes_svg.py`
- Test: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Write minimal implementation**

Tune only the right-middle cluster:
- `TRACK_TEXT_ANCHOR_BIASES`
- `TRACK_TEXT_BASELINE_OVERRIDE_OFFSETS`
- avoid broader lane or topology changes

**Step 2: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py -q`
Expected: PASS with no new label overlap regressions.

### Task 3: Verify and export artifacts

**Files:**
- Modify: `src/fzed_shunting/tools/segmented_routes_svg.py`
- Test: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Run related verification**

Run: `pytest tests/tools/test_segmented_routes_svg.py tests/domain/test_master_data.py tests/demo/test_topology_layout.py -q`
Expected: PASS

**Step 2: Regenerate artifacts**

Run:
- `python -m fzed_shunting.tools.segmented_routes_svg --output-dir artifacts/segmented_routes_experiment`
- `rsvg-convert artifacts/segmented_routes_experiment/segmented_physical_routes.svg -o artifacts/segmented_routes_experiment/segmented_physical_routes.png`

Expected: updated SVG and PNG in `artifacts/segmented_routes_experiment/`
