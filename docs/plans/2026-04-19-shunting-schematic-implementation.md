# Shunting Schematic Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current replay visualization with a metro-style schematic that keeps orientation cues, separates hook details from the main canvas, and moves distance catalog content out of the main view.

**Architecture:** Add a fixed schematic layout JSON plus a dedicated loader/renderer module under `src/fzed_shunting/demo/`. Keep solver and replay data unchanged. Refactor `app.py` so the replay step page consumes the same view model but renders into a schematic canvas, a hook-detail sidebar, and bottom detail tabs. Keep segmented physical-route generation as a separate distance-focused view.

**Tech Stack:** Python, Streamlit, SVG, Pydantic, pytest

---

### Task 1: Add the fixed schematic layout source

**Files:**
- Create: `data/master/schematic_layout.json`
- Create: `src/fzed_shunting/demo/schematic.py`
- Test: `tests/demo/test_topology_layout.py`

**Step 1: Write the failing test**

Add a test that loads the schematic layout and asserts:

- `联6` is on the main trunk
- area labels include `存车区` and `检修区`
- key branch endpoints such as `机库` and `洗南` exist

**Step 2: Run test to verify it fails**

Run: `pytest tests/demo/test_topology_layout.py -q`

Expected: FAIL because schematic loader/layout does not exist yet.

**Step 3: Write minimal implementation**

- Create `schematic_layout.json`
- Add `src/fzed_shunting/demo/schematic.py` with models and loader

**Step 4: Run test to verify it passes**

Run: `pytest tests/demo/test_topology_layout.py -q`

Expected: PASS

### Task 2: Replace the replay SVG with the schematic renderer

**Files:**
- Modify: `app.py`
- Test: `tests/demo/test_app_topology.py`

**Step 1: Write the failing test**

Add test coverage asserting `_build_topology_svg(...)` now:

- emits schematic area labels such as `存车区`
- does not emit the old background image class
- still renders route highlight and moving marker

**Step 2: Run test to verify it fails**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: FAIL because current SVG still uses the old background image renderer.

**Step 3: Write minimal implementation**

- Load schematic layout in `app.py`
- Build route motion path from the schematic points
- Render base tracks, area blocks, active path, changed tracks, source/target badges, and loco marker

**Step 4: Run test to verify it passes**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: PASS

### Task 3: Restructure the step UI into left canvas, right sidebar, bottom tabs

**Files:**
- Modify: `app.py`
- Test: `tests/demo/test_app_topology.py`

**Step 1: Write the failing test**

Add a test for new helper functions that build:

- hook sidebar rows
- current-state rows
- distance breakdown rows

**Step 2: Run test to verify it fails**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: FAIL because those helpers do not exist yet.

**Step 3: Write minimal implementation**

- Split `_render_step(...)` into helper-driven sections
- Use `st.columns([7, 3])`
- Move track status, vehicle details, verifier summary, and distance breakdown into tabs

**Step 4: Run test to verify it passes**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: PASS

### Task 4: Add top-level tabs for replay, overview, and distance catalog

**Files:**
- Modify: `app.py`
- Test: `tests/demo/test_app_topology.py`
- Optional reference: `src/fzed_shunting/tools/segmented_routes_svg.py`

**Step 1: Write the failing test**

Add a pure helper test asserting:

- overview content can be built without step state
- distance catalog rows are generated from `segmented_physical_routes.json`

**Step 2: Run test to verify it fails**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: FAIL because overview and distance helpers do not exist yet.

**Step 3: Write minimal implementation**

- Add top-level Streamlit tabs
- Create overview renderer using the schematic SVG without active path emphasis
- Create distance catalog table rows from segmented routes

**Step 4: Run test to verify it passes**

Run: `pytest tests/demo/test_app_topology.py -q`

Expected: PASS

### Task 5: Regression-check the segmented route experiment remains a distance view

**Files:**
- Modify: `tests/tools/test_segmented_routes_svg.py`
- Optional modify: `src/fzed_shunting/tools/segmented_routes_svg.py`

**Step 1: Write the failing test**

Add/adjust test coverage that the segmented route SVG still:

- exports route rows for all aggregated branches
- remains a catalog-style distance artifact
- is no longer assumed to be the main replay view

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py -q`

Expected: FAIL if the older assumptions conflict with the new structure.

**Step 3: Write minimal implementation**

Update tests and only touch implementation if needed to keep the distance artifact stable.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py -q`

Expected: PASS

### Task 6: Final verification

**Files:**
- Verify only

**Step 1: Run targeted verification**

Run: `pytest tests/demo/test_topology_layout.py tests/demo/test_app_topology.py tests/tools/test_segmented_routes_svg.py -q`

Expected: all selected tests pass

**Step 2: Run app smoke check**

Run: `.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8501`

Expected: Streamlit starts without import/runtime errors

**Step 3: Inspect key requirements**

Checklist:

- replay view uses schematic main canvas
- current hook details are in a right sidebar
- bottom tabs separate status/details/validation/distance
- overview tab shows stable full-yard schematic
- distance catalog is no longer embedded in the main replay canvas
