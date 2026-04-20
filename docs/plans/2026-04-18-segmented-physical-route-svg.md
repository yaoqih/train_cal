# Segmented Physical Route SVG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an experiment-first JSON representation for abbreviated segmented physical routes and export a simplified horizontal-ribbon straight-line continuous-network SVG with fixed left/right anchors, ordered corridor constraints, and log-scaled display span, without touching the Streamlit demo path.

**Architecture:** Keep `physical_routes.json`, `tracks.json`, and the two markdown business docs as source inputs. Add a new segmented-route JSON artifact keyed by aggregate branch codes, build a node-edge graph from the current master data, then overlay a small number of ordered-corridor constraints derived from the docs so serial relationships like `机棚 -> 机北` are preserved. The x-axis follows topology depth and stays fixed, the y-axis only spreads branches apart, display span for long routes uses `log1p` compression instead of linear growth, and each track is rendered as one continuous straight segment between two shared node positions.

**Tech Stack:** Python 3.11, Pydantic, pytest, existing `fzed_shunting` data/master-data utilities.

---

### Task 1: Lock The Segmented Route Contract

**Files:**
- Create: `tests/tools/test_segmented_routes_svg.py`
- Modify: `data/master/segmented_physical_routes.json`

**Step 1: Write the failing test**

Add tests asserting:

- `L2-L12` splits into `存5北`, `存5南`
- `Z1-L8` splits into `机北`, `机棚`
- `L19-修1尽头` splits into `修1库外`, `修1库内`
- `segments[*].track_code` all exist in `tracks.json`
- segment distance sum equals aggregate distance within a small tolerance

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_segmented_routes_cover_expected_multi_segment_branches -q`

Expected: FAIL because the segmented-route JSON or loader does not exist yet.

**Step 3: Write minimal implementation**

Create `data/master/segmented_physical_routes.json` with the approved branch-to-short-name mapping and physical-distance allocations for the split branches.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_segmented_routes_cover_expected_multi_segment_branches -q`

Expected: PASS.

### Task 2: Add Failing Tests For Ordered Corridor And Log Width

**Files:**
- Modify: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Write the failing test**

Add tests asserting:

- `联6` renders to the right of `修1库内`
- repair tracks stay on the left side of non-repair tracks
- connected tracks share the same node coordinates at junctions
- the overall node layout is much wider than it is tall
- a chosen main chain keeps increasing x from left to right
- `机棚` stays between `L8` and `机北`, and `机北` stays between `机棚` and `Z1`
- long branches do not inflate width linearly; width growth follows a compressed `log1p` rule
- SVG paths use straight lines and no cubic-curve commands
- canvas width grows enough to avoid horizontal compression

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_build_auto_layout_keeps_anchor_tracks_on_expected_sides -q`

Expected: FAIL because the current SVG exporter still depends on manual geometry output.

**Step 3: Write minimal implementation**

Expose a continuous-network layout builder that computes horizontal-ribbon node positions from the real topology graph with fixed x layers, ordered corridor overrides, and y-only repulsion.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_build_auto_layout_keeps_anchor_tracks_on_expected_sides -q`

Expected: PASS.

### Task 3: Add A Minimal Loader And SVG Exporter

**Files:**
- Create: `src/fzed_shunting/tools/segmented_routes_svg.py`
- Create: `src/fzed_shunting/tools/__init__.py` if import exposure needs adjustment
- Modify: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Write the failing test**

Add a test asserting the exporter returns SVG that includes:

- root `<svg`
- polyline/path content
- key labels such as `存5北`, `机棚`, `修1库内`
- at least one aggregate distance label such as `626.3m`

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_render_segmented_routes_svg_contains_expected_labels_and_paths -q`

Expected: FAIL because the exporter does not exist yet.

**Step 3: Write minimal implementation**

Implement a small loader and renderer that:

- reads `segmented_physical_routes.json`
- reads `tracks.json`
- checks the markdown docs for the approved ordered corridor relationships
- builds a shared-node continuous track network
- lays nodes out as a horizontal ribbon instead of a near-rectangular field
- grows canvas width from topology depth
- compresses overlong display span with `log1p`
- renders one SVG path per unique `track_code`
- emits aggregate route legend entries for split routes

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_render_segmented_routes_svg_contains_expected_labels_and_paths -q`

Expected: PASS.

### Task 4: Add Artifact Export And CLI Entry Path

**Files:**
- Modify: `src/fzed_shunting/tools/segmented_routes_svg.py`
- Modify: `tests/tools/test_segmented_routes_svg.py`

**Step 1: Write the failing test**

Add a test that calls the module entrypoint to export JSON + SVG into a temp directory and asserts both files are created and non-empty.

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_export_segmented_route_artifacts_writes_json_and_svg -q`

Expected: FAIL because the export entrypoint does not exist yet.

**Step 3: Write minimal implementation**

Add a CLI-friendly `main()` that writes:

- `segmented_physical_routes.json`
- `segmented_physical_routes.svg`

to a requested output directory.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_segmented_routes_svg.py::test_export_segmented_route_artifacts_writes_json_and_svg -q`

Expected: PASS.

### Task 5: Run Focused Verification

**Files:**
- No additional code changes unless regressions appear

**Step 1: Run focused test file**

Run: `pytest tests/tools/test_segmented_routes_svg.py -q`

Expected: PASS.

**Step 2: Run adjacent regression tests**

Run: `pytest tests/domain/test_master_data.py tests/demo/test_topology_layout.py -q`

Expected: PASS.

**Step 3: Generate the experiment artifacts**

Run: `python -m fzed_shunting.tools.segmented_routes_svg --output-dir artifacts/segmented_routes_experiment`

Expected: writes a JSON and SVG artifact pair for manual inspection.

**Step 4: Record integration boundary**

Summarize that the experiment is verified and still independent from `app.py`, so the next step can be a deliberate Streamlit integration rather than a mixed refactor.
