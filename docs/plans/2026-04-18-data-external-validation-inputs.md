# Data External Validation Inputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert `取送车计划/Data/*.xlsx` into a separate external-validation artifact set under `artifacts/external_validation_inputs/data/`, without regressing the existing monthly converter.

**Architecture:** Keep the existing monthly-plan converter unchanged at the entrypoint level and add a dedicated `Data` conversion path inside the same tool module. Reuse the existing track/length/scenario builders where possible, but add a `Data`-specific workbook discovery flow, flexible header detection, Start/End parsing, cross-sheet field backfill, and an assumptions summary so the one-off artifact remains auditable.

**Tech Stack:** Python, stdlib `zipfile`/`xml.etree.ElementTree`, pytest, existing `fzed_shunting` normalization and CLI solver.

---

### Task 1: Lock the `Data` workbook contract with failing tests

**Files:**
- Modify: `tests/tools/test_convert_external_validation_inputs.py`

**Step 1: Write the failing test**

Add tests that cover:

- discovering only scenario workbooks from `取送车计划/Data` and excluding `map.xlsx`
- deriving deterministic output names for `Data` scenarios
- mapping old alias names like `调梁库` / `喷漆库` / `洗罐库内` / `洗罐库外` / `洗罐线`
- normalizing richer raw repair values like `段修` / `厂修` / `临修` / `称重`
- backfilling missing model and repair fields from the opposite sheet for shared vehicle numbers

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: FAIL because the `Data` conversion path does not exist yet.

**Step 3: Write minimal implementation**

Implement only the helper functions needed for the new tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

### Task 2: Implement the isolated `Data` converter

**Files:**
- Modify: `src/fzed_shunting/tools/convert_external_validation_inputs.py`
- Modify: `tests/tools/test_convert_external_validation_inputs.py`

**Step 1: Expand the failing test**

Add coverage for:

- flexible Start/End header-row detection
- ignoring extra sheets like `map` and the older Chinese presentation sheets
- generating `conversion_summary.json` in the separate output folder
- recording backfill counts and assumptions for auditability

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: FAIL on the new `Data` conversion behaviors.

**Step 3: Write minimal implementation**

Implement:

- `Data` workbook discovery
- flexible single-sheet vehicle parsing
- shared-vehicle field merge/backfill
- `convert_data_external_validation_inputs(...)`
- a dedicated CLI command for the `Data` conversion path

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

### Task 3: Generate the `Data` artifacts and verify contract compatibility

**Files:**
- Create: `artifacts/external_validation_inputs/data/*.json`
- Create: `artifacts/external_validation_inputs/data/conversion_summary.json`
- Create: `artifacts/external_validation_inputs/data/conversion_assumptions.md`

**Step 1: Run the converter**

Run: `python -m fzed_shunting.tools.convert_external_validation_inputs convert-data --output-dir artifacts/external_validation_inputs/data`

Expected: one JSON per `Data` workbook plus a summary and assumptions file.

**Step 2: Verify structural compatibility**

Run: `python -m fzed_shunting.cli solve --input artifacts/external_validation_inputs/data/<file>.json`

Expected: each generated file is accepted by the current input contract, even if the solver does not find a plan.

**Step 3: Review the assumptions summary**

Confirm the summary documents:

- which workbooks were converted
- which fields were backfilled from the opposite sheet
- which alias mappings were applied
- whether any vehicles were skipped due to unresolved required fields

### Task 4: Final verification

**Files:**
- Verify: `tests/tools/test_convert_external_validation_inputs.py`
- Verify: `artifacts/external_validation_inputs/data/*.json`

**Step 1: Run focused tests**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

**Step 2: Run artifact compatibility checks**

Run the solver on the generated `Data` inputs.
Expected: command completes for every generated file.

**Step 3: Review diffs**

Check that only the dedicated `Data` artifacts and the converter/test/doc files changed.
