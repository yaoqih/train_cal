# XLSX Diff Validation Inputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert `标准化起点终点模板（9.4-9.8-9.9）.xlsx` plus `段内车型换长.xlsx` into four single-stage validation inputs that this repository can solve, using only vehicles present in both paired sheets.

**Architecture:** Add a one-off conversion module that parses `.xlsx` files without adding a new dependency, normalizes Excel aliases into repo track codes, derives target tracks from end-sheet positions, and emits four scenario JSON files plus a conversion summary. Keep solver behavior unchanged; the new code only prepares external validation data and verifies it against the existing input contract.

**Tech Stack:** Python, stdlib `zipfile`/`xml.etree.ElementTree`, pytest, existing `fzed_shunting` normalization and CLI solver.

---

### Task 1: Lock the conversion contract with tests

**Files:**
- Create: `tests/tools/test_convert_external_validation_inputs.py`

**Step 1: Write the failing test**

Add tests that cover:

- Excel track aliases mapping to repo track codes.
- `repair` values `段/厂/临/空/重` splitting into normalized `repairProcess` and `vehicleAttributes`.
- Pair diff generation keeping only shared vehicles and using end-sheet track as `targetTrack`.
- Summary output recording excluded `added_vehicle_nos` / `removed_vehicle_nos`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: FAIL because the conversion module does not exist yet.

**Step 3: Write minimal implementation**

Create the conversion module with only the helpers needed for the tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

### Task 2: Implement the one-off XLSX converter

**Files:**
- Create: `src/fzed_shunting/tools/convert_external_validation_inputs.py`

**Step 1: Expand failing tests**

Add coverage for:

- Parsing workbook sheets from raw `.xlsx`.
- Building `trackInfo` from the union of source/target tracks using master data lengths.
- Producing deterministic output filenames and scenario names.
- Converting enabled vehicle models through the length table.

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: FAIL on the new behaviors.

**Step 3: Write minimal implementation**

Implement:

- `.xlsx` worksheet reader using stdlib XML parsing.
- Alias/repair normalization helpers.
- Scenario builder for the four sheet pairs.
- JSON and summary payload generation entrypoint.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

### Task 3: Generate the four validation inputs and verify repository compatibility

**Files:**
- Create: `artifacts/external_validation_inputs/*.json`
- Create: `artifacts/external_validation_inputs/conversion_summary.json`

**Step 1: Run the converter**

Run: `python -m fzed_shunting.tools.convert_external_validation_inputs --output-dir artifacts/external_validation_inputs`

Expected: four scenario JSON files and one summary JSON are created.

**Step 2: Verify normalization/solver compatibility**

Run targeted validation commands against each generated JSON, at minimum:

- `python -m fzed_shunting.cli solve --input artifacts/external_validation_inputs/<file>.json`

Expected: each file is accepted by the current repo contract; if a scenario is unsolved, the output still proves the input is structurally valid and the summary should retain enough metadata for review.

**Step 3: Record artifacts**

Keep the generated JSON and summary as external validation data only. Do not modify solver logic in response to scenario difficulty.

### Task 4: Final verification

**Files:**
- Verify: `tests/tools/test_convert_external_validation_inputs.py`
- Verify: `artifacts/external_validation_inputs/*.json`

**Step 1: Run focused tests**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

**Step 2: Run artifact compatibility checks**

Run: `python -m fzed_shunting.cli solve --input ...` for all four files.
Expected: command completes for each artifact and outputs solver/verifier status.

**Step 3: Review generated summary**

Confirm the summary lists:

- paired sheet names
- scenario file names
- shared vehicle count
- excluded added vehicle numbers
- excluded removed vehicle numbers
- any models missing from the length table
