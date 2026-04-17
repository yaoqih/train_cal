# Monthly External Validation Hard Cutover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hard cut `convert_external_validation_inputs.py` over to convert every workbook under `取送车计划/1月-取送车计划` through `取送车计划/3月-取送车计划` into external validation JSON.

**Architecture:** Treat each monthly plan workbook as one start/end pair source. Read its own two worksheets, keep only vehicles shared between the start and end sheets, derive external validation payloads with existing alias and repair normalization, and emit one JSON plus one per-workbook summary row. Remove the old fixed four-scenario template contract.

**Tech Stack:** Python, stdlib `zipfile` and `xml.etree.ElementTree`, pytest, existing `fzed_shunting` normalization/master-data loaders.

---

### Task 1: Lock the new hard-cut contract with tests

**Files:**
- Modify: `tests/tools/test_convert_external_validation_inputs.py`

**Step 1: Write the failing tests**

Add coverage for:

- monthly workbook discovery across `1月-取送车计划` to `3月-取送车计划`
- per-workbook conversion using the workbook's own two sheets
- summary payload fields for workbook path, sheet names, scenario file, and scenario count

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: FAIL because the converter still uses the old fixed-template contract.

### Task 2: Implement the hard cutover

**Files:**
- Modify: `src/fzed_shunting/tools/convert_external_validation_inputs.py`

**Step 1: Write minimal implementation**

Implement:

- monthly workbook discovery under `取送车计划/*月-取送车计划`
- per-workbook sheet pairing from the workbook's own worksheets
- scenario naming and summary metadata derived from workbook filenames
- removal of the old fixed template entrypoint contract

**Step 2: Run focused tests**

Run: `pytest tests/tools/test_convert_external_validation_inputs.py -q`
Expected: PASS

### Task 3: Verify generated artifacts

**Files:**
- Verify: `artifacts/external_validation_inputs/*.json`

**Step 1: Run the converter**

Run: `python -m fzed_shunting.tools.convert_external_validation_inputs --output-dir artifacts/external_validation_inputs`
Expected: one JSON per monthly plan workbook and a summary JSON.

**Step 2: Spot-check compatibility**

Run targeted solver/normalization checks on a few generated scenarios to confirm the payload contract is still valid.
