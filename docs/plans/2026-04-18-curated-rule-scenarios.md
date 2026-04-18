# Curated Rule-Scenario Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `data/validation_inputs/` with 5 baseline yard files plus 91 hand-authored rule-scenario cases (64 positive / 27 negative) covering every rule branch in `福州东调车业务说明.md` chapters 3–8.

**Architecture:** JSON data artifacts only. No changes to the solver, the external-validation runner, or any existing code. Cases are authored by copying one of five baseline JSON files and applying rule-specific mutations to `vehicleInfo`. A single pytest structural validator runs after each batch to catch shape regressions early.

**Tech Stack:** Python 3.11 stdlib `json`, pytest, existing fzed-shunting repo layout.

---

## Authoring Conventions

All case files must conform to the shapes in the design spec (`docs/plans/2026-04-18-curated-rule-scenarios-design.md`, §6). The engineer uses these templates verbatim:

### Positive case template

```jsonc
{
  "metadata": {
    "spec_section": "3.2",
    "rule_id": "shed_work",
    "variant": "tanker",
    "baseline": "B5_busy_shed",
    "purpose": "调棚 WORK with 罐车 must route to 调棚:WORK area",
    "expected_bounds": {
      "must_be_solvable": true,
      "max_hook_count": null,
      "must_visit_tracks": ["调棚"],
      "close_door_constraint": null,
      "weigh_required_vehicles": []
    }
  },
  "trackInfo":     [ /* paste trackInfo array from the chosen baseline */ ],
  "vehicleInfo":   [ /* baseline vehicleInfo with mutations per case */ ],
  "locoTrackName": "机库"
}
```

### Negative case template

```jsonc
{
  "metadata": {
    "spec_section": "3.3",
    "rule_id": "isSpotting_on_storage_line",
    "variant": "cun1",
    "baseline": "B1_clean",
    "purpose": "isSpotting=是 must be rejected when targetTrack is a regular storage line",
    "expected_error": {
      "category": "illegal_input",
      "rule_citation": "spec 3.3 compat rule 2",
      "must_be_rejected_at": "input_normalization"
    }
  },
  "trackInfo":     [ /* from baseline */ ],
  "vehicleInfo":   [ /* includes the offending record */ ],
  "locoTrackName": "机库"
}
```

`expected_error.category` must be one of: `illegal_input`, `capacity_overflow`, `tow_limit_exceeded`, `close_door_violation`, `yard_allocation_violation`.

`expected_bounds.close_door_constraint`, when present, is a list whose entries are
`{"vehicle_no": "CD001", "forbidden_positions": [1, 2, 3], "target_track": "存4北"}` or similar.
Use `null` when no close-door constraint applies. Always a list or null; never a bare dict.

Naming: `case_<chapter>_<rule-slug>_<variant>.json`, under `positive/` or `negative/`.

---

## File Structure

```
data/validation_inputs/
├── README.md                              # Task 30
├── _baselines/
│   ├── B1_clean.json                      # Task 3
│   ├── B2_busy_storage.json               # Task 4
│   ├── B3_busy_yard_normal.json           # Task 5
│   ├── B4_busy_yard_inspection.json       # Task 6
│   └── B5_busy_shed.json                  # Task 7
├── positive/                              # Tasks 9-28
│   └── case_<ch>_<rule>_<variant>.json
└── negative/                              # Tasks 9-28
    └── case_<ch>_<rule>_<variant>.json

tests/data/
└── test_validation_inputs_structure.py    # Task 2
```

The validator in `tests/data/` owns all structural invariants. No other changes to `src/` or `tests/` trees.

---

## Task 1: Create the directory scaffold

**Files:**
- Create: `data/validation_inputs/_baselines/.gitkeep`
- Create: `data/validation_inputs/positive/.gitkeep`
- Create: `data/validation_inputs/negative/.gitkeep`

- [ ] **Step 1: Create empty directories via `.gitkeep`**

```bash
mkdir -p data/validation_inputs/_baselines data/validation_inputs/positive data/validation_inputs/negative
: > data/validation_inputs/_baselines/.gitkeep
: > data/validation_inputs/positive/.gitkeep
: > data/validation_inputs/negative/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add data/validation_inputs
git commit -m "Scaffold data/validation_inputs directory tree"
```

---

## Task 2: Structural validator (TDD)

Validator guards: (a) JSON loads, (b) required metadata fields present, (c) `metadata.baseline` is one of B1..B5, (d) `metadata.spec_section` follows `NN.NN`, (e) positive cases have `expected_bounds`, negative cases have `expected_error`, (f) `vehicleInfo[].trackName` appears in that file's `trackInfo` **unless** the case is under `negative/` with `expected_error.category=illegal_input` AND `rule_id` starts with `isSpotting_` or `走行线_as_source` (allowed to reference missing tracks as part of the bug being tested — explicitly whitelisted via `metadata.allow_unknown_trackname: true`), (g) no duplicate `(spec_section, rule_id, variant)` tuple across files, (h) file name matches `case_<spec_section_with_underscore>_<rule_id>_<variant>(_neg)?.json` pattern.

**Files:**
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_validation_inputs_structure.py`

- [ ] **Step 1: Create `tests/data/__init__.py` as empty file**

```bash
: > tests/data/__init__.py
```

- [ ] **Step 2: Write the failing structural validator test**

Create `tests/data/test_validation_inputs_structure.py`:

```python
"""Structural invariants for data/validation_inputs rule-scenario library."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "data" / "validation_inputs"
BASELINE_DIR = CORPUS_ROOT / "_baselines"
POSITIVE_DIR = CORPUS_ROOT / "positive"
NEGATIVE_DIR = CORPUS_ROOT / "negative"

ALLOWED_BASELINES = {
    "B1_clean",
    "B2_busy_storage",
    "B3_busy_yard_normal",
    "B4_busy_yard_inspection",
    "B5_busy_shed",
}

ALLOWED_ERROR_CATEGORIES = {
    "illegal_input",
    "capacity_overflow",
    "tow_limit_exceeded",
    "close_door_violation",
    "yard_allocation_violation",
}

FILENAME_RE = re.compile(
    r"^case_(\d+_\d+(?:_\d+)?)_([a-z0-9_]+)(?:_(\w+))?\.json$"
)
SPEC_SECTION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")


def _iter_cases(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("case_*.json"))


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("folder", [POSITIVE_DIR, NEGATIVE_DIR])
def test_case_filename_shape(folder: Path):
    for path in _iter_cases(folder):
        assert FILENAME_RE.match(path.name), (
            f"bad filename: {path.name}"
        )


@pytest.mark.parametrize("folder", [POSITIVE_DIR, NEGATIVE_DIR])
def test_case_metadata_shape(folder: Path):
    for path in _iter_cases(folder):
        payload = _load(path)
        assert isinstance(payload, dict), path
        meta = payload.get("metadata")
        assert isinstance(meta, dict), f"missing metadata in {path.name}"
        for field in ("spec_section", "rule_id", "variant", "baseline", "purpose"):
            assert field in meta, f"{path.name} missing metadata.{field}"
        assert SPEC_SECTION_RE.match(meta["spec_section"]), (
            f"{path.name} spec_section {meta['spec_section']} not NN.NN"
        )
        assert meta["baseline"] in ALLOWED_BASELINES, (
            f"{path.name} references unknown baseline {meta['baseline']!r}"
        )


def test_positive_cases_have_expected_bounds():
    for path in _iter_cases(POSITIVE_DIR):
        meta = _load(path)["metadata"]
        assert "expected_bounds" in meta, (
            f"{path.name} missing metadata.expected_bounds"
        )
        assert "expected_error" not in meta, (
            f"{path.name} positive case must not have expected_error"
        )


def test_negative_cases_have_expected_error():
    for path in _iter_cases(NEGATIVE_DIR):
        meta = _load(path)["metadata"]
        assert "expected_error" in meta, (
            f"{path.name} missing metadata.expected_error"
        )
        err = meta["expected_error"]
        assert err.get("category") in ALLOWED_ERROR_CATEGORIES, (
            f"{path.name} unknown expected_error.category {err.get('category')!r}"
        )
        assert "expected_bounds" not in meta, (
            f"{path.name} negative case must not have expected_bounds"
        )


def test_vehicle_track_references_exist():
    for folder in (POSITIVE_DIR, NEGATIVE_DIR):
        for path in _iter_cases(folder):
            payload = _load(path)
            meta = payload["metadata"]
            if meta.get("allow_unknown_trackname") is True:
                continue
            tracks = {row["trackName"] for row in payload["trackInfo"]}
            missing = {
                row["trackName"]
                for row in payload["vehicleInfo"]
                if row.get("trackName") not in tracks
            }
            assert not missing, (
                f"{path.name} vehicleInfo refers to tracks not in trackInfo: {missing}"
            )


def test_case_identity_is_unique():
    seen: dict[tuple[str, str, str], Path] = {}
    for folder in (POSITIVE_DIR, NEGATIVE_DIR):
        for path in _iter_cases(folder):
            meta = _load(path)["metadata"]
            key = (meta["spec_section"], meta["rule_id"], meta["variant"])
            assert key not in seen, (
                f"duplicate case identity {key}: {seen[key]} vs {path}"
            )
            seen[key] = path


def test_baseline_files_exist_and_parse():
    assert BASELINE_DIR.exists(), "baselines dir must exist"
    for baseline in ALLOWED_BASELINES:
        path = BASELINE_DIR / f"{baseline}.json"
        assert path.exists(), f"baseline missing: {path}"
        payload = _load(path)
        assert "trackInfo" in payload and payload["trackInfo"], path
        assert "vehicleInfo" in payload, path
        assert payload.get("locoTrackName") is not None, path
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: `test_baseline_files_exist_and_parse` FAILS with "baseline missing". All other tests PASS (vacuously, no cases yet).

- [ ] **Step 4: Commit**

```bash
git add tests/data/__init__.py tests/data/test_validation_inputs_structure.py
git commit -m "Add structural validator for data/validation_inputs rule library"
```

---

## Task 3: Baseline B1_clean

Source: `artifacts/external_validation_inputs/data/validation_2025_11_06_afternoon.json`. Trim `vehicleInfo` to 5 representative cars on mixed starting tracks; keep `trackInfo` verbatim.

**Files:**
- Create: `data/validation_inputs/_baselines/B1_clean.json`

- [ ] **Step 1: Copy source file and trim**

```bash
python3 - <<'PY'
import json
from pathlib import Path

src = Path("artifacts/external_validation_inputs/data/validation_2025_11_06_afternoon.json")
dst = Path("data/validation_inputs/_baselines/B1_clean.json")
payload = json.loads(src.read_text())

# Keep 5 cars with distinct starting tracks.
starts = ("存1", "存2", "修1库内", "预修", "存5南")
picked = []
for track in starts:
    for row in payload["vehicleInfo"]:
        if row["trackName"] == track:
            picked.append(row)
            break

assert len(picked) == 5, f"expected 5 starter cars, got {len(picked)}: {[r['trackName'] for r in picked]}"
payload["vehicleInfo"] = picked
payload["locoTrackName"] = "机库"

dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
```

- [ ] **Step 2: Run validator**

Run: `pytest tests/data/test_validation_inputs_structure.py::test_baseline_files_exist_and_parse -v`

Expected: still FAILS (B2–B5 missing). `B1_clean.json` loads correctly.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/_baselines/B1_clean.json
git commit -m "Add B1_clean baseline (trimmed Nov 6 afternoon)"
```

---

## Task 4: Baseline B2_busy_storage

Source: `artifacts/external_validation_inputs/validation_20260103W.json`. Use as-is; its 52 cars on storage lines make it the saturation baseline.

**Files:**
- Create: `data/validation_inputs/_baselines/B2_busy_storage.json`

- [ ] **Step 1: Copy source**

```bash
cp artifacts/external_validation_inputs/validation_20260103W.json data/validation_inputs/_baselines/B2_busy_storage.json
```

- [ ] **Step 2: Pretty-print and ensure `locoTrackName=机库`**

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("data/validation_inputs/_baselines/B2_busy_storage.json")
payload = json.loads(p.read_text())
payload["locoTrackName"] = "机库"
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
```

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/_baselines/B2_busy_storage.json
git commit -m "Add B2_busy_storage baseline (Jan 3W snapshot)"
```

---

## Task 5: Baseline B3_busy_yard_normal

Source: `artifacts/external_validation_inputs/data/validation_2025_12_08_noon.json`. Full 20 cars in 库内 + 12 in 调棚; represents a `NORMAL` (non-inspection) yard.

**Files:**
- Create: `data/validation_inputs/_baselines/B3_busy_yard_normal.json`

- [ ] **Step 1: Copy and sanitize**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = Path("artifacts/external_validation_inputs/data/validation_2025_12_08_noon.json")
dst = Path("data/validation_inputs/_baselines/B3_busy_yard_normal.json")
payload = json.loads(src.read_text())
payload["locoTrackName"] = "机库"
# Ensure no car already carries isSpotting=迎检; this baseline must be NORMAL.
for row in payload["vehicleInfo"]:
    if row.get("isSpotting") == "迎检":
        row["isSpotting"] = ""
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
```

- [ ] **Step 2: Commit**

```bash
git add data/validation_inputs/_baselines/B3_busy_yard_normal.json
git commit -m "Add B3_busy_yard_normal baseline (Dec 8 noon, NORMAL mode)"
```

---

## Task 6: Baseline B4_busy_yard_inspection

Synthesized from B3: flip 8 cars' `isSpotting` to `迎检` to trigger plan-level `INSPECTION` mode.

**Files:**
- Create: `data/validation_inputs/_baselines/B4_busy_yard_inspection.json`

- [ ] **Step 1: Synthesize**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = Path("data/validation_inputs/_baselines/B3_busy_yard_normal.json")
dst = Path("data/validation_inputs/_baselines/B4_busy_yard_inspection.json")
payload = json.loads(src.read_text())
count = 0
for row in payload["vehicleInfo"]:
    if count >= 8:
        break
    if row.get("targetTrack") == "大库":
        row["isSpotting"] = "迎检"
        count += 1
assert count == 8, f"only flipped {count} cars; expected 8"
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
```

- [ ] **Step 2: Commit**

```bash
git add data/validation_inputs/_baselines/B4_busy_yard_inspection.json
git commit -m "Add B4_busy_yard_inspection baseline (8 cars marked 迎检)"
```

---

## Task 7: Baseline B5_busy_shed

Source: `artifacts/external_validation_inputs/data/validation_2025_11_11_noon.json`. 11 cars in 调棚, variety on 存 lines.

**Files:**
- Create: `data/validation_inputs/_baselines/B5_busy_shed.json`

- [ ] **Step 1: Copy and sanitize**

```bash
python3 - <<'PY'
import json
from pathlib import Path
src = Path("artifacts/external_validation_inputs/data/validation_2025_11_11_noon.json")
dst = Path("data/validation_inputs/_baselines/B5_busy_shed.json")
payload = json.loads(src.read_text())
payload["locoTrackName"] = "机库"
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
```

- [ ] **Step 2: Run full validator**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: `test_baseline_files_exist_and_parse` now PASSES. Other tests pass vacuously.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/_baselines/B5_busy_shed.json
git commit -m "Add B5_busy_shed baseline (Nov 11 noon, 11 cars in 调棚)"
```

---

## Authoring Pattern for All Remaining Tasks

Every case-authoring task (Tasks 9–28) follows the same recipe. Apply it to every case row in the task's table.

1. Start from the baseline named in the row: copy `trackInfo` verbatim and `vehicleInfo` as a starting point.
2. Apply the mutation described in the row's "Vehicle mutation" column.
3. Fill `metadata` from the row's `(spec_section, rule_id, variant, baseline, purpose)` fields.
4. Fill `expected_bounds` (positive) or `expected_error` (negative) from the row's "Expected" column.
5. Save as `data/validation_inputs/<positive|negative>/case_<spec_section>_<rule>_<variant>.json` using underscores for the dot in `spec_section` (e.g., `3.2` → `3_2`).
6. Append a single commit per task containing all cases in that task.

For brevity, `spec_section` in filenames replaces `.` with `_` (e.g., `case_3_2_shed_work_tanker.json`).

Every negative case where `rule_id` starts with `走行线_as_source` or `order_` or involves a deliberately-invalid field **must** set `metadata.allow_unknown_trackname: true` if the offending input also references a track outside `trackInfo`.

### Running the validator

After each task, run:

```
pytest tests/data/test_validation_inputs_structure.py -v
```

Expected: all tests PASS. If a test fails, fix the case file before committing.

---

## Task 8: Sanity-run validator on current state

- [ ] **Step 1: Run full validator**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: all 6 tests PASS (5 baselines exist, no cases yet so case tests pass vacuously).

No commit; this is a checkpoint.

---

## Task 9: Chapter 3.2 调棚 cases (5 positive)

All from baseline **B5_busy_shed**. For each case, copy B5's `vehicleInfo`, remove one existing 调棚-bound car, add the new car described below. Append to `positive/`.

| # | Filename | Variant | Vehicle mutation | expected_bounds |
| - | -------- | ------- | ---------------- | --------------- |
| 1 | `case_3_2_shed_work_tanker.json` | `tanker` | Add car: `trackName=存5南, order="99", vehicleModel=GQ70, vehicleNo=T001, repairProcess=段修, vehicleLength=13.0, targetMode=AREA, targetTrack=调棚, targetAreaCode=调棚:WORK, isSpotting=是` | `must_be_solvable=true, must_visit_tracks=["调棚"]` |
| 2 | `case_3_2_shed_work_flatcar.json` | `flatcar` | Add: `NX70AF, vehicleLength=17.6, targetTrack=调棚, isSpotting=是, targetAreaCode=调棚:WORK` from 存1 | same |
| 3 | `case_3_2_shed_work_gondola.json` | `gondola` | Add: `C70E, vehicleLength=13.6, targetTrack=调棚, isSpotting=是, targetAreaCode=调棚:WORK` from 存2 | same |
| 4 | `case_3_2_shed_pre_repair_from_pre_repair.json` | `from_pre_repair` | Add: `C70, vehicleLength=14.3, targetTrack=调棚, isSpotting=""` (empty) from 预修 | `must_be_solvable=true, must_visit_tracks=["调棚"]` |
| 5 | `case_3_2_shed_pre_repair_from_cun5bei.json` | `from_cun5bei` | Same payload as #4 but starting from 存5北 | same |

- [ ] **Step 1: Create the 5 JSON case files**

For each row, author the JSON following the positive case template and the mutation described. Use a unique `vehicleNo` (`T001`..`T005`). Keep `expected_bounds.max_hook_count=null`, `close_door_constraint=null`, `weigh_required_vehicles=[]`.

- [ ] **Step 2: Run validator**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_3_2_*.json
git commit -m "Add Ch 3.2 调棚 positive cases (5 variants)"
```

---

## Task 10: Chapter 3.3 `isSpotting` positive cases (6 positive)

Covers 迎检 mode and SPOT 101/203/305/407 boundaries.

Baselines: 迎检 cases use **B4_busy_yard_inspection**; SPOT cases use **B3_busy_yard_normal**.

| # | Filename | Baseline | Variant | Vehicle mutation | expected_bounds |
| - | -------- | -------- | ------- | ---------------- | --------------- |
| 1 | `case_3_3_inspection_to_depot_random.json` | B4 | `to_depot_random` | Add: from 存5南, `targetTrack=大库, isSpotting=迎检, targetMode=AREA, targetAreaCode=大库:RANDOM` | `must_be_solvable=true, must_visit_tracks=["大库"]` |
| 2 | `case_3_3_inspection_to_spot_307.json` | B4 | `to_spot_307` | Add: from 存1, `targetTrack=修3库内, isSpotting=307, targetMode=SPOT, targetSpotCode="307"` | `must_be_solvable=true, must_visit_tracks=["修3库内"]` |
| 3 | `case_3_3_spot_101_boundary.json` | B3 | `spot_101` | Add: from 存1, `targetTrack=修1库内, isSpotting=101, targetMode=SPOT, targetSpotCode="101", vehicleLength=14.3` | `must_be_solvable=true` |
| 4 | `case_3_3_spot_203_mid.json` | B3 | `spot_203` | `targetTrack=修2库内, isSpotting=203, targetMode=SPOT, targetSpotCode="203", vehicleLength=14.3` from 存2 | same |
| 5 | `case_3_3_spot_305_long_car.json` | B3 | `spot_305` | `targetTrack=修3库内, isSpotting=305, targetMode=SPOT, targetSpotCode="305", vehicleLength=17.6` from 存3 | same |
| 6 | `case_3_3_spot_407_boundary.json` | B3 | `spot_407_boundary` | `targetTrack=修4库内, isSpotting=407, targetMode=SPOT, targetSpotCode="407", vehicleLength=14.3` from 存5南; note this requires `yardMode=INSPECTION` — set `metadata.purpose` to call out the cross-mode interaction and set baseline to **B4** instead | `must_be_solvable=true` |

(Row 6 correction: since 407 requires INSPECTION mode, use **B4_busy_yard_inspection** as the baseline for row 6. Rows 3–5 stay on B3.)

- [ ] **Step 1: Create the 6 case files**

- [ ] **Step 2: Run validator**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_3_3_inspection_*.json data/validation_inputs/positive/case_3_3_spot_*.json
git commit -m "Add Ch 3.3 isSpotting positive cases (迎检 + SPOT 101/203/305/407)"
```

---

## Task 11: Chapter 3.3 `isSpotting` negative cases (6 negative)

All negative. Baseline **B1_clean** unless stated.

| # | Filename | Variant | Offending row | expected_error |
| - | -------- | ------- | ------------- | -------------- |
| 1 | `case_3_3_spot_track_mismatch_neg.json` | `track_mismatch` | Add: `targetTrack=修3库内, isSpotting=203` (spot 203 belongs to 修2库内) from 存1 | `category=illegal_input, rule_citation="spec 3.3 compat rule 3"` |
| 2 | `case_3_3_area_on_cun1_neg.json` | `area_on_cun1` | Add: `targetTrack=存1, isSpotting=是` from 存2 | `category=illegal_input, rule_citation="spec 3.3 compat rule 2"` |
| 3 | `case_3_3_area_on_lin1_neg.json` | `area_on_lin1` | Add: `targetTrack=临1, isSpotting=是` from 存1. | same |
| 4 | `case_3_3_area_on_du5_neg.json` | `area_on_du5` | Add: `targetTrack=渡5, isSpotting=是` from 存1 | same |
| 5 | `case_3_3_isSpotting_unknown_value_neg.json` | `unknown_value` | Add: `targetTrack=大库, isSpotting=快送` (invalid literal) from 存1 | `category=illegal_input, rule_citation="spec 3.3 compat rule 4"` |
| 6 | `case_3_3_mixed_mode_conflict_neg.json` | `mixed_mode` | Add two cars: first `targetTrack=大库, isSpotting=迎检`; second `targetTrack=大库, isSpotting=205` (NORMAL-only spot). Baseline **B1_clean**. | `category=illegal_input, rule_citation="spec 3.3 compat rule 6"` |

- [ ] **Step 1: Create the 6 negative case files**

- [ ] **Step 2: Run validator**

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/negative/case_3_3_*.json
git commit -m "Add Ch 3.3 isSpotting negative cases (6 rejection paths)"
```

---

## Task 12: Chapter 3.4 `vehicleAttributes` positive cases (8 positive)

Baseline **B1_clean**. Each case adds 1–3 cars with specific `vehicleAttributes` values.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_3_4_weigh_to_depot.json` | `to_depot` | Add: from 存1, `targetTrack=大库, isSpotting=""`, `vehicleAttributes=称重`, `vehicleModel=GQ70, vehicleLength=13.0` | `must_be_solvable=true, must_visit_tracks=["机库","大库"], weigh_required_vehicles=["W001"]` |
| 2 | `case_3_4_weigh_to_cun1.json` | `to_cun1` | Add: from 存2, `targetTrack=存1`, `vehicleAttributes=称重` | `must_be_solvable=true, must_visit_tracks=["机库","存1"], weigh_required_vehicles=["W002"]` |
| 3 | `case_3_4_heavy_single.json` | `single` | Add: from 存1, `targetTrack=大库, vehicleAttributes=重车, vehicleModel=C70E, vehicleLength=14.3` | `must_be_solvable=true` |
| 4 | `case_3_4_heavy_pair.json` | `pair` | Add two heavy cars to 存1 targeting 大库 | `must_be_solvable=true` |
| 5 | `case_3_4_heavy_mix_with_empties.json` | `mix_empties` | Add 1 重车 + 3 空车 all targeting 大库 from 存1/2/3 | `must_be_solvable=true` |
| 6 | `case_3_4_close_door_to_cun1.json` | `to_cun1` | Add: from 存2, `targetTrack=存1, vehicleAttributes=关门车` (vehicleNo=CD001) | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD001","forbidden_positions":[],"target_track":"存1"}]` |
| 7 | `case_3_4_close_door_to_depot.json` | `to_depot` | Add: from 存2, `targetTrack=大库, vehicleAttributes=关门车` (vehicleNo=CD002) | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD002","forbidden_positions":[],"target_track":"大库"}]` |
| 8 | `case_3_4_close_door_to_shed.json` | `to_shed` | Add: from 存2, `targetTrack=调棚, isSpotting="", vehicleAttributes=关门车` (vehicleNo=CD003) | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD003","forbidden_positions":[],"target_track":"调棚"}]` |

Use vehicleNos: `W001`, `W002`, `H001`–`H005` (heavy), `CD001`–`CD003` (close-door).

- [ ] **Step 1: Create the 8 case files**

- [ ] **Step 2: Run validator**

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_3_4_*.json
git commit -m "Add Ch 3.4 vehicleAttributes cases (weigh + heavy + close-door)"
```

---

## Task 13: Chapter 4.3.1 普通 TRACK targets (5 positive)

Baseline **B1_clean**. Each case dispatches a single car to a different ordinary TRACK target.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_4_3_1_track_cun1.json` | `cun1` | Add: from 存5南, `targetTrack=存1, targetMode=TRACK` | `must_be_solvable=true, must_visit_tracks=["存1"]` |
| 2 | `case_4_3_1_track_cun5nan.json` | `cun5nan` | Add: from 存1, `targetTrack=存5南, targetMode=TRACK` | `must_visit_tracks=["存5南"]` |
| 3 | `case_4_3_1_track_pre_repair.json` | `pre_repair` | Add: from 存2, `targetTrack=预修, targetMode=TRACK` | `must_visit_tracks=["预修"]` |
| 4 | `case_4_3_1_track_jipeng.json` | `jipeng` | Add: from 存1, `targetTrack=机棚, targetMode=TRACK` | `must_visit_tracks=["机棚"]` |
| 5 | `case_4_3_1_track_jibei.json` | `jibei` | Add: from 存1, `targetTrack=机北, targetMode=TRACK`. **Set `locoTrackName="机北"`** for this case to cover the 机北 loco position (orthogonal coverage for spec §7.7). | `must_visit_tracks=["机北"]` |

- [ ] **Step 1: Create 5 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_4_3_1_*.json
git commit -m "Add Ch 4.3.1 普通 TRACK targets (5 variants)"
```

---

## Task 14: Chapter 4.3.4 AREA targets (4 positive)

Baseline **B1_clean**. 洗南 / 油 / 抛 / 轮.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_4_3_4_area_xinan.json` | `xinan` | Add from 存1: `targetTrack=洗南, isSpotting="", targetMode=AREA, targetAreaCode=洗南:WORK`. **Set `locoTrackName=""`** (empty) for this case to cover the empty loco default (orthogonal coverage for spec §7.7). | `must_visit_tracks=["洗南"]` |
| 2 | `case_4_3_4_area_you.json` | `you` | Add from 存1: `targetTrack=油, targetAreaCode=油:WORK` | `must_visit_tracks=["油"]` |
| 3 | `case_4_3_4_area_pao.json` | `pao` | Add from 存1: `targetTrack=抛, targetAreaCode=抛:WORK` | `must_visit_tracks=["抛"]` |
| 4 | `case_4_3_4_area_lun.json` | `lun` | Add from 存1: `targetTrack=轮, targetAreaCode=轮:OPERATE` | `must_visit_tracks=["轮"]` |

- [ ] **Step 1: Create 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_4_3_4_*.json
git commit -m "Add Ch 4.3.4 AREA targets (洗南/油/抛/轮)"
```

---

## Task 15: Chapter 4.3.5 大库 RANDOM × 修程 (3 positive)

Baseline **B3_busy_yard_normal**.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_4_3_5_depot_random_duanxiu.json` | `duanxiu` | Add from 存1: `repairProcess=段修, targetTrack=大库, targetMode=AREA, targetAreaCode=大库:RANDOM, vehicleLength=14.3` | `must_visit_tracks=["大库"]` |
| 2 | `case_4_3_5_depot_random_changxiu.json` | `changxiu` | Same but `repairProcess=厂修` | same |
| 3 | `case_4_3_5_depot_random_linxiu.json` | `linxiu` | Same but `repairProcess=临修` | same |

- [ ] **Step 1: Create 3 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_4_3_5_*.json
git commit -m "Add Ch 4.3.5 大库 RANDOM cases (段/厂/临 修程)"
```

---

## Task 16: Chapter 4.3.6 大库外 cases (5 positive)

Baseline **B3_busy_yard_normal**.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_4_3_6_outer_random.json` | `random` | Add from 存1: `targetTrack=大库外, targetMode=AREA, targetAreaCode=大库外:RANDOM` | `must_visit_tracks` one of `["修1库外","修2库外","修3库外","修4库外"]` — encode as `must_visit_one_of=["修1库外","修2库外","修3库外","修4库外"]` |
| 2 | `case_4_3_6_outer_xiu1.json` | `xiu1` | `targetTrack=修1库外, targetMode=TRACK` | `must_visit_tracks=["修1库外"]` |
| 3 | `case_4_3_6_outer_xiu2.json` | `xiu2` | `targetTrack=修2库外` | `["修2库外"]` |
| 4 | `case_4_3_6_outer_xiu3.json` | `xiu3` | `targetTrack=修3库外` | `["修3库外"]` |
| 5 | `case_4_3_6_outer_xiu4.json` | `xiu4` | `targetTrack=修4库外` | `["修4库外"]` |

Note: the `must_visit_one_of` key is a new shape — the validator does not check its contents, only the presence of top-level metadata keys, so this new key does not need a validator update.

- [ ] **Step 1: Create 5 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_4_3_6_*.json
git commit -m "Add Ch 4.3.6 大库外 cases (RANDOM + 4 specific 修N库外)"
```

---

## Task 17: Chapter 4 negative cases (2 negative)

| # | Filename | Baseline | Variant | Mutation | expected_error |
| - | -------- | -------- | ------- | -------- | -------------- |
| 1 | `case_4_3_2_cun4nan_as_target_neg.json` | B1 | `cun4nan_final` | Add from 存1: `targetTrack=存4南, targetMode=TRACK` | `category=illegal_input, rule_citation="spec 4.3 rule 2"` |
| 2 | `case_4_3_7_weigh_as_final_target_neg.json` | B1 | `weigh_as_final` | Add from 存1: `targetTrack=机库, targetMode=AREA, targetAreaCode=机库:WEIGH` (upstream tries to dispatch the WEIGH area directly) | `category=illegal_input, rule_citation="spec 4.3 rule 7"` |

- [ ] **Step 1: Create 2 negative files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/negative/case_4_3_*.json
git commit -m "Add Ch 4 negative cases (存4南 as target; 机库:WEIGH as final)"
```

---

## Task 18: Chapter 5.1 input-legality negative cases (6 negative)

Baseline **B1_clean** for all.

| # | Filename | Variant | Mutation | expected_error |
| - | -------- | ------- | -------- | -------------- |
| 1 | `case_5_1_source_du1_neg.json` | `du1` | Add: `trackName=渡1, order="1", targetTrack=存1` | `category=illegal_input, rule_citation="spec 5.1 rule 3"` |
| 2 | `case_5_1_source_lian6_neg.json` | `lian6` | Add: `trackName=联6, order="1", targetTrack=存1` | same |
| 3 | `case_5_1_target_lin1_neg.json` | `target_lin1` | Add from 存1: `targetTrack=临1, targetMode=TRACK` | `category=illegal_input, rule_citation="spec 5.1 rule 4"` |
| 4 | `case_5_1_target_lin2_neg.json` | `target_lin2` | Add from 存1: `targetTrack=临2, targetMode=TRACK` | same |
| 5 | `case_5_1_order_duplicate_neg.json` | `duplicate` | Add two cars both with `trackName=存1, order="5"` | `category=illegal_input, rule_citation="spec 5.1 rule 2"` |
| 6 | `case_5_1_order_non_integer_neg.json` | `non_integer` | Add: `trackName=存1, order="5.5"` | same |

- [ ] **Step 1: Create 6 negative files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/negative/case_5_1_*.json
git commit -m "Add Ch 5.1 input-legality negatives (6 rejection paths)"
```

---

## Task 19: Chapter 5.2 tow-limit cases (4 positive + 4 negative)

Baseline **B1_clean**. For tow-limit cases, clear out B1's original `vehicleInfo` in each case to keep inventory predictable; add only the tow-limit inputs.

| # | Filename | Folder | Variant | Mutation | Expected |
| - | -------- | ------ | ------- | -------- | -------- |
| 1 | `case_5_2_tow_empty_20cars.json` | positive | `20cars` | 20 empty cars on 存5北, all `targetTrack=大库` | `must_be_solvable=true` |
| 2 | `case_5_2_tow_empty_21cars_neg.json` | negative | `21cars` | 21 empty cars on 存5北, all `targetTrack=大库` | `category=tow_limit_exceeded, rule_citation="spec 5.2 rule 1"` |
| 3 | `case_5_2_tow_heavy_2cars.json` | positive | `heavy_2` | 2 重车 + 2 空车 on 存5北, all `targetTrack=大库`, 重车 via `vehicleAttributes=重车` | `must_be_solvable=true` |
| 4 | `case_5_2_tow_heavy_3cars_neg.json` | negative | `heavy_3` | 3 重车 + 2 空车 same target | `category=tow_limit_exceeded, rule_citation="spec 5.2 rule 2"` |
| 5 | `case_5_2_tow_heavy_equivalence_legal.json` | positive | `heavy_equiv_legal` | 1 重车 + 16 空车 all on 存5北 → 大库 (16 + 4 = 20) | `must_be_solvable=true` |
| 6 | `case_5_2_tow_heavy_equivalence_over_neg.json` | negative | `heavy_equiv_over` | 1 重车 + 17 空车 → total 21-equivalent | `category=tow_limit_exceeded, rule_citation="spec 5.2 rule 3"` |
| 7 | `case_5_2_l1_within_190m.json` | positive | `l1_at_190m` | 13 cars × 14.6m ≈ 189.8m on 存5北 → 存2; ensures path traverses L1 | `must_be_solvable=true` |
| 8 | `case_5_2_l1_over_190m_neg.json` | negative | `l1_over_190m` | 14 cars × 14.6m ≈ 204.4m on 存5北 → 存2 | `category=tow_limit_exceeded, rule_citation="spec 5.2 rule 4"` |

Use vehicleLength=14.3 for empties unless noted. All cars have consistent `repairProcess=段修`. Ensure `trackDistance` of 存5北 (367m) accommodates 21 cars; it does.

- [ ] **Step 1: Create all 8 case files (4 positive + 4 negative)**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_5_2_*.json data/validation_inputs/negative/case_5_2_*.json
git commit -m "Add Ch 5.2 tow-limit cases (20/21 empty, 2/3 heavy, equivalence, L1 190m)"
```

---

## Task 20: Chapter 5.3 track-length capacity (1 positive + 1 negative)

Baseline **B2_busy_storage**. Construct a case where one destination line is right at the capacity boundary.

Choose 存1 (trackDistance = 113.0m). Currently B2 has some cars on 存1; trim to a known cumulative length and add new cars to saturate.

| # | Filename | Folder | Variant | Mutation | Expected |
| - | -------- | ------ | ------- | -------- | -------- |
| 1 | `case_5_3_track_at_capacity.json` | positive | `at_capacity` | Replace B2 `vehicleInfo` with: (a) existing cars on 存1 adjusted to sum to 98.7m, (b) one new car from 存2 length 14.3m, `targetTrack=存1`. Post-arrival 存1 load = 113.0m exactly. | `must_be_solvable=true, must_visit_tracks=["存1"]` |
| 2 | `case_5_3_track_overflow_neg.json` | negative | `overflow` | Same as above but the incoming car has `vehicleLength=16.0`; post-arrival would be 114.7m > 113.0m. | `category=capacity_overflow, rule_citation="spec 5.3 rule 2"` |

- [ ] **Step 1: Create both case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_5_3_track_at_capacity.json data/validation_inputs/negative/case_5_3_track_overflow_neg.json
git commit -m "Add Ch 5.3 track-length boundary cases"
```

---

## Task 21: Chapter 5.5 weighing cases (3 positive + 1 negative)

Baseline **B1_clean**.

| # | Filename | Folder | Variant | Mutation | Expected |
| - | -------- | ------ | ------- | -------- | -------- |
| 1 | `case_5_5_weigh_single_car.json` | positive | `single_car` | Add one car with `vehicleAttributes=称重, targetTrack=大库` from 存1 | `weigh_required_vehicles=["W101"], must_visit_tracks=["机库","大库"]` |
| 2 | `case_5_5_weigh_three_cars_three_hooks.json` | positive | `three_cars_three_hooks` | Add 3 称重 cars (each `vehicleNo=W102/W103/W104`), three different targets (存1, 存2, 大库) | `weigh_required_vehicles=["W102","W103","W104"], must_visit_tracks=["机库"]` |
| 3 | `case_5_5_weigh_to_non_depot_final.json` | positive | `to_non_depot` | Add 1 称重 car targeting 存1 | `weigh_required_vehicles=["W105"], must_visit_tracks=["机库","存1"]` |
| 4 | `case_5_5_two_weigh_same_hook_neg.json` | negative | `two_weigh_same_hook` | Add 2 称重 cars both on 存1 targeting 大库 where the solver has no choice but to group them into a single hook (limit the rest of the yard so separate hooks are infeasible — easiest: reduce `trackInfo.存1.trackDistance` to 30m so only one hook fits). This is a pathological input; validate via `expected_error`. | `category=illegal_input, rule_citation="spec 5.5 rule 2"` |

Note: the fourth negative is about the business rule "a single hook processes at most 1 weigh car". Since we cannot guarantee the solver will try grouping, this case tests the *input-side* assertion that the plan MUST NOT schedule two weigh cars in the same hook; frame the negative as "two 称重 cars whose state forces co-processing" and rely on the runner's downstream verifier to catch.

- [ ] **Step 1: Create all 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_5_5_*.json data/validation_inputs/negative/case_5_5_*.json
git commit -m "Add Ch 5.5 weighing cases (single/multi + two-in-one-hook negative)"
```

---

## Task 22: Chapter 6 non-存4北 close-door cases (2 positive + 2 negative)

Baseline **B1_clean** (expand vehicleInfo to get hook sizes right).

| # | Filename | Folder | Variant | Mutation | Expected |
| - | -------- | ------ | ------- | -------- | -------- |
| 1 | `case_6_1_non_cun4bei_post10_cd_first.json` | positive | `post10_cd_first` | 10 cars on 存1, 1 关门车 (`CD201`) at `order=1`, all targeting 大库. Post-hook machine-tail count = 10. | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD201","forbidden_positions":[],"target_track":"大库"}]` |
| 2 | `case_6_1_non_cun4bei_post11_cd_first_neg.json` | negative | `post11_cd_first` | 11 cars on 存1, 关门车 at `order=1`, all targeting 大库. | `category=close_door_violation, rule_citation="spec 6.1 non-cun4bei"` |
| 3 | `case_6_1_non_cun4bei_post15_cd_first_neg.json` | negative | `post15_cd_first` | Same with 15 cars, 关门车 at `order=1`. | same |
| 4 | `case_6_1_non_cun4bei_post11_cd_second.json` | positive | `post11_cd_second` | 11 cars on 存1, 关门车 (`CD202`) at `order=2`, all targeting 大库. | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD202","forbidden_positions":[1],"target_track":"大库"}]` |

- [ ] **Step 1: Create all 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_6_1_non_cun4bei_*.json data/validation_inputs/negative/case_6_1_non_cun4bei_*.json
git commit -m "Add Ch 6 non-存4北 close-door cases (positions × machine-tail count)"
```

---

## Task 23: Chapter 6 存4北 close-door cases + multi-close-door (2 positive + 3 negative)

Baseline **B2_busy_storage** for 存4北 cases, **B1_clean** for multi.

| # | Filename | Folder | Baseline | Variant | Mutation | Expected |
| - | -------- | ------ | -------- | ------- | -------- | -------- |
| 1 | `case_6_1_cun4bei_cd_pos4.json` | positive | B2 | `pos4` | Add cars on 存5北 → 存4北: 3 normal cars at order=1,2,3 and 1 关门车 (`CD301`) at order=4 | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD301","forbidden_positions":[1,2,3],"target_track":"存4北"}]` |
| 2 | `case_6_1_cun4bei_cd_pos1_neg.json` | negative | B2 | `pos1` | 4 cars on 存5北 → 存4北, 关门车 at order=1 | `category=close_door_violation, rule_citation="spec 6.1 cun4bei"` |
| 3 | `case_6_1_cun4bei_cd_pos2_neg.json` | negative | B2 | `pos2` | 4 cars, 关门车 at order=2 | same |
| 4 | `case_6_1_cun4bei_cd_pos3_neg.json` | negative | B2 | `pos3` | 4 cars, 关门车 at order=3 | same |
| 5 | `case_6_1_multi_close_door_mixed.json` | positive | B1 | `multi_mixed` | 12 cars on 存1 → 存4北, positions 5 and 9 are 关门车 (`CD302`, `CD303`). | `must_be_solvable=true, close_door_constraint=[{"vehicle_no":"CD302","forbidden_positions":[1,2,3],"target_track":"存4北"},{"vehicle_no":"CD303","forbidden_positions":[1,2,3],"target_track":"存4北"}]` |

- [ ] **Step 1: Create all 5 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_6_1_cun4bei_*.json data/validation_inputs/positive/case_6_1_multi_*.json data/validation_inputs/negative/case_6_1_cun4bei_*.json
git commit -m "Add Ch 6 存4北 close-door cases + multi-close-door"
```

---

## Task 24: Chapter 7.1 调棚 SPOT cases (4 positive)

Baseline **B5_busy_shed**. For each, remove B5's existing car at the target spot and add a new car targeting that spot.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_7_1_shed_spot_1.json` | `spot_1` | Vacate 调棚 position 1 in B5 then add from 存1: `targetTrack=调棚, isSpotting=是, targetMode=SPOT, targetSpotCode="1"` | `must_visit_tracks=["调棚"]` |
| 2 | `case_7_1_shed_spot_2.json` | `spot_2` | Same for position 2 | same |
| 3 | `case_7_1_shed_spot_3.json` | `spot_3` | Same for position 3 | same |
| 4 | `case_7_1_shed_spot_4.json` | `spot_4` | Same for position 4 | same |

Note: for Ch 7.1 positions `1`–`4`, the spec treats them as named SPOTs within the 调棚:WORK area. The case declares `targetMode=SPOT, targetSpotCode="<N>"` and omits `targetAreaCode` (SPOT mode is more specific).

- [ ] **Step 1: Create 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_7_1_*.json
git commit -m "Add Ch 7.1 调棚 SPOT cases (positions 1–4)"
```

---

## Task 25: Chapter 7.2 + 7.3 work-area cases (4 positive)

Baseline **B1_clean**.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_7_2_xinan_spot_1.json` | `spot_1` | From 存1: `targetTrack=洗南, isSpotting=是, targetMode=SPOT, targetSpotCode="1"` | `must_visit_tracks=["洗南"]` |
| 2 | `case_7_2_xinan_spot_2.json` | `spot_2` | Same, spot 2 | same |
| 3 | `case_7_2_xinan_spot_3.json` | `spot_3` | Same, spot 3 | same |
| 4 | `case_7_3_weigh_then_jiku_final.json` | `weigh_then_jiku` | From 存1: 1 car with `vehicleAttributes=称重, targetTrack=机库, targetMode=TRACK` | `weigh_required_vehicles=["W301"], must_visit_tracks=["机库"]` |

- [ ] **Step 1: Create 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_7_2_*.json data/validation_inputs/positive/case_7_3_*.json
git commit -m "Add Ch 7.2/7.3 work-area cases (洗南 SPOT 1–3 + 机库 weigh+stay)"
```

---

## Task 26: Chapter 8.1 NORMAL/INSPECTION mode cases (1 positive + 1 negative + 2 positive)

| # | Filename | Folder | Baseline | Variant | Mutation | Expected |
| - | -------- | ------ | -------- | ------- | -------- | -------- |
| 1 | `case_8_1_normal_uses_01_to_05.json` | positive | B3 | `normal_01_05` | Add from 存1: `targetTrack=大库, targetMode=AREA, targetAreaCode=大库:RANDOM, vehicleLength=14.3` in NORMAL yard | `must_visit_tracks=one of 修1–4 库内 at spot 01–05; encode as must_visit_one_of=["修1库内","修2库内","修3库内","修4库内"]` |
| 2 | `case_8_1_normal_forces_06_neg.json` | negative | B3 | `forces_06` | Add: `targetTrack=修2库内, isSpotting=206, targetMode=SPOT, targetSpotCode="206"` from 存1 | `category=illegal_input, rule_citation="spec 8.3 rule 1"` |
| 3 | `case_8_1_inspection_uses_06_107.json` | positive | B4 | `inspection_106` | Add: `targetTrack=修1库内, isSpotting=106, targetMode=SPOT, targetSpotCode="106"` from 存1 | `must_visit_tracks=["修1库内"]` |
| 4 | `case_8_1_inspection_uses_07_307.json` | positive | B4 | `inspection_307` | Add: `targetTrack=修3库内, isSpotting=307, targetMode=SPOT, targetSpotCode="307"` from 存1 | `must_visit_tracks=["修3库内"]` |

- [ ] **Step 1: Create all 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_8_1_*.json data/validation_inputs/negative/case_8_1_*.json
git commit -m "Add Ch 8.1 NORMAL/INSPECTION spot-range cases"
```

---

## Task 27: Chapter 8.2 length-based spot allocation (4 positive)

Baseline **B3_busy_yard_normal**.

| # | Filename | Variant | Mutation | expected_bounds |
| - | -------- | ------- | -------- | --------------- |
| 1 | `case_8_2_long_176m_routes_to_3_4.json` | `long_176m` | Add from 存1: `vehicleLength=17.6, targetTrack=大库, targetMode=AREA, targetAreaCode=大库:RANDOM` | `must_visit_one_of=["修3库内","修4库内"]` |
| 2 | `case_8_2_long_192m_routes_to_3_4.json` | `long_192m` | Same with `vehicleLength=19.2` | same |
| 3 | `case_8_2_short_143m_prefers_1_2.json` | `short_143m` | `vehicleLength=14.3` | `must_visit_one_of=["修1库内","修2库内"]` |
| 4 | `case_8_2_short_160m_prefers_1_2.json` | `short_160m` | `vehicleLength=16.0` | same |

- [ ] **Step 1: Create 4 case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_8_2_*.json
git commit -m "Add Ch 8.2 length-based spot allocation cases"
```

---

## Task 28: Chapter 8 downgrade + 8.3 negative (1 positive + 1 negative)

| # | Filename | Folder | Baseline | Variant | Mutation | Expected |
| - | -------- | ------ | -------- | ------- | -------- | -------- |
| 1 | `case_8_2_downgrade_full_1_2_to_3_4.json` | positive | B3 | `downgrade_full_1_2` | Modify B3 to place 10 cars in 修1库内 and 10 in 修2库内 (fill all 1/2 NORMAL spots 01–05 × 2 lines = 10 spots with 10 cars), then add from 存1: short-car (vehicleLength=14.3) targeting 大库:RANDOM | `must_visit_one_of=["修3库内","修4库内"]` |
| 2 | `case_8_3_incompatible_spot_neg.json` | negative | B3 | `short_to_long_spot` | Add from 存1: `vehicleLength=14.3, targetTrack=修4库内, isSpotting=405, targetMode=SPOT, targetSpotCode="405"` where 405 is defined as a long-car-only slot per spec 8.2 | `category=yard_allocation_violation, rule_citation="spec 8.3 rule 2"` |

- [ ] **Step 1: Create both case files**

- [ ] **Step 2: Run validator**. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add data/validation_inputs/positive/case_8_2_downgrade_full_1_2_to_3_4.json data/validation_inputs/negative/case_8_3_incompatible_spot_neg.json
git commit -m "Add Ch 8 downgrade + 8.3 incompatible-spot negative"
```

---

## Task 29: Coverage audit test

Add a test that asserts the counts per chapter match the design spec.

**Files:**
- Modify: `tests/data/test_validation_inputs_structure.py`

- [ ] **Step 1: Append the audit test**

Append to `tests/data/test_validation_inputs_structure.py`:

```python
EXPECTED_COUNTS = {
    # (spec_section_prefix): (positive_count, negative_count)
    "3.2": (5, 0),
    "3.3": (6, 6),
    "3.4": (8, 0),
    "4.3.1": (5, 0),
    "4.3.2": (0, 1),
    "4.3.4": (4, 0),
    "4.3.5": (3, 0),
    "4.3.6": (5, 0),
    "4.3.7": (0, 1),
    "5.1": (0, 6),
    "5.2": (4, 4),
    "5.3": (1, 1),
    "5.5": (3, 1),
    "6.1": (4, 5),
    "7.1": (4, 0),
    "7.2": (3, 0),
    "7.3": (1, 0),
    "8.1": (3, 1),
    "8.2": (5, 0),
    "8.3": (0, 1),
}


def _count_by_prefix(folder: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _iter_cases(folder):
        meta = _load(path)["metadata"]
        section = meta["spec_section"]
        # Some chapters (e.g., 5.1) use spec_section="5.1" directly;
        # others (e.g., 4.3.1) use a three-level section id stored in the
        # file name rather than in metadata. Use metadata.spec_section as
        # the canonical key; tasks store it accordingly.
        counts[section] = counts.get(section, 0) + 1
    return counts


def test_coverage_counts_match_spec():
    pos_counts = _count_by_prefix(POSITIVE_DIR)
    neg_counts = _count_by_prefix(NEGATIVE_DIR)
    mismatches = []
    for section, (exp_pos, exp_neg) in EXPECTED_COUNTS.items():
        got_pos = pos_counts.get(section, 0)
        got_neg = neg_counts.get(section, 0)
        if got_pos != exp_pos or got_neg != exp_neg:
            mismatches.append(
                f"{section}: expected {exp_pos}+/{exp_neg}-, got {got_pos}+/{got_neg}-"
            )
    assert not mismatches, "\n".join(mismatches)


def test_total_case_count():
    pos = len(_iter_cases(POSITIVE_DIR))
    neg = len(_iter_cases(NEGATIVE_DIR))
    assert pos == 64, f"expected 64 positive cases, got {pos}"
    assert neg == 27, f"expected 27 negative cases, got {neg}"
```

Note: Tasks 13–16 use three-level spec sections in filenames (e.g., `4_3_1`) but `metadata.spec_section` is stored as the full three-level value ("4.3.1"). Ensure consistency when authoring cases.

- [ ] **Step 2: Update filename regex to allow three-level sections**

Already done in Task 2 — skip this step (kept as a placeholder so step numbering remains aligned).

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: all tests PASS, including the new `test_coverage_counts_match_spec` and `test_total_case_count`.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_validation_inputs_structure.py
git commit -m "Add coverage-count audit test against design matrix"
```

---

## Task 30: Authoring README

Produce a README that explains the directory and reprints the coverage matrix so the library is self-describing without reading the design spec.

**Files:**
- Create: `data/validation_inputs/README.md`

- [ ] **Step 1: Create the README**

Create `data/validation_inputs/README.md`:

```markdown
# Curated Rule-Scenario Library

Hand-authored JSON inputs exercising every rule branch in `福州东调车业务说明.md`
chapters 3–8. Paired with the design spec at
`docs/plans/2026-04-18-curated-rule-scenarios-design.md`.

## Layout

- `_baselines/` — 5 frozen starting yards (B1..B5). Case files cite one via
  `metadata.baseline`.
- `positive/` — legal inputs. Each file declares `metadata.expected_bounds` with
  soft assertions (solvability, must-visit tracks, close-door constraints,
  weighing requirements).
- `negative/` — illegal inputs. Each file declares `metadata.expected_error`
  with the rule citation that the planner is expected to reject on.

## Authoring a new case

1. Pick a baseline that fits the rule environment.
2. Copy its `trackInfo` verbatim into the new file.
3. Start from the baseline's `vehicleInfo` and mutate minimally to trigger
   the rule.
4. Fill `metadata` including `spec_section`, `rule_id`, `variant`, `baseline`,
   `purpose`, and either `expected_bounds` or `expected_error`.
5. Save as `case_<section_with_underscore>_<rule_slug>_<variant>.json`.
6. Run `pytest tests/data/test_validation_inputs_structure.py -v`.
7. If adding a new section to the coverage matrix, update `EXPECTED_COUNTS` in
   the validator test and the table below.

## Coverage matrix

| Chapter | Rule theme                           | Positive | Negative |
| ------- | ------------------------------------ | :------: | :------: |
| 3.2     | 调棚 WORK / PRE_REPAIR               |    5     |    0     |
| 3.3     | `isSpotting` 归一化                   |    6     |    6     |
| 3.4     | `vehicleAttributes` 称重/重车/关门车 |    8     |    0     |
| 4.3.1   | 普通 TRACK 目标                       |    5     |    0     |
| 4.3.2   | 存4南 当终点                          |    0     |    1     |
| 4.3.4   | 洗南/油/抛/轮 AREA                    |    4     |    0     |
| 4.3.5   | 大库 RANDOM × 修程                   |    3     |    0     |
| 4.3.6   | 大库外 RANDOM + 具体修N库外           |    5     |    0     |
| 4.3.7   | 上游下发 `机库:WEIGH`                 |    0     |    1     |
| 5.1     | 非法起点/终点/order                   |    0     |    6     |
| 5.2     | 牵引上限 (20 空 / 2 重 / 折算 / L1)   |    4     |    4     |
| 5.3     | 线路长度占用                          |    1     |    1     |
| 5.5     | 称重流程                              |    3     |    1     |
| 6.1     | 关门车规则                            |    4     |    5     |
| 7.1     | 调棚 SPOT 1–4                        |    4     |    0     |
| 7.2     | 洗南 SPOT 1–3                        |    3     |    0     |
| 7.3     | 机库随机 + 称重                       |    1     |    0     |
| 8.1     | NORMAL 01–05 / INSPECTION 01–07       |    3     |    1     |
| 8.2     | 长度→库位分配                         |    5     |    0     |
| 8.3     | 不兼容台位                            |    0     |    1     |
| **Total** |                                  | **64**   | **27**   |

## Baselines

| ID                        | Source                                                 |
| ------------------------- | ------------------------------------------------------ |
| `B1_clean`                | `validation_2025_11_06_afternoon.json` trimmed to 5 cars |
| `B2_busy_storage`         | `validation_20260103W.json`                            |
| `B3_busy_yard_normal`     | `validation_2025_12_08_noon.json` with 迎检 cleared    |
| `B4_busy_yard_inspection` | B3 with 8 cars flipped to `isSpotting=迎检`            |
| `B5_busy_shed`            | `validation_2025_11_11_noon.json`                      |
```

- [ ] **Step 2: Commit**

```bash
git add data/validation_inputs/README.md
git commit -m "Add curated rule-scenario library README"
```

---

## Task 31: Final full-suite run + cleanup

- [ ] **Step 1: Run the whole validation suite**

Run: `pytest tests/data/test_validation_inputs_structure.py -v`

Expected: all tests PASS.

- [ ] **Step 2: Run the whole project test suite as a regression check**

Run: `pytest -q`

Expected: no new failures compared to the baseline (pre-existing failures, if any, must be unchanged).

- [ ] **Step 3: Verify corpus layout**

```bash
ls data/validation_inputs/_baselines/ | wc -l   # expect 5
ls data/validation_inputs/positive/ | wc -l     # expect 64
ls data/validation_inputs/negative/ | wc -l     # expect 27
```

- [ ] **Step 4: If any step fails, fix inline and re-run**

No new commit unless a fix is required. The plan is complete when step 1 and step 3 pass.

---

## Out-of-scope reminders

Per design spec §10, the following are deliberately not in this plan:
- Wiring `data/validation_inputs/` into `scripts/run_external_validation_parallel.py`.
- Auto-generating cases from the spec.
- Golden HookPlan outputs.
- Solver or verifier changes.

These are future work; open separate plans when needed.
