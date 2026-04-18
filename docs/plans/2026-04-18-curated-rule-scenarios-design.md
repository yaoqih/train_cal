# Curated Rule-Scenario Library Design

## 1. Context

Existing external validation inputs at `artifacts/external_validation_inputs/` and its `data/` subfolder
hold 129 production-derived scenarios. A scan of those scenarios against
`福州东调车业务说明.md` (chapters 3–8) shows broad real-world breadth but extremely narrow
branch/edge coverage:

| Dimension                                         | Current state                                      | Coverage   |
| ------------------------------------------------- | -------------------------------------------------- | ---------- |
| `repairProcess` 段 / 厂 / 临                       | 9882 / 694 / 141                                   | ✅         |
| `vehicleModel` 罐 / 棚 / 敞 / 平                   | P70 / C70 / G70 / X70 families all present         | ✅         |
| `targetTrack` line coverage                       | all main lines present                             | ✅         |
| `locoTrackName`                                   | **all 127 files = 机库**; 机北 / empty never seen  | ❌         |
| `isSpotting` non-empty                            | **10717 rows all empty**: 是 / 否 / 迎检 / 101–407 | ❌         |
| `targetAreaCode`                                  | **only 大库:RANDOM**; 调棚 / 洗南 / 油 / 抛 / 轮 / 大库外 all zero | ❌ |
| `targetMode=SPOT` (101–407 exact spot)            | 0 rows                                             | ❌         |
| `vehicleAttributes=关门车`                         | 0 rows                                             | ❌         |
| `vehicleAttributes=称重`                           | 3 rows                                             | ⚠️         |
| `yardMode=INSPECTION` (迎检)                       | 0 rows                                             | ❌         |
| Ch 5.2 boundaries (20-empty / 2-loaded / L1 190m) | no constructed cases                               | ❌         |
| Ch 5.3 track length saturation/overflow           | no constructed cases                               | ❌         |
| Ch 6 close-door rules (both branches)             | 0 rows                                             | ❌         |
| Ch 8 大库 short/long allocation and downgrade      | no explicit stress cases                           | ❌         |
| Illegal-input / negative scenarios                | 0 files                                            | ❌         |

The curated rule-scenario library is an input-side test corpus that fills these gaps.
It is not a replacement for the production baseline — the two serve different purposes
and must stay separate.

## 2. Goal

Produce an auditable, compact corpus of hand-picked JSON inputs that, taken together,
exercise every normative rule and every illegal-input rejection path declared in
`福州东调车业务说明.md` chapters 3–8, across a small set of realistic baseline yards.

## 3. Non-Goals

- Not a replacement for `artifacts/external_validation_inputs/` (production baseline stays).
- No golden HookPlan outputs. Each positive case declares soft bounds only.
- No wiring into `scripts/run_external_validation_parallel.py` in this design; the
  runner already globs `validation_*.json` only, so `case_*.json` files are isolated by default.
- No changes to solver, move-generator, verifier, or normalization logic.

## 4. Location and Structure

```
data/validation_inputs/
├── README.md                              # directory guide + coverage matrix
├── _baselines/
│   ├── B1_clean.json                      # empty-ish yard, loco=机库
│   ├── B2_busy_storage.json               # 存1/2/3 near saturation
│   ├── B3_busy_yard_normal.json           # 库1–4 all populated, NORMAL mode
│   ├── B4_busy_yard_inspection.json       # synthesized from B3 with 迎检 markers
│   └── B5_busy_shed.json                  # 调棚 partially occupied
├── positive/
│   └── case_<ch>_<rule-slug>_<variant>.json
└── negative/
    └── case_<ch>_<rule-slug>_<variant>.json
```

Naming: `case_<chapter>_<rule-slug>_<variant>.json`. Examples:
- `case_3_3_isSpotting_inspection_to_depot.json`
- `case_5_2_tow_empty_20cars.json`
- `case_5_2_tow_empty_21cars_neg.json`
- `case_6_1_close_door_cun4bei_top3_neg.json`

`_neg` suffix is not required — negative cases live under `negative/` and their
`metadata.expected_error` field is the authoritative signal — but the suffix helps
when cross-referencing from the coverage matrix.

## 5. Baseline Selection

| ID                                | Source (real file)                                     | Justification                             |
| --------------------------------- | ------------------------------------------------------ | ----------------------------------------- |
| `B1_clean`                        | `validation_2025_11_06_afternoon.json` trimmed         | 21 cars on storage, cleanest bucket       |
| `B2_busy_storage`                 | `validation_20260103W.json`                            | 52 cars on storage, lightest elsewhere    |
| `B3_busy_yard_normal`             | `validation_2025_12_08_noon.json`                      | full 20 cars in 库内 + 12 in 调棚         |
| `B4_busy_yard_inspection`         | synthesized from B3                                    | 迎检 markers on 6–10 cars to trigger 8.1 INSPECTION branch |
| `B5_busy_shed`                    | `validation_2025_11_11_noon.json`                      | 11 cars in 调棚, plenty of 存 variety     |

Every baseline keeps the full `trackInfo` catalogue of the source file so that
rules touching auxiliary lines (`渡N`, `联6/7`, `临1–4`) remain valid inputs.

Baseline construction:
1. Read the real source JSON.
2. Keep `trackInfo` verbatim.
3. For B1: remove most `vehicleInfo` rows to leave 3–8 cars across mixed starting tracks.
4. For B2/B3/B5: retain as-is, or drop up to 10 scattered cars to keep the file readable.
5. For B4: start from B3, flip 6–10 cars' `isSpotting` to `迎检` to establish plan-level
   INSPECTION mode.
6. Normalize `locoTrackName` per baseline (documented below).

## 6. Case File Shape

### 6.1 Positive case

```jsonc
{
  "metadata": {
    "spec_section": "5.2",
    "rule_id": "tow_empty_upper_bound",
    "variant": "20cars",
    "baseline": "B1_clean",
    "purpose": "20 辆纯空车连挂必须被接受",
    "expected_bounds": {
      "must_be_solvable": true,
      "max_hook_count": null,
      "must_visit_tracks": [],
      "close_door_constraint": null,
      "weigh_required_vehicles": []
    }
  },
  "trackInfo":     [ /* from baseline */ ],
  "vehicleInfo":   [ /* baseline + rule-specific mutations */ ],
  "locoTrackName": "机库"
}
```

### 6.2 Negative case

```jsonc
{
  "metadata": {
    "spec_section": "3.3",
    "rule_id": "isSpotting_on_storage_line",
    "variant": "cun1",
    "baseline": "B1_clean",
    "purpose": "isSpotting=是 不能用于普通存车线",
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

### 6.3 Assertion philosophy

Soft bounds only. We assert:
- `must_be_solvable` true/false.
- Numeric bounds such as `max_hook_count` (optional upper bound).
- Track-visitation requirements such as `must_visit_tracks: ["机库"]` for 称重 cases.
- Positional constraints such as
  `close_door_constraint: {"target_track": "存4北", "closed_door_allowed_positions": [4, 5, 6]}`.
- `weigh_required_vehicles: [vehicleNo...]` to assert the weighing step is scheduled.

We do not assert concrete hook sequences or which spot a car lands on (the solver
retains freedom).

## 7. Coverage Matrix

Total: **91 cases** — positive 64 / negative 27 — distributed over 5 baselines.

### 7.1 Chapter 3 — Input Contract (25 cases: 19 pos / 6 neg)

| Section | Rule / Variant                                     | Pos | Neg | Baseline         |
| ------- | -------------------------------------------------- | :-: | :-: | ---------------- |
| 3.2     | 调棚 WORK × 车型 (罐 / 平 / 敞)                    |  3  |  —  | B5               |
| 3.2     | 调棚 PRE_REPAIR × 起点 (预修 / 存5北)              |  2  |  —  | B5               |
| 3.3     | `isSpotting=迎检` (to 大库 / to 修3库内 SPOT)      |  2  |  —  | B4               |
| 3.3     | `isSpotting=SPOT` 101 / 203 / 305 / 407 boundaries |  4  |  —  | B3               |
| 3.3     | SPOT 台位 ↔ targetTrack 不匹配                     |  —  |  1  | B3               |
| 3.3     | `isSpotting=是` on 存1 / 临1 / 渡5                 |  —  |  3  | B1               |
| 3.3     | `isSpotting=未知值`                                |  —  |  1  | B1               |
| 3.3     | 同批 NORMAL + INSPECTION 冲突                      |  —  |  1  | B3+B4            |
| 3.4     | 称重 × 目标 (机库 / 存1)                           |  2  |  —  | B1               |
| 3.4     | 重车 × (单 / 2 辆 / 重+空混编)                     |  3  |  —  | B1               |
| 3.4     | 关门车 × 目标 (存1 / 大库 / 调棚)                  |  3  |  —  | B1               |

### 7.2 Chapter 4 — Target Expression (19 cases: 17 pos / 2 neg)

| Section | Rule / Variant                                               | Pos | Neg | Baseline |
| ------- | ------------------------------------------------------------ | :-: | :-: | -------- |
| 4.3.1   | 普通 TRACK 目标 (存1 / 存5南 / 预修 / 机棚 / 机北)           |  5  |  —  | B1       |
| 4.3.2   | 存4南 当终点                                                  |  —  |  1  | B1       |
| 4.3.4   | 洗南 / 油 / 抛 / 轮 默认 AREA                                 |  4  |  —  | B1       |
| 4.3.5   | 大库 RANDOM × 修程 (段 / 厂 / 临)                            |  3  |  —  | B3       |
| 4.3.6   | 大库外 RANDOM / 具体 修1–4 库外                               |  1+4|  —  | B3       |
| 4.3.7   | 上游直接下发 `机库:WEIGH`                                     |  —  |  1  | B1       |

### 7.3 Chapter 5 — Legality & Hard Constraints (20 cases: 8 pos / 12 neg)

| Section | Rule / Variant                              | Pos | Neg | Baseline |
| ------- | ------------------------------------------- | :-: | :-: | -------- |
| 5.1     | 走行线当起点 (渡1 / 联6)                    |  —  |  2  | B1       |
| 5.1     | 临停线当终点 (临1 / 临2)                    |  —  |  2  | B1       |
| 5.1     | order 重复 / order 非整数                   |  —  |  2  | B1       |
| 5.2     | 20 纯空车 / 21 超限                         |  1  |  1  | B1       |
| 5.2     | 2 混编重车 / 3 混编重车                     |  1  |  1  | B1       |
| 5.2     | 1 重+16 空 / 1 重+17 空 (折算验证)           |  1  |  1  | B1       |
| 5.2     | L1 道岔机后 190m / 195m                     |  1  |  1  | B1       |
| 5.3     | trackDistance 刚满 / 溢出                   |  1  |  1  | B2       |
| 5.5     | 称重 × 变体 (单车 / 多车多钩 / 目标非机库)  |  3  |  —  | B1       |
| 5.5     | 单钩 2 辆称重车                             |  —  |  1  | B1       |

### 7.4 Chapter 6 — Close-Door Rules (9 cases: 4 pos / 5 neg)

| Section | Rule / Variant                                         | Pos | Neg | Baseline |
| ------- | ------------------------------------------------------ | :-: | :-: | -------- |
| 6.1     | 非存4北、机后=10、关门车首位 (允许)                    |  1  |  —  | B1       |
| 6.1     | 非存4北、机后=11 / 15、关门车首位                      |  —  |  2  | B1       |
| 6.1     | 非存4北、机后=11、关门车第 2 位 (允许)                  |  1  |  —  | B1       |
| 6.1     | 存4北、关门车第 4 位                                    |  1  |  —  | B2       |
| 6.1     | 存4北、关门车第 1 / 2 / 3 位                            |  —  |  3  | B2       |
| 6.1     | 一钩多辆关门车                                           |  1  |  —  | B1       |

### 7.5 Chapter 7 — Special Lines & Spot Numbering (8 cases: 8 pos)

| Section | Rule / Variant                        | Pos | Neg | Baseline |
| ------- | ------------------------------------- | :-: | :-: | -------- |
| 7.1     | 调棚 1 / 2 / 3 / 4 号位 SPOT          |  4  |  —  | B5       |
| 7.2     | 洗南 1 / 2 / 3 号位 SPOT              |  3  |  —  | B1       |
| 7.3     | 机库随机 + 称重执行顺序                |  1  |  —  | B1       |

### 7.6 Chapter 8 — 大库 Spot Allocation (10 cases: 8 pos / 2 neg)

| Section | Rule / Variant                                            | Pos | Neg | Baseline |
| ------- | --------------------------------------------------------- | :-: | :-: | -------- |
| 8.1     | NORMAL 用 01–05 / 强制 06                                 |  1  |  1  | B3       |
| 8.1     | INSPECTION 扩到 07 (106 / 307)                            |  2  |  —  | B4       |
| 8.2     | ≥17.6m → 3 / 4 库 (17.6m / 19.2m)                         |  2  |  —  | B3       |
| 8.2     | <17.6m → 优先 1 / 2 库 (14.3m / 16.0m)                     |  2  |  —  | B3       |
| 8.2     | 1 / 2 库位满时降级到 3 / 4 库                              |  1  |  —  | B3       |
| 8.3     | 未指定台位直接送入不兼容台位                              |  —  |  1  | B3       |

### 7.7 Orthogonal variations applied across cases

Beyond the per-rule variants enumerated above, these orthogonal mutations are
applied opportunistically so the 90-case corpus samples them:

| Dimension        | Values                                  | At least one case per value? |
| ---------------- | --------------------------------------- | ---------------------------- |
| `locoTrackName`  | 机库 / 机北 / 空                         | yes (one case each) |
| Vehicle length   | 14.3m / 17.6m / 19.2m                   | yes (via Ch 8 cases)         |
| Vehicle model    | 罐 / 棚 / 敞 / 平                        | yes (via Ch 3.2 / 3.4 cases) |
| Repair process   | 段修 / 厂修 / 临修                       | yes (via Ch 4.3.5 cases)     |

## 8. Authoring Workflow

Cases are hand-authored (not auto-generated). The authoring workflow for one case:

1. Pick the baseline that fits the rule's environmental need.
2. Start from a copy of the baseline JSON.
3. Mutate `vehicleInfo` minimally to trigger the rule: add / remove / edit rows,
   change `targetTrack`, `targetMode`, `isSpotting`, `vehicleAttributes`, `vehicleLength`.
4. Strip `vehicleInfo` down to the smallest set that still keeps the scenario
   self-consistent (typically ≤ 15 cars for rule cases, up to baseline size for
   saturation cases).
5. Fill `metadata` header with spec section, rule id, variant, baseline id, purpose,
   expected bounds or expected error.
6. Save under `positive/` or `negative/` with the naming convention.

## 9. Maintenance

- When `福州东调车业务说明.md` adds or revises a rule, the README coverage matrix
  is the canonical checklist for whether new cases are needed.
- Baseline files (`_baselines/`) are considered frozen once committed; further
  variants should be expressed as case-level mutations, not baseline edits.
- A case whose `metadata.baseline` no longer aligns with the referenced baseline
  (e.g., because rule evolved) must be either updated or removed in the same PR
  that changes the rule.

## 10. Out of Scope (defer)

- Wiring `data/validation_inputs/` into the external-validation parallel runner.
- Auto-generating cases from the spec.
- Golden HookPlan outputs.
- Mutation testing or property-based fuzzers.
