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
