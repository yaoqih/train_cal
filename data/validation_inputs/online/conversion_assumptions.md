# Online Conversion Assumptions

- Source workbooks under `取送车计划/new` are treated as single-snapshot online plan inputs, not paired Start/End snapshots.
- Each data row with a `车号` becomes one `vehicleInfo` row; both left and right table blocks are parsed.
- A sheet is used only when its title matches the workbook suffix, either directly as `20260402W` or by date/period such as `2026.04.02下午`; residual sheets with other dates are ignored.
- Blank `目标股道` means the vehicle is kept in the payload as a current-track occupancy row with `targetMode=SNAPSHOT` and `targetSource=ONLINE_EMPTY_TARGET`.
- Explicit online targets `修1/修2/修3/修4` are preserved as those short target names in JSON; normalization interprets them as the corresponding concrete inner-depot tracks.
- `目标股道=存4` is normalized to `存4北` because `存4南` is not a legal final destination.
- `车辆属性` from the workbook is preserved when present; otherwise legacy repair-derived attributes such as `重 -> 重车` are used.
- Total blank-target rows preserved as occupancy rows: 232

## Ignored Sheets
- 调车计划编制_20260402W.xlsx: 起点 (2)