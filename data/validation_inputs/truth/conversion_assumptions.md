# Data Conversion Assumptions

- Only `Start` and `End` sheets are treated as scenario sources; `map` and presentation sheets are ignored.
- The header row is detected by required columns `股道/序号/车型/车号/修程/备注` within the first six rows.
- Old aliases are normalized as follows: `调梁库 -> 调棚`, `喷漆库 -> 油`, `洗罐库内 -> 洗南`, `洗罐库外 -> 洗北`, `洗罐线 -> 洗南`.
- Raw repair values are normalized into the current contract with `称重 -> vehicleAttributes=称重`, `重 -> vehicleAttributes=重车`, direct `段修/厂修/临修` preserved, and other legacy labels folded into `段修`.
- Start-sheet model backfills from End: 0
- End-sheet model backfills from Start: 58
- Start-sheet repair backfills from End: 6
- End-sheet repair backfills from Start: 148