# External Validation Baseline

Current workflow baseline for full external validation:

- Script: `scripts/run_external_validation_parallel.py`
- Solver: `beam`
- `beam_width=8`
- `heuristic_weight=1.0`
- `max_workers=8`
- `timeout_seconds=60`
- `retry_no_solution_beam_width`: default progressive recovery enabled
  - retry widths: `16`, then `24`
  - applies only to scenarios that fail with `No solution found`
  - does not retry `timeout` or contract/data errors

Baseline run artifact:

- Run dir: `artifacts/external_validation_parallel_runs/beam8_timeout60_all109_iter19_progressive_recovery_v1`
- Summary: local artifact only, ignored by git:
  `artifacts/external_validation_parallel_runs/beam8_timeout60_all109_iter19_progressive_recovery_v1/summary.json`

Baseline result snapshot:

- `scenario_count=109`
- `solved=94`
- `timeout=1`
- `no_solution=1`
- `other=13`

Recovered scenarios in this baseline:

- `validation_20260210W.json` at `recovery_beam_width=16`
- `validation_20260310Z.json` at `recovery_beam_width=16`
- `validation_20260327Z.json` at `recovery_beam_width=24`

Known remaining unsolved scenarios:

- `validation_20260104Z.json`: `timeout`
- `validation_20260109Z.json`: `No solution found`

Reproduction command:

```bash
python scripts/run_external_validation_parallel.py \
  --output-dir artifacts/external_validation_parallel_runs/beam8_timeout60_all109_iter19_progressive_recovery_v1 \
  --solver beam \
  --beam-width 8 \
  --heuristic-weight 1.0 \
  --timeout-seconds 60 \
  --max-workers 8 \
  --retry-no-solution-beam-width 24
```
