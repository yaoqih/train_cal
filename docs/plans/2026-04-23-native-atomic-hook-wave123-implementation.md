# Native Atomic Hook Wave 1-3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完成原生勾求解器的 Wave 1/2/3 改造：切掉 `constructive_partial` 假成功语义，把 8.2 长度分配偏好正式并入目标模型，并在不牺牲准确性/鲁棒性的前提下加入通用降勾势能与 staging 代价。

**Architecture:** 本轮只做原生勾主线，不向后兼容旧 PUT 优化语义。先硬切求解结果语义与 incumbent 比较逻辑，再把大库随机目标从“完全等价 allowed set”升级成“allowed + preferred + fallback”的分层目标，最后在 constructive / search / LNS 里接入终态势能、staging 代价和 carry 合并偏好。

**Tech Stack:** Python, pytest, Pydantic models, repo-local batch validation scripts, native atomic hook solver (`ATTACH` / `DETACH`)

---

### Task 1: Wave 1 设计落地与失败测试

**Files:**
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_anytime_integration.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_astar_solver.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/workflow/test_runner.py`

**Step 1: 写 failing tests，固定 Wave 1 目标语义**

覆盖以下行为：
- `constructive_partial` 返回时不得再被当作 solved success。
- anytime / beam fallback 不能把 invalid partial 当主 incumbent。
- 对外批跑口径里，invalid partial 应归类为 unsolved / invalid，不得继续混入 solved-success。

**Step 2: 跑定向测试，确认 RED**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_anytime_integration.py`
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_astar_solver.py -k 'constructive_partial or verification'`

Expected:
- 新增测试失败，失败点明确是当前 partial success 语义仍存在。

**Step 3: 最小实现 Wave 1**

实现点：
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/result.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/astar_solver.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/anytime.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/workflow/runner.py`

改动目标：
- `constructive_partial` 只做 seed / artifact，不再代表 solved success。
- fallback chain 只接受 valid incumbent；invalid partial 不能短路后续搜索。
- telemetry / result summary 对 invalid partial 单独标识。

**Step 4: 跑 Wave 1 相关测试**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_anytime_integration.py tests/solver/test_astar_solver.py tests/workflow/test_runner.py`

Expected:
- 全通过。

### Task 2: Wave 2 failing tests 与目标模型升级

**Files:**
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/io/test_normalize_input.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_move_generator.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_heuristic.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/verify/test_plan_verifier.py`

**Step 1: 写 failing tests，固定 8.2 语义**

覆盖以下行为：
- 长车 `>=17.6m` 只能 `修3/4库内`。
- 短车 `<17.6m` 优先 `修1/2库内`。
- `修1/2` 满时，短车允许降级到 `修3/4`。
- verifier / goal / move generation 一致遵守分层目标。

**Step 2: 跑定向测试，确认 RED**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/io/test_normalize_input.py tests/solver/test_move_generator.py tests/solver/test_heuristic.py tests/verify/test_plan_verifier.py`

Expected:
- 新增 8.2 测试失败，证明当前 solver 仍把 `修1..4库内` 当完全等价。

**Step 3: 最小实现 Wave 2**

实现点：
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/io/normalize_input.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/domain/depot_spots.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/state.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/move_generator.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/heuristic.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/verify/plan_verifier.py`

改动目标：
- GoalSpec 增加 `preferred_target_tracks` / `fallback_target_tracks`。
- `大库:RANDOM` 分层目标正式入核。
- spot allocation、state goal check、heuristic、verifier 统一使用分层目标。

**Step 4: 跑 Wave 2 相关测试**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/io/test_normalize_input.py tests/solver/test_move_generator.py tests/solver/test_heuristic.py tests/verify/test_plan_verifier.py`

Expected:
- 全通过。

### Task 3: Wave 3 failing tests 与降勾势能接入

**Files:**
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_constructive.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_heuristic.py`
- Modify: `/Users/huyaoqi/Documents/train_cal/tests/solver/test_astar_solver.py`

**Step 1: 写 failing tests，固定通用降勾行为**

覆盖以下行为：
- constructive 对 staging 污染更敏感，优先减少未完成车停在 staging。
- carry 非空时，只有能减少未来 DETACH 组数的继续 ATTACH 才被偏好。
- 对同 hook_count 的候选，优先 preferred-target / 低 staging / 低 unfinished 势能。

**Step 2: 跑定向测试，确认 RED**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_constructive.py tests/solver/test_heuristic.py tests/solver/test_astar_solver.py -k 'staging or preferred or carry or partial'`

Expected:
- 新增测试失败。

**Step 3: 最小实现 Wave 3**

实现点：
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/heuristic.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/constructive.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/move_generator.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/lns.py`
- `/Users/huyaoqi/Documents/train_cal/src/fzed_shunting/solver/search.py`

改动目标：
- 引入终态势能：unfinished、staging pollution、preferred-target violation。
- constructive `_score_move(...)` 接入 staging 代价与 carry merge 奖励。
- LNS / plan quality 比较加入 `valid_complete -> hook_count -> purity` 层次。

**Step 4: 跑 Wave 3 相关测试**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_constructive.py tests/solver/test_heuristic.py tests/solver/test_astar_solver.py`

Expected:
- 全通过。

### Task 4: 集成回归与 validation_inputs 并行验收

**Files:**
- No code changes required if前面实现已完成

**Step 1: 跑核心回归集**

Run:
- `PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_move_generator.py tests/solver/test_astar_solver.py tests/solver/test_constructive.py tests/workflow/test_runner.py tests/io/test_cli_flow.py tests/io/test_benchmark_runner.py tests/solver/test_anytime_integration.py tests/solver/test_budget_fallback.py tests/verify/test_replay.py tests/demo/test_view_model.py tests/solver/test_heuristic.py tests/data/test_validation_inputs_structure.py`

Expected:
- 全通过。

**Step 2: 并行批跑 validation_inputs**

Run:
- `PYTHONPATH=src .venv/bin/python scripts/run_external_validation_parallel.py --input-dir data/validation_inputs/truth --output-dir artifacts/validation_inputs_truth_wave123 --solver beam --beam-width 8 --max-workers 8 --timeout-seconds 120`
- `PYTHONPATH=src .venv/bin/python <repo-local positive validation runner>`
- `PYTHONPATH=src .venv/bin/python <repo-local negative validation runner>`

Expected:
- truth invalid 清零或显著下降
- positive 64/64
- negative pass 数显著高于现基线 17/27

**Step 3: 汇总指标**

输出：
- valid truth 数
- invalid truth 数
- positive / negative pass rate
- hook_count median / p75 / p90 对比前一轮基线

## Post-Wave Search Tightening

在 Wave 1/2/3 完成后，主链还补了两类和原生勾定义一致的收敛优化：

1. `constructive partial resume-search`
   - 不再只做 `h=1` 的极窄 warm-start。
   - 当 constructive partial 已经回到 honest 的空 carry 状态时，会从最近的空 carry checkpoint 继续做一次真实搜索补完。
   - 这保持了 partial 只是 artifact 的语义，不会把 invalid partial 当 solved，但能把 constructive 已经铺好的前缀真正利用起来。

2. `search priority` 正式接入 purity 次序
   - beam/weighted/exact 的 queue priority 在主代价相同的情况下，会优先更纯的状态：
     `unfinished -> preferred violation -> staging pollution`
   - 这样主搜索与 constructive / LNS 的 preferred/fallback 目标模型统一，不再只有 constructive 知道“preferred 更好”。

3. `depot TRACK occupancy` 语义补齐
   - 显式 `TRACK -> 修N库内` 的车辆也会占用真实库位。
   - 否则 `大库:RANDOM` 的 preferred/fallback 判断会把已经被固定目标占满的 `修1/2库内` 误判成仍有空位，导致 8.2 类场景方向错误。
