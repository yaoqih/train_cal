# 福州东调车求解器架构

## 设计哲学：构造保底 + 搜索改良

生产调车系统的 SLA 约束是**任何合法输入都必须返回合法的勾计划**，而不是"找到最优解或返回空"。本求解器采用**两层架构**满足该约束：

```
┌─────────────────────────────────────────────────────────────┐
│                   solve_with_simple_astar_result            │
│                                                             │
│   ┌───────────────┐   ┌───────────────┐   ┌──────────────┐ │
│   │ Constructive  │   │    Anytime    │   │  Verification│ │
│   │   Baseline    │→→│   Search     │→→│   & Replay   │ │
│   │   (always     │   │  (改善质量)   │   │  (独立验证)  │ │
│   │   有解)       │   │               │   │              │ │
│   └───────────────┘   └───────────────┘   └──────────────┘ │
│          ↑                   ↑                              │
│          └── priority rules  └── exact A* + weighted + beam │
│              + LNS repair        + anytime fallback chain   │
└─────────────────────────────────────────────────────────────┘
```

- **下层（Constructive baseline）**：优先级规则分派器，每步按 7-tier 评分选择局部最优移动。保证"只要 move_generator 为非目标态产出至少一个合法动作，就能终止于合法 plan"。这层是 SLA 的保底。
- **上层（Anytime search）**：exact A* → weighted A* → beam search → 多个 greedy 变种。每阶段限定预算份额；只在改善勾数时替换增量解（incumbent）。
- **LNS 改良**：对 beam 和 LNS 模式解，用 4 种 destroy 算子（hotspot、worst-cost、target-cluster、equidistant）循环修补，缩短勾数。

最终保证：**给更多时间只会更短，不会更差**。

## 模块职责

所有求解代码位于 `src/fzed_shunting/solver/`：

| 模块 | 职责 | 行数 |
|---|---|---|
| `types.py` | `HookAction` 数据结构 | ~30 |
| `result.py` | `SolverResult` + `PlanVerificationError` | ~50 |
| `state.py` | 纯态空间助手（`_apply_move`、`_is_goal`、`_state_key` 等） | ~140 |
| `budget.py` | `SearchBudget`：时间 + 节点双预算 | ~50 |
| `heuristic.py` | 可采纳启发式（`h_distinct_transfer_pairs` 等） | ~270 |
| `move_generator.py` | 合法动作生成 + 关门车规则 + 容量校验 | ~650 |
| `constructive.py` | 优先级规则分派器（保底层） | ~360 |
| `search.py` | A\*/weighted/beam 核心循环 + 队列剪枝 | ~370 |
| `lns.py` | Destroy 算子 + 修补 + 计划质量比较 | ~320 |
| `anytime.py` | 7 阶段 fallback chain + 预算份额 | ~150 |
| `astar_solver.py` | 对外入口 + 构造阶段 + 验证胶水 + re-exports | ~340 |

## 对外 API

```python
from fzed_shunting.solver.astar_solver import (
    solve_with_simple_astar,          # 便捷 API，直接返回 plan
    solve_with_simple_astar_result,    # 完整 API，返回 SolverResult
)
```

核心签名：

```python
def solve_with_simple_astar_result(
    plan_input: NormalizedPlanInput,
    initial_state: ReplayState,
    master: MasterData | None = None,
    solver_mode: str = "exact",           # "exact" | "weighted" | "beam" | "lns"
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    time_budget_ms: float | None = None,  # 总预算
    node_budget: int | None = None,
    verify: bool = True,                   # 启用独立验证（默认）
    enable_anytime_fallback: bool = True,  # 启用 fallback chain
    enable_constructive_seed: bool = True, # 启用构造保底
) -> SolverResult
```

## 关键不变式

1. **SLA 保证**：当 `enable_constructive_seed=True`（默认）时，只要输入在 `_validate_final_track_goal_capacities` 预检通过，`result.plan` 永不为空。
2. **Anytime 单调性**：fallback chain 每阶段只在"产出更短 plan"时替换 incumbent；给更多时间只会改善，不会劣化。
3. **Verify 完整性**：`verify=True` 时，最终 plan 被 `plan_verifier.verify_plan` 独立重放，通过了才返回。
4. **最优性保留**：`solver_mode="exact"` 且 exact 阶段耗尽（未超预算）时，`is_proven_optimal=True`。

## Fallback Chain 顺序

```python
1. exact        heuristic_weight=1.0                  # 40% 总预算
2. weighted     heuristic_weight=max(w, 1.5)          # 8%
3. beam         beam_width=64,  heuristic_weight=1.5  # 8%
4. beam_greedy_64  beam_width=64,  heuristic_weight=5.0  # 8%
5. weighted_greedy heuristic_weight=5.0               # 12%
6. beam_greedy_128 beam_width=128, heuristic_weight=5.0  # 8%
7. beam_greedy_256 beam_width=256, heuristic_weight=5.0  # 8%
8. weighted_very_greedy heuristic_weight=10.0         # 8%
```

第一个产出非空 plan 的阶段即为最终解（除非后续阶段产出更短解）。若全部阶段都返回空，退回到构造保底。

## 启发式（`compute_admissible_heuristic`）

当前启发式取以下下界的**最大值**（保持可采纳性）：

- `h_distinct_transfer_pairs`：不同 (源, 目标) 对数 + 多目标源桶数
- `h_misplaced_vehicles`：车辆数 - 已在目标轨的车辆数
- `h_blocking`：每个错位车都需要至少 1 勾
- `h_weigh`：需称重车数
- `h_spot_evict`：SPOT 模式下需挪位的车

## 验证流程

`_attach_verification` 对最终 plan 调用 `verify/plan_verifier.verify_plan`，独立重放得到 `VerifyReport`：

- `is_valid`：所有规则通过
- `errors`：违反的规则列表（带 hook_no 定位）
- `hook_reports`：逐勾细粒度诊断

Partial plan（`fallback_stage="constructive_partial"`）不会抛出 `PlanVerificationError`，而是附带未通过的 report 返回；调用方自行决定是否接受。

## 生产预算建议

- 线上 p50 场景：`time_budget_ms=5_000`
- 线上 p99 复杂场景：`time_budget_ms=30_000`
- 离线 benchmark：`time_budget_ms=180_000`

## 当前求解状态

### 109 全量外部验证

最新基线跑的是 `exact` 模式、180s 超时（单条预算 175s）、8 worker 并发。工件：`artifacts/external_validation_parallel_runs/summary.json`。

当前状态应当拆开看，而不是简单说"99/99 可解"：

| 数据集 | 总数 | 返回 plan | verifier 通过 | verifier 失败 | 预检拒绝 | 空返 |
|---|---|---:|---:|---:|---:|---:|
| `external_validation_inputs` | 109 | 99 | 97 | 2 | 10 | 0 |

说明：

- `99` 条返回了 plan，其中 `97` 条 `is_valid=true`
- `2` 条是"有 plan 但 verifier 未通过"，都落在 `fallback_stage=constructive_partial`
- `10` 条不是搜索失败，而是最终容量预检阶段直接拒绝

**验证失败的 2 条**：

- `validation_20260105W.json`
- `validation_20260310W.json`

这 2 条的共同特征：构造保底层给出了可执行的 partial plan，但最终落位没有满足所有车辆的允许终点约束。详细勾数分布、阶段命中、耗时统计见 `docs/solver-deep-dive.md` §17。

### Fixed Valid 工件

本轮也用 `exact` 口径复核了两套固定验证工件：

| 工件 | 场景数 | valid | 总勾数 |
|---|---:|---:|---:|
| `typical_suite` | 13 | 13 | 21 |
| `typical_workflow_suite` | 17 | 17 | 39 |

固定集是稳定的。109 全量外部验证上更精确的表述是：

- `97/109` 达到"返回 plan 且 verifier 通过"
- `2/109` 仅达到"partial constructive plan 可返回，但 verifier 失败"
- `10/109` 在容量预检阶段被拒绝

离群点 `validation_20260205W`（89 车、20 个随机仓库目标）当前落在 200+ 勾；尝试过归一化前预分配和 move_generator top-1 候选两种压缩方案都破坏了搜索弹性，已回滚。下一步压缩需要 CP-SAT 子问题分解或 landmark 启发式。

## 贡献指南

- 新增 destroy 算子：在 `lns.py` 中添加 `_cut_points_*`，纳入 `_candidate_repair_cut_points`
- 新增 fallback 阶段：在 `anytime.py` 的 `fallback_stages` 追加元组，重新分配 `budget_share` 使总和 ≤ 0.6
- 新增启发式分量：在 `heuristic.py` 添加 `h_*` 函数，纳入 `compute_admissible_heuristic` 的 `max()` 集合；**必须保持可采纳性**（h ≤ 真实代价）
- 新增状态转移规则：在 `move_generator.py` 实现动作候选，在 `state.py._apply_move` 实现状态推进
