# 大库钩后调（depot-late-scheduling）设计

## 1. 需求

**业务动机**：大库检修工序周期最长，在一次调车计划的时间窗内，大库相关的取送往往要等到工序后段才具备条件（新车入库需要等既有库位腾空；出库车需要等修竣）。因此希望最终钩计划里，涉及大库的钩尽量集中在后半程。

**范围确认**：

- **方向**：进大库 + 出大库双向都要延后。
- **层面**：只调整**输出钩计划顺序**；不修改业务流程分段、不强行两阶段求解。
- **度量**：**平均较后**即可，不要求严格中点分界或先后完全分阶段。
- **强度**：先按硬约束思路实现（搜索强制优先延后大库钩），对外仍通过开关 `enable_depot_late_scheduling` 默认关闭；基准跑完验证影响后再决定是否翻转默认。

## 2. 术语与口径

**大库相关钩（depot-touching hook）** 定义：一条 `HookAction` H，当且仅当 `H.source_track ∈ DEPOT_INNER_TRACKS` 或 `H.target_track ∈ DEPOT_INNER_TRACKS` 时，为大库相关钩。

- `DEPOT_INNER_TRACKS = {"修1库内", "修2库内", "修3库内", "修4库内"}`
- 该常量已存在于 `src/fzed_shunting/solver/constructive.py:37`，直接复用。
- `修N库外` 是进出库衔接存车线，**不计入**大库相关（与《福州东车辆段业务说明》4.1 节口径一致）。

**延后得分（depot-earliness metric）** 定义：对长度为 N 的钩计划，

```
depot_earliness = Σ (N - i)   for each depot-touching hook at 1-indexed position i
```

`depot_earliness` 越小越好；0 表示没有大库钩或所有大库钩在末尾。该度量等价于"最大化大库钩的平均索引"，对同一 N 下不同排列有严格排序。

## 3. 设计哲学

保留现有两层架构（构造保底 + anytime 搜索）与核心不变式：

1. **主目标不动**：`总勾数最少` 仍是第一目标；大库延后是**严格次级**（词典序）目标，不得以增加 1 勾为代价换取更晚的大库钩。
2. **启发式不动**：`heuristic.py` 里的可采纳下界不变，不影响 exact 模式的最优性证明。
3. **关闭时零影响**：`enable_depot_late_scheduling=False`（默认）时，产出钩计划与当前基线**逐位相同**；新增字段只有 `SolverResult.depot_earliness`（观察用）。
4. **开启时最小侵入**：只在 cost 元组的次级键和终态后处理里引入大库延后考量；LNS 修补按 `(hook_count, depot_earliness, branch_count, length)` 词典序比较。

## 4. 实现方案

### 4.1 新模块 `solver/depot_late.py`

承载延后策略的全部纯函数：

```python
DEPOT_INNER_TRACKS = frozenset({"修1库内", "修2库内", "修3库内", "修4库内"})

def is_depot_hook(hook: HookAction) -> bool: ...

def depot_earliness(plan: Sequence[HookAction]) -> int: ...

def reorder_depot_late(
    plan: Sequence[HookAction],
    initial_state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[HookAction]:
    """贪心相邻交换，把大库钩尽量往后推。

    约束：
    - 交换前后 replay 都必须合法（replay_plan 成功且终态相同）。
    - 只在严格改善 depot_earliness 时接受交换。
    - 最多遍历 N*N 次（N 为钩数），实测 N<=50 可接受。
    """
```

- `is_depot_hook` 与 `depot_earliness` 无副作用，易单测。
- `reorder_depot_late` 使用 `verify.replay.replay_plan` 做等价性校验，不自行重算状态（借力现有验证器）。

### 4.2 搜索层改造 `solver/search.py`

`_priority()` 当前返回 `tuple[float, int, int, int, int]`。新增参数 `neg_depot_index_sum: int` 作为**次级**键：

```python
def _priority(*, cost, heuristic, blocker_bonus=0, solver_mode, heuristic_weight,
              neg_depot_index_sum: int = 0) -> tuple[...]:
    if solver_mode == "beam":
        ...
        return (cost + adjusted_heuristic, cost, neg_depot_index_sum, adjusted_heuristic, -blocker_bonus, heuristic)
    if solver_mode == "weighted":
        return (cost + heuristic_weight * heuristic, cost, neg_depot_index_sum, heuristic, -blocker_bonus)
    return (cost + heuristic, cost, neg_depot_index_sum, heuristic, -blocker_bonus)
```

- 次级键放在 `cost` 之后、启发式之前：保证词典序"先按 f 比较，f 相等再按大库延后度比较"。
- `enable_depot_late_scheduling=False` 时所有调用点传 `neg_depot_index_sum=0`，退化为旧行为。
- `QueueItem.priority` 的类型签名跟着拓宽为 `tuple[float, int, int, ...]`；无破坏性变更（仅增维度）。

**搜索时的增量度量**：终态指标 `depot_earliness = Σ(N-i)` 依赖最终 N，搜索中未知；改为维护：

```
depot_index_sum(partial_plan) = Σ i  for each depot hook at 1-indexed position i
```

- 每次 `_apply_move` 扩展一个新钩：若新钩是大库钩，`depot_index_sum += (L + 1)`（L 是扩展前的长度）；否则不变。
- **大库钩越靠后 ⇔ `depot_index_sum` 越大**。
- min-heap 用负值做次级键：`secondary = -depot_index_sum`，于是"更大的索引和"对应"更小的次级键"，优先扩展。
- 等价关系：对同一 N，`depot_earliness = K*N - depot_index_sum`（K 是大库钩数），两者相差一个只依赖 (N, K) 的常数，词典序排序结果一致。

### 4.3 LNS 比较 `solver/lns.py`

`_plan_quality()` 当前返回 `(len, branch_count, length_m)`。改为：

```python
def _plan_quality(plan, route_oracle, *, depot_late: bool = False) -> tuple:
    n = len(plan)
    branch, length = ...  # 不变
    if depot_late:
        earliness = depot_earliness(plan)
        return (n, earliness, branch, length)
    return (n, branch, length)
```

`_is_better_plan` 和 `_improve_incumbent_result` 透传 `depot_late` 标志。

### 4.4 对外 API `solver/astar_solver.py`

`solve_with_simple_astar_result` 新增一个 kwarg：

```python
def solve_with_simple_astar_result(
    ...,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult:
```

- 透传到 search、LNS、constructive（constructive 暂不改，只是把 flag 存到 debug_stats 以便对比）。
- 终态处理：`_attach_verification` 之前若 flag 开，跑一次 `reorder_depot_late`，把 plan 替换为重排后的版本，再送 verifier。
- **失败降级**：`reorder_depot_late` 内部任何交换导致 replay 失败都丢弃该交换，保持原序。最坏退化到"不重排"，不会引入新的无效计划。

### 4.5 SolverResult 增强 `solver/result.py`

新增可选字段：

```python
@dataclass
class SolverResult:
    ...
    depot_earliness: int | None = None  # 最终 plan 的 earliness 分数；None 表示未计算
    depot_hook_count: int | None = None
```

关闭开关时也可以填（只读观察），用于 bench 对比。

## 5. 测试策略

### 5.1 单元（`tests/solver/test_depot_late.py`）

1. `is_depot_hook` 覆盖：修1库内→存4北（True）、存5北→修3库内（True）、存1→调北（False）、修1库外→修1库内（True，因为 target 是库内）、修1库内→修1库外（True，因为 source 是库内）。
2. `depot_earliness` 覆盖：空计划、无大库钩、单大库钩位于首/中/末、多大库钩。
3. `reorder_depot_late`：
   - 可交换场景：两个独立钩，大库在前，交换后 earliness 严格减小。
   - 不可交换场景（依赖）：后钩 source == 前钩 target；必须拒绝。
   - 不改变终态：重排前后 `replay_plan` 终态 `track_sequences` 严格相等。

### 5.2 集成

1. **关闭开关零影响**：`typical_suite` (13) + `typical_workflow_suite` (17) 跑一遍关闭与开启，关闭版本的 `plan`、`hook_count` 与主线完全一致（byte-equal）。
2. **开启对 valid 的影响**：同两套 + `external_validation_inputs` (109)，开启版本必须满足：
   - 所有原先 valid 的条目仍然 valid（97/109 不降）。
   - 总勾数 ≤ 基线（词典序次级不应增加勾数）。
3. **earliness 改善统计**：在报告里打印开启前后 `depot_earliness` 分布；目标是均值下降、方差下降。

### 5.3 回归

1. 现有 150 条 solver 测试全绿。
2. 运行一次完整的 `external_validation_inputs` bench（180s/条，8 worker），对比 `artifacts/external_validation_parallel_runs/summary.json`。

## 6. 里程碑与开关策略

1. **里程碑 1**：`depot_late.py` + 单测 PR，不接通主链路。
2. **里程碑 2**：搜索/LNS/astar_solver 接通，默认 `enable_depot_late_scheduling=False`，所有集成测试关闭开关下绿。
3. **里程碑 3**：开启开关跑 `typical_suite` + `typical_workflow_suite` + 109 bench，对比 earliness 与 valid 率。
4. **里程碑 4**：基于里程碑 3 的数据，决定是否默认开启：
   - 若 valid 率不降、勾数不增：翻转默认为 True。
   - 若有可控劣化：保留默认 False，在调用方显式开启。
   - 若劣化严重：回到设计讨论。

## 7. 风险与回退

| 风险 | 触发 | 处理 |
|---|---|---|
| 次级键干扰 beam/weighted 剪枝，产生更差主目标 | 开启后 `external_validation_inputs` 勾数上升 | 把次级键范围压缩为"仅在 f、cost 都相等时生效"（更严格词典序），或退回纯后处理版 |
| `reorder_depot_late` 漏算一种依赖，导致 replay 失败 | 验证器报 invalid | 单测补、后处理捕获异常丢弃该次交换，最差等于不重排 |
| LNS 把词典序误当主目标追求，破坏 repair 进度 | LNS 阶段命中率下降 | `_plan_quality` 里次级权重只在 flag=True 时启用，默认保持旧元组 |
| 与已回滚的 preallocation 混淆 | 历史代码污染 | 新模块名 `depot_late`，与旧 `preallocation` 命名无交集；文件独立 |

## 8. 不做的事（YAGNI）

- 不建时间轴/时间窗模型（solver 不建模实际时钟）。
- 不改 `constructive.py` 的 tier 评分（先观察搜索层的效果，避免过度修改保底）。
- 不引入单独的"两阶段求解"；路线 C 的风险已在 preallocation 事件验证过。
- 不在业务层/IO 层动 `NormalizedPlanInput` 数据结构。
