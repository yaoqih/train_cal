# 大库钩后调（depot-late-scheduling）设计

## 1. 需求

**业务动机**：大库检修工序周期最长，在一次调车计划的时间窗内，大库相关的取送往往要等到工序后段才具备条件（新车入库需要等既有库位腾空；出库车需要等修竣）。因此希望最终钩计划里，涉及大库的钩尽量集中在后半程。

**范围确认**：

- **方向**：进大库 + 出大库双向都要延后。
- **层面**：只调整**输出钩计划顺序**；不修改业务流程分段、不强行两阶段求解。
- **度量**：**平均较后**即可，不要求严格中点分界或先后完全分阶段。
- **强度**：作为**必需行为**落地。开关 `enable_depot_late_scheduling` 默认 `True`，提供给 bench 做对照的 `False` 通路仍保留，不作为日常开关。里程碑 3 的基线数据揭示出搜索次级键对 weighted/beam 模式的干扰（详见 §7），已按 §4.2 的 mitigation 只在 `solver_mode == "exact"` 模式下启用搜索次级键。

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

保留现有两层架构（构造保底 + anytime 搜索）与核心不变式。大库延后现为必需行为（默认开启），但以下不变式仍然硬约束：

1. **主目标不动**：`总勾数最少` 仍是第一目标；大库延后是**严格次级**（词典序）目标，不得以增加 1 勾为代价换取更晚的大库钩。
2. **启发式不动**：`heuristic.py` 里的可采纳下界不变，不影响 exact 模式的最优性证明。
3. **关闭时零影响**（保留为 bench 对照通路）：`enable_depot_late_scheduling=False` 时，产出钩计划与基线**逐位相同**；`SolverResult.depot_earliness` 字段仍然填充（观察用）。
4. **默认开启，最小侵入**：搜索次级键仅在 `solver_mode == "exact"` 模式生效（§4.2 里程碑 4 mitigation）；LNS 修补按 `(hook_count, depot_earliness, branch_count, length)` 词典序比较；构造层同 tier 内有大库惩罚；终态过 `reorder_depot_late` 并在交换候选上再跑一遍完整验证器，不合法则回退原 plan。

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
- **里程碑 4 mitigation（已应用）**：在 `_solve_search_result` 里只有当 `enable_depot_late_scheduling and solver_mode == "exact"` 时才把真实的 `-depot_index_sum(next_plan)` 写入次级键；weighted/beam/greedy 模式恒传 `0`。理由：exact 模式有可采纳启发式保证 f-tie 时的词典序不会破坏主目标；而 weighted/beam/greedy 模式下 f 值几乎不会 tied，次级键会在非 tie 的情况下直接影响扩展/剪枝决策，实测在 `validation_20260112W.json`、`validation_20260116W.json` 上造成 hook 数反向上升（17→18、18→20）。这些模式下的大库延后靠 LNS 比较层与终态 `reorder_depot_late` 承担。
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

### 4.4 构造层注入 `solver/constructive.py`

`_score_move()` 当前返回的评分元组结构：

```python
score = (tier, is_spot_or_area_finalization_flag, -delta, -block_size, path_length, source_track, target_track)
```

**注入点**：在 `tier` 之后、`is_spot_or_area_finalization_flag` 之前新增一个 `depot_late_penalty` 分量：

```python
score = (
    tier,
    depot_late_penalty,  # 新增，0 或 1
    0 if is_spot_or_area_finalization else 1,
    -delta, -block_size, path_length, source_track, target_track,
)
```

**`depot_late_penalty` 计算**：

```python
depot_late_penalty = 1 if (
    enable_depot_late_scheduling
    and is_depot_hook(move)
    and _is_early_depot_phase(state, plan_input)
) else 0
```

**`_is_early_depot_phase` 定义**：

当存在任一"目标集合完全不含大库"的车辆 v 尚未到达其任一允许目标时，返回 `True`；否则 `False`。语义上即"仍有非大库目标未完成"。

```python
def _is_early_depot_phase(state: ReplayState, plan_input: NormalizedPlanInput) -> bool:
    for v in plan_input.vehicles:
        non_depot_targets = v.goal.allowed_target_tracks - DEPOT_INNER_TRACKS
        if not non_depot_targets:
            continue  # 纯大库目标的车不参与"早期相"判定
        current = _locate_vehicle(state, v.vehicle_no)
        if current not in v.goal.allowed_target_tracks:
            return True
    return False
```

**关键点**：

1. **不动 tier**：`depot_late_penalty` 只在**同 tier 内**起作用；当大库动作处于 tier 0/1/2（关门车/称重/forward progress）而同 tier 无非大库候选时，仍然被选中。避免破坏保底层的推进能力。
2. **phase 判定基于状态而非勾数**：与"非大库工序先完成"的业务直觉对齐；不需要估计最终 N。
3. **开关关闭时行为完全一致**：`enable_depot_late_scheduling=False` 时 `depot_late_penalty` 恒 0，元组多一维但排序结果同旧版。
4. `_choose_best_move` 透传 `enable_depot_late_scheduling` 和 `plan_input` 参数；主循环 `solve_constructive` 新增同名参数，默认 False。

### 4.5 对外 API `solver/astar_solver.py`

`solve_with_simple_astar_result` 新增一个 kwarg：

```python
def solve_with_simple_astar_result(
    ...,
    enable_depot_late_scheduling: bool = False,
) -> SolverResult:
```

- 透传到 search、LNS、constructive 各入口。
- 终态处理：`_attach_verification` 之前若 flag 开，跑一次 `reorder_depot_late`，把 plan 替换为重排后的版本，再送 verifier。
- **失败降级**：`reorder_depot_late` 内部任何交换导致 replay 失败都丢弃该交换，保持原序。最坏退化到"不重排"，不会引入新的无效计划。

### 4.6 SolverResult 增强 `solver/result.py`

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

### 5.1 单元

**`tests/solver/test_depot_late.py`**（Wave 1-A）

1. `is_depot_hook` 覆盖：修1库内→存4北（True）、存5北→修3库内（True）、存1→调北（False）、修1库外→修1库内（True，因为 target 是库内）、修1库内→修1库外（True，因为 source 是库内）。
2. `depot_earliness` / `depot_index_sum` 覆盖：空计划、无大库钩、单大库钩位于首/中/末、多大库钩；两个度量对同一 plan 的关系 `earliness + index_sum = K*N` 成立。
3. `reorder_depot_late`：
   - 可交换场景：两个独立钩，大库在前，交换后 earliness 严格减小。
   - 不可交换场景（依赖）：后钩 source == 前钩 target；必须拒绝。
   - 不改变终态：重排前后 `replay_plan` 终态 `track_sequences` 严格相等。
4. `_is_early_depot_phase`：全部非大库目标已就位 → False；有非大库目标未就位 → True；输入仅含大库目标车 → False（边界）。

**`tests/solver/test_search.py`**（Wave 2-C，若无则新建）

- `_priority()` 的次级键在 `neg_depot_index_sum=0` 时返回元组与旧版字节一致。
- 两个同 f 的节点，大库钩索引和更大的优先弹出。

**`tests/solver/test_lns.py`**（Wave 2-D，扩展现有文件）

- `_is_better_plan`：flag 关闭时同 len 不同 earliness 按 branch/length 比较；flag 开启时 earliness 胜出。

**`tests/solver/test_constructive.py`**（Wave 2-E，扩展现有文件）

- 构造两个同 tier 候选（一个大库一个非大库），开关开启且早期相时非大库被选中；开关关闭或晚期相时按旧版 delta/block_size 决定。

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

1. **里程碑 1（已完成）**：`depot_late.py` + 单测 PR，不接通主链路。
2. **里程碑 2（已完成）**：搜索/LNS/astar_solver 接通，默认 `enable_depot_late_scheduling=False`，所有集成测试关闭开关下绿。
3. **里程碑 3（已完成）**：开启开关跑 `typical_suite` + `typical_workflow_suite` + 109 bench，对比 earliness 与 valid 率。关键发现：在启发式模式（weighted/weighted_greedy）下，次级键把搜索带向 hook 数更差的子空间——`validation_20260112W.json` 17→18、`validation_20260116W.json` 18→20。LNS 的 `_plan_quality` 只在候选 plan 已完成时比较，不会让 plan 变长；构造层的 `depot_late_penalty` 只在同 tier 内生效；终态 `reorder_depot_late` 是纯相邻交换。这三层安全，真正的问题在搜索层的扩展/剪枝 tiebreak。
4. **里程碑 4（已完成）**：默认 `enable_depot_late_scheduling=True`；按 §7 风险栏 mitigation 把搜索次级键收窄到仅 `solver_mode == "exact"` 模式生效，heuristic 模式靠 LNS 与终态 reorder 承担延后诉求。保留 `False` 入参用于 bench 对照。回归覆盖扩展至包括 `validation_20260112W.json`、`validation_20260116W.json`，在 mitigation 后 hook count 回到基线。

## 7. 风险与回退

| 风险 | 触发 | 处理 |
|---|---|---|
| 次级键干扰 beam/weighted 剪枝，产生更差主目标 | 开启后 `external_validation_inputs` 勾数上升 | **已应用（里程碑 4）**：搜索层次级键只在 `solver_mode == "exact"` 激活；其他模式恒传 `0`，完全依靠 LNS 比较和终态 `reorder_depot_late`。若未来仍出现 hook 数反涨，进一步候选是把次级键压缩为"仅在 f、cost 都相等时生效"的更严格词典序，或退回纯后处理版 |
| `reorder_depot_late` 漏算一种依赖，导致 replay 失败 | 验证器报 invalid | 单测补、后处理捕获异常丢弃该次交换，最差等于不重排；`astar_solver` 中候选 plan 会在应用前跑一次完整 `verify_plan`，失败则回退原 plan（防止 `_apply_move` 未检测到的路径冲突） |
| LNS 把词典序误当主目标追求，破坏 repair 进度 | LNS 阶段命中率下降 | `_plan_quality` 里次级权重只在 flag=True 时启用；候选比较只发生在完整 plan 之间，不会让 plan 变长 |
| 与已回滚的 preallocation 混淆 | 历史代码污染 | 新模块名 `depot_late`，与旧 `preallocation` 命名无交集；文件独立 |

## 8. 不做的事（YAGNI）

- 不建时间轴/时间窗模型（solver 不建模实际时钟）。
- 不引入单独的"两阶段求解"；路线 C 的风险已在 preallocation 事件验证过。
- 不在业务层/IO 层动 `NormalizedPlanInput` 数据结构。
- 不在 move_generator 层硬过滤大库钩（会破坏批量联运路径，与已回滚的 preallocation 同病）。

## 9. 并行实现计划

本设计拆成 3 个波次，Wave 2 是独立 3 任务的并行主战场。所有任务落在同一分支（建议 `feat/depot-late-scheduling`）；Wave 2 的 agent 各自负责一个文件，不共享写路径。

### Wave 1 — 基础（并行 2 任务）

**W1-A · 核心模块 `solver/depot_late.py` + 单测**

- 交付：
  - `src/fzed_shunting/solver/depot_late.py` 全部纯函数（`is_depot_hook`、`depot_earliness`、`depot_index_sum`、`reorder_depot_late`、`_is_early_depot_phase`）。
  - `tests/solver/test_depot_late.py` 覆盖 §5.1。
  - 从 `constructive.py` 移出或共享 `DEPOT_INNER_TRACKS` 常量（`constructive.py` 改成 `from .depot_late import DEPOT_INNER_TRACKS`）。
- 独立性：不依赖本设计其他文件的改动。
- 完成门槛：`pytest tests/solver/test_depot_late.py` 全绿；`pytest tests/solver/` 无回归。

**W1-B · SolverResult 字段 `solver/result.py`**

- 交付：
  - `SolverResult` 新增 `depot_earliness: int | None = None`、`depot_hook_count: int | None = None`。
  - `SolverTelemetry`（如需要）同步加字段。
  - `tests/solver/test_result.py`（若存在）或新增小测，校验默认 None、序列化兼容。
- 独立性：纯数据结构扩展。
- 完成门槛：`pytest tests/solver/` 无回归。

### Wave 2 — 集成（并行 3 任务，均依赖 Wave 1-A 合入）

**W2-C · 搜索层 `solver/search.py`**

- 交付：
  - `_priority()` 新增 `neg_depot_index_sum: int = 0` 次级键。
  - 搜索主循环在扩展节点时维护 `depot_index_sum`（增量更新，O(1)/步）。
  - `_solve_search_result` 签名新增 `enable_depot_late_scheduling: bool = False`；关闭时所有次级键恒 0。
  - `QueueItem.priority` 类型标注同步加宽。
  - 新测：关闭开关下基准场景节点展开数与旧版完全一致（防意外影响）；开启开关下两个同 f 节点的扩展顺序符合预期。
- 可测边界：`tests/solver/test_search.py`（若无，新建）做最小单测。
- 依赖：W1-A（导入 `is_depot_hook`）。

**W2-D · LNS 比较 `solver/lns.py`**

- 交付：
  - `_plan_quality(..., depot_late: bool = False)`；开关关闭时元组同旧版。
  - `_is_better_plan` 和 `_improve_incumbent_result` 透传 `depot_late`。
  - 新测：构造两个同 `len` 不同 earliness 的假 plan，开关关闭时视为等价（按 branch/length 决定），开关开启时 earliness 小的胜出。
- 依赖：W1-A。

**W2-E · 构造层 `solver/constructive.py`**

- 交付：
  - `_score_move` 新增 `depot_late_penalty` 分量（插在 tier 之后）。
  - `_is_early_depot_phase` 复用 W1-A 的版本（在 `depot_late.py`，此处 import）。
  - `_choose_best_move` 和 `solve_constructive` 新增 `enable_depot_late_scheduling: bool = False`、`plan_input: NormalizedPlanInput | None = None` 参数（注意当前 `solve_constructive` 已有 plan_input）。
  - 新测：构造同 tier 两个候选（一个大库一个非大库），开关开启且早期相时非大库被选中；开关关闭或晚期相时按旧版 delta/block_size 决定。
- 依赖：W1-A。

### Wave 3 — 装配 & bench（单任务）

**W3-F · 对外 API + 后处理 + 集成测试**

- 交付：
  - `solve_with_simple_astar_result` 和 `solve_with_simple_astar` 新增 `enable_depot_late_scheduling: bool = False`，逐层透传到 W2-C/D/E。
  - `_attach_verification` 前注入 `reorder_depot_late` 调用（仅在开关开启时）。
  - `SolverResult.depot_earliness` 和 `depot_hook_count` 在所有返回点填值（开关关闭时也填，用于观察对比）。
  - 集成测试：`typical_suite` + `typical_workflow_suite` 关闭开关与主线逐字节比对；开启开关下 valid 率不降、总勾数不增。
  - 运行 `external_validation_inputs` bench（可本地 8 worker），把对比数据写入设计文档末尾。
- 依赖：Wave 1 + Wave 2 全部合入。

### 并行化策略

- **同一 git 分支 + 明确文件边界**：Wave 2 的 3 个 agent 各改不同文件，互不踩踏。
- **按文件夹隔离**：Wave 1-A 改 `solver/depot_late.py`（新文件）、`constructive.py`（仅常量 import）、`tests/solver/test_depot_late.py`；W1-B 改 `solver/result.py`。两者互不干扰，可完全并行。
- **共享约定**：所有参数统一命名 `enable_depot_late_scheduling`，避免后期重构。
- **合并顺序**：Wave 1 → 压栏等齐 → Wave 2 → 压栏等齐 → Wave 3。每 wave 完成后在本地跑一次 `pytest tests/solver/` 基线验证。

### 工具建议

- 采用 `superpowers:dispatching-parallel-agents` 技能驱动 Wave 2 的 3 agent 并行执行。
- 每个 agent 提交信息带 `[W?-?]` 前缀方便 rebase。
- Wave 3 用主会话跑，便于人工观察 bench 输出。
