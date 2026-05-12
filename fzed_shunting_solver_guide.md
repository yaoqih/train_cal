# fzed_shunting 求解器技术文档

> **适用人群**：刚接手本项目的工程师  
> **项目路径**：`src/fzed_shunting/`

---

## 1. 项目概述

`fzed_shunting` 是福州东动车所（FZED）的**调车计划自动求解器**，目标是为一个班次内所有动车组排出一套合法的调车钩次序列（Hook Plan），使每辆车从当前停靠位置移动到其目标位置，同时满足路径、容量、作业工位、称重等硬性约束。

**核心业务场景**：
- 动车所内有多条股道（存1～存5、修1～修4库内/外、临时备车线等），
- 调机（机车）一次能从某条股道的"北端"（可接入端）牵引若干辆车走一段路径，再停放到另一条股道的北端；
- 每辆车有自己的目标股道、目标停放位（Spot）或作业工位；
- 求解器输出的"计划"就是一系列 ATTACH（摘钩接车）→ DETACH（推放）的动作序列。

---

## 2. 整体架构与目录结构

```
src/fzed_shunting/
├── domain/          # 领域模型（股道、路径、约束规则）
├── io/              # 输入规范化
├── sim/             # 场景生成
├── solver/          # 求解算法核心（主体）
├── verify/          # 方案验证
├── workflow/        # 多阶段工作流
├── benchmark/       # 性能基准测试
├── demo/            # 可视化
├── tools/           # 辅助工具
└── cli.py           # 命令行入口
```

---

## 3. 领域模型（`domain/`）

### 3.1 主数据 `master_data.py`

`MasterData` 从 JSON 文件加载全场静态数据：
- **股道信息**：名称、容量（米）、端点节点；
- **物理路径**：股道之间的连接关系，用于图搜索；
- **区域定义**：大库、大库外、存车区等逻辑分组；
- **业务规则**：单端股道（single-end）、列车长度约束（如 L1 段限 190 m）。

### 3.2 路径预言机 `route_oracle.py`

`RouteOracle` 是路径查询与校验的核心组件：
- 基于主数据图，计算调机从当前股道到目标股道的**中间经过股道**（`path_tracks`）；
- 校验路径上是否被其他车辆堵塞（`validate_loco_access`）；
- 处理反向出清约束（reverse clearance）——某些路径需要目标股道的南端清空才能走通；
- 缓存常用路径查询结果。

常量 `TRACK_ENDPOINTS` 定义了每条股道的"下令端"（北端），新车辆永远放置在北端。

### 3.3 depot_spots.py — 库内停放位

修 1～4 库内的股道（"大库:RANDOM"区域）有独立的**停放位编号**（Spot）：
- `build_initial_spot_assignments`：根据初始车辆排列计算初始位置分配；
- `realign_spots_for_track_order`：每次 DETACH 后，按新的股道序列重新对齐位置分配；
- `exact_spot_reservations`：取出所有要求精确停放位的车辆的占位预留集合；
- `spot_candidates_for_vehicle`：枚举某辆车可选的合法停放位。

### 3.4 hook_constraints.py — 钩次约束

单次钩次内的车辆组合约束（业务规则）：
- 空车最多 20 辆；
- 重车最多 2 辆；
- 称重车最多 1 辆。

### 3.5 work_positions.py — 作业工位

调棚、洗南、油罐、抛（`WORK_POSITION_TRACKS`）等工位的停放等级（rank）规则：
- `SPOTTING`：数出来占位；
- `EXACT_NORTH_RANK`：必须在北端第 N 个位置；
- `EXACT_WORK_SLOT`：精确工位槽。

### 3.6 carry_order.py — 牵引顺序

调机在牵引状态下，随车列表为有序的，DETACH 只能从**尾端**（tail block）开始放置：
- `is_carried_tail_block(carry, vehicle_nos)`：判断 `vehicle_nos` 是否就是 carry 的尾段；
- `remove_carried_tail_block`：从 carry 末尾移除已放置的车组。

---

## 4. 输入/输出数据结构

### 4.1 输入规范化（`io/normalize_input.py`）

原始 JSON 输入 → `NormalizedPlanInput`（Pydantic 模型）：

```python
class NormalizedPlanInput:
    track_info: list[NormalizedTrackInfo]   # 股道名称与容量
    vehicles: list[NormalizedVehicle]        # 车辆列表
    loco_track_name: str                     # 调机初始位置
    yard_mode: str                           # 场区模式
    single_end_track_names: frozenset[str]   # 单端股道集合
```

```python
class NormalizedVehicle:
    vehicle_no: str          # 车号
    current_track: str       # 当前股道
    order: int               # 北端排列序号（越小越靠近北端）
    vehicle_model: str       # 车型
    vehicle_length: float    # 车长（米）
    goal: GoalSpec           # 目标规格
    need_weigh: bool         # 是否需要称重（必须过机库）
    is_heavy: bool           # 是否重车
    is_close_door: bool      # 是否关门车（存4北末端放置规则）
```

### 4.2 目标规格 `GoalSpec`

目标有 5 种模式（`target_mode`）：

| 模式 | 含义 |
|------|------|
| `TRACK` | 停到指定股道即可 |
| `AREA` | 停到指定区域（`allowed_target_tracks` 是区域内的所有股道） |
| `SPOT` | 必须停到精确停放位（`target_spot_code`） |
| `WORK_POSITION` | 必须在指定作业工位（`work_position_kind + target_rank`） |
| `SNAPSHOT` | 快照模式：按上班快照分配位置，允许的股道集合动态计算 |

目标允许有**首选**（`preferred_target_tracks`）和**兜底**（`fallback_target_tracks`）两级，只有首选全满时才允许使用兜底。

### 4.3 执行状态 `ReplayState`（`verify/replay.py`）

搜索过程中的可变状态：

```python
class ReplayState:
    track_sequences: dict[str, list[str]]   # 每条股道上的车辆序列（北端在前）
    loco_track_name: str                     # 调机当前所在股道
    loco_node: str | None                    # 调机当前接触节点
    weighed_vehicle_nos: set[str]            # 已称重车辆集合
    spot_assignments: dict[str, str]         # 车号 → 停放位编号
    loco_carry: tuple[str, ...]              # 调机当前携带的车辆（有序）
```

**不变量**：
- 车辆永远从北端（序列前端）被 ATTACH 取走；
- 车辆永远放置在目标股道的北端（前插）；
- `loco_carry` 中的车辆只能从尾端（tail）依次 DETACH。

### 4.4 动作类型 `HookAction`（`solver/types.py`）

```python
class HookAction:
    source_track: str        # 取车股道（ATTACH）或携带时调机所在股道（DETACH）
    target_track: str        # 放车目标股道（DETACH）或取车后调机停在哪里（ATTACH）
    vehicle_nos: list[str]   # 涉及车辆
    path_tracks: list[str]   # 完整行驶路径（含起止）
    action_type: str         # "ATTACH" 或 "DETACH"
```

---

## 5. 求解算法架构

`solve_with_simple_astar_result`（`solver/astar_solver.py`）是主入口，内部按固定顺序执行多个阶段，**构造法是第一个运行的**（作为安全网），A* 系列是后续的优化层：

```
solve_with_simple_astar_result()
│
├── Stage 0   构造法基线（占总预算 25%，上限 12 s）
│               ↳ 贪心 + W3-N 回溯，保证一定返回方案
│
├── Stage 0.5 暖启动补完（仅当构造法终止时 h=1）
│               ↳ 从构造法终止点做极短 A* 补上最后一步
│
├── Stage 0.6 部分续跑（从构造法的中间空载检查点继续）
│               ↳ 路径阻塞 → 尝试 route_tail / route_release 构造
│               ↳ 无阻塞  → 近目标局部续跑
│
├── Stage 1   精确 A*（exact，solver_mode="exact" 时）
│
├── Stage 2   Anytime Fallback 链（7个子阶段，见第 19 节）
│               → 仅在 Stage 1 未找到完整解时继续
│
├── Stage 3   LNS 改进（Large Neighborhood Search）
│               → 对已有完整解做"破坏-修复"迭代
│
├── Stage 4   方案压缩（plan_compressor）
│               → 三类局部重写，逐步缩短钩次数
│
└── Stage 5   验证（plan_verifier，verify=True 时）
                → 验证失败抛 PlanVerificationError
```

### 5.1 入口函数

```python
# solver/astar_solver.py
solve_with_simple_astar(plan_input, initial_state, ...)
    -> list[HookAction]

solve_with_simple_astar_result(plan_input, initial_state, ...)
    -> SolverResult
```

关键参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `solver_mode` | str | `"exact"` / `"weighted"` / `"beam"` / `"real_hook"` |
| `heuristic_weight` | float | 启发式权重（> 1 牺牲最优性换速度） |
| `beam_width` | int \| None | Beam 宽度（None 表示不剪枝） |
| `time_budget_ms` | float \| None | 总时间预算（毫秒） |
| `node_budget` | int \| None | 节点展开数上限 |
| `enable_anytime_fallback` | bool | 是否开启 fallback 链 |
| `enable_constructive_seed` | bool | 是否先跑构造法基线 |
| `enable_depot_late_scheduling` | bool | 是否优化库内进车时序（depot 延迟调度） |
| `verify` | bool | 完成后是否运行验证器（需传入 master） |

---

## 6. 核心搜索循环（`solver/search.py`）

### 6.1 队列节点 `QueueItem`

```python
@dataclass(order=True)
class QueueItem:
    priority: tuple              # 决定出队顺序的排序键
    seq: int                     # FIFO 序号（相同优先级时的决定性打破器）
    state_key: tuple             # 状态的正规化哈希键
    state: ReplayState           # 当前状态
    plan: list[HookAction]       # 到达本状态的动作序列
    cost: int                    # plan 长度（= g 值）
    structural_key: tuple        # 结构多样性键（beam 模式用）
    route_release_focus_tracks   # 路径释放焦点股道集合
    ...
```

### 6.2 主循环逻辑

```
while queue:
    if budget.exhausted(): break

    current = heappop(queue)            # 取出优先级最低的节点
    if cost 已经不是 best: continue     # 懒删除（closed set 优化）
    expanded_nodes += 1
    closed_state_keys.add(current.state_key)

    if is_goal(current.state):
        if exact/real_hook mode: return current.plan  # 找到最优解
        best_goal_plan = current.plan
        continue  # beam/weighted 模式：继续寻找更短的

    candidates = generate_move_candidates(...)   # 生成候选动作
    for candidate in candidates:
        applied = replay_candidate_steps(...)     # 重放验证合法性
        for branch in candidate_branches:        # 可能有多个"前缀分支"
            next_state = branch.final_state
            cost = len(next_plan)
            if state_key in best_cost and cost >= best_cost[state_key]: skip
            scoring = evaluate_candidate_steps(...)   # 评分
            priority = _priority(cost, heuristic, bonuses, penalties, ...)
            heappush(queue, QueueItem(...))

    if beam mode: _prune_queue(queue, beam_width)  # 剪枝
```

### 6.3 状态键（`state.py`）

状态键用于判断"是否已访问过相同状态"：

```python
state_key = (
    tuple(sorted非空股道序列),    # 各股道上的车辆排列
    loco_track_name,              # 调机位置
    loco_node,                    # 调机接触节点
    tuple(sorted已称重车辆),      # 称重状态
    tuple(spot_assignments),      # 停放位分配（排除大库随机车）
    loco_carry,                   # 携带车辆
)
```

---

## 7. 启发式函数（`solver/heuristic.py`）

A* 使用的可接受启发值（h）取以下四项中的**最大值**（确保可接受性）：

### 7.1 `h_distinct_transfer_pairs`（强制转移对数）

对每辆未到目标的车辆，计算其所在股道→目标股道的"转移对"数量：
- 目标唯一的车辆 → 每个不同的 `(来源, 目标)` 对算一次；
- 目标多元的车辆 → 每个来源股道算最多一次额外钩次。

这是从排列的角度给出的最紧下界。

### 7.2 `h_blocking`（阻塞惩罚）

对每条"目标股道"北端有阻塞车辆的情况 +1，
再用 **Tarjan 算法**检测股道之间的互阻塞强连通分量，每个 SCC 额外 +1（必须有一次临时中转才能打破循环）。

### 7.3 `h_weigh`（称重数量）

未称重的 `need_weigh` 车辆数量（每辆至少需要一次进机库 DETACH）。

### 7.4 `h_spot_evict`（精确位驱逐）

目标停放位被其他车辆占用的数量（被占用的车辆需要先被移走）。

### 7.5 `h_tight_capacity`（容量溢出驱逐，与 h_pairs 叠加）

当目标股道容量过载时，估算必须先驱逐再召回的"往返钩次"数量（加到 `h_pairs` 上，因两者不重叠）。

### 7.6 real-hook 模式的缩放

在 real-hook（ATTACH+DETACH 分开计算）模式中，所有基础下界 ×2（因原先每次"PUT"相当于 1 次 ATTACH + 1 次 DETACH），再加上 `h_carry_detach`（已携带车辆的最少 DETACH 次数）。

---

## 8. 候选动作生成（`solver/move_candidates.py` / `move_generator.py`）

### 8.1 原始动作（primitive）

`generate_real_hook_moves` 枚举当前状态下所有合法的 ATTACH / DETACH 动作：
- **ATTACH**：从任意非空股道的北端取 1～N 辆车（受钩次约束限制）；
- **DETACH**：当调机携带车辆时，把尾部的 1～M 辆车放到某条合法目标股道。

每个候选动作包含 `path_tracks`（路径），必须通过 `route_oracle` 验证无阻塞。

### 8.2 结构候选（structural）

在检测到特定结构性压力时，`generate_move_candidates` 会基于 `structural_intent` 生成**多步复合候选**：

| 候选类型（`reason`）| 触发条件 | 典型动作序列 |
|---|---|---|
| `route_release_*` | 存在路径阻塞 | 清除阻塞股道的车辆 |
| `goal_frontier_*` | 目标股道有前置工作 | 先移走阻塞者再放入目标车 |
| `work_position_*` | 作业工位未完成 | 按工位规则调整车序 |
| `chain_macro_*` | 债务链跨多条股道 | 一次性完成多步清理 |
| `resource_release` | 某条股道容量超载 | 先疏散再重新分配 |

### 8.3 MoveCandidate 数据结构

```python
@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]  # 单步或多步动作序列
    kind: str = "primitive"        # "primitive" 或 "structural"
    reason: str = ""               # 触发原因，见下表
    focus_tracks: tuple[str, ...] = ()  # 路径释放焦点股道
    structural_reserve: bool = False    # True → 搜索时同时入队改进前缀分支
```

结构候选的 `reason` 与优先级组：

| reason | 优先级组 |
|---|---|
| `route_release_frontier` | 0（最高） |
| `goal_frontier_source_opening` | 1 |
| `resource_release` / `capacity_release` | 1 |
| 其他 `goal_frontier_*`, `work_position_*`, `chain_macro_*` | 2+ |

### 8.4 候选编译与验证（`candidate_compiler.py`）

多步候选在进入搜索队列前，通过 `replay_candidate_steps` 重放验证：
- 每一步动作都必须在中间状态下合法；
- 若某步不合法则整个候选被丢弃；
- 返回完整的 `(前置状态, 动作, 后置状态)` 转移序列。

### 8.5 前缀分支入队（`search.py: _candidate_queue_branches`）

当 `candidate.structural_reserve=True` 且动作步数 > 1 时，搜索循环除入队完整候选外，还会入队**改进前缀**（每个前缀步骤都使进展键改善）：

```
进展键 = (target_sequence_defect_count, work_position_unfinished_count,
          front_blocker_count, capacity_overflow_track_count,
          capacity_debt_total, route_pressure, goal_track_blocker_count,
          unfinished_count, len(loco_carry))
```

- 若改进前缀 ≤ 3 个 → 全部入队；
- 若改进前缀 > 3 个 → 仅选 `{0, mid, last}` 3 个代表入队，防止分支爆炸。

---

## 9. 候选评分与优先级（`solver/search.py`）

### 9.1 优先级元组 `_priority()`

不同搜索模式下优先级元组内容不同，但总体结构是：

```python
(
    route_release_regression_penalty,   # 路径释放回退惩罚（越小越好）
    staging_churn_penalty,              # 临时股道反复来回惩罚
    carry_fragmentation_penalty,        # 拆散携带车组惩罚
    0 if exact_spot_priority else 1,    # 优先处理精确停放位
    0 if structural_progress else 1,    # 优先结构性进展
    -structural_progress_bonus,         # 结构进展奖励（越大优先级越高）
    0 if blocker_bonus else 1,          # 优先解除阻塞
    adjusted_score,                     # g + w*h + penalties - bonuses
    ...tiebreakers...
)
```

### 9.2 奖励（bonus）计算

**`blocker_bonus`**（`_blocking_goal_target_bonus`）：
- 如果 ATTACH 动作从某条阻塞股道取车 → +len(被阻塞的目标)；
- 路径释放奖励（`route_blockage_release_score`）；
- 路径释放延续奖励（`route_release_continuation_bonus`）；
- 精确停放位清除奖励（`exact_spot_clearance_bonus`）；
- 精确停放位暴露奖励（`exact_spot_seeker_exposure_bonus`）；
- 携带中暴露关键车辆奖励（`_carry_exposure_bonus`）。

**`structural_progress_bonus`**（`_structural_metric_progress_bonus`）：
根据前后结构指标变化量加权求和：

| 指标改善 | 权重 |
|---|---|
| 目标序列缺陷数减少 | ×18 |
| 容量超载股道数减少 | ×16 |
| 作业工位未完成数减少 | ×12 |
| 前端阻塞车辆减少 | ×8 |
| 路径压力减少 | ×4 |
| 目标股道阻塞减少 | ×3 |
| 未完成车辆数减少 | ×1 |
| 临时股道债务增加 | -1（惩罚） |

上限为 `step_count × 16`（防止多步候选虚高评分）。

### 9.3 惩罚（penalty）计算

- **`staging_churn_penalty`**：在临时股道（临1～4、存4南）反复来回时惩罚；
- **`carry_fragmentation_penalty`**：拆散正在携带车组（尤其是只放了一半）时惩罚；
- **`carry_growth_penalty`**：在已有携带车辆的情况下继续 ATTACH 更多（超过 10 辆时额外加重）；
- **`route_release_regression_penalty`**：在路径释放焦点有效期内转移到无法访问焦点股道时惩罚。

---

## 10. Beam Search 剪枝（`search.py: _prune_queue`）

每轮展开后，若队列超过 `beam_width`：

1. 对所有节点按优先级排序；
2. **保留最浅深度代表**（BEAM_SHALLOW_RESERVE=1）；
3. **保留阻塞器代表**（blocker_reserve=True 的节点，最多 1 个）；
4. **按结构候选股道分组保留**（每组至少保留 1 个，确保结构多样性）；
5. 若检测到"结构扰动压力"（临时股道债务过高），额外扩展搜索范围，按结构键选低债务候选；
6. 按优先级填满剩余位置。

被剪枝节点同时从 `best_cost` 字典中删除（使未来路径可以重新访问该状态）。

---

## 11. 构造法（`solver/constructive.py`）

### 11.1 算法概述

构造法是最底层的保底求解器，**一定能返回一个（可能不完整的）方案**。

核心算法：贪心前向搜索（greedy forward）+ **W3-N 有界回溯**：
- 每步评分所有候选，取最优者；
- 若启发值在 `stuck_threshold` 轮内无改善，则视为"卡住"；
- 回溯到最近的决策点，尝试次优方案；
- 最多回溯 `max_backtracks`（默认 5）次；
- 返回所有尝试中最好的结果。

### 11.2 动作分层（Tier）

每个候选动作按意图分为 7 个优先层：

| Tier | 含义 |
|---|---|
| 0 | DETACH 到首选目标（且满足偏好条件） |
| 1 | DETACH 到有效目标 |
| 2 | 暴露关键车辆 / 路径释放延续 / 清除精确位阻塞 / ATTACH 改进启发值 |
| 3 | 清除目标股道阻塞（blocker clear） |
| 4 | DETACH 改进启发值 |
| 5 | 放到临时股道（staging detach） |
| 6 | 其他（兜底） |

如果一个动作会"驱逐已满足目标的车辆"，其 Tier +100（极度避免）。

### 11.3 逆动作保护（Inverse Guard）

记录最近 12 步动作。如果一个候选动作与最近某步相反（把刚送走的一半以上车辆送回来），且有其他非逆向候选，则优先选非逆向。

---

## 12. 结构指标（`solver/structural_metrics.py`）

`compute_structural_metrics` 返回不可变 `StructuralMetrics` dataclass，用于指导搜索和评分：

| 字段 | 含义 |
|---|---|
| `unfinished_count` | 未到目标的车辆总数（**含 `loco_carry` 中的车辆**） |
| `target_sequence_defect_count` | 作业工位排列不满足顺序要求的缺陷数 |
| `target_sequence_defect_by_track` | 各股道的排列缺陷数 |
| `work_position_unfinished_count` | 未完成作业工位分配的车辆数 |
| `front_blocker_count` | 已满足目标的车辆堵在未完成车辆前面的数量 |
| `front_blocker_by_track` | 各股道的前端阻塞数 |
| `goal_track_blocker_count` | 未满足的车辆停在别人的目标股道上造成阻塞的数量 |
| `goal_track_blocker_by_track` | 各目标股道的阻塞数 |
| `capacity_overflow_track_count` | 超出容量的股道数 |
| `capacity_debt_by_track` | 各超载股道的超载量（米） |
| `staging_debt_count` | 临时股道上不应在那里的车辆总数 |
| `staging_debt_by_track` | 各临时股道的债务数 |
| `area_random_unfinished_count` | 区域随机目标（RANDOM）尚未满足的数量 |
| `preferred_violation_count` | 首选违规数（详见第 16 节） |
| `loco_carry_count` | 调机当前携带的车辆数 |

### 12.1 plan 形态分析（`summarize_plan_shape`）

```python
summarize_plan_shape(plan) -> dict:
    staging_hook_count            # 涉及任意临时股道的钩次数
    staging_to_staging_hook_count # 来源和目标都是临时股道的钩次数
    rehandled_vehicle_count       # 被移动超过 2 次的车辆数
    max_vehicle_touch_count       # 被移动最多次的单辆车的移动次数
```

临时股道集合为 `frozenset({"临1", "临2", "临3", "临4", "存4南"})`。

---

## 13. 结构意图（`solver/structural_intent.py`）

`build_structural_intent` 分析当前状态的宏观目标，返回 `StructuralIntent`：

- **`order_debts_by_track`**：每条目标股道的排列债务（哪些车需要调序）；
- **`delayed_commitments`**：因为前置工作尚未完成而被推迟的目标承诺；
- **`buffer_leases`**：需要临时借用缓冲股道的计划；
- **`debt_clusters`**：互相关联的债务集群，用于生成"链式宏候选"。

---

## 14. 债务链分析（`solver/debt_chain.py`）

`analyze_debt_chains` 把跨多条股道的连锁关系建模为"债务链"：

- 从某条"负债"股道出发（容量超载 / 排列不对 / 阻塞他人）；
- 通过路径连接，找到上游阻塞者和下游受影响者；
- 生成一条跨多条股道的宏观动作候选（`chain_macro_*`），一次性清理整条链。

---

## 15. 路径阻塞分析（`solver/route_blockage.py`）

`compute_route_blockage_plan` 分析当前哪些股道在"物理上"阻塞了重要路径：

- 对每辆未到目标的车辆，检查从当前股道到目标股道的路径；
- 若路径中有某条股道被其他车辆占用，记录为"阻塞事实"（`RouteBlockageFact`）；
- 返回 `RouteBlockagePlan`：包含所有阻塞事实和总阻塞压力（`total_blockage_pressure`）；
- 支持**路径释放焦点**（focus）机制：连续清理某条阻塞股道时给予奖励，短时间内不清理则惩罚。

---

## 16. 状态纯净度（`solver/purity.py`）

`compute_state_purity` 返回 `(unfinished_count, preferred_violation_count, staging_pollution_count)` 三元组作为次优先级 tiebreaker，帮助 A* 选择更"干净"的路径。

### 16.1 `unfinished_count`

未到目标的车辆数，**包含 `loco_carry` 中的车辆**。

### 16.2 `preferred_violation_count`

不是简单的"停在兜底股道"计数——一辆车被计入违规，当且仅当它满足目标（`goal_is_satisfied=True`）但满足的不是首选级别，具体为：

```python
def _counts_as_preferred_violation(vehicle, *, current_track, state, plan_input):
    return goal_is_satisfied(...) and (
        _is_fallback_while_preferred_still_feasible(...)  # 首选股道仍有空间且可达
        or not goal_is_preferred_satisfied(...)            # 目标满足但非首选方式
    )
```

### 16.3 `staging_pollution_count`

临时股道（`{"临1", "临2", "临3", "临4", "存4南"}`）上的车辆，其目标允许股道不包含该临时股道时，即被视为"污染"。

```python
staging_pollution_count = sum(
    1 for vno in state.track_sequences.get(track, [])
    if track in STAGING_TRACKS
    and track not in vehicle.goal.allowed_target_tracks
    for vehicle in [plan_input.vehicles_by_no[vno]]
)
```

---

## 17. 大邻域搜索 LNS（`solver/lns.py`）

在找到初始可行解后，LNS 通过"破坏-修复"循环改进解。

### 17.1 破坏算子（4 种，轮转使用）

`_candidate_repair_cut_points` 实现 4 种破坏策略，按 `per_strategy = repair_passes // 4` 轮次依次轮转：

| 算子 | 选择依据 |
|---|---|
| `hotspot` | 被触碰次数最多的股道（优先临时股道）；选该股道首次出现的钩次索引 |
| `worst-cost` | 代价最高的钩次（`len(path_tracks) × 100 + len(vehicle_nos)`） |
| `target-cluster` | 相邻两个钩次目标相同（连续同目标段的第一个索引） |
| `equidistant` | 等间距采样（`step = plan_length // (limit + 1)`） |

### 17.2 修复输入构造（`_build_repair_plan_input`）

从切割点出发重搜时，需要"冻结"已满足的车辆，只对需要移动的车辆求解：

- **已满足且不在可动集合中**的车辆 → 目标改为 `SNAPSHOT`（保持当前股道）；
- **可动集合**包括：`exact_spot_blockers`（精确位阻塞者）、`front_blockers`（前端阻塞者）、`capacity_release_front`（容量释放前端车辆）、`work_position_rank`（工位排序相关车辆）、`route_blockage` 中的阻塞者。

### 17.3 解优劣比较（`_is_better_plan`）

```python
def _plan_quality(plan, route_oracle, *, depot_late=False, purity_metrics=(0,0,0)):
    # 返回比较元组（越小越好）：
    # (len(plan), *purity_metrics, total_branch_count, total_length_m)
    # depot_late=True 时末尾追加 depot_earliness(plan)
```

`_is_better_plan(new, old)` 直接比较两个 quality 元组，新 < 旧则返回 True（接受改进）。

---

## 18. 方案验证（`verify/plan_verifier.py` / `replay.py`）

### 18.1 重放（replay.py）

`replay_plan` 从初始状态按计划逐步执行：
- 验证每步 ATTACH 时车辆确实在源股道北端；
- 验证每步 DETACH 时放置的是当前携带的尾段；
- 追踪称重状态（通过机库则标记已称重）；
- 维护停放位分配（每次 DETACH 后重新对齐）。

### 18.2 验证（plan_verifier.py）

`verify_plan` 在重放完成后检验：
- 所有车辆是否到达目标（包括精确位、工位、称重）；
- 每次钩次的车辆组合是否满足约束（重车、称重车数量）；
- 路径是否合法（无阻塞、长度符合限制）；
- 容量是否超出；
- 停放位分配是否有效。

返回 `PlanVerificationResult`，包含通过/失败状态和详细违规列表。

---

## 19. Anytime Fallback 链（`solver/anytime.py`）

`_run_anytime_fallback_chain` 管理时间预算内的多阶段搜索策略。实际共有 **7 个子阶段**，**只要当前 `best_complete_result` 仍为 None 时才继续**下一阶段（一旦找到完整解即退出链）：

| # | 阶段名 | solver_mode | heuristic_weight | beam_width | 预算占比 |
|---|---|---|---|---|---|
| 1 | `weighted` | weighted | max(w, 1.5) | None | 8% |
| 2 | `beam` | beam | max(w, 1.5) | bw or 64 | 8% |
| 3 | `beam_greedy_64` | beam | 5.0 | bw or 64 | 8% |
| 4 | `weighted_greedy` | weighted | 5.0 | None | **40%** |
| 5 | `beam_greedy_128` | beam | 5.0 | max(bw or 0, 128) | 8% |
| 6 | `beam_greedy_256` | beam | 5.0 | max(bw or 0, 256) | 8% |
| 7 | `weighted_very_greedy` | weighted | 10.0 | None | 8% |

其中 `w` 是调用方传入的 `heuristic_weight`，`bw` 是 `beam_width`。

每个阶段结束时：
- 若结果 `is_complete=True` → 更新 `best_complete_result`（仅当新解更优时）并**退出链**；
- 若结果 `is_complete=False` → 用 `_partial_candidate_score()` 比较局部解质量，更新 `best_partial_result`；
- 若所有阶段均未找到完整解 → 返回最优局部解。

最终解优劣通过 `_is_better_plan()` 比较，比较元组为 `(len, *purity_metrics, total_branch_count, total_length_m)`（启用 depot_late 时末尾再加 `depot_earliness`）。

---

## 20. 求解结果（`solver/result.py`）

`SolverResult` 是不可变 frozen dataclass，包含完整的求解输出：

```python
@dataclass(frozen=True)
class SolverResult:
    plan: list[HookAction]          # 钩次序列（最终计划）
    expanded_nodes: int             # 展开节点数
    generated_nodes: int            # 生成节点数
    closed_nodes: int               # 关闭节点数（已访问状态数）
    elapsed_ms: float               # 耗时（毫秒）
    is_complete: bool = False       # 是否所有车辆都到目标
    is_proven_optimal: bool = False # 是否证明最优（仅 exact 模式）
    fallback_stage: str | None = None
        # 取值："exact" / "weighted" / "beam" / "weighted_greedy" /
        #       "beam_greedy_128" / "beam_greedy_256" / "weighted_very_greedy" /
        #       "constructive" / "constructive_warm_start"
    partial_plan: list[HookAction] = field(default_factory=list)
        # 若不完整，最优局部计划
    partial_fallback_stage: str | None = None
    verification_report: Any | None = None        # 完整解的验证报告
    partial_verification_report: Any | None = None# 局部解的验证报告
    debug_stats: dict[str, Any] | None = None     # 调试统计信息
    telemetry: SolverTelemetry | None = None      # 生产可观测性遥测数据
    depot_earliness: int | None = None            # 进库最早钩次编号（LNS depot优化用）
    depot_hook_count: int | None = None           # 进库钩次总数
```

### 20.1 SolverTelemetry

`SolverTelemetry` 在生产环境中将结构化指标写入 JSON-lines 文件（环境变量 `FZED_SOLVER_TELEMETRY_PATH` 指定路径，`emit_telemetry()` 追加写入）：

```python
@dataclass(frozen=True)
class SolverTelemetry:
    # 输入规模
    input_vehicle_count: int
    input_track_count: int
    input_weigh_count: int
    input_close_door_count: int
    input_spot_count: int
    input_area_count: int
    input_work_position_count: int

    # 各阶段耗时（毫秒）
    constructive_ms: float
    exact_ms: float
    anytime_ms: float
    lns_ms: float
    verify_ms: float
    total_ms: float

    # 求解结果
    is_complete: bool
    plan_hook_count: int
    fallback_stage: str | None
    is_valid: bool
    partial_hook_count: int

    # 资源预算
    time_budget_ms: float | None
    node_budget: int | None
```

---

## 21. 目标逻辑与兜底触发（`solver/goal_logic.py`）

### 21.1 `goal_is_satisfied`

```python
goal_is_satisfied(vehicle, *, track_name, state, plan_input, ...) -> bool
```

按以下顺序检查：
1. **首选/兜底级别**：若当前在兜底股道，先检查是否允许（`goal_can_use_fallback_now`）；
2. **称重**：`need_weigh=True` 时要求已在 `weighed_vehicle_nos` 中；
3. **目标模式**：
   - `TRACK`/`AREA`/`SNAPSHOT`：`track_name in allowed_target_tracks`；
   - `SPOT`：`spot_assignments[vno] == target_spot_code`；
   - `大库:RANDOM`：`spot_assignments[vno] in spot_candidates_for_vehicle(...)`；
   - `WORK_POSITION`：工位排列约束满足；
   - `is_close_door` 在存4北：序列索引 ≥ 3（不得排在前 3 位）。

### 21.2 `goal_can_use_fallback_now`

```python
@lru_cache(maxsize=200_000)
def _cached_goal_can_use_fallback_now(...) -> bool
```

只有当以下**所有**首选股道都不可用时，才允许使用兜底股道：

- 无首选股道定义 → 始终返回 True；
- 首选股道属于 `大库:RANDOM` 组 → 该组所有停放位均已被占满；
- 对每条首选股道：无剩余长度余量 **或** 路径不可达（`route_oracle` 验证）。

该函数结果被 `lru_cache` 缓存（键包含状态快照），避免在同一状态内重复计算。

---

## 22. 方案压缩（`solver/plan_compressor.py`）

`compress_plan()` 在 LNS 之后对完整解做局部重写，参数：`max_window_size=10`，`max_passes=16`，所有改写均通过验证器守护。

每轮 pass 按顺序尝试三种策略：

### 22.1 Strategy 1: `_try_rebuild_single_source_window`

滑动窗口重建：取一段连续的"单一来源"钩次，用局部 A* 重搜得到更短替代段。

### 22.2 Strategy 2: `_try_merge_adjacent_same_source_same_target_pairs`

合并相邻同来源同目标的 ATTACH-DETACH 对：

```
ATTACH(src→src, vA)  DETACH(src→tgt, vA)
ATTACH(src→src, vB)  DETACH(src→tgt, vB)
                ↓
ATTACH(src→src, vA+vB)  DETACH(src→tgt, vA+vB)
```

### 22.3 Strategy 3: `_try_merge_adjacent_same_target_pairs`

合并相邻目标相同的钩次对（来源可不同）。

### 22.4 滑动窗口删除

对 `window_size` 从 2 到 `max_window_size` 的每个窗口大小，尝试直接删除该窗口内的全部钩次，若删后仍可验证通过则接受。

所有策略：在 `_verify_candidate()` 通过前不接受改写。

---

## 23. 验证恢复策略（`solver/profile.py` / `solver/validation_recovery.py`）

当主搜索以 beam=8 输出结果但验证失败时，触发自动恢复：

```
初始：beam_width=8，timeout=60s，total_timeout=180s
重试宽度序列：[8, 16, 24, 32]（= beam_width × [1,2,3,4]）
```

- 每次失败 → 将 beam_width 乘以下一个倍数再次求解；
- 验证通过后，若解质量"病态"（见下），仍继续扩大 beam 搜索：

```python
# validation_recovery_should_continue_after_success:
max_vehicle_touch_count > 80  OR  staging_to_staging_hook_count >= 8  OR  rehandled_vehicle_count > 80

# validation_recovery_should_escalate_after_success:
max_vehicle_touch_count > 80  OR  staging_to_staging_hook_count >= 32  OR  rehandled_vehicle_count > 80
```

相关常量（`profile.py`）：

```python
VALIDATION_DEFAULT_SOLVER = "beam"
VALIDATION_DEFAULT_BEAM_WIDTH = 8
VALIDATION_DEFAULT_TIMEOUT_SECONDS = 60.0
VALIDATION_TOTAL_TIMEOUT_SECONDS = 180.0
VALIDATION_PRIMARY_RESERVED_BUDGET_SECONDS = 75.0
VALIDATION_RETRY_BEAM_WIDTH_MULTIPLIER = 4   # 重试上限 = beam_width × 4
```

---

## 24. 工作流（`workflow/runner.py`）

`WorkflowRunner` 支持**多阶段串行求解**：
- 每个阶段是一个子场景（如先完成一批进库的车辆，再处理出库的）；
- 上一阶段的 `final_state` 作为下一阶段的 `initial_state`；
- 支持状态前向传递（已称重车辆、停放位分配跨阶段保留）。

---

## 25. 命令行接口（`cli.py`）

```bash
# 生成场景
python -m fzed_shunting generate-micro
python -m fzed_shunting generate-scenario --profile medium
python -m fzed_shunting generate-typical-suite

# 求解
python -m fzed_shunting solve --scenario scenario.json
python -m fzed_shunting solve-suite --suite-dir ./suite/

# 多阶段求解
python -m fzed_shunting solve-workflow --workflow workflow.json

# 验证方案
python -m fzed_shunting verify --plan plan.json --scenario scenario.json

# 基准测试
python -m fzed_shunting benchmark

# 比较不同求解配置
python -m fzed_shunting compare-suite --suite-dir ./suite/
```

---

## 26. 关键文件速查

| 你想了解... | 看这个文件 |
|---|---|
| 入口函数 / 搜索配置 | `solver/astar_solver.py` |
| 主搜索循环 / 优先级计算 | `solver/search.py` |
| 启发式函数 | `solver/heuristic.py` |
| 候选动作生成 | `solver/move_generator.py`, `move_candidates.py` |
| 候选编译与验证 | `solver/candidate_compiler.py` |
| 构造法（兜底） | `solver/constructive.py` |
| 结构分析 | `solver/structural_metrics.py`, `structural_intent.py` |
| 路径阻塞分析 | `solver/route_blockage.py`, `domain/route_oracle.py` |
| 状态表示 | `verify/replay.py` (ReplayState), `solver/state.py` |
| 输入数据格式 | `io/normalize_input.py` |
| 验证逻辑 | `verify/plan_verifier.py` |
| 主数据（股道/路径） | `domain/master_data.py`, `domain/route_oracle.py` |
| 停放位管理 | `domain/depot_spots.py` |
| Anytime 策略 | `solver/anytime.py` |
| LNS 改进 | `solver/lns.py` |
| 目标逻辑 / 兜底触发 | `solver/goal_logic.py` |
| 方案压缩 | `solver/plan_compressor.py` |
| 状态纯净度 | `solver/purity.py` |
| 验证恢复策略 | `solver/profile.py`, `solver/validation_recovery.py` |
| 时间/节点预算 | `solver/budget.py` |
| 遥测数据 | `solver/result.py` (SolverTelemetry), `FZED_SOLVER_TELEMETRY_PATH` env |

---

## 27. 核心约束速查

| 约束类型 | 相关代码 |
|---|---|
| 单次钩次最多 20 辆空车 | `domain/hook_constraints.py` |
| 单次钩次最多 2 辆重车 | `domain/hook_constraints.py` |
| 单次钩次最多 1 辆称重车 | `domain/hook_constraints.py` |
| 路径无阻塞 | `domain/route_oracle.py` |
| L1 段列车长度 ≤ 190 m | `domain/route_oracle.py` |
| 反向出清约束 | `domain/route_oracle.py` |
| 股道容量约束 | `domain/master_data.py` + `solver/structural_metrics.py` |
| 精确停放位（机库内） | `domain/depot_spots.py` + `solver/exact_spot.py` |
| 作业工位排列约束 | `domain/work_positions.py` + `solver/goal_logic.py` |
| 称重必须过机库 | `solver/heuristic.py` + `solver/goal_logic.py` |
| 存4北关门车放置规则 | `solver/constructive.py`（`_close_door_*` 系列函数） |
| 首选/兜底股道优先级 | `io/normalize_input.py` + `solver/goal_logic.py` |

---

## 28. 典型调试路径

**场景：求解器找不到完整解**
1. 检查 `SolverResult.fallback_stage`，确认最终落到哪个阶段；
2. 看 `debug_stats`，重点关注 `expanded_nodes`、`states_with_zero_moves`；
3. 若 `states_with_zero_moves` 高，说明有状态无合法动作 → 可能是容量/路径死锁；
4. 打开构造法 debug，看 `stuck_reason`；
5. 用 `verify` 子命令验证中间方案，定位违规的约束。

**场景：生成的方案钩次数过多**
1. 看 `plan_shape`（`summarize_plan_shape`）：`staging_to_staging_hook_count` 高说明临时股道来回多；
2. 检查 `structural_metrics.target_sequence_defect_count`：有缺陷说明排列未能满足；
3. 考虑增加 LNS 轮次或调整 beam_width；
4. 检查 `debt_chain` 分析是否覆盖了问题场景。

---

*文档生成日期：2026-05-12*
