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

---

## 8. 候选动作生成（`solver/move_candidates.py` / `move_generator.py`）

### 8.1 原始动作（primitive）

`generate_real_hook_moves` 枚举当前状态下所有合法的 ATTACH / DETACH 动作：
- **ATTACH**：从任意非空股道的北端取 1～N 辆车（受钩次约束限制）；
- **DETACH**：当调机携带车辆时，把尾部的 1～M 辆车放到某条合法目标股道。

每个候选动作包含 `path_tracks`（路径），必须通过 `route_oracle` 验证无阻塞。

### 8.2 结构候选（structural）

在检测到特定结构性压力时，`generate_move_candidates` 会基于 `structural_intent` 生成**多步复合候选**：

| 候选类型（`reason`）| 触发时机 | 做什么 |
|---|---|---|
| `route_release_frontier` | 某条股道上的车（`blocking_track`）堵住了别人的行驶路径 | 清除阻塞者 → 顺势把被堵的车推进一程 → 视情况恢复阻塞者 |
| `goal_frontier_source_opening` | 一条股道上已满足目标的车（`prefix`）压着还没满足的车 | 把前端已满足的 prefix 暂移 → 把被压的车送到目标 → prefix 放回 |
| `work_position_source_opening` | 作业工位目标股道的北端入口被无关车辆堵住 | 按 `blocking_prefix_vehicle_nos` 清走堵口的车，再把等待的车送进去 |
| `work_position_window` | 作业工位目标股道上 `pending_vehicle_nos` 排列顺序不对 | 按工位顺序要求逐步或批量送入等待车辆 |
| `work_position_free_fill` | 作业工位目标股道有空位，无需清障 | 直接批量送入 `pending_vehicle_nos` |
| `resource_release` / `capacity_release` | 股道超载、路径被堵、已满足车辆占着前端、或精确位被占 | 移走占位车辆（送到其目标或临时股道） |
| `chain_macro_<anchor>` | 多条股道形成连锁问题（≥3 条，`total_pressure ≥ 25`） | 以种子候选为起点，串联后续各段，一次性处理整条链 |

#### 8.2.1 整体生成流水线（`generate_move_candidates`）

```
1. generate_real_hook_moves()
      列出当前所有合法的单步 ATTACH / DETACH
   ↓
2. 原始候选过滤（4 道过滤器）
   ├─ _move_breaks_protected_commitment
   │    不拆散已满足的连续车辆段（CommittedBlock）或已就位的作业窗口
   ├─ _move_mixes_resource_debt_groups
   │    不把目标不同的车辆混进同一次 ATTACH
   ├─ _move_is_unowned_staging_churn
   │    不把"延迟承诺"车辆在临时股道之间乱搬
   └─ _move_attaches_unowned_delayed_staging_buffer
        不取出专门为延迟承诺车辆预留的缓冲存车
   ⚠ 例外：若动作被 _move_is_required_by_resource_debt 识别为
     清障必须（涉及 ResourceDebt 的 vehicle_nos），可绕过第一道过滤
   ↓
3. build_structural_intent() + summarize_debt_chains()
      分析当前局面，整理各股道的问题及严重度
   ↓
4. _generate_structural_candidates()
   ├─ 按 DebtCluster 压力降序遍历每条问题股道：
   │   ├─ OrderDebt → _build_work_position_source_opening_candidate
   │   │               _build_work_position_window_candidate（含 front_block_only 变体）
   │   │               _build_work_position_free_fill_candidate
   │   └─ ResourceDebt → _build_route_release_frontier_candidate
   │                      _build_goal_frontier_source_opening_candidate（FRONT_CLEARANCE）
   │                      _build_resource_release_candidate（其余类型）
   ├─ _generate_chain_macro_candidates()  → chain_macro_* 候选
   └─ _select_structural_candidates()     → 最多保留 6 个
   ↓
5. _trim_candidate_before_delayed_commitment()
      截掉候选中"时机还不到"的放车步骤
      （目标车辆在 DelayedCommitment 列表中，工位窗口尚未就绪）
   ↓
6. _dedup_candidates()  去重
   ↓
   结构候选（排前）+ 过滤后原始候选（排后），整体按 _candidate_search_sort_key 排序
```

#### 8.2.2 各类型构建逻辑详解

**`route_release_frontier`（路径解堵 + 顺手推进，`_build_route_release_frontier_candidate`）**

分三阶段拼出一串步骤：
1. **清除阻塞者**：优先调用 `_build_blocker_direct_goal_candidate` 把 `blocking_track` 北端的堵路车直接送到目标；若目标不可达，则由 `_build_resource_release_dispatch_candidate` 选一个临时股道过渡；
2. **循环推进前沿**（最多 3 次）：每轮调用 `_route_release_frontier_block` 找出被堵车辆中排在各来源股道北端的那批，直接送向目标，直到找不到新的前沿为止；
3. **恢复阻塞者**（可选）：若阻塞者的目标恰好是 `blocking_track`，由 `_restore_committed_blocker_steps` 生成把它从临时股道送回的步骤。

`focus_tracks = (blocking_track, source_track, target_track)`，在 Beam Search 多样性分组中三者合为同一个路径释放焦点。

**`goal_frontier_source_opening`（已满足车辆让路，`_build_goal_frontier_source_opening_candidate`）**

触发：某股道序列呈"已满足前缀 `prefix` + 未满足后续"格局。

构建步骤：
1. 找到 `prefix` 末尾之后第一辆未满足车（`transfer_block`）及其唯一目标股道；
2. 调用 `_build_source_prefix_split_detach_candidate`：暂移 `prefix` → 送出 `transfer_block` → `prefix` 归位；
3. 验证：执行后 `prefix` 仍满足目标，`transfer_block` 已在目标股道上满足——两个条件都成立才接受。

**`work_position_*`（作业工位系列，3 个子类型）**

由 `OrderDebt`（`order_debts_by_track`）驱动：

| 子函数 | reason | 适用情形 | 核心做法 |
|---|---|---|---|
| `_build_work_position_source_opening_candidate` | `work_position_source_opening` | 目标股道北端需先清 N 辆 | 按 `blocking_prefix_vehicle_nos` 疏散阻塞车，再整批送入 `pending_vehicle_nos` |
| `_build_work_position_window_candidate` | `work_position_window` | 需按顺序依次送入 | 逐步或批量完成工位窗口排序；`front_block_only=True` 变体仅处理最靠北的一辆 |
| `_build_work_position_free_fill_candidate` | `work_position_free_fill` | 目标股道有空位且无冲突 | 直接批量送入，无需清障 |

**`resource_release` / `capacity_release`（`_build_resource_release_candidate`）**

统一处理 4 种 `ResourceDebt`：
- `ROUTE_RELEASE`：支持 `allow_partial=True`，可只移走堵路前缀的一部分；
- `CAPACITY_RELEASE`：先直接调度，失败后再用 `_build_protected_prefix_resource_release_candidate`（保护已满足前缀再移后面的）；
- `FRONT_CLEARANCE`：优先走 `_build_goal_frontier_source_opening_candidate` 路径，失败才降级为通用 resource_release；
- `EXACT_SPOT_RELEASE`：直接移走占了别人精确坑位的车（`pressure` 权重最高，×5）。

**`chain_macro_<anchor>`（`_generate_chain_macro_candidates`）**

触发条件：债务链涉及 ≥3 条股道且 `total_pressure ≥ 25.0`，最多处理压力最大的前 2 条链。

构建步骤：
1. 从 `base_candidates` 里选出最优的"种子候选"（`_best_chain_seed_candidate`）；
2. 以种子的终态为起点，继续追加链上其他股道的步骤（最多 3 段，`pressure ≥ 120` 时取 4 段）；
3. 整体经 `replay_candidate_steps` 验证通过且不以 `loco_carry` 结束，才打包；
4. `reason = f"chain_macro_{anchor_track}"`，`focus_tracks` 覆盖链上所有涉及股道。

#### 8.2.3 结构候选筛选（`_select_structural_candidates`，limit=6）

候选超过 6 个时，分三轮筛选：

1. **按债务链槽位分配**：每条链分 3 个槽（`sequence` / `anchor_release` / `non_anchor_release`），`chain_macro_*` 进 `sequence` 槽，覆盖锚点股道的进 `anchor_release`，其余进 `non_anchor_release`；每槽只保留优先级最高的候选；
2. **按 `focus_tracks` 去重**：相同的 `focus_tracks` 组合只保留一个；
3. **按 `(reason, focus_tracks)` 去重**：进一步去重；
4. 若仍不足 6 个，按既有顺序补满。

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

构造法是最底层的保底求解器。它的目标不是先证明最优，而是先尽量**稳定地把局面往可解方向推**，并在预算内给后续 A* 一个较好的起点。

它有两个很重要的定位：
- **保底层**：哪怕后面的精确搜索预算不够，构造法也应尽量给出一份完整方案；实在做不完，也至少返回一份合法的部分方案。
- **统一合法性层**：它不是“手写业务捷径脚本”，而是和搜索层共用同一套候选生成、路径校验、容量校验、作业工位校验与重放验证逻辑。

可以把它理解成一句话：

> 构造法 = “每一步都挑当前最像正解的合法动作”，如果这样一路走下去卡住了，就回到最近几个关键岔路口，换第二名、第三名动作再试。

核心算法：贪心前向搜索（greedy forward）+ **W3-N 有界回溯**：
- 每轮先看当前状态是不是已经完工；
- 若未完工，枚举当前所有合法候选；
- 对候选逐个评分，选分数最低的那个执行；
- 如果连续很多轮都没有真正进展，就认定当前贪心路线走偏了；
- 回到最近的某个决策点，改选次优候选，再往前跑；
- 最多回溯 `max_backtracks`（默认 5）次；
- 所有尝试里，优先保留“已完成且钩数更少”的；若都没完成，则保留“离目标更近”的那条。

### 11.2 先建立一个直觉：它到底在“构造”什么

它构造的不是单个动作，而是一条**逐步成形的 Hook Plan**。

每一轮循环，构造法只做三件事：
1. 看当前局面里，哪些动作现在合法。
2. 判断这些动作里，哪个最值得先做。
3. 执行这个动作，把状态更新后进入下一轮。

所以它不是“先全局规划，再逐步执行”，而是：

```text
当前状态
  -> 找候选
  -> 给候选打分
  -> 选最优
  -> 执行
  -> 得到新状态
  -> 重复
```

这也是为什么它叫“构造法基线”：
- 它不试图一开始就看穿全局；
- 它靠一连串局部正确、业务合理的选择，把整份方案一点点构造出来。

### 11.3 一轮 greedy forward 具体怎么跑

单次前向尝试的主循环，按代码真实顺序大致是这样：

```text
while 还没到目标:
    1. 检查时间预算是否耗尽
    2. 计算当前结构指标（unfinished / blocker / capacity / sequence defect ...）
    3. 决定本轮只看原始动作，还是启用结构候选
    4. 生成所有合法候选
    5. 对每个候选做重放校验并评分
    6. 过滤明显不好的候选（例如最近刚做过的逆动作、重复状态）
    7. 选当前最优候选
    8. 执行候选中的每一步 HookAction
    9. 更新“最近动作”“最近状态”“空载安全检查点”“是否卡住”等信息
```

这里最关键的是第 3、4、5 步。

#### 11.3.1 什么时候只看 primitive，什么时候启用 structural candidate

构造法并不是每轮都强行用复杂宏动作。

它会先判断当前局面有没有明显的“结构性压力”：
- 作业工位顺序错了；
- 某些车已经在目标股道，但把没完成的车压在里面；
- 某条路径长期被前端车辆堵住；
- 临时股道上的“中转债务”开始堆积；
- 延迟承诺（delayed commitment）已经出现。

如果这些压力不明显，它就只看最基本的原子动作：
- 从某股道北端 ATTACH 一段车；
- 或把当前携带的尾段 DETACH 到某条合法股道。

如果压力明显，它就会调用 `generate_move_candidates(...)`，把“结构候选”也纳入比较，例如：
- 先挪开堵口车，再顺手把里面真正该出去的车送走；
- 先拆前缀，再把被压住的目标车送进去，再把前缀复位；
- 一次把某个工位窗口整理到位。

通俗理解：
- **primitive** 像“只走一步”；
- **structural candidate** 像“已经看出这是一整套连动作，干脆成组执行”。

#### 11.3.2 候选不是“想出来就算”，必须先重放验证

这点很重要。

构造法不会因为一个候选“看起来很聪明”就直接采用。每个候选都要先经过 `replay_candidate_steps(...)`：
- 候选里的每一步都重新在当前状态上模拟执行；
- 中间任何一步不合法，整个候选直接丢弃；
- 只有完整重放通过的候选，才有资格参加评分。

所以构造法本质上不是“拍脑袋规则系统”，而是：

> 先提出动作，再用统一的状态转移与业务规则去验它。

这保证了它和 A* 层看到的是同一个“合法动作世界”。

### 11.4 它到底怎么判断“哪个候选更好”

这里最容易让人误解。构造法不是只看一个启发值 `h`，而是看一个**多层排序键**。

也就是说，它并不是简单地问：

> “哪个动作让 h 最小？”

而是更像在问：

> “在业务上更该优先做的动作里，哪个又更能让局面变好？”

代码里大体有两层评分思想。

#### 11.4.1 第一层：先分 Tier，决定“动作意图”优先级

单步动作的优先层不是按学术最优性定义的，而是按业务直觉定义的。

当前代码的核心分层可以通俗理解为：

| Tier | 直觉含义 |
|---|---|
| 0 | 已经可以把车直接放到更优的最终位置，尤其还能减少首选违规 |
| 1 | 已经可以把车直接放到有效目标 |
| 2 | 这是明显的建设性动作：解路径堵塞、暴露关键车、延续解堵链、清精确位、支持关门车推进、或 ATTACH 后能让整体进度变好 |
| 3 | 主要是在清障，虽然还没直接完成目标，但能把局面打开 |
| 4 | DETACH 后能带来进展，但价值弱于前几层 |
| 5 | 暂存/中转类动作，必要时才做 |
| 6 | 其他兜底动作 |

这里的关键不是记住编号，而是理解排序哲学：
- **先做真正完成目标的动作**；
- **做不了最终动作时，优先做能打开局面的动作**；
- **再不行才做中转和铺垫**；
- **明显会把局面搞脏的动作放到最后**。

另外，很多本来看起来“还不错”的动作，会被额外降级到 Tier 5，例如：
- 把车又停回正在形成路径阻塞的股道；
- 刚清掉精确位问题，又把它堵回去；
- 把关门车推进所依赖的“推手车”提前用掉；
- 去重新抓刚放到临时股道、但其实还没到该回收时机的车。

也就是说，构造法不是只奖励“眼前进展”，还会主动避开**看似推进、实则埋雷**的动作。

#### 11.4.2 第二层：同层内再看结构进展、污染、路径长度等细节

如果两个候选 Tier 一样，就继续比较更细的指标，例如：
- 执行后未完成车辆是否更少；
- 作业工位顺序缺陷是否减少；
- 容量超载股道是否减少；
- 是否减少了目标股道前端阻塞；
- 是否增加了 staging 污染；
- 是否让 preferred/fallback 违例变多；
- 一次动作是否过长、是否把车埋得更深。

文档里可以把这个理解成：

> Tier 决定“方向对不对”，细粒度评分决定“同样方向下谁更干净、更省事”。

#### 11.4.3 它优化的不是一个数，而是一个“进展向量”

构造法内部长期跟踪的不是单个 `h`，而是一个进展键：

```python
(
    heuristic,
    preferred_violation_count,
    staging_pollution_count,
    unfinished_count + target_sequence_defect_count,
)
```

这表示它判断“局面是否更好”时，会同时看：
- 离目标还有多远；
- 是否在破坏首选目标；
- 是否把临时股道越用越脏；
- 是否真的在减少未完成任务，而不是只是换个地方堆着。

这比单纯盯着“还有几辆车没到位”要稳得多。

### 11.5 为什么有时会“走几步看起来没变好”

很多调车局面必须先清障，才能出现真正的目标动作。

例如：
- A 车应该进修1库内 3 号位；
- 但修1前面压着一段已经满足别的目标的车；
- 这时先把那段车临时挪走，表面看“目标车数量没减少”；
- 甚至启发值短时间内都不下降；
- 但这是必须的建设性准备。

因此代码把“无直接下降但属于 purposeful clearing 的动作”和“纯粹横跳的动作”分开看待：
- 对前者，允许连续更久都不降启发值；
- 对后者，较快认定为 stale。

这就是 `purposeful_stale` 和 `stale_rounds` 分离的原因。

它反映的是一个很重要的业务事实：

> “清障链条中的几步铺垫”不能和“无意义来回折腾”混为一谈。

### 11.6 它怎么判断“这条贪心路线走偏了”

构造法不是一旦某一步没变好就回溯，而是看一段连续过程。

主要有三种停止当前尝试的原因：
- **没有合法候选**：当前状态已经无路可走；
- **时间/步数预算耗尽**：这一轮尝试不能再继续；
- **长期无进展**：说明可能陷入局部最优或搬运循环。

尤其是“长期无进展”，代码分两类：
- `stale_rounds`：偏回归/横移动作连续太多；
- `purposeful_stale`：虽然是在清障，但太久都没有兑现成真实收益。

一旦认定卡住，就返回当前最好结果。但这里还有一个很关键的设计：**空载安全检查点**。

### 11.7 为什么只在“空载时”记安全检查点

构造法运行中会持续记住最近一个“调机未携带车辆”的状态。

原因很实际：
- 如果卡住时调机还挂着一串车，直接把这份部分方案交给后续阶段，后续处理会更难；
- 空载状态更“干净”，更适合让后续 A*、partial resume、tail rescue 接手；
- 业务上也更像一个自然的钩次边界。

因此，如果当前尝试在“带车状态”下卡住，构造法不会把这段半截 carry 强行作为部分结果输出，而是：
- 回退到最近一个空载检查点；
- 把那之前的计划作为 partial plan 返回。

这就是文档前面提到“部分续跑会从构造法中间空载检查点继续”的根本原因。

### 11.8 W3-N 有界回溯到底怎么工作

回溯不是把整条历史全部推翻重来，而是“在最近的若干关键选择点中，换一条分支再试”。

具体过程可以理解为：

1. 先完整跑一遍纯贪心。
2. 如果成功，直接结束。
3. 如果失败，从最近 `REWIND_WINDOW`（默认 30）个决策点里，倒着找一个还有备选动作没试过的位置。
4. 把那个位置的选择从“第 1 名”改成“第 2 名”或“第 3 名”。
5. 该位置之后的所有选择全部重新按贪心生成。
6. 重复最多 `max_backtracks` 次。

这里的“W3-N”可以粗暴理解成：
- **W**：只在最近的一个窗口里优先找回溯点，不做全局爆炸式回退；
- **3**：每个决策点最多尝试前 3 个备选；
- **N**：可重复多个回合，但有上限。

这个设计解决的是典型问题：

> 某个局面里，第一名候选看起来局部最优，但它会把关键通道占掉；第二名才是全局上更顺的。

纯贪心常死在这里，而有界回溯可以用很小代价修正这类误判。

### 11.9 一个通俗例子：为什么“构造法基线”有时先挪开，再送目标车

假设某条股道从北到南是：

```text
[X, Y, Z]
```

其中：
- `X` 已经满足自己的目标；
- `Y` 也暂时不该动；
- `Z` 才是真正急着要去修1工位的车；
- 但规则决定只能从北端取车。

那构造法不可能直接拿到 `Z`。它只能在候选里比较：
- 把 `X`、`Y` 暂时挪走；
- 还是去别处做别的事；
- 或者做某种复合结构候选：`前缀让路 -> 送出 Z -> 前缀复位`。

这时若结构候选可行，它往往会赢，因为它在评分上同时满足：
- 这是一个建设性清障动作；
- 清完以后 `Z` 可以马上进入有效目标；
- 被暂移的前缀还能恢复；
- 整体结构缺陷、未完成数或 blocker 压力会下降。

所以从阅读视角看，构造法并不是“乱挪车”，而是在做一种非常典型的调车动作：

> 先开门，再送核心车，再尽量恢复现场。

### 11.10 读日志或看结果时，怎么判断构造法当前在干什么

如果你在调试时看到构造法输出了一长串动作，可以先用下面这套思路判断它的阶段：

1. **如果频繁出现直接 DETACH 到目标股道/工位**：
   说明它在“兑现成果”。
2. **如果连续在搬前缀、临时清障，但目标车随后被送出**：
   说明它在走“结构性清障链”。
3. **如果大量动作集中在 staging 轨道之间来回**：
   说明它可能已经接近局部循环，要看是否即将 stale 或触发回溯。
4. **如果 partial plan 停在空载位置**：
   说明它主动把结果截在了更适合后续续跑的检查点。

### 11.11 这一层的价值和边界

构造法很强，但它不是万能的。

它擅长：
- 快速把大部分“明显该怎么拆”的局面拆开；
- 给 A* 提供接近终点的暖启动；
- 在复杂业务约束下提供稳定保底。

它的天然边界是：
- 仍然以局部评分为主，不保证全局最优；
- 对“先吃亏、后大赚”的深层策略看得不如搜索层远；
- 当多个清障方向都看起来差不多时，仍可能选错分支，需要靠回溯修正。

所以整体架构才会是：
- **先用构造法把局面做顺**；
- **再用 A* / anytime / LNS 去补最优性和长尾问题**。

### 11.12 逆动作保护（Inverse Guard）

记录最近 12 步动作。如果一个候选动作与最近某步相反（把刚送走的一半以上车辆送回来），且有其他非逆向候选，则优先选非逆向。

这条规则的意义非常朴素：
- 调车里允许“必要回收”；
- 但不允许 solver 在两个股道之间来回打摆子。

因此它不是绝对禁止逆动作，而是：
- **有别的合理动作时，优先不走回头路**；
- **真没别的路时，逆动作仍可作为兜底**。

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

`build_structural_intent` 分析当前状态的宏观目标，返回 `StructuralIntent`。

### 13.1 `StructuralIntent` 包含哪些信息

`StructuralIntent` 包含 7 类字段，供 `_generate_structural_candidates` 生成候选时使用：

**`committed_blocks_by_track`（各股道的稳定车辆段）**

记录每条股道上已经"稳稳就位、不该被动"的连续车辆段（`CommittedBlock`）。来源有两种：
- 北端连续满足目标的车辆（`goal_satisfied_contiguous_block`）；
- SPOTTING 工位上已全部满足的作业窗口段（`stable_work_position_window`）。

用于原始候选过滤：若某个 ATTACH 动作会拆散这样的段，就被 `_move_breaks_protected_commitment` 过滤掉（除非该动作是清障必需的）。

**`order_debts_by_track`（作业工位排列欠账）**

每条作业工位目标股道上有哪些排列问题（`OrderDebt`）：
- `defect_count`：当前排列缺陷数；
- `pending_vehicle_nos`：还没进去、等待入位的车辆；
- `blocking_prefix_vehicle_nos`：目标股道北端需要先移走的挡路车——通过 `_insertion_clear_count` 模拟插入来估算最少要清几辆；
- `kind_counts`：`SPOTTING` / `EXACT_NORTH_RANK` / `EXACT_WORK_SLOT` 各有几辆在等。

**`resource_debts`（4 种资源占用问题）**

| `kind` | 触发条件 | `vehicle_nos` 含义 | 压力权重 |
|---|---|---|---|
| `ROUTE_RELEASE` | 股道上的车辆堵住了别人的行驶路径 | 挡路的车 | ×3 |
| `CAPACITY_RELEASE` | 股道超载 | 前端需要移走的车 | ×1.5 |
| `FRONT_CLEARANCE` | 已满足目标的车堵在未满足车辆前面 | 前端已满足的车 | ×2 |
| `EXACT_SPOT_RELEASE` | 某辆车的目标精确位被另一辆车占着 | 占位的那辆车 | ×5 |

**`debt_clusters_by_track`（按股道汇总的综合严重度）**

把同一条股道上的 `OrderDebt` 和 `ResourceDebt` 合并成一个 `DebtCluster`，计算综合压力分：

```
pressure = defect_count×10 + len(pending)×1 + len(blocking_prefix)×2  # 排列欠账
         + 各 ResourceDebt 的 pressure × 对应权重                      # 资源占用
```

`debt_clusters_by_track` 按 `(-pressure, track_name)` 排序，`_generate_structural_candidates` 优先处理压力最高的股道。

**`staging_buffers`（临时股道可用情况）**

临1～4、存4南各自的 `free_length`（剩余空闲米数）、`occupied_vehicle_count` 以及 4 种角色分值（空股道额外 +2）。候选生成时，`_best_staging_track` 用这里的数据挑选最合适的临时停放目标。

**`delayed_commitments`（暂时放不进去的车辆）**

某辆车虽然有目标股道，但现在还不能送过去（`DelayedCommitment`），有两种原因：
- `work_position_window_not_ready`：放进去后工位窗口无法满足；
- `would_precede_unfinished_work_position_window`：会排在尚未完成的 SPOTTING 窗口前面。

被标记的车辆会在 `_trim_candidate_before_delayed_commitment` 阶段把涉及它们的放车步骤截掉，除非候选类型本身是 `work_position_*`（由工位系列专门处理）。

**`buffer_leases`（为等候车辆预留的临时停放计划）**

对每组 `DelayedCommitment` 车辆，记录其来源股道、目标股道、所需米数（`required_length`）及借用角色 `ORDER_BUFFER`，供候选生成时优先选用合适的缓冲股道。

### 13.2 `build_structural_intent` 调用链

```
build_structural_intent(plan_input, state, route_oracle)
│
├── compute_structural_metrics()        各股道基础指标（未完成数、排列缺陷数等）
├── compute_route_blockage_plan()       路径阻塞事实（哪条股道挡了哪些车的路）
├── compute_capacity_release_plan()     容量超载事实（各超载股道需释放多少米）
│
├── _committed_blocks_by_track()        → CommittedBlock（连续满足段 + 稳定工位窗口）
├── _delayed_commitments()              → DelayedCommitment（工位窗口未就绪 / 排序冲突）
├── _order_debts_by_track()             → OrderDebt（各目标股道的排列欠账）
│     └─ _insertion_clear_count()         模拟插入：最少要清几辆前缀才能让等待车顺利入位
├── _resource_debts()                   → ResourceDebt（ROUTE / CAPACITY / FRONT / SPOT 四种）
├── build_debt_clusters_by_track()      → DebtCluster（按压力聚合，降序排列）
├── _staging_buffers()                  → StagingBuffer（各临时股道的空闲状态快照）
└── _buffer_leases()                    → BufferLease（为延迟承诺车辆预留临时停放计划）
```

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
