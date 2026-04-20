# 求解过程深度拆解

> 本文面向刚上手这个项目的开发者。目标是让你在读完后：
>
> - 知道"求解"到底是怎么发生的——从 JSON 进来到 `plan` 吐出去的每一层
> - 认得每个文件扮演什么角色、哪一行代码干了最关键的事
> - 理解"为什么这样设计"——而不是"它是这样写的"
>
> 本文与 [solver-logic-explained.md](solver-logic-explained.md) 互补：那篇偏概念直觉，本篇偏代码路径与工程决策。两者对照读收益最大。

---

## 目录

1. [一张图先看懂全局](#1-一张图先看懂全局)
2. [数据流：从 JSON 到最终 plan](#2-数据流从-json-到最终-plan)
3. [核心数据结构四件套](#3-核心数据结构四件套)
4. [主入口：solve_with_simple_astar_result 拆解](#4-主入口solve_with_simple_astar_result-拆解)
5. [动作生成：求解的"燃料供应"](#5-动作生成求解的燃料供应)
6. [搜索主循环：A\*/Weighted/Beam 三模合一](#6-搜索主循环aweightedbeam-三模合一)
7. [启发式：五个分量的 max](#7-启发式五个分量的-max)
8. [状态键与去重：为什么不直接用字符串](#8-状态键与去重为什么不直接用字符串)
9. [构造保底：永远返回一条计划的兜底](#9-构造保底永远返回一条计划的兜底)
10. [Anytime Fallback Chain：七阶段递进](#10-anytime-fallback-chain七阶段递进)
11. [LNS：先求后磨的局部修复](#11-lns先求后磨的局部修复)
12. [终止判定：什么叫"解完了"](#12-终止判定什么叫解完了)
13. [Verify：独立口径的最终复核](#13-verify独立口径的最终复核)
14. [端到端示例：跟踪一次完整求解](#14-端到端示例跟踪一次完整求解)
15. [Debug 统计：怎么读懂搜索在干什么](#15-debug-统计怎么读懂搜索在干什么)
16. [源码阅读路线与模块依赖](#16-源码阅读路线与模块依赖)
17. [当前 benchmark 基线](#17-当前-benchmark-基线)

---

## 1. 一张图先看懂全局

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     solve_with_simple_astar_result                        │
│                      (src/fzed_shunting/solver/astar_solver.py)           │
│                                                                           │
│  Step 0: _validate_final_track_goal_capacities  ← 预检，直接 raise         │
│                                                                           │
│  Step 1: Constructive seed (保底)  ─┐                                     │
│          constructive.solve_constructive  │ 7 tier 贪心分派               │
│                                    │  不参与搜索，只作兜底                 │
│                                    ▼                                      │
│  Step 2: Search (主搜索)                                                  │
│          search._solve_search_result                                      │
│            exact / weighted / beam  (3 模式共用一个主循环)                 │
│                                    │                                      │
│                    ┌───────────────┼──────────────┐                       │
│                    ▼                              ▼                       │
│  Step 3a: Anytime chain (exact)     Step 3b: Post-repair (beam)           │
│          anytime.fallback_stages              lns._improve_incumbent      │
│          weighted → beam ×4 → …               4 个 destroy 算子           │
│          共 7 阶段，每阶段占 8%~12% 预算                                  │
│                                    │                                      │
│                                    ▼                                      │
│  Step 4: LNS mode (独立路径)                                              │
│          lns._solve_with_lns_result                                       │
│          先 seed → destroy → repair                                       │
│                                    │                                      │
│                                    ▼                                      │
│  Step 5: Constructive fallback                                            │
│          如果以上全部空返，用 Step 1 的 seed 兜底                         │
│                                    │                                      │
│                                    ▼                                      │
│  Step 6: _attach_verification (verify=True)                               │
│          verify.plan_verifier.verify_plan 独立复演                        │
│          不通过 → raise PlanVerificationError                             │
│                                    │                                      │
│                                    ▼                                      │
│                              SolverResult                                 │
│                    (plan + 遥测 + verification_report)                    │
└───────────────────────────────────────────────────────────────────────────┘
```

记住这张图的分层逻辑：**预检 → 兜底 → 主搜索 → 回退链 → 兜底替补 → 独立复核**。任何时候回头看代码，都能把它对回到这六层里的某一层。

---

## 2. 数据流：从 JSON 到最终 plan

```
原始 payload (dict/JSON)
    │  normalize_input.py → NormalizedPlanInput
    ▼
NormalizedPlanInput           ←─ track_info / vehicles / loco_track_name / yard_mode
    │  replay.build_initial_state
    ▼
ReplayState (初始)            ←─ track_sequences / loco_track_name /
    │                            weighed_vehicle_nos / spot_assignments
    │  solver.* (搜索)
    ▼
list[HookAction]              ←─ source/target/vehicle_nos/path_tracks
    │  plan_verifier.verify_plan
    ▼
PlanVerificationReport        ←─ is_valid / global_errors / hook_reports
    │
    ▼
SolverResult                   ←─ plan + verification_report + 遥测
```

这四次转换之间的边界都是纯数据（pydantic / dataclass），没有副作用，也没有隐藏状态。这是为什么 demo 回放、benchmark、多阶段 workflow 都能直接在中间任意一层切一刀重新用——**每一层的输入输出都是自包含的**。

---

## 3. 核心数据结构四件套

只要死磕这四个就够。

### 3.1 `NormalizedPlanInput` (io/normalize_input.py:39)

求解器唯一接受的输入结构。字段：

| 字段 | 含义 |
|---|---|
| `track_info` | 每条股道的名字和有效长度（容量上限的来源） |
| `vehicles` | 每辆车：当前股道、顺序、长度、`goal`、`need_weigh`、`is_heavy`、`is_close_door` |
| `loco_track_name` | 机车当前在哪条股道 |
| `yard_mode` | `NORMAL` 或 `INSPECTION`；影响大库台位开放集合 |

重点看 `NormalizedVehicle.goal`（`GoalSpec`）：

```python
class GoalSpec(BaseModel):
    target_mode: str               # "TRACK" | "AREA" | "SPOT"
    target_track: str              # 名义目标
    allowed_target_tracks: list[str]  # 真正参与判定的集合
    target_area_code: str | None   # 例如 "大库:RANDOM"、"调棚:WORK"
    target_spot_code: str | None   # 精确台位（SPOT 模式下）
```

这里最容易踩坑的点：**`allowed_target_tracks` 才是搜索器使用的字段，`target_track` 只是名义值**。比如 `大库:RANDOM` 的 `allowed_target_tracks` 会是 `["修1库内", "修2库内", "修3库内", "修4库内"]`。

### 3.2 `ReplayState` (verify/replay.py:11)

搜索里"一个状态"的实际载体。只有 4 个字段：

```python
class ReplayState(BaseModel):
    track_sequences: dict[str, list[str]]  # 每条股道当前车辆序列（北→南）
    loco_track_name: str                    # 机车当前位置
    weighed_vehicle_nos: set[str]           # 已称重车辆集合
    spot_assignments: dict[str, str]        # {vehicle_no: spot_code}
```

**为什么需要这四个？**

- `track_sequences` 是物理现场——没它就不知道谁挡着谁
- `loco_track_name` 是机车位置——目前主要用于展示与 workflow 阶段继承
- `weighed_vehicle_nos` 是历史——车离开过机库不代表称过，反之车现在不在机库也不代表没称过
- `spot_assignments` 是台位层——光到库线还不够，必须精确到"占了哪个台位"

这四个拼在一起就是**当前局面的完整刻画**。

### 3.3 `HookAction` (solver/types.py:6)

一条钩动作。所有动作类型都是 `PUT`：

```python
class HookAction(BaseModel):
    source_track: str        # 从哪条股道北端挂出来
    target_track: str        # 送到哪条股道尾部
    vehicle_nos: list[str]   # 必须是 source_track 的北端前缀
    path_tracks: list[str]   # 完整经过的股道序列（含首尾）
    action_type: str = "PUT"
```

**不变式**：

1. `vehicle_nos` 必须等于 `state.track_sequences[source_track][:len(vehicle_nos)]`——即北端前缀
2. `path_tracks[0] == source_track and path_tracks[-1] == target_track`
3. `path_tracks[1:-1]` 里任何股道不能被别的车占着（除非业务规则允许）

这三条不变式在搜索、replay、verify 三处都会被重复检查，任何一处漏掉都能被抓出来。

### 3.4 `SolverResult` (solver/result.py:11)

```python
@dataclass(frozen=True)
class SolverResult:
    plan: list[HookAction]
    expanded_nodes: int
    generated_nodes: int
    closed_nodes: int
    elapsed_ms: float
    is_proven_optimal: bool = False    # 只有 exact 跑完才 True
    fallback_stage: str | None = None  # 产出最终 plan 的阶段名
    verification_report: Any | None = None
    debug_stats: dict[str, Any] | None = None
```

注意 `fallback_stage`——这是调试生产问题的第一抓手。看到 `"constructive_partial"` 就知道搜索全挂了，看到 `"beam_greedy_256"` 就知道已经走到回退链很深的地方。

---

## 4. 主入口：`solve_with_simple_astar_result` 拆解

这个函数（`astar_solver.py:76-207`）是所有求解的唯一入口。它大约 130 行，但你必须逐段看懂，因为整个求解系统的调度逻辑都在这里。

我把它逐块拆开注释：

### 4.1 参数校验与预检（Line 90-97）

```python
_validate_solver_options(...)                       # 检查 solver_mode / heuristic_weight / beam_width
if verify and master is None:
    raise ValueError(...)                           # verify 必须有 master
_validate_final_track_goal_capacities(plan_input)   # 硬检查：最终目标股道装不下就直接拒
```

`_validate_final_track_goal_capacities` 很重要：它在进入任何搜索之前，先把"明摆着容量溢出"的场景拒掉。注意它只对 `target_mode == "TRACK"` 的目标做检查，因为 AREA/SPOT/RANDOM 的最终股道不唯一，不好静态算。

### 4.2 Stage 0：构造保底（Line 100-112）

```python
if enable_constructive_seed and enable_anytime_fallback and solver_mode == "exact":
    constructive_seed = _run_constructive_stage(...)
```

只在 **`solver_mode == "exact"` + `enable_anytime_fallback=True` + `enable_constructive_seed=True`** 三个条件同时满足时跑。为什么？

- LNS 模式自己会先求 seed，不需要再跑一遍
- beam / weighted 模式通常是用户手动指定的"我知道自己要什么"，尊重用户选择
- exact + anytime 是默认路径，必须有兜底

兜底预算 = `min(5000ms, max(200ms, time_budget_ms × 5%))`。抠这么紧是因为它是保险丝，不是主力。

### 4.3 主搜索调度（Line 114-144）

```python
if solver_mode == "lns":
    result = _solve_with_lns_result(...)
else:
    exact_time_budget_ms = time_budget_ms
    if solver_mode == "exact" and enable_anytime_fallback:
        exact_time_budget_ms = max(1.0, time_budget_ms * 0.4)  # exact 只用 40%
    result = _solve_search_result(...)
```

**关键决策**：如果是 exact + anytime，exact 阶段只拿 40% 总预算，剩下 60% 留给回退链。设计意图：宁可"快速确认 exact 打不出来就走回退"，也不"exact 磕到最后一刻然后没时间回退"。

### 4.4 Anytime 回退链（Line 160-173）

```python
if solver_mode == "exact" and enable_anytime_fallback and not result.is_proven_optimal:
    result = _anytime_run_fallback_chain(...)
```

只有 exact 模式、且没被证明最优（要么超时要么耗尽队列）时才触发。详见 [§10](#10-anytime-fallback-chain七阶段递进)。

### 4.5 Beam 后修补（Line 174-191）

```python
if solver_mode == "beam" and beam_width is not None:
    for _ in range(BEAM_POST_REPAIR_MAX_ROUNDS):     # = 1
        candidate = _improve_incumbent_result(..., repair_passes=1, max_rounds=1)
        if len(candidate.plan) >= len(improved.plan):
            break
        improved = candidate
```

beam 找到的解不一定最短，所以再跑一轮 LNS 修补。这里的参数被故意压得很小（`repair_passes=1`, `max_rounds=1`），是因为它是锦上添花。

### 4.6 兜底替补（Line 196-198）

```python
if not result.plan and constructive_seed is not None and constructive_seed.plan:
    result = constructive_seed
```

**SLA 契约的最后一道关**：如果前面所有搜索都打空，用 Stage 0 的 seed 顶上。这就是为什么即使 `constructive_partial`（未达标的部分解）也要返回——SLA 承诺"非空 plan"。

### 4.7 独立复核（Line 200-206）

```python
if verify:
    result = _attach_verification(result, plan_input, master, initial_state)
```

详见 [§13](#13-verify独立口径的最终复核)。

---

## 5. 动作生成：求解的"燃料供应"

`move_generator.generate_goal_moves`（move_generator.py:24）是整个系统里最厚的文件——650 行。每一次展开状态都要调用一次，它的产出质量直接决定搜索效果。

### 5.1 两类候选：直接动作 + 临停动作

```
generate_goal_moves
  ├── 对每条非空股道 source_track：
  │     ├── ① direct moves：把北端前缀直接送目标
  │     │     for prefix_size in range(len(seq), 0, -1):
  │     │         block = seq[:prefix_size]
  │     │         if 目标一致:
  │     │             candidate_targets = _candidate_targets(block, ...)
  │     │             for target_track in candidate_targets:
  │     │                 _build_candidate_move(...)  ← 硬过滤全部在这
  │     │
  │     └── ② staging moves：前挡车清障 / 路径清障
  │           staging_requests = _collect_staging_requests_for_source(...)
  │           for prefix_size in staging_requests:
  │               for target_track in _candidate_staging_targets(...):  ← 临停线候选
  │                   _build_candidate_move(...)
  └── _dedup_moves(moves)   ← 相同 (source, target, vehicles, path) 去重
```

### 5.2 北端前缀枚举的方向：从长到短

```python
for prefix_size in range(len(seq), 0, -1):  # 从整条股道到单辆车
```

为什么从长到短？因为**一次搬多辆的解通常更短**。如果长 block 能直达，就不想再生成短 block 的动作（`_should_skip_shorter_single_target_direct_move` 专门干这个裁剪）。

### 5.3 `_candidate_targets`：候选目标怎么决定（Line 298-322）

这是一条非常短但非常关键的路径：

```python
def _candidate_targets(block, plan_input, state, vehicle_by_no):
    # 条件 1：如果有人需要称重 → 强制送机库
    if 有车 need_weigh 且未称:
        return ["机库"]
    # 条件 2：正常情况 → goal.allowed_target_tracks
    targets = list(goal.allowed_target_tracks)
    # 条件 3：大库随机 → 按当前占用排序，"让满的更满"
    if goal.target_area_code == "大库:RANDOM":
        targets.sort(key=lambda t: (-当前占用长度(t), t))
    return targets
```

**三条设计决策**：

1. **称重的优先级被硬编码**：而不是让搜索器自己发现。原因：如果"搜着搜着才发现该先称"，搜索树会爆炸式膨胀。
2. **目标一致才生成直达动作**：混目标的 block 几乎永远不是一步到位，不如不生成。
3. **大库随机按满度排序**：避免车辆过度分散导致状态空间爆炸。

### 5.4 `_build_candidate_move`：硬过滤栈（Line 539-598）

动作被候选并不等于能进搜索。每个候选还要过以下 8 道硬过滤（任何一道失败都 `return None`）：

| # | 检查 | 代码位置 |
|---|---|---|
| 1 | `target_track == source_track` → 拒 | Line 552 |
| 2 | 关门车大钩规则 | `_violates_close_door_hook_rule`, Line 601 |
| 3 | 股道容量 | `_fits_capacity`, Line 284 |
| 4 | 作业区容量（调棚/洗南/油/抛） | `_fits_area_capacity`, Line 631 |
| 5 | 台位可分配性 | `allocate_spots_for_block`, depot_spots.py:64 |
| 6 | 路径拓扑存在 | `route_oracle.resolve_path_tracks` |
| 7 | 路径验证（中间净空、L1 190m、reverse clearance） | `route_oracle.validate_path` |
| 8 | 单钩车辆组合（空车≤20、重车≤2、折算≤20、称重≤1） | `validate_hook_vehicle_group`, hook_constraints.py:8 |

所有硬过滤都发生在动作生成阶段——搜索器拿到的动作已经是"业务合法"的。这是为什么搜索代码本身可以很薄：它不用管业务规则，只管组合策略。

### 5.5 临停清障：staging_requests

两种触发条件（`_collect_staging_requests_for_source`, Line 390-417）：

**A. 前挡车清障**：`_front_blocker_prefix_size`

如果某条股道的北端若干辆车已经在自己的合法目标上，但挡住了后面需要继续走的车，就生成 staging 动作把前缀挪到临停线。

**B. 路径清障**：`blocking_goal_targets_by_source`

如果某条股道上的车挡住了别的动作的 `path_tracks[1:-1]`，就把这条股道的车挪到临停线。

**临停线候选排序**（`_candidate_staging_targets`, Line 325-363）：

```python
优先级 = (
    track_type_priority,           # TEMPORARY 优先于 STORAGE
    source→staging + staging→goal, # 组合距离最小
    source→staging,                # 打平用源距离次优
    track_name,
)
```

并且 `MAX_STAGING_TARGETS = 2`——每个 staging request 最多生成 2 个临停目标候选，避免搜索树爆炸。

---

## 6. 搜索主循环：A\*/Weighted/Beam 三模合一

`search._solve_search_result`（search.py:47-214）是唯一的主循环实现。三种模式的差异**只在优先级计算和队列剪枝**，循环骨架完全一样。

### 6.1 循环骨架（伪代码）

```python
queue = 优先队列  # heapq
best_cost = {initial_key: 0}
while queue and not budget.exhausted():
    current = heappop(queue)
    # 跳过已被更优路径覆盖的旧版本
    if best_cost[current.state_key] != len(current.plan):
        continue
    if _is_goal(current.state):
        if exact: return current.plan  # 第一个到达目标就是最优
        else:     记为 best_goal_plan；继续搜
    moves = generate_goal_moves(current.state)
    for move in moves:
        next_state = _apply_move(current.state, move)
        next_key = _state_key(next_state)
        if next_key 已有更短路径: continue
        best_cost[next_key] = len(current.plan) + 1
        heappush(queue, next_item)
    if beam: _prune_queue(queue, beam_width)
return best_goal_plan or raise ValueError
```

### 6.2 优先级 tuple（`_priority`, Line 217-237）

优先级是 **5 元组**，按字典序比较：

```python
# exact
(cost + heuristic, cost, heuristic, -blocker_bonus, 0)

# weighted
(cost + heuristic_weight * heuristic, cost, heuristic, -blocker_bonus)

# beam
(cost + (heuristic - beam_heuristic_credit), cost, adjusted_heuristic,
 -blocker_bonus, heuristic)
```

**为什么要多列？**

- 第 0 位：f 值。是 A* 的核心打分
- 第 1 位：cost (=plan length)。f 值相同时偏向已走钩数少的（因为 h 更紧）
- 第 2 位：heuristic。再打平时偏向 h 更小的
- 第 3 位：`-blocker_bonus`。**把挡着其他目标路径的前缀整块清掉的动作，优先级被拉高**
- 第 4 位（仅 beam）：raw heuristic，保留原始信息

### 6.3 `blocker_bonus` 的巧妙之处

`_blocking_goal_target_bonus`（search.py:240-256）：

```python
def _blocking_goal_target_bonus(state, move, blocking_goal_targets_by_source):
    # 如果这一钩把整条 source_track 都清掉了
    # 而且 source_track 上的车原本在挡别人的目标路径
    # 给这个动作一个正的奖励，拉高优先级
    if move.target_track in blocking_targets:
        if len(move.vehicle_nos) == len(source_seq):
            return len(blocking_targets)
    return 0
```

直觉上：**"一次清空整条股道" 比 "分次搬" 更可能让后续动作解锁**。给这种动作加分，等于让搜索器偏向"先集中处理挡路股道"的决策。

### 6.4 Beam 剪枝的玄机（`_prune_queue`, Line 259-304）

简单 beam 是"保留前 N 个"，但这里多了两条保留规则：

```python
1. shallow_candidates[:1]  # 深度最浅的 1 个一定留（避免 beam 被深节点挤爆）
2. 第一个 blocker_bonus > 0 的 item 一定留  # 清路动作再穷也不能丢
3. 其余按 priority 前 N - 保留数 填
```

被裁掉的 item 还要从 `best_cost` 里同步删除，否则后续相同状态会被误判为"已探索过更优"而跳过。

### 6.5 预算两维：时间 + 节点

`SearchBudget`（budget.py:8）同时管两种预算：

```python
def exhausted(self) -> bool:
    if time_budget_ms is not None and elapsed_ms() >= time_budget_ms: return True
    if node_budget is not None and nodes_expanded >= node_budget: return True
    return False
```

每次 `heappop` 后调用一次 `budget.tick_expand()` 与 `budget.exhausted()`。超了就**跳出循环，返回当前 best_goal_plan**（可能是 None，也可能已经有一个次优解）。

---

## 7. 启发式：五个分量的 max

`heuristic.compute_admissible_heuristic`（heuristic.py:29）返回 **5 个可采纳下界的 max**：

```python
def value(self) -> int:
    return max(
        self.h_distinct_transfer_pairs,  # 不同 (源,目标) 对数
        self.h_blocking,                 # 需被清障的目标股道数
        self.h_weigh,                    # 还没称的称重车数
        self.h_spot_evict,               # 需挪位的 SPOT 占用者
    )
```

注意 `h_misplaced` 在 `HeuristicBreakdown` 里计算但不参与 max——它过于宽松，几乎总被其它分量 dominates。保留主要是为了遥测。

### 7.1 为什么是 max 而不是 sum？

**max 永远是可采纳的下界**（每一种分量都是一个独立的下界估计，真实代价 ≥ 任何一个下界）。sum 需要分量之间**独立**才可采纳——而这 5 个分量明显共享代价（同一钩可能同时消灭多个分量的"欠账"），不独立，所以 sum 会高估。

### 7.2 `h_distinct_transfer_pairs`：核心下界

```python
# heuristic.py:130-181
forced_pairs = set()  # (current_track, allowed_target_track) 对
multi_target_buckets = {}  # 多目标源 → 允许目标集合

for vehicle:
    if current in allowed: continue
    if len(allowed) == 1:
        forced_pairs.add((current, allowed[0]))
    else:
        multi_target_buckets[current].append(frozenset(allowed))

extra = 每个源有未被 forced 覆盖的多目标组 → +1
return len(forced_pairs) + extra
```

**直觉**：每一钩只搬一个 `(source, target)` 对。如果有 N 个不同的对需要被搬，至少要 N 钩。

**可采纳性证明草稿**：对于任意最优 plan，每一钩消除至多 1 个 forced pair；每个源的多目标组至少要 1 钩额外动作（无法被 forced pair 共享时）。所以真实代价 ≥ `|forced_pairs| + extra`。

### 7.3 `h_blocking`：挡路清障下界

```python
# heuristic.py:184-214
for goal_track in all_target_tracks_needed:
    seq = state.track_sequences[goal_track]
    # 数前缀中"不把这条股道当合法目标"的车辆数
    if 至少有 1 个 blocker:
        blocker_count += 1
return blocker_count
```

每条目标股道最多贡献 1——即"这条股道需要被清障至少 1 次"。这是一个很保守的下界，但永远可采纳。

### 7.4 `h_weigh`：称重下界

```python
# heuristic.py:217-229
return sum(
    1 for vehicle in need_weigh_vehicles
    if vehicle_no not in state.weighed_vehicle_nos
)
```

**每辆未称的称重车都需要至少 1 钩进机库**。单钩称重最多 1 辆，所以这是紧下界。

### 7.5 `h_spot_evict`：SPOT 占用冲突下界

```python
# heuristic.py:240-264
for vehicle with target_mode == "SPOT":
    current_occupant = 当前占着这个台位的人
    if current_occupant 不是自己 且不是 None:
        + 1  # 这个占用者必须先被挪走
```

### 7.6 `make_state_heuristic`：预计算的 closure

为什么搜索里不直接调 `compute_admissible_heuristic`？因为每次都要重新构造 `vehicle_by_no` 和 `weigh_vehicle_nos` 集合，几万次下来是一笔可观的开销。

`make_state_heuristic`（heuristic.py:67）预构造一次，返回一个只吃 `state` 的闭包，搜索主循环里调用的是这个闭包。

---

## 8. 状态键与去重：为什么不直接用字符串

`_state_key`（state.py:111-139）是整个搜索的性能关键点。它决定**什么样的两个状态被认为"等价"可以去重**。

### 8.1 键的构成

```python
return (
    tuple((track, tuple(seq)) for track, seq in sorted(state.track_sequences.items()) if seq),
    tuple(sorted(state.weighed_vehicle_nos)),
    spot_items,  # 见下
)
```

三部分：

1. 非空股道的车辆序列（排序后 tuple，可 hash）
2. 已称重集合（sorted tuple）
3. 台位分配（sorted tuple of pairs）

### 8.2 大库随机台位的 canonicalization

这一段特别关键（state.py:56-63 + 123-130）：

```python
def _canonical_random_depot_vehicle_nos(plan_input):
    if 有任何 SPOT 目标的车: return frozenset()  # 整体关闭 canonical
    return {vehicle_no for vehicle if target_area_code == "大库:RANDOM"}

# 在构造 state_key 时：
spot_items = tuple(
    (v_no, spot_code) for v_no, spot_code in sorted(state.spot_assignments.items())
    if not (v_no in canonical_random_depot_vehicle_nos and spot_code.isdigit())
)
```

**什么意思？** 如果一辆车的目标是 `大库:RANDOM`（不要求具体台位），那它被分配到 `101` 还是 `102` 在搜索上**不重要**——只要能放下就行。所以 state_key 里干脆不记具体台位数字。

**为什么要加 "if 有 SPOT 目标就关闭"？** 因为如果场景里有精确 SPOT 要求（比如 `304`），而"大库随机"车辆刚好占了 `304`，那台位差异会直接决定能不能完成目标，必须严格去重。

### 8.3 `best_cost` 的作用（search.py:86）

```python
best_cost = {initial_key: 0}  # state_key → 到达它的最小 cost
```

两个地方用：

- Line 102：`if best_cost.get(current.state_key) != current_cost: continue`——弹出来的节点如果已经被其他路径更短地到达过，直接跳过
- Line 161-163：新节点入队前检查`if state_key in best_cost and best_cost[state_key] <= cost: continue`

这是 Dijkstra 风格的去重。没有它的话，同一个 state_key 可能被几十倍重复展开。

---

## 9. 构造保底：永远返回一条计划的兜底

`constructive.solve_constructive`（constructive.py:51-177）是一个贪心分派器，它的核心承诺是：

> **只要 `generate_goal_moves` 在任何非目标态下能产出至少一个合法动作，构造保底就必能终止于一条合法 plan（即使不最优）。**

这就是 SLA 契约的根基。

### 9.1 主循环骨架

```python
for iteration in range(max_iterations):
    if _is_goal(state): return plan  # 达标
    if budget_exhausted: return 部分 plan
    moves = generate_goal_moves(state)
    if not moves: return 部分 plan  # 死锁
    best_move, tier = _choose_best_move(moves, state, ...)
    state = _apply_move(state, best_move)
    plan.append(best_move)
    # 启发值停滞检测
    if stale_rounds >= 6: return 部分 plan
```

### 9.2 七层分派优先级（`_score_move`, constructive.py:266-336）

```
tier 0: is_close_door_final    → 关门车进 存4北 且前进 (最紧急)
tier 1: is_weigh_to_jiku       → 带未称车进机库
tier 2: delta > 0              → 至少有一辆车从"错位"→"到位"
tier 3: clears_blocker         → delta==0 但清掉了目标股道上的挡路者
tier 4: delta == 0 非临停       → 横向移动（很少出现，次选）
tier 5: delta == 0 临停         → 临停清障（末选）
tier 6: delta < 0              → 回退（强烈不推荐）
```

同 tier 内再按 `(-delta, -block_size, path_length, src, tgt)` 排序——**同等优先级下，偏好一次搬多、路径短的动作**。

### 9.3 循环回退保护

`_is_inverse_of_recent`（constructive.py:249）：检查当前动作是不是刚才某钩的逆向（`A→B` 紧跟 `B→A` 且车辆相同）。**非逆向动作存在时，优先选非逆向**。窗口 12 钩。

否则贪心很容易陷入 `A→临1 → 临1→A` 的死循环。

### 9.4 `ConstructiveResult` 的三种退出

| 退出条件 | `reached_goal` | `stuck_reason` |
|---|---|---|
| `_is_goal` 满足 | True | None |
| 超时 | False | `"time budget exhausted"` |
| 无合法动作 | False | `"no legal moves"` |
| 启发值停滞 6 轮 | False | `"heuristic stale for 6 rounds"` |
| 达到最大迭代（1500） | False | `"max iterations reached"` |

后四种都会返回**部分 plan**，上层包装成 `fallback_stage="constructive_partial"` 返回。

---

## 10. Anytime Fallback Chain：七阶段递进

`anytime._run_anytime_fallback_chain`（anytime.py:22-135）在 exact 搜索没证明最优时逐阶段尝试不同策略。

### 10.1 七个阶段（按顺序）

| # | stage_name | solver_mode | h_weight | beam_width | 预算占比 |
|---|---|---|---|---|---|
| 1 | `weighted` | weighted | `max(w, 1.5)` | - | 8% |
| 2 | `beam` | beam | `max(w, 1.5)` | `beam_width or 64` | 8% |
| 3 | `beam_greedy_64` | beam | 5.0 | 64 | 8% |
| 4 | `weighted_greedy` | weighted | 5.0 | - | 12% |
| 5 | `beam_greedy_128` | beam | 5.0 | 128 | 8% |
| 6 | `beam_greedy_256` | beam | 5.0 | 256 | 8% |
| 7 | `weighted_very_greedy` | weighted | 10.0 | - | 8% |

加上 exact 本身的 40%，总计不超过 100%。

### 10.2 阶段间的 incumbent 替换规则

```python
for stage in fallback_stages:
    if current.plan:  # 已有解就不继续
        break
    candidate = solve_search_result(..., stage_kwargs)
    if not candidate.plan: continue
    if not current.plan or len(candidate.plan) < len(current.plan):
        current = replace(candidate, fallback_stage=stage_name)
```

**关键**：第一次拿到非空 plan 就跳出循环。每阶段只在"更短"时替换 incumbent（`len < current.plan`），所以**更多时间只会让 plan 更短，不会变长**（anytime 单调性）。

### 10.3 每阶段的时间预算

```python
share_ms = max(1.0, time_budget_ms * budget_share)  # 绝对上限
stage_time_budget_ms = min(remaining_ms, share_ms)  # 但不超过剩余预算
```

即使前面阶段超时了，后面阶段拿到的预算也会被压缩到"还剩多少就用多少"——整个 chain 严格尊重 `time_budget_ms`。

### 10.4 设计直觉：为什么这样排？

- `weighted` → `beam`：先试不完全 A*，再放宽剪枝
- `*_greedy_64` → `*_128` → `*_256`：beam 越宽越靠近完整搜索，但越慢
- `weighted_very_greedy` (w=10)：几乎成为"贪心最好优先"，用来在极端复杂场景下找**任何**可行解

线上分布（architecture.md）：80% 场景在 `beam` 阶段命中，13% 在 `weighted`，3% 直接 exact，剩下的在深度回退。

---

## 11. LNS：先求后磨的局部修复

`lns._solve_with_lns_result`（lns.py:18-59）和 `lns._improve_incumbent_result`（lns.py:62-127）。

### 11.1 LNS 模式完整流程

```python
def _solve_with_lns_result(...):
    # 1. 求一个 seed（有 beam_width 用 beam，否则 weighted）
    incumbent = _solve_search_result(solver_mode="beam" or "weighted", h_weight=max(w, 1.5))
    # 2. 迭代改良
    improved = _improve_incumbent_result(
        incumbent, repair_passes=4, max_rounds=None  # 无限轮直到不再改善
    )
    return improved
```

### 11.2 改良主循环

```python
while 有轮数预算:
    snapshots = replay_plan(incumbent_plan)  # 得到每一钩后的快照
    for cut_index in _candidate_repair_cut_points(incumbent_plan, 4):
        prefix = incumbent_plan[:cut_index]
        start_state = snapshots[cut_index]
        # 构造修补问题：把已满足车辆冻结成"保持原位"目标
        repair_input = _build_repair_plan_input(plan_input, start_state)
        repaired = _solve_search_result(repair_input, start_state, ...)
        candidate_plan = prefix + repaired.plan
        if _is_better_plan(candidate_plan, incumbent_plan):
            incumbent_plan = candidate_plan; improved=True; break
    if not improved: break
```

### 11.3 四个 destroy 算子（`_candidate_repair_cut_points`, lns.py:130-167）

LNS 的"去哪里下刀"本身就是一门学问。当前用四个算子 round-robin：

| 算子 | 逻辑 | 适用场景 |
|---|---|---|
| `_cut_points_hotspot` | 按 source/target 股道被触碰次数最多排序 | 现场有反复腾挪的热点 |
| `_cut_points_worst_cost` | 路径最长 + 车辆最多的钩 | 有一个"显得特别笨"的钩 |
| `_cut_points_target_cluster` | 连续多钩送同一目标的起点 | 能合并的邻接动作 |
| `_cut_points_equidistant` | 均匀切片 | 无特征时的盲打 |

每个算子产出 `repair_passes // 4 = 1` 个切点，最终合并去重，上限 `repair_passes` 个。

`_repair_cut_score`（lns.py:210）的 key：

```python
return (
    -repeated_touch_flag,  # 重复触碰过的切点优先
    -staging_flag,          # 涉及临停/存4南的切点优先
    -hotspot_score,         # 触碰次数越多越优先
    index,                  # 打平用 plan 顺序
)
```

### 11.4 冻结已满足车辆：`_build_repair_plan_input`

```python
for vehicle in plan_input.vehicles:
    current_track = _locate_vehicle(snapshot, vehicle_no)
    if 车已满足目标（_is_goal 的三条子条件）:
        frozen_goal = GoalSpec(
            target_mode="TRACK",
            target_track=current_track,
            allowed_target_tracks=[current_track],  ← 只允许保持原位
            ...
        )
    ...
```

这样后缀求解时，搜索器不会去动那些已经到位的车，只关心剩下的"未满足"车辆。**搜索空间被极大收窄**，这正是 LNS 能显著加速的原因。

### 11.5 `_plan_quality`：三级比较

```python
def _plan_quality(plan, route_oracle):
    return (
        len(plan),         # 1. 钩数
        total_branch_count,# 2. 分支总数
        total_length_m,    # 3. 路径总长度
    )
```

字典序比较。**钩数永远是主目标**，平局才看分支、再平才看长度。

---

## 12. 终止判定：什么叫"解完了"

`_is_goal`（state.py:11-38）是搜索能否停止的判官。它的判定远比"每辆车都到目标股道"复杂：

```python
for vehicle in plan_input.vehicles:
    # 1. 必须在允许的最终股道里
    if current_track not in vehicle.goal.allowed_target_tracks: return False
    # 2. 需称重的必须已称
    if vehicle.need_weigh and vehicle_no not in weighed_vehicle_nos: return False
    # 3. SPOT 模式：精确台位必须匹配
    if goal.target_mode == "SPOT":
        if spot_assignments[v_no] != target_spot_code: return False
    # 4. 大库随机：必须有兼容的台位分配
    if goal.target_area_code == "大库:RANDOM":
        if assigned_spot is None or not in spot_candidates_for_vehicle(...): return False
    # 5. 作业区：必须有兼容的工位分配
    if goal.target_area_code in {"调棚:WORK", ...}:
        if assigned_spot is None or not in spot_candidates_for_vehicle(...): return False
    # 6. 关门车：不能在存4北的前3位
    if vehicle.is_close_door and current_track == "存4北":
        if 存4北序列.index(v_no) < 3: return False
return True
```

六条全部通过才算达标。注意：**第 4/5 条比"到股道"严格得多**——光到大库线上不算，必须已经分到了台位（且台位兼容车型与 yard mode）。

---

## 13. Verify：独立口径的最终复核

`verify.plan_verifier.verify_plan`（plan_verifier.py:35-206）是整个系统的"质检员"。它**不依赖搜索器的任何内部状态**，从 `initial_state` 和 `hook_plan` 重新跑一遍完整 replay，然后独立判定。

### 13.1 两层检查

**全局检查**（对最终状态）：

- 每辆车在允许目标股道里
- 需称重的已称
- SPOT 目标匹配
- RANDOM/WORK_AREA 有合法台位
- 关门车不在 `存4北` 前 3 位
- 最终股道不是运行线
- 最终股道不容量溢出

**逐钩检查**（对每一钩）：

- `pathTracks` 存在且首尾匹配源/目标
- 车辆编号在已知集合里
- 单钩车辆组合合法（复用 `validate_hook_vehicle_group`）
- `route_oracle.validate_path`：路径拓扑 / 中间净空 / L1 / reverse clearance
- 关门车大钩规则（单钩 > 10 辆且目标非存4北时，首位不能是关门车）

### 13.2 为什么独立一层很重要

**搜索器知道的 ≠ 最终 plan 满足的**。举几个实际情况：

- 搜索器可能用错了 `state.track_sequences`（比如 workflow 阶段继承漏带字段）
- `HookAction.path_tracks` 和 route_oracle 的当前结果可能版本不一致
- 手动修改过的 plan 从外部加载进来（demo 里支持）

有独立 verifier 就能抓住**任何**这种错位。搜索代码改了 100 次，只要 verifier 通过就一定没违规。

### 13.3 `constructive_partial` 的特殊处理

```python
# astar_solver.py:272
best_effort = result.fallback_stage == "constructive_partial"
if not report.is_valid and not best_effort:
    raise PlanVerificationError(report)
return replace(result, verification_report=report)
```

部分 plan 即使验证失败也**不 raise**，而是附 report 返回。调用方自己决定要不要接受。这是 SLA 兜底的一部分——"宁可给你一个能看的半成品，也不抛异常"。

---

## 14. 端到端示例：跟踪一次完整求解

假设一个极小场景：

- 股道：`存2`、`临1`、`机库`、`修3库内`
- 车辆：`存2` 上依次是 `[A, B, C]`
- 目标：
  - `A` → 留在 `存2`（`target_mode="TRACK"`，`allowed=["存2"]`）
  - `B` → `修3库内` + 必须先称重（`need_weigh=True`，`target_area_code="大库:RANDOM"`）
  - `C` → `修3库内`

现在一步步看求解器干了什么。

### 14.1 Step 0 预检

`_validate_final_track_goal_capacities`：`存2` 容量够放 `A`、`修3库内` 容量够放 `B + C`。✅ 继续。

### 14.2 Step 1 构造保底（提前跑）

`constructive.solve_constructive` 尝试贪心：

```
迭代 1:
  state = {存2:[A,B,C], 临1:[], 机库:[], 修3库内:[]}
  _is_goal? → False (B 需要称重)
  moves = generate_goal_moves(...):
    从 存2 出发：
      prefix [A,B,C] 目标不一致 → 跳过直达
      prefix [A,B] 目标不一致 → 跳过
      prefix [A] 目标一致（→存2），但源=目标 → 跳过
      staging: 检测到 A 已在合法位置但挡住 B
        → 生成 [A]→临1
  打分：tier 3 (clears_blocker with delta=0) → 选它
  应用 → state = {存2:[B,C], 临1:[A], ...}

迭代 2:
  moves:
    从 存2: prefix [B,C] 目标不一致 → 跳过
            prefix [B] 需要称重 → candidate_targets=["机库"] → [B]→机库
            prefix [B,C] staging 无必要
  打分：tier 1 (weigh_to_jiku)
  应用 → state = {存2:[C], 临1:[A], 机库:[B], weighed:{B}}

迭代 3:
  moves:
    从 存2: [C]→修3库内
    从 机库: [B]→修3库内 (已称过重了)
    从 临1: [A]→存2
  打分：C→修3库内 delta=+1, B→修3库内 delta=+1
        都是 tier 2
        按 (-delta, -block_size, path_length) 打平后选一个
  假设选 [B]→修3库内 → state = {存2:[C], 临1:[A], 修3库内:[B]}

迭代 4:
  [C]→修3库内, tier 2 → 选
  state = {存2:[], 临1:[A], 修3库内:[B,C]}

迭代 5:
  [A]→存2 tier 2 → 选
  state = {存2:[A], 临1:[], 修3库内:[B,C]}

迭代 6:
  _is_goal? → True (全部满足)
  返回 plan 长度 5
```

构造保底得到：`[A→临1, B→机库, B→修3库内, C→修3库内, A→存2]`，长度 5。

### 14.3 Step 2 exact 搜索

主循环从初始状态开始。第一次展开会生成和迭代 1 相同的候选动作，但**所有动作都进队列**，按 f 值排序。

初始 f 值：
- `h_distinct_transfer_pairs` = 2 个强制对 (存2→机库 对 B, 存2→修3库内 对 C)
- `h_weigh` = 1
- `h_blocking` = 1
- `h = max(2, 1, 1, 0) = 2`
- `f = 0 + 2 = 2`

`[A]→临1` 动作：
- new state 里 B 到达存2 前端，但 B 还需要称重
- `h = max(2, 1, ...) = 2`
- `f = 1 + 2 = 3`
- `blocker_bonus = 1`（整块移动 + 目标是挡路清障）
- 优先级 = `(3, 1, 2, -1, 0)`

`[A,B]→?` 不生成（目标不一致）。

exact 会展开整棵树，直到找到长度最短的解。对这种小场景，通常会得到**长度 5 的最优解**（和构造保底一样）——但有可能更短的分支，比如如果 exact 发现 `[A,B,C]→?` 不存在，而 `[A]→临1 → [B,C]→?` 目标不一致，就会回退到和构造一样的路径。

### 14.4 Step 3 如果 exact 跑到头

`is_proven_optimal = True`，直接返回，不走 anytime chain。

### 14.5 Step 6 verify

重演整条 plan。每钩检查 `pathTracks`、`validate_hook_vehicle_group`、`route_oracle`。最终 state 检查每辆车在目标股道 + 台位合法。通过后附 `verification_report` 返回。

---

## 15. Debug 统计：怎么读懂搜索在干什么

`debug_stats` 是一个可选的 dict，传进去后搜索会把遥测数据写进来。生产上可以不传，调试时非常有用。

### 15.1 核心字段

| 字段 | 含义 | 看到异常怎么办 |
|---|---|---|
| `expanded_states` | 实际被展开的状态数 | 过大（>100k）说明搜索树过宽 |
| `generated_nodes` | 入队节点总数 | 通常 = expanded + 被裁剪/被更优覆盖 |
| `closed_states` | 已完全探索的状态键数 | 接近 expanded 说明没什么重复 |
| `candidate_moves_total` | 所有状态累计的候选动作数 | 除以 expanded 得到平均分叉数 |
| `candidate_direct_moves` / `candidate_staging_moves` | 直接 / 临停动作的计数 | staging >> direct 说明现场很堵 |
| `states_with_zero_moves` | 死锁状态数 | >0 需要检查业务约束是否过紧 |
| `moves_by_target` | 每个目标股道被选为候选的次数 | 找出热点 |
| `moves_by_source` | 每个源股道的次数 | 和 target 对照看腾挪流向 |
| `moves_by_block_size` | 按 block 大小统计 | 偏向 1 说明动作很碎 |
| `top_expansions` | 分叉最多的前 10 个展开点 | 手动跟这些点复现问题 |

### 15.2 构造保底特有统计

`constructive.solve_constructive` 还会写 `tier0_close_door_final` ~ `tier5_staging` 的计数，表明各层分派规则被触发的频率。如果 `tier5_staging` 占比过高，说明整个场景靠临停推进——可能暗示路径可达性有问题。

---

## 16. 源码阅读路线与模块依赖

### 16.1 自下而上的阅读顺序

```
第 1 步：input 和 state 的数据模型（先让自己知道在操作什么）
    io/normalize_input.py              ← NormalizedPlanInput, GoalSpec, NormalizedVehicle
    verify/replay.py                    ← ReplayState, build_initial_state

第 2 步：业务约束（搜索代码的"硬件规范"）
    domain/hook_constraints.py          ← 单钩车辆组合规则
    domain/depot_spots.py               ← 台位分配
    domain/route_oracle.py              ← 路径拓扑 + L1 + reverse clearance

第 3 步：动作生成（搜索的燃料）
    solver/move_generator.py            ← 最厚的文件，但把前面的点串起来

第 4 步：状态演进与判定
    solver/state.py                     ← _apply_move, _is_goal, _state_key
    solver/heuristic.py                 ← 5 个可采纳下界

第 5 步：搜索核心
    solver/budget.py                    ← SearchBudget
    solver/search.py                    ← 主循环 + 优先队列 + beam 剪枝
    solver/result.py                    ← SolverResult

第 6 步：上层调度
    solver/constructive.py              ← 7 tier 贪心保底
    solver/anytime.py                   ← 7 阶段回退链
    solver/lns.py                       ← 4 个 destroy 算子
    solver/astar_solver.py              ← 总入口

第 7 步：独立复验
    verify/plan_verifier.py
```

### 16.2 模块依赖示意

```
                   astar_solver (entry)
                 /         |          \
           constructive  anytime      lns
                 \         |          /
                  \        v         /
                   → search._solve_search_result
                         |
                   ┌─────┼──────┬───────────────┐
                   v     v      v               v
              state.py heuristic.py  move_generator.py  budget.py
                   |                      |
                   v                      v
                  replay.py      domain/{hook_constraints,
                                         depot_spots,
                                         route_oracle}
                                         |
                                         v
                                 domain/master_data.py
```

**自下而上无环**。任何一层都只依赖比它低的层，所以改动的影响范围清晰。

### 16.3 测试入口

对应的测试文件在 `tests/solver/` 下。想验证你对某层的理解，随时可以跑对应测试：

```bash
. .venv/bin/activate
pytest tests/solver/ -v
```

### 16.4 常见下手切入点

| 你想做什么 | 先读哪里 |
|---|---|
| 加一条业务规则 | `move_generator._build_candidate_move` + `verify/plan_verifier.py` |
| 换启发式 | `solver/heuristic.py` 添加 `h_*` 函数，纳入 `max` |
| 加新 destroy 算子 | `solver/lns.py` 的 `_cut_points_*`，补进 `_candidate_repair_cut_points` |
| 加 fallback 阶段 | `solver/anytime.py` 的 `fallback_stages` 列表 |
| 改主数据/拓扑 | `data/master/*.json` + `domain/master_data.py` |
| 加状态字段 | `verify/replay.py::ReplayState` + `solver/state.py::_state_key` + `_apply_move` |

### 16.5 一条底线

**任何时候对搜索做修改，必须重新跑 benchmark**：

```bash
python scripts/run_external_validation_parallel.py --solver exact --max-workers 8 --timeout-seconds 180
```

当前基线见下节 §17。任何改动让这个基线退化就需要讨论。

---

## 17. 当前 benchmark 基线

最新跑完的是 `exact` 模式、180s 超时（单条求解预算 175s）、8 worker 并发。口径见
`artifacts/external_validation_parallel_runs/summary.json`。

### 17.1 `external_validation_inputs` 109 条

| 分类 | 数量 | 说明 |
|---|---:|---|
| 总数 | 109 | 真实排班案例 |
| **预检拒绝** | **10** | `_validate_final_track_goal_capacities` 判定目标落位超轨道物理容量——真物理不可解，任何算法都解不了 |
| **返回 plan** | **99** | 所有容量可行的案例都产出了 plan |
| &nbsp;&nbsp;↳ verifier 通过（`is_valid=True`） | **97** | plan 完全合法，可直接进生产 |
| &nbsp;&nbsp;↳ verifier 未通过（`is_valid=False`） | **2** | 仅构造保底的 partial plan，不满足全部车辆终点约束 |
| 空返 / ValueError | **0** | SLA 保证生效 |

**Verifier 失败的 2 条**（都落在 `fallback_stage=constructive_partial`）：

- `validation_20260105W.json`：38 hooks
- `validation_20260310W.json`：22 hooks

这两条 plan 技术上"可执行"，但终态没让每辆车都抵达其 `allowed_target_tracks`。业务方按 partial 接还是拒，由上游决定——求解器已经把 `verification_report` 透传出来。

### 17.2 勾数分布（97 个 valid plan）

```
总勾数: 2732        avg: 28.2        min: 4        max: 284

勾数桶:
  00-10   1 (1.0%)
  11-20  26 (26.8%)
  21-30  45 (46.4%)  ← 绝大部分案例
  31-40  20 (20.6%)
  41-50   4 (4.1%)
  >100    1 (1.0%)   ← 离群点：validation_20260205W (89 车)

百分位: p50=25   p90=37   max=284
```

**离群点 205W**：89 辆车、20 条 `大库:RANDOM` 目标的高负载案例，当前 214-284 勾区间。已尝试过两种压缩方案（归一化前预分配、move_generator top-1 候选）都破坏了搜索的路径弹性，实验后已回滚。更激进的 CP-SAT 子问题分解或 landmark 启发式属于下一 milestone。

### 17.3 阶段命中分布（97 个 valid plan）

| 阶段 | 次数 | 含义 |
|---|---:|---|
| `beam` | 81 | 第二回退阶段，最常命中 |
| `weighted` | 13 | 第一回退阶段 |
| `exact` | 1 | 小规模案例直接最优 |
| `weighted_greedy` | 1 | 更深回退 |
| `constructive` | 1 | 所有搜索阶段都空返，纯构造保底救场 |

**注意**：2 个 invalid 的 `constructive_partial` 不在此表——它们走到构造保底但未达成完整 goal。

### 17.4 耗时（97 个 valid plan）

```
p50  2806 ms
p90 14004 ms
max 39706 ms
```

p90 恰好等于 14s 的 beam 阶段默认预算，说明大部分复杂案例是在 beam 阶段收敛的。

### 17.5 一句话总结

- **SLA 方面**：99/99 容量可行案例都返回 plan；真正不可解的 10 条在预检层被诚实拒绝
- **质量方面**：97/99 plan 独立 verifier 通过；2 条 invalid 来自单一的构造保底 partial，不是搜索 bug
- **压缩方面**：97% 案例 ≤50 勾，1 条离群（205W）留在 214-284 勾，需更重的结构性改动才能压下去

---

## 最后一句话

如果只记一句话：

> **求解器 = `动作合法性检查` + `优先级搜索` + `独立复核`。业务规则集中在动作生成器里；搜索只关心组合策略；verify 负责独立质检。三层解耦是整个系统可维护的根基。**

读到这里你应该已经能：

1. 看着 `fallback_stage` 倒推搜索在哪一层命中
2. 看着 `debug_stats` 定位哪个状态分叉爆炸
3. 在任何一层改代码时，知道影响会传导到哪些相邻模块

欢迎接手。

---

**文档归属**：本文档由项目主数据与求解器源码共同驱动；代码变更时应同步更新本文引用的行号与函数名。建议每次较大改动后 diff 一下本文 `§5` / `§6` / `§10` / `§11` 四节的描述是否还对。
