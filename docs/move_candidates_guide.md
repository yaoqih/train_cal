# move_candidates.py 模块详解

> 面向初学者的通俗指南。本文档对应文件：  
> `src/fzed_shunting/solver/move_candidates.py`

---

## 背景：这个模块解决什么问题？

调车场里有很多股道（track），每条股道上停着若干节车辆。  
调车机车（loco）每次可以做一个"钩挂动作"——  
把一组车辆从 A 股道推到 B 股道（ATTACH + DETACH，即一钩一放）。

这个模块的任务就是：**在每一步棋之前，生成一份"候选动作列表"**，  
让搜索算法从中挑一个最有希望的动作执行。

候选动作分两类：

| 类型 | 说明 |
|------|------|
| `primitive`（原始候选） | 单步合法钩挂动作，由底层 `generate_real_hook_moves()` 暴力枚举 |
| `structural`（结构候选） | 经过高层分析生成的多步计划，解决"欠债"问题 |

---

## 整体流水线（6 个阶段）

```
generate_real_hook_moves()         ← 第 1 阶段：枚举所有合法单步
        ↓
4 道过滤器（拦截坏动作）            ← 第 2 阶段：过滤原始候选
        ↓
build_structural_intent()          ← 第 3 阶段：分析局面，找出"欠债"
summarize_debt_chains()
        ↓
_generate_structural_candidates()  ← 第 4 阶段：根据欠债生成结构候选
        ↓
_trim_candidate_before_delayed_commitment()  ← 第 5 阶段：截断时机未到的步骤
        ↓
_dedup_candidates()                ← 第 6 阶段：去重，排序，输出最终列表
```

---

## 第 1 阶段：枚举原始动作

```python
primitive_moves = generate_real_hook_moves(plan_input, state, ...)
```

**做什么**：列出当前状态下所有物理上可行的单步钩挂动作。  
每个动作是一个 `HookAction`，包含：
- `action_type`：`"ATTACH"`（挂上）或 `"DETACH"`（放下）
- `source_track`：车辆当前在哪条股道
- `target_track`：要推去哪条股道
- `vehicle_nos`：这次移动的车辆编号列表

> 把它想象成：枚举棋盘上所有在规则内的走法，不管好坏。

---

## 第 2 阶段：4 道过滤器

原始动作太多，大多数都是无意义甚至有害的。  
以下 4 个过滤函数依次判断，任意一个返回 `True` 就丢弃该动作。

---

### 过滤器 ①：`_move_breaks_protected_commitment`
**不能拆散已保护的"承诺块"**

**什么是承诺块（CommittedBlock）？**  
当一组车辆已经按正确顺序停在正确股道上，系统会把它们标记为"已满足承诺"。  
这个过滤器保证不会把这些已经就位的车辆移走，打乱好不容易排好的序列。

**逻辑**：
- 如果是 ATTACH（把车从 source_track 拉走）：检查 source_track 上是否有保护块，且被移动的车辆与之重叠
- 如果是 DETACH（把车放到 target_track）：检查 target_track 是否是工位股道且已有保护块

**例外**：若动作被 `_move_is_required_by_resource_debt` 识别为清障必须（下文详述），则允许通过。

---

### 过滤器 ②：`_move_mixes_resource_debt_groups`
**不能把目标不同的车辆混在一次 ATTACH 中**

**背景**：某条股道上有一批"欠债车辆"（ResourceDebt），  
这些车辆里可能有的要去 A 站，有的要去 B 站。  
如果一次 ATTACH 把它们全钩上，就会把目标不同的车辆混在一起，  
后续无法分离到各自的目的地。

**逻辑**：检查被移动的多节车辆是否都属于同一个"目标组"（即同一个目的地股道）。  
如果有多个目标组，就拒绝此动作。

---

### 过滤器 ③：`_move_is_unowned_staging_churn`
**不能把"延迟承诺"车辆在临时股道之间乱搬**

**背景**：有些车辆处于"延迟承诺"状态（DelayedCommitment），  
意思是它们的目的地工位窗口还没开放，现在不该被送走，只能暂停在临时股道（staging track）等待。

**判定条件**（同时满足）：
1. 动作是 DETACH
2. 机车当前在临时股道，目标也是临时股道
3. 被移动的车辆中有"延迟承诺"车辆，或这些车辆当前位置本身已经满足目标（不应该再动）

---

### 过滤器 ④：`_move_attaches_unowned_delayed_staging_buffer`
**不能取出为"延迟承诺"车辆预留的缓冲存车**

**背景**：某些临时股道是专门留给"延迟承诺"车辆的缓冲区。  
如果把这些车辆钩走，缓冲区就被破坏了。

**判定**：动作是 ATTACH，源头是临时股道，且被钩走的车辆中有"延迟承诺"车辆。

---

### 例外：`_move_is_required_by_resource_debt`

有时候，即使动作触犯了过滤器①，也必须强行通过，  
因为它是解决"资源欠债"（ResourceDebt）的必要步骤。  
比如：必须先清走某股道前端的阻塞车辆，后面的车辆才能被取出。

**判定**：被移动的车辆中，有任何一节出现在某条 ResourceDebt 的 `vehicle_nos` 列表里。

---

## 第 3 阶段：分析局面

```python
intent = build_structural_intent(plan_input, state, ...)
debt_chain_summary = summarize_debt_chains(plan_input, state, intent=intent)
```

**做什么**：对当前局面做一次"体检"，找出所有"欠债"（Debt）。

### 两种欠债

| 欠债类型 | 含义 |
|----------|------|
| **OrderDebt（排序欠债）** | 某个工作位股道上，车辆的排列顺序不对，正确的车还没到位 |
| **ResourceDebt（资源欠债）** | 某条股道被不属于这里的车辆占用，导致其他车辆无法通行 |

### DebtCluster（欠债簇）

每条有问题的股道对应一个 `DebtCluster`，包含该股道的 OrderDebt 和所有 ResourceDebt，  
以及一个压力值（`pressure`），压力越大说明问题越紧迫。

### DebtChain（欠债链）

多个欠债簇之间可能相互依赖（A 的问题导致 B 也有问题），  
`summarize_debt_chains` 把它们串成"债务链"，方便统一处理。

---

## 第 4 阶段：生成结构候选

```python
structural_candidates = _generate_structural_candidates(...)
```

### 遍历顺序

按"压力从大到小"遍历每个 `DebtCluster`：

```
for cluster in _ordered_debt_clusters(intent):   # 按 pressure 降序
    if cluster.order_debt:                        # 先处理排序欠债
        ...
    for debt in cluster.resource_debts:           # 再处理资源欠债
        ...
```

---

### 针对 OrderDebt 的 3 种候选构建器

#### `_build_work_position_source_opening_candidate`
**"清路"候选：先把挡路的车移走，让目标车能进入工位**

- 找到那些"已承诺要去某工位"的车辆
- 找出拦在它们前面的障碍车辆
- 把障碍车辆移到其他股道，腾出通路

---

#### `_build_work_position_window_candidate`
**"直送"候选：把已承诺车辆直接送入工位窗口**

- 验证承诺车辆形成连续块
- 如果目标工位股道前端有障碍，先清障
- 把承诺车辆推进工位

`front_block_only=True` 变体：只处理最前端的一节承诺车辆，  
适用于工位空间有限、只能先送一节的情况。

---

#### `_build_work_position_free_fill_candidate`
**"填充"候选：用"自由工位"车辆把工位填满**

- 针对不限制精确位置的"自由工位"类型
- 从多条来源股道逐批移入目标工位
- 累积多步动作，尽量填满空位

---

### 针对 ResourceDebt 的 3 种候选构建器

#### `_build_route_release_frontier_candidate`（ROUTE_RELEASE 类型）
**"路线释放"候选：清障＋顺势推进**

这是最复杂的候选构建器：
1. 找到"前沿块"：在来源股道上，已经准备好移动到目标的车辆
2. 清走阻塞它们的障碍车辆
3. 顺势把前沿块逐步推进（最多推 3 次）
4. 如果障碍车辆原本就满足目标，事后把它们送回原位

---

#### `_build_goal_frontier_source_opening_candidate`（FRONT_CLEARANCE 类型）
**"前端开路"候选：把满足目标的前缀车辆送走，露出被堵的车**

- 源股道前端有一些已满足目标的车辆，但后面的车被它们堵住了
- 把这些"已就位但占着位置"的车辆移走，让后面的车能动

---

#### `_build_resource_release_candidate`（其余类型）
**"资源释放"通用候选**

处理 CAPACITY_RELEASE、FRONT_CLEARANCE、EXACT_SPOT_RELEASE 等欠债：
1. 先尝试直接把欠债车辆送到目标位置
2. 如果有"保护前缀"（前端已满足目标的车辆），则临时搬走前缀、释放后端、再把前缀送回

---

### `_generate_chain_macro_candidates`（宏候选）

**适用场景**：债务链有 3 条以上股道，且总压力 ≥ 25

**做什么**：把多个单步结构候选"串联"成一个长动作序列（最多 3~4 段），  
一次提交多步棋，让搜索算法能够"多想一步"。

**过程**：
1. 从基础候选中挑一个"种子"（通常是针对锚点股道的候选）
2. 执行种子后，重新分析局面
3. 找最佳的"后续候选"并拼接
4. 重复，直到达到步数上限或找不到合适的后续

---

### `_select_structural_candidates`（最多保留 6 个）

候选太多搜索会爆炸，这个函数把结构候选精简到至多 6 个。  
选择策略（优先级从高到低）：
1. 每条债务链，每个"槽位"（sequence / anchor_release / non_anchor_release）各保留最优候选
2. 不同 `focus_tracks` 各保留一个
3. 不同 `(reason, focus_tracks)` 组合各保留一个
4. 剩余配额从候选列表顺序填满

---

## 第 5 阶段：截断时机未到的步骤

```python
_trim_candidate_before_delayed_commitment(candidate, ...)
```

**问题**：某个多步候选里，最后几步是把"延迟承诺"车辆送入工位，  
但工位窗口还没开放，不能提前送进去。

**解决**：把候选步骤截到延迟承诺车辆的那一步之前，  
把"我该做的前期准备"保留下来，暂时不送车进工位。

**特殊情况**：如果候选本身就是 `work_position_` 类型，且车辆是严格工位（SPOTTING / EXACT_NORTH_RANK / EXACT_WORK_SLOT），则允许保留完整步骤（时机其实已到）。

---

## 第 6 阶段：去重与排序

### `_dedup_candidates`

两个候选若步骤完全相同（action_type + source_track + target_track + vehicle_nos + path_tracks 都一样），则视为重复，只保留第一个。

### `_candidate_search_sort_key`

排序优先级（从高到低）：

| 优先级 | 规则 |
|--------|------|
| 1 | `structural` 类型排在 `primitive` 前面 |
| 2 | `work_position_*` 原因 > `chain_macro_*` > 其他 |
| 3 | 步数少的排前面 |
| 4 | `focus_tracks` 字典序 |
| 5 | `structural_reserve=True` 排后（留作备选） |
| 6 | `reason` 字符串 |
| 7 | 步骤内容签名 |

最终输出：**结构候选在前，过滤后的原始候选在后**，整体按上述排序。

---

## 核心数据结构速查

### `MoveCandidate`

```python
@dataclass(frozen=True)
class MoveCandidate:
    steps: tuple[HookAction, ...]  # 由 1 或多个钩挂动作组成的计划
    kind: str = "primitive"        # "primitive" 或 "structural"
    reason: str = ""               # 候选生成原因，如 "work_position_window"
    focus_tracks: tuple[str, ...]  # 这次候选主要涉及的股道
    structural_reserve: bool = False  # 是否属于备用结构候选
```

### `HookAction`

```python
action_type: "ATTACH" | "DETACH"
source_track: str   # 车辆从哪条股道出发
target_track: str   # 推送到哪条股道
vehicle_nos: list[str]  # 被移动的车辆编号
```

---

## 一图总览

```
┌─────────────────────────────────────────────────────┐
│              generate_move_candidates()              │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ generate_real_hook  │  枚举所有合法单步钩挂
│ _moves()            │
└─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  过滤器（4 道）                                      │
│  ① _move_breaks_protected_commitment（不拆承诺块）  │
│  ② _move_mixes_resource_debt_groups（不混目标组）   │
│  ③ _move_is_unowned_staging_churn（不乱搬延迟车）  │
│  ④ _move_attaches_unowned_delayed_staging_buffer   │
│                          例外：ResourceDebt 必须动  │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  build_structural_intent() + summarize_debt_chains()│
│  → 找出 OrderDebt / ResourceDebt / DebtChain        │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  _generate_structural_candidates()                   │
│                                                      │
│  OrderDebt → source_opening / window / free_fill    │
│  ResourceDebt → route_release / goal_frontier /     │
│                 resource_release                     │
│  + chain_macro（宏候选）                             │
│  → _select_structural_candidates()（最多 6 个）     │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  _trim_candidate_before_delayed_commitment()         │
│  截断时机未到的"送车进工位"步骤                       │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  _dedup_candidates() + _candidate_search_sort_key() │
│  去重 → 排序 → 输出最终候选列表                      │
└─────────────────────────────────────────────────────┘
```

---

## 常见问题

**Q：`structural_reserve=True` 是什么意思？**  
A：这个候选虽然是结构候选，但被标记为"备用"，在排序中会略微靠后。  
通常是那些"有用但不那么紧迫"的候选。

**Q：为什么要区分 `primitive` 和 `structural`？**  
A：`primitive` 是原子动作（走一步棋），`structural` 是经过分析的多步计划（走一组棋）。  
搜索算法如果只看原始动作，可能需要很多步才能解开复杂局面；  
结构候选相当于"专家提示"，帮助搜索跳过无效的中间状态。

**Q：`focus_tracks` 有什么用？**  
A：用于候选去重和选择。同一组股道上的候选不需要保留太多，避免搜索树爆炸。

**Q：`DebtChain` 和 `DebtCluster` 的区别？**  
A：`DebtCluster` 是单条股道上的问题集合；`DebtChain` 是多个相互依赖的股道形成的问题链。  
处理 DebtChain 时需要协调多条股道，比单独处理 DebtCluster 更复杂。
