# fzed_shunting 当前求解算法流程详解

本文按当前代码状态说明 `fzed_shunting` 求解器的工作方式。它面向两类读者：

- 业务/产品侧：想知道系统到底怎样把一组车辆拆成可执行的调车钩计划。
- 开发/调试侧：想知道求解失败、勾数偏多、路径不通、校验失败时应该看哪些模块和指标。

先给一句总括：

> 当前求解器是一个“真实钩动作 + 强约束候选生成 + 启发式搜索 + 构造保底 + 局部修复 + 独立回放校验”的调车计划求解器。它不是只算最终每辆车在哪，而是从现场初始状态开始，一钩一钩生成、搜索、回放并验证一条可执行计划。

---

## 1. 它求解的不是静态分配，而是动态调车过程

福州东调车问题看起来像“把车送到目标股道”，但实际更复杂：

- 车辆按顺序停在线路上，前车会挡住后车。
- 机车要先能到达挂车股道，不能凭空挂车。
- 挂走的只能是股道北端可接近的一段连续车组。
- 挂在机车后的车辆也有顺序，卸车时只能卸机车携带序列的尾部连续车组。
- 目标可能是明确股道、作业区、精确台位或大库随机台位。
- 称重车在未称重前，实际目标先变成 `机库`。
- 路径上不能有中间股道被占用，还要满足 L1 限长和倒车余量。
- 单钩有业务上限：空车、重车、称重车、关门车都有约束。
- 最终计划必须能被独立 `replay` 和 `verifier` 复核。

所以求解器真正回答的是：

> 在当前现场状态下，下一钩有哪些合法动作？选哪一钩之后，现场如何变化？重复多少次后，所有车辆才能满足目标？

这就是一个有硬约束的状态空间搜索问题。

---

## 2. 端到端主流程

从命令行或 demo 调一次求解，大致会经过下面这条链路：

```text
JSON payload
  -> normalize_plan_input(...)
  -> build_initial_state(...)
  -> solve_with_validation_recovery_result(...)
  -> solve_with_simple_astar_result(...)
  -> replay_plan(...)
  -> verify_plan(...)
  -> hook_plan / final_state / debug_stats
```

对应核心文件：

| 阶段 | 作用 | 主要文件 |
|---|---|---|
| 读取主数据 | 加载股道、台位、区域、物理路径、业务规则 | `src/fzed_shunting/domain/master_data.py` |
| 输入归一化 | 把原始 payload 转成统一求解输入 | `src/fzed_shunting/io/normalize_input.py` |
| 初始状态 | 把车辆按股道和顺序整理成 `ReplayState` | `src/fzed_shunting/verify/replay.py` |
| 主求解 | 构造保底、搜索、恢复、压缩、遥测 | `src/fzed_shunting/solver/astar_solver.py` |
| 搜索循环 | exact / weighted / beam 共用的搜索核心 | `src/fzed_shunting/solver/search.py` |
| 动作生成 | 生成合法 `ATTACH` / `DETACH` 候选 | `src/fzed_shunting/solver/move_generator.py` |
| 回放校验 | 重放每一钩并检查业务/路径约束 | `src/fzed_shunting/verify/replay.py`, `src/fzed_shunting/verify/plan_verifier.py` |

CLI 默认走外部验证口径：`beam`、`beam_width=8`、`timeout_seconds=60`，并用 `solve_with_validation_recovery_result(...)` 做失败后的宽 beam 重试。这个设计保证命令行、demo、批量验证尽量使用同一套求解策略，而不是不同入口给出不一致结果。

---

## 3. 主数据：站场地图和业务规则

主数据在 `data/master/` 下，主要包括：

- `tracks.json`：股道类型、长度、是否可停车、是否可作为终点、连接节点、终端分支等。
- `spots.json`：台位定义。
- `areas.json`：区域定义。
- `physical_routes.json`：物理分支、长度、状态。
- `business_rules.json`：机车长度、路径中间股道是否要求净空等全局规则。

代码中用 `MasterData` 承载这些数据。可以把它理解成“地图 + 台位表 + 交通规则”。

求解器不会在代码里临时猜线路关系，而是通过 `RouteOracle` 使用主数据判断：

- 源股道到目标股道有没有路径。
- 机车当前位置能不能先到达挂车股道。
- `pathTracks` 是否等于完整路径。
- 中间股道是否被占用。
- 物理分支是否已确认。
- 经过 L1 时列长是否超过 190m。
- 倒车分支长度是否足够。

这也是当前算法能逐步接近生产使用的关键：路径可达性不是“名字能连上就行”，而是被独立路线模块约束。

---

## 4. 输入归一化：把业务目标变成可计算目标

`normalize_plan_input(...)` 会把原始 JSON 里的车辆、股道、目标信息转成 `NormalizedPlanInput`。

### 4.1 核心结构

`NormalizedPlanInput` 主要包含：

- `track_info`：本次场景中各股道可用长度。
- `vehicles`：每辆车的当前股道、顺序、长度、属性、目标。
- `loco_track_name`：机车初始股道。
- `yard_mode`：普通模式或迎检模式。
- `single_end_track_names`：单端股道集合。

每辆车的目标是 `GoalSpec`：

| 字段 | 含义 |
|---|---|
| `target_mode` | `TRACK` / `AREA` / `SPOT` |
| `target_track` | 名义目标股道 |
| `allowed_target_tracks` | 求解器实际允许落位的股道集合 |
| `preferred_target_tracks` | 优先股道 |
| `fallback_target_tracks` | 首选不可用时可退让的股道 |
| `target_area_code` | 作业区或随机区，例如 `大库:RANDOM` |
| `target_spot_code` | 精确台位 |

一个很重要的点：求解时主要看 `allowed_target_tracks` 和当前状态下的 `goal_effective_allowed_tracks(...)`，不是只看原始 `targetTrack`。

### 4.2 目标如何变成规则

常见目标会被归一化成下面几类：

- 明确股道：例如 `存1`、`存4北`、`机库`。
- 作业区域：例如 `调棚:WORK`、`洗南:WORK`、`油:WORK`、`抛:WORK`。
- 大库随机：例如 `大库:RANDOM`，允许进入 `修1库内` 到 `修4库内`，同时按车辆长度区分首选/备选。
- 大库精确台位：`SPOT` 模式下要求具体台位。
- 称重：车辆带称重属性且尚未称重时，当前有效目标先被改成 `机库`。
- 关门车：如果最终到 `存4北`，必须在第 4 位及以后，即索引至少为 3。

---

## 5. 状态模型：求解器眼中的“当前现场”

当前状态由 `ReplayState` 表示：

```python
class ReplayState(BaseModel):
    track_sequences: dict[str, list[str]]
    loco_track_name: str
    loco_node: str | None = None
    weighed_vehicle_nos: set[str] = set()
    spot_assignments: dict[str, str] = {}
    loco_carry: tuple[str, ...] = ()
```

每个字段都有实际业务含义：

| 字段 | 含义 |
|---|---|
| `track_sequences` | 每条股道当前车辆序列，按北端到南端排列 |
| `loco_track_name` | 机车当前所在股道 |
| `loco_node` | 机车当前所在端点，用于端点可达性判断 |
| `weighed_vehicle_nos` | 已经完成称重的车辆 |
| `spot_assignments` | 已占用台位，格式是 `{vehicle_no: spot_code}` |
| `loco_carry` | 当前挂在机车后的车辆序列 |

这比早期“直接从源股道 PUT 到目标股道”的模型更贴近真实调车：现在机车可以先挂车，形成 `loco_carry`，再按携带顺序卸到不同目标。

---

## 6. 真实钩动作：ATTACH 和 DETACH

当前主搜索使用 `generate_real_hook_moves(...)`，动作类型是两类：

### 6.1 ATTACH：挂车

`ATTACH` 表示机车到某条股道，把北端一段连续车辆挂到机车后面。

约束包括：

- 只能挂 `track_sequences[source_track]` 的北端前缀。
- 机车必须能从当前位置到达 `source_track` 的可接近端。
- 如果已经有 `loco_carry`，额外挂车必须能减少后续拆卸组数，否则没有价值。
- 如果机车已挂着未称重车，继续挂车要避免破坏称重车最后去机库的顺序。
- 不能违反单钩车辆组约束和关门车相关挂车规则。

`ATTACH` 后状态变化：

- 源股道前缀车辆被移除。
- 这些车辆追加到 `loco_carry` 尾部。
- 机车位置更新到源股道。
- 车辆原先占用的台位被释放。

### 6.2 DETACH：摘放

`DETACH` 表示把 `loco_carry` 尾部一段连续车辆放到某个目标股道或临停股道。

约束包括：

- 只能摘放 `loco_carry` 的尾部连续块。
- 目标可能是有效业务目标，也可能是临停/清障目标。
- 目标股道容量要够。
- 作业区台位容量要够。
- 大库/作业区/精确台位要能成功分配台位。
- 路径必须可达，`pathTracks` 必须完整且中间股道净空。
- 列长、L1、倒车分支余量必须通过。

`DETACH` 后状态变化：

- 被摘放车辆从 `loco_carry` 尾部移除。
- 车辆放到目标股道北端，也就是 prepend 到目标序列前面。
- 如果目标是 `机库`，这些车辆进入 `weighed_vehicle_nos`。
- 如果目标需要台位，则重新分配 `spot_assignments`。
- 机车位置更新到目标股道。

### 6.3 为什么要拆成两类动作

拆成 `ATTACH` / `DETACH` 的好处是：

- 能表达机车后挂多组车再分批摘放。
- 能表达“先挂挡车，再决定临停还是送目标”的真实操作。
- 能用 `loco_carry` 描述挂车顺序，避免不符合物理顺序的卸车动作。
- 每个动作都算一钩，更接近业务统计口径。

---

## 7. 候选动作生成：先过滤违法动作，再交给搜索

搜索本身不负责“想象所有可能违法动作再处罚”。当前设计是：

> `move_generator` 尽量只生成合法候选；搜索只在合法候选中选择。

这能显著缩小搜索空间，也能避免搜索器靠权重把违法动作“优化”成看似更短的计划。

### 7.1 ATTACH 候选怎么来

当 `loco_carry` 为空时，候选来源包括：

- 直接挂取未完成目标车辆所在股道的北端前缀。
- 挂走挡在目标车前面的前缀车组。
- 挂走占用关键路径、导致机车或车辆无法到达目标的阻塞车组。
- 挂走容量释放所需的前端车辆。
- 挂走占用精确台位或作业台位的车辆。

当 `loco_carry` 非空时，只允许“继续挂上有助于减少后续摘放组数”的车辆。否则每次额外挂车都只会增加复杂度和返工。

### 7.2 DETACH 候选怎么来

当 `loco_carry` 非空时，会枚举尾部连续块：

- 可以直接放到有效目标股道。
- 如果车未称重，目标会优先是 `机库`。
- 如果直接目标不可行，可以放到临停线或可用存车线做清障。
- 如果摘放一部分能暴露后面更关键的车辆，也会生成相关动作。

临停目标按优先级排序：

- 优先 `TEMPORARY` 类型股道。
- 必要时使用 `STORAGE` 类型股道作为 fallback。
- 结合源到临停、临停到后续目标的距离排序。
- 如果当前存在路径阻塞压力，会优先选择能降低阻塞压力的临停目标。

### 7.3 动作硬约束

动作进入搜索前，会经过多层硬过滤：

| 约束 | 说明 |
|---|---|
| 单钩车辆数 | 纯空车最多 20 辆；重车最多 2 辆；重车折算后不超过 20 辆空车 |
| 称重 | 单钩称重最多 1 辆；未称重车有效目标先是 `机库` |
| 关门车 | 大钩首位关门车受限制；到 `存4北` 时还要保证最终位置不在前三 |
| 股道容量 | 目标股道有效长度不能超 |
| 作业区容量 | `调棚/洗南/油/抛` 有各自容量上限 |
| 台位分配 | 大库、作业区、精确台位必须能分配合法空台位 |
| 机车可达 | `ATTACH` 前机车必须能到达源股道 |
| 路径可达 | `DETACH` 路径必须存在且完整 |
| 路径净空 | 中间股道被占用则不可走 |
| L1 限长 | 经过 L1 时列长不能超过 190m |
| 倒车余量 | 需要倒车的分支长度必须容纳车辆加机车 |

这套过滤是算法正确性的底座。后面的搜索、beam、LNS 只是在合法动作之间选更短、更稳的路径。

---

## 8. 目标判定：什么叫“已经完成”

`_is_goal(...)` 会检查：

- 机车不能还挂着车，即 `loco_carry` 必须为空。
- 每辆车都满足 `goal_is_satisfied(...)`。

`goal_is_satisfied(...)` 不只是判断“车在允许股道上”，还会检查：

- 称重车是否已经进入过 `机库`。
- `SPOT` 目标是否拿到指定台位。
- `大库:RANDOM` 是否分配到合法库内台位。
- 作业区目标是否分配到合法作业位。
- 关门车在 `存4北` 是否位于第 4 位及以后。
- 首选/备选股道策略是否被满足。

所以最终完成态是“位置 + 称重历史 + 台位 + 顺序规则”共同满足，而不是简单的车辆到轨。

---

## 9. 搜索核心：同一个循环支持 exact / weighted / beam

`_solve_search_result(...)` 是搜索主循环。它做的事情可以用下面伪代码理解：

```text
把初始状态放入优先队列
记录初始状态的最优已知成本

while 队列非空且预算未耗尽:
    取出当前优先级最高的状态
    如果当前状态已完成:
        exact 直接返回
        weighted/beam 记录当前最好完整解，继续看是否还有更好解

    生成当前状态下所有合法 ATTACH/DETACH 候选
    for 每个候选动作:
        应用动作得到 next_state
        计算 next_state 的状态键
        如果以前用更少钩数到过同一状态，跳过
        计算启发式、纯净度、路径阻塞奖励/惩罚
        入队

    如果是 beam 模式:
        只保留 beam_width 个较好的队列状态

如果预算耗尽:
    返回最好完整解；如果没有完整解，则返回最好 partial_plan
```

关键字段：

- `cost`：已经用了多少钩。
- `heuristic`：从当前状态到完成态至少还需要多少钩的下界估计。
- `priority`：队列排序用的综合指标。
- `state_key`：状态去重键。
- `best_cost`：到达某状态的最小已知钩数。
- `best_goal_plan`：非 exact 模式下发现的最好完整解。
- `best_partial_plan`：预算耗尽时用于诊断和后续恢复的最佳部分计划。

### 9.1 状态去重为什么复杂

`_state_key(...)` 不只是记录每条股道有哪些车，还包含：

- 机车股道。
- 机车端点。
- 已称重车辆集合。
- 台位分配。
- 当前 `loco_carry`。

同时，大库随机车辆的非保留随机台位会做一定规范化，避免同等随机分配造成大量重复状态。

这个状态键决定了“两个现场局面是否等价”。如果漏掉称重、台位、机车端点或 `loco_carry`，搜索就可能错误合并两个业务上完全不同的状态。

### 9.2 exact 模式

`exact` 使用 `cost + heuristic` 排序。启发式必须是可采纳的，也就是不能高估真实剩余钩数。

在预算足够且搜索能自然结束时，`exact` 可以证明最优。当前入口在开启 fallback 时，主 exact 阶段只拿一部分预算；这样做是为了避免 exact 在复杂场景里把时间全部耗尽，导致后面的恢复策略完全没有机会。

### 9.3 weighted 模式

`weighted` 使用 `cost + heuristic_weight * heuristic` 排序。

它更贪心，通常更快，但不保证最优。适合在复杂场景中先快速找可行解。

### 9.4 beam 模式

`beam` 也是启发式搜索，但每轮会修剪队列，只保留宽度内的候选。

当前 beam 的优先级会额外考虑：

- 路径阻塞回归惩罚。
- 是否在清除挡路目标。
- 已用钩数。
- 启发式值。
- 首选/备选纯净度。
- 临停污染程度。
- 结构多样性保留，避免 beam 全部落入同一种高返工路线。

beam 是当前外部验证和 CLI 默认口径，因为它在大场景里比 exact 更可控。

### 9.5 lns 模式

`lns` 不是一个完全独立的求解器，而是“先求一条解，再局部破坏并重求尾部”的改良层：

1. 先用 beam 或 weighted 求 seed。
2. 找若干切点。
3. 固定前缀，从切点状态重新搜索后半段。
4. 如果新计划更短、路径更好、纯净度更好，就接受。

切点来源包括：

- 热点股道：被反复触达的区域。
- 高成本钩：路径长或分支多的钩。
- 同目标连续段：可能可以合并。
- 均匀采样：防止只盯热点。

---

## 10. 启发式：告诉搜索“离目标还多远”

当前真实钩模型用 `compute_admissible_heuristic_real_hook(...)`。

它的核心思想：

- 还在股道上的车，通常至少需要 `ATTACH + DETACH` 两钩。
- 已经挂在 `loco_carry` 上的车，只需要后续 `DETACH` 组。
- 称重、台位、容量释放、前端挡车都要作为剩余工作下界。

主要分量包括：

| 分量 | 含义 |
|---|---|
| distinct transfer pairs | 不同源到目标组合至少需要的转移次数 |
| blocking | 目标股道前端挡车和互相阻塞环 |
| weigh | 尚未称重的车辆数 |
| spot evict | 精确台位被别人占用时的腾挪需求 |
| tight capacity | 容量紧张时，已在目标股道的车可能也要临时让位 |
| carry detach groups | 当前 `loco_carry` 尾部按目标可合并的最少摘放组 |

启发式不是业务规则本身，而是搜索排序工具。硬规则仍由动作生成和 verifier 保证。

---

## 11. 构造保底：先用规则快速造一条路

在 `exact` 和 `beam` 入口下，`solve_with_simple_astar_result(...)` 会先尝试 `constructive`。

它不是全局搜索，而是一个带回溯的规则分派器：

```text
while 未完成:
    生成当前状态下的合法动作
    给每个动作打分
    选当前最好的动作
    如果长期没有进展:
        回退到近期决策点，尝试第 2 / 第 3 备选动作
```

它的价值不是“保证最短”，而是：

- 快速得到一个完整解或接近完整的 partial。
- 给 beam/exact 主搜索提供兜底。
- 为后续 partial resume、route release、goal frontier 等恢复策略提供真实中间状态。

### 11.1 构造评分关注什么

构造层会优先选择：

- 直接完成目标的 `DETACH`。
- 能改善首选股道落位的摘放。
- 能暴露关键携带车辆的动作。
- 能清除路径阻塞或精确台位阻塞的动作。
- 能推动关门车位置合法化的动作。
- 能降低启发式的 `ATTACH`。
- 能清除前端挡车的动作。

它会压低优先级的动作包括：

- 把已满足目标的车又挪走。
- 重新把未完成车从临停线抓起后又形成循环。
- 把车停到会制造新路径阻塞的位置。
- 不必要的 staging-to-staging 返工。

构造层也会记录最近状态和最近反向动作，尽量避免“刚搬出去又搬回来”的振荡。

---

## 12. 分层恢复：不是一失败就重跑

复杂调车场景里，主搜索经常不是“完全不会解”，而是卡在某类局部问题。当前入口把这些局部问题拆成几个恢复层。

### 12.1 warm start

如果构造层只差很少一步，例如启发式 `h=1`，入口会从构造 partial 的最终状态启动一个短预算 exact 搜索补尾。

### 12.2 partial resume

构造层可能中途失败，但已经经过很多合法动作。入口会找一些 `loco_carry` 为空的 checkpoint，从这些真实中间状态继续搜索，而不是从头重来。

这比“整个场景重新 beam”更精准，因为前缀已经完成了大量确定工作。

### 12.3 route release

有些 partial 不是目标还远，而是路径被已停车辆或临停车挡住。`route_blockage` 会计算：

- 哪些股道是阻塞源。
- 哪些车辆被阻塞。
- 哪些源/目标对受影响。
- 当前阻塞压力是多少。

随后入口会优先尝试：

- 释放挡路股道。
- 对 route-release partial 做短尾搜索。
- 避免把刚清出来的路径又重新堵住。

### 12.4 relaxed constructive

严格防振荡通常能减少返工，但少数局面需要从临停线重新抓车才能收尾。入口会在主搜索前给一个短预算的 relaxed constructive retry。

### 12.5 goal frontier

还有一种常见尾部问题：前端车辆已经满足目标，但它挡住了后面未完成车辆。这个问题不是全局大搜索，而是一个局部“前沿挡车”问题。入口会尝试 goal-frontier tail completion。

### 12.6 validation recovery

CLI/demo/批量验证外层还有 `solve_with_validation_recovery_result(...)`。当默认 beam 解不完整，或者得到的完整解结构上有高返工风险时，会按配置扩大 beam width 重试。

默认 beam width 是 8，重试宽度会扩到更宽的 beam，用同一总预算继续找更稳的计划。

---

## 13. 计划压缩和后处理

求出完整计划后，入口还会做几类后处理。

### 13.1 LNS polish

如果完整解不是 exact 证明最优，并且计划较长，会用剩余预算跑 LNS polish，尝试缩短勾数。

### 13.2 verifier-guarded compression

`plan_compressor` 会尝试局部合并或删除冗余窗口，例如相邻同源同目标的挂摘组合。但每次改写都必须：

- 能 replay 到一致的终态。
- 通过完整 verifier。

失败的改写直接丢弃，不影响原计划。

### 13.3 depot-late reorder

当 `enable_depot_late_scheduling=True` 且条件允许时，会尝试把大库相关钩往后挪，减少过早入库。但候选重排也要经过 verifier 防护；如果校验不通过，就保留原计划。

CLI 外部验证入口当前会关闭这个选项，避免验证口径被展示优化影响。

---

## 14. 回放与校验：最终安全网

求解器不是生成 plan 后直接相信自己，而是再走一遍独立检查。

### 14.1 replay_plan

`replay_plan(...)` 从初始 `ReplayState` 开始逐钩执行：

- `ATTACH`：校验源股道前缀，移除车辆，追加到 `loco_carry`。
- `DETACH`：校验源股道等于机车当前股道，移除 `loco_carry` 尾部块，prepend 到目标股道。
- 更新机车位置和端点。
- 更新称重集合和台位分配。

如果计划里任何一钩不符合状态转移规则，回放会直接失败。

### 14.2 verify_plan

`verify_plan(...)` 做更完整的规则检查：

- `hookNo` 必须从 1 连续递增。
- 每钩车辆必须存在。
- 单钩车辆组约束必须满足。
- `ATTACH` 前机车必须能到达源股道。
- 每钩 `pathTracks` 必须首尾正确且匹配完整路径。
- 路径中间股道不能有干涉。
- 分支状态、L1 限长、倒车余量必须满足。
- 最终每辆车必须满足目标。
- 最终股道容量不能超。
- 关门车顺序必须合法。

完整结果如果 verifier 不通过，会抛 `PlanVerificationError`。partial plan 不会被当作完成解，只会附带 `partial_verification_report` 供诊断。

---

## 15. 一条简单计划如何被理解

假设某条股道北端有：

```text
存1: [A, B, C]
```

其中 `A`、`B` 可以一起去 `存4北`，`C` 暂时不动。

真实钩模型下，计划不会是一个抽象的“把 A/B 从存1放到存4北”，而至少是：

```text
1. ATTACH 存1 -> 存1: [A, B]
2. DETACH 存1 -> 存4北: [A, B], pathTracks=[存1, ..., 存4北]
```

第 1 钩检查：

- 机车能不能到 `存1`。
- `[A, B]` 是不是 `存1` 北端前缀。
- 单钩车辆组是否合法。

第 2 钩检查：

- `[A, B]` 是不是 `loco_carry` 尾部可摘放块。
- `存4北` 容量是否够。
- 路径是否可达、中间是否净空。
- 如果有关门车，最终位置是否可能合法。

如果 `A` 要称重，则在未称重前，`DETACH` 的有效目标会先变成 `机库`，等进入 `机库` 后才算已称重，再继续往最终目标走。

---

## 16. 如何读 debug_stats 和 telemetry

遇到“为什么慢、为什么 partial、为什么勾数多”，优先看这些字段。

### 16.1 SolverResult

| 字段 | 含义 |
|---|---|
| `is_complete` | 是否得到完整可交付 plan |
| `is_proven_optimal` | 是否 exact 证明最优 |
| `fallback_stage` | 完整 plan 来自哪个阶段 |
| `partial_fallback_stage` | partial 来自哪个阶段 |
| `expanded_nodes` | 展开的搜索状态数 |
| `generated_nodes` | 生成的搜索状态数 |
| `elapsed_ms` | 求解耗时 |
| `depot_earliness` | 大库钩过早程度观测值 |
| `depot_hook_count` | 大库相关钩数 |

常见 `fallback_stage`：

- `exact`：exact 搜索找到。
- `weighted` / `weighted_greedy` / `weighted_very_greedy`：fallback 的 weighted 阶段找到。
- `beam` / `beam_greedy_64` / `beam_greedy_128` / `beam_greedy_256`：beam 阶段找到。
- `constructive`：构造保底找到。
- `constructive_warm_start`：构造 partial + 短 exact 补尾。
- `constructive_route_release`：路径释放导向的构造恢复。
- `route_blockage_tail_clearance`：路径阻塞尾部清障恢复。
- `goal_frontier_tail_completion`：前沿挡车尾部恢复。

### 16.2 debug_stats

高价值字段：

| 字段 | 用法 |
|---|---|
| `expanded_states` | 搜索展开规模 |
| `candidate_moves_total` | 候选动作总量 |
| `max_candidate_moves_per_state` | 单状态动作爆炸程度 |
| `top_expansions` | 哪些状态生成动作最多 |
| `search_best_partial_score` | 搜索预算耗尽时最佳 partial 的启发式/未完成情况 |
| `initial_route_blockage_plan` | 初始路径阻塞压力 |
| `partial_route_blockage_plan` | partial 结束时路径阻塞压力 |
| `initial_capacity_release_plan` | 初始容量释放压力 |
| `plan_shape_metrics` | 计划形状：临停钩、反复触达、最大车辆触达次数 |
| `partial_structural_metrics` | partial 的未完成数、临停债、前端挡车等 |
| `plan_compression` | 压缩前后勾数和接受改写次数 |

### 16.3 telemetry

如果设置 `FZED_SOLVER_TELEMETRY_PATH`，求解器会把结构化 telemetry 追加为 JSON lines。

它包含：

- 输入规模：车辆数、股道数、称重车数、关门车数、SPOT/AREA 数。
- 各阶段耗时：constructive、exact、anytime、lns、verify。
- 输出状态：是否完整、勾数、fallback 阶段、校验状态。
- 预算：时间预算、节点预算。

这适合后续接生产监控或批量分析。

---

## 17. 各模块职责速查

| 文件 | 核心职责 |
|---|---|
| `src/fzed_shunting/cli.py` | 命令行入口，加载数据、调用求解、输出 plan |
| `src/fzed_shunting/workflow/runner.py` | 多阶段 workflow，阶段间继承现场状态 |
| `src/fzed_shunting/io/normalize_input.py` | 输入合同、目标归一化、属性解析 |
| `src/fzed_shunting/domain/master_data.py` | 主数据模型和加载 |
| `src/fzed_shunting/domain/route_oracle.py` | 路径、机车可达、分支、净空、限长、倒车余量 |
| `src/fzed_shunting/domain/depot_spots.py` | 大库/作业区/机库台位分配 |
| `src/fzed_shunting/domain/hook_constraints.py` | 单钩车辆组硬约束 |
| `src/fzed_shunting/domain/carry_order.py` | `loco_carry` 尾部摘放顺序 |
| `src/fzed_shunting/solver/types.py` | `HookAction` |
| `src/fzed_shunting/solver/state.py` | 状态转移、目标判断、状态键 |
| `src/fzed_shunting/solver/move_generator.py` | 生成合法 `ATTACH` / `DETACH` 候选 |
| `src/fzed_shunting/solver/search.py` | exact / weighted / beam 主搜索循环 |
| `src/fzed_shunting/solver/heuristic.py` | 可采纳启发式和真实钩启发式 |
| `src/fzed_shunting/solver/constructive.py` | 构造保底和规则分派 |
| `src/fzed_shunting/solver/anytime.py` | fallback chain |
| `src/fzed_shunting/solver/lns.py` | 局部破坏/修复改良 |
| `src/fzed_shunting/solver/route_blockage.py` | 路径阻塞压力分析 |
| `src/fzed_shunting/solver/capacity_release.py` | 容量释放压力分析 |
| `src/fzed_shunting/solver/structural_metrics.py` | 结构指标、计划形状指标 |
| `src/fzed_shunting/solver/plan_compressor.py` | verifier 防护下的局部压缩 |
| `src/fzed_shunting/solver/validation_recovery.py` | 外部验证口径的 beam 重试策略 |
| `src/fzed_shunting/verify/replay.py` | 逐钩回放 |
| `src/fzed_shunting/verify/plan_verifier.py` | 独立最终校验 |

---

## 18. 当前算法的优点

### 18.1 物理动作更真实

`ATTACH` / `DETACH` 和 `loco_carry` 让计划更接近真实调车，而不是把“挂车 + 走行 + 摘放”压成一个抽象 PUT。

### 18.2 硬约束前置

候选动作生成阶段就过滤路径、容量、台位、称重、关门车、单钩上限等硬规则。搜索不用在违法动作里做权重赌博。

### 18.3 多层恢复通用

route release、partial resume、goal frontier、validation recovery 都是围绕通用结构问题设计的，不是针对单个样例硬编码。

### 18.4 校验独立

最终交付前必须 replay + verify。即使搜索器或压缩器内部逻辑有偏差，verifier 仍有机会截住错误计划。

### 18.5 可观测性较强

`debug_stats`、结构指标、路径阻塞计划、容量释放计划和 telemetry 能帮助定位“为什么慢、为什么失败、为什么返工多”。

---

## 19. 当前算法的边界

### 19.1 exact 不等于总能证明最优

exact 只有在预算足够、搜索完整结束时才有最优证明。复杂场景下 exact 可能超预算，系统会转向 weighted/beam/constructive/recovery，返回可行解或更好的启发式解。

### 19.2 beam 是工程默认，不是数学最优

beam 通过限制队列宽度换取速度和稳定性。它适合大规模外部验证和生产型响应，但不提供严格全局最优保证。

### 19.3 构造解是保底，不应被误读为最短

constructive 的目标是快速找到可行路或高质量 partial。它有回溯和防振荡，但本质仍是规则分派，不是全局最优算法。

### 19.4 LNS 和压缩是改良，不是无限搜索

LNS 和 plan compression 都有预算和窗口限制，只接受明确更好的候选。它们能减少返工，但不会穷尽所有可能重排。

### 19.5 主数据准确性决定路径正确性

路径可达、端点、倒车分支、线路长度都依赖 `data/master`。如果主数据缺失或错误，算法再精细也无法得出正确物理计划。

---

## 20. 推荐的源码阅读顺序

如果要继续深入，建议按下面顺序读：

1. `src/fzed_shunting/io/normalize_input.py`：先理解目标和输入合同。
2. `src/fzed_shunting/verify/replay.py`：理解状态如何变化。
3. `src/fzed_shunting/domain/route_oracle.py`：理解路径为什么会通过或失败。
4. `src/fzed_shunting/solver/move_generator.py`：理解哪些动作会被生成。
5. `src/fzed_shunting/solver/state.py`：理解 `_apply_move`、`_is_goal`、`_state_key`。
6. `src/fzed_shunting/solver/search.py`：理解 exact/weighted/beam 主循环。
7. `src/fzed_shunting/solver/astar_solver.py`：理解入口调度、恢复、压缩、校验。
8. `src/fzed_shunting/verify/plan_verifier.py`：理解最终验收标准。

看 bug 时反过来：

1. 先看 verifier 错误。
2. 再看 replay 是否能走通。
3. 再看动作是否本该被生成或过滤。
4. 再看搜索为何没选中。
5. 最后才调整搜索优先级或恢复策略。

---

## 21. 最短心智模型

如果只记一版，可以记这 6 句话：

1. 输入先被归一化成车辆目标、股道容量、机车位置和业务属性。
2. 当前现场由 `ReplayState` 表示，核心是股道序列、机车位置、称重历史、台位分配和 `loco_carry`。
3. 主搜索每次只从合法 `ATTACH` / `DETACH` 动作里选下一钩。
4. `exact/weighted/beam` 共用一个搜索循环，区别主要是优先级和队列修剪。
5. 构造保底、partial resume、route release、goal frontier、LNS 和压缩是围绕主搜索的工程增强层。
6. 最终计划必须通过独立 replay 和 verifier，partial plan 只能用于诊断，不能当完整解交付。
