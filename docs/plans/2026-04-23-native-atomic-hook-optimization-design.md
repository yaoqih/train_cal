# Native Atomic Hook Optimization Design

**Date:** 2026-04-23

## Goal

把当前“块级 PUT 动作 + 原生勾补丁”的混合求解器，重新定义成纯原生勾求解器：

- 每一钩只表示一个原子化动作：`ATTACH` 或 `DETACH`
- 目标函数只优化原生勾数，不再通过 PUT 计划编译近似
- 求解、验证、回放、导出都围绕同一套原生勾语义收敛

这份文档只做分析和方案，不包含代码改动。

## Executive Summary

当前仓库里已经存在 `real_hook` 路径，但它还不是纯原生勾优化，而是“在 PUT 框架外面包一层 ATTACH/DETACH 外观”：

- `src/fzed_shunting/solver/types.py` 里的 `HookAction` 默认动作仍是 `PUT`
- `src/fzed_shunting/solver/constructive.py`、`src/fzed_shunting/solver/anytime.py`、`src/fzed_shunting/solver/lns.py` 的主工作流仍以 PUT 搜索为中心
- `src/fzed_shunting/solver/heuristic.py` 中的 `real_hook` 启发式，本质上仍是 PUT 下界乘 2 再加 carry 修正
- `src/fzed_shunting/solver/astar_solver.py` 的 `real_hook` 路径超时后还会回退到 PUT 精确解，再由 `src/fzed_shunting/solver/real_hook_compiler.py` 编译成 ATTACH/DETACH

这意味着：

1. 现在的 `real_hook` 不是“原生勾最优”，只是“PUT 最优的一个近似投影”
2. 现有搜索空间、状态键、验证流程并没有为多次摘挂的原生勾问题完全重建
3. 求解器在鲁棒性和可解释性上仍背着 PUT 时代的历史包袱

结论很明确：如果“勾”的定义已经改成原生的摘钩/挂钩原子操作，那求解器也必须硬切到原生勾模型，不能继续让 PUT 当真实目标、再把结果编译成 ATTACH/DETACH。

## Current Diagnosis

## 1. 当前系统的真实建模状态

### 1.1 动作层仍是双系统并存

从代码看，当前存在两套动作语义：

- PUT 语义
  - `src/fzed_shunting/solver/move_generator.py::generate_goal_moves`
  - `src/fzed_shunting/solver/constructive.py`
  - `src/fzed_shunting/solver/anytime.py`
  - `src/fzed_shunting/solver/lns.py`
- 原生勾语义
  - `src/fzed_shunting/solver/move_generator.py::generate_real_hook_moves`
  - `src/fzed_shunting/solver/state.py::_apply_attach/_apply_detach`
  - `src/fzed_shunting/solver/heuristic.py::make_state_heuristic_real_hook`

但这两套并不是平行的一等公民，而是 PUT 为主、原生勾为辅。最直接的证据是：

- `real_hook` 求解失败或预算不足时，会退回 PUT 搜索
- PUT 计划之后再被 `compile_put_to_real_hook()` 编译成 ATTACH/DETACH

这条回退链本身就说明：当前系统并没有真正相信原生勾模型是唯一真源。

### 1.2 当前 `real_hook` 只是原生勾的第一版近似

已有 `real_hook` 路径并非完全无价值，它已经做了两件正确的事情：

- 引入了 `ReplayState.loco_carry`
- 允许在 carry 非空时继续 `ATTACH`，以及按 carry 前缀 `DETACH`

但从问题定义上看，它仍然不够彻底：

- `real_hook` 启发式仍依赖 PUT 下界
- `constructive` / `lns` / `anytime` 仍不是 native-first
- 验证链条没有真正把原生勾当成默认主线
- 导出的 ATTACH 还带着伪目标 `LOCO` 的语义痕迹

### 1.3 当前实现里已经暴露出几个和原生勾直接冲突的问题

#### A. 目标函数不纯

`src/fzed_shunting/solver/real_hook_compiler.py` 的存在本身就是问题。它默认：

- PUT 是主动作
- ATTACH/DETACH 只是 PUT 的展示展开
- 连续同目标 PUT 可以合并成更少原生勾

这套逻辑在“ATTACH/DETACH 才是真实动作”之后就不成立了。因为：

- 原生勾最优解不一定能由 PUT 最优解编译出来
- 原生勾的最优 batching 顺序，可能要求先多次 ATTACH 再多次 DETACH
- PUT 世界里一次“搬运”被当作一个动作，但原生勾世界里它至少是两步

#### B. 原生勾状态判重不完整

`src/fzed_shunting/solver/state.py::_state_key()` 当前不包含 `loco_track_name`。

在 PUT 模型下，这个问题相对没那么突出；但在原生勾模型下，`loco_track_name` 是显式状态的一部分，因为：

- carry 非空时，下一步 `DETACH` 的源位置依赖机车当前位置
- staging / detour / 多次摘挂的可行性依赖“当前挂着车停在哪”
- 同样的股道分布，如果机车在不同位置，后续 legal moves 不是同一个集合

因此原生勾硬切后，`loco_track_name` 不是展示字段，而是状态空间的一等公民。

#### C. real_hook 路径的 verify 闭环不干净

`src/fzed_shunting/solver/astar_solver.py` 里 `real_hook` 分支有单独的早返回结构。当前实现虽然可以产出 ATTACH/DETACH 计划，但整个分支不是围绕“native plan 必须走统一 verify”设计出来的。

这在纯原生勾模型里不可接受。最佳实践应该是：

- 无论 exact / weighted / beam / lns 哪条路径
- 只要最终结果是原生勾计划
- 都必须走同一套 replay + verify 复核

#### D. ATTACH 外部契约带有历史包袱

当前 `ATTACH` 动作把 `target_track` 设成 `"LOCO"`，这更像内部中间态，而不是稳定的业务动作契约。它带来两个问题：

- verifier / replay / UI 需要理解一个“伪股道”
- `pathTracks` 对 ATTACH 的语义不自然，不利于通用验证

如果只考虑原生勾，最佳实践应该是把 ATTACH 的外部语义也收敛成“真实发生在哪条股道”，而不是保留 `LOCO` 这种过渡性概念。

## 2. 为什么原生勾优化和 PUT 优化不是一回事

原先的业务抽象是：

- 一钩 = 从 A 线取一个当前可取连续车组，送到 B 线

新的业务抽象是：

- 一钩 = 一次 `ATTACH` 或一次 `DETACH`

这不是“计数口径小修”，而是问题本体变了。

### 2.1 PUT 是“完整搬运”

PUT 把一次搬运压缩成一个复合动作：

1. 从源线挂出一组车
2. 把车带到目标线
3. 在目标线解开

它天然隐藏了两个原生勾动作之间的中间状态。

### 2.2 原生勾会显式暴露中间状态

ATTACH/DETACH 模型会显式暴露：

- 当前机后还挂着什么
- 现在是否可以继续挂其他车
- 应该先在哪个目标线解哪一部分
- 多次摘挂后 carry 中的相对顺序是什么

这些中间状态才是“降低原生勾数”的真正来源。因为节省钩数的关键，不再只是“少搬一次”，而是：

- 一次 DETACH 前是否值得先多次 ATTACH
- 当前 carry 是否应该保留一部分继续走
- 哪些车应该作为同一批 DETACH
- 哪些 ATTACH 顺序会导致后续 DETACH 组数增加

### 2.3 因此原生勾最优不能靠编译得到

从 PUT 最优解编译出 ATTACH/DETACH，只能得到“某个由 PUT 投影出来的原生勾解”，不能保证是原生勾最优解。

推荐明确放弃以下旧思路：

- “先求 PUT 最优，再编译”
- “real_hook 找不到就自动退回 PUT”
- “启发式先按 PUT 算，再乘 2”

这些办法短期上手快，但会把新问题永远锁死在旧问题的影子里。

## Hard-Cut Decisions

## 3. 本轮方案的硬切决策

### 决策 1：求解器主目标只剩原生勾数

定义：

- `hook_count = atomic_attach_count + atomic_detach_count`

任何 solver 的 primary objective 都只能是这个值。

### 决策 2：PUT 不再参与优化链路

PUT 可以保留成：

- 历史输入兼容适配器
- 对比工具
- 回放导入辅助格式

但它不能再出现在“真正的优化路径”上。

### 决策 3：所有求解模式共享同一个 native action space

不是只有 `real_hook` 一个孤立模式 native 化，而是：

- `exact`
- `weighted`
- `beam`
- `lns`
- `constructive`

最终都应该以 `ATTACH/DETACH` 为动作空间。

### 决策 4：状态空间必须对 carry 语义完全诚实

native state 至少必须包含：

- `track_sequences`
- `loco_track_name`
- `loco_carry`
- `weighed_vehicle_nos`
- `spot_assignments`

而且 state key 也必须忠实反映这些对可行性有影响的字段。

### 决策 5：verification 必须直接验证 native plan

最终验证对象只能是 solver 实际返回的 ATTACH/DETACH 序列，而不是它的 PUT 影子。

## Native Problem Formulation

## 4. 原生勾问题的推荐建模

### 4.1 状态定义

推荐把原生勾状态定义为：

1. `track_sequences`
   - 每条股道从可达端到远端的车辆顺序
2. `loco_track_name`
   - 机车当前停留位置
3. `loco_carry`
   - 当前机后挂着的车辆序列
   - 顺序定义为“下一次可优先 DETACH 的前缀在前”
4. `weighed_vehicle_nos`
   - 已完成称重的车辆集合
5. `spot_assignments`
   - 当前台位/工位占用

这个状态定义的优点是：

- 足够表达多次 ATTACH / 多次 DETACH
- 能把称重、台位、关门车、阻挡关系统一留在一个状态里
- replay、verify、UI 可以共用一套状态真源

### 4.2 原生勾动作定义

推荐内部统一成两个动作：

#### ATTACH

- 含义：从某条源股道摘出一个当前可取的连续前缀，接到机后
- 作用：
  - 源线删除该前缀
  - `loco_carry` 在尾部追加该前缀
  - 机车位置更新到该操作股道

#### DETACH

- 含义：从 `loco_carry` 的前缀摘下一段，放入目标股道
- 作用：
  - `loco_carry` 删除该前缀
  - 目标线在可达端接收该前缀
  - 机车位置更新到目标股道
  - 同步更新称重和 spot assignment

### 4.3 carry 顺序的推荐语义

推荐继续沿用当前实现的有序 carry 模型：

- ATTACH 追加到 carry 尾部
- DETACH 只能从 carry 前缀执行

原因：

- 这是当前仓库里已经存在并且最可实现的一套中间态语义
- 它比“carry 无序集合”更贴近物理可操作顺序
- 它能稳定地表达“多次挂、分批摘”的组合结构

后续如果现场确认真实物理顺序与此相反，再单独调整；但当前方案文档里不建议把 carry 顺序抽象成无序，否则搜索空间会失真。

### 4.4 搜索层补充原则

原生勾问题里，大量 setup / staging / fallback 分支在主代价上会长时间并列。要让 solver 真正收敛，工程实现上还需要两类次级原则：

- `purity` 次序
  - 在相同主代价下，优先 `unfinished` 更少、`preferred violation` 更少、`staging pollution` 更少的状态。
  - 这不是修改目标函数，而是在主代价并列区间里给搜索一个更贴近终态质量的方向。

- `partial resume`
  - constructive partial 不能直接当 solved success。
  - 但如果它已经回到 honest 的空 carry checkpoint，就应该允许主搜索从这个 checkpoint 继续补完，而不是总是从初态重开。

### 4.5 目标判定

原生勾模式的 goal 仍然应该是：

- `loco_carry` 为空
- 所有车辆已落在允许终点
- 所有称重要求满足
- 所有 spot / area / close-door 规则满足

也就是说，native 化只改变动作和成本，不改变终态业务约束。

## Optimization Objective

## 5. 目标函数与 tie-break 原则

推荐采用严格字典序：

1. 最小化原生勾数
2. 最小化 loaded path 成本
3. 最小化反向分支 / 路径复杂度 / depot earliness 等次目标

注意点：

- 所有“减少路径长度”“减少倒车”“晚进库”都只能当 secondary objective
- 不能为了这些次目标牺牲原生勾数
- 也不能再把 PUT 勾数当作 primary objective 的替代指标

## Search Architecture Recommendation

## 6. 推荐的 native-first 求解架构

### 6.1 推荐方案

推荐方案只有一个：**统一搜索框架，统一 native action space。**

保留现有搜索基础设施：

- A*
- weighted A*
- beam search
- anytime fallback
- LNS

但把它们全部切换到原生勾动作生成和原生勾启发式上。

### 6.2 不推荐方案

以下两种都不推荐继续演化：

#### 方案 A：继续 PUT 主导，real_hook 只做外层包装

问题：

- 优化目标不诚实
- 结果解释困难
- 无法从根上压缩原生勾数

#### 方案 B：保留 real_hook，但 fallback 到 PUT

问题：

- 可解性看似更高，实则把结果质量拉回旧世界
- 一旦 fallback 命中，输出就不再是真原生勾优化结果
- 很难判断“解得差”到底是 native 本身问题，还是 fallback 混入造成

## Hierarchical Algorithm Design

## 7. 分层算法设计

如果不把视野限制在“现有 A* / beam / LNS 框架的小修小补”里，那么原生勾问题最值得认真考虑的广义算法方向，就是 **分层算法**。

它的核心思想不是直接在全量原子 `ATTACH/DETACH` 空间里硬搜，而是先决定“这一趟车机后应该带哪些车、准备分几批在哪些目标家族落下”，再把它细化成具体原子勾动作。

### 7.1 为什么原生勾问题天然适合分层

原生勾问题比 PUT 模型多出的难点，本质上集中在 `carry`：

- 哪些车应该在同一趟里一起挂
- 挂的顺序怎么影响后续可拆分性
- 当前 `carry` 应该分成几批 DETACH
- 是否值得先中转 staging 再继续挂

这些决策既不是纯局部动作选择，也不是完全展开后的逐勾搜索最擅长表达的东西。它们更像是“趟次级 / 批次级”决策。

因此更自然的分层方式是：

- 上层：决定趟次结构和批次结构
- 下层：验证并细化成原子 `ATTACH/DETACH`

### 7.2 推荐的两层分解单位

推荐把一个更高层的宏单元定义为 **trip**：

- trip 起点：`loco_carry` 为空
- trip 中间：执行一段 `ATTACH* + DETACH*`
- trip 终点：`loco_carry` 再次为空

这一定义的好处是：

- 它天然对应现场里一趟完整取送车行为
- 成本可以直接写成该 trip 内的 `ATTACH` 数 + `DETACH` 数
- 上层不需要跟踪无限长的 carry 生命周期，问题被切成若干段空载到空载的闭环

### 7.3 上层需要决定什么

上层不直接决定每一钩，而是决定每个 trip 的宏结构。推荐上层输出以下信息：

1. trip 顺序
   - 先处理哪一组业务，再处理哪一组业务
2. pickup batches
   - 每个 trip 准备从哪些源线拿哪些 block family
3. detach partition
   - 同一趟 `carry` 计划分成几批落下
4. target family
   - 每一批更偏向哪类目标线 / 目标区域，而不是立即钉死到唯一路径
5. staging permission
   - 这一趟是否允许中转，以及允许哪些 staging 类型

注意，上层不应该过早承诺：

- 精确 `pathTracks`
- 精确台位编号
- 精确 ATTACH 次序细节
- 精确 DETACH 路径细节

否则就失去分层的意义。

### 7.4 下层负责什么

下层的职责是把单个 trip 细化成一个真实可执行的 native hook 子计划。

它需要在给定：

- 当前精确 yard state
- 上层给出的 trip 规格

的条件下，求出：

- 一串具体的 `ATTACH` / `DETACH`
- 每一步真实路径
- 每一步后的精确状态

并保证：

- 容量可行
- 单端线方向可行
- 称重可行
- spot / area 可行
- close-door 可行

这层本质上是一个 **局部精细求解器**，可以用：

- exact A*
- beam / weighted A*
- CP-SAT 小子问题
- 有界 DP / branch-and-bound

来做。

### 7.5 层间接口应该怎样定义

分层算法成败的关键，不是“分两层”这句话本身，而是层间接口是否干净。

推荐的接口形式：

#### 上层输出

每个 trip 输出一个 `TripSpec`，至少包含：

- `candidate_pickups`
- `candidate_drop_groups`
- `allowed_target_families`
- `staging_budget`
- `max_attach_count`
- `max_detach_count`

#### 下层返回

下层不只返回 success / fail，而应返回三类结果：

1. `feasible plan`
   - 给出精确原子计划和实际成本
2. `infeasible with reason`
   - 例如 carry 分组不可落、单端线方向冲突、容量不满足
3. `repair suggestion`
   - 例如“必须拆成两批 DETACH”“必须换 staging family”“不能先挂这组车”

如果没有这第三类反馈，上层就只能盲猜，分层会非常低效。

### 7.6 推荐的上层算法

上层不是在解物理细节，而是在解组合结构。比较合适的有三类：

#### A. Beam / Tabu / VNS over trip structures

把一个解表示成：

- 若干 trip 的有序序列
- 每个 trip 的 pickup / drop 分组方案

再通过邻域操作改进：

- 合并两个 trip
- 拆开一个过大的 trip
- 调整某个 trip 的 detach partition
- 交换两个 trip 的顺序
- 改写某批车的归属 trip

这是最现实、最容易先落地的方案。

#### B. Master problem + feasibility oracle

把上层建成：

- set partitioning
- generalized assignment
- precedence-constrained batching

之类的主问题，再让下层作为 feasibility oracle。

这更接近 logic-based Benders 或 column-generation 风格，理论上更漂亮，但工程复杂度也更高。

#### C. Learned policy for trip proposals

上层也可以用学习模型提议：

- 哪些车值得合并到同一趟
- 哪些 detach grouping 更可能低勾数
- 哪些 trip 顺序更稳定

但推荐把学习只用于提议和排序，不直接替代可行性求解。

### 7.7 推荐的下层算法

下层最适合做“小而强”的可行性与细化问题。

推荐优先级：

1. native exact / weighted A*
   - 适合小 trip、需要强解释性
2. native beam + repair
   - 适合中等 trip、追求速度
3. CP-SAT 或 branch-and-bound
   - 适合约束特别紧、需要修复局部坏 trip

不推荐让下层再次回到 PUT 模型，否则分层上层输出的 native trip 结构会被下层重新扭曲。

### 7.8 分层算法的目标函数该怎样拆

推荐把全局目标分成：

- 上层优化“结构性成本”
- 下层验证“真实可执行成本”

上层的近似成本可以用：

- trip 数
- 预计 `ATTACH` 数
- 预计 `DETACH` 数
- 预计 staging 惩罚
- 预计目标分裂惩罚

下层则返回真实成本：

- 实际原生勾数
- 实际路径代价
- 实际 reverse / depot late 次目标

也就是说，上层负责 **提出低勾数结构假设**，下层负责 **确认这个结构在物理约束下是否真的低勾数**。

### 7.9 分层算法最关键的邻域设计

如果上层走 beam / tabu / VNS，邻域设计比底层求解器选择更重要。

最有价值的邻域通常不是“改一钩”，而是：

1. `merge-trips`
   - 把两趟合并成一趟，测试是否能减少总 DETACH 组数
2. `split-trip`
   - 把一个过大的 carry 拆成两趟，换取更强可行性
3. `repartition-detach`
   - 重做某个 trip 的落车分组
4. `swap-pickup-order`
   - 调整同一趟内 pickup 顺序，改善 carry 前缀结构
5. `staging-rewrite`
   - 把一次 staging 绕行改到另一类临停线

这类邻域才真正对应原生摘挂问题的结构。

### 7.10 为什么分层算法是广义算法里的首选

因为它同时解决了三个单层原子搜索很难兼顾的问题：

1. **规模**
   - 不需要在全量原子动作空间里一次展开所有分支
2. **结构**
   - 能显式表达“同一趟 carry 的合并价值”
3. **可解释性**
   - 最终答案不仅是逐钩序列，还能解释成“这几趟各自处理什么”

对于原生勾问题，真正的优化收益往往来自“趟次结构设计正确”，而不是某一步原子动作单独挑得多漂亮。

### 7.11 推荐的工程落地方向

如果后面真的实施，推荐的落地方向不是一步到位重写全部 solver，而是：

1. 先保留现有 native hook 子求解器，作为下层 refine engine
2. 在其上增加上层 `TripSpec` 搜索器
3. 先做 beam / tabu 版本的上层
4. 等 trip 邻域稳定后，再考虑 CP-SAT master 或学习增强

换句话说：

- **短期最佳路线**：分层启发式
- **中期最佳路线**：分层 + 精确子问题修复
- **长期最佳路线**：分层 + 学习排序 / 提议

## Move Generation Recommendation

## 8. 原生勾动作生成的最佳实践

### 8.1 carry 为空时：枚举 ATTACH

当 `loco_carry` 为空时：

- 枚举所有非空源股道
- 从可达端枚举合法前缀
- 对每个前缀做业务约束校验
- 产出 ATTACH 候选

### 8.2 carry 非空时：DETACH 优先，但允许继续 ATTACH

原生勾真正的价值就在这里。

当 `loco_carry` 非空时，不能简单规定“只能 DETACH”；最佳实践应该是：

- 优先枚举所有合法 DETACH 前缀
- 同时允许继续 ATTACH
- 但继续 ATTACH 必须经过“后续可分批 DETACH 可行性”检查

### 8.3 继续 ATTACH 的判定，不应该只看目标交集

当前实现里，继续 ATTACH 的主要判定思路偏向“block 的目标集合是否和 carry 当前目标集合有交集”。这太弱，也太特例化。

更稳健的做法应该是检查：

- 当前 carry 与新 block 拼接后
- 是否仍然可以被拆成若干个连续 DETACH 组
- 每个组是否存在非空共同可落目标集合
- 是否满足称重、staging、容量与关门车规则

也就是说，判定依据应该是 **combined carry 的可分组可落性**，而不是简单的目标集合 overlap。

### 8.4 保留必要的 dominance 剪枝，但不误杀可解路径

推荐保留以下 dominance：

- 同源、同目标闭包、同可行性时，优先更长前缀
- 同 carry 前缀、同目标线、同 path 成本时，只保留更优路线
- staging 候选只保留有限个主候选

但要避免两类误杀：

- 只保留“最长前缀”，导致较短前缀才是真正可解路径时被剪掉
- 只保留“目标集合重合”的继续 ATTACH，导致需要中间 staging 的可解路径被剪掉

## Heuristic Recommendation

## 9. 原生勾启发式应该怎样重建

### 9.1 现状问题

当前 `real_hook` 启发式基本是：

- PUT 下界乘 2
- 再补一个 carry 的最小 DETACH 组数

这比完全没有 native heuristic 强，但仍然不够：

- 它对“未来还需要多少 ATTACH”估计太粗
- 它默认 PUT 世界里的转移结构仍然是原生勾世界的主下界
- 对 carry 顺序、继续 ATTACH、多次 staging 的刻画不够直接

### 9.2 推荐把下界拆成 attach / detach 两部分

推荐的 native heuristic 结构：

- `h_attach_lb`
  - 剩余至少还需要多少次 ATTACH
- `h_detach_lb`
  - 当前 carry 和剩余待运车辆至少还需要多少次 DETACH
- `h_weigh_lb`
  - 称重要求带来的额外下界
- `h_spot_evict_lb`
  - spot / area 挪位带来的额外下界
- `h_blocking_lb`
  - 为释放被埋车组至少还要发生多少原生勾

exact 模式下推荐使用 admissible 组合，例如：

`h_native = max(h_attach_lb + h_detach_lb, h_weigh_lb, h_spot_evict_lb, h_blocking_lb)`

这样做的好处是：

- 语义直接对应原生勾
- 不再依赖 PUT 作为中间翻译层
- 可单独分析“attach 爆炸”还是“detach 分组爆炸”

### 9.3 weighted / beam 可以再加非 admissible 排序特征

为提高可解性，可以在非 exact 模式里额外加入排序特征：

- 更少的目标分裂数
- 更少的 carry 组数
- 更高的直接到位比例
- 更低的 staging 倾向

但这些只能用于排序，不应用来污染 exact 的 admissible heuristic。

## Robustness and Solvability

## 10. 如何提升鲁棒性、可解性、通用性

### 10.1 鲁棒性

原生勾硬切后，必须优先修三件事：

1. `state_key` 必须纳入对 native 可行性有影响的字段
2. native 结果必须走统一 verify
3. 不允许通过 PUT fallback“伪修复”native 的失败

否则系统会出现两类坏现象：

- 搜索层说可解，但 native replay / verify 并不稳定
- native 解不出来时，被 PUT fallback 掩盖真正根因

### 10.2 可解性

只靠一个 exact native 搜索不够，推荐保留完整的 native anytime 链：

1. native exact
2. native weighted
3. native beam
4. native beam greedy
5. native LNS / repair

如果要保底，也应该是 **native constructive**，而不是 PUT constructive。

### 10.3 通用性

native 模型的通用性强于 PUT，前提是不要再把它写死成特例：

- 多次 ATTACH / 多次 DETACH 应该是天然支持，而不是补丁
- staging 不应只服务某几个业务点，而应是统一候选目标类型
- verify / replay / UI 不应假设“每次搬运一定一挂一摘”

## Verification and Contract Recommendation

## 11. 推荐的验证与导出契约

### 11.1 solver 内部契约

内部统一输出：

- `action_type in {"ATTACH", "DETACH"}`
- 不再在优化路径里生成 `PUT`

### 11.2 ATTACH 的推荐外部语义

推荐把 ATTACH 的契约收敛成“操作真实发生在哪条股道”，而不是继续使用 `targetTrack="LOCO"` 这种过渡性字段。

建议：

- ATTACH 的 `sourceTrack` = 被摘出的股道
- ATTACH 的 `targetTrack` = 同一股道
- ATTACH 的 `pathTracks` = `[sourceTrack]` 或未来扩展为机车到位路线

这样做的好处：

- verifier 不需要理解伪股道 `LOCO`
- UI / 导出更自然
- replay 语义也更一致

### 11.3 DETACH 的语义保持直观

- `sourceTrack` = 当前 carry 所在位置
- `targetTrack` = 落车股道
- `pathTracks` = 完整 loaded 路径

### 11.4 replay / verifier 的职责边界

推荐：

- replay 负责状态推进真相
- verifier 负责路径、容量、称重、spot、close-door、最终目标约束
- 两者都以 native action 为第一公民

## Recommended Implementation Scope Later

## 12. 后续真正改造时的文件落点

如果后续开始实现，主要影响面应集中在：

- `src/fzed_shunting/solver/types.py`
- `src/fzed_shunting/solver/state.py`
- `src/fzed_shunting/solver/move_generator.py`
- `src/fzed_shunting/solver/heuristic.py`
- `src/fzed_shunting/solver/search.py`
- `src/fzed_shunting/solver/astar_solver.py`
- `src/fzed_shunting/solver/constructive.py`
- `src/fzed_shunting/solver/anytime.py`
- `src/fzed_shunting/solver/lns.py`
- `src/fzed_shunting/verify/replay.py`
- `src/fzed_shunting/verify/plan_verifier.py`
- `app.py`

其中优先级建议是：

1. `state.py` / `move_generator.py` / `heuristic.py`
2. `search.py` / `astar_solver.py`
3. `verify/replay`
4. `constructive` / `anytime` / `lns`
5. UI / 导出

## Acceptance Criteria

## 13. 后续实现完成时，应该满足的验收标准

### A. 目标纯度

- solver 的优化链路不再依赖 PUT
- 不再存在 `real_hook_compiled` 这种 fallback stage

### B. 正确性

- native plan 全量通过 replay + verify
- `state_key` 不再把不同 native 状态错误合并
- carry 顺序语义在测试中被明确锁定

### C. 结果质量

- 同一场景下，原生勾数不再通过 PUT 编译近似
- 多次摘挂、合并 DETACH、称重插入等场景能产生更自然的原生勾解

### D. 工程可维护性

- 原生勾动作是唯一真源
- replay / verify / UI / export 不再维护 PUT 与 ATTACH/DETACH 两套优化主语义

## Final Recommendation

## 14. 最终建议

最佳实践不是“继续修 real_hook 这个补丁模式”，而是：

**以原生勾为唯一真实动作，做一次明确的硬切重构。**

具体地说：

- 在概念上，彻底放弃 PUT 作为优化动作
- 在状态上，显式拥抱 `loco_carry + loco_track_name`
- 在搜索上，把 exact / weighted / beam / lns 统一迁移到 native action space
- 在更广义的算法层，优先把 trip / batch / detach partition 做成分层优化对象
- 在验证上，让 native plan 走完整 replay + verify 闭环

只有这样，新的“勾数”定义才会从业务口径真正落实到求解器本体，而不是停留在展示层。
