# L7 stage1-3 处理流程新手讲解

这份文档讲的是当前 `L7_CLOSED_TOPOLOGY` 模式下，项目自动生成并执行的前 3 个核心阶段：

- `stage1`: `phase1_pre_repair_buffering`
- `stage2`: `phase2_depot_area_marshalling`
- `stage3`: `phase3_ji_to_depot_allocation`

代码入口主要在两个文件：

- `src/fzed_shunting/workflow/l7_closed_topology_mode.py`: 负责把原始输入拆成阶段计划，也就是决定每个阶段“想让哪些车去哪里”。
- `src/fzed_shunting/workflow/runner.py`: 负责真正按阶段调用求解器，并把上一阶段结束后的现场状态传给下一阶段。

先说一句人话版总览：这套流程不是一上来就把所有车直接塞到最终位置，而是先把现场整理成适合入库的形态，再从大库区把该出去的车批量拉到 `存4北`，最后把已经准备好的入库车推到 `修1/修2/修3/修4/轮` 的精确目标位置。

## 1. 这 3 个阶段到底想解决什么

当前 L7 场景的特殊点是：一开始 `L15-L16` 这条联 7 相关通路是阶段性封锁的。

所以系统不能假装一开始所有路都通，也不能只看最终目标硬搜。它要先按现场约束把问题拆开：

1. `stage1`: 联 7 封锁时，先把大库入库需求整理到机区缓冲线，顺手清掉必要障碍。
2. `stage2`: 联 7 打开后，在大库区内部整理要出库或要让路的车，把它们尽量少批次拉到 `存4北`。
3. `stage3`: 把 stage1 已经整理好的入库骨架，从机区缓冲线推进到大库最终精确位置。

可以把它想成搬家：

- `stage1` 是先把要送进大库的东西打包放到门口。
- `stage2` 是把大库门里面挡路或要出去的东西先挪走。
- `stage3` 是门口的东西正式进屋，并摆到指定位置。

## 2. 入口：什么时候会进入 L7 stage1-3 流程

入口在 `solve_workflow()`。

如果输入 payload 里有：

```json
{
  "operationMode": "L7_CLOSED_TOPOLOGY"
}
```

并且还没有手工提供 `workflowStages`，runner 会自动调用：

```python
build_l7_closed_topology_workflow_payload(master, payload)
```

这个函数会把单个原始任务扩展成一个多阶段 workflow。当前实际会生成 4 个阶段：

1. `phase1_pre_repair_buffering`
2. `phase2_depot_area_marshalling`
3. `phase3_ji_to_depot_allocation`
4. `final_exact_settle_and_cleanup`

这份文档重点讲前 3 个。第 4 个是收尾阶段：冻结 stage3 已经完成的大库结果，再处理剩余没有归位的非大库车辆。

## 3. 阶段生成前先做什么：给每辆车打标签

在真正生成阶段之前，代码会先做两件事：

1. `normalize_plan_input(payload, master)`: 把输入规范化，统一字段、轨道、目标等。
2. `_build_vehicle_stage_facts(payload, normalized)`: 给每辆车整理一份 `VehicleStageFacts`。

`VehicleStageFacts` 可以理解为“这辆车在阶段规划里需要用到的事实卡片”。里面包含：

- 车号、当前位置、顺序、长度。
- 修程，比如 `段修`、`厂修`。
- 最终目标线和目标股位。
- 当前是不是在大库区域。
- 最终是不是要去 `修1/修2/修3/修4/轮`。
- 是否已经在最终位置。
- 是否属于固定大库驻留车。
- 是否需要称重。
- 是否重车。
- 是否关门车。
- 是否需要 stage2 从大库区拉出去。

后面的阶段不是直接看原始 `vehicleInfo` 硬判断，而是尽量基于这些事实卡片做统一规划。

## 4. stage1：先形成入库骨架，只做必要清障

stage1 名字是：

```text
phase1_pre_repair_buffering
```

它的描述是：

```text
联7封锁下，先形成上游入库骨架，并只做必要清障
```

### 4.1 stage1 的业务目标

stage1 的核心不是“把所有车都归位”。它只关心一件大事：

把后面要进大库的车，先整理到机区缓冲线，形成一套后续能推进大库的入库骨架。

当前缓冲线是：

```text
机南、机棚、机北1、机北2、机北3
```

stage1 同时会处理一些必须清掉的车，比如：

- 当前在 `存4北/存4南` 上，但挡住后续交换空间的车。
- 当前在机区缓冲线上，但它本身不是要进大库的车。
- 某条线上前面有非大库车挡住后面大库车时，需要先把前面的障碍车挪走。

但 stage1 不希望为了追求“看起来完成更多”而乱动太多车。它的原则是：优先形成主顺序骨架，必要清障，其他能不动就不动。

### 4.2 stage1 为什么要“打包”

调车不是拿鼠标点一辆车瞬移。实际动作通常是从某条线的前端拿一段连续车辆。

所以 stage1 先把车辆整理成 `Phase1LayoutPackage` / `Phase1Block` 这一类包或块。

简单说，每个包大概回答：

- 这包车来自哪条线。
- 包里有哪些车。
- 它们是不是要进机区缓冲线。
- 如果要进缓冲线，优先放哪条缓冲线。
- 这包车是主骨架、必要清障，还是可选清理。
- 这包车总长度多少，能不能塞进目标线。
- 这包车有没有称重、重车、关门车等特殊限制。

这里很重要：stage1 不是一辆一辆随便排，而是尽量用“连续块”表达现场上可执行的动作单元。

### 4.3 stage1 怎么区分主任务和清障

stage1 里大致有三类执行层：

| 层级 | 代码名 | 人话解释 |
| --- | --- | --- |
| 主骨架 | `L1_BACKBONE` | 真正要为后续入库准备的车，通常要放到机区缓冲线 |
| 必要清障 | `L2_REQUIRED_CLEAR` | 不清掉就会挡住主骨架、`存4`、机区缓冲等关键空间 |
| 可选清理 | `L3_OPTIONAL_CLEANUP` | 顺手做会变好，但不应该挤占主任务资源 |

这体现了当前设计方向：前面阶段要做建设性的整理，不要把问题全部压给后面，也不要把后续修补当主逻辑。

### 4.4 stage1 怎么选择哪些来源线先处理

代码会把来源线按角色分类，例如：

- `wash_gate`: `洗北/洗南/油` 这类清洗、油漆相关入口。
- `work_gate`: `调棚/预修`。
- `work_support`: `抛/调北/机棚`。
- `receiving_storage`: `存5北/存5南`。
- `yard_storage`: `存1/存2/存3`。
- `cun4_clear`: `存4北/存4南`。
- `ji_clear`: 机区缓冲线。

然后给每条来源线算一个“值不值得打开”的判断：

- 能释放多少后续入库车。
- 清障成本高不高。
- 是不是关键咽喉位置。
- 是否会消耗过多缓冲线容量。
- 是否只是低收益、弱相关的尾部清理。

最后不是无限选，而是有主槽位和弹性槽位的概念：

- 主骨架优先选择核心来源线。
- 少量弹性来源可以补充。
- 低收益来源会延后。

这就是为什么有些车明明最终也要进大库，但 stage1 可能先不动它：不是忘了，而是当前缓冲容量和主顺序优先级不允许全部展开。

### 4.5 stage1 怎么分波次执行

虽然 workflow 里 stage1 是一个阶段，但 runner 看到 `phase1WavePlans` 后，会把它拆成多个 wave 来执行。

当前 stage1 通常有 3 个 wave：

| wave | 代码名 | 人话解释 |
| --- | --- | --- |
| A | `wave_a_backbone` | 先做主骨架和最必要的清障 |
| B | `wave_b_pressure_relief` | 再做高收益减压，把后面会卡住的地方提前松开 |
| C | `wave_c_cleanup` | 最后做少量收尾清理 |

注意，这 3 个 wave 是累积式的。也就是说：

- wave A 只要求完成第一批目标。
- wave B 会在 A 的基础上增加目标。
- wave C 再继续增加目标。

runner 每执行完一个 wave，就用真实执行后的现场状态作为下一个 wave 的初始状态。

### 4.6 stage1 给车辆设置什么目标

`_phase1_goal()` 的规则很直接：

- 固定大库驻留车：留在当前轨道。
- 如果这辆车在清障或局部完成计划里：去对应临时线或局部目标线。
- 如果这辆车被选入入库骨架：去分配好的机区缓冲线。
- 其他车：保持当前线不动。

所以 stage1 的目标不是最终目标，而是阶段目标。

例如一辆最终要去 `修1` 的车，stage1 可能先让它去 `机南`，因为 `机南` 是后续入库的缓冲骨架位置。

### 4.7 stage1 具体怎么从目标变成动作

这里要分清两层：

- `l7_closed_topology_mode.py` 负责设定 stage1 的阶段目标。
- 真正的摘挂动作不是在这里手写出来的，而是交给通用求解器搜索出来。

runner 执行 stage1 时会走 `_solve_wave_stage()`。每个 wave 都会被临时变成一个子 stage：

1. 取当前 wave 的 `vehicleGoals`。
2. 调 `_build_stage_payload()`，把“当前车在哪”和“这一波目标去哪”合成普通求解 payload。
3. 调 `build_demo_view_model(..., plan_payload=None, initial_state_override=working_state)`。
4. `plan_payload=None` 表示没有手写动作计划，要让求解器自己搜索 hook。
5. 求解器生成 `ATTACH/DETACH` 计划，verifier 再复验。
6. 执行完这个 wave 后，runner 用最后一步的 `track_sequences/loco_track_name/weighed_vehicle_nos/spot_assignments/loco_carry` 更新现场，再进入下一个 wave。

说人话：stage1 这层只是告诉求解器“这一波哪些车应该去机区缓冲线、哪些车应该清走、哪些车别动”。具体怎么挂、怎么走路径、怎么摘，是下面的普通搜索求解器算出来的。

stage1 传给求解器的 `stagePolicy` 也不是摆设。里面的：

- `packageAssignments` / `layoutAssignments`
- `packageTargetRanks` / `layoutTargetRanks`
- `deferredVehicleNos`
- `phase1Diagnostics`

会在搜索评分里起作用。比如 `_phase1_structure_bonus_penalty()` 会奖励车进入分配好的缓冲线，惩罚已经明确延后的车被错误塞进缓冲线，也会惩罚隐藏车被乱动。

所以 stage1 的具体实施逻辑是：

```text
先做阶段规划 -> 拆成 wave -> 每个 wave 生成目标 payload -> 通用求解器搜索 hook -> verifier 复验 -> 状态传给下一 wave
```

它不是只“写目标就结束”，但它本身也不手工拼每一勾动作。

## 5. stage2：大库区内部编组，把要出去的车拉到存4北

stage2 名字是：

```text
phase2_depot_area_marshalling
```

它的描述是：

```text
联7开放后，先在大库区内部形成出库链，再尽量少次整列拉到存4北
```

### 5.1 stage2 为什么存在

stage3 要把 stage1 准备好的车推入大库最终位置。但如果大库里面已经有车挡着，或者有些车本来最终应该去 `存4北`，直接 stage3 会很难。

所以 stage2 先处理大库区内部的车，目标是：

- 固定驻留车不要乱动。
- 应留在大库的车留住。
- 应该离开大库的车，尽量合并成批次拉到 `存4北`。
- 有些虽然最终就在 `存4北`，但为了给 stage3 让路，也可能一起参与转移。

### 5.2 stage2 先把大库车分成几类

`_build_stage2_plan()` 会把大库区域车辆分成几组。这里先把话说直：我之前把 `fixedDepotResidentVehicleNos` 和 `depotStayVehicles` 写成并列分类，是不准确的。

更准确地说：

- `depotStayVehicles` 是“本阶段留在大库区域”的集合。
- `fixedDepotResidentVehicleNos` 是其中更强的一类“硬锚点”，表示这辆车已经在大库合适位置上，stage2 不应该为了拉别的车把它动掉。

说人话就是：固定驻留不是另一类车，它就是“留在大库不动”的车里面，被额外打了一个“绝对别动我”的标记。

代码里如果一辆车是固定驻留车，会先进入 `depot_stay_members`，同时它的车号也会出现在 `fixedDepotResidentVehicleNos` 里。所以固定驻留可以理解成“大库留存里的特殊保护标签”，不是另一批独立车辆。

| 类别 | 字段 | 人话解释 |
| --- | --- | --- |
| 大库留存 | `depotStayVehicles` | 当前仍应留在大库区域的车 |
| 固定驻留保护 | `fixedDepotResidentVehicleNos` | `depotStayVehicles` 里的硬锚点，不能被当作解锁车或出库车拉走 |
| 最终存4 | `cun4FinalVehicles` | 最终目标是 `存4北` 的车 |
| 大库出库 | `depotOutboundVehicles` | 当前在大库，但最终不该留在大库，需要拉出 |

这里有一个关键保护：固定大库驻留车会从 `cun4FinalVehicles` 和 `depotOutboundVehicles` 里剔除，避免被误当成需要转移的车。

### 5.3 stage2 怎么把车组成出库组

大库里每条线上的车不是随便合并。代码会按：

- 当前轨道。
- 当前顺序。
- 是否连续。
- 目标类型是否一致。
- 修程是否一致。
- 最终目标族是否一致。

来构造 `Phase2OutboundGroup`。

如果两辆车在同一条线并且相邻，而且目标性质一致，才会被合成同一个 group。

这符合现场逻辑：能连着摘的一段车，才适合作为一个操作单元。

### 5.4 stage2 的 layer 是什么

有了 group 之后，代码会再构造 `Phase2TrackLayer`。

layer 可以理解成“从某条大库线前端往里看的第几层”。

举个例子，某条线从外到内是：

```text
[A, B, C, D]
```

如果 `A/B` 是第一组，`C/D` 是第二组，那么第二组虽然也需要处理，但必须先暴露前面的第一组。于是 layer 记录了：

- 当前层是哪条线。
- 这是第几层。
- 这一层有哪些车。
- 为了暴露这一层，前缀里有哪些车。

这能帮助 stage2 判断：哪些车是必须拉的，哪些车是为了让后面那批车暴露出来而不得不动的。

### 5.5 stage2 怎么决定真正转移哪些车

`_build_phase2_execution_plan()` 会选出真正要转移到 `存4北` 的车。

核心几类是：

- `mustPullVehicleNos`: 本来就需要从大库拉出去的车。
- `predecessorUnlockVehicleNos`: 自己未必需要出去，但挡住了后面的必拉车辆，所以要先拉。
- `phase3ClearanceVehicleNos`: 最终在 `存4北`，但当前位置会影响 stage3 入库通道或大库空间，所以要清出来。
- `deferredTailVehicleNos`: 暂时不拉，留给后续或收尾。

这一步很重要：stage2 不是把所有可疑车辆都拉走，而是按“必须拉、为了解锁必须拉、为了 stage3 清路必须拉”的顺序控制规模。

### 5.6 stage2 为什么要拆批次

调车有硬约束，不能一口气想拉多少就拉多少。

当前 stage2 转移批次会检查：

- 普通空车数量不能超过上限。
- 重车最多 2 辆。
- 重车按更高等效数量计入限制。
- 关门车不能放在某些会违反规则的位置。
- 称重车最多 1 辆，并且需要在尾部。
- L1 转移总长度不能超过 `193.0m`。

所以 `_split_phase2_layers_for_hook_constraints()` 和 `_build_phase2_collection_batches()` 会把原本计划的一批车拆成合法批次。

人话说：宁愿多分一批，也不能生成现场规则不允许的一大钩。

### 5.7 stage2 为什么执行时还会 rebuild

从业务意图上说，你的理解是对的：stage1 和 stage2 处理的主来源不同。

- stage1 主要处理大库外其他区域，把要入库的车整理到机区缓冲线。
- stage2 主要处理大库区域，把该出库或该让路的车转移到 `存4北`。

所以正常情况下，stage1 不应该改动大库内部车列；stage2 也不应该依赖 stage1 先替它整理大库。换句话说，stage2 的业务对象就是大库区域车辆，它应该能基于大库区域自己的初始状态来计划。

那为什么代码里还会有：

```python
rebuild_phase2_execution_policy_for_runtime(...)
```

原因不是“stage1 和 stage2 混在一起了”，也不是“stage2 必须靠 stage1 改完现场才能算”。更直接地说：这是执行层做的一次保险复核。

静态 `phase2ExecutionPlan` 是 workflow 生成时根据原始 facts 做出来的；runner 真正跑 stage2 时，会拿当前状态再扫一遍大库线，确认前缀、锚点和尾部阻塞关系有没有变化。如果 stage1 确实没碰大库内部，且中间没有外部改写，那么 rebuild 结果理论上应该和静态计划差不多。

它会根据当前真实 `track_sequences` 重新扫描大库线：

- 当前每条线前面实际是谁。
- 哪些车已经暴露。
- 哪些车被固定驻留车或留库车挡住。
- 哪些尾部车辆应该延后。

然后重新生成 runtime 版本的 execution plan。

所以 rebuild 更像一个“阶段交接前的现场复核”，不是 stage2 的业务前置条件。它主要防止几类问题：

- 静态计划生成后，实际执行状态和原始输入不一致。
- stage1 wave 执行、用户自定义 workflow、恢复执行等场景导致当前车列与原始 facts 有偏差。
- 大库线内有固定驻留锚点时，尾部车辆虽然在静态分类里看似可处理，但 runtime 前缀上实际不能越过锚点。
- 某些可选尾部车不值得为了本阶段强拉，应该作为 `deferredTailVehicleNos` 延后。

如果业务上明确保证 stage1 永远不触碰大库区、且 workflow 不会被外部改写，那么 rebuild 可以理解成防御性校准；它不应该改变 stage2 的业务边界，也不应该成为解释 stage2 的主线。

### 5.8 stage2 怎么真正生成动作

runner 进入 `_solve_phase2_execution_stage()` 后，会按 `collectionBatches` 执行。

每一批大致是：

1. 根据 `phase2ExecutionPlan.trackLayers` 建立 `pending_by_track`。
2. 按 `collectionBatches` 一批一批处理。
3. 在每一批里，从各条大库线找“当前前缀已经暴露”的 layer。
4. 用 `RouteOracle.validate_loco_access()` 判断机车能不能到这条线前端。
5. 用 `RouteOracle.resolve_clear_path_tracks()` 算出挂车路径。
6. 对可达 layer 追加一个 `ATTACH` 到 `plan_payload`。
7. 同步用 `_apply_move()` 更新临时 `working_state`，这样下一次选择看到的是动作后的现场。
8. 当前批次挂完后，如果机车挂着车，就追加一个去 `存4北` 的 `DETACH`。
9. 再次用 `_apply_move()` 更新临时状态，进入下一批。

这就是 stage2 描述里“尽量少次整列拉到存4北”的实际含义：不是一辆一辆送，而是先挂多段，再集中摘到交换线。

stage2 和 stage1 最大不同在这里：stage2 不是完全交给搜索器自由搜。它会先在 workflow 层手工构造一份 `plan_payload`：

```text
ATTACH 某条大库线前缀车辆
ATTACH 另一条大库线前缀车辆
...
DETACH 到存4北
```

然后再调用：

```python
build_demo_view_model(..., plan_payload=plan_payload)
```

`plan_payload` 不是 `None` 时，`build_demo_view_model()` 不会再让求解器自由搜索完整计划，而是把这份外部动作计划转成 demo hook，计算路径信息，并交给 replay/verifier 去复验。

所以 stage2 具体实施可以拆成三层：

```text
目标层：哪些车本阶段要去存4北，哪些车保持
编组层：把大库线前缀车辆组成 layer 和 collection batch
动作层：直接生成 ATTACH/DETACH plan_payload，再复验
```

如果必要 layer 被挡住，stage2 还有一段运行时补救逻辑：尝试 `_phase2_build_release_task()`，把挡路车释放到可用缓冲线。这个 release task 也会变成明确的 `ATTACH/DETACH` 动作追加到 `plan_payload`，不是口头目标。

### 5.9 stage2 给车辆设置什么目标

`_phase2_goal()` 的规则大致是：

- 固定大库驻留车：保持当前轨道。
- stage1 已放到机区缓冲线的入库骨架：继续保持在缓冲线。
- 本阶段选中要转移的车：目标设为 `存4北`，来源标记 `PHASE2_TRANSFER_TO_CUN4`。
- 本阶段计划延后的尾部车：保持当前线。
- 大库留存车：保持当前大库线。
- 其他车：阶段保持。

所以 stage2 的主角其实是大库区内部车辆，不是 stage1 准备好的入库骨架。

这里再补一句：stage2 的 `vehicleGoals` 主要用于描述阶段结果和 verifier 复验口径；真正的主动作来自 `_solve_phase2_execution_stage()` 生成的 `plan_payload`。这也是为什么说 stage2 是“目标 + action”两条线一起工作。

## 6. stage3：把入库骨架推到最终大库位置

stage3 名字是：

```text
phase3_ji_to_depot_allocation
```

它的描述是：

```text
将已成形的入库骨架直接推进到大库最终精确位置
```

### 6.1 stage3 的业务目标

到 stage3 时，前两步理论上已经完成了两件事：

- stage1 已经把要入库的车整理到了机区缓冲线。
- stage2 已经把大库里需要出去或需要让路的车转移到了 `存4北`。

于是 stage3 就不再做大范围整理，而是开始恢复最终目标：

- 最终要去 `修1/修2/修3/修4/轮` 的车，使用原始最终目标。
- 如果有精确股位 `isSpotting`，也在 stage3 恢复。
- 非大库入库车不在 stage3 强行归最终目标，只动态保持当前位置。

### 6.2 stage3 给车辆设置什么目标

`_phase3_goal()` 非常关键：

- 固定大库驻留车：继续保持。
- `needs_depot_batch == True` 的车：恢复原始最终目标。
- 其他车：目标设为当前轨道，`targetSource = PHASE3_DYNAMIC_CURRENT_HOLD`。

换句话说，stage3 是大库精确入位阶段，不是全场所有车最终归位阶段。

这也是为什么后面还有第 4 阶段：stage3 专心解决大库入库，不把非大库车辆的收尾也塞进来。

### 6.3 stage3 具体怎么从目标变成动作

stage3 初始目标来自 `_phase3_goal()`，但 runner 真正执行前还会做一次 `_resolve_phase3_depot_targets()`。

这一步主要干三件事：

1. 对 `PHASE3_DYNAMIC_CURRENT_HOLD` 的车，把目标改成“当前所在轨道保持不动”。这些车不是 stage3 主角。
2. 对目标是 `大库` 或 `大库:RANDOM` 的车，选择一个具体的 `修1/修2/修3/修4`。
3. 根据当前大库股位占用，用 `realign_spots_for_track_order()` 检查这个选择能不能放得下、股位能不能对齐。

如果车当前已经在 `修1/修2/修3/修4`，并且它本来就是留在大库里的，stage3 会把它具体化成当前轨道保持，避免没必要地搬动。

如果需要选择具体大库线，`_choose_phase3_depot_track()` 会看：

- 目标里有没有 preferred/fallback/allowed tracks。
- 车长是否适合某些线。
- 当前每条修线已经计划放多少车。
- 股位重排后是否合法。

选好后，stage3 会生成 `phase3WavePlans`。这和 stage1 的 wave 类似，但分组依据是目标线：

```text
轮 -> 修4 -> 修3 -> 修2 -> 修1
```

runner 看到 `phase3WavePlans` 后，也会走 `_solve_wave_stage()`。每个 wave 只激活当前目标线的入库车，其他车临时保持当前轨道。

stage3 和 stage1 一样，通常不手工生成 `plan_payload`。它调用：

```python
build_demo_view_model(..., plan_payload=None, initial_state_override=working_state)
```

也就是说，stage3 的动作仍然由通用求解器搜索出来。不同的是，stage3 的目标已经非常具体：

- 哪些车要进 `轮/修4/修3/修2/修1`。
- 哪些车保持当前轨道。
- 随机大库目标已经尽量落到具体修线。
- 精确股位或可重排股位已经在解析阶段考虑过。

说人话：stage3 不是“把目标写回最终目标就完事”。它会先把动态大库目标落到具体修线，再按目标线拆波次，然后每一波交给求解器生成实际 hook。

### 6.4 stage3 为什么是当前主要瓶颈之一

从历史诊断结果看，stage1 和 stage2 的通过率通常比 stage3 高，stage3 的 `ji_to_depot_allocation` 是高尾部耗时和失败的主要来源。

这很好理解：stage3 要把前面准备好的车精确塞进 `修1/修2/修3/修4/轮`，还要满足路线、长度、股位、称重、关门车、重车等约束。它已经不是“先放到某个缓冲线”这种相对宽松的目标，而是更接近最终业务结果。

所以排查 L7 问题时，不要只看“stage3 失败了”，还要回头看：

- stage1 是否把入库骨架排得太散。
- stage1 是否把不该留下的压力留给后面。
- stage2 是否真的清出了 stage3 需要的通道和空间。
- stage2 的 `deferredTailVehicleNos` 是计划性延后，还是运行时真的被挡住。

## 7. runner 怎么把阶段串起来

多阶段 workflow 的关键不是“生成 3 个 payload”这么简单，而是每阶段执行完后，现场状态会传给下一阶段。

runner 会继承：

- `track_sequences`: 每条线当前车列顺序。
- `loco_track_name`: 机车当前在哪。
- `weighed_vehicle_nos`: 哪些车已经称重。
- `spot_assignments`: 大库精确股位占用。
- `loco_carry`: 机车当前是否还挂着车。

所以 stage2 不是从原始输入开始，stage3 也不是从原始输入开始。它们都站在上一阶段真实执行结果上继续算。

这点对理解问题非常重要：如果 stage3 卡住，原因可能是 stage3 自己目标难，也可能是 stage1/stage2 给它留下的现场形态不好。

## 8. 为什么看到的执行阶段可能不止 3 个

从 workflow payload 看，前 3 个阶段是：

1. stage1
2. stage2
3. stage3

但实际 runner 里，如果某个 stage 带了 wave plans，它会拆开执行。

stage1 当前有 `phase1WavePlans`，所以你在运行结果里可能看到多个同名：

```text
phase1_pre_repair_buffering
phase1_pre_repair_buffering
phase1_pre_repair_buffering
phase2_depot_area_marshalling
phase3_ji_to_depot_allocation
...
```

这不是重复 bug，而是 stage1 的 3 个波次。

理解方式：

- 业务阶段还是 stage1。
- 执行层面拆成 wave A/B/C。
- runner 最后会把多个 wave 的 hook 和 step 合并成一个阶段结果。

## 9. 调试时最该看的字段

### 9.1 stage1 看这些

在 payload 的：

```text
workflowStages[0].stagePolicy.phase1Diagnostics
```

重点看：

- 选中了哪些包。
- 哪些车被放进缓冲线。
- 哪些车被延后。
- 每个来源线的角色和优先级。
- 每个 wave 增加了哪些 block。
- `packageAssignments` / `layoutAssignments`: 车到缓冲线的分配。
- `packageTargetRanks` / `layoutTargetRanks`: 同一目标内的顺序。

如果 stage3 后面很难，先看 stage1 是否形成了合理的主骨架。

### 9.2 stage2 看这些

在 payload 的：

```text
workflowStages[1].stagePolicy.phase2Diagnostics
workflowStages[1].stagePolicy.phase2ExecutionPlan
```

重点看：

- `depotStayVehicles`: 哪些车留在大库。
- `fixedDepotResidentVehicles`: 哪些车固定不动。
- `depotOutboundVehicles`: 哪些车必须从大库出去。
- `cun4FinalVehicles`: 哪些车最终去 `存4北`。
- `phase2OutboundGroups`: 大库出库组怎么合并。
- `phase2TrackLayers`: 每条线按前缀暴露形成了哪些层。
- `mustPullVehicleNos`: 必须拉的车。
- `predecessorUnlockVehicleNos`: 为了解锁后车而必须先拉的车。
- `phase3ClearanceVehicleNos`: 为 stage3 清路的车。
- `deferredTailVehicleNos`: 计划延后的尾部车。
- `collectionBatches`: 最终按约束拆出来的转移批次。

运行后还要看 view diagnostics 里的 `phase2Runtime`，它能告诉你 runtime 重建后实际执行了多少、延后了多少、延后原因是什么。

### 9.3 stage3 看这些

在 payload 的：

```text
workflowStages[2].vehicleGoals
```

重点看：

- 入库车是否恢复到了原始最终目标。
- 精确股位 `isSpotting` 是否恢复。
- 非入库车是否是 `PHASE3_DYNAMIC_CURRENT_HOLD`。
- 固定大库驻留车是否还是 `FIXED_DEPOT_RESIDENT`。

如果 stage3 失败，还要结合 solver 的 verifier errors、失败 hook、最终 track_sequences 一起看。

## 10. 新手读代码的推荐顺序

建议不要一上来钻进所有 helper。按这个顺序读会顺很多：

1. `solve_workflow()`  
   先理解 workflow 是怎么逐阶段执行和继承现场状态的。

2. `build_l7_closed_topology_workflow_payload()`  
   看 L7 模式到底生成了哪几个阶段，每阶段的 `stagePolicy` 和 `vehicleGoals` 是什么。

3. `_build_vehicle_stage_facts()`  
   看每辆车被贴了哪些业务标签。

4. `_build_phase1_plan()`  
   看 stage1 如何选主骨架、缓冲线和 wave。

5. `_phase1_goal()` / `_phase1_wave_goal()`  
   看 stage1 最终给每辆车设置了什么阶段目标。

6. `_build_stage2_plan()`  
   看大库车辆如何分成留存、出库、最终存4、固定驻留。

7. `_build_phase2_execution_plan()`  
   看 stage2 怎么选必须转移的车、怎么拆合法批次。

8. `_solve_phase2_execution_stage()`  
   看 stage2 运行时怎么根据真实现场挂车、释放障碍、集中摘到 `存4北`。

9. `_phase3_goal()`  
   看 stage3 为什么只恢复入库车最终目标。

## 11. 一句话总结

当前 stage1-3 的本质是一个“先整理入口、再清空大库压力、最后精确入库”的分阶段求解流程。

它不是 3 次独立求解，也不是简单把最终目标切成 3 份。真正关键的是：

- stage1 用机区缓冲线形成入库骨架。
- stage2 根据大库真实前缀顺序，把必须出库或必须让路的车合法批量转移到 `存4北`。
- stage3 只对入库车恢复最终精确目标，集中解决大库入位。
- runner 把每阶段执行后的真实现场传下去，所以前一阶段的选择会直接影响后一阶段难度。

理解了这条主线，再看具体代码时就不会迷路：所有复杂的包、层、波次、批次，本质上都是为了让这条业务链条在真实现场约束下可执行、少勾数、少修补。
