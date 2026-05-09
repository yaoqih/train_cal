# Structural Intent Frontloaded Planning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move work-position order repair, staging role assignment, route release, and capacity release into one frontloaded structural-planning layer so the solver avoids long tail repair loops.

**Architecture:** Add a single immutable `StructuralIntent` model that converts current state facts into group/resource/commitment decisions, then generate a small number of multi-hook `MoveCandidate` objects from that model. The candidates still compile to ordinary `HookAction` steps and are applied through the existing replay/verifier path, so search, UI, and validation stay hook-level.

**Tech Stack:** Python dataclasses, current `fzed_shunting` solver modules, existing `HookAction`, `ReplayState`, `RouteOracle`, structural metrics, route blockage, capacity release, pytest.

---

## 背景判断

当前求解器已经有三类有价值的事实模型：

- `structural_metrics.py` 能发现未完成、前端阻塞、目标序列缺陷、容量超长。
- `route_blockage.py` 能识别通路被哪些股道/车辆阻塞。
- `capacity_release.py` 能识别目标股道容量压力和应优先释放的前端车辆。

问题在于这些事实大多只进入评分、beam 偏好或 `astar_solver.py` 尾部补全。尾部补全能救一些样例，但它太晚：前面已经可能把非 SPOTTING 同目标车塞进调棚，把临停当自由池反复倒车，或者把已经正确的目标车组又拿出来。此时再靠尾部修，会产生大量 staging-to-staging、重复触碰和超长勾数。

下一阶段不应继续新增 tail fallback，也不应恢复旧的 `work_position_sequence` 专用候选层。正确方向是建立一个统一的结构意图层：在动作生成阶段就决定哪些车组应延迟承诺、哪些目标结构应保护、哪些资源必须先打开、哪个临停承担什么角色。

## 设计原则

1. 一个结构模型，不要两套规划器。
   `StructuralIntent` 是唯一的新结构事实入口。work-position、route blockage、capacity release、staging role、commitment 都从这里读，后续候选生成也只从这里产生结构候选。

2. 宏动作只是候选组织，不是绕过物理规则。
   所有结构候选最终都是 `MoveCandidate.steps: tuple[HookAction, ...]`。每一步仍由现有 `_apply_move`、route oracle、capacity preview、work-position preview 校验。

3. 目标序列用偏序和窗口，不用唯一硬编码序列。
   调棚/洗南/油/抛的 SPOTTING 是 rank window；FREE、SNAPSHOT、普通 TRACK 是弹性或可保留车。模型只表达必须满足的顺序关系和禁止提前承诺的关系。

4. 冻结是带原因的保护，不是绝对禁止移动。
   已经满足目标结构的连续车组应默认保护；只有明确资源债务可以打破，比如 route access、capacity release、front clearance。每次破坏都要在候选理由里记录。

5. 临停是带角色的短生命周期资源。
   临停不能只是“哪里能放就放哪里”。候选应记录 `ORDER_BUFFER`、`ROUTE_RELEASE`、`CAPACITY_RELEASE`、`SOURCE_REMAINDER` 等角色，并保证同一候选中不把同一缓冲线重复用于互相冲突的用途。

6. 普通一钩动作继续存在。
   结构候选用于跨越短期看似退步但长期正确的局部重排。简单问题仍由 primitive moves 解决，避免为了宏动作牺牲小样例效率。

## 核心抽象

### StructuralIntent

建议创建 `src/fzed_shunting/solver/structural_intent.py`。

```python
@dataclass(frozen=True)
class StructuralIntent:
    committed_blocks_by_track: dict[str, tuple[CommittedBlock, ...]]
    order_debts_by_track: dict[str, OrderDebt]
    resource_debts: tuple[ResourceDebt, ...]
    staging_buffers: tuple[StagingBuffer, ...]
    delayed_commitments: tuple[DelayedCommitment, ...]
```

含义：

- `committed_blocks_by_track`: 当前已经形成合法最终结构、应保护的连续块。
- `order_debts_by_track`: 调棚/洗南/油/抛等作业位线的顺序窗口缺口、阻塞块、候选插入块。
- `resource_debts`: route、capacity、front-access 等必须先打开的资源债。
- `staging_buffers`: 当前可用临停资源及适合角色。
- `delayed_commitments`: 目标股道正确但当前不该立刻落位的车，比如会压坏 SPOTTING 窗口的同目标非 SPOTTING 车。

### StructuralCandidate

内部候选可以独立建模，但输出只允许转成 `MoveCandidate`：

```python
@dataclass(frozen=True)
class StructuralCandidate:
    kind: str
    reason: str
    focus_tracks: tuple[str, ...]
    required_buffers: tuple[BufferReservation, ...]
    steps: tuple[HookAction, ...]
```

`StructuralCandidate` 不进入搜索层；`generate_move_candidates()` 只返回 `MoveCandidate`，避免扩散类型系统。

## 候选类型

### 1. Work-Position Window Repair

业务目标：一次修复某条作业位线的 rank window，而不是一辆一辆尾部补。

候选形态：

1. 识别目标线当前窗口。
2. 找出必须临时清走的北端 prefix。
3. 找出应先插入的 SPOTTING/EXACT block。
4. 把同目标但会压坏窗口的 FREE/SNAPSHOT 车标记为 `ORDER_BUFFER` 或 delayed。
5. 编译为：清 prefix -> 暂存 -> 取目标块 -> 插入目标线 -> 恢复或延后 prefix。

必须约束：

- 不生成会让 `preview_work_positions_after_prepend()` hard invalid 的候选。
- 不把已满足且无资源债务的 committed block 拿出来。
- 多辆连续 SPOTTING 能成组处理，避免重复清同一前缀。

### 2. Resource Release Candidate

业务目标：先打开会改变可行性的资源，不在尾部才发现路不通或容量不够。

资源债来源：

- route blocker: 来自 `compute_route_blockage_plan()`。
- capacity pressure: 来自 `compute_capacity_release_plan()`。
- front blocker: 来自 `compute_structural_metrics()`。

候选形态：

- 取 blocking track 的必要前缀；
- 按债务类型选择 staging role；
- 优先放到不阻塞后续 work-position/route 的缓冲线；
- 如果 blocker 本身已有明确目标且可直接送达，优先直接送目标，不临停。

### 3. Commitment-Preserving Primitive Filtering

业务目标：普通一钩动作不能持续破坏已经正确的目标结构。

实现方式：

- 不直接删除 primitive moves，先给它们增加结构破坏惩罚或降低排序；
- 如果动作会移动 committed block，必须命中某个 `ResourceDebt.required_vehicle_nos`；
- debug_stats 记录被保护机制降权或拒绝的候选数量。

这一层很关键：只增加宏动作但仍允许普通动作随便破坏目标结构，尾部高勾数仍会出现。

## 搜索融合方式

`generate_move_candidates()` 应改为：

1. 生成 primitive candidates。
2. 构造 `StructuralIntent`。
3. 从 intent 生成少量 structural candidates。
4. 对 primitive 做 commitment-aware 排序/过滤。
5. 去重后返回同一个 `list[MoveCandidate]`。

搜索层不用知道具体业务 planner。`search.py` 已经支持多步候选：

- `_apply_candidate_steps()` 逐步应用；
- 成本按真实 hook 数计；
- `_evaluate_candidate_steps()` 会累计 blocker bonus 和 route release 影响。

下一阶段只需要补一个轻量结构保留机制：当某条 work track 存在 order debt 时，beam 至少保留一个该 track 的最佳 structural candidate 结果。不要恢复旧的 work-position 专用 reserve metadata；应使用通用 `focus_tracks` 和 `structural_reserve`。

## 尾部逻辑迁移边界

短期允许保留 `astar_solver.py` 现有 tail completion 作为安全网，但新逻辑不得继续往尾部加。

迁移顺序：

1. 先把 shared legality helpers 抽到 `candidate_compiler.py`。
   例如 exact attach、exact detach、staging detach、candidate replay。
2. `StructuralIntent` 候选和旧 tail 都调用这些 helper。
3. 当 full validation 显示结构候选覆盖主要 work-position tail case 后，删除对应 tail 分支。
4. 每删除一段 tail，必须有替代候选测试和 validation hard case 覆盖。

不允许出现：

- `move_candidates.py` 自己实现一套 detach 合法性；
- `astar_solver.py` 又保留另一套 work-position special planner；
- route/capacity/work-position 三套候选各自选择 staging。

## 自审和修改结论

本节是对初稿方案的二次审查结果。审查后做了三处收敛：第一，冻结从硬约束改成带债务豁免的保护；第二，结构候选只作为 `MoveCandidate` 输入搜索，不新增独立搜索器；第三，尾部逻辑不再扩展，只允许把通用合法性 helper 抽出复用。

### 问题 1：会不会变成又一个大规划器？

风险真实存在。控制办法是：`StructuralIntent` 只做事实归一和候选生成，不做全局搜索；它最多生成每类前 N 个候选。全局选择仍交给已有 beam/A*。

### 问题 2：冻结会不会导致不可解？

会，如果做成硬约束。因此本方案改为“默认保护 + 资源债豁免”。任何移动 committed block 的候选必须说明是为了解决哪一个 route/capacity/front/order debt。

### 问题 3：宏动作会不会隐藏非法中间态？

不会，只要所有宏动作都编译成 `HookAction` 并逐步 replay。测试必须验证每个 structural candidate 的每一步都能 `_apply_move`，不能只验证最终状态。

### 问题 4：会不会牺牲简单案例速度？

不会作为默认假设；实现上 structural candidate 数量要小，并且只在 intent 有结构债时生成。没有 order/resource debt 的状态只返回 primitive moves。

### 问题 5：会不会过拟合调棚？

如果只写调棚规则会过拟合。因此模型必须基于 `work_position_kind` 和 allowed rank window，不基于具体车号或具体文件。调棚、洗南、油、抛都走同一套 rank-window 逻辑。

### 问题 6：如何证明它解决尾部大量修补？

不能只看可解率。必须同时看：

- `hook_count` p50/p90/max；
- `staging_to_staging_hook_count`；
- `max_vehicle_touch_count`；
- `target_sequence_defect_count` 在搜索前中期是否下降；
- tail fallback 命中次数是否下降；
- hard cases 是否在普通搜索阶段完成，而不是 tail 完成。

### 修改后排除的做法

以下做法不进入下一阶段：

- 不恢复旧的 `generate_work_position_sequence_candidates()` 专用 planner。
- 不在 `astar_solver.py` 继续增加 work-position tail special case。
- 不把调棚写成车号级或样例级规则。
- 不把 committed block 做成不可移动硬约束。
- 不引入 CP-SAT/RL/新全局规划框架。
- 不让 structural candidate 绕过 `_apply_move` 或 verifier。

### 方案合理性的最终判断

修改后的方案是合适的。它不是继续给尾部补救加分支，而是把尾部高勾数的根因提前建模：顺序窗口、临停角色、资源释放、目标结构保护、延迟承诺。它保留现有 hook-level 物理模型和 verifier，只在动作生成层增加结构候选，因此能和当前代码自然融合，也能逐步替换尾部 repair。

这也是它比单纯“债务驱动”更强的地方：债务只说明哪里坏了，`StructuralIntent` 还说明哪些车应该一起动、哪些位置应保护、哪些临停被哪个业务目的占用，以及什么时候才允许最终落位。

## 实施计划

### Task 1: StructuralIntent 数据模型和基础事实归一

**Files:**
- Create: `src/fzed_shunting/solver/structural_intent.py`
- Create: `tests/solver/test_structural_intent.py`
- Modify: none initially

**Step 1: 写失败测试**

测试内容：

- route blocker 和 capacity blocker 同时出现在 `StructuralIntent.resource_debts`。
- SPOTTING rank window 缺口生成 `OrderDebt`。
- 已满足目标的连续块生成 `CommittedBlock`。

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_structural_intent.py
```

Expected: fail because module does not exist.

**Step 2: 实现最小模型**

实现只读 dataclass 和 `build_structural_intent(plan_input, state, route_oracle)`：

- 调 `compute_structural_metrics()`；
- 调 `compute_route_blockage_plan()`；
- 调 `compute_capacity_release_plan()`；
- 识别 work-position tracks 的 window debt；
- 识别简单 committed block。

**Step 3: 验证**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_structural_intent.py tests/solver/test_structural_metrics.py tests/solver/test_capacity_release.py tests/solver/test_route_blockage.py
```

### Task 2: Commitment 保护规则

**Files:**
- Modify: `src/fzed_shunting/solver/structural_intent.py`
- Modify: `src/fzed_shunting/solver/move_candidates.py`
- Test: `tests/solver/test_move_generator.py`
- Test: `tests/solver/test_structural_intent.py`

**Step 1: 写失败测试**

测试内容：

- 一个已满足 work-position/fixed target 的 committed block，普通 staging churn 不应作为高优先候选出现。
- 如果同一 block 是 route/capacity/front debt 的 required block，则允许移动。

**Step 2: 实现 primitive candidate 结构降权/过滤**

在 `generate_move_candidates()` 中：

- 构造 intent；
- 对 primitive move 判断是否破坏 committed block；
- 无资源债豁免时标记为 low priority 或过滤；
- debug_stats 增加 `protected_primitive_rejected_count` 和 `protected_primitive_allowed_by_debt_count`。

**Step 3: 验证**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_move_generator.py tests/solver/test_structural_intent.py
```

### Task 3: Work-Position Window Structural Candidate

**Files:**
- Create: `src/fzed_shunting/solver/candidate_compiler.py`
- Modify: `src/fzed_shunting/solver/structural_intent.py`
- Modify: `src/fzed_shunting/solver/move_candidates.py`
- Test: `tests/solver/test_structural_intent.py`
- Test: `tests/solver/test_move_generator.py`

**Step 1: 写失败测试**

测试内容：

- 同目标非 SPOTTING 车在插入 SPOTTING 前不被送入目标线。
- 对同一 target track，一次候选能处理连续 SPOTTING block。
- structural candidate 的所有 steps 可以逐步 `_apply_move`。

**Step 2: 抽 shared compiler**

从 `astar_solver.py` 尾部 helper 中抽出小而通用的能力：

- exact prefix attach；
- exact block detach；
- staging detach candidate；
- candidate replay/rank。

不要一次搬完整 tail planner，只抽合法性和编译原语。

**Step 3: 生成 window repair candidates**

在 `generate_move_candidates()` 中：

- 当 intent 有 `OrderDebt` 时生成最多 1-3 个 structural candidates；
- 使用 `ORDER_BUFFER` 预留 staging；
- 输出 `MoveCandidate(kind="structural", reason="work_position_window_repair", focus_tracks=(target_track,), structural_reserve=True)`。

**Step 4: 验证**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_structural_intent.py tests/solver/test_move_generator.py tests/solver/test_astar_solver.py
```

### Task 4: Resource Release Structural Candidate

**Files:**
- Modify: `src/fzed_shunting/solver/structural_intent.py`
- Modify: `src/fzed_shunting/solver/move_candidates.py`
- Test: `tests/solver/test_structural_intent.py`
- Test: `tests/solver/test_move_generator.py`
- Test: `tests/solver/test_capacity_release.py`
- Test: `tests/solver/test_route_blockage.py`

**Step 1: 写失败测试**

测试内容：

- route blocker 车组优先直接去目标；不能直接去目标时才临停。
- capacity release 的 front_release_vehicle_nos 被生成候选。
- 同一候选中同一 staging buffer 不能同时作为 `ORDER_BUFFER` 和 `ROUTE_RELEASE`。

**Step 2: 实现资源候选**

实现顺序：

1. route release candidate；
2. capacity release candidate；
3. front blocker candidate。

每类最多生成少量候选，并共享 staging role scorer。

**Step 3: 验证**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_structural_intent.py tests/solver/test_move_generator.py tests/solver/test_capacity_release.py tests/solver/test_route_blockage.py
```

### Task 5: Beam 结构保留和尾部降级观测

**Files:**
- Modify: `src/fzed_shunting/solver/search.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Test: `tests/solver/test_astar_solver.py`
- Test: `tests/solver/test_validation_recovery.py`

**Step 1: 写失败测试**

测试内容：

- 有 order debt 的 work track 至少保留一个 `structural_reserve` candidate 后继状态。
- tail fallback debug_stats 能统计是否被调用。

**Step 2: 实现通用 reserve**

不要恢复旧的 work-position 专用字段。使用：

- `candidate.structural_reserve`
- `candidate.focus_tracks`
- beam prune 时按 focus track 保留少量结构状态。

**Step 3: 加观测**

debug_stats 增加：

- `structural_intent_candidate_count`
- `structural_candidate_steps_total`
- `structural_reserve_kept_count`
- `tail_completion_invocation_count`

**Step 4: 验证**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/solver/test_astar_solver.py tests/solver/test_validation_recovery.py
```

### Task 6: Hard Cases 和全量分布验证

**Files:**
- Modify only if runner needs extra summary fields: `scripts/run_external_validation_parallel.py`
- Test: existing validation suites
- Artifacts: `artifacts/validation_inputs_positive_structural_intent_*`, `artifacts/validation_inputs_truth_structural_intent_*`

**Step 1: 先跑硬案例**

至少包括：

- `data/validation_inputs/online/validation_20260403Z.json`
- `data/validation_inputs/online/validation_20260401Z.json`
- 之前 positive/truth 不可解或高勾数案例。

**Step 2: 跑 selected tests**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/solver/test_move_generator.py \
  tests/solver/test_astar_solver.py \
  tests/solver/test_external_validation_parallel_runs.py \
  tests/solver/test_validation_recovery.py \
  tests/solver/test_structural_metrics.py \
  tests/solver/test_capacity_release.py \
  tests/solver/test_route_blockage.py \
  tests/solver/test_goal_logic.py
```

**Step 3: 跑 positive/truth 分布**

输出指标：

- solved / total；
- solve time p50/p90/max；
- hook count p50/p90/max；
- bucketed hook distribution；
- `staging_to_staging_hook_count` p90/max；
- `max_vehicle_touch_count` p90/max；
- tail fallback invocation count。

**Step 4: 通过门槛**

不以“某个样例能解”为门槛。通过标准：

- 可解性不低于 rollback 前；
- truth/positive 高勾数尾部明显下降；
- tail fallback 命中率下降；
- 没有新增 verifier/replay 不合法方案；
- 没有出现 `move_candidates.py` 和 `astar_solver.py` 两套 work-position planner。

## 最终收敛目标

阶段完成后，求解器应形成如下结构：

```text
current state
  -> StructuralIntent
  -> primitive candidates + structural candidates
  -> one search queue
  -> HookAction replay / verifier
  -> minimal tail safety net
```

进一步稳定后，再删除被 structural candidates 覆盖的 tail completion 分支，留下的 tail 只处理预算耗尽时的 partial recovery，不再承担主要求解职责。
