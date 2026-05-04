# Work Track Rank Position Rules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 硬切换调棚、洗南、油、抛的对位规则：彻底移除旧固定作业台位码模型，改为基于最终线路序列的北端/南端序位约束。

**Architecture:** 保留现有北到南 `track_sequences` 和北端 prepend 模型，把作业线对位判断改成 `north_rank` / `south_rank` 的纯函数。归一化后的工作线目标使用独立 `target_mode="WORK_POSITION"`，不再复用 `AREA` / `SPOT`；大库、库内精确台位、机库称重继续使用 `spot_assignments`。作业线完全退出 `WORK_AREA_SPOTS`、固定 group capacity 和旧 `调棚:1` / `洗南:1` spot code 体系。

**Tech Stack:** Python, Pydantic `GoalSpec`, pytest, existing `fzed_shunting` solver/replay/verifier stack.

---

## 1. 总体改造边界

这次不是简单改 `WORK_AREA_SPOTS`。

旧模型：

```text
调棚:WORK -> 调棚:1-4
洗南:WORK -> 洗南:1-3
油:WORK -> 油:1-2
抛:WORK -> 抛:1-2
```

新模型：

```text
线路最终序列 = [北端车辆, ..., 南端车辆]
north_rank = index + 1
south_rank = len(seq) - index
```

| 线路 | `isSpotting=是` | `isSpotting=否/空` | 具体台位 |
|---|---|---|---|
| 抛 / 油 | `south_rank in {1,2}` | 任意序位，长度放得下 | `north_rank == N` |
| 洗南 | `south_rank in {2,3,4}` | 任意序位，长度放得下 | `north_rank == N` |
| 调棚 | `south_rank in {3,4,5,6}` | 任意序位，长度放得下 | `north_rank == N` |

长度容量继续使用：

```text
sum(vehicleLength on track) <= trackInfo.trackDistance
```

这条容量链路当前已经存在于 `move_generator.py`、`goal_logic.py`、`plan_verifier.py` 等模块，后续要复用，不要重新引入固定台位总数 `T`。

本计划是需求硬切换，不保留作业线旧模型的向后兼容路径。实现完成后：

```text
工作线对位事实源 = 最终 track_sequences
工作线容量事实源 = 线路长度 + 车辆长度累计
工作线不得再产生或消费固定 spot code
工作线目标不得再用 target_area_code 表达
```

旧 `调棚:WORK`、`洗南:WORK`、`油:WORK`、`抛:WORK` 只能作为迁移前的错误来源被清理，不能作为新求解分支继续存在。

---

## 2. 修改后的求解问题：算法原理

这次规则变化的本质，是把作业线从“固定台位分配问题”改成“最终线路序列构造问题”。

旧模型把调棚、洗南、油、抛都当成一组固定格子：

```text
车辆 -> 分配到某个 spot code
例如：洗南:1、洗南:2、洗南:3
```

新模型不再关心这些局部编号。作业线真正可验证的事实只有一条：

```text
最终线路序列 = [北端车辆, ..., 南端车辆]
```

所有作业线台位语义都从这个序列动态推导：

```text
north_rank = 车辆在最终序列中从北往南数第几辆
south_rank = 车辆在最终序列中从南往北数第几辆
```

因此，作业线不应该再产生 `调棚:1`、`洗南:2` 这类持久 spot assignment。它们不是独立状态，只是 `track_sequences` 的派生属性。

### 2.1 状态表达：只保留真实状态，不保留派生台位

求解状态里应该继续保留：

```text
track_sequences[track] = [北端车辆, ..., 南端车辆]
loco_carry = [靠机车内侧, ..., 靠外侧]
spot_assignments = 真实固定台位占用
```

其中 `spot_assignments` 只适合表达真实固定资源：

```text
大库 n01-n05/n07
机库:WEIGH
```

作业线 rank 不应写入 `spot_assignments`，原因有三个：

1. `north_rank` / `south_rank` 由最终序列唯一决定，重复存一份会出现双真源。
2. 作业线后续从北端 prepend 车辆时，`north_rank` 会变化；如果提前写死 spot code，状态会变脏。
3. 每次求解都会重新读取现场状态，不需要让某个北侧台位号跨求解稳定。

所以新模型下的状态判断应该是：

```text
final_seq = state.track_sequences[target_track]
rank = compute_rank(final_seq, vehicle_no)
```

而不是：

```text
state.spot_assignments[vehicle_no] == "洗南:2"
```

### 2.2 动作语义：作业线是北端插入、南端名次自然形成

当前挂车 / 摘车顺序模型应保持：

```text
ATTACH:
  从源线路北端取 prefix
  追加到 loco_carry 的尾部

DETACH:
  从 loco_carry 尾部摘出一段
  prepend 到目标线路北端
```

对目标作业线来说，一钩摘入后的新序列是：

```text
new_seq = incoming_block + existing_seq
```

这条公式非常关键。它说明：

- `incoming_block` 里的车会在原有车辆北边。
- `existing_seq` 里的车全部在新放入车辆南边。
- 同一钩 `incoming_block` 内部的相对顺序，也会决定谁在谁北边。

例如空洗南要求目标车在南边第 2 辆：

```text
incoming_block = [TARGET]
existing_seq = []
new_seq = [TARGET]
TARGET south_rank = 1 -> 不满足
```

但同一钩带一个南侧垫车时：

```text
incoming_block = [TARGET, FILLER]
existing_seq = []
new_seq = [TARGET, FILLER]
TARGET south_rank = 2 -> 满足
```

如果线路上原来已经有南侧车辆：

```text
incoming_block = [TARGET]
existing_seq = [OLD_SOUTH]
new_seq = [TARGET, OLD_SOUTH]
TARGET south_rank = 2 -> 满足
```

因此动作生成不能只看“目标车单独放进去是否满足”，还要看同一钩 block 和已有线路车辆共同形成的最终序列。

### 2.3 单调性：南端名次和北端名次的变化方向不同

在作业线只从北端 prepend 的前提下，有一个非常有用的单调性：

```text
new_seq = prepended_cars + old_seq
```

对 `old_seq` 里已经存在的车辆：

```text
south_rank 不变
north_rank 增加 len(prepended_cars)
```

举例：

```text
old_seq = [A, TARGET, B]
TARGET north_rank = 2
TARGET south_rank = 2

prepend [X, Y]

new_seq = [X, Y, A, TARGET, B]
TARGET north_rank = 4
TARGET south_rank = 2
```

这个单调性直接决定剪枝策略。

`isSpotting=是` 是南端名次约束：

```text
抛 / 油: south_rank in {1,2}
洗南: south_rank in {2,3,4}
调棚: south_rank in {3,4,5,6}
```

只要车辆已经进入目标作业线，后续从北端继续放车不会改变它的 `south_rank`。所以如果一辆 `SPOTTING` 车刚放入目标线时 `south_rank` 错了，后续单靠北端 prepend 不能修好；除非把它再搬走重排。

因此：

```text
SPOTTING 放入目标作业线时，可以硬判断 south_rank。
明显错位的候选动作可以直接剪掉。
```

具体台位是北端名次约束：

```text
target_rank = N
要求最终 north_rank == N
```

但 `north_rank` 会随着后续北端 prepend 增大。因此具体台位不能简单要求“刚放进去就等于 N”。

例如目标最终要北端第 3：

```text
先放 TARGET:
  [TARGET]
  north_rank = 1

后续北端补入 [A, B]:
  [A, B, TARGET]
  north_rank = 3 -> 最终满足
```

所以：

```text
EXACT_NORTH_RANK 是最终状态硬约束，不是每次 DETACH 后的即时硬约束。
动作生成可以记录 rank_gap 并用于排序，但必须尊重单调性：
  north_rank < target_rank -> 可通过后续 prepend 修复，保留。
  north_rank == target_rank -> 当前满足，保留。
  north_rank > target_rank -> 已经过位，作为目标工作线候选拒绝。
```

### 2.4 三类输入对应三类约束

归一化后的作业线目标统一使用：

```text
target_mode = "WORK_POSITION"
target_track = 具体工作线
allowed_target_tracks = [具体工作线]
target_area_code = None
target_spot_code = None
work_position_kind = "FREE" | "SPOTTING" | "EXACT_NORTH_RANK"
target_rank = 具体北端序位或 None
```

`target_mode="WORK_POSITION"` 的目的不是增加一层复杂度，而是彻底阻断旧 `AREA` / `SPOT` 逻辑继续消费工作线目标。所有求解层只需要判断：

```python
if vehicle.goal.work_position_kind is not None:
    ...
```

作业线目标最终可以统一成三类约束：

```text
FREE:
  只要求车辆最终在线路上。
  不要求 north_rank / south_rank。
  只受线路长度容量约束。

SPOTTING:
  要求车辆最终在线路上。
  要求 south_rank 落在对应窗口。
  因为 south_rank 对后续北端 prepend 不变，放入时就应尽量硬判断。

EXACT_NORTH_RANK:
  要求车辆最终在线路上。
  要求最终 north_rank == target_rank。
  当前 north_rank 可以暂时小于 target_rank，等待后续 prepend 修复。
  当前 north_rank 不能大于 target_rank；一旦过位，继续 prepend 只会更错。
  最终 verifier 必须严格判断 north_rank == target_rank。
```

这三个约束都可以只依赖最终序列判断：

```python
def work_position_satisfied(seq, vehicle_no, kind, target_rank=None):
    if vehicle_no not in seq:
        return False
    if kind == "FREE":
        return True
    if kind == "SPOTTING":
        return south_rank(seq, vehicle_no) in allowed_south_ranks(track_code)
    if kind == "EXACT_NORTH_RANK":
        return north_rank(seq, vehicle_no) == target_rank
```

这比旧的固定 spot allocation 更干净：输入归一化只负责把业务字段转成约束类型，求解器只负责构造最终序列。

### 2.5 容量约束：长度是硬约束，台位总数不是事实源

新规则下不需要确认每条作业线一共有多少个固定台位 `T`。原因是：

1. 现场真正限制车辆能否放入的是线路有效长度。
2. 作业台位窗口只描述“目标车最终相对南端/北端的位置”，不是整条线最多只能放几辆。
3. `否/空` 允许放在对位位置、对位北边、对位南边的部分范围，本质上已经不是固定格子容量。

所以容量判断应该始终是：

```text
sum(vehicleLength on target track) <= target_track.effective_length
```

旧的固定计数限制：

```text
调棚:WORK <= 4
洗南:WORK <= 3
油:WORK <= 2
抛:WORK <= 2
```

不能再作为工作线放车的硬约束。它最多只能作为旧数据解释，不应参与新求解。

### 2.6 候选动作生成：从“找空台位”改成“构造可行序列”

旧动作生成大致是：

```text
选择一段 block
选择目标线
检查固定 group 是否有空位
分配 spot code
```

新动作生成应改成：

```text
选择一段 block
选择目标线
检查长度容量
预演 new_seq = block + existing_seq
对 block 内有作业线目标的车，检查序位约束
```

对 `SPOTTING`，应做面向低勾数的强过滤：

```text
如果 TARGET 放入后 south_rank 不在允许窗口，拒绝这个候选。
```

但这个判断必须基于完整 `new_seq`，而不是只看目标车：

```text
south_count = len(new_seq) - index(TARGET) - 1
south_rank = south_count + 1
```

这条强过滤的含义是：不要把目标车先放进目标工作线的错误位置，再期待后续搬走重排。那种动作通常会增加勾数，也会让搜索陷入“先制造错位、再修复错位”的低质量分支。若极端场景确实需要临时缓存，应由统一临停机制选择其他可用缓存线解决，而不是让目标工作线兼任错误台位缓存。

对 `EXACT_NORTH_RANK`，不能简单要求“当前就满足”，但可以利用单调性做安全剪枝：

```text
当前 north_rank < target_rank:
  后续还可以从北端 prepend 车辆把它推到更南，可能变成满足。

当前 north_rank == target_rank:
  当前满足，但后续如果继续 prepend 到同线，会被破坏，需要最终再核。

当前 north_rank > target_rank:
  已经太靠南，单靠继续 prepend 只会更靠南。该目标工作线候选应拒绝。
```

因此对具体台位应同时使用安全剪枝和排序评分：

```text
rank_gap = target_rank - current_north_rank

rank_gap > 0:
  还缺北侧补车，可保留。

rank_gap == 0:
  当前好状态，应加分，但不能永久锁死。

rank_gap < 0:
  已经太靠南，单靠后续 prepend 无法修好；作为目标工作线候选应拒绝。
```

这个评分不是为了保留旧兼容逻辑，而是为了低勾数：优先选择能直接构造最终序列的动作，尽量减少“放错再搬”的反复操作。

### 2.7 启发式：奖励正确序列，避免旧 spot 冲突思维

搜索评分应该从旧的“台位冲突”转向“序列接近最终目标”。

启发式应服务两个目标：

```text
1. 可解性：不要把可能通过后续北端 prepend 修好的 EXACT_NORTH_RANK 分支剪掉。
2. 低勾数：强烈偏好一次性构造正确最终序列，避免先错放再重排。
```

必须直接引入基础 rank 启发式：

```text
SPOTTING:
  放入后 south_rank 在窗口内 -> 明显奖励
  放入后 south_rank 不在窗口内 -> 不生成目标工作线候选

EXACT_NORTH_RANK:
  当前 north_rank == target_rank -> 奖励
  current_north_rank < target_rank -> 小惩罚，表示还需要北侧补车
  current_north_rank > target_rank -> 不生成目标工作线候选

FREE:
  不产生 rank 奖惩，只看是否进入目标线和容量。
```

这里要避免两个旧思维：

1. 不要让工作线具体台位进入 depot exact spot 冲突消解。
2. 不要因为工作线 `spot_assignments` 为空就认为没有满足目标。

最终“是否满足”的唯一标准，应由 `goal_is_satisfied(...)` 从最终 `track_sequences` 计算。

### 2.8 搜索难度变化：更正确，但复杂点转移了

修改后不是单纯更难，也不是单纯更简单，而是难点换了位置。

变简单的部分：

```text
不再维护作业线固定 spot code。
不再处理伪台位占用冲突。
否/空不再被旧 capacity=2/3/4 错误卡死。
最终校验可以统一回到序列事实。
```

变难的部分：

```text
SPOTTING 可能需要同钩带垫车，动作生成要看完整 block。
EXACT_NORTH_RANK 是最终序列约束，不能只看单步局部满足。
搜索需要理解后续 prepend 会改变 north_rank。
```

总体判断：

```text
求解空间会稍微变大，因为 FREE 和具体北端序位不再被固定格子提前锁死。
但错误剪枝会减少，可行性会更接近真实业务。
对于生产场景，这是更好的模型：状态更少、真源更少、验证更直接。
```

换句话说，算法实现上需要更认真处理序列；但业务建模上更简单、更通用，也更不容易被测试样例外的真实场景击穿。

对生产可解性和低勾数的要求，应落实成搜索策略：

```text
1. 候选 block 以连续车组为单位，优先一钩完成目标车和必要垫车的共同放置。
2. SPOTTING 目标车不生成错位入目标工作线的候选，减少无效回搬。
3. EXACT_NORTH_RANK 保留可通过后续 prepend 修复的分支，但用 rank_gap 排序。
4. FREE 车辆不制造伪台位冲突，只参与长度容量和阻挡关系。
5. 最终目标函数仍以勾数优先，再考虑路径/道岔/移动成本。
```

还有一个重要细节：向目标工作线 prepend 新 block 时，不能只检查新 block 里的车。目标线上原有车辆的 `south_rank` 不变，但 `north_rank` 会增加，因此已有 `EXACT_NORTH_RANK` 车辆可能被新 block 推过目标位置。

动作预演应检查：

```text
incoming_block 内的 WORK_POSITION 车辆
existing_seq 内已有的 WORK_POSITION 车辆
```

其中：

```text
SPOTTING:
  检查 final south_rank 是否落窗。

EXACT_NORTH_RANK:
  final north_rank > target_rank -> hard violation
  final north_rank == target_rank -> 当前满足，排序靠前
  final north_rank < target_rank -> 仍可通过后续 prepend 修复，保留并排序靠后
```

### 2.9 最终校验：只相信最终序列，不相信中间标记

verifier 应在计划回放结束后重新计算：

```text
1. 每条线路长度是否超限。
2. 每辆 FREE 作业线目标车是否在目标线。
3. 每辆 SPOTTING 作业线目标车 south_rank 是否落入窗口。
4. 每辆 EXACT_NORTH_RANK 目标车 north_rank 是否等于 target_rank。
5. depot 固定台位和机库称重位是否满足原固定资源规则。
```

错误信息也应该从旧的“spot wrong”改成可读 rank：

```text
Vehicle 123 on 洗南 has south_rank=1, expected one of {2,3,4}
Vehicle 456 on 调棚 has north_rank=2, expected 4
```

这样现场排查时可以直接看到“差的是南边第几辆 / 北边第几辆”，而不是反查旧 spot code。

### 2.10 推荐的核心不变量

后续实现和测试都应围绕这些不变量：

```text
Invariant 1:
  track_sequences 始终按北到南存储。

Invariant 2:
  DETACH 到作业线始终是 prepend 到北端。

Invariant 3:
  工作线 rank 始终从 track_sequences 动态计算，不写入 spot_assignments。

Invariant 4:
  SPOTTING 看 south_rank，具体台位看 north_rank。

Invariant 5:
  作业线容量看车辆长度累计，不看固定台位总数 T。

Invariant 6:
  SPOTTING 错位入目标工作线不生成候选；EXACT_NORTH_RANK 若 north_rank > target_rank 也不生成候选，north_rank <= target_rank 时用 rank_gap 排序并最终校验。

Invariant 7:
  最终 verifier 重新计算所有 rank，不信任动作生成时的临时判断。
```

---

## 3. 分层修改清单

### 3.1 输入归一化层

**Files:**
- Modify: `src/fzed_shunting/io/normalize_input.py`
- Test: `tests/io/test_normalize_input.py`

**当前问题:**

- `isSpotting=是` 会变成 `调棚:WORK`、`洗南:WORK` 等固定作业区。
- `isSpotting=否/空` 在调棚会变成 `调棚:PRE_REPAIR`，不是自由放置。
- 工作线数字台位只在 `targetMode=SPOT` 的 `WORK_SPOT_BY_TARGET` 里支持，而且是局部 `1/2/3/4`。
- `isSpotting` 直接填数字时只支持 3 位大库台位，不支持工作线具体北端序位。

**必须修改:**

必须给 `GoalSpec` 增加作业线序位字段，彻底切断工作线和 depot `SPOT` 的语义复用：

```python
work_position_kind: str | None = None
# allowed values: "FREE", "SPOTTING", "EXACT_NORTH_RANK"
target_rank: int | None = None
```

同时约定：

```text
target_mode = "WORK_POSITION"
target_area_code = None
target_spot_code = None
```

工作线目标不要再归类为 `AREA`。如果仍用 `AREA`，`structural_metrics`、`telemetry`、`constructive`、`soft_target_template` 等二级路径会继续把它当作旧作业区目标处理。

归一化规则：

```text
targetTrack in {"调棚","洗南","油","抛"} and isSpotting == "是":
  target_mode = "WORK_POSITION"
  target_track = targetTrack
  allowed_target_tracks = [targetTrack]
  target_area_code = None
  target_spot_code = None
  work_position_kind = "SPOTTING"
  target_rank = None

targetTrack in {"调棚","洗南","油","抛"} and isSpotting in {"", "否"}:
  target_mode = "WORK_POSITION"
  target_track = targetTrack
  allowed_target_tracks = [targetTrack]
  target_area_code = None
  target_spot_code = None
  work_position_kind = "FREE"
  target_rank = None

targetTrack in {"调棚","洗南","油","抛"} and isSpotting is digit:
  target_mode = "WORK_POSITION"
  target_track = targetTrack
  allowed_target_tracks = [targetTrack]
  target_area_code = None
  target_spot_code = None
  work_position_kind = "EXACT_NORTH_RANK"
  target_rank = int(isSpotting)
```

`targetMode=SPOT` 且 `targetTrack` 是工作线时，也应转成 `work_position_kind="EXACT_NORTH_RANK"`，不要再生成 `target_spot_code="调棚:1"` 这种固定码。

工作线目标不再生成 `target_area_code="<track>:WORK"`、`WORK_FREE`、`WORK_EXACT` 这类伪 area code。新规则下，工作线目标的最小表达就是：

```text
target_track
work_position_kind
target_rank
```

这样后续搜索、校验、输出都不会误用旧 `AREA` spot candidate 逻辑。

显式输入也要硬切换：

```text
targetMode=AREA + targetAreaCode in {"调棚:WORK","调棚:PRE_REPAIR","洗南:WORK","油:WORK","抛:WORK"}:
  拒绝输入，提示旧工作线 area code 已废弃，必须使用 targetTrack + isSpotting / targetSpotCode。

targetMode=SPOT + targetTrack in {"调棚","洗南","油","抛"}:
  解释为 WORK_POSITION / EXACT_NORTH_RANK。

targetMode=SPOT + targetSpotCode like "调棚:2":
  拒绝输入，工作线具体台位只接受纯数字北端序位。
```

旧常量要同步清理：

```text
WORK_AREA_DEFAULTS:
  删除 调棚/洗南/油/抛 的固定作业区映射。
  保留 轮、大库、大库外、修N库内 等非本次工作线对位目标。

AREA_ALLOWED_TRACKS:
  删除 调棚:WORK、调棚:PRE_REPAIR、洗南:WORK、油:WORK、抛:WORK。
  保留大库/大库外/接车/存车/预修/洗罐等非本次固定台位组语义。

WORK_SPOT_BY_TARGET:
  删除整个工作线具体台位映射。
```

**测试重点:**

- `isSpotting=是` 的调棚、洗南、油、抛都得到 `SPOTTING`。
- `isSpotting=""` 和 `否` 都得到 `FREE`。
- 工作线 `isSpotting="3"` 得到 `EXACT_NORTH_RANK target_rank=3`。
- 工作线 `targetMode=SPOT targetSpotCode="3"` 得到同样结果。
- 上述工作线目标的 `target_mode == "WORK_POSITION"`，`target_area_code is None`，`target_spot_code is None`。
- 显式旧 `targetAreaCode="调棚:WORK"` / `targetAreaCode="洗南:WORK"` 被拒绝。
- 工作线 `targetSpotCode="调棚:2"` 被拒绝，`targetSpotCode="2"` 被接受为 `target_rank=2`。
- 工作线具体台位 `0`、负数、非数字应拒绝。
- 大库 `101/205` 仍走原 depot 精确台位逻辑。

Run:

```bash
pytest -q tests/io/test_normalize_input.py
```

---

### 3.2 作业线序位领域层

**Files:**
- Create: `src/fzed_shunting/domain/work_positions.py`
- Test: `tests/domain/test_work_positions.py`

**职责:**

把所有“南边第几辆 / 北边第几辆”的判断集中到一个领域模块，避免散落在搜索、回放和校验里。

必须提供的核心 API：

```python
WORK_POSITION_TRACKS = {"调棚", "洗南", "油", "抛"}

SPOTTING_SOUTH_RANKS = {
    "抛": frozenset({1, 2}),
    "油": frozenset({1, 2}),
    "洗南": frozenset({2, 3, 4}),
    "调棚": frozenset({3, 4, 5, 6}),
}

def is_work_position_track(track_code: str) -> bool: ...

def north_rank(seq: list[str], vehicle_no: str) -> int | None: ...

def south_rank(seq: list[str], vehicle_no: str) -> int | None: ...

def work_position_satisfied(vehicle, track_name: str, state) -> bool: ...

def preview_work_positions_after_prepend(
    *,
    target_track: str,
    incoming_vehicle_nos: list[str],
    existing_vehicle_nos: list[str],
    vehicle_by_no: dict[str, NormalizedVehicle],
) -> WorkPositionPreview: ...
```

必须返回结构化预演结果：

```python
class WorkPositionPreview(BaseModel):
    valid: bool
    hard_violations: list[str] = []
    evaluations: dict[str, WorkPositionEvaluation] = {}

class WorkPositionEvaluation(BaseModel):
    vehicle_no: str
    kind: str
    north_rank: int | None = None
    south_rank: int | None = None
    target_rank: int | None = None
    rank_gap: int | None = None
    satisfied_now: bool
```

`preview_work_positions_after_prepend(...)` 用于动作生成时预演：

```text
new_seq = incoming_vehicle_nos + existing_vehicle_nos
```

至少要检查：

- 本次放入工作线的 `SPOTTING` 车辆是否满足 `south_rank`。
- 本次放入工作线的 `EXACT_NORTH_RANK` 车辆当前 `north_rank` 和 `rank_gap`，用于候选排序和最终校验。
- 目标线上已有 `EXACT_NORTH_RANK` 车辆是否会因为本次 prepend 被推到 `north_rank > target_rank`。

不要把 `EXACT_NORTH_RANK` 当成必须在每次摘入后立即稳定的硬约束。工作线具体台位只是本次求解最终状态的北端序位要求；下一次求解会重新读取现场状态并重新计算。当前计划执行过程中，后续从北端 prepend 车辆可能把某辆车从 `north_rank=1` 推到 `north_rank=2`，这反而可能是满足具体台位的必要动作。

对目标工作线候选，`SPOTTING` 错位应作为 hard violation；`EXACT_NORTH_RANK` 若 `north_rank > target_rank` 也应作为 hard violation。只有 `north_rank < target_rank` 是可修复偏差，只记录 `rank_gap`，不作为 hard violation。

**测试重点:**

- `[C, B, A]` 中 `A south_rank=1`、`B south_rank=2`、`C south_rank=3`。
- 抛/油 `south_rank=1/2` 满足，`3` 不满足。
- 洗南 `south_rank=2/3/4` 满足，`1/5` 不满足。
- 调棚 `south_rank=3/4/5/6` 满足，`1/2/7` 不满足。
- 具体台位 `target_rank=2` 只在 `north_rank=2` 时满足。
- 具体台位 `target_rank=2` 预演时，`north_rank=1` 可保留，`north_rank=3` hard violation。
- 已在目标线且 `target_rank=2` 的车辆，若本次 prepend 后变成 `north_rank=3`，预演必须 hard violation。

Run:

```bash
pytest -q tests/domain/test_work_positions.py
```

---

### 3.3 台位分配层

**Files:**
- Modify: `src/fzed_shunting/domain/depot_spots.py`
- Test: `tests/io/test_depot_spot_validation.py`
- Test: `tests/verify/test_replay.py`

**当前问题:**

`depot_spots.py` 同时负责：

- 大库台位。
- 机库称重位。
- 作业线固定工作位。

新规则下，作业线不能继续在这里分配 `调棚:1`、`洗南:1` 这类固定码。否则最终判断会被旧码误导。

**必须修改:**

- 删除 `WORK_AREA_SPOTS` 的工作线语义。`机库:WEIGH` 如果仍需要常量，应迁移到更准确的名字，例如 `FIXED_FUNCTION_SPOTS` 或 `SPECIAL_FIXED_SPOTS`，不要继续挂在 `WORK_AREA_SPOTS` 下。
- 大库 `101-105/107` 保持现状。
- `allocate_spots_for_block(...)` 对工作线 `work_position_kind` 返回 `{}`，不再写 `spot_assignments`。
- `exact_spot_reservations(...)` 只收集 depot 精确台位，不收集工作线 `target_rank`。
- `realign_spots_for_track_order(...)` 对 depot 继续重排台位，对工作线只保留非该线路车辆的已有 depot/WEIGH assignment，不给工作线车辆补固定码。
- `_requires_spot_assignment(...)` 对 `target_mode="WORK_POSITION"` 必须返回 `False`。
- `_exact_work_area_spot_candidates(...)` 应删除；工作线具体台位不再经过 depot spot candidate 体系。
- `spot_candidates_for_vehicle(...)` 不再接受 `调棚:WORK` / `洗南:WORK` / `油:WORK` / `抛:WORK`，这些旧 area code 若出现应在归一化阶段报错或被测试覆盖为不可能出现。

**影响:**

`spot_assignments` 将只表示“真实固定台位占用”：

```text
大库 n01-n05/n07
机库:WEIGH
```

作业线对位不再从 `spot_assignments` 读，而从 `track_sequences` 动态算。

**测试重点:**

- 调棚 `isSpotting=是` 车辆进入调棚后，不再产生 `{"v": "调棚:1"}`。
- 大库 `101`、随机大库、迎检 `106/107` 原行为不变。
- 机库称重 `机库:WEIGH` 原行为不变。

Run:

```bash
pytest -q tests/io/test_depot_spot_validation.py tests/verify/test_replay.py
```

---

### 3.4 状态更新和回放层

**Files:**
- Modify: `src/fzed_shunting/solver/state.py`
- Modify: `src/fzed_shunting/verify/replay.py`
- Test: `tests/verify/test_replay.py`
- Test: `tests/solver/test_astar_solver.py` targeted state tests

**当前状态:**

`ATTACH` / `DETACH` 顺序模型已经正确：

```text
ATTACH: 源线路北端前缀 -> loco_carry 尾部追加
DETACH: loco_carry 尾部块 -> 目标线路北端 prepend
```

这层不需要改方向。

**需要改的点:**

`DETACH` 后当前会调用：

```python
realign_spots_for_track_order(...)
```

后续要保证：

- depot 线路继续重排 depot 台位。
- 工作线只更新 `track_sequences`，不生成固定作业台位码。
- 工作线位置合法性不要依赖 replay 的 spot assignment，而由 `goal_is_satisfied(...)` 和动作生成层判断。

**测试重点:**

- `DETACH` 到工作线后，目标线路序列仍然 prepend。
- `spot_assignments` 不包含工作线固定码。
- 对同一个 replay final state，用 `goal_is_satisfied(...)` 能正确判断工作线 `south_rank` / `north_rank`。

Run:

```bash
pytest -q tests/verify/test_replay.py
```

---

### 3.5 目标满足判断层

**Files:**
- Modify: `src/fzed_shunting/solver/goal_logic.py`
- Test: `tests/solver/test_goal_logic.py`
- Test: `tests/verify/test_plan_verifier.py`

**当前问题:**

`goal_is_satisfied(...)` 现在对：

- `SPOT`：看 `state.spot_assignments[vno] == target_spot_code`。
- 作业区：看 assigned spot 是否在 `spot_candidates_for_vehicle(...)`。

这不适合新规则。

**必须修改:**

增加工作线分支，优先于普通 `AREA` spot 分支：

```python
if vehicle.goal.work_position_kind is not None:
    return work_position_satisfied(vehicle, track_name=track_name, state=state)
```

具体语义：

```text
FREE:
  track_name in allowed_target_tracks

SPOTTING:
  track_name in allowed_target_tracks
  and south_rank(vehicle) in window

EXACT_NORTH_RANK:
  track_name in allowed_target_tracks
  and north_rank(vehicle) == target_rank
```

长度容量不在 `goal_is_satisfied` 单车函数里重复做，继续由动作生成和 verifier 的整线容量检查负责。

**测试重点:**

- 洗南 `isSpotting=是` 在 `[NORTH, TARGET, SOUTH]` 中满足，因为 `TARGET south_rank=2`。
- 洗南 `isSpotting=是` 在 `[TARGET]` 中不满足，因为 `south_rank=1`。
- 调棚 `isSpotting=是` 在 `south_rank=3/4/5/6` 满足。
- 工作线 `否/空` 只要在目标线就满足。
- 工作线具体台位 `3` 要求 `north_rank=3`。
- 大库 depot `SPOT` 不受影响。

Run:

```bash
pytest -q tests/solver/test_goal_logic.py tests/verify/test_plan_verifier.py
```

---

### 3.6 动作生成层

**Files:**
- Modify: `src/fzed_shunting/solver/move_generator.py`
- Test: `tests/solver/test_move_generator.py`

**当前问题:**

`_build_candidate_move(...)` 现在用两层旧限制：

```python
_fits_area_capacity(...)
allocate_spots_for_block(...)
```

其中 `_fits_area_capacity(...)` 是固定辆数限制：

```text
调棚:WORK = 4
洗南:WORK = 3
油:WORK = 2
抛:WORK = 2
```

新规则下这个限制不成立。作业线容量应该看长度，不看固定计数。

**必须修改:**

1. 删除工作线 `_fits_area_capacity(...)` 分支。

`AREA_CAPACITY_LIMITS` 常量应删除，不保留空壳。工作线容量只允许走 `_fits_capacity(...)` 的长度判断。

保留 `_fits_capacity(...)`：

```text
current_length + block_length <= trackDistance
```

2. 对工作线目标，增加序位预演：

```python
if is_work_position_track(target_track):
    preview = preview_work_positions_after_prepend(...)
    if preview.hard_violations:
        return None
```

3. 预演范围：

```text
new_seq = block + state.track_sequences[target_track]
```

至少判断本次 block 内所有工作线约束车辆：

- `SPOTTING` 必须放入后立即满足 `south_rank`。
- `EXACT_NORTH_RANK` 不应一律要求立即满足 `north_rank`，因为后续北端 prepend 可能把它推到目标序位；但如果已经 `north_rank > target_rank`，应拒绝该候选。`north_rank < target_rank` 时必须记录 `rank_gap` 参与候选排序，最终以整条计划结束时为准。
- `FREE` 不限制序位。

同时也要判断目标线既有车辆：

```text
existing WORK_POSITION vehicles in target track:
  SPOTTING: south_rank 不会被 prepend 改变，通常仍保持原判断。
  EXACT_NORTH_RANK: north_rank 会被 prepend 增大，可能从满足变成过位，必须预演。
```

所有调用 `allocate_spots_for_block(...)` 的候选构造路径都要加同一判断：

```text
WORK_POSITION:
  不调用 allocate_spots_for_block(...)
  调 preview_work_positions_after_prepend(...)

depot / weigh:
  继续走 allocate_spots_for_block(...)
```

当前文件里不止 `_build_candidate_move(...)` 一处调用 spot allocation，实施时要用 `rg "allocate_spots_for_block|_fits_area_capacity|AREA_CAPACITY_LIMITS"` 逐处清掉工作线分支。

4. 处理“用同一钩带垫车”的情况：

例如洗南 `TARGET` 要南边第 2 辆，空线时单独放入不满足：

```text
[TARGET] -> south_rank=1, reject as final placement
```

但如果一钩放入：

```text
[TARGET, FILLER] -> TARGET south_rank=2, allow
```

这个能力很重要，不能只支持已有线路上先有垫车。

**测试重点:**

- 洗南 `isSpotting=是` 单独入空洗南应不作为有效目标 move。
- 洗南 `isSpotting=是` 与一个尾部垫车同钩入空洗南应允许。
- 抛/油 `isSpotting=是` 单独入空线允许。
- 调棚 `isSpotting=是` 至少需要在 block 或已有线路南侧有 2 辆车，否则不能作为有效目标 move。
- 工作线 `否/空` 只受长度容量限制。
- 具体台位 `targetSpotCode=2` 的单步预演要能算出当前 `north_rank`：`north_rank=1` 保留，`north_rank=2` 排序靠前，`north_rank=3` 拒绝。
- 已满足 `EXACT_NORTH_RANK` 的既有车辆被新 block prepend 推到过位时，候选必须被拒绝。

Run:

```bash
pytest -q tests/solver/test_move_generator.py
```

---

### 3.7 搜索、启发式和构造评分层

**Files:**
- Modify: `src/fzed_shunting/solver/heuristic.py`
- Modify: `src/fzed_shunting/solver/exact_spot.py`
- Modify: `src/fzed_shunting/solver/constructive.py`
- Modify as needed: `src/fzed_shunting/solver/search.py`
- Modify: `src/fzed_shunting/solver/lns.py`
- Modify: `src/fzed_shunting/solver/soft_target_template.py`
- Modify: `src/fzed_shunting/solver/structural_metrics.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Test: `tests/solver/test_astar_solver.py`
- Test: `tests/solver/test_constructive.py`

**当前问题:**

搜索里很多逻辑把“精确台位”和“作业区台位”都理解为 `spot_assignments`：

- `exact_spot.py` 只适合 depot 固定台位冲突。
- `_h_spot_evict(...)` 看 `spot_assignments` 反查占位者。
- `constructive._move_displaces_satisfied_vehicle(...)` 手写了作业区 spot assignment 是否存在。
- `lns._spot_goal_satisfied(...)` 仍按旧工作线 area code + spot assignment 判断。
- `soft_target_template` 用 `allocate_spots_for_block(...)` 判断 soft target 是否可分配，工作线不应被旧 spot allocation 影响。
- `structural_metrics.area_random_unfinished_count` 和 `astar_solver` telemetry 以 `target_area_code is not None` 统计 area，工作线切到 `WORK_POSITION` 后要改名或单独统计。

新规则下，工作线具体台位不是固定 spot code，而是最终 `north_rank`；工作线对位是最终 `south_rank`。

**必须修改:**

1. depot 精确台位逻辑只处理 depot `SPOT`。

不要让工作线 exact rank 进入 `exact_spot.py`。

2. 构造层判断“已满足车辆是否被挪走”时，改为复用 `goal_is_satisfied(...)`。

不要继续手写：

```python
if vehicle.goal.target_area_code in {"调棚:WORK", ...}:
    state.spot_assignments.get(vno) is not None
```

`constructive._vehicle_needs_precise_placement(...)` 也要更新：

```text
WORK_POSITION + FREE:
  不属于精准台位目标，只要最终在线路上即可。

WORK_POSITION + SPOTTING / EXACT_NORTH_RANK:
  属于精准序位目标，需要纳入“精准完成”排序。

depot SPOT / 大库 RANDOM:
  保持原固定台位逻辑。
```

3. `lns.py` 必须复用 `goal_is_satisfied(...)`，删除旧工作线 `_spot_goal_satisfied(...)` 分支。

LNS 在 `_build_repair_plan_input(...)` 冻结已满足车辆时，工作线车辆不能再写入 `target_spot_code=snapshot.spot_assignments.get(...)`。冻结策略应是：

```text
depot SPOT / 大库 RANDOM:
  可以保留现有 spot freeze 逻辑。

WORK_POSITION:
  如果 goal_is_satisfied(...) 为真，冻结为 TRACK 目标即可：
    target_mode="TRACK"
    target_track=current_track
    allowed_target_tracks=[current_track]
    work_position_kind=None
    target_rank=None
  不要生成任何工作线 spot code。
```

4. `soft_target_template.py` 的候选可分配判断要区分目标类型。

```text
WORK_POSITION:
  allocatable = 线路长度容量足够
  不调用 allocate_spots_for_block(...)

depot / weigh:
  保留 spot allocation 判断。
```

5. `structural_metrics.py` 和 `astar_solver.py` telemetry 要把工作线从 area 统计中拆出。

必须扩展或改名指标：

```text
input_work_position_count
work_position_unfinished_count
area_random_unfinished_count 只统计真正 AREA/RANDOM 类目标
```

不能继续用 `target_area_code is not None` 把工作线混入 area 指标。

6. 引入基础 rank 评分，保证低勾数倾向。

```text
work_rank_satisfaction_bonus:
  DETACH 后让 SPOTTING / EXACT_NORTH_RANK 满足，加分

work_rank_wrong_final_penalty:
  把工作线约束车辆放入目标线但放错序位，扣分或直接不生成
```

7. 不实现复杂 rank blocker，先用 `rank_gap` 和已有阻挡/连续车组机制控制搜索。

```text
EXACT_NORTH_RANK:
  rank_gap > 0 -> 需要北侧补车，保留并排序靠后
  rank_gap = 0 -> 当前 rank 满足，排序靠前
  rank_gap < 0 -> 已太靠南，作为目标工作线候选不生成

SPOTTING south_rank:
  入目标工作线时不满足 -> 不生成候选
```

这能保持实现简洁，同时不会牺牲主要可解性：`EXACT_NORTH_RANK` 的可修复分支还在，`SPOTTING` 的低质量错位分支被移除。

**测试重点:**

- depot exact spot 相关测试仍通过。
- 工作线 exact rank 不触发 depot exact spot clearance/reblock。
- 已满足工作线 `SPOTTING` 车辆被搬离时，构造评分能识别为 displace satisfied。
- 搜索能解决一个小场景：洗南目标车需要南边第 2，和垫车同钩放入。
- LNS repair 不会把已满足工作线车辆冻结成旧 spot code。
- soft target template 不会因为工作线没有 `spot_assignments` 而判为不可分配。
- telemetry / structural metrics 不再把工作线 `WORK_POSITION` 混入旧 area random 统计。

Run:

```bash
pytest -q tests/solver/test_astar_solver.py tests/solver/test_constructive.py tests/solver/test_heuristic.py
```

---

### 3.8 校验层

**Files:**
- Modify: `src/fzed_shunting/verify/plan_verifier.py`
- Modify: `src/fzed_shunting/verify/replay.py`
- Test: `tests/verify/test_plan_verifier.py`
- Test: `tests/verify/test_replay.py`

**当前状态:**

最终校验已经调用 `goal_is_satisfied(...)`。只要第 3.5 层改好，最终目标校验自然能覆盖新规则。

**需要补强:**

- 保持最终整线容量检查：

```text
occupied_length <= effective_cap
```

- 增加错误信息可读性。当前错误是：

```text
final track/spot/weigh state does not satisfy goal
```

建议对工作线序位错误输出更具体：

```text
Vehicle X on 洗南 has south_rank=1, expected one of {2,3,4}
Vehicle Y on 调棚 has north_rank=2, expected 4
```

这会让现场验证比单纯 “spot wrong” 更容易定位。

- `verify/replay.py` 中 `realign_spots_for_track_order(...)` 只能重排 depot / 机库称重固定资源。DETACH 到工作线时，回放可以继续调用统一函数，但该函数必须对工作线返回“清除本线旧固定码后的 spot_assignments”，不能生成 `调棚:1` / `洗南:1`。

**测试重点:**

- 错误 south_rank 的 `isSpotting=是` 计划被 verifier 拒绝。
- 错误 north_rank 的具体台位计划被 verifier 拒绝。
- 工作线自由放置计划只要容量足够就通过。
- replay DETACH 到调棚/洗南/油/抛 后 `spot_assignments` 不含该工作线车辆。

Run:

```bash
pytest -q tests/verify/test_plan_verifier.py tests/verify/test_replay.py
```

---

### 3.9 Demo / 输出层

**Files:**
- Modify: `src/fzed_shunting/demo/view_model.py`
- Modify: `src/fzed_shunting/cli.py`
- Modify: `src/fzed_shunting/benchmark/runner.py`
- Modify: `src/fzed_shunting/workflow/runner.py`
- Modify: `app.py` if UI displays spot assignments directly
- Test: `tests/demo/test_view_model.py`
- Test: `tests/demo/test_app_topology.py`

**当前问题:**

Demo 和输出里可能展示：

```text
finalSpotAssignments = {"V": "调棚:1"}
```

新规则下工作线不再有固定 `调棚:1` 这种含义。

**必须修改:**

保留 depot / 机库的 `spot_assignments` 展示，同时新增工作线派生展示。这个派生展示可以由 `work_positions.py` 统一生成，避免 UI 自己重算一套规则：

```text
workPositionAssignments = {
  "V": {
    "track": "洗南",
    "northRank": 2,
    "southRank": 3,
    "rule": "SPOTTING",
    "satisfied": true
  }
}
```

如果接口不需要新增字段，也至少把显示文本改成：

```text
洗南 北2 / 南3
```

不要再显示 `洗南:1` 这种旧局部台位码。

所有输出 JSON 中：

```text
finalSpotAssignments:
  只包含 depot / 机库称重固定资源。

workPositionAssignments:
  包含调棚/洗南/油/抛的 northRank、southRank、rule、satisfied。
```

**测试重点:**

- depot finalSpotAssignments 仍显示 `101`。
- 工作线展示显示 rank，而不是旧 work spot code。
- UI 中车号后面的对位信息和最终状态一致。
- CLI / benchmark / workflow 输出不再把工作线 rank 塞进 `finalSpotAssignments`。

Run:

```bash
pytest -q tests/demo/test_view_model.py tests/demo/test_app_topology.py
```

---

### 3.10 数据、测试和文档层

**Files:**
- Modify: `data/master/spots.json`
- Modify: `docs/fzed-shunting-spot-vehicle-handling.md`
- Modify: `docs/fzed-shunting-current-algorithm-flow.md`
- Modify: `tests/io/test_normalize_input.py`
- Modify: `tests/verify/test_replay.py`
- Modify: `tests/solver/test_astar_solver.py`
- Modify: `tests/solver/test_constructive.py`
- Modify: `tests/solver/test_heuristic.py`
- Modify as needed: any tests asserting `调棚:WORK` / `洗南:WORK` / `油:WORK` / `抛:WORK` / `调棚:1`

**必须修改:**

删除 `spots.json` 中工作线固定 group：

```json
{"code":"调棚:1-4","track_code":"调棚","category":"WORK_GROUP","capacity":4}
```

必须从算法事实源中删除。硬切换完成后，`data/master/spots.json` 不应再包含调棚/洗南/油/抛的 `WORK_GROUP` 记录；如果加载到这些旧记录，应视为配置错误，而不是兼容处理。

文档要同步写明：

```text
作业线对位 = 最终序位约束
大库对位 = 固定库内台位
机库称重 = 特殊固定功能位
```

必须清理的旧断言包括：

```text
target_area_code == "调棚:WORK"
target_area_code == "洗南:WORK"
target_area_code == "油:WORK"
target_area_code == "抛:WORK"
spot_assignments["..."] == "调棚:1"
spot_assignments["..."] == "调棚:PRE_REPAIR"
replay.final_state.spot_assignments["WG1"] == "调棚:1"
```

这些测试不能改成“兼容旧结果仍可接受”，必须改成新硬切换断言。

---

## 4. 推荐实施顺序

### Task 1: 建立作业线序位领域模块

**Files:**
- Create: `src/fzed_shunting/domain/work_positions.py`
- Create: `tests/domain/test_work_positions.py`

**Why first:** 先把 rank 规则做成纯函数，后面所有层复用，避免重复实现。

Run:

```bash
pytest -q tests/domain/test_work_positions.py
```

### Task 2: 修改输入归一化

**Files:**
- Modify: `src/fzed_shunting/io/normalize_input.py`
- Modify: `tests/io/test_normalize_input.py`

**Why second:** 先让输入能表达 `WORK_POSITION + FREE/SPOTTING/EXACT_NORTH_RANK`，并停止产生旧 `target_area_code` / `target_spot_code`。

Run:

```bash
pytest -q tests/io/test_normalize_input.py
```

### Task 3: 改目标满足判断

**Files:**
- Modify: `src/fzed_shunting/solver/goal_logic.py`
- Create/Modify: `tests/solver/test_goal_logic.py`

**Why third:** 让最终正确性先成立。

Run:

```bash
pytest -q tests/solver/test_goal_logic.py tests/verify/test_plan_verifier.py
```

### Task 4: 移除工作线固定 spot assignment

**Files:**
- Modify: `src/fzed_shunting/domain/depot_spots.py`
- Modify: `src/fzed_shunting/solver/state.py`
- Modify: `src/fzed_shunting/verify/replay.py`
- Modify tests expecting `调棚:1` / `洗南:1`

**Why fourth:** 避免旧 spot code 和新 rank rule 并存，造成双重真源。

Run:

```bash
pytest -q tests/io/test_depot_spot_validation.py tests/verify/test_replay.py
```

### Task 5: 清理旧工作线 spot 消费路径

**Files:**
- Modify: `src/fzed_shunting/solver/heuristic.py`
- Modify: `src/fzed_shunting/solver/exact_spot.py`
- Modify: `src/fzed_shunting/solver/constructive.py`
- Modify: `src/fzed_shunting/solver/lns.py`
- Modify: `src/fzed_shunting/solver/soft_target_template.py`
- Modify: `src/fzed_shunting/solver/structural_metrics.py`
- Modify: `src/fzed_shunting/solver/astar_solver.py`
- Modify related solver tests

**Why fifth:** 在动作生成使用新 preview 前，先确保所有搜索辅助层不会继续把工作线当旧 spot/area 目标。

Run:

```bash
pytest -q tests/solver/test_heuristic.py tests/solver/test_constructive.py tests/solver/test_astar_solver.py
```

### Task 6: 改动作生成

**Files:**
- Modify: `src/fzed_shunting/solver/move_generator.py`
- Modify: `tests/solver/test_move_generator.py`

**Why sixth:** 防止生成明显不可能成为最终对位的工作线放车动作，并把 `rank_gap` 纳入候选排序。

Run:

```bash
pytest -q tests/solver/test_move_generator.py
```

### Task 7: 改 verifier 错误信息和 demo 输出

**Files:**
- Modify: `src/fzed_shunting/verify/plan_verifier.py`
- Modify: `src/fzed_shunting/demo/view_model.py`
- Modify: `src/fzed_shunting/cli.py`
- Modify: `src/fzed_shunting/benchmark/runner.py`
- Modify: `src/fzed_shunting/workflow/runner.py`
- Modify: `app.py`
- Modify demo/verifier tests

**Why seventh:** 最终校验和输出必须展示新 rank 事实，不能再输出旧工作线 spot code。

Run:

```bash
pytest -q tests/verify/test_plan_verifier.py tests/demo/test_view_model.py tests/demo/test_app_topology.py
```

### Task 8: 清理数据和旧测试断言

**Files:**
- Modify: `data/master/spots.json`
- Modify: `tests/io/test_normalize_input.py`
- Modify: `tests/verify/test_replay.py`
- Modify: `tests/solver/test_astar_solver.py`
- Modify: `tests/solver/test_constructive.py`
- Modify docs

Run:

```bash
pytest -q tests/io tests/verify tests/solver/test_astar_solver.py tests/solver/test_constructive.py
```

### Task 9: 全量回归

先跑静态残留扫描：

```bash
rg -n "调棚:WORK|调棚:PRE_REPAIR|洗南:WORK|油:WORK|抛:WORK|WORK_SPOT_BY_TARGET|AREA_CAPACITY_LIMITS|WORK_GROUP|调棚:1|洗南:1|油:1|抛:1" src tests data app.py
```

期望：

```text
只允许出现在“拒绝旧输入”的错误信息测试或历史说明文档中；
不得出现在求解逻辑、master 数据、通过性断言中。
```

Run:

```bash
pytest -q
```

如果外部验证样例耗时过长，可以先跑分层套件：

```bash
pytest -q tests/io tests/domain tests/verify tests/solver/test_move_generator.py tests/solver/test_goal_logic.py
```

---

## 5. 关键风险

1. `targetMode=SPOT` 不能继续一概表示“固定 spot code”。

工作线具体台位是 `north_rank`，大库具体台位才是固定 spot code。混用会让 `exact_spot.py`、`spot_assignments`、`state_key` 产生错误判断。

2. 工作线 `isSpotting=是` 的满足状态可能依赖同一钩里的垫车。

例如洗南南边第 2，空线单车不满足，但 `[TARGET, FILLER]` 同钩放入可以满足。

3. `否/空` 不应再被固定计数限制。

只要线路长度放得下，工作线自由放置应该允许超过旧 `WORK_GROUP capacity`。

4. 不要把工作线 rank 写入 `spot_assignments`。

`spot_assignments` 继续服务固定台位：大库和机库称重。工作线 rank 是从 `track_sequences` 派生的动态事实。

5. 最终校验必须是最终序列校验。

不能只在刚 DETACH 的瞬间判断一次就永久认为满足。任何后续 prepend 都可能改变本次计划内的 `north_rank`，因此具体台位号必须在本次求解最终状态再核。它不需要跨求解长期稳定；下一次求解会重新读取现场状态并重新计算。

---

## 6. 最小可验收场景

### 场景 A：抛丸对位

```text
目标车: isSpotting=是, targetTrack=抛
最终抛线: [NORTH, TARGET]
TARGET south_rank=1 -> 满足
```

### 场景 B：洗南对位需要垫车

```text
目标车: isSpotting=是, targetTrack=洗南
最终洗南: [TARGET, FILLER]
TARGET south_rank=2 -> 满足
```

如果最终是：

```text
[TARGET]
TARGET south_rank=1 -> 不满足
```

### 场景 C：调棚对位

```text
目标车: isSpotting=是, targetTrack=调棚
最终调棚: [A, B, TARGET, S1, S2]
TARGET south_rank=3 -> 满足
```

### 场景 D：工作线具体台位

```text
目标车: targetTrack=调棚, targetSpotCode=2
最终调棚: [FILLER, TARGET, SOUTH]
TARGET north_rank=2 -> 满足
```

### 场景 E：工作线自由放置

```text
目标车: targetTrack=洗南, isSpotting=否
最终洗南任意 north_rank/south_rank
只要长度放得下 -> 满足
```

### 场景 F：具体台位不能过位

```text
目标车: targetTrack=调棚, targetSpotCode=2

候选 1:
最终调棚: [FILLER, TARGET]
TARGET north_rank=2 -> 满足

候选 2:
最终调棚: [A, B, TARGET]
TARGET north_rank=3 -> 过位，不应生成为目标工作线候选

候选 3:
当前调棚: [TARGET]
TARGET north_rank=1 -> 可保留，后续再 prepend 一辆可能满足
```

### 场景 G：prepend 不能破坏既有具体台位

```text
既有调棚: [A, TARGET, S]
TARGET target_rank=2
TARGET 当前 north_rank=2 -> 满足

候选 prepend [NEW]:
新调棚: [NEW, A, TARGET, S]
TARGET north_rank=3 -> 过位
该候选必须拒绝，避免后续再搬回造成多余勾
```

### 场景 H：旧工作线 spot code 被彻底拒绝

```text
输入: targetMode=AREA, targetAreaCode=洗南:WORK
结果: INVALID_INPUT

输入: targetMode=SPOT, targetTrack=洗南, targetSpotCode=洗南:2
结果: INVALID_INPUT

输入: targetMode=SPOT, targetTrack=洗南, targetSpotCode=2
结果: WORK_POSITION / EXACT_NORTH_RANK / target_rank=2
```
