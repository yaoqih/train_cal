# 福州东调车业务覆盖矩阵

本文档用于把当前实现与《福州东调车业务说明.md》《福州东车辆段业务说明.md》的对齐状态固化成可审计矩阵。

截至当前版本：

- 单阶段典型场景：`13`
- 多阶段 workflow 典型场景：`17`
- 全量测试：`177 passed`

## 已覆盖且已验证

| 业务规则 | 当前实现/验证 | 状态 |
| --- | --- | --- |
| `机库` 可作为最终目的地 | 单阶段 `single_direct`；workflow `dispatch_jiku_final`、`weigh_jiku_final` | 已验证 |
| `存4南` 不能作为正式终点 | `normalize_input` / workflow rejection tests | 已验证 |
| `存4南` 可作为过程临停清障线 | 单阶段 `cun4nan_staging`；move generator / A* / verifier tests | 已验证 |
| `存4北` 为正式出段集结线 | 多个单阶段/ workflow 终态场景 | 已验证 |
| `调棚:WORK` 与 `调棚:PRE_REPAIR` 二分 | 单阶段 `dispatch_work_spot`、`dispatch_pre_repair`；workflow `pre_repair_departure` | 已验证 |
| `迎检 -> INSPECTION` | 单阶段 `inspection_depot`；workflow `inspection_departure` | 已验证 |
| `洗南:WORK` 默认归一化 | normalize tests；单阶段 `wash_work_area`；workflow `wash_depot_departure` / `tank_wash_direct_departure` | 已验证 |
| `油:WORK` 默认归一化 | normalize tests；单阶段 `paint_work_area`；workflow `paint_depot_departure` | 已验证 |
| `抛:WORK` 默认归一化 | normalize tests；单阶段 `shot_work_area`；workflow `shot_depot_departure` | 已验证 |
| `轮:OPERATE` 默认归一化 | normalize tests；单阶段 `wheel_operate`；workflow `wheel_departure` / `depot_wheel_departure` | 已验证 |
| `大库:RANDOM` / `大库外:RANDOM` | normalize tests；workflow `outer_depot_departure` 等 | 已验证 |
| 大库精准台位 `101~407` | verifier / solver / workflow `dispatch_depot_departure`、`depot_wheel_departure` | 已验证 |
| 称重必须经过 `机库:WEIGH` | 单阶段 `weigh_then_store`；workflow `weigh_then_departure`、`weigh_jiku_final` | 已验证 |
| 单钩称重最多 `1` 辆 | move generator / verifier tests | 已验证 |
| 纯空车最多 `20` 辆 | move generator / verifier tests | 已验证 |
| 混编重车最多 `2` 辆 | verifier tests | 已验证 |
| `1` 辆重车按 `4` 辆空车折算 | hook constraints / verifier tests | 已验证 |
| `L1 190m` 约束 | route oracle / move generator / verifier tests | 已验证 |
| 关门车 `存4北 top-3` 禁止 | solver / verifier / workflow `close_door_departure` | 已验证 |
| 非 `存4北` 且单钩 > `10` 时关门车不得首位 | solver / verifier / 单阶段 `close_door_non_cun4bei` | 已验证 |
| 前挡车/中间路径占用可临停清障 | 单阶段 `front_blocker`、`path_blocker`；A* / move generator tests | 已验证 |
| 混编折算超限 block 在 move generation 阶段即被剪枝 | `move_generator` 混编重车折算超限且无 L1 干扰场景测试，证明不会退化成非法临停 move | 已验证 |
| 主预修 / 机棚补充预修链路 | workflow `main_pre_repair_departure`、`jipeng_departure` | 已验证 |
| demo 外部计划对比摘要口径 | `view_model` + `app` tests，统一验证钩数、差值、失败钩号、solverError 展示摘要 | 已验证 |
| workflow 阶段间变化摘要 | `app` tests，验证每阶段机车位置变化、车辆迁移、称重增量与台位变化摘要 | 已验证 |

## 已覆盖但仍属启发式实现

| 能力 | 说明 | 状态 |
| --- | --- | --- |
| 多钩清障与临停重排 | 当前可解一批前挡车/路径阻塞案例，但整体仍是启发式 V1，不保证全局最优 | 部分等价 |
| LNS 改良 | 当前已支持热点切点排序 + 已满足车辆目标冻结的局部 repair；未接入 CP-SAT repair | 部分等价 |

## 明确未完成或不能宣称等价

| 业务项 | 当前状态 | 原因 |
| --- | --- | --- |
| 几何级连续拓扑回放 | demo 已支持二维布局 + 每钩微时间轴 SVG 动画标记 | 已验证 |
| 严格全局最优大实例求解 | 未完成 | 当前主力仍是 block 级 A* / weighted / beam / lns |

## 当前典型工件

### 单阶段

- `artifacts/typical_suite.json`
- `13` 个场景
- 当前 `13/13` 可解

### Workflow

- `artifacts/typical_workflow_suite.json`
- `17` 个场景
- 当前 `17/17` 可解

## 下一批建议

1. 继续补“业务规则 -> 测试 -> 典型工件”空白，优先容量/重车的更复杂组合场景。
2. 若要推进演示等价，下一步应做更强的拓扑动画或阶段对比视图，而不是继续堆 solver 参数。
3. 若要推进大实例质量，下一步优先补 CP-SAT repair 或更强的 LNS destroy/repair，而不是继续增加固定规则。
