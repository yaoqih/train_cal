# 福州东调车求解器 Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从业务说明出发，完成福州东调车问题的模拟数据生成、求解器开发、验证评测、性能改进与可视化演示闭环。

**Architecture:** 采用“数据主线 + 可验证求解内核 + 局部改良 + 回放可视化”的分层路线。先建立可重复的场景与真值校验，再开发 block 级字典序 A* 主求解器，随后引入 LNS + CP-SAT 做大实例增强，最后做基于回放的演示界面。

**Tech Stack:** Python 3.11+, pytest, pydantic/dataclasses, networkx 或自定义图结构, OR-Tools CP-SAT, Typer/Click CLI, FastAPI or Streamlit/React for demo, JSON/JSONL/Parquet for scenario artifacts.

---

## 0. 总体策略

整个项目按 6 个阶段推进：

1. 领域模型与主数据固化
2. 模拟数据与 benchmark 体系
3. 核心求解器 V1
4. 验证、回放、评测体系
5. 大实例优化与改进
6. 可视化演示与交付

每个阶段都必须有：

- 明确产物
- 可运行脚本
- 自动化测试
- 样例输入输出
- 下一阶段可复用的稳定接口

不要先做“功能很多的 UI”。先把内核做成一个能批量跑、能回放、能打分的 research-grade tool。

## 1. 阶段一：领域模型与主数据固化

### 1.1 目标

把两份业务说明固化成可执行的 machine-readable model，结束“文档理解不一致”的问题。

### 1.2 核心产物

- `data/master/tracks.json`
- `data/master/physical_routes.json`
- `data/master/spots.json`
- `data/master/business_rules.yaml`
- `src/fzed_shunting/domain/*.py`
- `tests/domain/*`

### 1.3 需要定义的核心对象

- `Vehicle`
- `Track`
- `Spot`
- `Area`
- `PathSegment`
- `PhysicalRoute`
- `GoalSpec`
- `PlanInput`
- `State`
- `Block`
- `HookAction`
- `PlanResult`

### 1.4 要落实的业务边界

- 上游显式给 `targetTrack/targetMode/targetAreaCode/targetSpotCode`
- 算法只负责合法性校验和技术性派生
- `yardMode` 是整次计划级别
- `trackDistance` 与 `physicalDistance` 严格分离
- `机库` 可以是最终目标线
- `存4南` 只能临停
- `needWeigh` 与 `REPAIR_STAGING` 必须规则化

### 1.5 验收标准

- 任意一个输入样例都可以被解析成稳定内部对象
- 关键主数据都有 schema 校验
- 对文档中的 15 到 20 条关键业务规则，有单元测试逐条覆盖

### 1.6 推荐用时

3 到 5 天

## 2. 阶段二：模拟数据与 benchmark 体系

### 2.1 目标

建立一个不是“瞎随机”的场景生成器，而是可控分布、可分难度、可回放、可用于性能评估的数据体系。

### 2.2 设计原则

模拟数据要分三层：

1. 规则合成数据
2. 业务分布驱动数据
3. 手工构造极端/对抗数据

### 2.3 数据类型

#### A. 微型 exact 场景

用途：

- 验证状态转移正确
- 验证最优性
- 开发启发函数

特征：

- 3 到 6 条活跃线
- 5 到 20 辆车
- 少量工位和特殊规则

#### B. 中型评测场景

用途：

- 验证搜索规模
- 对比不同启发函数和剪枝

特征：

- 15 到 60 辆车
- 多目标线、多临停、多挡车
- 含称重、关门车、大库

#### C. 大型近真实场景

用途：

- 压测
- LNS/beam/weighted A* 改进验证
- demo 演示

特征：

- 60 到 200+ 辆车
- 多个冲突窗口
- 更接近真实股道利用率

### 2.4 场景生成器要支持的参数

- 总车辆数
- 初始线分布
- 目标类型分布：`TRACK / AREA / SPOT`
- 车型分布
- 重车比例
- 称重比例
- 关门车比例
- 大库目标比例
- 挡车深度分布
- 临停需求强度
- 拥堵度
- `yardMode`

### 2.5 benchmark 数据目录建议

- `data/scenarios/micro/*.json`
- `data/scenarios/medium/*.json`
- `data/scenarios/large/*.json`
- `data/scenarios/adversarial/*.json`
- `data/benchmarks/manifest.json`

### 2.6 每个场景应包含

- 初始状态
- 目标定义
- 元数据标签
- 生成参数
- 难度标签
- 若有：已知最优值或当前最好值

### 2.7 验收标准

- 至少生成 100 个 micro、200 个 medium、100 个 large 场景
- 每个场景都能通过输入合法性校验
- 至少 20 个 micro 场景有人为或 exact 方式确认最优值

### 2.8 推荐用时

4 到 6 天

## 3. 阶段三：核心求解器 V1

### 3.1 目标

实现一个能在 micro/medium 场景上稳定给出合法计划的 block 级字典序 A* 求解器。

### 3.2 核心模块

- `normalizer`
- `state_codec`
- `block_builder`
- `route_oracle`
- `move_generator`
- `cost_model`
- `heuristics`
- `search_astar`
- `verifier`

### 3.3 开发顺序

1. 状态定义与哈希
2. block 构造规则
3. 单钩动作生成
4. 单钩合法性校验
5. 状态转移
6. 目标状态判定
7. A* 搜索
8. 结果回放

### 3.4 V1 必须先实现的启发函数

- `h_unfinished_goals`
- `h_blocking_moves`
- `h_pending_weigh`
- `h_spot_conflict`
- `h_capacity_release`

先做保守 admissible 版本，正确优先。

### 3.5 V1 不要急着做

- 强化学习动作排序
- 复杂神经启发函数
- 多机车扩展
- 连续时间调度
- 复杂前端动画

### 3.6 验收标准

- 所有 micro 场景都返回合法计划或明确无解
- 至少 80% 的 medium 场景在设定时限内返回合法计划
- micro 场景与 exact 对照的最优钩数一致率达到目标值

### 3.7 推荐用时

7 到 12 天

## 4. 阶段四：验证、回放、评测体系

### 4.1 目标

让每个求解结果都可以独立复验，避免“看起来能跑，但其实违规”。

### 4.2 要建设的能力

#### A. 独立 verifier

输入：

- 初始状态
- 计划钩序

输出：

- 是否合法
- 第几钩失败
- 失败原因
- 失败时的状态快照

#### B. replay engine

作用：

- 重放每一钩
- 生成中间状态快照
- 为可视化提供数据

#### C. benchmark runner

作用：

- 批量运行全部场景
- 输出成功率、钩数、节点数、耗时、gap

#### D. regression dashboard data

输出：

- CSV/JSONL 指标汇总
- 便于后续画趋势图

### 4.3 核心指标

- legality pass rate
- solved rate
- first solution time
- best solution time
- expanded nodes
- closed nodes
- total hooks
- total switch cost
- total path cost
- optimality gap

### 4.4 验收标准

- 所有 solver 输出必须经过 verifier
- 每次改算法都能批量跑 benchmark
- 任一回归都能定位到具体场景与失败钩号

### 4.5 推荐用时

3 到 5 天

## 5. 阶段五：大实例优化与改进

### 5.1 目标

在保证合法性的前提下，让 large 场景的首解更快、解更好。

### 5.2 建议的优化路线

#### A. 搜索层优化

- better dominance rules
- symmetry reduction
- route cache
- move cache
- transposition table
- weighted A*
- beam search

#### B. LNS 改进

destroy 邻域：

- 临停拥堵窗口
- 大库入口窗口
- 机库称重窗口
- 高 rehandle 窗口
- 高 switch-cost 窗口

repair 方法：

- 小窗口 DP
- 中窗口 CP-SAT
- 必要时启发式 repair

#### C. 学习辅助，只做排序不做决策

- 动作优先级模型
- 邻域选择模型
- 场景难度预测器

### 5.3 验收标准

- large 场景首解时间显著下降
- 在固定时间预算下，平均钩数或综合目标明显优于 V1
- 没有引入 legality 回归

### 5.4 推荐用时

7 到 14 天

## 6. 阶段六：可视化演示与交付

### 6.1 目标

做一个能展示“输入 -> 求解 -> 回放 -> 指标”的演示系统，而不是只做漂亮动画。

### 6.2 演示系统建议结构

#### A. 场景选择页

- 选择 micro / medium / large
- 选择典型业务标签
- 展示输入摘要

#### B. 求解页

- 选择 solver 模式：exact / weighted / beam / LNS
- 显示求解进度、节点数、当前 incumbent

#### C. 回放页

- 逐钩播放
- 高亮取车 block、目标线、路径
- 显示每钩后的线内顺位变化

#### D. 指标页

- 总钩数
- 道岔代价
- 路径代价
- 求解耗时
- 与 baseline 比较

### 6.3 UI 不是重点，但必须支持

- 单步回放
- 自动播放
- 失败钩定位
- 状态快照下载
- 结果 JSON 下载

### 6.4 建议技术路线

如果要快速出 demo：

- 后端 CLI + JSON 输出
- 前端先用 Streamlit 或单页 React

如果后期要正式产品化：

- FastAPI + React
- 独立 replay API

### 6.5 验收标准

- 至少能展示 5 个典型场景
- 每个场景可完整回放
- 可切换不同 solver 结果做比较

### 6.6 推荐用时

4 到 7 天

## 7. 建议里程碑

### Milestone 1：领域模型冻结

产物：

- 主数据文件
- 输入 schema
- 业务规则测试

完成标志：

- 业务规则不再靠口头补充

### Milestone 2：可重复 benchmark

产物：

- 场景生成器
- benchmark 数据集
- exact micro 对照集

完成标志：

- 算法改动可以被量化

### Milestone 3：V1 solver 可用

产物：

- exact/weighted A* 主求解器
- verifier
- replay

完成标志：

- micro 全过，medium 大部分可解

### Milestone 4：大实例增强

产物：

- beam/LNS/CP-SAT repair
- 性能报告

完成标志：

- large 场景可稳定给出高质量方案

### Milestone 5：demo 可演示

产物：

- 场景选择
- 求解运行
- 逐钩回放
- 指标对比

完成标志：

- 能对外演示完整闭环

## 8. 优先级建议

如果资源有限，按这个优先级做：

1. 主数据与 schema
2. 场景生成器
3. verifier + replay
4. exact/weighted A*
5. benchmark runner
6. LNS + CP-SAT
7. demo UI

因为：

- 没有数据和 verifier，后面所有优化都不可信
- 没有 replay，可视化也只是表面演示
- 没有 benchmark，就无法判断算法有没有进步

## 9. 风险与防坑

### 9.1 常见风险

- 过早做复杂 UI，核心还不稳定
- 直接做大模型或 RL，缺少强 verifier
- 主数据没固化，导致代码里规则散落
- 场景生成只有随机，没有业务分布
- 只比耗时，不比 legality 和目标值

### 9.2 对策

- 一切 solver 输出必须经过 verifier
- 所有大改动都跑 benchmark
- 任何新启发式先在 micro/medium 回归
- 先做 CLI，再做交互式 demo

## 10. 你接下来最应该做什么

如果你现在准备真正开工，最建议的前三步是：

1. 先把 `data/master` 和 `domain schema` 建出来。
2. 然后做 `scenario generator + verifier + replay`。
3. 最后才开始写 `block-level A*`。

这是因为：

- 没有统一主数据，求解器会反复返工
- 没有 verifier/replay，solver 输出不可信
- 没有模拟数据，优化无从评测

## 11. 实施顺序建议

### 第 1 周

- 冻结主数据格式
- 建好 domain model
- 写关键规则单测

### 第 2 周

- 完成场景生成器
- 完成 benchmark manifest
- 完成 verifier 原型

### 第 3 周

- 完成状态表示
- 完成动作生成
- 完成状态转移

### 第 4 周

- 完成 exact/weighted A*
- 跑通 micro/medium

### 第 5 周

- 完成 replay 与结果指标汇总
- 形成第一版性能报告

### 第 6 周及以后

- 做 beam/LNS/CP-SAT repair
- 做 large 场景压测
- 做可视化 demo

## 12. 建议的最小可交付闭环

最小但完整的版本应该是：

- 一份 machine-readable 主数据
- 一批可重复场景
- 一个 block 级求解器
- 一个独立 verifier
- 一个 replay 导出器
- 一个简单的可视化页面

只要这 6 件事齐了，这个项目就已经从“想法”进入“工程系统”。
