# 福州东调车求解器 V0

当前目录已经包含一个可运行的最小闭环：

- 主数据模型
- 输入合同归一化
- 模拟场景生成
- route oracle 与物理分支校验
- 简化版 block 级 A* 求解器
- plan verifier / replay
- benchmark runner
- Streamlit demo

## 环境

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install ortools pytest streamlit fastapi uvicorn
```

## 测试

```bash
. .venv/bin/activate
pytest -q
```

当前验证基线已更新为：`177 passed`

## 生成一个场景

```bash
. .venv/bin/activate
python -m fzed_shunting.cli generate-micro --output artifacts/sample_micro.json --seed 42 --vehicle-count 4
```

或按 profile 生成：

```bash
. .venv/bin/activate
python -m fzed_shunting.cli generate-scenario --output artifacts/sample_adversarial.json --profile adversarial --seed 42 --vehicle-count 4
python -m fzed_shunting.cli generate-scenario --output artifacts/sample_medium.json --profile medium --seed 42 --vehicle-count 12
python -m fzed_shunting.cli generate-typical-suite --output artifacts/typical_suite.json
python -m fzed_shunting.cli generate-typical-workflow-suite --output artifacts/typical_workflow_suite.json
```

## 求解一个场景

```bash
. .venv/bin/activate
python -m fzed_shunting.cli solve --input artifacts/sample_micro.json
python -m fzed_shunting.cli solve --input artifacts/sample_micro.json --solver weighted --heuristic-weight 2.0
python -m fzed_shunting.cli solve --input artifacts/sample_micro.json --solver beam --beam-width 16
python -m fzed_shunting.cli solve --input artifacts/sample_micro.json --solver lns --beam-width 16
python -m fzed_shunting.cli solve-suite --input artifacts/typical_suite.json --output-dir artifacts/typical_suite_solved --solver lns --beam-width 8
python -m fzed_shunting.cli solve-workflow --input artifacts/sample_workflow.json
python -m fzed_shunting.cli solve-suite --input artifacts/typical_workflow_suite.json --output-dir artifacts/typical_workflow_suite_solved --solver lns --beam-width 8
python -m fzed_shunting.cli compare-suite --input artifacts/typical_suite.json --output-dir artifacts/typical_suite_compare --solver exact --solver weighted --solver beam --solver lns --weighted-heuristic-weight 2.0 --beam-width 8 --lns-beam-width 8
```

`solve-suite` 现在可同时处理两类 suite：

- 单阶段 `scenario.payload.vehicleInfo`
- 多阶段 `workflow.payload.initialVehicleInfo + workflowStages`

workflow suite 的每个导出文件会额外包含：

- `scenario_type=workflow`
- `stage_count`
- `stages[].hookCount / finalTracks / finalSpotAssignments / verifierErrors`

`compare-suite` 会：

- 对同一份固定 suite 逐个运行多个 solver
- 在 `output-dir/<solver>/` 下各自导出 per-scenario plan 与 `suite_report.json`
- 在 `output-dir/comparison_report.json` 汇总每个 solver 的 `validCount / averageHookCount`

## 校验一个求解结果

先把 `solve` 输出保存为 `artifacts/sample_plan.json`，然后：

```bash
. .venv/bin/activate
python -m fzed_shunting.cli verify --input artifacts/sample_micro.json --plan artifacts/sample_plan.json
```

## 跑 benchmark

```bash
. .venv/bin/activate
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark --scenario-count 10 --vehicle-count 4 --seed-start 0
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark_weighted --scenario-count 10 --vehicle-count 4 --seed-start 100 --solver weighted --heuristic-weight 2.0
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark_beam --scenario-count 10 --vehicle-count 4 --seed-start 200 --solver beam --beam-width 16
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark_lns --scenario-count 10 --vehicle-count 4 --seed-start 300 --solver lns --beam-width 16
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark_mixed --scenario-count 10 --vehicle-count 4 --seed-start 100 --no-direct-only
python -m fzed_shunting.cli benchmark --output-dir artifacts/benchmark_adv --profile adversarial --scenario-count 10 --vehicle-count 4 --seed-start 200 --no-direct-only
```

或：

```bash
. .venv/bin/activate
python scripts/run_benchmark.py
```

## 启动 demo

```bash
. .venv/bin/activate
streamlit run app.py
```

在页面里填入某个场景 JSON 路径，例如：

`artifacts/sample_micro.json`

页面现已支持：

- 直接选择 `exact / weighted / beam`
- 直接选择 `exact / weighted / beam / lns`
- 为 weighted 设置 `Heuristic Weight`
- 为 beam / lns 设置 `Beam Width`
- 可选加载外部 `plan.json`
- 自动识别 `typical_suite.json`，并在页面中选择具体典型场景
- 自动识别 `typical_workflow_suite.json`，并在页面中选择具体多阶段典型场景
- 逐步显示线路图回放卡片，高亮当前路径、变化股道、占用和机车位置
- 几何拓扑 SVG 回放：按股道二维布局显示线路，并支持每钩内部的微时间轴运动标记
- 逐钩显示 verifier 错误
- 对比多种 solver 的结果摘要
- 加载外部 `plan.json` 时显示“外部计划 vs 当前求解器”的钩数/校验差异摘要
- 自动播放逐钩回放
- 下载钩计划 JSON 与回放快照 JSON

典型演示场景：

- `artifacts/typical_suite.json` 包含 `13` 个固定单阶段场景，当前 `13/13` 可解
- `artifacts/typical_workflow_suite.json` 包含 `17` 个固定多阶段场景：`dispatch_depot_departure / weigh_then_departure / weigh_jiku_final / multi_vehicle_dispatch_merge / wash_depot_departure / wheel_departure / paint_depot_departure / shot_depot_departure / outer_depot_departure / dispatch_jiku_final / pre_repair_departure / main_pre_repair_departure / jipeng_departure / depot_wheel_departure / tank_wash_direct_departure / close_door_departure / inspection_departure`

## 当前能力边界

当前版本已经实现并验证的内容：

- `TRACK / AREA / SPOT` 归一化
- 多阶段 workflow 执行：阶段间继承 `track_sequences / loco_track_name / weighed_vehicle_nos / spot_assignments`，且后续阶段输入快照会继承上一阶段最终机车位置
- `迎检 -> INSPECTION`
- 走行线/临停线/机库/存4南 等基本边界
- `存4南` 已验证可作为过程临停清障线，但仍不能作为正式终点
- 称重先经过机库
- 单钩称重最多 `1` 辆
- 纯空车 `20` 辆上限、重车 `<=2` 且按 `4` 辆空车折算
- 大库随机区长车优先 3/4 库
- 大库精准台位与随机台位分配；`06/07` 台位仅在 `INSPECTION` 模式开放
- 调棚/洗南/油/抛工位组容量上限
- 调棚/洗南/油/抛作业工位与调棚尽头预修位已统一进入 `spot_assignments`
- 多阶段 workflow 典型链路已验证：调棚 -> 大库 -> 存4北、机库称重 -> 存4北、称重后机库终到、双车分流 -> 汇合出段、洗南 -> 大库 -> 存4北、轮 -> 存4北、油 -> 大库 -> 存4北、抛 -> 大库 -> 存4北、大库外 -> 存4北、调棚 -> 机库终到、调棚预修 -> 存4北、主预修 -> 存4北、机棚 -> 存4北、大库 -> 轮 -> 存4北、临修罐车 洗南 -> 存4北、关门车 -> 存4北 top-3 约束、迎检大库 -> 存4北
- `存4北` 的关门车 top-3 禁止
- 物理分支可达性、reverse 分支余量校验、`L1 190m` 约束
- 基于 `business_rules.json` 的机车长度与路径清空规则
- 中间股道占用导致的运行干涉校验与剪枝
- source / target 两侧 reverse 分支余量校验
- `pathTracks` 完整经过股道序列输出与校验
- 路径拓扑主数据化：`tracks.json` 提供股道端点/接入节点，`physical_routes.json` 提供分支节点
- 北侧前缀 block 动作生成
- 临停清障：可把挡车或挡路径股道上的 block 临时送往临停线，再回送最终目标
- 容量感知且 route-aware 的简化 A* / 热点局部修复 LNS
- 支持 `exact / weighted / beam` 三种求解模式
- 支持 `lns` 局部改良模式
- `solve-suite` 已支持单阶段 suite 与 workflow suite 的统一批量求解和逐场景导出
- 搜索状态键已纳入已称重集合与大库台位分配，避免错误状态合并
- 搜索终止判定已纳入称重完成、精确台位、大库随机台位和 `存4北` 关门车顺位约束
- profile 化场景生成：`micro / medium / large / adversarial`
- 场景生成会自动补齐称重车所需 `机库` 和大库目标所需 `修1~4库内/库外` trackInfo，减少先天无解样例
- 生成场景包含 `scenarioMeta.profile / seed / vehicleCount / tags`
- benchmark 统计项：`solved_count / unsolved_count / solved_rate / average_hook_count_on_solved / average_path_length_m_on_solved / average_branch_count_on_solved / average_expanded_nodes_on_solved / average_generated_nodes_on_solved / error_summary`
- benchmark 可按 profile 和 solver 跑批，结果中保留 `profile / solver / heuristic_weight / beam_width`
- `solve` 输出增强：`solver / vehicleCount / remark`
- demo 回放项：逐钩动作、变化股道、当前机车位置、当前已称重车辆、当前库内台位分配、最终台位分配、每钩路径长度/反向分支 remark、校验结果、逐钩 verifier 错误、外部计划回放、solver 对比摘要
- demo 已支持直接加载 workflow payload 和 workflow suite，并按阶段查看逐钩回放与拓扑图；workflow 页面已增加阶段总览、阶段进度和线路图图例
- workflow 页面现已增加“阶段间状态变化摘要”，可直接查看每阶段机车位置变化、车辆迁移、称重增量和台位变化

仍未完成的高复杂度业务能力：

- 中间折返节点已支持“显式主数据配置”版本：起终点继续按 terminal branch 校验，路径中间股道可通过 `reverse_branches` 额外声明必须校验的倒车分支；当前首批已在 `机棚 -> Z1-L8` 上启用
- 复杂多钩清障与临停优化仍是启发式 V1，尚未做到全局最优级别的中转搜索
- 当前 `lns` 已升级为热点局部修复版本：会优先围绕临停/存4南/重复触达热点切点做 repair，并在局部 repair 时冻结已满足目标的车辆；但仍未接入 CP-SAT repair，也不宣称严格全局最优
- demo 已支持 solver 对比、失败钩定位、自动播放、结果下载，以及基于独立布局主数据的几何拓扑 SVG 回放；当前仍是 demo 级几何播放，不宣称连续动力学/精确调车物理仿真
