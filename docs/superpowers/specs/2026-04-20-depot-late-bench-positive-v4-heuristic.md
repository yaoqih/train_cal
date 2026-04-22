# 大库后调车 bench 分析报告

- 输入集：`data/validation_inputs/truth`（64 场景）
- 预算：`timeout_seconds=120` / `solver_time_budget_ms=115000`
- 求解器：`solver=exact`，`anytime_fallback=True`
- 并行度：6 workers
- OFF 运行：`enable_depot_late_scheduling=False`
- ON  运行：`enable_depot_late_scheduling=True`

## 1. 整体成败

| 指标 | OFF | ON | 变化 |
|---|---:|---:|---|
| 场景总数 | 64 | 64 | - |
| 返回 plan | 64 (100.0%) | 64 (100.0%) | +0 |
| verifier 通过 | 52 (81.2%) | 52 (81.2%) | +0 |
| verifier 失败（solved=True, is_valid=False） | 12 | 12 | +0 |
| 容量预检拒绝 | 0 | 0 | +0 |
| 空返（no solution within budget） | 0 | 0 | +0 |
| 其他错误 | 0 | 0 | +0 |

## 2. 勾数分布（solved 场景）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 64 | 64 |
| 总和 | 2779 | 2779 |
| 均值 | 43.42 | 43.42 |
| 中位数 | 8.5 | 8.5 |
| p95 | 292 | 292 |
| max | 292 | 292 |

## 3. 大库钩指标（solved 场景）

| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |
|---|---:|---:|---:|---:|
| depot_hook_count | 4.42 | 7 | 4.42 | 7 |
| depot_earliness | 243.98 | 1754 | 242.05 | 1754 |

**Earliness 总和**：OFF=15615，ON=15491，绝对下降 124，相对下降 0.8% 。

## 4. Fallback stage 命中

| Stage | OFF | ON |
|---|---:|---:|
| `beam` | 18 | 18 |
| `constructive_partial` | 12 | 12 |
| `exact` | 34 | 34 |

## 5. 配对比较（solved in both）

配对样本数：**64**

### 5.1 勾数对比

- ON 更少勾数：**0** 场景（0.0%）
- 持平：**64** 场景（100.0%）
- ON 更多勾数：**0** 场景（0.0%）—— **关键回归指标**

### 5.2 Earliness 对比（大库钩数相同的子集）

同 depot_hook_count 子集（N=64）：
- ON earliness 更小（大库更靠后）：**19** 场景
- 持平：**42** 场景
- ON earliness 更大（大库更靠前）：**3** 场景
- 总 earliness 绝对下降：**124**

全部配对（不同 depot_count 也纳入，仅供参考）：
- 总 earliness 变化：**-124**

### 5.3 勾数回归明细（ON 比 OFF 多）

**无勾数回归。**

### 5.4 勾数下降明细（ON 比 OFF 少）

（无）

### 5.5 Earliness top improvements（同大库钩数）

| Scenario | depot_count | OFF earl | ON earl | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---:|---|---|
| `case_3_2_shed_pre_repair_from_pre_repair.json` | 7 | 1038 | 1020 | **-18** | beam | beam |
| `case_3_4_heavy_mix_with_empties.json` | 5 | 25 | 10 | **-15** | exact | exact |
| `case_5_5_weigh_three_cars_three_hooks.json` | 3 | 18 | 8 | **-10** | exact | exact |
| `case_3_4_weigh_to_depot.json` | 3 | 9 | 3 | **-6** | exact | exact |
| `case_4_3_1_track_cun1.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_4_3_1_track_jibei.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_4_3_1_track_jipeng.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_4_3_4_area_xinan.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_4_3_4_area_you.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_5_5_weigh_single_car.json` | 3 | 9 | 3 | **-6** | exact | exact |
| `case_7_2_xinan_spot_1.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_7_2_xinan_spot_2.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_7_2_xinan_spot_3.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_7_3_weigh_then_jiku_final.json` | 2 | 7 | 1 | **-6** | exact | exact |
| `case_3_4_heavy_pair.json` | 3 | 8 | 3 | **-5** | exact | exact |

## 6. 运行时间分布（solved 场景，elapsed_ms）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 64 | 64 |
| 均值 (ms) | 1545 | 1500 |
| 中位数 (ms) | 83 | 75 |
| p95 (ms) | 9201 | 9202 |
| max (ms) | 9202 | 9202 |

## 7. 大库钩平均位置分布（fraction = 平均索引 / N）

对每个 solved 且有大库钩的场景计算：
- `avg_frac = depot_index_sum / (K * N)` = 大库钩平均位置占 plan 长度的比例
- `theoretical_max = (2N - K + 1) / (2N)` = 若把全部 K 个大库钩塞到 plan 最后 K 位时的平均比例
  （该上限不考虑依赖约束，纯粹是 pigeonhole）

| Bucket (avg depot position fraction) | OFF | ON | 理论上限 |
|---|---:|---:|---:|
| < 0.50 | 43 | 33 | 0 |
| 0.50-0.60 | 7 | 0 | 0 |
| 0.60-0.70 | 3 | 3 | 3 |
| 0.70-0.80 | 2 | 3 | 2 |
| 0.80-0.90 | 3 | 8 | 21 |
| ≥ 0.90 | 2 | 13 | 34 |

- OFF 均值：0.394，ON 均值：0.503，理论上限均值：0.890
- ON 相对 OFF 向理论上限的空间收敛度：均值 31.4%（0% = 没挪，100% = 达到理论上限）
  - p25/p50/p75: 0.0% / 0.0% / 100.0%

## 8. 结论

- 配对样本 64 个场景，主目标总勾数 OFF=2779、ON=2779，变化 +0（净改善或持平）。
- **没有任何场景的勾数因开启而增加**，mitigation 生效。
- Earliness 总和 OFF=15615、ON=15491，变化 -124。

