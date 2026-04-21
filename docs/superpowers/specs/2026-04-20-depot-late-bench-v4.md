# 大库后调车 bench 分析报告

- 输入集：`data/validation_inputs/truth`（127 场景）
- 预算：`timeout_seconds=120` / `solver_time_budget_ms=115000`
- 求解器：`solver=exact`，`anytime_fallback=True`
- 并行度：6 workers
- OFF 运行：`enable_depot_late_scheduling=False`
- ON  运行：`enable_depot_late_scheduling=True`

## 1. 整体成败

| 指标 | OFF | ON | 变化 |
|---|---:|---:|---|
| 场景总数 | 127 | 127 | - |
| 返回 plan | 117 (92.1%) | 117 (92.1%) | +0 |
| verifier 通过 | 116 (91.3%) | 116 (91.3%) | +0 |
| verifier 失败（solved=True, is_valid=False） | 1 | 1 | +0 |
| 容量预检拒绝 | 10 | 10 | +0 |
| 空返（no solution within budget） | 0 | 0 | +0 |
| 其他错误 | 0 | 0 | +0 |

## 2. 勾数分布（solved 场景）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 总和 | 3289 | 3275 |
| 均值 | 28.11 | 27.99 |
| 中位数 | 26 | 26 |
| p95 | 41 | 41 |
| max | 203 | 203 |

## 3. 大库钩指标（solved 场景）

| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |
|---|---:|---:|---:|---:|
| depot_hook_count | 9.13 | 13 | 9.13 | 13 |
| depot_earliness | 177.91 | 324 | 174.75 | 314 |

**Earliness 总和**：OFF=20816，ON=20446，绝对下降 370，相对下降 1.8% 。

## 4. Fallback stage 命中

| Stage | OFF | ON |
|---|---:|---:|
| `UNSOLVED (final arrangement exceeds track capacity)` | 10 | 10 |
| `beam` | 96 | 96 |
| `constructive` | 1 | 1 |
| `constructive_partial` | 1 | 1 |
| `exact` | 2 | 2 |
| `weighted` | 15 | 15 |
| `weighted_greedy` | 1 | 1 |
| `weighted_very_greedy` | 1 | 1 |

## 5. 配对比较（solved in both）

配对样本数：**117**

### 5.1 勾数对比

- ON 更少勾数：**4** 场景（3.4%）
- 持平：**112** 场景（95.7%）
- ON 更多勾数：**1** 场景（0.9%）—— **关键回归指标**

### 5.2 Earliness 对比（大库钩数相同的子集）

同 depot_hook_count 子集（N=117）：
- ON earliness 更小（大库更靠后）：**21** 场景
- 持平：**96** 场景
- ON earliness 更大（大库更靠前）：**0** 场景
- 总 earliness 绝对下降：**370**

全部配对（不同 depot_count 也纳入，仅供参考）：
- 总 earliness 变化：**-370**

### 5.3 勾数回归明细（ON 比 OFF 多）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_2025_11_10_afternoon.json` | 40 | 41 | **+1** | beam | beam |

### 5.4 勾数下降明细（ON 比 OFF 少）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_2025_09_09_noon.json` | 49 | 43 | **-6** | beam | beam |
| `validation_2025_09_08_noon.json` | 40 | 36 | **-4** | beam | beam |
| `validation_20260121Z.json` | 43 | 39 | **-4** | beam | beam |
| `validation_20260318W.json` | 39 | 38 | **-1** | beam | beam |

### 5.5 Earliness top improvements（同大库钩数）

| Scenario | depot_count | OFF earl | ON earl | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---:|---|---|
| `validation_2025_09_09_noon.json` | 9 | 351 | 293 | **-58** | beam | beam |
| `validation_20260121Z.json` | 8 | 273 | 218 | **-55** | beam | beam |
| `validation_2025_09_08_noon.json` | 9 | 282 | 254 | **-28** | beam | beam |
| `validation_20260225Z.json` | 8 | 88 | 64 | **-24** | weighted | weighted |
| `validation_20260123W.json` | 8 | 56 | 37 | **-19** | weighted | weighted |
| `validation_20260206W.json` | 6 | 73 | 54 | **-19** | beam | beam |
| `validation_20260312W.json` | 11 | 83 | 64 | **-19** | weighted | weighted |
| `validation_20260116W.json` | 13 | 122 | 104 | **-18** | weighted | weighted |
| `validation_20260203Z.json` | 10 | 162 | 147 | **-15** | beam | beam |
| `validation_20260318W.json` | 10 | 306 | 291 | **-15** | beam | beam |
| `validation_20260202W.json` | 8 | 107 | 93 | **-14** | beam | beam |
| `validation_2025_12_08_noon.json` | 7 | 86 | 73 | **-13** | beam | beam |
| `validation_2025_11_11_noon.json` | 7 | 217 | 206 | **-11** | beam | beam |
| `validation_20260224W.json` | 7 | 68 | 57 | **-11** | beam | beam |
| `validation_20260228W.json` | 10 | 86 | 75 | **-11** | weighted | weighted |

## 6. 运行时间分布（solved 场景，elapsed_ms）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 均值 (ms) | 3782 | 3832 |
| 中位数 (ms) | 944 | 899 |
| p95 (ms) | 14222 | 15663 |
| max (ms) | 24908 | 27881 |

## 7. 大库钩平均位置分布（fraction = 平均索引 / N）

对每个 solved 且有大库钩的场景计算：
- `avg_frac = depot_index_sum / (K * N)` = 大库钩平均位置占 plan 长度的比例
- `theoretical_max = (2N - K + 1) / (2N)` = 若把全部 K 个大库钩塞到 plan 最后 K 位时的平均比例
  （该上限不考虑依赖约束，纯粹是 pigeonhole）

| Bucket (avg depot position fraction) | OFF | ON | 理论上限 |
|---|---:|---:|---:|
| < 0.50 | 106 | 99 | 0 |
| 0.50-0.60 | 10 | 14 | 0 |
| 0.60-0.70 | 1 | 4 | 4 |
| 0.70-0.80 | 0 | 0 | 20 |
| 0.80-0.90 | 0 | 0 | 79 |
| ≥ 0.90 | 0 | 0 | 14 |

- OFF 均值：0.373，ON 均值：0.386，理论上限均值：0.835
- ON 相对 OFF 向理论上限的空间收敛度：均值 4.1%（0% = 没挪，100% = 达到理论上限）
  - p25/p50/p75: 0.0% / 0.0% / 0.0%

## 8. 结论

- 配对样本 117 个场景，主目标总勾数 OFF=3289、ON=3275，变化 -14（净改善或持平）。
- 仍有 **1** 个场景勾数上升，需进一步排查。
- Earliness 总和 OFF=20816、ON=20446，变化 -370。

