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
| verifier 通过 | 116 (91.3%) | 115 (90.6%) | -1 |
| verifier 失败（solved=True, is_valid=False） | 1 | 2 | +1 |
| 容量预检拒绝 | 10 | 10 | +0 |
| 空返（no solution within budget） | 0 | 0 | +0 |
| 其他错误 | 0 | 0 | +0 |

## 2. 勾数分布（solved 场景）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 总和 | 3285 | 3259 |
| 均值 | 28.08 | 27.85 |
| 中位数 | 26 | 26 |
| p95 | 40 | 41 |
| max | 203 | 203 |

## 3. 大库钩指标（solved 场景）

| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |
|---|---:|---:|---:|---:|
| depot_hook_count | 9.13 | 13 | 9.12 | 13 |
| depot_earliness | 177.44 | 324 | 170.44 | 308 |

**Earliness 总和**：OFF=20761，ON=19942，绝对下降 819，相对下降 3.9% 。

## 4. Fallback stage 命中

| Stage | OFF | ON |
|---|---:|---:|
| `UNSOLVED (final arrangement exceeds track capacity)` | 10 | 10 |
| `beam` | 96 | 97 |
| `constructive` | 1 | 1 |
| `constructive_partial` | 1 | 2 |
| `exact` | 2 | 2 |
| `weighted` | 15 | 14 |
| `weighted_greedy` | 1 | 1 |
| `weighted_very_greedy` | 1 | 0 |

## 5. 配对比较（solved in both）

配对样本数：**117**

### 5.1 勾数对比

- ON 更少勾数：**4** 场景（3.4%）
- 持平：**111** 场景（94.9%）
- ON 更多勾数：**2** 场景（1.7%）—— **关键回归指标**

### 5.2 Earliness 对比（大库钩数相同的子集）

同 depot_hook_count 子集（N=116）：
- ON earliness 更小（大库更靠后）：**19** 场景
- 持平：**97** 场景
- ON earliness 更大（大库更靠前）：**0** 场景
- 总 earliness 绝对下降：**620**

全部配对（不同 depot_count 也纳入，仅供参考）：
- 总 earliness 变化：**-819**

### 5.3 勾数回归明细（ON 比 OFF 多）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_20260228W.json` | 17 | 19 | **+2** | weighted | beam |
| `validation_2025_11_10_afternoon.json` | 40 | 41 | **+1** | beam | beam |

### 5.4 勾数下降明细（ON 比 OFF 少）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_20260310W.json` | 40 | 22 | **-18** | weighted_very_greedy | constructive_partial |
| `validation_2025_09_09_noon.json` | 49 | 43 | **-6** | beam | beam |
| `validation_2025_09_08_noon.json` | 40 | 36 | **-4** | beam | beam |
| `validation_20260318W.json` | 39 | 38 | **-1** | beam | beam |

### 5.5 Earliness top improvements（同大库钩数）

| Scenario | depot_count | OFF earl | ON earl | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---:|---|---|
| `validation_20260203Z.json` | 10 | 162 | 45 | **-117** | beam | beam |
| `validation_20260130W.json` | 11 | 155 | 75 | **-80** | beam | beam |
| `validation_2025_09_09_noon.json` | 9 | 351 | 293 | **-58** | beam | beam |
| `validation_20260116W.json` | 13 | 122 | 78 | **-44** | weighted | weighted |
| `validation_20260224W.json` | 7 | 68 | 27 | **-41** | beam | beam |
| `validation_20260225Z.json` | 8 | 88 | 48 | **-40** | weighted | weighted |
| `validation_20260112W.json` | 9 | 78 | 45 | **-33** | weighted | weighted |
| `validation_20260228W.json` | 10 | 86 | 55 | **-31** | weighted | beam |
| `validation_2025_09_08_noon.json` | 9 | 282 | 254 | **-28** | beam | beam |
| `validation_20260202W.json` | 8 | 107 | 79 | **-28** | beam | beam |
| `validation_20260206W.json` | 6 | 73 | 48 | **-25** | beam | beam |
| `validation_20260123W.json` | 8 | 56 | 34 | **-22** | weighted | weighted |
| `validation_20260312W.json` | 11 | 83 | 64 | **-19** | weighted | weighted |
| `validation_20260318W.json` | 10 | 306 | 291 | **-15** | beam | beam |
| `validation_2025_11_07_noon.json` | 4 | 68 | 56 | **-12** | beam | beam |

## 6. 运行时间分布（solved 场景，elapsed_ms）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 均值 (ms) | 3850 | 3956 |
| 中位数 (ms) | 955 | 1066 |
| p95 (ms) | 14640 | 17874 |
| max (ms) | 27952 | 27850 |

## 7. 大库钩平均位置分布（fraction = 平均索引 / N）

对每个 solved 且有大库钩的场景计算：
- `avg_frac = depot_index_sum / (K * N)` = 大库钩平均位置占 plan 长度的比例
- `theoretical_max = (2N - K + 1) / (2N)` = 若把全部 K 个大库钩塞到 plan 最后 K 位时的平均比例
  （该上限不考虑依赖约束，纯粹是 pigeonhole）

| Bucket (avg depot position fraction) | OFF | ON | 理论上限 |
|---|---:|---:|---:|
| < 0.50 | 106 | 95 | 0 |
| 0.50-0.60 | 10 | 11 | 0 |
| 0.60-0.70 | 1 | 5 | 4 |
| 0.70-0.80 | 0 | 5 | 20 |
| 0.80-0.90 | 0 | 1 | 79 |
| ≥ 0.90 | 0 | 0 | 14 |

- OFF 均值：0.373，ON 均值：0.402，理论上限均值：0.835
- ON 相对 OFF 向理论上限的空间收敛度：均值 8.3%（0% = 没挪，100% = 达到理论上限）
  - p25/p50/p75: 0.0% / 0.0% / 0.0%

## 8. 结论

- 配对样本 117 个场景，主目标总勾数 OFF=3285、ON=3259，变化 -26（净改善或持平）。
- 仍有 **2** 个场景勾数上升，需进一步排查。
- Earliness 总和 OFF=20761、ON=19942，变化 -819。

