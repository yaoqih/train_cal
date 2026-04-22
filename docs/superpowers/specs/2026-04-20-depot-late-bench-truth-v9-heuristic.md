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
| verifier 通过 | 117 (92.1%) | 117 (92.1%) | +0 |
| verifier 失败（solved=True, is_valid=False） | 0 | 0 | +0 |
| 容量预检拒绝 | 10 | 10 | +0 |
| 空返（no solution within budget） | 0 | 0 | +0 |
| 其他错误 | 0 | 0 | +0 |

## 2. 勾数分布（solved 场景）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 总和 | 3749 | 3749 |
| 均值 | 32.04 | 32.04 |
| 中位数 | 26 | 26 |
| p95 | 59 | 59 |
| max | 354 | 354 |

## 3. 大库钩指标（solved 场景）

| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |
|---|---:|---:|---:|---:|
| depot_hook_count | 9.15 | 13 | 9.15 | 13 |
| depot_earliness | 199.91 | 386 | 189.38 | 371 |

**Earliness 总和**：OFF=23390，ON=22157，绝对下降 1233，相对下降 5.3% 。

## 4. Fallback stage 命中

| Stage | OFF | ON |
|---|---:|---:|
| `UNSOLVED (final arrangement exceeds track capacity)` | 10 | 10 |
| `beam` | 95 | 95 |
| `constructive` | 2 | 2 |
| `exact` | 2 | 2 |
| `weighted` | 15 | 15 |
| `weighted_greedy` | 2 | 2 |
| `weighted_very_greedy` | 1 | 1 |

## 5. 配对比较（solved in both）

配对样本数：**117**

### 5.1 勾数对比

- ON 更少勾数：**0** 场景（0.0%）
- 持平：**117** 场景（100.0%）
- ON 更多勾数：**0** 场景（0.0%）—— **关键回归指标**

### 5.2 Earliness 对比（大库钩数相同的子集）

同 depot_hook_count 子集（N=117）：
- ON earliness 更小（大库更靠后）：**46** 场景
- 持平：**71** 场景
- ON earliness 更大（大库更靠前）：**0** 场景
- 总 earliness 绝对下降：**1233**

全部配对（不同 depot_count 也纳入，仅供参考）：
- 总 earliness 变化：**-1233**

### 5.3 勾数回归明细（ON 比 OFF 多）

**无勾数回归。**

### 5.4 勾数下降明细（ON 比 OFF 少）

（无）

### 5.5 Earliness top improvements（同大库钩数）

| Scenario | depot_count | OFF earl | ON earl | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---:|---|---|
| `validation_20260203Z.json` | 10 | 162 | 45 | **-117** | beam | beam |
| `validation_2025_09_08_afternoon.json` | 8 | 196 | 94 | **-102** | beam | beam |
| `validation_20260120W.json` | 14 | 314 | 224 | **-90** | beam | beam |
| `validation_20260130W.json` | 11 | 155 | 81 | **-74** | beam | beam |
| `validation_20260210W.json` | 9 | 235 | 174 | **-61** | beam | beam |
| `validation_20260121W.json` | 11 | 188 | 133 | **-55** | beam | beam |
| `validation_20260116W.json` | 13 | 122 | 78 | **-44** | weighted | weighted |
| `validation_20260224W.json` | 7 | 68 | 29 | **-39** | beam | beam |
| `validation_20260228W.json` | 10 | 86 | 48 | **-38** | weighted | weighted |
| `validation_2025_11_07_noon.json` | 4 | 148 | 111 | **-37** | beam | beam |
| `validation_20260326W.json` | 8 | 95 | 58 | **-37** | beam | beam |
| `validation_20260209W.json` | 9 | 176 | 143 | **-33** | beam | beam |
| `validation_20260204Z.json` | 9 | 108 | 80 | **-28** | beam | beam |
| `validation_20260225Z.json` | 8 | 88 | 60 | **-28** | weighted | weighted |
| `validation_20260114W.json` | 11 | 131 | 105 | **-26** | beam | beam |

## 6. 运行时间分布（solved 场景，elapsed_ms）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 均值 (ms) | 2839 | 2827 |
| 中位数 (ms) | 923 | 905 |
| p95 (ms) | 9202 | 9205 |
| max (ms) | 17027 | 16072 |

## 7. 大库钩平均位置分布（fraction = 平均索引 / N）

对每个 solved 且有大库钩的场景计算：
- `avg_frac = depot_index_sum / (K * N)` = 大库钩平均位置占 plan 长度的比例
- `theoretical_max = (2N - K + 1) / (2N)` = 若把全部 K 个大库钩塞到 plan 最后 K 位时的平均比例
  （该上限不考虑依赖约束，纯粹是 pigeonhole）

| Bucket (avg depot position fraction) | OFF | ON | 理论上限 |
|---|---:|---:|---:|
| < 0.50 | 105 | 89 | 0 |
| 0.50-0.60 | 11 | 18 | 0 |
| 0.60-0.70 | 1 | 6 | 4 |
| 0.70-0.80 | 0 | 3 | 20 |
| 0.80-0.90 | 0 | 1 | 77 |
| ≥ 0.90 | 0 | 0 | 16 |

- OFF 均值：0.371，ON 均值：0.419，理论上限均值：0.838
- ON 相对 OFF 向理论上限的空间收敛度：均值 12.0%（0% = 没挪，100% = 达到理论上限）
  - p25/p50/p75: 0.0% / 0.0% / 15.6%

## 8. 结论

- 配对样本 117 个场景，主目标总勾数 OFF=3749、ON=3749，变化 +0（净改善或持平）。
- **没有任何场景的勾数因开启而增加**，mitigation 生效。
- Earliness 总和 OFF=23390、ON=22157，变化 -1233。

