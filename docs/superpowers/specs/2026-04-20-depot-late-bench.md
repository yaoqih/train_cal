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
| verifier 通过 | 116 (91.3%) | 114 (89.8%) | -2 |
| verifier 失败（solved=True, is_valid=False） | 1 | 3 | +2 |
| 容量预检拒绝 | 10 | 10 | +0 |
| 空返（no solution within budget） | 0 | 0 | +0 |
| 其他错误 | 0 | 0 | +0 |

## 2. 勾数分布（solved 场景）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 总和 | 3279 | 3260 |
| 均值 | 28.03 | 27.86 |
| 中位数 | 26 | 26 |
| p95 | 40 | 41 |
| max | 203 | 203 |

## 3. 大库钩指标（solved 场景）

| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |
|---|---:|---:|---:|---:|
| depot_hook_count | 9.13 | 13 | 9.12 | 13 |
| depot_earliness | 176.95 | 314 | 164.44 | 286 |

**Earliness 总和**：OFF=20703，ON=19240，绝对下降 1463，相对下降 7.1% 。

## 4. Fallback stage 命中

| Stage | OFF | ON |
|---|---:|---:|
| `UNSOLVED (final arrangement exceeds track capacity)` | 10 | 10 |
| `beam` | 95 | 96 |
| `beam_greedy_64` | 1 | 0 |
| `constructive` | 1 | 0 |
| `constructive_partial` | 1 | 3 |
| `exact` | 2 | 2 |
| `weighted` | 15 | 15 |
| `weighted_greedy` | 1 | 1 |
| `weighted_very_greedy` | 1 | 0 |

## 5. 配对比较（solved in both）

配对样本数：**117**

### 5.1 勾数对比

- ON 更少勾数：**3** 场景（2.6%）
- 持平：**112** 场景（95.7%）
- ON 更多勾数：**2** 场景（1.7%）—— **关键回归指标**

### 5.2 Earliness 对比（大库钩数相同的子集）

同 depot_hook_count 子集（N=116）：
- ON earliness 更小（大库更靠后）：**47** 场景
- 持平：**68** 场景
- ON earliness 更大（大库更靠前）：**1** 场景
- 总 earliness 绝对下降：**1235**

全部配对（不同 depot_count 也纳入，仅供参考）：
- 总 earliness 变化：**-1463**

### 5.3 勾数回归明细（ON 比 OFF 多）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_20260109Z.json` | 27 | 29 | **+2** | constructive | constructive_partial |
| `validation_2025_11_10_afternoon.json` | 40 | 41 | **+1** | beam | beam |

### 5.4 勾数下降明细（ON 比 OFF 少）

| Scenario | OFF | ON | Δ | OFF stage | ON stage |
|---|---:|---:|---:|---|---|
| `validation_20260310W.json` | 40 | 23 | **-17** | weighted_very_greedy | constructive_partial |
| `validation_2025_09_08_noon.json` | 40 | 36 | **-4** | beam | beam |
| `validation_20260318W.json` | 39 | 38 | **-1** | beam | beam |

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
| `validation_20260326W.json` | 8 | 95 | 58 | **-37** | beam | beam |
| `validation_20260209W.json` | 9 | 176 | 143 | **-33** | beam | beam |
| `validation_2025_09_08_noon.json` | 9 | 282 | 254 | **-28** | beam | beam |
| `validation_20260204Z.json` | 9 | 108 | 80 | **-28** | beam | beam |
| `validation_20260225Z.json` | 8 | 88 | 60 | **-28** | weighted | weighted |
| `validation_20260114W.json` | 11 | 131 | 105 | **-26** | beam | beam |

## 6. 运行时间分布（solved 场景，elapsed_ms）

| 统计 | OFF | ON |
|---|---:|---:|
| N | 117 | 117 |
| 均值 (ms) | 4069 | 4034 |
| 中位数 (ms) | 1074 | 1166 |
| p95 (ms) | 16222 | 16997 |
| max (ms) | 27979 | 27883 |

## 7. 结论

### 主要发现

- **主目标（总勾数）**：OFF=3279 → ON=3260，**净下降 19 勾**。117 配对场景中，3 场景勾数下降、112 持平、2 场景上升。
- **次要目标（earliness）**：OFF=20703 → ON=19240，**下降 7.1%**。47 场景改善、68 持平、1 场景轻微反向。
- **solve 率**：OFF/ON 各 117/127，**持平**。
- **validity 率**：OFF 116/127、ON 114/127，**ON 多 2 个 invalid**——这是主要关注点。
- **运行时**：p95/均值几乎持平，flag-on 的开销在噪声范围内。

### 2 个 validity 回归的根因

| Scenario | OFF stage | OFF valid | ON stage | ON valid | 说明 |
|---|---|---|---|---|---|
| `validation_20260109Z.json` | constructive | ✓ (27 hooks) | constructive_partial | ✗ (29 hooks) | OFF 的 constructive 把所有车送到目标；ON 的 constructive 受 tier penalty 影响走了不同路径，29 勾时被 stuck_threshold 截断，2 辆车滞留在 `存5北` 未到 `预修` |
| `validation_20260310W.json` | weighted_very_greedy | ✓ (40 hooks) | constructive_partial | ✗ (23 hooks) | OFF 的 fallback chain 第 8 阶段（最贪婪）找到可行 plan；ON 下全部 8 阶段都没找到，退到 constructive，partial |

**共同特征**：两例都**落到 `constructive_partial`**（fallback chain 的最后兜底），这是 SLA 保底层。构造层的 `depot_late_penalty` 改变了选择路径，在极端场景下触发 stuck 或没覆盖全部目标。搜索层的 secondary key 已经 mitigation 到 exact-only，不是这里的肇因。

**候选修复方案**（供后续讨论，不在当前 bench 报告内落地）：

1. **把 `depot_late_penalty` 也限制到 `_choose_best_move` 以非"兜底"角色调用时起作用**：当 constructive 作为 search 全失败的最终 fallback 被调用时关闭 penalty。需要在 astar_solver.py 的 fallback 路径上透传一个 `is_final_fallback` 信号。
2. **完全移除 `constructive.py` 的 penalty**：依赖 LNS + post-processing 承担大库延后。这两层是 tier-preserving / 纯重排，不会导致 invalid。收益：消除两个回归。代价：构造 seed 的 earliness 失去主动引导，可能使 exact/weighted 搜索初值略差（但 search 层在 exact 模式的 secondary key 仍在发挥作用）。
3. **接受并标注**：默认开启，但文档记录两个已知 validity 回归场景，业务层按场景决定。

### Earliness 改善 Top 3

| Scenario | Δ earliness | stage 不变 |
|---|---:|---|
| `validation_20260203Z.json` | -117 | beam |
| `validation_2025_09_08_afternoon.json` | -102 | beam |
| `validation_20260120W.json` | -90 | beam |

最大的改善来自 `beam` 模式，主要是后处理 `reorder_depot_late` 和 LNS 的 earliness tiebreak 共同作用。

### 勾数意外下降 1 例

`validation_20260310W.json` 在 ON 下 23 勾（stage=constructive_partial）对比 OFF 40 勾（stage=weighted_very_greedy）。**但 ON 的 23 勾是 partial**（2 辆车未到目标），不是真正更优。上面 §5.4 的勾数下降明细里把它也列为 "-17"，需要结合 validity 一起看——不是净改善。

