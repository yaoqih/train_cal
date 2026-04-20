# Right-Middle SVG Polish Design

**Date:** 2026-04-19

## Goal

在不改变当前连续线路拓扑和整体横向 ribbon 形态的前提下，继续细化右中部小线簇的视觉质量，让 `存1 / 临2 / 渡5 / 渡6 / 机北` 一带更松、更顺眼。

## Scope

本轮只做局部顺眼化，不触碰以下内容：

- 不改 `segmented_physical_routes.json`
- 不改主节点拓扑顺序
- 不重排整体层级
- 不把实验脚本并入 `app.py`

允许的改动：

- 微调右中部几条短线的标签锚点偏移
- 为个别标签增加专用上下基线偏移
- 对局部接入段做很小的视觉错位，但不改变连接关系

## Target Issues

当前右中部仍有三个值得修的小问题：

1. `存1` 与 `临2` 标签横向距离偏近
2. `渡5` 标签贴到 `渡4` 斜线附近
3. `机北` 标签与上方 `渡7` 接入线的气口偏紧

## Approach

采用“标签优先、折角最小”的方案：

- 先用测试把右中部的最小标签间距和标签到邻近轨道的最小清障距离固化
- 优先改 `TRACK_TEXT_ANCHOR_BIASES`
- 如果单纯横向偏移不够，再对个别轨道补很小的 `TRACK_TEXT_BASELINE_OVERRIDE_OFFSETS`
- 只在必要时再调整右中部短线的 join offset；若标签层即可解决，则不碰线条

## Verification

- `tests/tools/test_segmented_routes_svg.py`
- `tests/domain/test_master_data.py`
- `tests/demo/test_topology_layout.py`
- 重导出：
  - `artifacts/segmented_routes_experiment/segmented_physical_routes.svg`
  - `artifacts/segmented_routes_experiment/segmented_physical_routes.png`
