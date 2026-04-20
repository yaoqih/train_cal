"""Analyze depot-late-scheduling bench outputs (flag OFF vs ON).

Reads two summary.json files produced by run_external_validation_parallel.py
and writes a Markdown analysis report covering:

- Solve success and validity
- Hook count distribution
- Depot earliness distribution
- Fallback stage breakdown
- Per-scenario regressions and wins
- Runtime distribution

Usage:
    python scripts/analyze_depot_late_bench.py \\
        --off artifacts/depot_late_bench_off \\
        --on artifacts/depot_late_bench_on \\
        --out docs/superpowers/specs/2026-04-20-depot-late-bench.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScenarioResult:
    scenario: str
    solved: bool
    hook_count: int | None
    is_valid: bool | None
    is_proven_optimal: bool | None
    fallback_stage: str | None
    depot_earliness: int | None
    depot_hook_count: int | None
    elapsed_ms: float | None
    error: str | None


def load_summary(path: Path) -> tuple[dict, list[ScenarioResult]]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    results = []
    for r in payload.get("results", []):
        results.append(
            ScenarioResult(
                scenario=r["scenario"],
                solved=r.get("solved", False),
                hook_count=r.get("hook_count"),
                is_valid=r.get("is_valid"),
                is_proven_optimal=r.get("is_proven_optimal"),
                fallback_stage=r.get("fallback_stage"),
                depot_earliness=r.get("depot_earliness"),
                depot_hook_count=r.get("depot_hook_count"),
                elapsed_ms=r.get("elapsed_ms"),
                error=r.get("error"),
            )
        )
    return payload, results


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = int(round((len(s) - 1) * pct / 100))
    return s[k]


def pct(n: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{100*n/total:.1f}%"


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "sum": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def summarize_stage(results: list[ScenarioResult]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in results:
        if not r.solved:
            c[f"UNSOLVED ({r.error[:40] if r.error else 'unknown'})"] += 1
        else:
            c[r.fallback_stage or "unknown"] += 1
    return c


def by_scenario(results: list[ScenarioResult]) -> dict[str, ScenarioResult]:
    return {r.scenario: r for r in results}


def make_report(
    meta_off: dict, off: list[ScenarioResult],
    meta_on: dict, on: list[ScenarioResult],
) -> str:
    off_by = by_scenario(off)
    on_by = by_scenario(on)
    all_scenarios = sorted(set(off_by) | set(on_by))

    lines: list[str] = []
    w = lines.append

    w("# 大库后调车 bench 分析报告")
    w("")
    w(f"- 输入集：`data/validation_inputs/truth`（{len(all_scenarios)} 场景）")
    w(f"- 预算：`timeout_seconds={meta_off.get('timeout_seconds')}` / "
      f"`solver_time_budget_ms={meta_off.get('solver_time_budget_ms')}`")
    w(f"- 求解器：`solver={meta_off.get('solver')}`，"
      f"`anytime_fallback={meta_off.get('enable_anytime_fallback')}`")
    w(f"- 并行度：6 workers")
    w(f"- OFF 运行：`enable_depot_late_scheduling={meta_off.get('enable_depot_late_scheduling', False)}`")
    w(f"- ON  运行：`enable_depot_late_scheduling={meta_on.get('enable_depot_late_scheduling', True)}`")
    w("")

    # === 1. 整体成败 ===
    w("## 1. 整体成败")
    w("")
    w("| 指标 | OFF | ON | 变化 |")
    w("|---|---:|---:|---|")

    def count_solved(rs):
        return sum(1 for r in rs if r.solved)
    def count_valid(rs):
        return sum(1 for r in rs if r.solved and r.is_valid)
    def count_unsolved_preflight(rs):
        return sum(1 for r in rs if not r.solved and r.error and "capacity" in (r.error or ""))
    def count_unsolved_nosolution(rs):
        return sum(1 for r in rs if not r.solved and r.error and "no solution" in (r.error or ""))
    def count_unsolved_other(rs):
        return sum(1 for r in rs if not r.solved and not (r.error and ("capacity" in r.error or "no solution" in r.error)))
    def count_invalid(rs):
        return sum(1 for r in rs if r.solved and r.is_valid is False)

    total = len(off)
    w(f"| 场景总数 | {len(off)} | {len(on)} | - |")
    w(f"| 返回 plan | {count_solved(off)} ({pct(count_solved(off), total)}) | "
      f"{count_solved(on)} ({pct(count_solved(on), total)}) | "
      f"{count_solved(on)-count_solved(off):+d} |")
    w(f"| verifier 通过 | {count_valid(off)} ({pct(count_valid(off), total)}) | "
      f"{count_valid(on)} ({pct(count_valid(on), total)}) | "
      f"{count_valid(on)-count_valid(off):+d} |")
    w(f"| verifier 失败（solved=True, is_valid=False） | {count_invalid(off)} | {count_invalid(on)} | "
      f"{count_invalid(on)-count_invalid(off):+d} |")
    w(f"| 容量预检拒绝 | {count_unsolved_preflight(off)} | {count_unsolved_preflight(on)} | "
      f"{count_unsolved_preflight(on)-count_unsolved_preflight(off):+d} |")
    w(f"| 空返（no solution within budget） | {count_unsolved_nosolution(off)} | {count_unsolved_nosolution(on)} | "
      f"{count_unsolved_nosolution(on)-count_unsolved_nosolution(off):+d} |")
    w(f"| 其他错误 | {count_unsolved_other(off)} | {count_unsolved_other(on)} | "
      f"{count_unsolved_other(on)-count_unsolved_other(off):+d} |")
    w("")

    # === 2. 勾数分布 ===
    w("## 2. 勾数分布（solved 场景）")
    w("")
    off_hooks = [r.hook_count for r in off if r.solved and r.hook_count is not None]
    on_hooks = [r.hook_count for r in on if r.solved and r.hook_count is not None]
    dist_off = distribution(off_hooks)
    dist_on = distribution(on_hooks)
    w("| 统计 | OFF | ON |")
    w("|---|---:|---:|")
    w(f"| N | {dist_off['n']} | {dist_on['n']} |")
    w(f"| 总和 | {dist_off['sum']} | {dist_on['sum']} |")
    w(f"| 均值 | {dist_off['mean']:.2f} | {dist_on['mean']:.2f} |" if dist_off['mean'] else "| 均值 | - | - |")
    w(f"| 中位数 | {dist_off['median']} | {dist_on['median']} |")
    w(f"| p95 | {dist_off['p95']} | {dist_on['p95']} |")
    w(f"| max | {dist_off['max']} | {dist_on['max']} |")
    w("")

    # === 3. 大库钩数和 earliness ===
    w("## 3. 大库钩指标（solved 场景）")
    w("")
    off_depot = [r.depot_hook_count for r in off if r.solved and r.depot_hook_count is not None]
    on_depot = [r.depot_hook_count for r in on if r.solved and r.depot_hook_count is not None]
    off_earl = [r.depot_earliness for r in off if r.solved and r.depot_earliness is not None]
    on_earl = [r.depot_earliness for r in on if r.solved and r.depot_earliness is not None]
    w("| 指标 | OFF 均值 | OFF p95 | ON 均值 | ON p95 |")
    w("|---|---:|---:|---:|---:|")
    if off_depot and on_depot:
        w(f"| depot_hook_count | {statistics.mean(off_depot):.2f} | {percentile(off_depot, 95)} | "
          f"{statistics.mean(on_depot):.2f} | {percentile(on_depot, 95)} |")
    if off_earl and on_earl:
        w(f"| depot_earliness | {statistics.mean(off_earl):.2f} | {percentile(off_earl, 95)} | "
          f"{statistics.mean(on_earl):.2f} | {percentile(on_earl, 95)} |")
    w("")
    if off_earl and on_earl:
        total_off = sum(off_earl)
        total_on = sum(on_earl)
        w(f"**Earliness 总和**：OFF={total_off}，ON={total_on}，"
          f"绝对下降 {total_off - total_on}，相对下降 "
          f"{100*(total_off - total_on)/total_off:.1f}% 。")
        w("")

    # === 4. fallback stage 分布 ===
    w("## 4. Fallback stage 命中")
    w("")
    off_stages = summarize_stage(off)
    on_stages = summarize_stage(on)
    all_stages = sorted(set(off_stages) | set(on_stages))
    w("| Stage | OFF | ON |")
    w("|---|---:|---:|")
    for s in all_stages:
        w(f"| `{s}` | {off_stages.get(s, 0)} | {on_stages.get(s, 0)} |")
    w("")

    # === 5. 配对比较（同一 scenario，OFF vs ON） ===
    w("## 5. 配对比较（solved in both）")
    w("")
    matched = [
        (off_by[s], on_by[s]) for s in all_scenarios
        if s in off_by and s in on_by
        and off_by[s].solved and on_by[s].solved
        and off_by[s].hook_count is not None and on_by[s].hook_count is not None
    ]
    w(f"配对样本数：**{len(matched)}**")
    w("")
    if matched:
        hook_deltas = [o.hook_count - f.hook_count for f, o in matched]
        hook_wins = sum(1 for d in hook_deltas if d < 0)  # ON fewer hooks
        hook_same = sum(1 for d in hook_deltas if d == 0)
        hook_regress = sum(1 for d in hook_deltas if d > 0)
        w("### 5.1 勾数对比")
        w("")
        w(f"- ON 更少勾数：**{hook_wins}** 场景（{pct(hook_wins, len(matched))}）")
        w(f"- 持平：**{hook_same}** 场景（{pct(hook_same, len(matched))}）")
        w(f"- ON 更多勾数：**{hook_regress}** 场景（{pct(hook_regress, len(matched))}）—— **关键回归指标**")
        w("")

        earl_pairs = [(f, o) for f, o in matched if f.depot_earliness is not None and o.depot_earliness is not None]
        if earl_pairs:
            earl_deltas = [o.depot_earliness - f.depot_earliness for f, o in earl_pairs]
            earl_wins = sum(1 for d in earl_deltas if d < 0)
            earl_same = sum(1 for d in earl_deltas if d == 0)
            earl_regress = sum(1 for d in earl_deltas if d > 0)
            w("### 5.2 Earliness 对比（大库钩数相同的子集）")
            w("")
            same_depot_pairs = [(f, o) for f, o in earl_pairs if f.depot_hook_count == o.depot_hook_count]
            if same_depot_pairs:
                same_deltas = [o.depot_earliness - f.depot_earliness for f, o in same_depot_pairs]
                w(f"同 depot_hook_count 子集（N={len(same_depot_pairs)}）：")
                w(f"- ON earliness 更小（大库更靠后）：**{sum(1 for d in same_deltas if d < 0)}** 场景")
                w(f"- 持平：**{sum(1 for d in same_deltas if d == 0)}** 场景")
                w(f"- ON earliness 更大（大库更靠前）：**{sum(1 for d in same_deltas if d > 0)}** 场景")
                w(f"- 总 earliness 绝对下降：**{-sum(same_deltas)}**")
                w("")
            w("全部配对（不同 depot_count 也纳入，仅供参考）：")
            w(f"- 总 earliness 变化：**{sum(earl_deltas):+d}**")
            w("")

        # Regression details
        w("### 5.3 勾数回归明细（ON 比 OFF 多）")
        w("")
        regressions = [(f, o) for f, o in matched if o.hook_count > f.hook_count]
        if regressions:
            w("| Scenario | OFF | ON | Δ | OFF stage | ON stage |")
            w("|---|---:|---:|---:|---|---|")
            for f, o in sorted(regressions, key=lambda x: x[1].hook_count - x[0].hook_count, reverse=True)[:20]:
                w(f"| `{f.scenario}` | {f.hook_count} | {o.hook_count} | "
                  f"**+{o.hook_count - f.hook_count}** | {f.fallback_stage} | {o.fallback_stage} |")
            w("")
        else:
            w("**无勾数回归。**")
            w("")

        # Wins
        w("### 5.4 勾数下降明细（ON 比 OFF 少）")
        w("")
        wins = [(f, o) for f, o in matched if o.hook_count < f.hook_count]
        if wins:
            w("| Scenario | OFF | ON | Δ | OFF stage | ON stage |")
            w("|---|---:|---:|---:|---|---|")
            for f, o in sorted(wins, key=lambda x: x[0].hook_count - x[1].hook_count, reverse=True)[:20]:
                w(f"| `{f.scenario}` | {f.hook_count} | {o.hook_count} | "
                  f"**{o.hook_count - f.hook_count:+d}** | {f.fallback_stage} | {o.fallback_stage} |")
            w("")
        else:
            w("（无）")
            w("")

        # Earliness improvements (same depot hook count)
        w("### 5.5 Earliness top improvements（同大库钩数）")
        w("")
        if same_depot_pairs:
            sorted_pairs = sorted(same_depot_pairs, key=lambda x: x[1].depot_earliness - x[0].depot_earliness)
            w("| Scenario | depot_count | OFF earl | ON earl | Δ | OFF stage | ON stage |")
            w("|---|---:|---:|---:|---:|---|---|")
            for f, o in sorted_pairs[:15]:
                delta = o.depot_earliness - f.depot_earliness
                if delta >= 0:
                    break
                w(f"| `{f.scenario}` | {f.depot_hook_count} | {f.depot_earliness} | "
                  f"{o.depot_earliness} | **{delta:+d}** | {f.fallback_stage} | {o.fallback_stage} |")
            w("")

    # === 6. 运行时间分布 ===
    w("## 6. 运行时间分布（solved 场景，elapsed_ms）")
    w("")
    off_ms = [r.elapsed_ms for r in off if r.solved and r.elapsed_ms is not None]
    on_ms = [r.elapsed_ms for r in on if r.solved and r.elapsed_ms is not None]
    d_off_ms = distribution(off_ms)
    d_on_ms = distribution(on_ms)
    w("| 统计 | OFF | ON |")
    w("|---|---:|---:|")
    w(f"| N | {d_off_ms['n']} | {d_on_ms['n']} |")
    if d_off_ms['mean'] and d_on_ms['mean']:
        w(f"| 均值 (ms) | {d_off_ms['mean']:.0f} | {d_on_ms['mean']:.0f} |")
        w(f"| 中位数 (ms) | {d_off_ms['median']:.0f} | {d_on_ms['median']:.0f} |")
        w(f"| p95 (ms) | {d_off_ms['p95']:.0f} | {d_on_ms['p95']:.0f} |")
        w(f"| max (ms) | {d_off_ms['max']:.0f} | {d_on_ms['max']:.0f} |")
    w("")

    # === 7. 结论 ===
    w("## 7. 结论")
    w("")
    if matched:
        total_off_hooks = sum(f.hook_count for f, _ in matched)
        total_on_hooks = sum(o.hook_count for _, o in matched)
        hook_change = total_on_hooks - total_off_hooks
        w(f"- 配对样本 {len(matched)} 个场景，主目标总勾数 OFF={total_off_hooks}、ON={total_on_hooks}，"
          f"变化 {hook_change:+d}（{'净回归' if hook_change > 0 else '净改善或持平'}）。")
        if hook_regress == 0:
            w("- **没有任何场景的勾数因开启而增加**，mitigation 生效。")
        else:
            w(f"- 仍有 **{hook_regress}** 个场景勾数上升，需进一步排查。")
        if earl_pairs:
            total_earl_off = sum(f.depot_earliness for f, _ in earl_pairs)
            total_earl_on = sum(o.depot_earliness for _, o in earl_pairs)
            w(f"- Earliness 总和 OFF={total_earl_off}、ON={total_earl_on}，"
              f"变化 {total_earl_on - total_earl_off:+d}。")
    w("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True, help="OFF bench output dir (containing summary.json)")
    parser.add_argument("--on", type=Path, required=True, help="ON bench output dir")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown path")
    args = parser.parse_args()

    meta_off, off = load_summary(args.off)
    meta_on, on = load_summary(args.on)
    report = make_report(meta_off, off, meta_on, on)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
