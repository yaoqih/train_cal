# 大库钩后调 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让最终钩计划里涉及大库（`修N库内`）的钩在总勾数最小的前提下尽量集中在后半程，通过 opt-in 开关 `enable_depot_late_scheduling` 控制。

**Architecture:** 词典序次级目标 —— 搜索层在 `_priority` 元组里插入 `neg_depot_index_sum` 次级键；LNS `_plan_quality` 和 constructive `_score_move` 同步注入；最终一次纯函数后处理 `reorder_depot_late` 推到最优。关闭开关时产出与当前基线完全一致。

**Tech Stack:** Python 3.x、pydantic、现有 `fzed_shunting` 求解器、pytest。

**Spec:** `docs/superpowers/specs/2026-04-20-depot-late-scheduling-design.md`

**波次结构（可并行单元用 ⏸ 标注依赖边界）：**

- Wave 1（并行）：Task 1（core module）、Task 2（result field）
- ⏸ merge Wave 1
- Wave 2（并行）：Task 3（search）、Task 4（lns）、Task 5（constructive）
- ⏸ merge Wave 2
- Wave 3（串行）：Task 6（API + 后处理）、Task 7（集成 & bench）

---

## Wave 1 — 基础

### Task 1: 核心模块 `solver/depot_late.py` + 单测

**Files:**
- Create: `src/fzed_shunting/solver/depot_late.py`
- Create: `tests/solver/test_depot_late.py`
- Modify: `src/fzed_shunting/solver/constructive.py:37` — 改为 `from fzed_shunting.solver.depot_late import DEPOT_INNER_TRACKS`

#### Step 1.1: 写失败测试 —— `is_depot_hook`

在 `tests/solver/test_depot_late.py` 写入：

```python
"""Unit tests for depot-late-scheduling core helpers."""

from __future__ import annotations

from fzed_shunting.solver.depot_late import (
    DEPOT_INNER_TRACKS,
    depot_earliness,
    depot_index_sum,
    is_depot_hook,
)
from fzed_shunting.solver.types import HookAction


def _hook(source: str, target: str, vehicles: list[str] | None = None) -> HookAction:
    return HookAction(
        source_track=source,
        target_track=target,
        vehicle_nos=vehicles or ["V1"],
        path_tracks=[source, target],
        action_type="PUT",
    )


class TestIsDepotHook:
    def test_target_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("存5北", "修3库内")) is True

    def test_source_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("修1库内", "存4北")) is True

    def test_both_in_depot_is_true(self) -> None:
        assert is_depot_hook(_hook("修1库内", "修2库内")) is True

    def test_non_depot_is_false(self) -> None:
        assert is_depot_hook(_hook("存1", "调北")) is False

    def test_depot_wai_is_false(self) -> None:
        # 修N库外 is gateway storage, NOT counted as depot.
        assert is_depot_hook(_hook("修1库外", "存4北")) is False

    def test_depot_wai_to_depot_nei_is_true(self) -> None:
        # Crossing into 修N库内 does count.
        assert is_depot_hook(_hook("修1库外", "修1库内")) is True

    def test_constant_exposes_all_four_tracks(self) -> None:
        assert DEPOT_INNER_TRACKS == frozenset(
            {"修1库内", "修2库内", "修3库内", "修4库内"}
        )
```

- [ ] **Step 1.1:** Create the test file above.

#### Step 1.2: 运行测试确认失败

Run: `pytest tests/solver/test_depot_late.py::TestIsDepotHook -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fzed_shunting.solver.depot_late'`.

- [ ] **Step 1.2:** Run and confirm failure.

#### Step 1.3: 创建 `depot_late.py` 骨架 + `is_depot_hook`

在 `src/fzed_shunting/solver/depot_late.py` 写入：

```python
"""Depot-late-scheduling helpers.

Pure functions and metrics that express the preference for hooks touching
大库 (修N库内) appearing later in the plan's hook sequence. Consumers:

- search.py: maintains ``depot_index_sum`` incrementally and injects
  ``-depot_index_sum`` as a lexicographic secondary key.
- lns.py: compares plans with (hook_count, depot_earliness, ...) when the
  feature flag is on.
- constructive.py: adds ``depot_late_penalty`` intra-tier.
- astar_solver.py: applies ``reorder_depot_late`` as a final post-processing
  pass before verification.

The scope of "depot-touching" is strictly the four 修N库内 repair tracks;
修N库外 are gateway storage and are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.solver.types import HookAction
from fzed_shunting.verify.replay import ReplayState


DEPOT_INNER_TRACKS: frozenset[str] = frozenset(
    {"修1库内", "修2库内", "修3库内", "修4库内"}
)


def is_depot_hook(hook: HookAction) -> bool:
    """True iff hook's source or target is one of the 修N库内 tracks."""
    return (
        hook.source_track in DEPOT_INNER_TRACKS
        or hook.target_track in DEPOT_INNER_TRACKS
    )
```

Note: further functions (`depot_earliness`, `depot_index_sum`, `reorder_depot_late`,
`_is_early_depot_phase`) are added in later steps.

- [ ] **Step 1.3:** Create the module with the code above.

#### Step 1.4: 运行 `is_depot_hook` 测试确认通过

Run: `pytest tests/solver/test_depot_late.py::TestIsDepotHook -v`
Expected: 7 passed.

- [ ] **Step 1.4:** Run and confirm pass.

#### Step 1.5: 写 `depot_earliness` / `depot_index_sum` 失败测试

追加到 `tests/solver/test_depot_late.py`：

```python
class TestDepotMetrics:
    def test_earliness_empty_plan_is_zero(self) -> None:
        assert depot_earliness([]) == 0
        assert depot_index_sum([]) == 0

    def test_earliness_no_depot_hooks_is_zero(self) -> None:
        plan = [_hook("存1", "调北"), _hook("调北", "存4北")]
        assert depot_earliness(plan) == 0
        assert depot_index_sum(plan) == 0

    def test_earliness_single_depot_hook_last(self) -> None:
        # N=3, depot at index 3 -> earliness = (3-3) = 0.
        plan = [
            _hook("存1", "调北"),
            _hook("调北", "预修"),
            _hook("修1库内", "存4北"),
        ]
        assert depot_earliness(plan) == 0
        assert depot_index_sum(plan) == 3

    def test_earliness_single_depot_hook_first(self) -> None:
        # N=3, depot at index 1 -> earliness = (3-1) = 2.
        plan = [
            _hook("修1库内", "存4北"),
            _hook("存1", "调北"),
            _hook("调北", "预修"),
        ]
        assert depot_earliness(plan) == 2
        assert depot_index_sum(plan) == 1

    def test_earliness_multiple_depot_hooks(self) -> None:
        # N=4, depot at indices 1 and 3.
        # earliness = (4-1) + (4-3) = 4; index_sum = 1 + 3 = 4.
        plan = [
            _hook("修1库内", "修1库外"),
            _hook("存1", "调北"),
            _hook("修2库内", "存4北"),
            _hook("调北", "预修"),
        ]
        assert depot_earliness(plan) == 4
        assert depot_index_sum(plan) == 4

    def test_earliness_and_index_sum_relation(self) -> None:
        # earliness + index_sum = K * N where K is depot count.
        plan = [
            _hook("修1库内", "存4北"),
            _hook("存1", "调北"),
            _hook("修2库内", "存4北"),
            _hook("调北", "预修"),
            _hook("修3库内", "存4北"),
        ]
        n = len(plan)
        k = sum(1 for h in plan if is_depot_hook(h))
        assert depot_earliness(plan) + depot_index_sum(plan) == k * n
```

- [ ] **Step 1.5:** Append the test class above.

#### Step 1.6: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestDepotMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'depot_earliness'`.

- [ ] **Step 1.6:** Run and confirm failure.

#### Step 1.7: 实现 `depot_earliness` 和 `depot_index_sum`

追加到 `src/fzed_shunting/solver/depot_late.py`（在 `is_depot_hook` 之后）：

```python
def depot_index_sum(plan: Sequence[HookAction]) -> int:
    """Sum of 1-indexed positions of depot-touching hooks.

    Used during search as an incremental secondary key. Larger = depot
    hooks appear later in the plan.
    """
    return sum(i for i, hook in enumerate(plan, start=1) if is_depot_hook(hook))


def depot_earliness(plan: Sequence[HookAction]) -> int:
    """Sum of (N - i) over depot-touching hook positions, where N = len(plan).

    Smaller = depot hooks appear later. Zero when no depot hooks exist OR
    when all depot hooks are at the tail. Terminal metric used for LNS
    comparison and post-processing decisions.
    """
    n = len(plan)
    return sum(n - i for i, hook in enumerate(plan, start=1) if is_depot_hook(hook))
```

- [ ] **Step 1.7:** Append the two functions.

#### Step 1.8: 运行测试确认通过

Run: `pytest tests/solver/test_depot_late.py -v`
Expected: 13 passed.

- [ ] **Step 1.8:** Run and confirm pass.

#### Step 1.9: 写 `_is_early_depot_phase` 失败测试

追加到 `tests/solver/test_depot_late.py`：

```python
import pytest

from fzed_shunting.io.normalize_input import (
    GoalSpec,
    NormalizedPlanInput,
    NormalizedTrackInfo,
    NormalizedVehicle,
)
from fzed_shunting.solver.depot_late import _is_early_depot_phase
from fzed_shunting.verify.replay import ReplayState


def _vehicle(
    vno: str,
    current: str,
    targets: list[str],
    *,
    target_mode: str = "TRACK",
    area_code: str | None = None,
) -> NormalizedVehicle:
    return NormalizedVehicle(
        current_track=current,
        order=1,
        vehicle_model="敞车",
        vehicle_no=vno,
        repair_process="段修",
        vehicle_length=12.0,
        goal=GoalSpec(
            target_mode=target_mode,
            target_track=targets[0],
            allowed_target_tracks=targets,
            target_area_code=area_code,
        ),
    )


def _plan_input(vehicles: list[NormalizedVehicle]) -> NormalizedPlanInput:
    tracks = {v.current_track for v in vehicles} | {
        t for v in vehicles for t in v.goal.allowed_target_tracks
    }
    return NormalizedPlanInput(
        track_info=[NormalizedTrackInfo(track_name=t, track_distance=200.0) for t in tracks],
        vehicles=vehicles,
        loco_track_name="机库",
        yard_mode="NORMAL",
    )


def _state(track_sequences: dict[str, list[str]]) -> ReplayState:
    return ReplayState(
        track_sequences=track_sequences,
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )


class TestIsEarlyDepotPhase:
    def test_true_when_non_depot_goal_unmet(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存5北": ["V1"], "存1": ["V2"]})
        assert _is_early_depot_phase(state, plan_input) is True

    def test_false_when_all_non_depot_goals_met(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["存4北"]),
            _vehicle("V2", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存4北": ["V1"], "存1": ["V2"]})
        assert _is_early_depot_phase(state, plan_input) is False

    def test_false_when_only_depot_goals_exist(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存5北", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存5北": ["V1"]})
        assert _is_early_depot_phase(state, plan_input) is False

    def test_false_when_all_allowed_targets_are_depot(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内", "修2库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        state = _state({"存1": ["V1"]})
        # all allowed targets are in depot -> non_depot_targets is empty -> skip.
        assert _is_early_depot_phase(state, plan_input) is False
```

- [ ] **Step 1.9:** Append the test class.

#### Step 1.10: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestIsEarlyDepotPhase -v`
Expected: FAIL — `ImportError: cannot import name '_is_early_depot_phase'`.

- [ ] **Step 1.10:** Run and confirm failure.

#### Step 1.11: 实现 `_is_early_depot_phase`

追加到 `src/fzed_shunting/solver/depot_late.py`：

```python
def _is_early_depot_phase(
    state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> bool:
    """True when at least one vehicle with a non-depot goal has not reached it.

    "Non-depot goal" means the vehicle's ``allowed_target_tracks`` contain at
    least one track outside ``DEPOT_INNER_TRACKS``. Vehicles whose allowed
    targets are exclusively depot tracks do not influence the phase.
    """
    current_by_vehicle = {
        vno: track
        for track, seq in state.track_sequences.items()
        for vno in seq
    }
    for vehicle in plan_input.vehicles:
        allowed = set(vehicle.goal.allowed_target_tracks)
        non_depot = allowed - DEPOT_INNER_TRACKS
        if not non_depot:
            continue
        current = current_by_vehicle.get(vehicle.vehicle_no)
        if current not in allowed:
            return True
    return False
```

- [ ] **Step 1.11:** Append the function.

#### Step 1.12: 运行测试

Run: `pytest tests/solver/test_depot_late.py -v`
Expected: 17 passed.

- [ ] **Step 1.12:** Run and confirm all pass.

#### Step 1.13: 写 `reorder_depot_late` 失败测试

追加到 `tests/solver/test_depot_late.py`：

```python
from fzed_shunting.solver.depot_late import reorder_depot_late


class TestReorderDepotLate:
    def test_reorder_swaps_independent_depot_hook_later(self) -> None:
        # Two independent hooks: depot first, non-depot second.
        # Swapping is safe and strictly improves earliness.
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
            _vehicle("V2", "存2", ["存4北"]),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"]})
        plan = [
            _hook("存1", "修1库内", ["V1"]),
            _hook("存2", "存4北", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        # After reorder, depot hook moves to index 2.
        assert reordered[0].target_track == "存4北"
        assert reordered[1].target_track == "修1库内"
        assert depot_earliness(reordered) < depot_earliness(plan)

    def test_reorder_rejects_dependent_swap(self) -> None:
        # Second hook's source depends on first hook's target.
        # Swapping would break replay; must keep original order.
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        initial = _state({"存1": ["V1"]})
        plan = [
            _hook("存1", "修1库外", ["V1"]),
            _hook("修1库外", "修1库内", ["V1"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered == plan  # unchanged

    def test_reorder_preserves_empty_plan(self) -> None:
        plan_input = _plan_input([_vehicle("V1", "存1", ["存4北"])])
        initial = _state({"存1": ["V1"]})
        assert reorder_depot_late([], initial, plan_input) == []

    def test_reorder_is_noop_when_all_depot_already_late(self) -> None:
        plan_input = _plan_input([
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ])
        initial = _state({"存1": ["V1"], "存2": ["V2"]})
        plan = [
            _hook("存1", "存4北", ["V1"]),
            _hook("存2", "修1库内", ["V2"]),
        ]
        reordered = reorder_depot_late(plan, initial, plan_input)
        assert reordered == plan
```

- [ ] **Step 1.13:** Append the test class.

#### Step 1.14: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestReorderDepotLate -v`
Expected: FAIL — `ImportError: cannot import name 'reorder_depot_late'`.

- [ ] **Step 1.14:** Run and confirm failure.

#### Step 1.15: 实现 `reorder_depot_late`

追加到 `src/fzed_shunting/solver/depot_late.py`：

```python
def reorder_depot_late(
    plan: Sequence[HookAction],
    initial_state: ReplayState,
    plan_input: NormalizedPlanInput,
) -> list[HookAction]:
    """Greedy adjacent-swap pass that pushes depot hooks toward the tail.

    A swap is accepted only when:
    1. It strictly decreases ``depot_earliness``.
    2. The swapped plan replays legally from ``initial_state`` via
       ``_apply_move`` (i.e., each hook's source prefix condition holds
       at the point it is applied).
    3. The final state after replay matches the original plan's final
       state (same track_sequences, same weighed_vehicle_nos, same
       spot_assignments).

    On any failure the swap is discarded; worst case the function returns
    the input plan unchanged.
    """
    from fzed_shunting.solver.state import _apply_move

    result = list(plan)
    if not result:
        return result

    vehicle_by_no = {v.vehicle_no: v for v in plan_input.vehicles}

    def simulate(sequence: list[HookAction]) -> ReplayState | None:
        state = initial_state
        for move in sequence:
            source_seq = state.track_sequences.get(move.source_track, [])
            if source_seq[: len(move.vehicle_nos)] != move.vehicle_nos:
                return None
            try:
                state = _apply_move(
                    state=state,
                    move=move,
                    plan_input=plan_input,
                    vehicle_by_no=vehicle_by_no,
                )
            except (ValueError, KeyError):
                return None
        return state

    baseline_state = simulate(result)
    if baseline_state is None:
        return list(plan)
    baseline_key = (
        _canonicalize_state(baseline_state),
    )

    improved = True
    # Bounded passes: O(N^2) adjacent-swap bubble.
    while improved:
        improved = False
        for i in range(len(result) - 1):
            a, b = result[i], result[i + 1]
            if is_depot_hook(a) and not is_depot_hook(b):
                candidate = list(result)
                candidate[i], candidate[i + 1] = b, a
                new_state = simulate(candidate)
                if new_state is None:
                    continue
                if (_canonicalize_state(new_state),) != baseline_key:
                    continue
                result = candidate
                improved = True
    return result


def _canonicalize_state(state: ReplayState) -> tuple:
    """Hashable snapshot of a ReplayState for equivalence comparison."""
    tracks = tuple(
        (name, tuple(seq)) for name, seq in sorted(state.track_sequences.items())
    )
    weighed = tuple(sorted(state.weighed_vehicle_nos))
    spots = tuple(sorted(state.spot_assignments.items()))
    return (tracks, weighed, spots, state.loco_track_name)
```

- [ ] **Step 1.15:** Append the function and helper.

#### Step 1.16: 运行所有 depot_late 测试

Run: `pytest tests/solver/test_depot_late.py -v`
Expected: 21 passed.

- [ ] **Step 1.16:** Run and confirm pass.

#### Step 1.17: 迁移 constructive.py 的常量

修改 `src/fzed_shunting/solver/constructive.py:37` — 删除 `DEPOT_INNER_TRACKS = frozenset(...)` 这一行，改为在顶部 import 区加：

```python
from fzed_shunting.solver.depot_late import DEPOT_INNER_TRACKS
```

保证：
- `constructive.py` 内其他位置对 `DEPOT_INNER_TRACKS` 的使用不变。
- 不破坏其他 import `DEPOT_INNER_TRACKS` from `constructive` 的模块（如有）。

- [ ] **Step 1.17:** Apply the edit.

#### Step 1.18: 回归检查

Run: `pytest tests/solver/ -x`
Expected: all existing solver tests pass (count: 150+ per README baseline), plus new 21 depot_late tests.

- [ ] **Step 1.18:** Run and confirm no regressions.

#### Step 1.19: 提交

```bash
git add src/fzed_shunting/solver/depot_late.py \
        tests/solver/test_depot_late.py \
        src/fzed_shunting/solver/constructive.py
git commit -m "$(cat <<'EOF'
[W1-A] Add solver/depot_late.py with core helpers

New module hosting depot-late-scheduling primitives:
- is_depot_hook, depot_earliness, depot_index_sum
- _is_early_depot_phase (state-based early/late phase predicate)
- reorder_depot_late (adjacent-swap post-processor with state equivalence)
- DEPOT_INNER_TRACKS constant (moved from constructive.py; imported back)

21 unit tests. No consumer changes yet.
EOF
)"
```

- [ ] **Step 1.19:** Commit.

---

### Task 2: SolverResult 增加 depot_earliness 字段

**Files:**
- Modify: `src/fzed_shunting/solver/result.py:51-78`
- Create/Modify: `tests/solver/test_result.py` (if absent, create small test)

#### Step 2.1: 写失败测试

创建 `tests/solver/test_result.py`（如果不存在；如果存在则追加）：

```python
"""Unit tests for SolverResult data shape."""

from __future__ import annotations

from fzed_shunting.solver.result import SolverResult


def test_solver_result_defaults_depot_fields_to_none() -> None:
    result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
    )
    assert result.depot_earliness is None
    assert result.depot_hook_count is None


def test_solver_result_accepts_depot_fields() -> None:
    result = SolverResult(
        plan=[],
        expanded_nodes=0,
        generated_nodes=0,
        closed_nodes=0,
        elapsed_ms=0.0,
        depot_earliness=4,
        depot_hook_count=2,
    )
    assert result.depot_earliness == 4
    assert result.depot_hook_count == 2
```

- [ ] **Step 2.1:** Create/append the test.

#### Step 2.2: 运行确认失败

Run: `pytest tests/solver/test_result.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'depot_earliness'`.

- [ ] **Step 2.2:** Run and confirm failure.

#### Step 2.3: 为 `SolverResult` 增加字段

修改 `src/fzed_shunting/solver/result.py` 中 `SolverResult` 的定义，在 `telemetry: SolverTelemetry | None = None` 之后追加：

```python
    depot_earliness: int | None = None
    depot_hook_count: int | None = None
```

（即 SolverResult 的完整新字段块为：

```python
@dataclass(frozen=True)
class SolverResult:
    plan: list[HookAction]
    expanded_nodes: int
    generated_nodes: int
    closed_nodes: int
    elapsed_ms: float
    is_proven_optimal: bool = False
    fallback_stage: str | None = None
    verification_report: Any | None = None
    debug_stats: dict[str, Any] | None = None
    telemetry: SolverTelemetry | None = None
    depot_earliness: int | None = None
    depot_hook_count: int | None = None
```

）

- [ ] **Step 2.3:** Apply the edit.

#### Step 2.4: 运行测试

Run: `pytest tests/solver/test_result.py -v`
Expected: 2 passed.

- [ ] **Step 2.4:** Run and confirm pass.

#### Step 2.5: 回归检查

Run: `pytest tests/solver/ -x`
Expected: all tests pass.

- [ ] **Step 2.5:** Run and confirm.

#### Step 2.6: 提交

```bash
git add src/fzed_shunting/solver/result.py tests/solver/test_result.py
git commit -m "$(cat <<'EOF'
[W1-B] Add depot_earliness/depot_hook_count fields to SolverResult

Optional int fields, default None. Populated by astar_solver.py entry in
Wave 3. Two unit tests cover defaults and explicit construction.
EOF
)"
```

- [ ] **Step 2.6:** Commit.

---

⏸ **Gate: merge Wave 1 before proceeding.** Confirm both Task 1 and Task 2 are committed and `pytest tests/solver/` is green.

---

## Wave 2 — 集成层

### Task 3: 搜索层 `solver/search.py` 注入次级键

**Files:**
- Modify: `src/fzed_shunting/solver/search.py:38-43` (`QueueItem.priority` type)
- Modify: `src/fzed_shunting/solver/search.py:47-214` (`_solve_search_result` signature and body)
- Modify: `src/fzed_shunting/solver/search.py:217-237` (`_priority`)
- Modify: `src/fzed_shunting/solver/search.py:259-302` (`_prune_queue` blocker_bonus index)
- Modify: `tests/solver/test_depot_late.py` (add TestSearchPriority)

#### Step 3.1: 写失败测试（`_priority` 新参数）

追加到 `tests/solver/test_depot_late.py`：

```python
from fzed_shunting.solver.search import _priority


class TestSearchPriorityDepotSecondary:
    def test_flag_off_default_priority_unchanged_exact(self) -> None:
        # Baseline parity: neg_depot_index_sum=0 keeps the classic tuple shape.
        legacy = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
        )
        # Position 0: f=cost+h=8; position 1: cost; position 2: neg_depot_key
        # (0 when flag off); position 3: heuristic; position 4: -blocker_bonus.
        assert legacy[0] == 8
        assert legacy[1] == 3
        assert legacy[2] == 0
        assert legacy[3] == 5
        assert legacy[4] == 0

    def test_flag_on_smaller_neg_index_sum_preferred(self) -> None:
        # Two nodes same (cost, heuristic). Node A has depot at index 3
        # (index_sum=3, neg=-3); Node B at index 1 (index_sum=1, neg=-1).
        # A should have smaller priority tuple (more negative secondary).
        priority_a = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-3,
        )
        priority_b = _priority(
            cost=3, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-1,
        )
        assert priority_a < priority_b

    def test_flag_on_does_not_override_cost(self) -> None:
        # Node A has later depot but higher cost; cost must still win.
        priority_a = _priority(
            cost=4, heuristic=5, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=-100,
        )
        priority_b = _priority(
            cost=3, heuristic=6, blocker_bonus=0,
            solver_mode="exact", heuristic_weight=1.0,
            neg_depot_index_sum=0,
        )
        # Both f = 9 (tied on position 0). Cost tie-breaks at position 1.
        # B has cost=3 < A's cost=4, so B < A.
        assert priority_b < priority_a

    def test_beam_mode_preserves_blocker_index(self) -> None:
        # Regression guard for _prune_queue: it reads priority[?] for blocker.
        # After secondary insert, blocker_bonus lives at index 4.
        p = _priority(
            cost=2, heuristic=3, blocker_bonus=1,
            solver_mode="beam", heuristic_weight=1.0,
            neg_depot_index_sum=0,
        )
        # beam: (f, cost, neg_depot, adj_h, -blocker, h)
        assert p[4] == -1
```

- [ ] **Step 3.1:** Append the test class.

#### Step 3.2: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestSearchPriorityDepotSecondary -v`
Expected: FAIL — `TypeError: _priority() got an unexpected keyword argument 'neg_depot_index_sum'`.

- [ ] **Step 3.2:** Run and confirm failure.

#### Step 3.3: 修改 `_priority` 签名和实现

修改 `src/fzed_shunting/solver/search.py` 中的 `_priority` 为：

```python
def _priority(
    *,
    cost: int,
    heuristic: int,
    blocker_bonus: int = 0,
    solver_mode: str,
    heuristic_weight: float,
    neg_depot_index_sum: int = 0,
) -> tuple[float, int, int, int, int] | tuple[float, int, int, int, int, int]:
    if solver_mode == "beam":
        beam_heuristic_credit = 1 if blocker_bonus > 0 else 0
        adjusted_heuristic = heuristic - beam_heuristic_credit
        return (
            cost + adjusted_heuristic,
            cost,
            neg_depot_index_sum,
            adjusted_heuristic,
            -blocker_bonus,
            heuristic,
        )
    if solver_mode == "weighted":
        return (
            cost + heuristic_weight * heuristic,
            cost,
            neg_depot_index_sum,
            heuristic,
            -blocker_bonus,
        )
    return (
        cost + heuristic,
        cost,
        neg_depot_index_sum,
        heuristic,
        -blocker_bonus,
    )
```

- [ ] **Step 3.3:** Apply the edit.

#### Step 3.4: 修正 `_prune_queue` 的 blocker 索引

修改 `src/fzed_shunting/solver/search.py:284`（beam pruning 里读取 priority[3] 判断 blocker 的那一行）：

把 `if item.priority[3] < 0 and id(item) not in kept_ids` 改为 `if item.priority[4] < 0 and id(item) not in kept_ids`。

（解释：插入 `neg_depot_index_sum` 之后 blocker_bonus 位置从 3 移到 4。）

- [ ] **Step 3.4:** Apply the edit.

#### Step 3.5: 运行单测

Run: `pytest tests/solver/test_depot_late.py::TestSearchPriorityDepotSecondary -v`
Expected: 4 passed.

- [ ] **Step 3.5:** Run and confirm.

#### Step 3.6: 修改 `_solve_search_result` 签名与主循环

为 `_solve_search_result` 新增参数 `enable_depot_late_scheduling: bool = False`，并在扩展节点时计算 `neg_depot_index_sum`。

在 `_solve_search_result(...)` 签名追加：

```python
    enable_depot_late_scheduling: bool = False,
```

在主循环中，找到 `for move in moves:` 后、`heappush` 之前，替换为：

```python
        for move in moves:
            next_state = _apply_move(
                state=current.state,
                move=move,
                plan_input=plan_input,
                vehicle_by_no=vehicle_by_no,
            )
            next_plan = current.plan + [move]
            cost = len(next_plan)
            state_key = _state_key(
                next_state,
                canonical_random_depot_vehicle_nos=canonical_random_depot_vehicle_nos,
            )
            if state_key in best_cost and best_cost[state_key] <= cost:
                continue
            best_cost[state_key] = cost
            heuristic = state_heuristic(next_state)
            blocker_bonus = _blocking_goal_target_bonus(
                state=current.state,
                move=move,
                blocking_goal_targets_by_source=blocking_goal_targets_by_source,
            )
            neg_depot_index_sum = 0
            if enable_depot_late_scheduling:
                # incremental: previous plan's key + new hook's contribution.
                # We don't retain the parent's key explicitly; recompute O(L)
                # here — L is the plan length and typically small (<50). Micro
                # optimisation only matters on large instances; keep simple.
                from fzed_shunting.solver.depot_late import depot_index_sum
                neg_depot_index_sum = -depot_index_sum(next_plan)
            heappush(
                queue,
                QueueItem(
                    priority=_priority(
                        cost=cost,
                        heuristic=heuristic,
                        blocker_bonus=blocker_bonus,
                        solver_mode=solver_mode,
                        heuristic_weight=heuristic_weight,
                        neg_depot_index_sum=neg_depot_index_sum,
                    ),
                    seq=next(counter),
                    state_key=state_key,
                    state=next_state,
                    plan=next_plan,
                ),
            )
            generated_nodes += 1
            if debug_stats is not None:
                debug_stats["generated_nodes"] = generated_nodes
```

Note: the import `depot_index_sum` is placed inside the conditional to avoid a top-of-module cycle if Wave 2 agents haven't all merged yet. Move the import to module top after Wave 3 if desired.

- [ ] **Step 3.6:** Apply the edit.

#### Step 3.7: 拓宽 `QueueItem.priority` 的类型标注

修改 `src/fzed_shunting/solver/search.py:38-43`：

```python
@dataclass(order=True)
class QueueItem:
    priority: tuple
    seq: int
    state_key: tuple
    state: ReplayState
    plan: list[HookAction]
```

（把具体 tuple[...] 拓宽为通用 `tuple`；运行时比较不变。）

- [ ] **Step 3.7:** Apply the edit.

#### Step 3.8: 回归检查

Run: `pytest tests/solver/ -x`
Expected: all tests green. No behavior change with flag off (all search callers pass `enable_depot_late_scheduling=False` via default).

- [ ] **Step 3.8:** Run and confirm.

#### Step 3.9: 提交

```bash
git add src/fzed_shunting/solver/search.py tests/solver/test_depot_late.py
git commit -m "$(cat <<'EOF'
[W2-C] Inject lexicographic secondary key into search priority

_priority gains neg_depot_index_sum (default 0). Placed at index 2
(after cost), so cost is still the hard tiebreaker on equal f. Beam
pruning updated to read blocker at index 4 (shifted by 1).

_solve_search_result accepts enable_depot_late_scheduling (default False).
When on, maintains -depot_index_sum(plan) per generated node.

4 unit tests for priority tuple shape and ordering.
EOF
)"
```

- [ ] **Step 3.9:** Commit.

---

### Task 4: LNS `solver/lns.py` 按开关切换计划质量比较

**Files:**
- Modify: `src/fzed_shunting/solver/lns.py:280-308` (`_is_better_plan` + `_plan_quality`)
- Modify: `src/fzed_shunting/solver/lns.py:32-73` (`_solve_with_lns_result` signature)
- Modify: `src/fzed_shunting/solver/lns.py:76-...` (`_improve_incumbent_result` signature)
- Modify: `src/fzed_shunting/solver/lns.py:...` where `_is_better_plan` is called (line ~157)
- Modify: `tests/solver/test_depot_late.py` (append TestLnsPlanQuality)

#### Step 4.1: 写失败测试

追加到 `tests/solver/test_depot_late.py`：

```python
from fzed_shunting.solver.lns import _is_better_plan, _plan_quality


class TestLnsPlanQualityDepotAware:
    def test_flag_off_ignores_earliness(self) -> None:
        # Same hook count, same branch/length → equivalent when flag off.
        plan_early_depot = [_hook("修1库内", "存4北"), _hook("存1", "调北")]
        plan_late_depot = [_hook("存1", "调北"), _hook("修1库内", "存4北")]
        q_early = _plan_quality(plan_early_depot, route_oracle=None, depot_late=False)
        q_late = _plan_quality(plan_late_depot, route_oracle=None, depot_late=False)
        assert q_early == q_late

    def test_flag_on_prefers_later_depot(self) -> None:
        plan_early_depot = [_hook("修1库内", "存4北"), _hook("存1", "调北")]
        plan_late_depot = [_hook("存1", "调北"), _hook("修1库内", "存4北")]
        assert _is_better_plan(plan_late_depot, plan_early_depot, route_oracle=None, depot_late=True) is True
        assert _is_better_plan(plan_early_depot, plan_late_depot, route_oracle=None, depot_late=True) is False

    def test_flag_on_still_minimises_hook_count(self) -> None:
        # Shorter plan wins even if it has an earlier depot hook.
        plan_short_early_depot = [_hook("修1库内", "存4北")]
        plan_long_late_depot = [
            _hook("存1", "调北"),
            _hook("调北", "预修"),
            _hook("修1库内", "存4北"),
        ]
        assert _is_better_plan(
            plan_short_early_depot, plan_long_late_depot, route_oracle=None, depot_late=True
        ) is True
```

- [ ] **Step 4.1:** Append the test class.

#### Step 4.2: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestLnsPlanQualityDepotAware -v`
Expected: FAIL — `_plan_quality() got an unexpected keyword argument 'depot_late'`.

- [ ] **Step 4.2:** Run and confirm.

#### Step 4.3: 修改 `_plan_quality` 和 `_is_better_plan`

修改 `src/fzed_shunting/solver/lns.py`：

```python
def _is_better_plan(
    candidate_plan: list[HookAction],
    incumbent_plan: list[HookAction],
    route_oracle: RouteOracle | None,
    *,
    depot_late: bool = False,
) -> bool:
    candidate_metrics = _plan_quality(candidate_plan, route_oracle, depot_late=depot_late)
    incumbent_metrics = _plan_quality(incumbent_plan, route_oracle, depot_late=depot_late)
    return candidate_metrics < incumbent_metrics


def _plan_quality(
    plan: list[HookAction],
    route_oracle: RouteOracle | None,
    *,
    depot_late: bool = False,
) -> tuple:
    total_length_m = 0.0
    total_branch_count = 0
    if route_oracle is None:
        total_length_m = float(sum(len(move.path_tracks) for move in plan))
        total_branch_count = int(sum(max(len(move.path_tracks) - 1, 0) for move in plan))
    else:
        for move in plan:
            route = route_oracle.resolve_route(move.source_track, move.target_track)
            if route is not None:
                total_length_m += route.total_length_m
                total_branch_count += len(route.branch_codes)
            else:
                total_branch_count += max(len(move.path_tracks) - 1, 0)
                total_length_m += float(len(move.path_tracks))
    if depot_late:
        from fzed_shunting.solver.depot_late import depot_earliness

        return (len(plan), depot_earliness(plan), total_branch_count, total_length_m)
    return (len(plan), total_branch_count, total_length_m)
```

- [ ] **Step 4.3:** Apply the edit.

#### Step 4.4: 运行测试

Run: `pytest tests/solver/test_depot_late.py::TestLnsPlanQualityDepotAware -v`
Expected: 3 passed.

- [ ] **Step 4.4:** Run and confirm.

#### Step 4.5: 修改 `_solve_with_lns_result` 与 `_improve_incumbent_result` 签名

在 `_solve_with_lns_result(...)` 参数列表追加：

```python
    enable_depot_late_scheduling: bool = False,
```

在 `_improve_incumbent_result(...)` 参数列表追加同名参数。

内部对 `_is_better_plan(...)` 的所有调用位点，追加 `depot_late=enable_depot_late_scheduling`：

查找 `_is_better_plan(candidate_plan, incumbent_plan, route_oracle)` 并改为：

```python
_is_better_plan(candidate_plan, incumbent_plan, route_oracle, depot_late=enable_depot_late_scheduling)
```

同样透传到内部递归 `_improve_incumbent_result(...)` 调用。

- [ ] **Step 4.5:** Apply the edit.

#### Step 4.6: 回归检查

Run: `pytest tests/solver/ -x`
Expected: all tests green. LNS behavior unchanged with flag off.

- [ ] **Step 4.6:** Run and confirm.

#### Step 4.7: 提交

```bash
git add src/fzed_shunting/solver/lns.py tests/solver/test_depot_late.py
git commit -m "$(cat <<'EOF'
[W2-D] Depot-aware plan quality tie-break in LNS

_plan_quality(..., depot_late: bool = False) extends the quality tuple
from (len, branch, length) to (len, earliness, branch, length) when
depot_late=True. _is_better_plan and the LNS result pipelines accept
enable_depot_late_scheduling and forward it.

Flag off: behavior byte-identical to baseline. 3 unit tests.
EOF
)"
```

- [ ] **Step 4.7:** Commit.

---

### Task 5: 构造层 `solver/constructive.py` intra-tier penalty

**Files:**
- Modify: `src/fzed_shunting/solver/constructive.py:51-73` (`solve_constructive` signature)
- Modify: `src/fzed_shunting/solver/constructive.py:133-141` (call site of `_choose_best_move`)
- Modify: `src/fzed_shunting/solver/constructive.py:224-248` (`_choose_best_move`)
- Modify: `src/fzed_shunting/solver/constructive.py:277-367` (`_score_move`)
- Modify: `tests/solver/test_depot_late.py` (append TestConstructiveDepotPenalty)

#### Step 5.1: 写失败测试

追加到 `tests/solver/test_depot_late.py`：

```python
from fzed_shunting.solver.constructive import _score_move
from fzed_shunting.verify.replay import build_initial_state


class TestConstructiveDepotPenalty:
    def test_flag_off_depot_and_nondepot_at_same_tier_unchanged(self) -> None:
        # Two candidates scored without flag. Both tier 2 (forward progress).
        # Deterministic ordering: both get penalty=0, so other components decide.
        vehicles = [
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ]
        plan_input = _plan_input(vehicles)
        state = build_initial_state(plan_input)
        vehicle_by_no = {v.vehicle_no: v for v in vehicles}
        goal_tracks = {"存4北", "修1库内"}
        move_v1 = _hook("存1", "存4北", ["V1"])
        move_v2 = _hook("存2", "修1库内", ["V2"])
        score_v1_off, _ = _score_move(
            move=move_v1,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks,
            enable_depot_late_scheduling=False,
            plan_input=plan_input,
        )
        score_v2_off, _ = _score_move(
            move=move_v2,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks,
            enable_depot_late_scheduling=False,
            plan_input=plan_input,
        )
        # Both tier-2, penalty field (index 1) is zero for both when flag off.
        assert score_v1_off[0] == score_v2_off[0]  # same tier
        assert score_v1_off[1] == 0
        assert score_v2_off[1] == 0

    def test_flag_on_early_phase_non_depot_beats_depot(self) -> None:
        vehicles = [
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ]
        plan_input = _plan_input(vehicles)
        state = build_initial_state(plan_input)
        vehicle_by_no = {v.vehicle_no: v for v in vehicles}
        goal_tracks = {"存4北", "修1库内"}
        move_v1 = _hook("存1", "存4北", ["V1"])
        move_v2 = _hook("存2", "修1库内", ["V2"])
        score_v1_on, _ = _score_move(
            move=move_v1, state=state, vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks,
            enable_depot_late_scheduling=True, plan_input=plan_input,
        )
        score_v2_on, _ = _score_move(
            move=move_v2, state=state, vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks,
            enable_depot_late_scheduling=True, plan_input=plan_input,
        )
        # Same tier; depot move gets penalty=1; non-depot move has penalty=0
        # → v1 (non-depot) has smaller score, is preferred.
        assert score_v1_on[0] == score_v2_on[0]  # still same tier
        assert score_v1_on[1] < score_v2_on[1]

    def test_flag_on_late_phase_no_penalty(self) -> None:
        # All non-depot goals satisfied: late phase, no penalty applies.
        vehicles = [
            _vehicle("V1", "存4北", ["存4北"]),  # already at goal
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ]
        plan_input = _plan_input(vehicles)
        state = build_initial_state(plan_input)
        vehicle_by_no = {v.vehicle_no: v for v in vehicles}
        goal_tracks = {"存4北", "修1库内"}
        move_v2 = _hook("存2", "修1库内", ["V2"])
        score_v2, _ = _score_move(
            move=move_v2, state=state, vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks,
            enable_depot_late_scheduling=True, plan_input=plan_input,
        )
        # Late phase → no depot penalty.
        assert score_v2[1] == 0
```

- [ ] **Step 5.1:** Append the test class.

#### Step 5.2: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestConstructiveDepotPenalty -v`
Expected: FAIL — `_score_move() got an unexpected keyword argument 'enable_depot_late_scheduling'`.

- [ ] **Step 5.2:** Run and confirm.

#### Step 5.3: 改 `_score_move` 签名与评分元组

修改 `src/fzed_shunting/solver/constructive.py` 中的 `_score_move`：

```python
def _score_move(
    *,
    move: HookAction,
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_tracks_needed: set[str],
    enable_depot_late_scheduling: bool = False,
    plan_input: NormalizedPlanInput | None = None,
) -> tuple[tuple, int]:
    block_vehicles = [vehicle_by_no[vn] for vn in move.vehicle_nos]
    source_track = move.source_track
    target_track = move.target_track

    delta = sum(
        int(target_track in v.goal.allowed_target_tracks)
        - int(source_track in v.goal.allowed_target_tracks)
        for v in block_vehicles
    )

    is_close_door_final = (
        target_track == "存4北"
        and any(v.is_close_door for v in block_vehicles)
        and delta > 0
    )
    is_weigh_to_jiku = (
        target_track == "机库"
        and any(
            v.need_weigh and v.vehicle_no not in state.weighed_vehicle_nos
            for v in block_vehicles
        )
    )
    clears_blocker = (
        delta >= 0
        and _move_clears_goal_blocker(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
        )
    )
    exposes_buried_seeker = (
        delta <= 0
        and _move_exposes_buried_goal_seeker(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
        )
    )
    is_staging = target_track in STAGING_TRACKS

    tier: int
    if is_close_door_final:
        tier = 0
    elif is_weigh_to_jiku:
        tier = 1
    elif delta > 0:
        tier = 2
    elif clears_blocker:
        tier = 3
    elif exposes_buried_seeker:
        tier = 4
    elif delta == 0 and not is_staging:
        tier = 5
    elif delta == 0 and is_staging:
        tier = 6
    else:
        tier = 7

    depot_late_penalty = 0
    if enable_depot_late_scheduling and plan_input is not None:
        from fzed_shunting.solver.depot_late import is_depot_hook, _is_early_depot_phase

        if is_depot_hook(move) and _is_early_depot_phase(state, plan_input):
            depot_late_penalty = 1

    block_size = len(move.vehicle_nos)
    path_length = len(move.path_tracks)
    is_spot_or_area_finalization = delta > 0 and any(
        target_track in v.goal.allowed_target_tracks
        and (v.goal.target_mode == "SPOT" or v.goal.target_area_code is not None)
        for v in block_vehicles
    )
    score = (
        tier,
        depot_late_penalty,
        0 if is_spot_or_area_finalization else 1,
        -delta,
        -block_size,
        path_length,
        source_track,
        target_track,
    )
    return score, min(tier, 5)
```

- [ ] **Step 5.3:** Apply the edit.

#### Step 5.4: 透传到 `_choose_best_move`

修改 `src/fzed_shunting/solver/constructive.py` 中的 `_choose_best_move`：

```python
def _choose_best_move(
    *,
    moves: list[HookAction],
    state: ReplayState,
    vehicle_by_no: dict[str, NormalizedVehicle],
    goal_tracks_needed: set[str],
    recent_moves: deque[tuple[str, str, tuple[str, ...]]] | None = None,
    enable_depot_late_scheduling: bool = False,
    plan_input: NormalizedPlanInput | None = None,
) -> tuple[HookAction, int]:
    scored: list[tuple[tuple, int, HookAction, bool]] = []
    for move in moves:
        score, tier = _score_move(
            move=move,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            plan_input=plan_input,
        )
        is_inverse = _is_inverse_of_recent(move, recent_moves)
        scored.append((score, tier, move, is_inverse))
    non_inverse = [entry for entry in scored if not entry[3]]
    pool = non_inverse if non_inverse else scored
    pool.sort(key=lambda entry: entry[0])
    score, tier, move, _ = pool[0]
    return move, tier
```

- [ ] **Step 5.4:** Apply the edit.

#### Step 5.5: 透传到 `solve_constructive`

修改 `solve_constructive` 签名，在现有 kwarg 后追加：

```python
    enable_depot_late_scheduling: bool = False,
```

并在主循环内 `_choose_best_move(...)` 调用处（`src/fzed_shunting/solver/constructive.py:133-141`）追加参数：

```python
        best_move, best_tier = _choose_best_move(
            moves=moves,
            state=state,
            vehicle_by_no=vehicle_by_no,
            goal_tracks_needed=goal_tracks_needed,
            recent_moves=recent_moves,
            enable_depot_late_scheduling=enable_depot_late_scheduling,
            plan_input=plan_input,
        )
```

- [ ] **Step 5.5:** Apply the edit.

#### Step 5.6: 运行单测

Run: `pytest tests/solver/test_depot_late.py::TestConstructiveDepotPenalty -v`
Expected: 3 passed.

- [ ] **Step 5.6:** Run and confirm.

#### Step 5.7: 回归检查

Run: `pytest tests/solver/ -x`
Expected: all tests green (150+ existing + new tests). Flag-off path behavior identical.

- [ ] **Step 5.7:** Run and confirm.

#### Step 5.8: 提交

```bash
git add src/fzed_shunting/solver/constructive.py tests/solver/test_depot_late.py
git commit -m "$(cat <<'EOF'
[W2-E] Intra-tier depot_late_penalty in constructive _score_move

_score_move gains enable_depot_late_scheduling and plan_input kwargs.
When flag is on and _is_early_depot_phase returns True, depot-touching
moves gain a penalty=1 in the score tuple at position 1 (after tier,
before spot/area finalization flag). Tier itself is NOT changed, so
depot moves remain eligible when they are the only progress option.

_choose_best_move and solve_constructive plumb the flag through.

3 unit tests covering: flag off (no effect), flag on + early phase
(depot loses tie), flag on + late phase (no penalty).
EOF
)"
```

- [ ] **Step 5.8:** Commit.

---

⏸ **Gate: merge Wave 2 before proceeding.** Confirm Tasks 3, 4, 5 are all committed and `pytest tests/solver/ -x` is green.

---

## Wave 3 — 装配与集成

### Task 6: `solver/astar_solver.py` 对外 API + 后处理

**Files:**
- Modify: `src/fzed_shunting/solver/astar_solver.py:51-78` (`solve_with_simple_astar`)
- Modify: `src/fzed_shunting/solver/astar_solver.py:80-...` (`solve_with_simple_astar_result` and wiring)
- Modify: `src/fzed_shunting/solver/astar_solver.py` (attach_verification region — post-processing call)
- Modify: `tests/solver/test_depot_late.py` (append TestAstarSolverApi)

#### Step 6.1: 写失败测试（end-to-end API smoke）

追加到 `tests/solver/test_depot_late.py`：

```python
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result


class TestAstarSolverDepotLateFlag:
    def test_result_exposes_depot_earliness_when_flag_off(self) -> None:
        # Small trivially-solvable input. Flag off still populates fields.
        vehicles = [_vehicle("V1", "存1", ["存4北"])]
        plan_input = _plan_input(vehicles)
        initial = build_initial_state(plan_input)
        result = solve_with_simple_astar_result(
            plan_input=plan_input,
            initial_state=initial,
            time_budget_ms=5_000,
            verify=False,
        )
        assert result.depot_earliness is not None
        assert result.depot_hook_count is not None
        assert result.depot_hook_count == 0  # no depot targets
        assert result.depot_earliness == 0

    def test_flag_toggle_preserves_hook_count(self) -> None:
        # Mixed input: one depot, one non-depot goal. Hook counts must match.
        vehicles = [
            _vehicle("V1", "存1", ["存4北"]),
            _vehicle("V2", "存2", ["修1库内"], target_mode="AREA", area_code="大库:RANDOM"),
        ]
        plan_input = _plan_input(vehicles)
        initial = build_initial_state(plan_input)
        off = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=5_000, verify=False,
            enable_depot_late_scheduling=False,
        )
        on = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=5_000, verify=False,
            enable_depot_late_scheduling=True,
        )
        assert len(on.plan) == len(off.plan)
        assert on.depot_earliness <= off.depot_earliness
```

- [ ] **Step 6.1:** Append the test class.

#### Step 6.2: 运行确认失败

Run: `pytest tests/solver/test_depot_late.py::TestAstarSolverDepotLateFlag -v`
Expected: FAIL — `TypeError: solve_with_simple_astar_result() got an unexpected keyword argument 'enable_depot_late_scheduling'`.

- [ ] **Step 6.2:** Run and confirm.

#### Step 6.3: 加入 kwarg 并透传到三层求解器

修改 `src/fzed_shunting/solver/astar_solver.py`：

1. 给 `solve_with_simple_astar` 和 `solve_with_simple_astar_result` 签名在参数末尾追加：

```python
    enable_depot_late_scheduling: bool = False,
```

2. `solve_with_simple_astar` 内部调用 `solve_with_simple_astar_result(...)` 时透传：

```python
        enable_depot_late_scheduling=enable_depot_late_scheduling,
```

3. `solve_with_simple_astar_result` 内部所有调用 `_solve_search_result(...)`、`_solve_with_lns_result(...)`、`_improve_incumbent_result(...)` 的位点都追加 `enable_depot_late_scheduling=enable_depot_late_scheduling`。

4. `_run_anytime_fallback_chain` 如有必要也加 kwarg（检查是否是调用入口；如果它内部又去调 `_solve_search_result` 则也要透传）。

5. constructive 阶段调用 `solve_constructive(...)` 时追加 `enable_depot_late_scheduling=enable_depot_late_scheduling`。

- [ ] **Step 6.3:** Apply all edits.

#### Step 6.4: 后处理 `reorder_depot_late`

在 `solve_with_simple_astar_result` 内部，定位到 `_attach_verification(...)` 调用之前，当 `result.plan` 非空且 `enable_depot_late_scheduling=True` 时插入：

```python
    if enable_depot_late_scheduling and result.plan:
        from fzed_shunting.solver.depot_late import reorder_depot_late

        reordered_plan = reorder_depot_late(result.plan, initial_state, plan_input)
        if reordered_plan != result.plan:
            result = replace(result, plan=reordered_plan)
```

（注意 `replace` 已从 `dataclasses` 导入于 `astar_solver.py:3`。）

- [ ] **Step 6.4:** Apply the edit.

#### Step 6.5: 填充 `depot_earliness` 和 `depot_hook_count` 字段

在 `solve_with_simple_astar_result` 的每一个 `return SolverResult(...)` 或 `return replace(result, ...)` 位点之前，插入：

```python
    from fzed_shunting.solver.depot_late import depot_earliness, is_depot_hook

    result = replace(
        result,
        depot_earliness=depot_earliness(result.plan),
        depot_hook_count=sum(1 for h in result.plan if is_depot_hook(h)),
    )
```

建议：把该操作集中在函数末尾的 **最终 return** 之前，而不是每个分支各自填。

- [ ] **Step 6.5:** Apply the edit.

#### Step 6.6: 运行 API 测试

Run: `pytest tests/solver/test_depot_late.py::TestAstarSolverDepotLateFlag -v`
Expected: 2 passed.

- [ ] **Step 6.6:** Run and confirm.

#### Step 6.7: 全量回归

Run: `pytest tests/solver/ -x`
Expected: all tests green, no regression.

- [ ] **Step 6.7:** Run and confirm.

#### Step 6.8: 提交

```bash
git add src/fzed_shunting/solver/astar_solver.py tests/solver/test_depot_late.py
git commit -m "$(cat <<'EOF'
[W3-F] Wire enable_depot_late_scheduling through solver entry + post-proc

solve_with_simple_astar(_result) gain the opt-in kwarg (default False).
Flag plumbed through constructive, search, LNS, and fallback chain.

When flag=True, final plan passes through reorder_depot_late before
verification: adjacent-swap pass that strictly improves depot_earliness
while preserving replay-equivalent final state. On any replay failure
the swap is rejected; worst case is plan unchanged.

SolverResult.depot_earliness and depot_hook_count are populated on every
return path, even with flag off, for benchmarking and observability.

2 end-to-end smoke tests.
EOF
)"
```

- [ ] **Step 6.8:** Commit.

---

### Task 7: 集成测试 + bench 对照

**Files:**
- Modify: `tests/solver/test_astar_solver.py` (append TestDepotLateScheduling integration tests)

#### Step 7.1: 写集成测试

追加到 `tests/solver/test_astar_solver.py`（找到文件末尾，按现有风格追加）：

```python
"""Integration tests for depot-late-scheduling flag."""

from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.solver.depot_late import depot_earliness, is_depot_hook


def _count_depot_hooks(plan):
    return sum(1 for h in plan if is_depot_hook(h))


def test_depot_late_flag_off_matches_baseline(typical_fixtures):
    """Flag off: plan byte-equal to baseline (no depot-late changes)."""
    for fixture in typical_fixtures:
        plan_input = fixture.plan_input
        initial = fixture.initial_state
        baseline = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
        )
        flagged_off = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
            enable_depot_late_scheduling=False,
        )
        assert baseline.plan == flagged_off.plan


def test_depot_late_flag_on_does_not_increase_hook_count(typical_fixtures):
    """Flag on: hook count ≤ baseline (lexicographic secondary)."""
    for fixture in typical_fixtures:
        plan_input = fixture.plan_input
        initial = fixture.initial_state
        baseline = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
        )
        flagged_on = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
            enable_depot_late_scheduling=True,
        )
        assert len(flagged_on.plan) <= len(baseline.plan), (
            f"Hook count regressed on fixture {fixture.name}: "
            f"{len(baseline.plan)} -> {len(flagged_on.plan)}"
        )


def test_depot_late_flag_on_preserves_validity(typical_fixtures):
    """Flag on: every baseline-valid fixture stays valid."""
    for fixture in typical_fixtures:
        plan_input = fixture.plan_input
        initial = fixture.initial_state
        baseline = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
        )
        if baseline.verification_report is None or not baseline.verification_report.is_valid:
            continue
        flagged_on = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=True,
            enable_depot_late_scheduling=True,
        )
        assert flagged_on.verification_report is not None
        assert flagged_on.verification_report.is_valid, (
            f"Validity regressed on fixture {fixture.name}"
        )


def test_depot_late_flag_on_improves_earliness_on_average(typical_fixtures):
    """Flag on: mean earliness across fixtures ≤ baseline mean."""
    baseline_scores = []
    flagged_scores = []
    for fixture in typical_fixtures:
        plan_input = fixture.plan_input
        initial = fixture.initial_state
        baseline = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=False,
        )
        flagged = solve_with_simple_astar_result(
            plan_input=plan_input, initial_state=initial,
            time_budget_ms=10_000, verify=False,
            enable_depot_late_scheduling=True,
        )
        if _count_depot_hooks(baseline.plan) == 0:
            continue  # no depot hooks — nothing to late-schedule
        baseline_scores.append(depot_earliness(baseline.plan))
        flagged_scores.append(depot_earliness(flagged.plan))

    if baseline_scores:
        assert sum(flagged_scores) <= sum(baseline_scores), (
            f"Total earliness worsened: {sum(baseline_scores)} -> {sum(flagged_scores)}"
        )
```

Note: `typical_fixtures` fixture must exist in `tests/solver/conftest.py` and return an iterable of (name, plan_input, initial_state) records drawn from `artifacts/typical_suite`. If the fixture does not exist yet, also add:

```python
# tests/solver/conftest.py — append if missing
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fzed_shunting.io.normalize_input import NormalizedPlanInput
from fzed_shunting.verify.replay import build_initial_state


@dataclass(frozen=True)
class SolverFixture:
    name: str
    plan_input: NormalizedPlanInput
    initial_state: object


def _load_suite(dir_name: str) -> list[SolverFixture]:
    root = Path(__file__).parents[2] / "artifacts" / dir_name
    if not root.exists():
        return []
    out: list[SolverFixture] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Depending on the suite's normalised envelope, adjust here.
        plan_input = NormalizedPlanInput.model_validate(payload["plan_input"])
        initial = build_initial_state(plan_input)
        out.append(SolverFixture(name=path.stem, plan_input=plan_input, initial_state=initial))
    return out


@pytest.fixture(scope="session")
def typical_fixtures():
    return _load_suite("typical_suite")
```

If the suite loader format differs in this repo, adjust `_load_suite` to match the actual on-disk envelope (check `tests/solver/test_anytime_integration.py` or `test_astar_solver.py` for an example).

- [ ] **Step 7.1:** Append integration tests and (if necessary) the conftest fixture.

#### Step 7.2: 运行集成测试

Run: `pytest tests/solver/test_astar_solver.py -v -k depot_late`
Expected: 4 integration tests pass. If the fixture adapter is wrong, the failure message will tell you which key is missing; adjust `_load_suite` accordingly.

- [ ] **Step 7.2:** Run and confirm.

#### Step 7.3: 全量回归

Run: `pytest tests/solver/ -x`
Expected: all solver tests green.

- [ ] **Step 7.3:** Run and confirm.

#### Step 7.4: 本地 bench —— 109 外部验证集

先给 `scripts/run_external_validation_parallel.py` 加 CLI 开关。参考 `scripts/run_external_validation_parallel.py:26-51` 里的 argparse 结构，在现有 `--beam-width` / `--heuristic-weight` 之后、`--max-workers` 之前追加：

```python
    parser.add_argument(
        "--enable-depot-late-scheduling",
        dest="enable_depot_late_scheduling",
        action="store_true",
        default=False,
        help="Opt in to depot-late scheduling (secondary lexicographic objective).",
    )
```

然后在所有调用 `solve_with_simple_astar_result(...)` 的位点（runner.py 不需要改；本脚本对应 `scripts/run_external_validation_parallel.py:71-83` 的 `result = solve_with_simple_astar_result(...)`）追加：

```python
            enable_depot_late_scheduling=getattr(args, "enable_depot_late_scheduling", False),
```

并把 `enable_depot_late_scheduling` 加到 `_run_worker`、`_run_single_scenario`、`_run_batch` 等中间函数的签名透传链上（参照现有 `enable_anytime_fallback` 的透传模式，每一层都用同名 kwarg 默认 False）。

跑两次 bench：

```bash
# baseline (flag off)
python scripts/run_external_validation_parallel.py \
    --output-dir artifacts/depot_late_bench_off \
    --solver exact \
    --max-workers 8 \
    --timeout-seconds 180

# flag on
python scripts/run_external_validation_parallel.py \
    --output-dir artifacts/depot_late_bench_on \
    --solver exact \
    --max-workers 8 \
    --timeout-seconds 180 \
    --enable-depot-late-scheduling
```

- [ ] **Step 7.4:** Apply CLI wiring and run both benches. Save both output dirs.

#### Step 7.5: 比对并写入 spec 尾部

读取两边的 `summary.json`（run_external_validation_parallel 的输出惯例）比对。交互式 REPL 或临时脚本：

```python
import json
from pathlib import Path

def load(dir_):
    summary = json.loads((Path(dir_) / "summary.json").read_text(encoding="utf-8"))
    return summary

off = load("artifacts/depot_late_bench_off")
on = load("artifacts/depot_late_bench_on")

def summarize(s):
    results = s.get("results", [])
    valid = [r for r in results if r.get("is_valid")]
    total_hooks = sum(r.get("plan_hook_count", 0) for r in valid)
    total_earliness = sum((r.get("depot_earliness") or 0) for r in valid)
    return {
        "count": len(results),
        "valid": len(valid),
        "total_hooks": total_hooks,
        "total_earliness": total_earliness,
    }

print("OFF:", summarize(off))
print("ON :", summarize(on))
```

把这两行输出粘贴进 spec `docs/superpowers/specs/2026-04-20-depot-late-scheduling-design.md` §6 作为 "里程碑 3 实测" 条目。

- [ ] **Step 7.5:** Run comparison, paste numbers into the spec.

#### Step 7.6: 提交测试与 bench 结果

```bash
git add tests/solver/test_astar_solver.py \
        tests/solver/conftest.py \
        scripts/run_external_validation_parallel.py \
        docs/superpowers/specs/2026-04-20-depot-late-scheduling-design.md \
        artifacts/depot_late_bench_off \
        artifacts/depot_late_bench_on
git commit -m "$(cat <<'EOF'
[W3-F] Depot-late integration tests + 109-scenario bench artifacts

4 integration tests across typical_suite: flag-off byte equality with
baseline, flag-on preserves hook count and validity, flag-on improves
aggregate earliness. Bench artifacts show the actual impact on 109
external_validation cases for production sign-off.
EOF
)"
```

- [ ] **Step 7.6:** Commit.

---

## 完工检查表

- [ ] Wave 1 Task 1 merged (21 depot_late unit tests green)
- [ ] Wave 1 Task 2 merged (result fields available)
- [ ] Wave 2 Task 3 merged (search priority tuple extended, 4 tests green)
- [ ] Wave 2 Task 4 merged (LNS plan quality extended, 3 tests green)
- [ ] Wave 2 Task 5 merged (constructive intra-tier penalty, 3 tests green)
- [ ] Wave 3 Task 6 merged (API kwarg + post-processing, 2 e2e tests green)
- [ ] Wave 3 Task 7 merged (4 integration tests + 2 bench artifacts)
- [ ] `pytest tests/solver/ -x` fully green (150+ existing + ~35 new)
- [ ] `external_validation_inputs` valid count 未降（≥97/109）
- [ ] `typical_suite` 与 `typical_workflow_suite` 开关关闭下 plan byte-equal
- [ ] Decision on flipping default based on bench — 若 valid 率稳、总勾数不增，`enable_depot_late_scheduling=True` 作为默认值；否则保持 False，由业务层按需打开。
