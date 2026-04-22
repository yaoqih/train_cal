"""Isolate the 12 previously-failing positive scenarios and report solver status.

Used to validate the W3-N backtracking constructive fix. Prior baseline: 0/12
reached VALID under `enable_depot_late_scheduling=False`. Target: lift count
above zero; each reached goal proves backtracking broke a local minimum.

Run: ``python scripts/verify_backtrack_12.py``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.verify.replay import build_initial_state

FAILING = [
    "case_3_2_shed_work_gondola.json",
    "case_3_3_spot_101_boundary.json",
    "case_3_3_spot_203_mid.json",
    "case_3_3_spot_305_long_car.json",
    "case_4_3_5_depot_random_changxiu.json",
    "case_4_3_5_depot_random_duanxiu.json",
    "case_4_3_5_depot_random_linxiu.json",
    "case_8_1_normal_uses_01_to_05.json",
    "case_8_2_long_176m_routes_to_3_4.json",
    "case_8_2_long_192m_routes_to_3_4.json",
    "case_8_2_short_143m_prefers_1_2.json",
    "case_8_2_short_160m_prefers_1_2.json",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    master = load_master_data(repo_root / "data" / "master")
    results: dict[str, tuple] = {}

    for name in FAILING:
        payload_path = repo_root / "data" / "validation_inputs" / "positive" / name
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        normalized = normalize_plan_input(payload, master)
        initial = build_initial_state(normalized)
        t_start = time.time()
        try:
            r = solve_with_simple_astar_result(
                normalized,
                initial,
                master=master,
                solver_mode="exact",
                time_budget_ms=120_000,
                verify=True,
                enable_depot_late_scheduling=False,
            )
            elapsed = time.time() - t_start
            valid = r.verification_report.is_valid if r.verification_report else None
            results[name] = (len(r.plan), valid, r.fallback_stage, elapsed)
            print(
                f"{name:<50} hooks={len(r.plan):>3} valid={valid} "
                f"stage={r.fallback_stage} t={elapsed:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            results[name] = ("error", str(exc)[:60])
            print(f"{name:<50} ERROR: {str(exc)[:80]}")

    valid_count = sum(
        1 for v in results.values() if len(v) >= 2 and v[1] is True
    )
    total_time = sum(
        v[3] for v in results.values() if len(v) >= 4 and isinstance(v[3], float)
    )
    avg_time = total_time / max(len(results), 1)
    print(f"\nSummary: {valid_count}/12 now VALID, avg time per scenario: {avg_time:.1f}s")


if __name__ == "__main__":
    main()
