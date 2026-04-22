#!/usr/bin/env python3
"""
Experimental CP-SAT solver for depot shunting — minimum-hooks model.

真·一勾定义:
  PICK(track, n)  — 从股道近端连续挑走 n 辆 (1 钩)
  PLACE(track, n) — 从机车摆下 n 辆到股道近端 (1 钩)
  目标: 最小化总钩数; 次目标(可选 --depot-late): 最小化大库延后度

物理模型 (严格):
  Stock position 0 = near end (north end), higher = deeper.
  Loco position 0 = most recently coupled (top of stack).
  PICK(k, n) :  track[k][0..n-1] → loco[0..n-1];
                track[k] shifts LEFT  by n;
                existing loco[i] shifts to loco[i+n].
  PLACE(k, n):  loco[0..n-1] → track[k][0..n-1];
                track[k][i] shifts to track[k][i+n];
                remaining loco[i] shifts to loco[i-n].

Ignored: target track length overflow.
Enforced: reverse-branch clearance (consist length during travel must fit).

Usage:
  python tools/cpsat_solver.py --input <json> [--time-limit 60] [--max-hooks 60]
                                [--depot-late] [--master data/master]
  python tools/cpsat_solver.py --sweep data/validation_inputs/positive [--time-limit 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from ortools.sat.python import cp_model
except ImportError:
    print("ortools not installed. Run: pip install ortools", file=sys.stderr)
    sys.exit(1)

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input

_LENGTH_SCALE = 10   # metres → decimetres
_INF_CLEARANCE = 99_999


# ─── reverse-clearance precomputation ─────────────────────────────────────────

def _precompute_clearance(
    oracle: RouteOracle,
    track_names: list[str],
) -> dict[tuple[int, int], int]:
    n = len(track_names)
    result: dict[tuple[int, int], int] = {}
    for i, src in enumerate(track_names):
        for j, dst in enumerate(track_names):
            if i == j:
                result[(i, j)] = _INF_CLEARANCE
                continue
            route = oracle.resolve_route(src, dst)
            if route is None:
                result[(i, j)] = _INF_CLEARANCE
                continue
            path_tracks = oracle.resolve_path_tracks(src, dst)
            if path_tracks is None:
                result[(i, j)] = _INF_CLEARANCE
                continue
            rev_branches = oracle._resolve_reverse_branch_codes(path_tracks, route)
            if not rev_branches:
                result[(i, j)] = _INF_CLEARANCE
                continue
            min_len = min(
                (oracle.master.physical_routes[br].total_length_m
                 for br in rev_branches
                 if br in oracle.master.physical_routes),
                default=float("inf"),
            )
            result[(i, j)] = int(min_len * _LENGTH_SCALE) if min_len != float("inf") else _INF_CLEARANCE
    return result


# ─── CP-SAT model ─────────────────────────────────────────────────────────────

def build_and_solve(
    *,
    initial_seqs: dict[str, list[str]],
    vehicle_lengths_dm: dict[str, int],
    goal_tracks: dict[str, list[str]],
    track_names: list[str],
    clearance_dm: dict[tuple[int, int], int],
    loco_length_dm: int,
    max_hooks: int,
    time_limit_s: float,
    depot_late_target_tracks: list[str],
    enable_depot_late: bool,
) -> dict:
    t0 = time.time()

    track_idx: dict[str, int] = {t: i for i, t in enumerate(track_names)}
    T = len(track_names)

    # Vehicle list (1-indexed; 0 = empty slot)
    all_vnos: list[str] = []
    for seq in initial_seqs.values():
        all_vnos.extend(seq)
    all_vnos = list(dict.fromkeys(all_vnos))
    V = len(all_vnos)
    vno_to_id: dict[str, int] = {v: i + 1 for i, v in enumerate(all_vnos)}

    if V == 0:
        return {"status": "OPTIMAL", "hooks": 0, "plan": [], "elapsed_s": 0.0}

    # P = max vehicles per track slot (must accommodate worst-case goal track load)
    max_in_track = max((len(s) for s in initial_seqs.values()), default=1)
    from collections import Counter
    goal_load: Counter = Counter()
    for allowed in goal_tracks.values():
        for t in allowed:
            goal_load[t] += 1
    max_goal_load = max(goal_load.values(), default=0)
    P = max(max_in_track + 2, max_goal_load + 1, 8)
    P = min(P, V + 1)
    PL = V  # loco capacity = total vehicles

    # Initial state arrays: TV0[k][p]
    TV0: list[list[int]] = [[0] * P for _ in range(T)]
    for tname, seq in initial_seqs.items():
        if tname not in track_idx:
            continue
        k = track_idx[tname]
        for p, vno in enumerate(seq):
            if p < P:
                TV0[k][p] = vno_to_id[vno]

    # Goal: for each vehicle id, list of allowed final track indices
    goal_track_indices: dict[int, list[int]] = {}
    for vno, allowed in goal_tracks.items():
        vid = vno_to_id.get(vno)
        if vid is None:
            continue
        idxs = [track_idx[t] for t in allowed if t in track_idx]
        if idxs:
            goal_track_indices[vid] = idxs

    # Vehicle length table indexed by vehicle id (0 = empty → 0 dm)
    len_by_id: list[int] = [0] * (V + 1)
    for vno, vid in vno_to_id.items():
        len_by_id[vid] = vehicle_lengths_dm.get(vno, 0)

    depot_track_set = frozenset(
        track_idx[t] for t in depot_late_target_tracks if t in track_idx
    )

    H = max_hooks
    max_pick_n = min(P, V)
    max_place_n = min(PL, V)

    # ── model ─────────────────────────────────────────────────────────────────
    model = cp_model.CpModel()

    # State: TV[h][k][p] vehicle at track k position p after h hooks
    TV = [
        [[model.new_int_var(0, V, f"TV_{h}_{k}_{p}") for p in range(P)]
         for k in range(T)]
        for h in range(H + 1)
    ]
    # Loco: LV[h][p]
    LV = [
        [model.new_int_var(0, V, f"LV_{h}_{p}") for p in range(PL)]
        for h in range(H + 1)
    ]

    # Initial state
    for k in range(T):
        for p in range(P):
            model.add(TV[0][k][p] == TV0[k][p])
    for p in range(PL):
        model.add(LV[0][p] == 0)

    # Hook operation booleans
    pick = [
        [[model.new_bool_var(f"pk_{h}_{k}_{n}") for n in range(1, max_pick_n + 1)]
         for k in range(T)]
        for h in range(H)
    ]
    place = [
        [[model.new_bool_var(f"pl_{h}_{k}_{n}") for n in range(1, max_place_n + 1)]
         for k in range(T)]
        for h in range(H)
    ]
    noop = [model.new_bool_var(f"noop_{h}") for h in range(H)]

    for h in range(H):
        all_ops: list[cp_model.IntVar] = [noop[h]]
        for k in range(T):
            all_ops.extend(pick[h][k])
            all_ops.extend(place[h][k])
        model.add_exactly_one(all_ops)

    # Noops come last (symmetry breaking)
    for h in range(H - 1):
        model.add_implication(noop[h], noop[h + 1])

    # ── feasibility guards ─────────────────────────────────────────────────────
    # PICK(k, n) requires at least n real vehicles in track k at time h
    # (TV[h][k][n-1] >= 1 means the nth slot is non-empty).
    # PLACE(k, n) requires at least n vehicles on loco at time h
    # (LV[h][n-1] >= 1).
    for h in range(H):
        for k in range(T):
            for ni, n in enumerate(range(1, max_pick_n + 1)):
                b = pick[h][k][ni]
                slot = n - 1  # 0-indexed
                if slot < P:
                    model.add(TV[h][k][slot] >= 1).only_enforce_if(b)
                else:
                    model.add(b == 0)  # impossible: track never has n vehicles
            for ni, n in enumerate(range(1, max_place_n + 1)):
                b = place[h][k][ni]
                slot = n - 1
                if slot < PL:
                    model.add(LV[h][slot] >= 1).only_enforce_if(b)
                else:
                    model.add(b == 0)

    # ── per-track "is hook h operating on track k?" boolean ───────────────────
    # operates_on[h][k] = 1 iff hook h does something on track k
    operates_on = [
        [model.new_bool_var(f"op_{h}_{k}") for k in range(T)]
        for h in range(H)
    ]
    for h in range(H):
        for k in range(T):
            all_on_k = list(pick[h][k]) + list(place[h][k])
            # operates_on[h][k] ↔ (any pick or place on k)
            model.add_bool_or(all_on_k).only_enforce_if(operates_on[h][k])
            model.add(sum(all_on_k) == 0).only_enforce_if(operates_on[h][k].Not())
            # noop → not operating on any k
            model.add_implication(noop[h], operates_on[h][k].Not())

    # ── transition constraints ─────────────────────────────────────────────────
    # For each track k and each time step h:
    #   If NOT operating on k at h → TV[h+1][k] = TV[h][k]
    #   If PICK(k, n)              → left-shift TV[h][k] by n; update loco
    #   If PLACE(k, n)             → prepend loco[0..n-1] to TV[h][k]; update loco

    # LOCO: similarly, if not picking/placing anything → LV[h+1] = LV[h]
    # But loco changes whenever ANY hook fires (either loco gains or loses vehicles)
    # We handle loco transition per hook operation.

    # Precompute: noop_or_not_on_k_bools for each (h, k)
    # "track k is untouched at hook h" = not operates_on[h][k]
    # "loco is untouched at hook h" = noop[h]

    for h in range(H):
        b_noop = noop[h]

        # ── loco unchanged if noop ──────────────────────────────────────────
        for p in range(PL):
            model.add(LV[h + 1][p] == LV[h][p]).only_enforce_if(b_noop)

        # ── per-track transitions ───────────────────────────────────────────
        for k in range(T):
            not_on_k = operates_on[h][k].Not()

            # Track k unchanged when not operated on
            for p in range(P):
                model.add(TV[h + 1][k][p] == TV[h][k][p]).only_enforce_if(not_on_k)

            # PICK(k, n) for each n
            for ni, n in enumerate(range(1, max_pick_n + 1)):
                b = pick[h][k][ni]

                # Track k: shift left by n
                for p in range(P):
                    src = p + n
                    if src < P:
                        model.add(TV[h + 1][k][p] == TV[h][k][src]).only_enforce_if(b)
                    else:
                        model.add(TV[h + 1][k][p] == 0).only_enforce_if(b)

                # Loco: positions 0..n-1 = picked vehicles; rest shift right by n
                for p in range(PL):
                    if p < n:
                        src_p = p  # TV[h][k][p] → LV[h+1][p]
                        if src_p < P:
                            model.add(LV[h + 1][p] == TV[h][k][src_p]).only_enforce_if(b)
                        else:
                            model.add(LV[h + 1][p] == 0).only_enforce_if(b)
                    else:
                        shift_src = p - n  # LV[h][p-n] → LV[h+1][p]
                        if 0 <= shift_src < PL:
                            model.add(LV[h + 1][p] == LV[h][shift_src]).only_enforce_if(b)
                        else:
                            model.add(LV[h + 1][p] == 0).only_enforce_if(b)

            # PLACE(k, n) for each n
            for ni, n in enumerate(range(1, max_place_n + 1)):
                b = place[h][k][ni]

                # Track k: positions 0..n-1 = loco[0..n-1]; rest shift by +n
                for p in range(P):
                    if p < n:
                        src_p = p  # LV[h][p] → TV[h+1][k][p]
                        if src_p < PL:
                            model.add(TV[h + 1][k][p] == LV[h][src_p]).only_enforce_if(b)
                        else:
                            model.add(TV[h + 1][k][p] == 0).only_enforce_if(b)
                    else:
                        shift_src = p - n  # TV[h][k][p-n] → TV[h+1][k][p]
                        if 0 <= shift_src < P:
                            model.add(TV[h + 1][k][p] == TV[h][k][shift_src]).only_enforce_if(b)
                        else:
                            model.add(TV[h + 1][k][p] == 0).only_enforce_if(b)

                # Loco: shift left by n (remove first n vehicles)
                for p in range(PL):
                    shift_src = p + n  # LV[h][p+n] → LV[h+1][p]
                    if shift_src < PL:
                        model.add(LV[h + 1][p] == LV[h][shift_src]).only_enforce_if(b)
                    else:
                        model.add(LV[h + 1][p] == 0).only_enforce_if(b)

    # ── goal constraints ───────────────────────────────────────────────────────
    for vid in range(1, V + 1):
        allowed_k = goal_track_indices.get(vid)
        if not allowed_k:
            continue

        # in_track_k_final[k] = 1 iff vehicle vid found in TV[H][k][*]
        in_any_goal: list[cp_model.IntVar] = []
        for k in allowed_k:
            pos_bools = []
            for p in range(P):
                b = model.new_bool_var(f"gf_{vid}_{k}_{p}")
                model.add(TV[H][k][p] == vid).only_enforce_if(b)
                model.add(TV[H][k][p] != vid).only_enforce_if(b.Not())
                pos_bools.append(b)
            in_k = model.new_bool_var(f"ink_f_{vid}_{k}")
            model.add_bool_or(pos_bools).only_enforce_if(in_k)
            model.add(sum(pos_bools) == 0).only_enforce_if(in_k.Not())
            in_any_goal.append(in_k)
        model.add_bool_or(in_any_goal)

    # ── loco must be empty at end ──────────────────────────────────────────────
    for p in range(PL):
        model.add(LV[H][p] == 0)

    # ── reverse-branch clearance ───────────────────────────────────────────────
    constrained_pairs = [
        (i, j, clr) for (i, j), clr in clearance_dm.items()
        if clr < _INF_CLEARANCE
    ]

    if constrained_pairs:
        max_single_len = max(len_by_id) if len_by_id else 0
        loco_vlen = [
            [model.new_int_var(0, max_single_len, f"lvlen_{h}_{p}") for p in range(PL)]
            for h in range(H + 1)
        ]
        loco_total_vlen = [
            model.new_int_var(0, sum(len_by_id), f"loco_vt_{h}")
            for h in range(H + 1)
        ]
        for h in range(H + 1):
            for p in range(PL):
                model.add_element(LV[h][p], len_by_id, loco_vlen[h][p])
            model.add(loco_total_vlen[h] == sum(loco_vlen[h]))

        for h in range(H - 1):
            for k1, k2, clr in constrained_pairs:
                max_vlen = clr - loco_length_dm
                b_pair = model.new_bool_var(f"pair_{h}_{k1}_{k2}")
                model.add_bool_and([operates_on[h][k1], operates_on[h + 1][k2]]).only_enforce_if(b_pair)
                model.add_bool_or([operates_on[h][k1].Not(), operates_on[h + 1][k2].Not()]).only_enforce_if(b_pair.Not())
                if max_vlen <= 0:
                    # Even empty loco too long — forbid this transition
                    model.add_bool_or([operates_on[h][k1].Not(), operates_on[h + 1][k2].Not()])
                else:
                    model.add(loco_total_vlen[h + 1] <= max_vlen).only_enforce_if(b_pair)

    # ── objective ──────────────────────────────────────────────────────────────
    n_active = model.new_int_var(0, H, "n_active")
    model.add(n_active == sum(1 - noop[h] for h in range(H)))

    if not enable_depot_late:
        model.minimize(n_active)
    else:
        depot_final_bools = []
        for vid in range(1, V + 1):
            for k in depot_track_set:
                for p in range(P):
                    b = model.new_bool_var(f"df_{vid}_{k}_{p}")
                    model.add(TV[H][k][p] == vid).only_enforce_if(b)
                    model.add(TV[H][k][p] != vid).only_enforce_if(b.Not())
                    depot_final_bools.append(b)
        n_depot = model.new_int_var(0, V, "n_depot")
        model.add(n_depot <= sum(depot_final_bools))
        model.minimize(n_active * (H + 1) - n_depot)

    # ── solve ──────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8

    status = solver.solve(model)
    elapsed = time.time() - t0

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_name, "hooks": None, "plan": [], "elapsed_s": round(elapsed, 2)}

    hooks_used = solver.value(n_active)
    id_to_vno = {v: k for k, v in vno_to_id.items()}
    plan = []
    for h in range(H):
        if solver.value(noop[h]):
            break
        for k in range(T):
            for ni, n in enumerate(range(1, max_pick_n + 1)):
                if solver.value(pick[h][k][ni]):
                    vehs = [
                        id_to_vno[solver.value(TV[h][k][p])]
                        for p in range(n)
                        if solver.value(TV[h][k][p]) != 0
                    ]
                    plan.append({"step": h, "op": "PICK", "track": track_names[k],
                                 "count": n, "vehicles": vehs})
            for ni, n in enumerate(range(1, max_place_n + 1)):
                if solver.value(place[h][k][ni]):
                    vehs = [
                        id_to_vno[solver.value(LV[h][p])]
                        for p in range(n)
                        if solver.value(LV[h][p]) != 0
                    ]
                    plan.append({"step": h, "op": "PLACE", "track": track_names[k],
                                 "count": n, "vehicles": vehs})

    return {
        "status": status_name,
        "hooks": hooks_used,
        "plan": plan,
        "elapsed_s": round(elapsed, 2),
    }


# ─── problem loader ────────────────────────────────────────────────────────────

def load_problem(input_path: Path, master_dir: Path) -> dict:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    master = load_master_data(master_dir)
    oracle = RouteOracle(master)
    plan_input = normalize_plan_input(payload, master, allow_internal_loco_tracks=True)

    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for v in plan_input.vehicles:
        grouped[v.current_track].append((v.order, v.vehicle_no))
    initial_seqs: dict[str, list[str]] = {
        t: [vno for _, vno in sorted(entries)]
        for t, entries in grouped.items()
    }

    goal_tracks: dict[str, list[str]] = {}
    for v in plan_input.vehicles:
        allowed = list(v.goal.allowed_target_tracks)
        if not allowed and v.goal.target_track:
            allowed = [v.goal.target_track]
        goal_tracks[v.vehicle_no] = allowed

    vehicle_lengths_dm = {
        v.vehicle_no: int(round(v.vehicle_length * _LENGTH_SCALE))
        for v in plan_input.vehicles
    }

    # Only include tracks that are actually referenced in this problem
    track_set: set[str] = set(initial_seqs.keys())
    for allowed in goal_tracks.values():
        track_set.update(allowed)
    track_names = sorted(track_set)

    clearance_dm = _precompute_clearance(oracle, track_names)
    loco_length_dm = int(round(master.business_rules.loco_length_m * _LENGTH_SCALE))

    from fzed_shunting.io.normalize_input import AREA_ALLOWED_TRACKS
    depot_late_tracks = list({
        t
        for code, tracks in AREA_ALLOWED_TRACKS.items()
        if "大库" in code or "修" in code
        for t in tracks
    })

    return {
        "initial_seqs": initial_seqs,
        "vehicle_lengths_dm": vehicle_lengths_dm,
        "goal_tracks": goal_tracks,
        "track_names": track_names,
        "clearance_dm": clearance_dm,
        "loco_length_dm": loco_length_dm,
        "depot_late_target_tracks": depot_late_tracks,
    }


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="CP-SAT depot shunting solver (experimental)")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--sweep", type=Path)
    ap.add_argument("--master", type=Path, default=Path("data/master"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--max-hooks", type=int, default=60)
    ap.add_argument("--depot-late", action="store_true")
    ap.add_argument("--max-vehicles", type=int, default=None,
                    help="Skip cases with more than this many vehicles (sweep mode)")
    args = ap.parse_args()

    master_dir = (
        args.master if args.master.is_absolute()
        else _REPO / args.master
    )

    counts: dict[str, int] = {"OPTIMAL": 0, "UNKNOWN": 0, "INFEASIBLE": 0, "ERROR": 0, "SKIPPED": 0}

    def solve_one(path: Path) -> None:
        print(f"\n{'='*60}")
        print(f"Input: {path.name}")
        try:
            problem = load_problem(path, master_dir)
        except Exception as e:
            print(f"  Load error: {e}")
            import traceback; traceback.print_exc()
            counts["ERROR"] += 1
            return

        V = sum(len(s) for s in problem["initial_seqs"].values())
        T = len(problem["track_names"])
        print(f"  Vehicles: {V}, Tracks: {T}, MaxHooks: {args.max_hooks}")
        if args.max_vehicles is not None and V > args.max_vehicles:
            print(f"  SKIPPED (V={V} > --max-vehicles={args.max_vehicles})")
            counts["SKIPPED"] += 1
            return

        result = build_and_solve(
            initial_seqs=problem["initial_seqs"],
            vehicle_lengths_dm=problem["vehicle_lengths_dm"],
            goal_tracks=problem["goal_tracks"],
            track_names=problem["track_names"],
            clearance_dm=problem["clearance_dm"],
            loco_length_dm=problem["loco_length_dm"],
            max_hooks=args.max_hooks,
            time_limit_s=args.time_limit,
            depot_late_target_tracks=problem["depot_late_target_tracks"],
            enable_depot_late=args.depot_late,
        )
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        print(f"  Status:  {status}")
        print(f"  Hooks:   {result['hooks']}")
        print(f"  Elapsed: {result['elapsed_s']}s")
        if result["plan"]:
            print("  Plan:")
            for step in result["plan"]:
                print(f"    [{step['step']}] {step['op']:5s} {step['track']} ×{step['count']}  {step['vehicles']}")

    if args.sweep:
        files = sorted(args.sweep.glob("*.json"))
        print(f"Sweeping {len(files)} files from {args.sweep}")
        for f in files:
            solve_one(f)
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY")
        total = len(files)
        for k, v in sorted(counts.items()):
            if v:
                print(f"  {k}: {v}/{total}")
        print(f"  Total: {total}")
    elif args.input:
        solve_one(args.input)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
