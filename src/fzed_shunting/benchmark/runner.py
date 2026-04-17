from __future__ import annotations

import csv
import json
from pathlib import Path

from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.sim.generator import generate_scenario
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state, replay_plan
from fzed_shunting.workflow.runner import solve_workflow


DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[3] / "data" / "master"


def run_benchmark(
    output_dir: Path,
    scenario_count: int = 10,
    vehicle_count: int = 4,
    seed_start: int = 0,
    direct_only: bool = True,
    profile: str = "micro",
    solver: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
) -> dict:
    master = load_master_data(DEFAULT_MASTER_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    solved_count = 0
    solved_hook_counts: list[int] = []
    solved_path_lengths: list[float] = []
    solved_branch_counts: list[int] = []
    solved_expanded_nodes: list[int] = []
    solved_generated_nodes: list[int] = []
    error_summary: dict[str, int] = {}
    route_oracle = RouteOracle(master)
    for idx in range(scenario_count):
        seed = seed_start + idx
        scenario = generate_scenario(
            master,
            seed=seed,
            vehicle_count=vehicle_count,
            profile=profile,
            direct_only=direct_only,
        )
        scenario_path = output_dir / f"scenario_{seed}.json"
        scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")

        normalized = normalize_plan_input(scenario, master)
        initial = build_initial_state(normalized)
        try:
            solver_result = solve_with_simple_astar_result(
                normalized,
                initial,
                master=master,
                solver_mode=solver,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
            )
            plan = solver_result.plan
            hook_plan = [
                {
                    "hookNo": move_idx,
                    "actionType": move.action_type,
                    "sourceTrack": move.source_track,
                    "targetTrack": move.target_track,
                    "vehicleNos": move.vehicle_nos,
                    "pathTracks": move.path_tracks,
                }
                for move_idx, move in enumerate(plan, start=1)
            ]
            path_length_m, branch_count = _collect_plan_route_metrics(route_oracle, hook_plan)
            verify_report = verify_plan(master, normalized, hook_plan)
            solved = verify_report.is_valid
            if solved:
                solved_count += 1
                solved_hook_counts.append(len(hook_plan))
                solved_path_lengths.append(path_length_m)
                solved_branch_counts.append(branch_count)
                solved_expanded_nodes.append(solver_result.expanded_nodes)
                solved_generated_nodes.append(solver_result.generated_nodes)
            else:
                for error in verify_report.errors:
                    error_summary[error] = error_summary.get(error, 0) + 1
            results.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "scenario_path": str(scenario_path),
                    "hook_count": len(hook_plan),
                    "path_length_m": path_length_m,
                    "branch_count": branch_count,
                    "elapsed_ms": solver_result.elapsed_ms,
                    "expanded_nodes": solver_result.expanded_nodes,
                    "generated_nodes": solver_result.generated_nodes,
                    "closed_nodes": solver_result.closed_nodes,
                    "solver": solver,
                    "solved": solved,
                    "errors": verify_report.errors,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "scenario_path": str(scenario_path),
                    "hook_count": None,
                    "path_length_m": None,
                    "branch_count": None,
                    "elapsed_ms": None,
                    "expanded_nodes": None,
                    "generated_nodes": None,
                    "closed_nodes": None,
                    "solver": solver,
                    "solved": False,
                    "errors": [str(exc)],
                }
            )
            error_summary[str(exc)] = error_summary.get(str(exc), 0) + 1

    unsolved_count = scenario_count - solved_count
    report = {
        "scenario_count": scenario_count,
        "vehicle_count": vehicle_count,
        "seed_start": seed_start,
        "profile": profile,
        "direct_only": direct_only,
        "solver": solver,
        "heuristic_weight": heuristic_weight,
        "beam_width": beam_width,
        "solved_count": solved_count,
        "unsolved_count": unsolved_count,
        "solved_rate": solved_count / scenario_count if scenario_count else 0.0,
        "average_hook_count_on_solved": (
            sum(solved_hook_counts) / len(solved_hook_counts) if solved_hook_counts else None
        ),
        "average_path_length_m_on_solved": (
            sum(solved_path_lengths) / len(solved_path_lengths) if solved_path_lengths else None
        ),
        "average_branch_count_on_solved": (
            sum(solved_branch_counts) / len(solved_branch_counts) if solved_branch_counts else None
        ),
        "average_expanded_nodes_on_solved": (
            sum(solved_expanded_nodes) / len(solved_expanded_nodes)
            if solved_expanded_nodes
            else None
        ),
        "average_generated_nodes_on_solved": (
            sum(solved_generated_nodes) / len(solved_generated_nodes)
            if solved_generated_nodes
            else None
        ),
        "error_summary": error_summary,
        "results": results,
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results_csv(output_dir / "benchmark_results.csv", results)
    return report


def run_solver_comparison_suite(
    *,
    suite_path: Path,
    output_dir: Path,
    solver_configs: list[dict],
) -> dict:
    master = load_master_data(DEFAULT_MASTER_DIR)
    suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
    scenarios = suite_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("suite payload must contain scenarios[]")

    output_dir.mkdir(parents=True, exist_ok=True)
    solver_reports: list[dict] = []
    for config in solver_configs:
        solver = str(config["solver"])
        solver_dir = output_dir / solver
        solver_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        valid_count = 0
        total_hook_count = 0
        for index, scenario in enumerate(scenarios, start=1):
            if not isinstance(scenario, dict) or not isinstance(scenario.get("payload"), dict):
                raise ValueError(f"invalid scenario entry at index {index}")
            scenario_name = str(scenario.get("name", f"scenario_{index}"))
            try:
                scenario_result = _solve_suite_payload(
                    master=master,
                    payload=scenario["payload"],
                    solver=solver,
                    heuristic_weight=float(config.get("heuristic_weight", 1.0)),
                    beam_width=config.get("beam_width"),
                )
            except Exception as exc:  # noqa: BLE001
                scenario_result = _build_suite_error_result(
                    master=master,
                    payload=scenario["payload"],
                    solver=solver,
                    error=str(exc),
                )
            scenario_result["name"] = scenario_name
            scenario_result["description"] = scenario.get("description", "")
            results.append(scenario_result)
            total_hook_count += int(scenario_result["hook_count"])
            if scenario_result["is_valid"]:
                valid_count += 1
            (solver_dir / f"{scenario_name}.plan.json").write_text(
                json.dumps(scenario_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        solver_report = {
            "solver": solver,
            "heuristicWeight": float(config.get("heuristic_weight", 1.0)),
            "beamWidth": config.get("beam_width"),
            "validCount": valid_count,
            "scenarioCount": len(results),
            "averageHookCount": total_hook_count / len(results) if results else None,
            "results": [
                {
                    "name": item["name"],
                    "hookCount": item["hook_count"],
                    "isValid": item["is_valid"],
                    "solverErrors": item.get("solver_errors", []),
                }
                for item in results
            ],
        }
        solver_reports.append(solver_report)
        (solver_dir / "suite_report.json").write_text(
            json.dumps(solver_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    comparison_report = {
        "suite": suite_payload.get("suite", suite_path.stem),
        "scenario_count": len(scenarios),
        "solvers": solver_reports,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(comparison_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return comparison_report


def _collect_plan_route_metrics(route_oracle: RouteOracle, hook_plan: list[dict]) -> tuple[float, int]:
    total_length_m = 0.0
    total_branch_count = 0
    for hook in hook_plan:
        route = route_oracle.resolve_route(hook["sourceTrack"], hook["targetTrack"])
        if route is None:
            continue
        total_length_m += route.total_length_m
        total_branch_count += len(route.branch_codes)
    return total_length_m, total_branch_count


def _solve_suite_payload(
    *,
    master,
    payload: dict,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
) -> dict:
    if _is_workflow_payload(payload):
        result = solve_workflow(
            master,
            payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
        )
        stage_payloads = [
            {
                "name": stage.name,
                "description": stage.description,
                "isValid": stage.view.summary.is_valid if stage.view else False,
                "hookCount": stage.view.summary.hook_count if stage.view else 0,
                "finalTracks": stage.view.summary.final_tracks if stage.view else [],
                "finalSpotAssignments": stage.view.final_spot_assignments if stage.view else {},
                "failedHookNos": stage.view.failed_hook_nos if stage.view else [],
                "verifierErrors": stage.view.verifier_errors if stage.view else [],
            }
            for stage in result.stages
        ]
        return {
            "scenario_type": "workflow",
            "solver": solver,
            "vehicle_count": len(payload.get("initialVehicleInfo", [])),
            "stage_count": result.stage_count,
            "hook_count": sum(stage["hookCount"] for stage in stage_payloads),
            "stages": stage_payloads,
            "hook_plan": [],
            "final_state": None,
            "is_valid": all(stage["isValid"] for stage in stage_payloads),
            "verifier_errors": [],
            "solver_errors": [],
        }
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    plan = solve_with_simple_astar_result(
        normalized,
        initial,
        master=master,
        solver_mode=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
    ).plan
    route_oracle = RouteOracle(master)
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in normalized.vehicles}
    hook_plan = [
        {
            "hookNo": idx,
            "actionType": item.action_type,
            "sourceTrack": item.source_track,
            "targetTrack": item.target_track,
            "vehicleCount": len(item.vehicle_nos),
            "vehicleNos": item.vehicle_nos,
            "pathTracks": item.path_tracks,
            "remark": _build_hook_remark(
                route_oracle.validate_path(
                    source_track=item.source_track,
                    target_track=item.target_track,
                    path_tracks=item.path_tracks,
                    train_length_m=sum(length_by_vehicle[vehicle_no] for vehicle_no in item.vehicle_nos),
                )
            ),
        }
        for idx, item in enumerate(plan, start=1)
    ]
    replay = replay_plan(
        initial,
        hook_plan,
        plan_input=normalized,
    )
    verify_report = verify_plan(master, normalized, hook_plan)
    return {
        "scenario_type": "single",
        "solver": solver,
        "vehicle_count": len(normalized.vehicles),
        "hook_plan": hook_plan,
        "hook_count": len(hook_plan),
        "final_state": replay.final_state.model_dump(mode="json"),
        "is_valid": verify_report.is_valid,
        "verifier_errors": verify_report.errors,
        "solver_errors": [],
    }


def _build_suite_error_result(
    *,
    master,
    payload: dict,
    solver: str,
    error: str,
) -> dict:
    if _is_workflow_payload(payload):
        return {
            "scenario_type": "workflow",
            "solver": solver,
            "vehicle_count": len(payload.get("initialVehicleInfo", [])),
            "stage_count": len(payload.get("workflowStages", [])),
            "hook_count": 0,
            "stages": [],
            "hook_plan": [],
            "final_state": None,
            "is_valid": False,
            "verifier_errors": [],
            "solver_errors": [error],
        }
    normalized = normalize_plan_input(payload, master)
    return {
        "scenario_type": "single",
        "solver": solver,
        "vehicle_count": len(normalized.vehicles),
        "hook_plan": [],
        "hook_count": 0,
        "final_state": None,
        "is_valid": False,
        "verifier_errors": [],
        "solver_errors": [error],
    }


def _is_workflow_payload(payload: dict) -> bool:
    workflow_stages = payload.get("workflowStages")
    return isinstance(workflow_stages, list)


def _build_hook_remark(validation) -> str:
    parts: list[str] = []
    if validation.total_length_m is not None:
        parts.append(f"route={validation.total_length_m:.1f}m")
    if validation.branch_codes:
        parts.append("branches=" + ",".join(validation.branch_codes))
    if validation.reverse_branch_codes:
        parts.append("reverse=" + ",".join(validation.reverse_branch_codes))
    return "; ".join(parts)


def _write_results_csv(path: Path, results: list[dict]) -> None:
    fieldnames = [
        "seed",
        "profile",
        "scenario_path",
        "hook_count",
        "path_length_m",
        "branch_count",
        "elapsed_ms",
        "expanded_nodes",
        "generated_nodes",
        "closed_nodes",
        "solver",
        "solved",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = dict(item)
            row["errors"] = " | ".join(row.get("errors", []))
            writer.writerow({name: row.get(name) for name in fieldnames})
