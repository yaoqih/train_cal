from __future__ import annotations

import json
from pathlib import Path

import typer

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import PathValidationResult, RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.benchmark.runner import run_benchmark, run_solver_comparison_suite
from fzed_shunting.sim.generator import (
    generate_micro_scenario,
    generate_scenario,
    generate_typical_suite,
    generate_typical_workflow_suite,
)
from fzed_shunting.solver.astar_solver import solve_with_simple_astar_result
from fzed_shunting.solver.profile import (
    VALIDATION_DEFAULT_BEAM_WIDTH,
    VALIDATION_DEFAULT_SOLVER,
    VALIDATION_DEFAULT_TIMEOUT_SECONDS,
    validation_time_budget_ms,
)
from fzed_shunting.solver.validation_recovery import solve_with_validation_recovery_result
from fzed_shunting.workflow.runner import solve_workflow
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import build_initial_state, replay_plan


app = typer.Typer(no_args_is_help=True)
DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


@app.command("generate-micro")
def generate_micro(
    output: Path = typer.Option(..., exists=False, dir_okay=False),
    seed: int = typer.Option(0),
    vehicle_count: int = typer.Option(6),
    direct_only: bool = typer.Option(True),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    scenario = generate_micro_scenario(
        master,
        seed=seed,
        vehicle_count=vehicle_count,
        direct_only=direct_only,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(str(output))


@app.command("generate-scenario")
def generate_scenario_cmd(
    output: Path = typer.Option(..., exists=False, dir_okay=False),
    profile: str = typer.Option("micro"),
    seed: int = typer.Option(0),
    vehicle_count: int = typer.Option(6),
    direct_only: bool = typer.Option(False),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    scenario = generate_scenario(
        master,
        seed=seed,
        vehicle_count=vehicle_count,
        profile=profile,
        direct_only=direct_only,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(str(output))


@app.command("generate-typical-suite")
def generate_typical_suite_cmd(
    output: Path = typer.Option(..., exists=False, dir_okay=False),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    suite = generate_typical_suite(master)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(str(output))


@app.command("generate-typical-workflow-suite")
def generate_typical_workflow_suite_cmd(
    output: Path = typer.Option(..., exists=False, dir_okay=False),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    suite = generate_typical_workflow_suite(master)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(str(output))


@app.command("solve")
def solve(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    solver: str = typer.Option(VALIDATION_DEFAULT_SOLVER),
    heuristic_weight: float = typer.Option(1.0),
    beam_width: int | None = typer.Option(VALIDATION_DEFAULT_BEAM_WIDTH),
    timeout_seconds: float | None = typer.Option(
        VALIDATION_DEFAULT_TIMEOUT_SECONDS,
        help="Solver budget profile aligned with external validation. Use 0 to disable.",
    ),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    payload = json.loads(input.read_text(encoding="utf-8"))
    result = _solve_payload(
        master=master,
        payload=payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=_cli_time_budget_ms(timeout_seconds),
    )
    typer.echo(json.dumps(result, ensure_ascii=False))


@app.command("solve-suite")
def solve_suite(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
    solver: str = typer.Option(VALIDATION_DEFAULT_SOLVER),
    heuristic_weight: float = typer.Option(1.0),
    beam_width: int | None = typer.Option(VALIDATION_DEFAULT_BEAM_WIDTH),
    timeout_seconds: float | None = typer.Option(
        VALIDATION_DEFAULT_TIMEOUT_SECONDS,
        help="Per-scenario solver budget profile aligned with external validation. Use 0 to disable.",
    ),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    suite_payload = json.loads(input.read_text(encoding="utf-8"))
    scenarios = suite_payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise typer.BadParameter("input must be a suite JSON with scenarios[]")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict) or not isinstance(scenario.get("payload"), dict):
            raise typer.BadParameter(f"invalid scenario entry at index {index}")
        scenario_name = str(scenario.get("name", f"scenario_{index}"))
        try:
            scenario_result = _solve_suite_payload(
                master=master,
                payload=scenario["payload"],
                solver=solver,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
                time_budget_ms=_cli_time_budget_ms(timeout_seconds),
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
        (output_dir / f"{scenario_name}.plan.json").write_text(
            json.dumps(scenario_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "suite": suite_payload.get("suite", input.stem),
        "scenarioCount": len(results),
        "solver": solver,
        "heuristicWeight": heuristic_weight,
        "beamWidth": beam_width,
        "results": [
            {
                "name": item["name"],
                "description": item.get("description", ""),
                "scenarioType": item.get("scenario_type", "single"),
                "hookCount": item["hook_count"],
                "stageCount": item.get("stage_count"),
                "isValid": item["is_valid"],
                "vehicleCount": item["vehicle_count"],
                "verifierErrors": item["verifier_errors"],
                "solverErrors": item.get("solver_errors", []),
            }
            for item in results
        ],
    }
    (output_dir / "suite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(report, ensure_ascii=False))


@app.command("solve-workflow")
def solve_workflow_cmd(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    solver: str = typer.Option(VALIDATION_DEFAULT_SOLVER),
    heuristic_weight: float = typer.Option(1.0),
    beam_width: int | None = typer.Option(VALIDATION_DEFAULT_BEAM_WIDTH),
    timeout_seconds: float | None = typer.Option(
        VALIDATION_DEFAULT_TIMEOUT_SECONDS,
        help="Per-stage solver budget profile aligned with external validation. Use 0 to disable.",
    ),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    payload = json.loads(input.read_text(encoding="utf-8"))
    result = solve_workflow(
        master,
        payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=_cli_time_budget_ms(timeout_seconds),
    )
    typer.echo(
        json.dumps(
            {
                "stageCount": result.stage_count,
                "stages": [
                    {
                        "name": stage.name,
                        "description": stage.description,
                        "isValid": stage.view.summary.is_valid if stage.view else False,
                        "hookCount": stage.view.summary.hook_count if stage.view else 0,
                        "finalTracks": stage.view.summary.final_tracks if stage.view else [],
                        "finalSpotAssignments": stage.view.final_spot_assignments if stage.view else {},
                        "workPositionAssignments": stage.view.final_work_position_assignments if stage.view else {},
                        "failedHookNos": stage.view.failed_hook_nos if stage.view else [],
                        "verifierErrors": stage.view.verifier_errors if stage.view else [],
                    }
                    for stage in result.stages
                ],
            },
            ensure_ascii=False,
        )
    )


def _solve_payload(
    *,
    master,
    payload: dict,
    solver: str | None,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None = None,
) -> dict:
    normalized = normalize_plan_input(payload, master)
    initial = build_initial_state(normalized)
    effective_solver = solver or VALIDATION_DEFAULT_SOLVER
    effective_beam_width = (
        beam_width
        if beam_width is not None
        else VALIDATION_DEFAULT_BEAM_WIDTH
        if effective_solver == "beam"
        else None
    )
    effective_time_budget_ms = (
        time_budget_ms
        if time_budget_ms is not None
        else validation_time_budget_ms(VALIDATION_DEFAULT_TIMEOUT_SECONDS)
    )
    result = solve_with_validation_recovery_result(
        normalized,
        initial,
        master=master,
        solver_mode=effective_solver,
        heuristic_weight=heuristic_weight,
        beam_width=effective_beam_width,
        time_budget_ms=effective_time_budget_ms,
        enable_depot_late_scheduling=False,
        solve_result_fn=solve_with_simple_astar_result,
    )
    if not result.is_complete:
        return {
            "scenario_type": "single",
            "solver": effective_solver,
            "fallback_stage": result.fallback_stage,
            "partial_fallback_stage": result.partial_fallback_stage,
            "is_proven_optimal": result.is_proven_optimal,
            "debug_stats": result.debug_stats or {},
            "vehicle_count": len(normalized.vehicles),
            "hook_plan": [],
            "hook_count": 0,
            "partial_hook_count": len(result.partial_plan),
            "final_state": None,
            "is_valid": False,
            "verifier_errors": [],
            "solver_errors": [
                "No complete solver plan found within validation-profile recovery"
            ],
        }
    plan = result.plan
    route_oracle = RouteOracle(master)
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in normalized.vehicles}
    hook_plan = [
        _build_hook_payload(idx, item, route_oracle, length_by_vehicle)
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
        "solver": effective_solver,
        "fallback_stage": result.fallback_stage,
        "is_proven_optimal": result.is_proven_optimal,
        "debug_stats": result.debug_stats or {},
        "vehicle_count": len(normalized.vehicles),
        "hook_plan": hook_plan,
        "hook_count": len(hook_plan),
        "final_state": replay.final_state.model_dump(mode="json"),
        "is_valid": verify_report.is_valid,
        "verifier_errors": verify_report.errors,
        "solver_errors": [],
    }


def _solve_suite_payload(
    *,
    master,
    payload: dict,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None = None,
) -> dict:
    if _is_workflow_payload(payload):
        result = solve_workflow(
            master,
            payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
            time_budget_ms=time_budget_ms,
        )
        stage_payloads = [
            {
                "name": stage.name,
                "description": stage.description,
                "isValid": stage.view.summary.is_valid if stage.view else False,
                "hookCount": stage.view.summary.hook_count if stage.view else 0,
                "finalTracks": stage.view.summary.final_tracks if stage.view else [],
                "finalSpotAssignments": stage.view.final_spot_assignments if stage.view else {},
                "workPositionAssignments": stage.view.final_work_position_assignments if stage.view else {},
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
    return _solve_payload(
        master=master,
        payload=payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
    )


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


def _cli_time_budget_ms(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is not None and timeout_seconds <= 0:
        return None
    return validation_time_budget_ms(timeout_seconds)


def _build_hook_payload(
    idx: int,
    item,
    route_oracle: RouteOracle,
    length_by_vehicle: dict[str, float],
) -> dict:
    validation = route_oracle.validate_path(
        source_track=item.source_track,
        target_track=item.target_track,
        path_tracks=item.path_tracks,
        train_length_m=sum(length_by_vehicle[vehicle_no] for vehicle_no in item.vehicle_nos),
    )
    return {
        "hookNo": idx,
        "actionType": item.action_type,
        "sourceTrack": item.source_track,
        "targetTrack": item.target_track,
        "vehicleCount": len(item.vehicle_nos),
        "vehicleNos": item.vehicle_nos,
        "pathTracks": item.path_tracks,
        "remark": _build_hook_remark(validation),
    }


def _build_hook_remark(validation: PathValidationResult) -> str:
    parts: list[str] = []
    if validation.total_length_m is not None:
        parts.append(f"route={validation.total_length_m:.1f}m")
    if validation.branch_codes:
        parts.append("branches=" + ",".join(validation.branch_codes))
    if validation.reverse_branch_codes:
        parts.append("reverse=" + ",".join(validation.reverse_branch_codes))
    return "; ".join(parts)


@app.command("verify")
def verify(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    plan: Path = typer.Option(..., exists=True, dir_okay=False),
):
    master = load_master_data(DEFAULT_MASTER_DIR)
    payload = json.loads(input.read_text(encoding="utf-8"))
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    normalized = normalize_plan_input(payload, master)
    hook_plan = plan_payload["hook_plan"] if "hook_plan" in plan_payload else plan_payload
    report = verify_plan(master, normalized, hook_plan)
    typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))


@app.command("benchmark")
def benchmark(
    output_dir: Path = typer.Option(Path("artifacts/benchmark"), dir_okay=True, file_okay=False),
    scenario_count: int = typer.Option(10),
    vehicle_count: int = typer.Option(4),
    seed_start: int = typer.Option(0),
    direct_only: bool = typer.Option(True),
    profile: str = typer.Option("micro"),
    solver: str = typer.Option("exact"),
    heuristic_weight: float = typer.Option(1.0),
    beam_width: int | None = typer.Option(None),
):
    report = run_benchmark(
        output_dir=output_dir,
        scenario_count=scenario_count,
        vehicle_count=vehicle_count,
        seed_start=seed_start,
        direct_only=direct_only,
        profile=profile,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
    )
    typer.echo(json.dumps(report, ensure_ascii=False))


@app.command("compare-suite")
def compare_suite(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
    solver: list[str] = typer.Option(["exact", "weighted", "beam", "lns"]),
    weighted_heuristic_weight: float = typer.Option(2.0),
    beam_width: int = typer.Option(8),
    lns_beam_width: int = typer.Option(8),
):
    solver_configs: list[dict] = []
    for solver_name in solver:
        if solver_name == "weighted":
            solver_configs.append({"solver": solver_name, "heuristic_weight": weighted_heuristic_weight})
        elif solver_name == "beam":
            solver_configs.append({"solver": solver_name, "beam_width": beam_width})
        elif solver_name == "lns":
            solver_configs.append({"solver": solver_name, "beam_width": lns_beam_width})
        else:
            solver_configs.append({"solver": solver_name})
    report = run_solver_comparison_suite(
        suite_path=input,
        output_dir=output_dir,
        solver_configs=solver_configs,
    )
    typer.echo(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    app()
