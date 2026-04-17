from __future__ import annotations

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.domain.route_oracle import PathValidationResult, RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.astar_solver import solve_with_simple_astar
from fzed_shunting.verify.plan_verifier import verify_plan
from fzed_shunting.verify.replay import ReplayState, build_initial_state, replay_plan


class DemoHook(BaseModel):
    hook_no: int
    action_type: str
    source_track: str
    target_track: str
    vehicle_count: int
    vehicle_nos: list[str] = Field(default_factory=list)
    path_tracks: list[str] = Field(default_factory=list)
    branch_codes: list[str] = Field(default_factory=list)
    route_length_m: float | None = None
    reverse_branch_codes: list[str] = Field(default_factory=list)
    required_reverse_clearance_m: float | None = None
    remark: str = ""


class DemoTrackNode(BaseModel):
    track_code: str
    is_occupied: bool
    vehicle_nos: list[str] = Field(default_factory=list)
    has_loco: bool = False
    is_in_active_path: bool = False
    is_changed: bool = False


class DemoTrackMap(BaseModel):
    track_nodes: dict[str, DemoTrackNode] = Field(default_factory=dict)
    active_path_tracks: list[str] = Field(default_factory=list)
    changed_tracks: list[str] = Field(default_factory=list)


class DemoTopologyNode(BaseModel):
    track_code: str


class DemoTopologyGraph(BaseModel):
    nodes: dict[str, DemoTopologyNode] = Field(default_factory=dict)
    edge_keys: list[tuple[str, str]] = Field(default_factory=list)
    active_edge_keys: list[tuple[str, str]] = Field(default_factory=list)


class DemoTransitionFrame(BaseModel):
    frame_index: int
    progress: float
    current_track: str
    passed_tracks: list[str] = Field(default_factory=list)


class DemoStep(BaseModel):
    step_index: int
    hook: DemoHook | None = None
    loco_track_name: str
    changed_tracks: list[str] = Field(default_factory=list)
    track_sequences: dict[str, list[str]] = Field(default_factory=dict)
    weighed_vehicle_nos: list[str] = Field(default_factory=list)
    spot_assignments: dict[str, str] = Field(default_factory=dict)
    verifier_errors: list[str] = Field(default_factory=list)
    track_map: DemoTrackMap = Field(default_factory=DemoTrackMap)
    topology_graph: DemoTopologyGraph = Field(default_factory=DemoTopologyGraph)
    transition_frames: list[DemoTransitionFrame] = Field(default_factory=list)


class DemoSummary(BaseModel):
    hook_count: int
    vehicle_count: int
    is_valid: bool
    error_count: int
    final_tracks: list[str] = Field(default_factory=list)
    weighed_vehicle_count: int
    assigned_spot_count: int


class DemoViewModel(BaseModel):
    summary: DemoSummary
    verifier_errors: list[str] = Field(default_factory=list)
    hook_plan: list[DemoHook] = Field(default_factory=list)
    steps: list[DemoStep] = Field(default_factory=list)
    final_spot_assignments: dict[str, str] = Field(default_factory=dict)
    failed_hook_nos: list[int] = Field(default_factory=list)
    track_map: DemoTrackMap = Field(default_factory=DemoTrackMap)
    topology_graph: DemoTopologyGraph = Field(default_factory=DemoTopologyGraph)
    comparison_summary: dict[str, object] | None = None


class DemoWorkflowViewModel(BaseModel):
    workflow: object


def select_demo_payload(
    payload: dict,
    selected_name: str | None = None,
) -> tuple[dict, list[str], str | None]:
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        return payload, [], None

    valid_scenarios = [
        item for item in scenarios if isinstance(item, dict) and isinstance(item.get("payload"), dict)
    ]
    if not valid_scenarios:
        raise ValueError("suite payload does not contain any valid scenarios")

    scenario_names = [str(item.get("name", f"scenario_{idx + 1}")) for idx, item in enumerate(valid_scenarios)]
    active_name = selected_name if selected_name in scenario_names else scenario_names[0]
    active_index = scenario_names.index(active_name)
    selected_payload = valid_scenarios[active_index]["payload"]
    return selected_payload, scenario_names, active_name


def build_demo_view_model(
    master: MasterData,
    payload: dict,
    plan_payload: list[dict] | dict | None = None,
    solver: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
    initial_state_override: ReplayState | None = None,
) -> DemoViewModel:
    normalized = normalize_plan_input(
        payload,
        master,
        allow_internal_loco_tracks=initial_state_override is not None,
    )
    initial = initial_state_override or build_initial_state(normalized)
    route_oracle = RouteOracle(master)
    length_by_vehicle = {vehicle.vehicle_no: vehicle.vehicle_length for vehicle in normalized.vehicles}
    raw_hook_plan = _resolve_hook_plan(
        master=master,
        normalized=normalized,
        initial=initial,
        route_oracle=route_oracle,
        length_by_vehicle=length_by_vehicle,
        plan_payload=plan_payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
    )
    hook_plan = [DemoHook.model_validate(item) for item in raw_hook_plan]
    replay = replay_plan(
        initial,
        [
            {
                "hookNo": hook.hook_no,
                "actionType": hook.action_type,
                "sourceTrack": hook.source_track,
                "targetTrack": hook.target_track,
                "vehicleNos": hook.vehicle_nos,
                "pathTracks": hook.path_tracks,
            }
            for hook in hook_plan
        ],
        plan_input=normalized,
    )
    verify_report = verify_plan(
        master,
        normalized,
        [
            {
                "hookNo": hook.hook_no,
                "actionType": hook.action_type,
                "sourceTrack": hook.source_track,
                "targetTrack": hook.target_track,
                "vehicleNos": hook.vehicle_nos,
                "pathTracks": hook.path_tracks,
            }
            for hook in hook_plan
        ],
        initial_state_override=initial,
    )
    hook_error_by_no = {
        report.hook_no: list(report.errors)
        for report in verify_report.hook_reports
        if report.errors
    }
    final_state = replay.final_state
    summary = DemoSummary(
        hook_count=len(hook_plan),
        vehicle_count=len(normalized.vehicles),
        is_valid=verify_report.is_valid,
        error_count=len(verify_report.errors),
        final_tracks=sorted(track for track, seq in final_state.track_sequences.items() if seq),
        weighed_vehicle_count=len(final_state.weighed_vehicle_nos),
        assigned_spot_count=len(final_state.spot_assignments),
    )
    visible_track_codes = _visible_track_codes_from_snapshots(replay.snapshots, hook_plan)
    topology_graph = _build_topology_graph(
        visible_track_codes=visible_track_codes,
        edge_keys=_collect_edge_keys(hook_plan),
        active_edge_keys=[],
    )
    comparison_summary = None
    if plan_payload is not None:
        solver_hook_count: int | None = None
        solver_error: str | None = None
        try:
            solver_plan = solve_with_simple_astar(
                normalized,
                initial,
                master=master,
                solver_mode=solver,
                heuristic_weight=heuristic_weight,
                beam_width=beam_width,
            )
            solver_hook_count = len(solver_plan)
        except Exception as exc:  # noqa: BLE001
            solver_error = str(exc)
        external_hook_count = len(hook_plan)
        comparison_summary = {
            "solverHookCount": solver_hook_count,
            "externalHookCount": external_hook_count,
            "hookCountDelta": (
                external_hook_count - solver_hook_count
                if solver_hook_count is not None
                else None
            ),
            "externalIsValid": verify_report.is_valid,
            "failedHookNos": sorted(hook_error_by_no),
            "solverError": solver_error,
        }
    return DemoViewModel(
        summary=summary,
        verifier_errors=verify_report.errors,
        hook_plan=hook_plan,
        steps=_build_steps(
            replay.snapshots,
            hook_plan,
            hook_error_by_no,
            visible_track_codes=visible_track_codes,
            global_edge_keys=topology_graph.edge_keys,
        ),
        final_spot_assignments=dict(sorted(final_state.spot_assignments.items())),
        failed_hook_nos=sorted(hook_error_by_no),
        track_map=_build_track_map(
            snapshot=replay.snapshots[0],
            hook=None,
            changed_tracks=[],
            visible_track_codes=visible_track_codes,
        ),
        topology_graph=topology_graph,
        comparison_summary=comparison_summary,
    )


def build_demo_workflow_view_model(
    master: MasterData,
    payload: dict,
    solver: str = "exact",
    heuristic_weight: float = 1.0,
    beam_width: int | None = None,
) -> DemoWorkflowViewModel:
    from fzed_shunting.workflow.runner import solve_workflow

    return DemoWorkflowViewModel(
        workflow=solve_workflow(
            master,
            payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
        )
    )


def _resolve_hook_plan(
    *,
    master: MasterData,
    normalized,
    initial: ReplayState,
    route_oracle: RouteOracle,
    length_by_vehicle: dict[str, float],
    plan_payload: list[dict] | dict | None,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
) -> list[dict]:
    if plan_payload is None:
        plan = solve_with_simple_astar(
            normalized,
            initial,
            master=master,
            solver_mode=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width,
        )
        return [
            _build_demo_hook(
                idx,
                move,
                route_oracle=route_oracle,
                length_by_vehicle=length_by_vehicle,
            ).model_dump(mode="json")
            for idx, move in enumerate(plan, start=1)
        ]
    raw_hooks = plan_payload.get("hook_plan", plan_payload) if isinstance(plan_payload, dict) else plan_payload
    resolved: list[dict] = []
    for idx, hook in enumerate(raw_hooks, start=1):
        validation = route_oracle.validate_path(
            source_track=hook["sourceTrack"],
            target_track=hook["targetTrack"],
            path_tracks=list(hook.get("pathTracks", [])),
            train_length_m=sum(length_by_vehicle[vehicle_no] for vehicle_no in hook["vehicleNos"]),
        )
        resolved.append(
            {
                "hook_no": hook.get("hookNo", idx),
                "action_type": hook["actionType"],
                "source_track": hook["sourceTrack"],
                "target_track": hook["targetTrack"],
                "vehicle_count": hook.get("vehicleCount", len(hook["vehicleNos"])),
                "vehicle_nos": list(hook["vehicleNos"]),
                "path_tracks": list(hook.get("pathTracks", [])),
                "branch_codes": list(validation.branch_codes),
                "route_length_m": validation.total_length_m,
                "reverse_branch_codes": list(validation.reverse_branch_codes),
                "required_reverse_clearance_m": validation.required_reverse_clearance_m,
                "remark": hook.get("remark", _build_hook_remark(validation)),
            }
        )
    return resolved


def _build_demo_hook(idx, move, route_oracle: RouteOracle, length_by_vehicle: dict[str, float]) -> DemoHook:
    validation = route_oracle.validate_path(
        source_track=move.source_track,
        target_track=move.target_track,
        path_tracks=list(move.path_tracks),
        train_length_m=sum(length_by_vehicle[vehicle_no] for vehicle_no in move.vehicle_nos),
    )
    return DemoHook(
        hook_no=idx,
        action_type=move.action_type,
        source_track=move.source_track,
        target_track=move.target_track,
        vehicle_count=len(move.vehicle_nos),
        vehicle_nos=list(move.vehicle_nos),
        path_tracks=list(move.path_tracks),
        branch_codes=list(validation.branch_codes),
        route_length_m=validation.total_length_m,
        reverse_branch_codes=list(validation.reverse_branch_codes),
        required_reverse_clearance_m=validation.required_reverse_clearance_m,
        remark=_build_hook_remark(validation),
    )


def _build_hook_remark(validation: PathValidationResult) -> str:
    parts: list[str] = []
    if validation.total_length_m is not None:
        parts.append(f"route={validation.total_length_m:.1f}m")
    if validation.branch_codes:
        parts.append("branches=" + ",".join(validation.branch_codes))
    if validation.reverse_branch_codes:
        parts.append("reverse=" + ",".join(validation.reverse_branch_codes))
    return "; ".join(parts)


def _build_steps(
    snapshots: list[ReplayState],
    hook_plan: list[DemoHook],
    hook_error_by_no: dict[int, list[str]],
    *,
    visible_track_codes: list[str],
    global_edge_keys: list[tuple[str, str]],
) -> list[DemoStep]:
    steps: list[DemoStep] = []
    for index, snapshot in enumerate(snapshots):
        previous = snapshots[index - 1] if index > 0 else None
        hook = hook_plan[index - 1] if index > 0 else None
        changed_tracks = _detect_changed_tracks(previous, snapshot, hook)
        steps.append(
            DemoStep(
                step_index=index,
                hook=hook,
                loco_track_name=snapshot.loco_track_name,
                changed_tracks=changed_tracks,
                track_sequences={track: list(seq) for track, seq in snapshot.track_sequences.items()},
                weighed_vehicle_nos=sorted(snapshot.weighed_vehicle_nos),
                spot_assignments=dict(sorted(snapshot.spot_assignments.items())),
                verifier_errors=list(hook_error_by_no.get(hook.hook_no if hook else 0, [])),
                track_map=_build_track_map(
                    snapshot=snapshot,
                    hook=hook,
                    changed_tracks=changed_tracks,
                    visible_track_codes=visible_track_codes,
                ),
                topology_graph=_build_topology_graph(
                    visible_track_codes=visible_track_codes,
                    edge_keys=global_edge_keys,
                    active_edge_keys=_path_to_edge_keys(hook.path_tracks if hook is not None else []),
                ),
                transition_frames=_build_transition_frames(hook),
            )
        )
    return steps


def _build_track_map(
    snapshot: ReplayState,
    hook: DemoHook | None,
    changed_tracks: list[str],
    visible_track_codes: list[str],
) -> DemoTrackMap:
    active_path_tracks = list(hook.path_tracks) if hook is not None else []
    nodes: dict[str, DemoTrackNode] = {}
    for track_code in visible_track_codes:
        vehicle_nos = list(snapshot.track_sequences.get(track_code, []))
        nodes[track_code] = DemoTrackNode(
            track_code=track_code,
            is_occupied=bool(vehicle_nos),
            vehicle_nos=vehicle_nos,
            has_loco=snapshot.loco_track_name == track_code,
            is_in_active_path=track_code in active_path_tracks,
            is_changed=track_code in changed_tracks,
        )
    return DemoTrackMap(
        track_nodes=nodes,
        active_path_tracks=active_path_tracks,
        changed_tracks=list(changed_tracks),
    )


def _visible_track_codes(normalized) -> list[str]:
    track_codes = {info.track_name for info in normalized.track_info}
    return sorted(track_codes)


def _visible_track_codes_from_snapshots(
    snapshots: list[ReplayState],
    hook_plan: list[DemoHook],
) -> list[str]:
    track_codes: set[str] = set()
    for snapshot in snapshots:
        track_codes.update(snapshot.track_sequences.keys())
    for hook in hook_plan:
        track_codes.update(hook.path_tracks)
    return sorted(track_codes)


def _detect_changed_tracks(
    previous: ReplayState | None,
    current: ReplayState,
    hook: DemoHook | None,
) -> list[str]:
    if previous is None:
        return sorted(track for track, seq in current.track_sequences.items() if seq)
    changed = {
        track
        for track in set(previous.track_sequences) | set(current.track_sequences)
        if previous.track_sequences.get(track, []) != current.track_sequences.get(track, [])
    }
    if hook is not None:
        changed.add(hook.source_track)
        changed.add(hook.target_track)
    return sorted(changed)


def _collect_edge_keys(hook_plan: list[DemoHook]) -> list[tuple[str, str]]:
    edge_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for hook in hook_plan:
        for edge_key in _path_to_edge_keys(hook.path_tracks):
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edge_keys.append(edge_key)
    return edge_keys


def _path_to_edge_keys(path_tracks: list[str]) -> list[tuple[str, str]]:
    return [
        (path_tracks[index], path_tracks[index + 1])
        for index in range(len(path_tracks) - 1)
    ]


def _build_topology_graph(
    *,
    visible_track_codes: list[str],
    edge_keys: list[tuple[str, str]],
    active_edge_keys: list[tuple[str, str]],
) -> DemoTopologyGraph:
    return DemoTopologyGraph(
        nodes={
            track_code: DemoTopologyNode(track_code=track_code)
            for track_code in visible_track_codes
        },
        edge_keys=list(edge_keys),
        active_edge_keys=list(active_edge_keys),
    )


def _build_transition_frames(hook: DemoHook | None) -> list[DemoTransitionFrame]:
    if hook is None or len(hook.path_tracks) <= 1:
        return []
    segment_count = len(hook.path_tracks) - 1
    frames_per_segment = 3
    total_frames = segment_count * frames_per_segment
    frames: list[DemoTransitionFrame] = []
    for frame_index in range(total_frames + 1):
        progress = frame_index / total_frames
        track_position = progress * segment_count
        current_index = min(int(track_position), segment_count)
        current_track = hook.path_tracks[current_index]
        if frame_index == total_frames:
            current_track = hook.path_tracks[-1]
            passed_tracks = list(hook.path_tracks)
        else:
            passed_tracks = list(hook.path_tracks[: current_index + 1])
        frames.append(
            DemoTransitionFrame(
                frame_index=frame_index,
                progress=progress,
                current_track=current_track,
                passed_tracks=passed_tracks,
            )
        )
    return frames
