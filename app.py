from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import time
from html import escape

import streamlit as st
from PIL import Image

from fzed_shunting.demo.layout import (
    build_route_polyline,
    load_topology_layout,
    LayoutPoint,
    point_at_progress,
    route_to_svg_path,
)
from fzed_shunting.demo.schematic import build_schematic_route, load_schematic_layout
from fzed_shunting.demo.view_model import (
    build_demo_view_model,
    build_demo_workflow_view_model,
    select_demo_payload,
)
from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.tools.segmented_routes_svg import load_segmented_physical_routes


MASTER_DIR = Path(__file__).resolve().parent / "data" / "master"
_TOPOLOGY_LAYOUT = None
_SCHEMATIC_LAYOUT = None
_MASTER_DATA = None


def _get_master_data():
    global _MASTER_DATA
    if _MASTER_DATA is None:
        _MASTER_DATA = load_master_data(MASTER_DIR)
    return _MASTER_DATA


def _payload_cache_key(payload) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_demo_view_model(
    payload_key: str,
    plan_payload_key: str | None,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None,
    compare_external_plan: bool,
):
    payload = json.loads(payload_key)
    plan_payload = json.loads(plan_payload_key) if plan_payload_key else None
    return build_demo_view_model(
        _get_master_data(),
        payload,
        plan_payload=plan_payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
        compare_external_plan=compare_external_plan,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_workflow_view_model(
    payload_key: str,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None,
):
    payload = json.loads(payload_key)
    return build_demo_workflow_view_model(
        _get_master_data(),
        payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
        time_budget_ms=time_budget_ms,
    )


def main():
    st.set_page_config(page_title="福州东调车 Demo", layout="wide")
    st.title("福州东调车 Demo")
    st.caption("按场景文件回放 block 级求解结果，并显示 verifier 结论。")

    scenario_path = st.text_input("Scenario JSON 路径", value="")
    plan_path = st.text_input("可选 Plan JSON 路径", value="")
    auto_solver = st.checkbox(
        "自动选择算法（只设超时）",
        value=True,
        help="开启后自动从 exact 起步、失败或超时自动回退到 weighted/beam/LNS；关闭则手动指定 Solver。",
    )
    if auto_solver:
        timeout_seconds = st.number_input(
            "超时（秒）",
            min_value=1.0,
            max_value=600.0,
            value=20.0,
            step=1.0,
            help="总时间预算。超过预算求解器按 anytime 链条返回最好已知解。",
        )
        solver = "exact"
        heuristic_weight = 1.0
        beam_width = 32
        time_budget_ms: float | None = float(timeout_seconds) * 1000.0
    else:
        solver = st.selectbox("Solver", options=["exact", "weighted", "beam", "lns"], index=0)
        heuristic_weight = st.number_input("Heuristic Weight", min_value=1.0, value=1.0, step=0.5)
        beam_width = st.number_input("Beam Width", min_value=1, value=16, step=1)
        time_budget_ms = None
    compare_solvers = st.checkbox("Compare Solvers", value=False)
    compare_external_plan = st.checkbox(
        "对比外部 Plan 与当前 Solver",
        value=False,
        disabled=not bool(plan_path),
        help="默认只回放外部 Plan；开启后会额外运行一次当前 Solver 做钩数对比。",
    )
    if not scenario_path:
        st.info("先通过 CLI 生成一个 scenario.json，或使用 `artifacts/typical_suite.json` 中的典型场景，再粘贴路径。")
        return

    path = Path(scenario_path)
    if not path.exists():
        st.error("文件不存在")
        return

    master = _get_master_data()
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if _is_workflow_payload(raw_payload):
        _render_workflow_demo(
            master=master,
            payload=raw_payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width if solver in {"beam", "lns"} else None,
            time_budget_ms=time_budget_ms,
        )
        return
    selected_payload, scenario_names, active_scenario_name = select_demo_payload(raw_payload)
    payload = selected_payload
    if _is_workflow_payload(payload):
        if scenario_names:
            selected_name = st.selectbox(
                "典型 Workflow 场景",
                options=scenario_names,
                index=scenario_names.index(active_scenario_name) if active_scenario_name else 0,
            )
            payload, _, active_scenario_name = select_demo_payload(raw_payload, selected_name=selected_name)
            scenario_meta = next(
                (
                    item
                    for item in raw_payload.get("scenarios", [])
                    if item.get("name") == active_scenario_name
                ),
                None,
            )
            if scenario_meta and scenario_meta.get("description"):
                st.caption(str(scenario_meta["description"]))
        _render_workflow_demo(
            master=master,
            payload=payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width if solver in {"beam", "lns"} else None,
            time_budget_ms=time_budget_ms,
        )
        return
    if scenario_names:
        selected_name = st.selectbox(
            "典型场景",
            options=scenario_names,
            index=scenario_names.index(active_scenario_name) if active_scenario_name else 0,
        )
        payload, _, active_scenario_name = select_demo_payload(raw_payload, selected_name=selected_name)
        scenario_meta = next(
            (
                item
                for item in raw_payload.get("scenarios", [])
                if item.get("name") == active_scenario_name
            ),
            None,
        )
        if scenario_meta and scenario_meta.get("description"):
            st.caption(str(scenario_meta["description"]))
    vehicle_display_metadata = _build_vehicle_display_metadata(payload)
    plan_payload = None
    if plan_path:
        candidate = Path(plan_path)
        if not candidate.exists():
            st.error("Plan 文件不存在")
            return
        plan_payload = json.loads(candidate.read_text(encoding="utf-8"))
    try:
        with st.spinner("求解中…"):
            view = _cached_demo_view_model(
                _payload_cache_key(payload),
                _payload_cache_key(plan_payload) if plan_payload is not None else None,
                solver,
                heuristic_weight,
                beam_width if solver in {"beam", "lns"} else None,
                time_budget_ms,
                compare_external_plan,
            )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    summary_cols = st.columns(6)
    summary_cols[0].metric("车辆数", view.summary.vehicle_count)
    summary_cols[1].metric("钩数", view.summary.hook_count)
    summary_cols[2].metric("已称重车辆", view.summary.weighed_vehicle_count)
    summary_cols[3].metric("最终占用线", len(view.summary.final_tracks))
    summary_cols[4].metric("库内台位", view.summary.assigned_spot_count)
    summary_cols[5].metric("Verifier", "PASS" if view.summary.is_valid else "FAIL")

    if view.verifier_errors:
        st.error("校验未通过")
        st.json(view.verifier_errors)
    else:
        st.success("校验通过")

    if view.comparison_summary is not None:
        st.subheader("外部计划对比")
        panel = _build_comparison_panel(view.comparison_summary)
        compare_cols = st.columns(len(panel["metrics"]))
        for column, metric in zip(compare_cols, panel["metrics"], strict=False):
            column.metric(metric["label"], metric["value"])
        for caption in panel["captions"]:
            st.caption(caption)

    if compare_solvers and not plan_payload:
        st.subheader("Solver 对比")
        rows = []
        for solver_name in ["exact", "weighted", "beam", "lns"]:
            try:
                compare_view = _cached_demo_view_model(
                    _payload_cache_key(payload),
                    None,
                    solver_name,
                    heuristic_weight,
                    beam_width if solver_name in {"beam", "lns"} else None,
                    time_budget_ms,
                    False,
                )
                rows.append(
                    {
                        "solver": solver_name,
                        "isValid": compare_view.summary.is_valid,
                        "hooks": compare_view.summary.hook_count,
                        "vehicles": compare_view.summary.vehicle_count,
                        "weighed": compare_view.summary.weighed_vehicle_count,
                        "assignedSpots": compare_view.summary.assigned_spot_count,
                        "errors": " | ".join(compare_view.verifier_errors[:2]),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "solver": solver_name,
                        "isValid": False,
                        "hooks": None,
                        "vehicles": None,
                        "weighed": None,
                        "assignedSpots": None,
                        "errors": str(exc),
                    }
                )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    replay_tab, vehicles_tab, plan_tab, overview_tab, distance_tab = st.tabs(
        ["回放", "车辆分布", "钩计划", "线路总览", "距离构成"]
    )

    with replay_tab:
        st.subheader("逐钩回放")
        control_cols = st.columns([2, 1, 1])
        with control_cols[0]:
            step_index = st.slider(
                "Step",
                min_value=0,
                max_value=len(view.steps) - 1,
                value=view.failed_hook_nos[0] if view.failed_hook_nos else 0,
            )
        with control_cols[1]:
            autoplay = st.checkbox("Auto Play", value=False)
        with control_cols[2]:
            autoplay_interval_ms = st.number_input(
                "Interval (ms)", min_value=100, value=600, step=100
            )
        step_container = st.container()
        if autoplay:
            for auto_index in range(len(view.steps)):
                with step_container:
                    _render_step(view, auto_index, vehicle_display_metadata=vehicle_display_metadata)
                time.sleep(autoplay_interval_ms / 1000)
        else:
            with step_container:
                _render_step(view, step_index, vehicle_display_metadata=vehicle_display_metadata)

    with vehicles_tab:
        st.caption("初始车辆分布与各自目的地，用来核对求解器的输入。")
        roster_rows = _build_vehicle_roster_rows(payload)
        if roster_rows:
            st.dataframe(roster_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("本场景没有初始车辆信息。")

    with plan_tab:
        download_cols = st.columns(2)
        download_cols[0].download_button(
            "下载钩计划 JSON",
            data=json.dumps(
                {
                    "hook_plan": [
                        {
                            "hookNo": hook.hook_no,
                            "actionType": hook.action_type,
                            "sourceTrack": hook.source_track,
                            "targetTrack": hook.target_track,
                            "vehicleCount": hook.vehicle_count,
                            "vehicleNos": hook.vehicle_nos,
                            "pathTracks": hook.path_tracks,
                            "remark": hook.remark,
                        }
                        for hook in view.hook_plan
                    ],
                    "verifierErrors": view.verifier_errors,
                    "failedHookNos": view.failed_hook_nos,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="hook_plan.json",
            mime="application/json",
        )
        download_cols[1].download_button(
            "下载回放快照 JSON",
            data=json.dumps(
                {
                    "steps": [
                        {
                            "stepIndex": step.step_index,
                            "hookNo": step.hook.hook_no if step.hook else None,
                            "locoTrackName": step.loco_track_name,
                            "changedTracks": step.changed_tracks,
                            "trackSequences": step.track_sequences,
                            "weighedVehicleNos": step.weighed_vehicle_nos,
                            "spotAssignments": step.spot_assignments,
                            "verifierErrors": step.verifier_errors,
                        }
                        for step in view.steps
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="replay_steps.json",
            mime="application/json",
        )

        st.dataframe(
            [
                {
                    "hookNo": hook.hook_no,
                    "actionType": hook.action_type,
                    "sourceTrack": hook.source_track,
                    "targetTrack": hook.target_track,
                    "vehicleCount": hook.vehicle_count,
                    "vehicleNos": " ".join(hook.vehicle_nos),
                    "pathTracks": " -> ".join(hook.path_tracks),
                    "routeLengthM": hook.route_length_m,
                    "reverseBranches": " ".join(hook.reverse_branch_codes),
                    "remark": hook.remark,
                }
                for hook in view.hook_plan
            ],
            use_container_width=True,
            hide_index=True,
        )
        if view.failed_hook_nos:
            st.warning(f"失败钩号: {', '.join(str(item) for item in view.failed_hook_nos)}")

    with overview_tab:
        st.caption("固定方位示意图用于理解全场结构，路径和状态不在这里叠加。")
        st.markdown(
            _build_topology_svg(
                view.topology_graph,
                None,
                hook=None,
                show_all_labels=True,
            ),
            unsafe_allow_html=True,
        )

    with distance_tab:
        st.caption("距离构成：聚合分支与分段股道，用于核对路径距离数据。")
        st.dataframe(
            _build_distance_catalog_rows(),
            use_container_width=True,
            hide_index=True,
        )


def _render_workflow_demo(
    master,
    payload,
    solver: str,
    heuristic_weight: float,
    beam_width: int | None,
    time_budget_ms: float | None = None,
):
    with st.spinner("求解中…"):
        workflow_view = _cached_workflow_view_model(
            _payload_cache_key(payload),
            solver,
            heuristic_weight,
            beam_width,
            time_budget_ms,
        )
    workflow = workflow_view.workflow
    st.subheader("多轮 Workflow")
    st.caption(f"共 {workflow.stage_count} 个阶段。")
    stage_names = [stage.name for stage in workflow.stages]
    stage_index = st.selectbox(
        "阶段",
        options=list(range(len(stage_names))),
        index=0,
        format_func=lambda idx: stage_names[idx],
    )
    st.progress(_workflow_progress_value(stage_index=stage_index, stage_count=workflow.stage_count))
    st.dataframe(
        _build_workflow_stage_rows(workflow),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(
        _build_workflow_transition_rows(workflow),
        use_container_width=True,
        hide_index=True,
    )
    stage = workflow.stages[stage_index]
    if stage.description:
        st.caption(stage.description)
    view = stage.view
    if view is None:
        st.error("阶段无结果")
        return
    vehicle_display_metadata = _build_vehicle_display_metadata(stage.input_payload)

    summary_cols = st.columns(6)
    summary_cols[0].metric("阶段车辆数", view.summary.vehicle_count)
    summary_cols[1].metric("阶段钩数", view.summary.hook_count)
    summary_cols[2].metric("已称重车辆", view.summary.weighed_vehicle_count)
    summary_cols[3].metric("最终占用线", len(view.summary.final_tracks))
    summary_cols[4].metric("库内台位", view.summary.assigned_spot_count)
    summary_cols[5].metric("Verifier", "PASS" if view.summary.is_valid else "FAIL")

    if view.verifier_errors:
        st.error("阶段校验未通过")
        st.json(view.verifier_errors)

    st.markdown("**阶段钩计划**")
    st.dataframe(
        [
            {
                "hookNo": hook.hook_no,
                "actionType": hook.action_type,
                "sourceTrack": hook.source_track,
                "targetTrack": hook.target_track,
                "vehicleCount": hook.vehicle_count,
                "vehicleNos": " ".join(hook.vehicle_nos),
                "pathTracks": " -> ".join(hook.path_tracks),
                "routeLengthM": hook.route_length_m,
                "remark": hook.remark,
            }
            for hook in view.hook_plan
        ],
        use_container_width=True,
        hide_index=True,
    )

    step_index = st.slider(
        "阶段 Step",
        min_value=0,
        max_value=len(view.steps) - 1,
        value=0,
        key=f"workflow-step-{stage_index}",
    )
    _render_step(view, step_index, vehicle_display_metadata=vehicle_display_metadata)


def _is_workflow_payload(payload: dict) -> bool:
    return isinstance(payload.get("workflowStages"), list)


def _render_step(view, step_index: int, *, vehicle_display_metadata: dict[str, dict[str, str]] | None = None):
    step = view.steps[step_index]
    vehicle_meta = vehicle_display_metadata or {}
    if step.hook is None:
        st.info("Step 0 为初始状态。")
    else:
        st.write(f"第 {step.hook.hook_no} 钩: {_hook_title(step.hook)}")
        st.caption(
            f"车辆: {_format_hook_vehicle_text(step.hook.vehicle_nos, vehicle_meta)} | "
            f"路径: {' -> '.join(step.hook.path_tracks)}"
        )
        if step.hook.remark:
            st.caption(step.hook.remark)
        if step.verifier_errors:
            st.error(step.verifier_errors)

    transition_frame = None
    canvas_col, sidebar_col = st.columns([7, 3])
    with canvas_col:
        st.markdown("**方位示意回放**")
        if step.transition_frames:
            frame_index = st.slider(
                "钩内动画帧",
                min_value=0,
                max_value=len(step.transition_frames) - 1,
                value=len(step.transition_frames) - 1,
                key=f"transition-frame-{step.step_index}",
            )
            transition_frame = step.transition_frames[frame_index]
        _render_topology_graph(step.topology_graph, step.track_map, hook=step.hook, transition_frame=transition_frame, spot_assignments=step.spot_assignments, vehicle_target_tracks=view.vehicle_target_tracks)
    with sidebar_col:
        st.markdown("**当前钩摘要**")
        _render_hook_sidebar(step, vehicle_target_tracks=view.vehicle_target_tracks)

    detail_tabs = st.tabs(["股道变化", "车辆明细", "校验结果", "本钩路径距离"])
    with detail_tabs[0]:
        rows = _build_step_state_rows(step.track_map, vehicle_meta)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("当前无需要关注的股道状态。")
    with detail_tabs[1]:
        _render_vehicle_detail_panel(step, view, vehicle_meta)
    with detail_tabs[2]:
        _render_verifier_panel(step, view)
    with detail_tabs[3]:
        rows = _build_distance_breakdown_rows(step.hook)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("初始状态无路径距离构成。")


def _render_topology_graph(topology_graph, track_map, hook=None, transition_frame=None, spot_assignments=None, vehicle_target_tracks=None):
    st.markdown(
        _build_topology_svg(
            topology_graph,
            track_map,
            hook=hook,
            transition_frame=transition_frame,
            spot_assignments=spot_assignments,
            vehicle_target_tracks=vehicle_target_tracks,
        ),
        unsafe_allow_html=True,
    )


def _build_topology_graph_dot(topology_graph, track_map) -> str:
    lines = [
        "graph shunting_topology {",
        '  graph [bgcolor="transparent", pad="0.3", nodesep="0.35", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fontname="PingFang SC", fontsize=11, color="#d8d1c2", penwidth=1.0];',
        '  edge [fontname="PingFang SC", color="#c5c0b5", penwidth=1.6];',
    ]
    active_edges = set(topology_graph.active_edge_keys)
    changed_tracks = set(track_map.changed_tracks)
    active_tracks = set(track_map.active_path_tracks)
    for track_code, node in topology_graph.nodes.items():
        track_state = track_map.track_nodes.get(track_code)
        if track_state is None:
            continue
        fill = "#f0ebe2"
        border = "#d8d1c2"
        if track_state.has_loco:
            fill = "#fde7cf"
            border = "#d8772a"
        elif track_state.is_in_active_path:
            fill = "#dcf0ec"
            border = "#1d6f6d"
        elif track_state.is_occupied:
            fill = "#f7f2ea"
            border = "#8f8574"
        if track_state.is_changed:
            border = "#d8772a"
        status_parts: list[str] = []
        if track_state.has_loco:
            status_parts.append("机车")
        status_parts.append(f"占用 {len(track_state.vehicle_nos)}" if track_state.is_occupied else "空")
        label = f'{track_code}\\n{" / ".join(status_parts)}'
        lines.append(
            f'  "{track_code}" [label="{label}", fillcolor="{fill}", color="{border}"];'
        )
    for left, right in topology_graph.edge_keys:
        edge_color = "#c5c0b5"
        penwidth = 1.6
        if (left, right) in active_edges or (right, left) in active_edges:
            edge_color = "#1d6f6d"
            penwidth = 3.0
        elif left in changed_tracks or right in changed_tracks:
            edge_color = "#d8772a"
            penwidth = 2.2
        lines.append(
            f'  "{left}" -- "{right}" [color="{edge_color}", penwidth={penwidth}];'
        )
    lines.append("}")
    return "\n".join(lines)


def _build_topology_svg(
    topology_graph,
    track_map,
    hook=None,
    transition_frame=None,
    animate: bool = False,
    show_all_labels: bool = False,
    spot_assignments: dict | None = None,
    vehicle_target_tracks: dict | None = None,
) -> str:
    layout = _get_schematic_layout()
    track_nodes = track_map.track_nodes if track_map is not None else {}
    active_tracks = set(track_map.active_path_tracks) if track_map is not None else set()
    changed_tracks = set(track_map.changed_tracks) if track_map is not None else set()
    occupied_tracks = {
        track_code
        for track_code, node in track_nodes.items()
        if node.is_occupied or node.has_loco
    }
    source_track = hook.source_track if hook is not None else None
    target_track = hook.target_track if hook is not None else None
    if source_track is not None:
        active_tracks.add(source_track)
    if target_track is not None:
        active_tracks.add(target_track)

    route = build_schematic_route(layout, hook.path_tracks if hook is not None else [])
    motion_path = route_to_svg_path(route)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
            f'{layout.canvas_width:.0f} {layout.canvas_height:.0f}" '
            'class="topology-svg">'
        ),
        "<style>",
        ".schematic-bg{fill:#fcfaf5;}",
        ".schematic-area{fill:#f4f0e6;stroke:#ddd4c1;stroke-width:1.5;}",
        ".schematic-area-label{font-family:PingFang SC,sans-serif;font-size:24px;font-weight:700;fill:#8c816d;text-anchor:middle;}",
        ".schematic-track{fill:none;stroke:#cbc3b3;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;}",
        ".schematic-track-mainline{stroke:#b7ae9d;stroke-width:11;}",
        ".schematic-track-active{fill:none;stroke:#0f766e;stroke-width:14;stroke-linecap:round;stroke-linejoin:round;}",
        ".schematic-track-changed{fill:none;stroke:#d97706;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:10 8;}",
        ".schematic-track-label{font-family:PingFang SC,sans-serif;font-size:16px;fill:#574f44;text-anchor:middle;}",
        ".schematic-track-label-active{font-family:PingFang SC,sans-serif;font-size:18px;font-weight:700;fill:#0f4f4a;text-anchor:middle;}",
        ".schematic-track-label-key{font-family:PingFang SC,sans-serif;font-size:16px;font-weight:600;fill:#6b5f4b;text-anchor:middle;}",
        ".schematic-badge{fill:#fffaf0;stroke:#d9cdb5;stroke-width:1.5;}",
        ".schematic-badge-text{font-family:PingFang SC,sans-serif;font-size:12px;font-weight:700;fill:#6a5d45;text-anchor:middle;dominant-baseline:middle;}",
        ".sb-parked{fill:#0f766e;stroke:#ffffff;stroke-width:1.5;}",
        ".sb-parked-txt{font-family:PingFang SC,sans-serif;font-size:11px;font-weight:700;fill:#ffffff;text-anchor:middle;dominant-baseline:middle;}",
        ".sb-transit{fill:#d97706;stroke:#ffffff;stroke-width:1.5;}",
        ".sb-transit-txt{font-family:PingFang SC,sans-serif;font-size:11px;font-weight:700;fill:#ffffff;text-anchor:middle;dominant-baseline:middle;}",
        ".schematic-endpoint{fill:#fdf7ed;stroke:#0f766e;stroke-width:3;}",
        ".schematic-endpoint-target{fill:#fff2e0;stroke:#d97706;stroke-width:3;}",
        ".schematic-endpoint-text{font-family:PingFang SC,sans-serif;font-size:12px;font-weight:700;text-anchor:middle;dominant-baseline:middle;}",
        ".moving-block-marker{fill:#0f766e;stroke:#ffffff;stroke-width:3;}",
        ".moving-block-label{font-family:PingFang SC,sans-serif;font-size:22px;fill:#ffffff;text-anchor:middle;dominant-baseline:middle;}",
        ".loco-marker{fill:#d97706;stroke:#ffffff;stroke-width:3;}",
        ".route-motion-path{fill:none;stroke:none;}",
        "</style>",
        f'<rect class="schematic-bg" x="0" y="0" width="{layout.canvas_width:.1f}" height="{layout.canvas_height:.1f}" />',
    ]

    for area in layout.areas:
        parts.append(
            f'<rect class="schematic-area" x="{area.x:.1f}" y="{area.y:.1f}" width="{area.width:.1f}" '
            f'height="{area.height:.1f}" rx="24" />'
        )
        parts.append(
            f'<text class="schematic-area-label" x="{area.center.x:.1f}" y="{area.y + 34:.1f}">{escape(area.label)}</text>'
        )

    for geometry in layout.track_geometries.values():
        if geometry.track_code in active_tracks:
            parts.append(f'<path class="schematic-track-active" d="{_track_polyline_to_svg(geometry.points)}" />')
        else:
            base_class = "schematic-track"
            if geometry.is_mainline:
                base_class += " schematic-track-mainline"
            parts.append(f'<path class="{base_class}" d="{_track_polyline_to_svg(geometry.points)}" />')

    if motion_path:
        parts.append(f'<path id="route-motion-path" class="route-motion-path" d="{motion_path}" />')

    for track_code in sorted(changed_tracks - set(hook.path_tracks if hook is not None else [])):
        geometry = layout.track_geometries.get(track_code)
        if geometry is None:
            continue
        parts.append(f'<path class="schematic-track-changed" d="{_track_polyline_to_svg(geometry.points)}" />')

    visible_labels = {
        track_code
        for track_code, geometry in layout.track_geometries.items()
        if geometry.always_visible or show_all_labels
    }
    visible_labels.update(active_tracks)
    visible_labels.update(changed_tracks)
    visible_labels.update(occupied_tracks)

    for track_code in sorted(visible_labels):
        geometry = layout.track_geometries.get(track_code)
        if geometry is None:
            continue
        label_class = "schematic-track-label"
        if geometry.always_visible and track_code not in active_tracks and track_code not in changed_tracks:
            label_class = "schematic-track-label-key"
        if track_code in active_tracks or track_code in changed_tracks:
            label_class = "schematic-track-label-active"
        parts.append(
            f'<text class="{label_class}" x="{geometry.label_anchor.x:.1f}" y="{geometry.label_anchor.y:.1f}">{escape(track_code)}</text>'
        )

    vtt = vehicle_target_tracks or {}
    for track_code in sorted(occupied_tracks):
        geometry = layout.track_geometries.get(track_code)
        node = track_nodes.get(track_code)
        if geometry is None or node is None or not node.vehicle_nos:
            continue
        in_place = sum(1 for v in node.vehicle_nos if track_code in vtt.get(v, []))
        not_in_place = len(node.vehicle_nos) - in_place
        bx = geometry.label_anchor.x + 34.0
        by = geometry.label_anchor.y - 12.0
        if in_place > 0 and not_in_place > 0:
            parts.append(f'<circle class="sb-parked" cx="{bx - 11:.1f}" cy="{by:.1f}" r="10" />')
            parts.append(f'<text class="sb-parked-txt" x="{bx - 11:.1f}" y="{by + 1:.1f}">{in_place}</text>')
            parts.append(f'<circle class="sb-transit" cx="{bx + 11:.1f}" cy="{by:.1f}" r="10" />')
            parts.append(f'<text class="sb-transit-txt" x="{bx + 11:.1f}" y="{by + 1:.1f}">{not_in_place}</text>')
        elif in_place > 0:
            parts.append(f'<circle class="sb-parked" cx="{bx:.1f}" cy="{by:.1f}" r="12" />')
            parts.append(f'<text class="sb-parked-txt" x="{bx:.1f}" y="{by + 1:.1f}">{in_place}</text>')
        else:
            parts.append(f'<circle class="sb-transit" cx="{bx:.1f}" cy="{by:.1f}" r="12" />')
            parts.append(f'<text class="sb-transit-txt" x="{bx:.1f}" y="{by + 1:.1f}">{not_in_place}</text>')

    if source_track is not None:
        parts.append(_schematic_endpoint_svg(layout, source_track, label="起", css_class="schematic-endpoint"))
    if target_track is not None:
        parts.append(_schematic_endpoint_svg(layout, target_track, label="终", css_class="schematic-endpoint-target"))

    if hook is not None and hook.vehicle_nos:
        if animate and motion_path:
            marker_label = escape(str(len(hook.vehicle_nos)))
            parts.append('<g class="moving-block-marker">')
            parts.append('<circle class="moving-block-marker" r="18" cx="0" cy="0" />')
            parts.append(f'<text class="moving-block-label" x="0" y="1">{marker_label}</text>')
            parts.append(
                '<animateMotion dur="2.4s" repeatCount="indefinite" rotate="auto">'
                '<mpath href="#route-motion-path" />'
                '</animateMotion>'
            )
            parts.append("</g>")
        elif transition_frame is not None:
            point = point_at_progress(route, transition_frame.progress) if motion_path else None
            if point is None:
                point = route.points[0] if route.points else LayoutPoint(x=0.0, y=0.0)
            parts.append(f'<circle class="moving-block-marker" cx="{point.x:.1f}" cy="{point.y:.1f}" r="18" />')
            parts.append(
                f'<text class="moving-block-label" x="{point.x:.1f}" y="{point.y + 1:.1f}">{len(hook.vehicle_nos)}</text>'
            )

    if track_map is not None:
        loco_track = next((track_code for track_code, node in track_nodes.items() if node.has_loco), None)
    else:
        loco_track = None
    if loco_track is not None and loco_track in layout.track_geometries:
        loco_center = layout.track_geometries[loco_track].center
        parts.append(
            f'<rect class="loco-marker" x="{loco_center.x - 12:.1f}" y="{loco_center.y - 48:.1f}" width="24" height="24" rx="6" />'
        )

    parts.append("</svg>")
    return "".join(parts)


def _track_polyline_to_svg(points) -> str:
    if not points:
        return ""
    first = points[0]
    segments = [f"M {first.x:.1f} {first.y:.1f}"]
    for point in points[1:]:
        segments.append(f"L {point.x:.1f} {point.y:.1f}")
    return " ".join(segments)


def _track_badge_width(track_code: str) -> float:
    return max(64.0, 18.0 * len(track_code) + 26.0)


def _background_rect_svg(rect, css_class: str) -> str:
    center_x = rect.x + rect.width / 2.0
    center_y = rect.y + rect.height / 2.0
    rotation = rect.rotation_deg
    transform = ""
    if abs(rotation) > 1e-6:
        transform = f' transform="rotate({rotation:.1f} {center_x:.1f} {center_y:.1f})"'
    return (
        f'<rect class="{css_class}" x="{rect.x:.1f}" y="{rect.y:.1f}" '
        f'width="{rect.width:.1f}" height="{rect.height:.1f}" rx="10"{transform} />'
    )


def _build_background_anchor_route(layout, track_codes: list[str]):
    points: list[LayoutPoint] = []
    for track_code in track_codes:
        geometry = layout.track_geometries.get(track_code)
        if geometry is None:
            continue
        anchor = geometry.background_anchor or geometry.center
        if not points or points[-1] != anchor:
            points.append(anchor)
    if not points:
        return build_route_polyline(layout, track_codes)
    total_length_px = 0.0
    cumulative_lengths = [0.0]
    for start, end in zip(points, points[1:], strict=False):
        total_length_px += ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        cumulative_lengths.append(total_length_px)
    from fzed_shunting.demo.layout import RoutePolyline

    return RoutePolyline(
        track_codes=list(track_codes),
        points=points,
        total_length_px=total_length_px,
        cumulative_lengths=cumulative_lengths,
    )


def _should_render_track_overlay(track_code, node, active_tracks: set[str], changed_tracks: set[str]) -> bool:
    if track_code in active_tracks or track_code in changed_tracks:
        return True
    if node is None:
        return False
    return node.has_loco


def _get_topology_layout():
    global _TOPOLOGY_LAYOUT
    if _TOPOLOGY_LAYOUT is None:
        _TOPOLOGY_LAYOUT = load_topology_layout(MASTER_DIR, _get_master_data())
    return _TOPOLOGY_LAYOUT


def _get_schematic_layout():
    global _SCHEMATIC_LAYOUT
    if _SCHEMATIC_LAYOUT is None:
        _SCHEMATIC_LAYOUT = load_schematic_layout(MASTER_DIR)
    return _SCHEMATIC_LAYOUT


def _schematic_endpoint_svg(layout, track_code: str, *, label: str, css_class: str) -> str:
    geometry = layout.track_geometries.get(track_code)
    if geometry is None:
        return ""
    center_x = geometry.label_anchor.x
    center_y = geometry.label_anchor.y - 26.0
    return (
        f'<g><circle class="{css_class}" cx="{center_x:.1f}" cy="{center_y:.1f}" r="13" />'
        f'<text class="schematic-endpoint-text" x="{center_x:.1f}" y="{center_y + 1:.1f}">{escape(label)}</text></g>'
    )


def _hook_title(hook) -> str:
    if hook.action_type == "ATTACH":
        return f"挂车 ← {hook.source_track}"
    if hook.action_type == "DETACH":
        return f"摘车 → {hook.target_track}"
    return f"{hook.source_track} → {hook.target_track}"


def _build_hook_sidebar_rows(step) -> list[dict[str, str]]:
    if step.hook is None:
        return [
            {"label": "状态", "value": "初始状态"},
            {"label": "机车位置", "value": step.loco_track_name},
            {"label": "变化股道", "value": "无"},
        ]

    route_length = f"{step.hook.route_length_m:.1f}m" if step.hook.route_length_m is not None else "未知"
    action_type = step.hook.action_type
    track_rows: list[dict[str, str]]
    if action_type == "ATTACH":
        track_rows = [{"label": "挂车股道", "value": step.hook.source_track}]
    elif action_type == "DETACH":
        track_rows = [{"label": "摘车股道", "value": step.hook.target_track}]
    else:
        track_rows = [
            {"label": "起点", "value": step.hook.source_track},
            {"label": "终点", "value": step.hook.target_track},
        ]
    return [
        {"label": "当前钩", "value": f"第 {step.hook.hook_no} 钩"},
        {"label": "动作", "value": action_type},
        *track_rows,
        {"label": "车辆数", "value": str(step.hook.vehicle_count)},
        {"label": "机车位置", "value": step.loco_track_name},
        {"label": "路径长度", "value": route_length},
        {"label": "变化股道", "value": ", ".join(step.changed_tracks) if step.changed_tracks else "无"},
    ]


def _build_step_state_rows(
    track_map,
    vehicle_display_metadata: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    vehicle_meta = vehicle_display_metadata or {}
    rows: list[dict[str, str]] = []
    for track_code, node in track_map.track_nodes.items():
        if not (node.is_in_active_path or node.is_changed or node.is_occupied or node.has_loco):
            continue
        state_parts: list[str] = []
        if node.has_loco:
            state_parts.append("机车")
        if node.is_in_active_path:
            state_parts.append("当前路径")
        if node.is_changed:
            state_parts.append("本步变化")
        state_parts.append(f"占用 {len(node.vehicle_nos)}" if node.is_occupied else "空")
        rows.append(
            {
                "trackCode": track_code,
                "state": " / ".join(state_parts),
                "vehicles": (
                    _format_hook_vehicle_text(node.vehicle_nos, vehicle_meta)
                    if node.vehicle_nos
                    else "无车辆"
                ),
            }
        )
    rows.sort(key=lambda item: item["trackCode"])
    return rows


def _build_distance_breakdown_rows(hook) -> list[dict[str, object]]:
    if hook is None:
        return []
    master = _get_master_data()
    rows: list[dict[str, object]] = []
    running_total = 0.0
    for index, track_code in enumerate(hook.path_tracks, start=1):
        track = master.tracks.get(track_code)
        if track is None:
            continue
        running_total += track.effective_length_m
        rows.append(
            {
                "order": index,
                "trackCode": track_code,
                "trackName": track.name,
                "effectiveLengthM": track.effective_length_m,
                "cumulativeEffectiveLengthM": round(running_total, 1),
            }
        )
    return rows


def _build_distance_catalog_rows() -> list[dict[str, object]]:
    master = _get_master_data()
    routes = load_segmented_physical_routes(MASTER_DIR, master)
    rows: list[dict[str, object]] = []
    for route in routes.values():
        rows.append(
            {
                "displayName": route.display_name,
                "branchCode": route.branch_code,
                "endpointSpan": f"{route.left_node or '?'} -> {route.right_node or '?'}",
                "segments": " / ".join(segment.track_code for segment in route.segments),
                "totalPhysicalDistanceM": route.aggregate_physical_distance_m,
            }
        )
    return rows


def _render_hook_sidebar(step, vehicle_target_tracks: dict | None = None) -> None:
    route_tracks = step.hook.path_tracks if step.hook is not None else []
    track_nodes = step.track_map.track_nodes if step.track_map else {}
    vtt = vehicle_target_tracks or {}
    cards = []
    for row in _build_hook_sidebar_rows(step):
        cards.append(
            f"""
            <div class="hook-sidebar-card">
              <div class="hook-sidebar-label">{escape(row['label'])}</div>
              <div class="hook-sidebar-value">{escape(row['value'])}</div>
            </div>
            """
        )

    chip_parts = []
    for track_code in route_tracks:
        node = track_nodes.get(track_code)
        vehicles = node.vehicle_nos if node and node.vehicle_nos else []
        in_place = sum(1 for v in vehicles if track_code in vtt.get(v, []))
        not_in_place = len(vehicles) - in_place
        if vehicles:
            count_html = ""
            if in_place > 0:
                count_html += f"<span class='rc-parked'>{in_place}</span>"
            if not_in_place > 0:
                count_html += f"<span class='rc-transit'>{not_in_place}</span>"
            chip_parts.append(
                f"<span class='route-chip route-chip-occupied'>"
                f"{escape(track_code)}{count_html}"
                f"</span>"
            )
        else:
            chip_parts.append(f"<span class='route-chip'>{escape(track_code)}</span>")
    chips = "".join(chip_parts) or "<span class='route-chip route-chip-muted'>初始状态</span>"

    st.markdown(
        """
        <style>
        .hook-sidebar-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 10px;
          margin-bottom: 12px;
        }
        .hook-sidebar-card {
          padding: 12px 14px;
          border-radius: 14px;
          background: linear-gradient(180deg, #fffaf1 0%, #f4ecde 100%);
          border: 1px solid #e3d7c0;
        }
        .hook-sidebar-label {
          font-size: 12px;
          color: #7c6f59;
          margin-bottom: 4px;
        }
        .hook-sidebar-value {
          font-size: 16px;
          font-weight: 700;
          color: #2f2a22;
        }
        .route-chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 8px;
        }
        .route-chip {
          padding: 6px 10px;
          border-radius: 999px;
          background: #dcf2ee;
          color: #0f4f4a;
          font-size: 13px;
          font-weight: 600;
        }
        .route-chip-occupied {
          background: #fff0d6;
          color: #7c3f00;
          border: 1px solid #e8a82a;
        }
        .route-chip-count {
          display: inline-block;
          background: #d97706;
          color: #ffffff;
          border-radius: 999px;
          font-size: 11px;
          padding: 1px 6px;
          margin-left: 5px;
        }
        .rc-parked {
          display: inline-block;
          background: #0f766e;
          color: #ffffff;
          border-radius: 999px;
          font-size: 11px;
          padding: 1px 6px;
          margin-left: 5px;
        }
        .rc-transit {
          display: inline-block;
          background: #d97706;
          color: #ffffff;
          border-radius: 999px;
          font-size: 11px;
          padding: 1px 6px;
          margin-left: 3px;
        }
        .route-chip-muted {
          background: #f1ede5;
          color: #7c6f59;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='hook-sidebar-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.markdown("**路径条**")
    st.markdown(f"<div class='route-chip-row'>{chips}</div>", unsafe_allow_html=True)


def _build_vehicle_roster_rows(payload: dict) -> list[dict[str, object]]:
    """Build per-vehicle overview rows from raw scenario payload.

    Includes current track, destination (目的地), and attributes so operators
    can see the full vehicle distribution and their goals at a glance.
    """

    rows: list[dict[str, object]] = []
    source = payload.get("initialVehicleInfo") or payload.get("vehicleInfo") or []
    # For workflow stages, targetTrack is not on initialVehicleInfo; fall back
    # to vehicleInfo for per-stage goals when both present.
    target_map: dict[str, str] = {}
    for item in payload.get("vehicleInfo", []) or []:
        vehicle_no = str(item.get("vehicleNo", ""))
        target = item.get("targetTrack")
        if vehicle_no and target:
            target_map[vehicle_no] = str(target)
    for item in source:
        vehicle_no = str(item.get("vehicleNo", ""))
        rows.append(
            {
                "vehicleNo": vehicle_no,
                "currentTrack": str(item.get("trackName", "")),
                "targetTrack": target_map.get(vehicle_no) or str(item.get("targetTrack", "") or ""),
                "vehicleModel": str(item.get("vehicleModel", "")),
                "vehicleLength": item.get("vehicleLength", ""),
                "repairProcess": str(item.get("repairProcess", "")),
                "attributes": str(item.get("vehicleAttributes", "") or ""),
            }
        )
    rows.sort(key=lambda r: (r["currentTrack"], r["vehicleNo"]))
    return rows


def _build_vehicle_display_metadata(payload: dict) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    sources = []
    initial_vehicle_info = payload.get("initialVehicleInfo")
    if isinstance(initial_vehicle_info, list):
        sources.append(initial_vehicle_info)
    vehicle_info = payload.get("vehicleInfo")
    if isinstance(vehicle_info, list):
        sources.append(vehicle_info)
    for source in sources:
        for item in source:
            vehicle_no = str(item.get("vehicleNo", "")).strip()
            if not vehicle_no:
                continue
            attributes_value = str(item.get("vehicleAttributes", "") or "").strip()
            metadata[vehicle_no] = {
                "requirement": _format_vehicle_requirement_text(item),
                "attributes": attributes_value or "无",
                "length": _format_vehicle_length_text(item.get("vehicleLength")),
            }
    return metadata


def _format_vehicle_requirement_text(vehicle_info: dict) -> str:
    target_track = str(vehicle_info.get("targetTrack", "") or "").strip()
    spotting_value = str(vehicle_info.get("isSpotting", "") or "").strip()
    target_spot_code = str(vehicle_info.get("targetSpotCode", "") or "").strip()
    if target_spot_code:
        return target_spot_code
    if spotting_value.isdigit() or spotting_value == "迎检":
        return spotting_value
    if spotting_value == "是":
        return f"{target_track}作业区" if target_track else "需要对位"
    if target_track:
        return target_track
    return "无"


def _format_vehicle_length_text(length_value) -> str:
    try:
        return f"{float(length_value):.1f}m"
    except (TypeError, ValueError):
        return "未知"


def _format_vehicle_display_text(
    vehicle_no: str,
    vehicle_display_metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    vehicle_meta = (vehicle_display_metadata or {}).get(vehicle_no)
    if not vehicle_meta:
        return vehicle_no
    return (
        f"{vehicle_no}(对位={vehicle_meta['requirement']}，"
        f"属性={vehicle_meta['attributes']}，"
        f"长度={vehicle_meta['length']})"
    )


def _format_hook_vehicle_text(
    vehicle_nos: list[str],
    vehicle_display_metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    return " ".join(
        _format_vehicle_display_text(vehicle_no, vehicle_display_metadata)
        for vehicle_no in vehicle_nos
    )


def _render_vehicle_detail_panel(step, view, vehicle_display_metadata: dict[str, dict[str, str]] | None = None) -> None:
    vehicle_meta = vehicle_display_metadata or {}
    if step.hook is not None and step.hook.vehicle_nos:
        action_type = step.hook.action_type
        st.dataframe(
            [
                {
                    "vehicleNo": _format_vehicle_display_text(vehicle_no, vehicle_meta),
                    **({} if action_type == "DETACH" else {"sourceTrack": step.hook.source_track}),
                    **({} if action_type == "ATTACH" else {"targetTrack": step.hook.target_track}),
                }
                for vehicle_no in step.hook.vehicle_nos
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("当前步骤无车辆移动。")
    if step.weighed_vehicle_nos:
        st.caption(f"本步已称重车辆: {_format_hook_vehicle_text(step.weighed_vehicle_nos, vehicle_meta)}")
    if step.spot_assignments:
        st.markdown("**当前台位分配**")
        st.dataframe(
            [
                {
                    "vehicleNo": _format_vehicle_display_text(vehicle_no, vehicle_meta),
                    "spotCode": spot_code,
                }
                for vehicle_no, spot_code in step.spot_assignments.items()
            ],
            use_container_width=True,
            hide_index=True,
        )
    elif view.final_spot_assignments:
        st.markdown("**最终台位分配**")
        st.dataframe(
            [
                {
                    "vehicleNo": _format_vehicle_display_text(vehicle_no, vehicle_meta),
                    "spotCode": spot_code,
                }
                for vehicle_no, spot_code in view.final_spot_assignments.items()
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_verifier_panel(step, view) -> None:
    if step.verifier_errors:
        st.error("当前步骤校验未通过")
        st.write(step.verifier_errors)
    elif view.verifier_errors:
        st.warning("整体计划存在校验问题，但当前步骤未命中错误。")
        st.write(view.verifier_errors[:2])
    else:
        st.success("当前步骤与整体计划校验通过")


@lru_cache(maxsize=1)
def _topology_background_data_uri(path: str, crop_box: tuple[int, int, int, int] | None) -> str:
    image = Image.open(path).convert("RGB")
    if crop_box is not None:
        image = image.crop(crop_box)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _get_topology_background_data_uri(layout) -> str | None:
    background = layout.background_image
    if background is None:
        return None
    image_path = (MASTER_DIR / background.path).resolve()
    return _topology_background_data_uri(str(image_path), background.crop_box)


def _render_track_grid(
    track_sequences: dict[str, list[str]],
    highlighted_tracks: set[str],
):
    active_tracks = sorted(
        track for track, seq in track_sequences.items() if seq or track in highlighted_tracks
    )
    if not active_tracks:
        st.info("当前无车辆占用。")
        return

    st.markdown("**当前股道状态**")
    columns = st.columns(3)
    for index, track in enumerate(active_tracks):
        sequence = track_sequences.get(track, [])
        with columns[index % 3]:
            label = f"{track} *" if track in highlighted_tracks else track
            st.markdown(f"**{label}**")
            if sequence:
                st.code(
                    "\n".join(f"{position}. {vehicle_no}" for position, vehicle_no in enumerate(sequence, start=1)),
                    language=None,
                )
            else:
                st.caption("空")


def _render_track_map(track_map):
    ordered_tracks = list(track_map.active_path_tracks)
    ordered_tracks.extend(
        track_code
        for track_code, node in track_map.track_nodes.items()
        if track_code not in ordered_tracks and node.is_occupied
    )
    ordered_tracks.extend(
        track_code
        for track_code in track_map.changed_tracks
        if track_code not in ordered_tracks
    )
    if not ordered_tracks:
        ordered_tracks = list(track_map.track_nodes.keys())

    st.markdown(
        """
        <style>
        .track-map-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px;
          margin: 8px 0 14px 0;
        }
        .track-map-card {
          border-radius: 12px;
          padding: 10px 12px;
          border: 1px solid #d8d1c2;
          background: linear-gradient(180deg, #faf7f2 0%, #f0ebe2 100%);
        }
        .track-map-card.active {
          border-color: #1d6f6d;
          background: linear-gradient(180deg, #eefaf8 0%, #dcf0ec 100%);
        }
        .track-map-card.changed {
          box-shadow: inset 0 0 0 1px #d8772a;
        }
        .track-map-card.loco {
          background: linear-gradient(180deg, #fff5e8 0%, #fde7cf 100%);
        }
        .track-map-title {
          font-weight: 700;
          margin-bottom: 3px;
        }
        .track-map-meta {
          font-size: 12px;
          color: #5d5a54;
          margin-bottom: 4px;
        }
        .track-map-body {
          font-size: 13px;
          color: #23211d;
          word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption(_track_map_legend_markdown())

    cards: list[str] = []
    for track_code in ordered_tracks:
        node = track_map.track_nodes[track_code]
        classes = ["track-map-card"]
        if node.is_in_active_path:
            classes.append("active")
        if node.is_changed:
            classes.append("changed")
        if node.has_loco:
            classes.append("loco")
        status_parts: list[str] = []
        if node.has_loco:
            status_parts.append("机车")
        status_parts.append(f"占用 {len(node.vehicle_nos)}" if node.is_occupied else "空")
        body = " ".join(node.vehicle_nos) if node.vehicle_nos else "无车辆"
        cards.append(
            f"""
            <div class="{' '.join(classes)}">
              <div class="track-map-title">{track_code}</div>
              <div class="track-map-meta">{' / '.join(status_parts)}</div>
              <div class="track-map-body">{body}</div>
            </div>
            """
        )

    st.markdown(
        f"<div class='track-map-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def _workflow_progress_value(*, stage_index: int, stage_count: int) -> float:
    if stage_count <= 1:
        return 1.0
    return stage_index / (stage_count - 1)


def _build_workflow_stage_rows(workflow) -> list[dict]:
    rows: list[dict] = []
    for index, stage in enumerate(workflow.stages, start=1):
        view = stage.view
        rows.append(
            {
                "stageIndex": index,
                "stageName": stage.name,
                "hookCount": view.summary.hook_count if view else 0,
                "finalTracks": ", ".join(view.summary.final_tracks) if view else "",
                "isValid": view.summary.is_valid if view else False,
            }
        )
    return rows


def _build_workflow_transition_rows(workflow) -> list[dict]:
    rows: list[dict] = []
    previous_tracks: dict[str, str] = {}
    previous_weighed: set[str] = set()
    previous_spots: dict[str, str] = {}

    for index, stage in enumerate(workflow.stages, start=1):
        view = stage.view
        if view is None or not view.steps:
            rows.append(
                {
                    "stageIndex": index,
                    "stageName": stage.name,
                    "locoTransition": "无",
                    "movedVehicles": "无",
                    "newWeighedVehicles": "无",
                    "spotChanges": "无",
                }
            )
            continue
        first_step = view.steps[0]
        final_step = view.steps[-1]
        current_tracks = {
            vehicle_no: track_name
            for track_name, seq in final_step.track_sequences.items()
            for vehicle_no in seq
        }
        current_weighed = set(final_step.weighed_vehicle_nos)
        current_spots = dict(final_step.spot_assignments)

        moved_parts: list[str] = []
        for vehicle_no in sorted(current_tracks):
            previous_track = previous_tracks.get(vehicle_no)
            current_track = current_tracks[vehicle_no]
            if previous_track is None:
                previous_track = _find_vehicle_track(first_step.track_sequences, vehicle_no) or current_track
            if previous_track != current_track:
                moved_parts.append(f"{vehicle_no}({previous_track}->{current_track})")

        spot_parts: list[str] = []
        for vehicle_no in sorted(set(previous_spots) | set(current_spots)):
            previous_spot = previous_spots.get(vehicle_no, "无")
            current_spot = current_spots.get(vehicle_no, "无")
            if previous_spot != current_spot:
                spot_parts.append(f"{vehicle_no}({previous_spot}->{current_spot})")

        rows.append(
            {
                "stageIndex": index,
                "stageName": stage.name,
                "locoTransition": f"{first_step.loco_track_name} -> {final_step.loco_track_name}",
                "movedVehicles": "；".join(moved_parts) if moved_parts else "无",
                "newWeighedVehicles": (
                    " ".join(sorted(current_weighed - previous_weighed))
                    if current_weighed - previous_weighed
                    else "无"
                ),
                "spotChanges": "；".join(spot_parts) if spot_parts else "无",
            }
        )

        previous_tracks = current_tracks
        previous_weighed = current_weighed
        previous_spots = current_spots
    return rows


def _find_vehicle_track(track_sequences: dict[str, list[str]], vehicle_no: str) -> str | None:
    for track_name, seq in track_sequences.items():
        if vehicle_no in seq:
            return track_name
    return None


def _build_comparison_panel(comparison_summary: dict[str, object] | None) -> dict[str, list[dict] | list[str]]:
    if not comparison_summary:
        return {"metrics": [], "captions": []}

    metrics = [
        {"label": "外部钩数", "value": comparison_summary["externalHookCount"]},
        {
            "label": "求解器钩数",
            "value": (
                comparison_summary["solverHookCount"]
                if comparison_summary["solverHookCount"] is not None
                else "N/A"
            ),
        },
        {
            "label": "钩数差值",
            "value": (
                comparison_summary["hookCountDelta"]
                if comparison_summary["hookCountDelta"] is not None
                else "N/A"
            ),
        },
        {
            "label": "外部计划校验",
            "value": "PASS" if comparison_summary["externalIsValid"] else "FAIL",
        },
    ]
    captions: list[str] = []
    failed_hook_nos = comparison_summary.get("failedHookNos") or []
    if failed_hook_nos:
        captions.append("失败钩号: " + ", ".join(str(item) for item in failed_hook_nos))
    solver_error = comparison_summary.get("solverError")
    if solver_error:
        captions.append(f"求解器对比结果: {solver_error}")
    return {"metrics": metrics, "captions": captions}


def _track_map_legend_markdown() -> str:
    return "Active Path = 绿色, Changed Track = 橙色描边, Loco Track = 橙色底色"


if __name__ == "__main__":
    main()
