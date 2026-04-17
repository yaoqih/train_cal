from __future__ import annotations

import json
from pathlib import Path
import time
from html import escape

import streamlit as st

from fzed_shunting.demo.layout import (
    build_route_polyline,
    load_topology_layout,
    point_at_progress,
    route_to_svg_path,
)
from fzed_shunting.demo.view_model import (
    build_demo_view_model,
    build_demo_workflow_view_model,
    select_demo_payload,
)
from fzed_shunting.domain.master_data import load_master_data


MASTER_DIR = Path(__file__).resolve().parent / "data" / "master"
_TOPOLOGY_LAYOUT = None


def main():
    st.set_page_config(page_title="福州东调车 Demo", layout="wide")
    st.title("福州东调车 Demo")
    st.caption("按场景文件回放 block 级求解结果，并显示 verifier 结论。")

    scenario_path = st.text_input("Scenario JSON 路径", value="")
    plan_path = st.text_input("可选 Plan JSON 路径", value="")
    solver = st.selectbox("Solver", options=["exact", "weighted", "beam", "lns"], index=0)
    heuristic_weight = st.number_input("Heuristic Weight", min_value=1.0, value=1.0, step=0.5)
    beam_width = st.number_input("Beam Width", min_value=1, value=16, step=1)
    compare_solvers = st.checkbox("Compare Solvers", value=False)
    if not scenario_path:
        st.info("先通过 CLI 生成一个 scenario.json，或使用 `artifacts/typical_suite.json` 中的典型场景，再粘贴路径。")
        return

    path = Path(scenario_path)
    if not path.exists():
        st.error("文件不存在")
        return

    master = load_master_data(MASTER_DIR)
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if _is_workflow_payload(raw_payload):
        _render_workflow_demo(
            master=master,
            payload=raw_payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width if solver in {"beam", "lns"} else None,
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
    plan_payload = None
    if plan_path:
        candidate = Path(plan_path)
        if not candidate.exists():
            st.error("Plan 文件不存在")
            return
        plan_payload = json.loads(candidate.read_text(encoding="utf-8"))
    try:
        view = build_demo_view_model(
            master,
            payload,
            plan_payload=plan_payload,
            solver=solver,
            heuristic_weight=heuristic_weight,
            beam_width=beam_width if solver in {"beam", "lns"} else None,
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
                compare_view = build_demo_view_model(
                    master,
                    payload,
                    solver=solver_name,
                    heuristic_weight=heuristic_weight,
                    beam_width=beam_width if solver_name in {"beam", "lns"} else None,
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

    st.subheader("钩计划")
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

    st.subheader("逐钩回放")
    autoplay = st.checkbox("Auto Play", value=False)
    autoplay_interval_ms = st.number_input("Auto Play Interval (ms)", min_value=100, value=600, step=100)
    step_index = st.slider(
        "Step",
        min_value=0,
        max_value=len(view.steps) - 1,
        value=view.failed_hook_nos[0] if view.failed_hook_nos else 0,
    )
    step_container = st.container()
    if autoplay:
        for auto_index in range(len(view.steps)):
            with step_container:
                _render_step(view, auto_index)
            time.sleep(autoplay_interval_ms / 1000)
    else:
        with step_container:
            _render_step(view, step_index)


def _render_workflow_demo(master, payload, solver: str, heuristic_weight: float, beam_width: int | None):
    workflow_view = build_demo_workflow_view_model(
        master,
        payload,
        solver=solver,
        heuristic_weight=heuristic_weight,
        beam_width=beam_width,
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
    _render_step(view, step_index)


def _is_workflow_payload(payload: dict) -> bool:
    return isinstance(payload.get("workflowStages"), list)


def _render_step(view, step_index: int):
    step = view.steps[step_index]
    if step.hook is None:
        st.info("Step 0 为初始状态。")
    else:
        st.write(
            f"第 {step.hook.hook_no} 钩: {step.hook.source_track} -> {step.hook.target_track}"
        )
        st.caption(
            f"车辆: {' '.join(step.hook.vehicle_nos)} | 路径: {' -> '.join(step.hook.path_tracks)}"
        )
        if step.hook.remark:
            st.caption(step.hook.remark)
        if step.verifier_errors:
            st.error(step.verifier_errors)

    detail_cols = st.columns(2)
    detail_cols[0].write(f"当前机车位置: {step.loco_track_name}")
    detail_cols[1].write(
        f"本步变化线: {', '.join(step.changed_tracks) if step.changed_tracks else '无'}"
    )
    if step.weighed_vehicle_nos:
        st.caption(f"已称重车辆: {' '.join(step.weighed_vehicle_nos)}")
    if step.spot_assignments:
        st.markdown("**当前台位分配**")
        st.dataframe(
            [
                {"vehicleNo": vehicle_no, "spotCode": spot_code}
                for vehicle_no, spot_code in step.spot_assignments.items()
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**线路图回放**")
    transition_frame = None
    if step.transition_frames:
        frame_index = st.slider(
            "钩内动画帧",
            min_value=0,
            max_value=len(step.transition_frames) - 1,
            value=len(step.transition_frames) - 1,
            key=f"transition-frame-{step.step_index}",
        )
        transition_frame = step.transition_frames[frame_index]
    _render_topology_graph(step.topology_graph, step.track_map, hook=step.hook, transition_frame=transition_frame)
    _render_track_map(step.track_map)
    _render_track_grid(step.track_sequences, highlighted_tracks=set(step.changed_tracks))
    if view.final_spot_assignments:
        st.subheader("最终台位分配")
        st.dataframe(
            [
                {"vehicleNo": vehicle_no, "spotCode": spot_code}
                for vehicle_no, spot_code in view.final_spot_assignments.items()
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_topology_graph(topology_graph, track_map, hook=None, transition_frame=None):
    st.markdown(
        _build_topology_svg(
            topology_graph,
            track_map,
            hook=hook,
            transition_frame=transition_frame,
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


def _build_topology_svg(topology_graph, track_map, hook=None, transition_frame=None, animate: bool = False) -> str:
    layout = _get_topology_layout()
    active_tracks = set(track_map.active_path_tracks)
    changed_tracks = set(track_map.changed_tracks)
    route = build_route_polyline(layout, hook.path_tracks if hook is not None else [])
    motion_path = route_to_svg_path(route)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
            f'{layout.canvas_width:.0f} {layout.canvas_height:.0f}" '
            'class="topology-svg">'
        ),
        "<style>",
        ".track-path{fill:none;stroke:#c5c0b5;stroke-width:10;stroke-linecap:round;stroke-linejoin:round;}",
        ".track-path.active-path{stroke:#1d6f6d;stroke-width:12;}",
        ".track-path.changed-track{stroke:#d8772a;stroke-width:11;}",
        ".track-label{font-family:PingFang SC, sans-serif;font-size:24px;fill:#2e2a24;text-anchor:middle;}",
        ".track-badge{fill:#faf7f2;stroke:#d8d1c2;stroke-width:1.5;rx:12;}",
        ".track-badge.active{fill:#dcf0ec;stroke:#1d6f6d;}",
        ".track-badge.changed{fill:#fff0e0;stroke:#d8772a;}",
        ".track-badge.loco{fill:#fde7cf;stroke:#d8772a;}",
        ".moving-block-marker{fill:#1d6f6d;stroke:#ffffff;stroke-width:3;}",
        ".moving-block-label{font-family:PingFang SC, sans-serif;font-size:22px;fill:#ffffff;text-anchor:middle;dominant-baseline:middle;}",
        ".loco-marker{fill:#d8772a;stroke:#ffffff;stroke-width:3;}",
        ".route-motion-path{fill:none;stroke:none;}",
        "</style>",
    ]

    rendered_track_codes = set(topology_graph.nodes) | set(layout.track_geometries)
    for track_code in sorted(rendered_track_codes):
        geometry = layout.track_geometries.get(track_code)
        node = track_map.track_nodes.get(track_code)
        if geometry is None or node is None:
            continue
        path_class = "track-path"
        if track_code in active_tracks:
            path_class += " active-path"
        elif track_code in changed_tracks:
            path_class += " changed-track"
        path_d = _track_polyline_to_svg(geometry.points)
        parts.append(f'<path class="{path_class}" d="{path_d}" />')

    if motion_path:
        parts.append(f'<path id="route-motion-path" class="route-motion-path" d="{motion_path}" />')

    for track_code in sorted(rendered_track_codes):
        geometry = layout.track_geometries.get(track_code)
        node = track_map.track_nodes.get(track_code)
        if geometry is None or node is None:
            continue
        badge_class = "track-badge"
        if node.has_loco:
            badge_class += " loco"
        elif track_code in active_tracks:
            badge_class += " active"
        elif track_code in changed_tracks:
            badge_class += " changed"
        badge_x = geometry.center.x - 36
        badge_y = geometry.center.y - 18
        parts.append(
            f'<rect class="{badge_class}" x="{badge_x:.1f}" y="{badge_y:.1f}" width="72" height="36" rx="12" />'
        )
        parts.append(
            f'<text class="track-label" x="{geometry.center.x:.1f}" y="{geometry.center.y + 8:.1f}">{escape(track_code)}</text>'
        )

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
        elif transition_frame is not None and motion_path:
            point = point_at_progress(route, transition_frame.progress)
            parts.append(
                f'<circle class="moving-block-marker" cx="{point.x:.1f}" cy="{point.y:.1f}" r="18" />'
            )
            parts.append(
                f'<text class="moving-block-label" x="{point.x:.1f}" y="{point.y + 1:.1f}">{len(hook.vehicle_nos)}</text>'
            )

    loco_track = next(
        (track_code for track_code, node in track_map.track_nodes.items() if node.has_loco),
        None,
    )
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


def _get_topology_layout():
    global _TOPOLOGY_LAYOUT
    if _TOPOLOGY_LAYOUT is None:
        _TOPOLOGY_LAYOUT = load_topology_layout(MASTER_DIR, load_master_data(MASTER_DIR))
    return _TOPOLOGY_LAYOUT


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
