from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict, deque
from dataclasses import dataclass
from html import escape
from heapq import heappop, heappush
import json
from math import log1p
from pathlib import Path
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field

from fzed_shunting.demo.layout import LayoutPoint
from fzed_shunting.domain.master_data import MasterData, load_master_data


class SegmentedRouteSegment(BaseModel):
    track_code: str
    physical_distance_m: float


class SegmentedPhysicalRoute(BaseModel):
    branch_code: str
    display_name: str
    aggregate_physical_distance_m: float
    status: str
    left_node: str | None = None
    right_node: str | None = None
    segments: tuple[SegmentedRouteSegment, ...] = Field(default_factory=tuple)


class DirectedTrackSpec(BaseModel):
    track_code: str
    start_node: str
    end_node: str
    display_span_px: float


@dataclass
class TrackRoutePlan:
    track_code: str
    endpoint_nodes: tuple[str, str]
    start_point: LayoutPoint
    end_point: LayoutPoint
    preferred_lane_y: float
    actual_lane_y: float
    x_min: float
    x_max: float
    primary_distance: int
    is_primary_track: bool
    locked_to_fixed_lane: bool
    start_join_offset_px: float = 0.0
    end_join_offset_px: float = 0.0


class ContinuousTrackGeometry(BaseModel):
    track_code: str
    endpoint_nodes: tuple[str, str]
    endpoints: tuple[LayoutPoint, LayoutPoint]
    label_anchor: LayoutPoint
    path_d: str


class ContinuousNetworkLayout(BaseModel):
    canvas_width: float
    canvas_height: float
    node_positions: dict[str, LayoutPoint] = Field(default_factory=dict)
    track_geometries: dict[str, ContinuousTrackGeometry] = Field(default_factory=dict)


PRIMARY_CORRIDOR_TRACK_CODES = (
    "联6",
    "渡2",
    "机北1",
    "机北2",
    "渡5",
    "渡6",
    "渡7",
    "预修",
    "渡9",
    "渡10",
    "联7",
    "渡12",
    "渡13",
)

ROUTE_LANE_SPACING = 28.0
DIRECT_CONNECTOR_TRACK_CODES = {
    "机北2",
    "渡4",
    "渡5",
    "渡6",
    "渡7",
    "洗油北",
    "机南",
}
ROUTE_GRID_BASELINE_NODE_NAMES = (
    "L13",
    "L14",
    "L15",
    "L16",
    "L17",
    "L18",
    "Z3",
)
FIXED_NODE_LANE_INDICES = {
    "L13": 0,
    "L14": 0,
    "L15": 0,
    "L16": 0,
    "L17": 0,
    "L18": 0,
    "Z3": 0,
    "L5": -2,
    "Z2": -2,
    "L6": -4,
    "Z1": -4,
    "机棚北口": -4,
    "L8": -4,
    "L7": -6,
    "调棚北口": -6,
    "调棚尽头": -6,
    "机库尽头": -8,
}
TRACK_TEXT_LAYOUTS: dict[str, str] = {
    "存5北": "above",
    "存5南": "above",
    "存4北": "above",
    "存4南": "above",
    "存3": "above",
    "存2": "above",
    "存1": "above",
    "预修": "above",
    "机北3": "above",
    "机棚": "above",
    "渡7": "above",
    "渡6": "above",
    "渡5": "above",
    "机北2": "above",
    "机北1": "above",
    "调北": "below",
    "调棚": "below",
    "机库": "below",
    "渡4": "below",
    "洗油北": "below",
    "机南": "above",
    "油": "below",
    "洗北": "below",
    "洗南": "below",
    "渡13": "above",
    "修1库外": "below",
    "修2库外": "below",
    "修3库外": "below",
    "修4库外": "below",
}
TRACK_TEXT_ANCHOR_BIASES = {
    "存1": 40.0,
    "渡7": -10.0,
    "渡6": 18.0,
    "渡4": -8.0,
    "渡5": 8.0,
    "机北2": 28.0,
    "机棚": -10.0,
    "机北3": -20.0,
    "调棚": -8.0,
    "调北": 10.0,
    "机库": -72.0,
    "油": 8.0,
    "洗北": 54.0,
    "渡11": -40.0,
    "修1库外": -72.0,
    "修2库外": -40.0,
    "轮": 72.0,
}
TRACK_TEXT_BASELINE_OVERRIDE_OFFSETS: dict[str, tuple[float, float]] = {
    "机北2": (-58.0, -36.0),
    "渡6": (-58.0, -36.0),
    "渡5": (62.0, 84.0),
    "洗油北": (14.0, 36.0),
    "油": (4.0, 26.0),
}
NODE_X_ALIGNMENT_GROUPS = (
    ("机棚北口", "调棚北口", "机库尽头"),
)


def load_segmented_physical_routes(
    base_dir: Path,
    master: MasterData,
) -> dict[str, SegmentedPhysicalRoute]:
    raw = json.loads((base_dir / "segmented_physical_routes.json").read_text(encoding="utf-8"))
    routes: dict[str, SegmentedPhysicalRoute] = {}
    for item in raw:
        route = SegmentedPhysicalRoute.model_validate(item)
        if not route.segments:
            raise ValueError(f"branch {route.branch_code} has no segments")
        for segment in route.segments:
            if segment.track_code not in master.tracks:
                raise ValueError(f"unknown track code {segment.track_code} in {route.branch_code}")
        total = sum(segment.physical_distance_m for segment in route.segments)
        if abs(total - route.aggregate_physical_distance_m) > 0.11:
            raise ValueError(
                f"branch {route.branch_code} segment total {total} does not match "
                f"aggregate {route.aggregate_physical_distance_m}"
            )
        routes[route.branch_code] = route
    return routes


def render_segmented_routes_svg(base_dir: Path, master: MasterData | None = None) -> str:
    resolved_master = master or load_master_data(base_dir)
    routes = load_segmented_physical_routes(base_dir, resolved_master)
    network_layout = build_continuous_network_layout(base_dir, resolved_master)
    canvas_width = network_layout.canvas_width
    canvas_height = network_layout.canvas_height

    unique_track_codes: list[str] = []
    seen_track_codes: set[str] = set()
    for route in routes.values():
        for segment in route.segments:
            if segment.track_code in seen_track_codes:
                continue
            if segment.track_code not in network_layout.track_geometries:
                continue
            seen_track_codes.add(segment.track_code)
            unique_track_codes.append(segment.track_code)

    legend_x = canvas_width + 32.0
    view_width = canvas_width + 460.0
    view_height = max(canvas_height, 64.0 + len(routes) * 30.0)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '
            f'{view_width:.1f} {view_height:.1f}" class="segmented-routes-svg">'
        ),
        "<style>",
        ".svg-bg{fill:#fbfaf5;}",
        ".track-panel{fill:#f3f0e6;stroke:#d5cebf;stroke-width:1;}",
        ".segmented-track{fill:none;stroke:#2f4f5f;stroke-width:8;stroke-linecap:round;stroke-linejoin:round;}",
        (
            ".track-label{fill:#1f2933;font-size:24px;font-family:'Hiragino Sans GB','PingFang SC',sans-serif;"
            "font-weight:600;paint-order:stroke;stroke:#fbfaf5;stroke-width:7;stroke-linejoin:round;}"
        ),
        (
            ".track-distance{fill:#52606d;font-size:16px;font-family:'Hiragino Sans GB','PingFang SC',sans-serif;"
            "paint-order:stroke;stroke:#fbfaf5;stroke-width:5;stroke-linejoin:round;}"
        ),
        ".legend-title{fill:#102a43;font-size:26px;font-family:'Hiragino Sans GB','PingFang SC',sans-serif;font-weight:700;}",
        ".legend-text{fill:#243b53;font-size:18px;font-family:'Hiragino Sans GB','PingFang SC',sans-serif;}",
        "</style>",
        f'<rect class="svg-bg" x="0" y="0" width="{view_width:.1f}" height="{view_height:.1f}" />',
        (
            f'<rect class="track-panel" x="{canvas_width + 12.0:.1f}" y="12.0" '
            f'width="430.0" height="{view_height - 24.0:.1f}" rx="18" ry="18" />'
        ),
        f'<text class="legend-title" x="{legend_x:.1f}" y="44.0">聚合物理距离实验图</text>',
    ]

    for track_code in unique_track_codes:
        geometry = network_layout.track_geometries[track_code]
        track = resolved_master.tracks[track_code]
        label_baseline_y, distance_baseline_y = _track_text_baseline_ys(
            track_code=track_code,
            anchor=geometry.label_anchor,
        )
        parts.append(f'<path class="segmented-track" d="{geometry.path_d}" />')
        parts.append(
            f'<text class="track-label" x="{geometry.label_anchor.x:.1f}" '
            f'y="{label_baseline_y:.1f}" text-anchor="middle">{escape(track_code)}</text>'
        )
        parts.append(
            f'<text class="track-distance" x="{geometry.label_anchor.x:.1f}" '
            f'y="{distance_baseline_y:.1f}" text-anchor="middle">{track.effective_length_m:.1f}m</text>'
        )

    for index, route in enumerate(routes.values()):
        segment_codes = " / ".join(segment.track_code for segment in route.segments)
        y = 76.0 + index * 30.0
        parts.append(
            f'<text class="legend-text" x="{legend_x:.1f}" y="{y:.1f}">'
            f'{escape(route.display_name)} | {escape(segment_codes)} | '
            f'{route.aggregate_physical_distance_m:.1f}m</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def build_continuous_network_layout(
    base_dir: Path,
    master: MasterData | None = None,
) -> ContinuousNetworkLayout:
    resolved_master = master or load_master_data(base_dir)
    routes = load_segmented_physical_routes(base_dir, resolved_master)
    track_specs = _build_directed_track_specs(routes, resolved_master)
    raw_x = _solve_node_x_coordinates(track_specs)
    raw_y = _solve_node_y_seeds(track_specs)
    undirected_adjacency = _build_node_adjacency(track_specs.values())
    primary_corridor = _build_primary_corridor(track_specs)
    primary_node_names = set(primary_corridor)
    distance_to_primary, _ = _measure_primary_proximity(
        undirected_adjacency=undirected_adjacency,
        primary_corridor=primary_corridor,
    )
    raw_reference_y = _build_node_reference_y_positions(
        raw_y=raw_y,
        track_specs=track_specs,
        primary_corridor=primary_corridor,
    )

    ordered_nodes = sorted(raw_x, key=lambda node_name: (raw_x[node_name], node_name))
    left_margin = 180.0
    right_margin = 170.0
    max_raw_x = max(raw_x.values(), default=0.0)
    canvas_width = max(4600.0, max_raw_x + left_margin + right_margin + 120.0)
    top_margin = 160.0
    bottom_margin = 160.0
    min_raw_y = min(raw_reference_y.values(), default=0.0)
    max_raw_y = max(raw_reference_y.values(), default=0.0)
    canvas_height = max(860.0, (max_raw_y - min_raw_y) + top_margin + bottom_margin)
    vertical_shift = top_margin - min_raw_y

    positions: dict[str, LayoutPoint] = {}
    for node_name in ordered_nodes:
        positions[node_name] = LayoutPoint(
            x=canvas_width - right_margin - raw_x[node_name],
            y=raw_reference_y[node_name] + vertical_shift,
        )
    _align_node_x_groups(
        positions=positions,
        track_specs=track_specs,
    )

    primary_y = _resolve_route_grid_primary_y(
        positions=positions,
        primary_corridor=primary_corridor,
        fallback_y=canvas_height / 2.0,
    )
    route_plans = _build_track_route_plans(
        track_specs=track_specs,
        positions=positions,
        primary_node_names=primary_node_names,
        distance_to_primary=distance_to_primary,
    )
    _snap_track_route_lanes_to_grid(
        route_plans,
        primary_y=primary_y,
        lane_spacing=ROUTE_LANE_SPACING,
    )
    _separate_track_route_lanes(
        route_plans,
        primary_y=primary_y,
        lane_spacing=ROUTE_LANE_SPACING,
    )
    _assign_route_plan_join_offsets(route_plans)

    track_geometries: dict[str, ContinuousTrackGeometry] = {}
    label_spans: dict[str, tuple[float, float]] = {}
    for spec in track_specs.values():
        polyline = _build_track_polyline(
            spec=spec,
            positions=positions,
            primary_node_names=primary_node_names,
            distance_to_primary=distance_to_primary,
            route_plan=route_plans.get(spec.track_code),
        )
        label_anchor = _polyline_midpoint(polyline)
        label_spans[spec.track_code] = _dominant_horizontal_span(polyline)
        track_geometries[spec.track_code] = ContinuousTrackGeometry(
            track_code=spec.track_code,
            endpoint_nodes=(spec.start_node, spec.end_node),
            endpoints=(positions[spec.start_node], positions[spec.end_node]),
            label_anchor=label_anchor,
            path_d=_polyline_path(polyline),
        )

    _apply_track_text_anchor_biases(track_geometries, label_spans)
    _separate_track_label_anchors(track_geometries, label_spans)

    return ContinuousNetworkLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        node_positions=positions,
        track_geometries=track_geometries,
    )


def export_segmented_route_artifacts(
    base_dir: Path,
    output_dir: Path,
    master: MasterData | None = None,
) -> dict[str, Path]:
    resolved_master = master or load_master_data(base_dir)
    routes = load_segmented_physical_routes(base_dir, resolved_master)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "segmented_physical_routes.json"
    svg_path = output_dir / "segmented_physical_routes.svg"

    json_path.write_text(
        json.dumps(_serialize_routes(routes, resolved_master), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    svg_path.write_text(render_segmented_routes_svg(base_dir, resolved_master), encoding="utf-8")

    return {
        "json_path": json_path,
        "svg_path": svg_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Export segmented physical route experiment artifacts.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "master"),
        help="Master data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[3] / "artifacts" / "segmented_routes_experiment"),
        help="Output directory for JSON and SVG artifacts.",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    export_segmented_route_artifacts(base_dir, output_dir)
    return 0


def _serialize_routes(
    routes: dict[str, SegmentedPhysicalRoute],
    master: MasterData,
) -> list[dict]:
    serialized: list[dict] = []
    for route in routes.values():
        serialized.append(
            {
                "branch_code": route.branch_code,
                "display_name": route.display_name,
                "aggregate_physical_distance_m": route.aggregate_physical_distance_m,
                "status": route.status,
                "left_node": route.left_node,
                "right_node": route.right_node,
                "segments": [
                    {
                        "track_code": segment.track_code,
                        "track_name": master.tracks[segment.track_code].name,
                        "effective_length_m": master.tracks[segment.track_code].effective_length_m,
                        "physical_distance_m": segment.physical_distance_m,
                    }
                    for segment in route.segments
                ],
            }
        )
    return serialized


def _collect_track_codes(routes: dict[str, SegmentedPhysicalRoute]) -> list[str]:
    track_codes: list[str] = []
    seen: set[str] = set()
    for route in routes.values():
        for segment in route.segments:
            if segment.track_code in seen:
                continue
            seen.add(segment.track_code)
            track_codes.append(segment.track_code)
    return track_codes


def _collect_track_edges(
    master: MasterData,
    track_codes: list[str],
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for track_code in track_codes:
        endpoint_nodes = master.tracks[track_code].endpoint_nodes
        if len(endpoint_nodes) != 2:
            continue
        edges.append((track_code, endpoint_nodes[0], endpoint_nodes[1]))
    return edges


def _build_node_adjacency(
    edge_specs: Iterable[DirectedTrackSpec],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for spec in edge_specs:
        adjacency.setdefault(spec.start_node, set()).add(spec.end_node)
        adjacency.setdefault(spec.end_node, set()).add(spec.start_node)
    return adjacency


def _build_primary_corridor(
    track_specs: dict[str, DirectedTrackSpec],
) -> tuple[str, ...]:
    explicit_corridor = _build_explicit_primary_corridor(track_specs)
    if explicit_corridor:
        return explicit_corridor

    directed_adjacency: dict[str, list[DirectedTrackSpec]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    node_names: set[str] = set()
    for spec in track_specs.values():
        directed_adjacency[spec.start_node].append(spec)
        indegree[spec.end_node] += 1
        indegree.setdefault(spec.start_node, 0)
        node_names.add(spec.start_node)
        node_names.add(spec.end_node)

    queue = deque(sorted(node_name for node_name in node_names if indegree[node_name] == 0))
    topological_order: list[str] = []
    while queue:
        node_name = queue.popleft()
        topological_order.append(node_name)
        for spec in sorted(directed_adjacency[node_name], key=lambda item: (item.end_node, item.track_code)):
            indegree[spec.end_node] -= 1
            if indegree[spec.end_node] == 0:
                queue.append(spec.end_node)

    anchor_node = "5号门" if "5号门" in node_names else topological_order[0]
    best_score = {node_name: float("-inf") for node_name in node_names}
    parent_node: dict[str, str] = {}
    best_score[anchor_node] = 0.0
    for node_name in topological_order:
        if best_score[node_name] == float("-inf"):
            continue
        for spec in directed_adjacency[node_name]:
            candidate = best_score[node_name] + spec.display_span_px
            if candidate <= best_score[spec.end_node]:
                continue
            best_score[spec.end_node] = candidate
            parent_node[spec.end_node] = node_name

    reachable_nodes = [node_name for node_name in node_names if best_score[node_name] > float("-inf")]
    sink_node = max(reachable_nodes, key=lambda node_name: (best_score[node_name], node_name))
    corridor_nodes = [sink_node]
    while corridor_nodes[-1] != anchor_node:
        corridor_nodes.append(parent_node[corridor_nodes[-1]])
    corridor_nodes.reverse()
    return tuple(corridor_nodes)


def _build_explicit_primary_corridor(
    track_specs: dict[str, DirectedTrackSpec],
) -> tuple[str, ...] | None:
    corridor_nodes: list[str] = []
    current_node: str | None = None
    for track_code in PRIMARY_CORRIDOR_TRACK_CODES:
        spec = track_specs.get(track_code)
        if spec is None:
            return None
        if current_node is None:
            corridor_nodes.extend([spec.start_node, spec.end_node])
            current_node = spec.end_node
            continue
        if spec.start_node == current_node:
            corridor_nodes.append(spec.end_node)
            current_node = spec.end_node
            continue
        if spec.end_node == current_node:
            corridor_nodes.append(spec.start_node)
            current_node = spec.start_node
            continue
        return None
    return tuple(corridor_nodes)


def _build_node_reference_y_positions(
    raw_y: dict[str, float],
    track_specs: dict[str, DirectedTrackSpec],
    primary_corridor: tuple[str, ...],
) -> dict[str, float]:
    track_bands = {track_code: _track_alignment_band(track_code) for track_code in track_specs}
    incident_bands: dict[str, list[float]] = defaultdict(list)
    for spec in track_specs.values():
        band = track_bands[spec.track_code]
        incident_bands[spec.start_node].append(band)
        incident_bands[spec.end_node].append(band)

    max_abs_seed = max((abs(value) for value in raw_y.values()), default=1.0) or 1.0
    band_spacing = 95.0
    fine_spacing = 18.0
    primary_node_names = set(primary_corridor)
    reference_y: dict[str, float] = {}
    for node_name, bands in incident_bands.items():
        if node_name in primary_node_names:
            reference_y[node_name] = (raw_y.get(node_name, 0.0) / max_abs_seed) * 14.0
            continue
        reference_y[node_name] = (
            -(sum(bands) / len(bands)) * band_spacing
            + (raw_y.get(node_name, 0.0) / max_abs_seed) * fine_spacing
        )
    baseline_y = _resolve_reference_lane_baseline(reference_y, primary_corridor)
    for node_name, lane_index in FIXED_NODE_LANE_INDICES.items():
        if node_name not in reference_y:
            continue
        reference_y[node_name] = baseline_y - lane_index * ROUTE_LANE_SPACING
    return reference_y


def _resolve_reference_lane_baseline(
    reference_y: dict[str, float],
    primary_corridor: tuple[str, ...],
) -> float:
    baseline_nodes = [node_name for node_name in ROUTE_GRID_BASELINE_NODE_NAMES if node_name in reference_y]
    if not baseline_nodes:
        baseline_nodes = [node_name for node_name in primary_corridor if node_name in reference_y]
    if not baseline_nodes:
        return 0.0
    return sum(reference_y[node_name] for node_name in baseline_nodes) / len(baseline_nodes)


def _resolve_route_grid_primary_y(
    positions: dict[str, LayoutPoint],
    primary_corridor: tuple[str, ...],
    fallback_y: float,
) -> float:
    baseline_nodes = [node_name for node_name in ROUTE_GRID_BASELINE_NODE_NAMES if node_name in positions]
    if not baseline_nodes:
        baseline_nodes = [node_name for node_name in primary_corridor if node_name in positions]
    if not baseline_nodes:
        return fallback_y
    return sum(positions[node_name].y for node_name in baseline_nodes) / len(baseline_nodes)


def _align_node_x_groups(
    *,
    positions: dict[str, LayoutPoint],
    track_specs: dict[str, DirectedTrackSpec],
) -> None:
    node_adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for spec in track_specs.values():
        node_adjacency[spec.start_node].append((spec.end_node, spec.display_span_px))
        node_adjacency[spec.end_node].append((spec.start_node, spec.display_span_px))

    for group in NODE_X_ALIGNMENT_GROUPS:
        existing_nodes = [node_name for node_name in group if node_name in positions]
        if len(existing_nodes) < 2:
            continue
        target_x_candidates = [
            _aligned_node_target_x(
                node_name=node_name,
                positions=positions,
                node_adjacency=node_adjacency,
            )
            for node_name in existing_nodes
        ]
        target_x = sum(target_x_candidates) / len(target_x_candidates)
        for node_name in existing_nodes:
            positions[node_name] = LayoutPoint(x=target_x, y=positions[node_name].y)


def _aligned_node_target_x(
    *,
    node_name: str,
    positions: dict[str, LayoutPoint],
    node_adjacency: dict[str, list[tuple[str, float]]],
) -> float:
    neighbors = [
        (neighbor_name, span_px)
        for neighbor_name, span_px in node_adjacency.get(node_name, [])
        if neighbor_name in positions
    ]
    if not neighbors:
        return positions[node_name].x
    if len(neighbors) == 1:
        neighbor_name, span_px = neighbors[0]
        neighbor_x = positions[neighbor_name].x
        direction = -1.0 if positions[node_name].x < neighbor_x else 1.0
        return neighbor_x + direction * span_px

    ordered_neighbors = sorted(neighbors, key=lambda item: positions[item[0]].x)
    left_neighbor, left_span = ordered_neighbors[0]
    right_neighbor, right_span = ordered_neighbors[-1]
    left_x = positions[left_neighbor].x
    right_x = positions[right_neighbor].x
    total_span = left_span + right_span
    if total_span < 1e-6:
        return positions[node_name].x
    return (left_x * right_span + right_x * left_span) / total_span


def _track_alignment_band(track_code: str) -> float:
    explicit_bands = {
        "存5北": 3.0,
        "存5南": 3.0,
        "修4库外": 3.0,
        "修4": 3.0,
        "存4北": 2.0,
        "存4南": 2.0,
        "存3": 2.0,
        "存2": 1.8,
        "修3库外": 2.0,
        "修3": 2.0,
        "机北3": 1.2,
        "机棚": 1.0,
        "预修": 0.8,
        "修2库外": 1.0,
        "修2": 1.0,
        "渡4": -0.6,
        "调北": -1.0,
        "调棚": -1.2,
        "存1": -0.7,
        "修1库外": -0.2,
        "修1": -0.2,
        "轮": -1.0,
        "抛": -0.8,
        "机库": -1.5,
        "洗油北": -1.5,
        "油": -2.1,
        "洗北": -2.0,
        "洗南": -2.6,
    }
    return explicit_bands.get(track_code, 0.0)


def _measure_primary_proximity(
    undirected_adjacency: dict[str, set[str]],
    primary_corridor: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    primary_node_names = set(primary_corridor)
    distance_to_primary: dict[str, int] = {}
    nearest_primary_index: dict[str, int] = {}
    for node_name in undirected_adjacency:
        if node_name in primary_node_names:
            distance_to_primary[node_name] = 0
            nearest_primary_index[node_name] = primary_corridor.index(node_name)
            continue
        queue = deque([(node_name, 0)])
        visited = {node_name}
        while queue:
            current_node, distance = queue.popleft()
            if current_node in primary_node_names:
                distance_to_primary[node_name] = distance
                nearest_primary_index[node_name] = primary_corridor.index(current_node)
                break
            for neighbor in sorted(undirected_adjacency[current_node]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return distance_to_primary, nearest_primary_index


def _build_track_route_plans(
    track_specs: dict[str, DirectedTrackSpec],
    positions: dict[str, LayoutPoint],
    primary_node_names: set[str],
    distance_to_primary: dict[str, int],
) -> dict[str, TrackRoutePlan]:
    route_plans: dict[str, TrackRoutePlan] = {}
    for spec in track_specs.values():
        start_point = positions[spec.start_node]
        end_point = positions[spec.end_node]
        if abs(start_point.x - end_point.x) < 1e-6:
            continue

        if abs(start_point.y - end_point.y) < 1e-6:
            preferred_lane_y = start_point.y
        else:
            start_on_primary = spec.start_node in primary_node_names
            end_on_primary = spec.end_node in primary_node_names
            if start_on_primary and not end_on_primary:
                preferred_lane_y = end_point.y
            elif end_on_primary and not start_on_primary:
                preferred_lane_y = start_point.y
            else:
                start_distance = distance_to_primary[spec.start_node]
                end_distance = distance_to_primary[spec.end_node]
                preferred_lane_y = end_point.y if start_distance < end_distance else start_point.y

        route_plans[spec.track_code] = TrackRoutePlan(
            track_code=spec.track_code,
            endpoint_nodes=(spec.start_node, spec.end_node),
            start_point=start_point,
            end_point=end_point,
            preferred_lane_y=preferred_lane_y,
            actual_lane_y=preferred_lane_y,
            x_min=min(start_point.x, end_point.x),
            x_max=max(start_point.x, end_point.x),
            primary_distance=max(
                distance_to_primary.get(spec.start_node, 0),
                distance_to_primary.get(spec.end_node, 0),
            ),
            is_primary_track=spec.track_code in PRIMARY_CORRIDOR_TRACK_CODES,
            locked_to_fixed_lane=False,
        )
    return route_plans


def _snap_track_route_lanes_to_grid(
    route_plans: dict[str, TrackRoutePlan],
    *,
    primary_y: float,
    lane_spacing: float,
) -> None:
    for route_plan in route_plans.values():
        explicit_lane_index = _fixed_track_lane_index(route_plan.track_code)
        if route_plan.is_primary_track and explicit_lane_index is None:
            continue
        if explicit_lane_index is None:
            explicit_lane_index = round((primary_y - route_plan.preferred_lane_y) / lane_spacing)
        else:
            route_plan.locked_to_fixed_lane = True
        snapped_lane_y = primary_y - explicit_lane_index * lane_spacing
        route_plan.preferred_lane_y = snapped_lane_y
        route_plan.actual_lane_y = snapped_lane_y


def _separate_track_route_lanes(
    route_plans: dict[str, TrackRoutePlan],
    *,
    primary_y: float,
    lane_spacing: float,
    clearance: float = 20.0,
    min_shared_span: float = 60.0,
    max_iterations: int = 24,
) -> None:
    for _ in range(max_iterations):
        changed = False
        ordered_plans = sorted(route_plans.values(), key=lambda item: (item.actual_lane_y, item.track_code))
        for index, first_plan in enumerate(ordered_plans):
            for second_plan in ordered_plans[index + 1 :]:
                shared_span = min(first_plan.x_max, second_plan.x_max) - max(
                    first_plan.x_min,
                    second_plan.x_min,
                )
                if shared_span < min_shared_span:
                    continue
                if set(first_plan.endpoint_nodes) & set(second_plan.endpoint_nodes):
                    continue
                gap = second_plan.actual_lane_y - first_plan.actual_lane_y
                if gap >= clearance:
                    continue

                first_target = second_plan.actual_lane_y - lane_spacing
                second_target = first_plan.actual_lane_y + lane_spacing
                if first_plan.locked_to_fixed_lane and second_plan.locked_to_fixed_lane:
                    continue
                if first_plan.locked_to_fixed_lane:
                    if abs(second_target - second_plan.actual_lane_y) < 1e-6:
                        continue
                    second_plan.actual_lane_y = second_target
                    changed = True
                    break
                if second_plan.locked_to_fixed_lane:
                    if abs(first_target - first_plan.actual_lane_y) < 1e-6:
                        continue
                    first_plan.actual_lane_y = first_target
                    changed = True
                    break
                first_penalty = _route_lane_move_penalty(first_plan, first_target, primary_y)
                second_penalty = _route_lane_move_penalty(second_plan, second_target, primary_y)
                if first_penalty <= second_penalty:
                    if abs(first_target - first_plan.actual_lane_y) < 1e-6:
                        continue
                    first_plan.actual_lane_y = first_target
                else:
                    if abs(second_target - second_plan.actual_lane_y) < 1e-6:
                        continue
                    second_plan.actual_lane_y = second_target
                changed = True
                break
            if changed:
                break
        if not changed:
            return


def _assign_route_plan_join_offsets(
    route_plans: dict[str, TrackRoutePlan],
    *,
    join_spacing: float = 14.0,
) -> None:
    groups: dict[tuple[str, int], list[tuple[str, TrackRoutePlan, float]]] = defaultdict(list)
    for route_plan in route_plans.values():
        route_plan.start_join_offset_px = 0.0
        route_plan.end_join_offset_px = 0.0

        if abs(route_plan.start_point.y - route_plan.actual_lane_y) >= 1e-6:
            start_side = 1 if route_plan.end_point.x > route_plan.start_point.x else -1
            groups[(route_plan.endpoint_nodes[0], start_side)].append(
                (
                    "start",
                    route_plan,
                    abs(route_plan.start_point.y - route_plan.actual_lane_y),
                )
            )
        if abs(route_plan.end_point.y - route_plan.actual_lane_y) >= 1e-6:
            end_side = 1 if route_plan.start_point.x > route_plan.end_point.x else -1
            groups[(route_plan.endpoint_nodes[1], end_side)].append(
                (
                    "end",
                    route_plan,
                    abs(route_plan.end_point.y - route_plan.actual_lane_y),
                )
            )

    for connectors in groups.values():
        for index, (connector_kind, route_plan, _) in enumerate(
            sorted(connectors, key=lambda item: (item[2], item[1].track_code))
        ):
            extra_offset = index * join_spacing
            if connector_kind == "start":
                route_plan.start_join_offset_px = extra_offset
            else:
                route_plan.end_join_offset_px = extra_offset


def _route_lane_move_penalty(
    route_plan: TrackRoutePlan,
    target_y: float,
    primary_y: float,
) -> float:
    current_outward_distance = abs(route_plan.actual_lane_y - primary_y)
    target_outward_distance = abs(target_y - primary_y)
    penalty = abs(target_y - route_plan.preferred_lane_y)
    if target_outward_distance + 1e-6 < current_outward_distance:
        penalty += 500.0
    if route_plan.is_primary_track:
        penalty += 10_000.0
    penalty += max(0, 2 - route_plan.primary_distance) * 40.0
    penalty += max(0.0, 0.6 - abs(_track_alignment_band(route_plan.track_code))) * 20.0
    return penalty


def _fixed_track_lane_index(track_code: str) -> int | None:
    explicit_lane_indices = {
        "存5北": 8,
        "存5南": 8,
        "存4北": 7,
        "存4南": 7,
        "存3": 6,
        "存2": 5,
        "预修": 0,
        "存1": -2,
        "机北3": -4,
        "机棚": -4,
        "调北": -6,
        "调棚": -6,
        "机库": -8,
        "油": -9,
        "洗北": -10,
        "洗南": -11,
    }
    return explicit_lane_indices.get(track_code)


def _should_use_direct_connector(
    track_code: str,
    start_point: LayoutPoint,
    end_point: LayoutPoint,
) -> bool:
    is_turnout_link = track_code in DIRECT_CONNECTOR_TRACK_CODES or (
        track_code.startswith("渡") or track_code.startswith("临")
    )
    if not is_turnout_link:
        return False
    return abs(start_point.x - end_point.x) >= 24.0


def _build_track_polyline(
    spec: DirectedTrackSpec,
    positions: dict[str, LayoutPoint],
    primary_node_names: set[str],
    distance_to_primary: dict[str, int],
    route_plan: TrackRoutePlan | None = None,
) -> tuple[LayoutPoint, ...]:
    start_point = positions[spec.start_node]
    end_point = positions[spec.end_node]
    if _should_use_direct_connector(spec.track_code, start_point, end_point):
        return (start_point, end_point)

    if abs(start_point.x - end_point.x) < 1e-6 or abs(start_point.y - end_point.y) < 1e-6:
        if route_plan is None or abs(start_point.y - end_point.y) < 1e-6 and abs(route_plan.actual_lane_y - start_point.y) < 1e-6:
            return (start_point, end_point)

    if route_plan is not None:
        return _build_laned_polyline(
            start_point=start_point,
            end_point=end_point,
            lane_y=route_plan.actual_lane_y,
            start_join_offset_px=route_plan.start_join_offset_px,
            end_join_offset_px=route_plan.end_join_offset_px,
        )

    start_on_primary = spec.start_node in primary_node_names
    end_on_primary = spec.end_node in primary_node_names
    if start_on_primary and not end_on_primary:
        candidate_points = (
            start_point,
            LayoutPoint(x=start_point.x, y=end_point.y),
            end_point,
        )
    elif end_on_primary and not start_on_primary:
        candidate_points = (
            start_point,
            LayoutPoint(x=end_point.x, y=start_point.y),
            end_point,
        )
    else:
        start_distance = distance_to_primary[spec.start_node]
        end_distance = distance_to_primary[spec.end_node]
        if start_distance < end_distance:
            candidate_points = (
                start_point,
                LayoutPoint(x=start_point.x, y=end_point.y),
                end_point,
            )
        else:
            candidate_points = (
                start_point,
                LayoutPoint(x=end_point.x, y=start_point.y),
                end_point,
            )
    return _dedupe_polyline(candidate_points)


def _build_laned_polyline(
    *,
    start_point: LayoutPoint,
    end_point: LayoutPoint,
    lane_y: float,
    start_join_offset_px: float = 0.0,
    end_join_offset_px: float = 0.0,
) -> tuple[LayoutPoint, ...]:
    direction = 1.0 if end_point.x >= start_point.x else -1.0
    total_span = abs(end_point.x - start_point.x)
    if total_span < 24.0:
        return (start_point, end_point)

    start_delta = abs(start_point.y - lane_y)
    end_delta = abs(end_point.y - lane_y)
    start_offset = _lane_connector_offset(start_delta) + start_join_offset_px
    end_offset = _lane_connector_offset(end_delta) + end_join_offset_px
    max_offset_budget = max(0.0, total_span - 12.0)
    total_offset = start_offset + end_offset
    if total_offset > max_offset_budget and total_offset > 1e-6:
        scale = max_offset_budget / total_offset
        start_offset *= scale
        end_offset *= scale

    points: list[LayoutPoint] = [start_point]
    lane_start_x = start_point.x
    if start_delta >= 1e-6 and start_offset > 1e-6:
        lane_start_x = start_point.x + direction * start_offset
        points.append(LayoutPoint(x=lane_start_x, y=lane_y))
    elif start_delta >= 1e-6:
        return (start_point, end_point)

    lane_end_x = end_point.x
    if end_delta >= 1e-6 and end_offset > 1e-6:
        lane_end_x = end_point.x - direction * end_offset
    elif end_delta >= 1e-6:
        return (start_point, end_point)

    if abs(lane_end_x - lane_start_x) > 1e-6:
        points.append(LayoutPoint(x=lane_end_x, y=lane_y))

    points.append(end_point)
    return _dedupe_polyline(points)


def _lane_connector_offset(vertical_delta: float) -> float:
    if vertical_delta < 1e-6:
        return 0.0
    return min(72.0, max(18.0, vertical_delta * 0.8))


def _build_directed_track_specs(
    routes: dict[str, SegmentedPhysicalRoute],
    master: MasterData,
) -> dict[str, DirectedTrackSpec]:
    specs: dict[str, DirectedTrackSpec] = {}
    for route in routes.values():
        route_nodes = _resolve_route_node_chain(route, master)
        for segment, start_node, end_node in zip(route.segments, route_nodes, route_nodes[1:]):
            spec = DirectedTrackSpec(
                track_code=segment.track_code,
                start_node=start_node,
                end_node=end_node,
                display_span_px=_log_display_span(segment.physical_distance_m),
            )
            existing = specs.get(segment.track_code)
            if existing is not None and existing != spec:
                raise ValueError(
                    f"track {segment.track_code} has conflicting directed specs: "
                    f"{existing.model_dump()} vs {spec.model_dump()}"
                )
            specs[segment.track_code] = spec
    return specs


def _resolve_route_node_chain(
    route: SegmentedPhysicalRoute,
    master: MasterData,
) -> list[str]:
    current_node = _resolve_route_start_node(route, master)
    route_nodes = [current_node]
    for segment in route.segments:
        endpoint_a, endpoint_b = master.tracks[segment.track_code].endpoint_nodes
        if endpoint_a == current_node:
            current_node = endpoint_b
        elif endpoint_b == current_node:
            current_node = endpoint_a
        else:
            raise ValueError(
                f"route {route.branch_code} cannot continue through {segment.track_code} "
                f"from node {route_nodes[-1]}"
            )
        route_nodes.append(current_node)
    return route_nodes


def _resolve_route_start_node(
    route: SegmentedPhysicalRoute,
    master: MasterData,
) -> str:
    first_segment_endpoints = master.tracks[route.segments[0].track_code].endpoint_nodes
    left_node = route.left_node
    if left_node in first_segment_endpoints:
        return left_node

    if len(route.segments) > 1:
        second_segment_endpoints = set(master.tracks[route.segments[1].track_code].endpoint_nodes)
        endpoint_a, endpoint_b = first_segment_endpoints
        if endpoint_a in second_segment_endpoints and endpoint_b not in second_segment_endpoints:
            return endpoint_b
        if endpoint_b in second_segment_endpoints and endpoint_a not in second_segment_endpoints:
            return endpoint_a

    right_node = route.right_node
    endpoint_a, endpoint_b = first_segment_endpoints
    if right_node == endpoint_a:
        return endpoint_b
    if right_node == endpoint_b:
        return endpoint_a
    raise ValueError(
        f"route {route.branch_code} cannot resolve start node from "
        f"{first_segment_endpoints} with aliases left={route.left_node} right={route.right_node}"
    )


def _log_display_span(length_m: float) -> float:
    return 110.0 + 130.0 * log1p(max(length_m, 0.0) / 50.0)


def _solve_node_x_coordinates(
    track_specs: dict[str, DirectedTrackSpec],
) -> dict[str, float]:
    node_names = sorted(
        {
            node_name
            for spec in track_specs.values()
            for node_name in (spec.start_node, spec.end_node)
        }
    )
    node_index = {node_name: index for index, node_name in enumerate(node_names)}

    matrix_rows: list[list[float]] = []
    target_values: list[float] = []
    for spec in track_specs.values():
        row = [0.0] * len(node_names)
        row[node_index[spec.start_node]] = -1.0
        row[node_index[spec.end_node]] = 1.0
        matrix_rows.append(row)
        target_values.append(spec.display_span_px)

    anchor_node = "5号门" if "5号门" in node_index else node_names[0]
    anchor_row = [0.0] * len(node_names)
    anchor_row[node_index[anchor_node]] = 1.0
    matrix_rows.append(anchor_row)
    target_values.append(0.0)

    solved = np.linalg.lstsq(
        np.array(matrix_rows, dtype=float),
        np.array(target_values, dtype=float),
        rcond=None,
    )[0]
    x_coordinates = {
        node_name: float(solved[node_index[node_name]])
        for node_name in node_names
    }
    min_x = min(x_coordinates.values(), default=0.0)
    if min_x < 0.0:
        x_coordinates = {
            node_name: value - min_x
            for node_name, value in x_coordinates.items()
        }
    return x_coordinates


def _solve_node_y_seeds(
    track_specs: dict[str, DirectedTrackSpec],
) -> dict[str, float]:
    node_names = sorted(
        {
            node_name
            for spec in track_specs.values()
            for node_name in (spec.start_node, spec.end_node)
        }
    )
    if len(node_names) < 3:
        return {node_name: 0.0 for node_name in node_names}

    node_index = {node_name: index for index, node_name in enumerate(node_names)}
    adjacency: dict[str, list[tuple[str, float]]] = {node_name: [] for node_name in node_names}
    for spec in track_specs.values():
        adjacency[spec.start_node].append((spec.end_node, spec.display_span_px))
        adjacency[spec.end_node].append((spec.start_node, spec.display_span_px))

    distance_matrix = np.zeros((len(node_names), len(node_names)), dtype=float)
    for start_node in node_names:
        shortest = _shortest_weighted_node_distances(adjacency, start_node)
        for end_node, distance in shortest.items():
            distance_matrix[node_index[start_node], node_index[end_node]] = distance

    centering = np.eye(len(node_names)) - np.ones((len(node_names), len(node_names))) / len(node_names)
    gram = -0.5 * centering.dot(distance_matrix**2).dot(centering)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    if len(sorted_indices) < 2 or eigenvalues[sorted_indices[1]] <= 0.0:
        return {node_name: 0.0 for node_name in node_names}

    second_axis = eigenvectors[:, sorted_indices[1]] * np.sqrt(eigenvalues[sorted_indices[1]])
    return {
        node_name: float(second_axis[node_index[node_name]])
        for node_name in node_names
    }


def _shortest_weighted_node_distances(
    adjacency: dict[str, list[tuple[str, float]]],
    start_node: str,
) -> dict[str, float]:
    distances = {node_name: float("inf") for node_name in adjacency}
    distances[start_node] = 0.0
    queue: list[tuple[float, str]] = [(0.0, start_node)]
    while queue:
        distance, node_name = heappop(queue)
        if distance != distances[node_name]:
            continue
        for neighbor, weight in adjacency[node_name]:
            candidate = distance + weight
            if candidate >= distances[neighbor]:
                continue
            distances[neighbor] = candidate
            heappush(queue, (candidate, neighbor))
    return distances


def _dedupe_polyline(
    points: Iterable[LayoutPoint],
) -> tuple[LayoutPoint, ...]:
    deduped: list[LayoutPoint] = []
    for point in points:
        if deduped and abs(deduped[-1].x - point.x) < 1e-6 and abs(deduped[-1].y - point.y) < 1e-6:
            continue
        deduped.append(point)
    if len(deduped) == 1:
        deduped.append(deduped[0])
    return tuple(deduped)


def _polyline_path(points: tuple[LayoutPoint, ...]) -> str:
    commands = [f"M {points[0].x:.1f} {points[0].y:.1f}"]
    for point in points[1:]:
        commands.append(f"L {point.x:.1f} {point.y:.1f}")
    return " ".join(commands)


def _polyline_midpoint(points: tuple[LayoutPoint, ...]) -> LayoutPoint:
    segment_lengths: list[float] = []
    total_length = 0.0
    for index in range(len(points) - 1):
        start_point = points[index]
        end_point = points[index + 1]
        length = ((end_point.x - start_point.x) ** 2 + (end_point.y - start_point.y) ** 2) ** 0.5
        segment_lengths.append(length)
        total_length += length
    midpoint_distance = total_length / 2.0
    traversed = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if traversed + segment_length < midpoint_distance:
            traversed += segment_length
            continue
        start_point = points[index]
        end_point = points[index + 1]
        if segment_length < 1e-6:
            return start_point
        ratio = (midpoint_distance - traversed) / segment_length
        return LayoutPoint(
            x=start_point.x + (end_point.x - start_point.x) * ratio,
            y=start_point.y + (end_point.y - start_point.y) * ratio,
        )
    return points[-1]


def _dominant_horizontal_span(
    points: tuple[LayoutPoint, ...],
) -> tuple[float, float]:
    best_length = -1.0
    best_span: tuple[float, float] | None = None
    for start_point, end_point in zip(points, points[1:]):
        if abs(start_point.y - end_point.y) >= 1e-6:
            continue
        left = min(start_point.x, end_point.x)
        right = max(start_point.x, end_point.x)
        length = right - left
        if length <= best_length:
            continue
        best_length = length
        best_span = (left, right)
    if best_span is not None:
        return best_span
    return (min(points[0].x, points[-1].x), max(points[0].x, points[-1].x))


def _track_text_layout_kind(track_code: str) -> str:
    explicit = TRACK_TEXT_LAYOUTS.get(track_code)
    if explicit is not None:
        return explicit
    lane_index = _fixed_track_lane_index(track_code)
    if lane_index is None:
        return "center"
    if lane_index <= -6:
        return "below"
    return "above"


def _track_text_baseline_offsets(track_code: str) -> tuple[float, float]:
    override = TRACK_TEXT_BASELINE_OVERRIDE_OFFSETS.get(track_code)
    if override is not None:
        return override
    kind = _track_text_layout_kind(track_code)
    if kind == "above":
        return (-26.0, -6.0)
    if kind == "below":
        return (22.0, 44.0)
    return (-16.0, 26.0)


def _track_text_baseline_ys(
    *,
    track_code: str,
    anchor: LayoutPoint,
) -> tuple[float, float]:
    label_offset_y, distance_offset_y = _track_text_baseline_offsets(track_code)
    return anchor.y + label_offset_y, anchor.y + distance_offset_y


def _track_text_boxes(
    track_code: str,
    anchor: LayoutPoint,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    label_baseline_y, distance_baseline_y = _track_text_baseline_ys(
        track_code=track_code,
        anchor=anchor,
    )
    label_width = max(48.0, len(track_code) * 18.0)
    distance_width = 78.0
    label_box = (
        anchor.x - label_width / 2.0,
        label_baseline_y - 30.0,
        anchor.x + label_width / 2.0,
        label_baseline_y + 4.0,
    )
    distance_box = (
        anchor.x - distance_width / 2.0,
        distance_baseline_y - 18.0,
        anchor.x + distance_width / 2.0,
        distance_baseline_y + 4.0,
    )
    return label_box, distance_box


def _track_text_group_box(
    track_code: str,
    anchor: LayoutPoint,
) -> tuple[float, float, float, float]:
    label_box, distance_box = _track_text_boxes(track_code, anchor)
    return (
        min(label_box[0], distance_box[0]),
        min(label_box[1], distance_box[1]),
        max(label_box[2], distance_box[2]),
        max(label_box[3], distance_box[3]),
    )


def _apply_track_text_anchor_biases(
    track_geometries: dict[str, ContinuousTrackGeometry],
    label_spans: dict[str, tuple[float, float]],
) -> None:
    for track_code, delta_x in TRACK_TEXT_ANCHOR_BIASES.items():
        if track_code not in track_geometries:
            continue
        _shift_track_label_anchor(
            track_geometries=track_geometries,
            label_spans=label_spans,
            track_code=track_code,
            delta_x=delta_x,
        )


def _separate_track_label_anchors(
    track_geometries: dict[str, ContinuousTrackGeometry],
    label_spans: dict[str, tuple[float, float]],
    *,
    padding: float = 8.0,
    max_iterations: int = 40,
) -> None:
    for _ in range(max_iterations):
        changed = False
        ordered_codes = sorted(
            track_geometries,
            key=lambda track_code: (
                _track_text_group_box(track_code, track_geometries[track_code].label_anchor)[1],
                track_geometries[track_code].label_anchor.x,
                track_code,
            ),
        )
        for index, first_code in enumerate(ordered_codes):
            first_geometry = track_geometries[first_code]
            first_box = _track_text_group_box(first_code, first_geometry.label_anchor)
            for second_code in ordered_codes[index + 1 :]:
                second_geometry = track_geometries[second_code]
                second_box = _track_text_group_box(second_code, second_geometry.label_anchor)
                if min(first_box[2], second_box[2]) <= max(first_box[0], second_box[0]):
                    continue
                if min(first_box[3], second_box[3]) <= max(first_box[1], second_box[1]):
                    continue
                _move_track_label_pair_apart(
                    track_geometries=track_geometries,
                    label_spans=label_spans,
                    first_code=first_code,
                    second_code=second_code,
                    padding=padding,
                )
                changed = True
                break
            if changed:
                break
        if not changed:
            return


def _track_label_box(
    track_code: str,
    anchor: LayoutPoint,
) -> tuple[float, float, float, float]:
    return _track_text_group_box(track_code, anchor)


def _move_track_label_pair_apart(
    *,
    track_geometries: dict[str, ContinuousTrackGeometry],
    label_spans: dict[str, tuple[float, float]],
    first_code: str,
    second_code: str,
    padding: float,
) -> None:
    first_geometry = track_geometries[first_code]
    second_geometry = track_geometries[second_code]
    first_left, _, first_right, _ = _track_text_group_box(first_code, first_geometry.label_anchor)
    second_left, _, second_right, _ = _track_text_group_box(second_code, second_geometry.label_anchor)
    if first_geometry.label_anchor.x <= second_geometry.label_anchor.x:
        required_gap = first_right + padding - second_left
        if required_gap <= 0.0:
            return
        _shift_track_label_anchor(
            track_geometries=track_geometries,
            label_spans=label_spans,
            track_code=first_code,
            delta_x=-required_gap,
        )
        first_left, _, first_right, _ = _track_text_group_box(
            first_code,
            track_geometries[first_code].label_anchor,
        )
        second_left, _, _, _ = _track_text_group_box(
            second_code,
            track_geometries[second_code].label_anchor,
        )
        remaining_gap = first_right + padding - second_left
        if remaining_gap > 0.0:
            _shift_track_label_anchor(
                track_geometries=track_geometries,
                label_spans=label_spans,
                track_code=second_code,
                delta_x=remaining_gap,
            )
        return

    required_gap = second_right + padding - first_left
    if required_gap <= 0.0:
        return
    _shift_track_label_anchor(
        track_geometries=track_geometries,
        label_spans=label_spans,
        track_code=first_code,
        delta_x=required_gap,
    )
    first_left, _, _, _ = _track_text_group_box(first_code, track_geometries[first_code].label_anchor)
    _, _, second_right, _ = _track_text_group_box(second_code, track_geometries[second_code].label_anchor)
    remaining_gap = second_right + padding - first_left
    if remaining_gap > 0.0:
        _shift_track_label_anchor(
            track_geometries=track_geometries,
            label_spans=label_spans,
            track_code=second_code,
            delta_x=-remaining_gap,
        )


def _shift_track_label_anchor(
    *,
    track_geometries: dict[str, ContinuousTrackGeometry],
    label_spans: dict[str, tuple[float, float]],
    track_code: str,
    delta_x: float,
) -> None:
    geometry = track_geometries[track_code]
    anchor = geometry.label_anchor
    left_limit, right_limit = label_spans[track_code]
    width = max(max(48.0, len(track_code) * 18.0), 78.0)
    min_x = left_limit + width / 2.0
    max_x = right_limit - width / 2.0
    if min_x > max_x:
        return
    target_x = min(max(anchor.x + delta_x, min_x), max_x)
    if abs(target_x - anchor.x) < 1e-6:
        return
    geometry.label_anchor = LayoutPoint(x=target_x, y=anchor.y)


if __name__ == "__main__":
    raise SystemExit(main())
