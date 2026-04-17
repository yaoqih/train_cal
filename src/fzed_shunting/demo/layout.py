from __future__ import annotations

from math import hypot
from pathlib import Path
import json

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData


class LayoutPoint(BaseModel):
    x: float
    y: float


class TrackGeometry(BaseModel):
    track_code: str
    points: list[LayoutPoint] = Field(default_factory=list)
    center: LayoutPoint
    length_px: float


class TopologyLayout(BaseModel):
    canvas_width: float
    canvas_height: float
    node_points: dict[str, LayoutPoint] = Field(default_factory=dict)
    track_geometries: dict[str, TrackGeometry] = Field(default_factory=dict)


class RoutePolyline(BaseModel):
    track_codes: list[str] = Field(default_factory=list)
    points: list[LayoutPoint] = Field(default_factory=list)
    total_length_px: float = 0.0
    cumulative_lengths: list[float] = Field(default_factory=list)


def load_topology_layout(base_dir: Path, master: MasterData) -> TopologyLayout:
    raw = json.loads((base_dir / "topology_layout.json").read_text(encoding="utf-8"))
    node_points = {
        node_name: LayoutPoint(x=float(coords[0]), y=float(coords[1]))
        for node_name, coords in raw["nodePoints"].items()
    }
    track_geometries: dict[str, TrackGeometry] = {}
    for track_code, track in master.tracks.items():
        endpoint_nodes = list(track.endpoint_nodes)
        if len(endpoint_nodes) != 2:
            continue
        points = [node_points[endpoint_nodes[0]], node_points[endpoint_nodes[1]]]
        track_geometries[track_code] = TrackGeometry(
            track_code=track_code,
            points=points,
            center=_midpoint(points[0], points[1]),
            length_px=_polyline_length(points),
        )
    return TopologyLayout(
        canvas_width=float(raw["canvas"]["width"]),
        canvas_height=float(raw["canvas"]["height"]),
        node_points=node_points,
        track_geometries=track_geometries,
    )


def build_route_polyline(layout: TopologyLayout, track_codes: list[str]) -> RoutePolyline:
    if not track_codes:
        return RoutePolyline()
    if len(track_codes) == 1:
        geometry = layout.track_geometries[track_codes[0]]
        return RoutePolyline(
            track_codes=list(track_codes),
            points=[geometry.center],
            total_length_px=0.0,
            cumulative_lengths=[0.0],
        )

    points: list[LayoutPoint] = [layout.track_geometries[track_codes[0]].center]
    for index, track_code in enumerate(track_codes[:-1]):
        next_track_code = track_codes[index + 1]
        shared_node = _find_shared_node(
            layout.track_geometries[track_code],
            layout.track_geometries[next_track_code],
        )
        if shared_node is not None and not _same_point(points[-1], shared_node):
            points.append(shared_node)
        next_center = layout.track_geometries[next_track_code].center
        if not _same_point(points[-1], next_center):
            points.append(next_center)

    cumulative_lengths = [0.0]
    total_length_px = 0.0
    for start, end in zip(points, points[1:], strict=False):
        total_length_px += hypot(end.x - start.x, end.y - start.y)
        cumulative_lengths.append(total_length_px)
    return RoutePolyline(
        track_codes=list(track_codes),
        points=points,
        total_length_px=total_length_px,
        cumulative_lengths=cumulative_lengths,
    )


def point_at_progress(route: RoutePolyline, progress: float) -> LayoutPoint:
    if not route.points:
        return LayoutPoint(x=0.0, y=0.0)
    if len(route.points) == 1 or route.total_length_px <= 1e-9:
        return route.points[0]

    clipped = min(1.0, max(0.0, progress))
    target_length = route.total_length_px * clipped
    for index in range(1, len(route.points)):
        previous_length = route.cumulative_lengths[index - 1]
        current_length = route.cumulative_lengths[index]
        if target_length > current_length and index < len(route.points) - 1:
            continue
        segment_length = current_length - previous_length
        if segment_length <= 1e-9:
            return route.points[index]
        ratio = (target_length - previous_length) / segment_length
        start = route.points[index - 1]
        end = route.points[index]
        return LayoutPoint(
            x=start.x + (end.x - start.x) * ratio,
            y=start.y + (end.y - start.y) * ratio,
        )
    return route.points[-1]


def route_to_svg_path(route: RoutePolyline) -> str:
    if not route.points:
        return ""
    first = route.points[0]
    segments = [f"M {first.x:.1f} {first.y:.1f}"]
    for point in route.points[1:]:
        segments.append(f"L {point.x:.1f} {point.y:.1f}")
    return " ".join(segments)


def _find_shared_node(left: TrackGeometry, right: TrackGeometry) -> LayoutPoint | None:
    for left_point in left.points:
        for right_point in right.points:
            if _same_point(left_point, right_point):
                return left_point
    return None


def _same_point(left: LayoutPoint, right: LayoutPoint) -> bool:
    return abs(left.x - right.x) <= 1e-9 and abs(left.y - right.y) <= 1e-9


def _midpoint(left: LayoutPoint, right: LayoutPoint) -> LayoutPoint:
    return LayoutPoint(x=(left.x + right.x) / 2.0, y=(left.y + right.y) / 2.0)


def _polyline_length(points: list[LayoutPoint]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for start, end in zip(points, points[1:], strict=False):
        total += hypot(end.x - start.x, end.y - start.y)
    return total
