from __future__ import annotations

from math import hypot
from pathlib import Path
import json
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

from fzed_shunting.domain.master_data import MasterData


class LayoutPoint(BaseModel):
    x: float
    y: float


class LayoutRect(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation_deg: float = 0.0


class TrackGeometry(BaseModel):
    track_code: str
    points: list[LayoutPoint] = Field(default_factory=list)
    center: LayoutPoint
    label_anchor: LayoutPoint
    background_anchor: LayoutPoint | None = None
    background_rect: LayoutRect | None = None
    length_px: float


class BackgroundImageSpec(BaseModel):
    path: str
    crop_box: tuple[int, int, int, int] | None = None


class TopologyLayout(BaseModel):
    canvas_width: float
    canvas_height: float
    background_image: BackgroundImageSpec | None = None
    node_points: dict[str, LayoutPoint] = Field(default_factory=dict)
    track_geometries: dict[str, TrackGeometry] = Field(default_factory=dict)


class RoutePolyline(BaseModel):
    track_codes: list[str] = Field(default_factory=list)
    points: list[LayoutPoint] = Field(default_factory=list)
    total_length_px: float = 0.0
    cumulative_lengths: list[float] = Field(default_factory=list)


def load_topology_layout(base_dir: Path, master: MasterData) -> TopologyLayout:
    raw = json.loads((base_dir / "topology_layout.json").read_text(encoding="utf-8"))
    background_image = None
    if raw.get("backgroundImage") is not None:
        crop_box = raw["backgroundImage"].get("crop")
        background_image = BackgroundImageSpec(
            path=str(raw["backgroundImage"]["path"]),
            crop_box=tuple(int(value) for value in crop_box) if crop_box is not None else None,
        )
    node_points = {
        node_name: LayoutPoint(x=float(coords[0]), y=float(coords[1]))
        for node_name, coords in raw["nodePoints"].items()
    }
    pptx_layout_rects = _load_pptx_track_rects(
        base_dir.parent.parent / "现场布局图.pptx",
        canvas_width=float(raw["canvas"]["width"]),
        canvas_height=float(raw["canvas"]["height"]),
    )
    track_geometries: dict[str, TrackGeometry] = {}
    raw_track_geometries = raw.get("trackGeometries", {})
    for track_code, spec in raw_track_geometries.items():
        points = [_resolve_point(point_spec, node_points) for point_spec in spec["points"]]
        center = _resolve_point(spec.get("center"), node_points) if spec.get("center") is not None else _polyline_midpoint(points)
        label_anchor = (
            _resolve_point(spec.get("labelAnchor"), node_points)
            if spec.get("labelAnchor") is not None
            else center
        )
        background_anchor = (
            _resolve_point(spec.get("backgroundAnchor"), node_points)
            if spec.get("backgroundAnchor") is not None
            else None
        )
        background_rect = None
        if background_anchor is None:
            background_rect = pptx_layout_rects.get(track_code)
            if background_rect is not None:
                background_anchor = LayoutPoint(
                    x=background_rect.x + background_rect.width / 2.0,
                    y=background_rect.y + background_rect.height / 2.0,
                )
        track_geometries[track_code] = TrackGeometry(
            track_code=track_code,
            points=points,
            center=center,
            label_anchor=label_anchor,
            background_anchor=background_anchor,
            background_rect=background_rect,
            length_px=_polyline_length(points),
        )

    for track_code, track in master.tracks.items():
        if track_code in track_geometries:
            continue
        endpoint_nodes = list(track.endpoint_nodes)
        if len(endpoint_nodes) != 2:
            continue
        points = [node_points[endpoint_nodes[0]], node_points[endpoint_nodes[1]]]
        center = _polyline_midpoint(points)
        background_rect = pptx_layout_rects.get(track_code)
        background_anchor = None
        if background_rect is not None:
            background_anchor = LayoutPoint(
                x=background_rect.x + background_rect.width / 2.0,
                y=background_rect.y + background_rect.height / 2.0,
            )
        track_geometries[track_code] = TrackGeometry(
            track_code=track_code,
            points=points,
            center=center,
            label_anchor=center,
            background_anchor=background_anchor,
            background_rect=background_rect,
            length_px=_polyline_length(points),
        )
    return TopologyLayout(
        canvas_width=float(raw["canvas"]["width"]),
        canvas_height=float(raw["canvas"]["height"]),
        background_image=background_image,
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

    points: list[LayoutPoint] = []
    for index, track_code in enumerate(track_codes):
        geometry = layout.track_geometries[track_code]
        start_point = geometry.center
        end_point = geometry.center
        if index > 0:
            previous_geometry = layout.track_geometries[track_codes[index - 1]]
            previous_shared = _find_shared_node(previous_geometry, geometry)
            if previous_shared is not None:
                start_point = previous_shared
        if index < len(track_codes) - 1:
            next_geometry = layout.track_geometries[track_codes[index + 1]]
            next_shared = _find_shared_node(geometry, next_geometry)
            if next_shared is not None:
                end_point = next_shared

        segment = _polyline_section(geometry.points, start_point, end_point)
        if not segment:
            segment = [start_point, end_point] if not _same_point(start_point, end_point) else [start_point]
        _extend_unique(points, segment)

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


def _load_pptx_track_rects(
    pptx_path: Path,
    *,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, LayoutRect]:
    if not pptx_path.exists():
        return {}

    slide_width = 9_144_000.0
    slide_height = 6_858_000.0
    fit_track_codes = [
        "存5南",
        "存5北",
        "存4南",
        "存4北",
        "存3",
        "渡13",
        "渡12",
        "联7",
        "渡10",
        "渡9",
        "存2",
        "预修",
        "渡7",
        "存1",
        "渡11",
        "临3",
        "洗北",
        "机棚",
        "机库",
        "调棚",
        "临4",
        "轮",
        "洗南",
    ]
    alias_to_track = {
        "联6线": "联6",
        "修1内": "修1库内",
        "修1外": "修1库外",
        "修2内": "修2库内",
        "修2外": "修2库外",
        "修3内": "修3库内",
        "修3外": "修3库外",
        "修4内": "修4库内",
        "修4外": "修4库外",
    }
    ignored_texts = {
        "",
        "页-1",
        "释义",
        "检修库",
        "北",
        "：临停线",
        "：辆渡线",
        "：存车线",
        "临",
        "渡",
        "存",
    }
    targets = set(fit_track_codes) | {
        "渡1",
        "渡2",
        "渡3",
        "临1",
        "临2",
        "渡4",
        "渡5",
        "渡6",
        "渡8",
        "机北",
        "调北",
        "联6",
        "油",
        "抛",
        "存4南",
        "存5南",
        "修1库内",
        "修1库外",
        "修2库内",
        "修2库外",
        "修3库内",
        "修3库外",
        "修4库内",
        "修4库外",
    }
    slide_boxes: dict[str, dict[str, float]] = {}
    with ZipFile(pptx_path) as pptx_zip:
        root = ET.fromstring(pptx_zip.read("ppt/slides/slide1.xml"))
    namespaces = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    }

    def parse_xfrm(xfrm) -> dict[str, float]:
        if xfrm is None:
            return {
                "off_x": 0.0,
                "off_y": 0.0,
                "ext_cx": 0.0,
                "ext_cy": 0.0,
                "ch_off_x": 0.0,
                "ch_off_y": 0.0,
                "ch_ext_cx": 0.0,
                "ch_ext_cy": 0.0,
                "rot": 0.0,
            }
        off = xfrm.find("./a:off", namespaces)
        ext = xfrm.find("./a:ext", namespaces)
        ch_off = xfrm.find("./a:chOff", namespaces)
        ch_ext = xfrm.find("./a:chExt", namespaces)
        ext_cx = float(ext.get("cx")) if ext is not None else 0.0
        ext_cy = float(ext.get("cy")) if ext is not None else 0.0
        return {
            "off_x": float(off.get("x")) if off is not None else 0.0,
            "off_y": float(off.get("y")) if off is not None else 0.0,
            "ext_cx": ext_cx,
            "ext_cy": ext_cy,
            "ch_off_x": float(ch_off.get("x")) if ch_off is not None else 0.0,
            "ch_off_y": float(ch_off.get("y")) if ch_off is not None else 0.0,
            "ch_ext_cx": float(ch_ext.get("cx")) if ch_ext is not None else ext_cx,
            "ch_ext_cy": float(ch_ext.get("cy")) if ch_ext is not None else ext_cy,
            "rot": float(xfrm.get("rot", "0")),
        }

    def compose_group(parent: dict[str, float], group: dict[str, float]) -> dict[str, float]:
        scale_x = parent["ext_cx"] / parent["ch_ext_cx"] if parent["ch_ext_cx"] else 1.0
        scale_y = parent["ext_cy"] / parent["ch_ext_cy"] if parent["ch_ext_cy"] else 1.0
        return {
            "off_x": parent["off_x"] + (group["off_x"] - parent["ch_off_x"]) * scale_x,
            "off_y": parent["off_y"] + (group["off_y"] - parent["ch_off_y"]) * scale_y,
            "ext_cx": group["ext_cx"] * scale_x,
            "ext_cy": group["ext_cy"] * scale_y,
            "ch_off_x": group["ch_off_x"],
            "ch_off_y": group["ch_off_y"],
            "ch_ext_cx": group["ch_ext_cx"],
            "ch_ext_cy": group["ch_ext_cy"],
            "rot": parent["rot"] + group["rot"],
        }

    def transform_box(parent: dict[str, float], child: dict[str, float]) -> dict[str, float]:
        scale_x = parent["ext_cx"] / parent["ch_ext_cx"] if parent["ch_ext_cx"] else 1.0
        scale_y = parent["ext_cy"] / parent["ch_ext_cy"] if parent["ch_ext_cy"] else 1.0
        return {
            "off_x": parent["off_x"] + (child["off_x"] - parent["ch_off_x"]) * scale_x,
            "off_y": parent["off_y"] + (child["off_y"] - parent["ch_off_y"]) * scale_y,
            "ext_cx": child["ext_cx"] * scale_x,
            "ext_cy": child["ext_cy"] * scale_y,
            "rot": parent["rot"] + child["rot"],
        }

    def extract_text(shape) -> str:
        return "".join(
            (node.text or "")
            for node in shape.findall(".//a:t", namespaces)
        ).strip()

    def walk(node, parent_transform: dict[str, float]) -> None:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "grpSp":
            group_transform = compose_group(
                parent_transform,
                parse_xfrm(node.find("./p:grpSpPr/a:xfrm", namespaces)),
            )
            for child in node:
                if child.tag.rsplit("}", 1)[-1] in {"grpSp", "sp"}:
                    walk(child, group_transform)
            return
        if tag != "sp":
            return
        track_code = extract_text(node)
        if track_code in ignored_texts:
            return
        track_code = alias_to_track.get(track_code, track_code)
        if track_code not in targets:
            return
        box = transform_box(parent_transform, parse_xfrm(node.find("./p:spPr/a:xfrm", namespaces)))
        slide_boxes[track_code] = box

    root_transform = {
        "off_x": 0.0,
        "off_y": 0.0,
        "ext_cx": slide_width,
        "ext_cy": slide_height,
        "ch_off_x": 0.0,
        "ch_off_y": 0.0,
        "ch_ext_cx": slide_width,
        "ch_ext_cy": slide_height,
        "rot": 0.0,
    }
    sp_tree = root.find(".//p:spTree", namespaces)
    if sp_tree is None:
        return {}
    for child in sp_tree:
        if child.tag.rsplit("}", 1)[-1] in {"grpSp", "sp"}:
            walk(child, root_transform)

    fit_boxes = {
        track_code: slide_boxes[track_code]
        for track_code in fit_track_codes
        if track_code in slide_boxes
    }
    if len(fit_boxes) < 3:
        return {}

    image_points = {
        "存5南": (1439.0, 51.0),
        "存5北": (1941.0, 51.0),
        "存4南": (1296.0, 193.0),
        "存4北": (1737.0, 196.0),
        "存3": (1738.0, 327.0),
        "渡13": (458.0, 483.0),
        "渡12": (560.0, 482.0),
        "联7": (706.0, 446.0),
        "渡10": (887.0, 447.0),
        "渡9": (989.0, 446.0),
        "存2": (1738.0, 452.0),
        "预修": (1330.0, 574.0),
        "渡7": (1614.0, 559.0),
        "存1": (1738.0, 599.0),
        "渡11": (564.0, 741.0),
        "临3": (1100.0, 869.0),
        "洗北": (942.0, 959.0),
        "机棚": (1415.0, 724.0),
        "机库": (1549.0, 1090.0),
        "调棚": (1416.0, 874.0),
        "临4": (1055.0, 648.0),
        "轮": (320.0, 1000.0),
        "洗南": (707.0, 1022.0),
    }
    fit_codes = [
        track_code
        for track_code in fit_track_codes
        if track_code in slide_boxes and track_code in image_points
    ]

    matrix = [
        [
            slide_boxes[track_code]["off_x"] + slide_boxes[track_code]["ext_cx"] / 2.0,
            slide_boxes[track_code]["off_y"] + slide_boxes[track_code]["ext_cy"] / 2.0,
            1.0,
        ]
        for track_code in fit_codes
    ]
    target_x = [image_points[track_code][0] for track_code in fit_codes]
    target_y = [image_points[track_code][1] for track_code in fit_codes]

    try:
        import numpy as np
    except Exception:
        return {}

    matrix_np = np.array(matrix, dtype=float)
    target_x_np = np.array(target_x, dtype=float)
    target_y_np = np.array(target_y, dtype=float)
    coef_x = np.linalg.lstsq(matrix_np, target_x_np, rcond=None)[0]
    coef_y = np.linalg.lstsq(matrix_np, target_y_np, rcond=None)[0]

    def transform_point(x: float, y: float) -> tuple[float, float]:
        point = np.array([x, y, 1.0], dtype=float)
        return float(point @ coef_x), float(point @ coef_y)

    width_scale = float(coef_x[0])
    height_scale = float(coef_y[1])
    rects: dict[str, LayoutRect] = {}
    for track_code, box in slide_boxes.items():
        x, y = transform_point(box["off_x"], box["off_y"])
        rects[track_code] = LayoutRect(
            x=x,
            y=y,
            width=max(22.0, box["ext_cx"] * width_scale),
            height=max(18.0, box["ext_cy"] * height_scale),
            rotation_deg=box["rot"] / 60_000.0,
        )
    return rects


def _resolve_point(spec, node_points: dict[str, LayoutPoint]) -> LayoutPoint:
    if isinstance(spec, str):
        return node_points[spec]
    if isinstance(spec, dict):
        return LayoutPoint(x=float(spec["x"]), y=float(spec["y"]))
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return LayoutPoint(x=float(spec[0]), y=float(spec[1]))
    raise ValueError(f"Unsupported point spec: {spec!r}")


def _midpoint(left: LayoutPoint, right: LayoutPoint) -> LayoutPoint:
    return LayoutPoint(x=(left.x + right.x) / 2.0, y=(left.y + right.y) / 2.0)


def _polyline_length(points: list[LayoutPoint]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for start, end in zip(points, points[1:], strict=False):
        total += hypot(end.x - start.x, end.y - start.y)
    return total


def _polyline_midpoint(points: list[LayoutPoint]) -> LayoutPoint:
    if not points:
        return LayoutPoint(x=0.0, y=0.0)
    if len(points) == 1:
        return points[0]
    return _point_on_polyline(points, _polyline_length(points) / 2.0)


def _point_on_polyline(points: list[LayoutPoint], distance: float) -> LayoutPoint:
    if not points:
        return LayoutPoint(x=0.0, y=0.0)
    if len(points) == 1:
        return points[0]
    clipped = min(max(distance, 0.0), _polyline_length(points))
    traversed = 0.0
    for start, end in zip(points, points[1:], strict=False):
        segment_length = hypot(end.x - start.x, end.y - start.y)
        if traversed + segment_length < clipped:
            traversed += segment_length
            continue
        if segment_length <= 1e-9:
            return end
        ratio = (clipped - traversed) / segment_length
        return LayoutPoint(
            x=start.x + (end.x - start.x) * ratio,
            y=start.y + (end.y - start.y) * ratio,
        )
    return points[-1]


def _polyline_section(points: list[LayoutPoint], start_point: LayoutPoint, end_point: LayoutPoint) -> list[LayoutPoint]:
    if not points:
        return []
    if _same_point(start_point, end_point):
        return [start_point]
    start_distance = _distance_along_polyline(points, start_point)
    end_distance = _distance_along_polyline(points, end_point)
    if start_distance is None or end_distance is None:
        return [start_point, end_point]

    reverse = start_distance > end_distance
    left = end_distance if reverse else start_distance
    right = start_distance if reverse else end_distance
    section = [_point_on_polyline(points, left)]
    traversed = 0.0
    for point, next_point in zip(points, points[1:], strict=False):
        segment_length = hypot(next_point.x - point.x, next_point.y - point.y)
        traversed += segment_length
        if left < traversed < right:
            section.append(next_point)
    section.append(_point_on_polyline(points, right))
    if reverse:
        section.reverse()
    return _dedupe_points(section)


def _distance_along_polyline(points: list[LayoutPoint], target: LayoutPoint) -> float | None:
    traversed = 0.0
    for start, end in zip(points, points[1:], strict=False):
        if _same_point(start, target):
            return traversed
        segment_length = hypot(end.x - start.x, end.y - start.y)
        if _point_on_segment(start, end, target):
            if segment_length <= 1e-9:
                return traversed
            return traversed + hypot(target.x - start.x, target.y - start.y)
        traversed += segment_length
    if _same_point(points[-1], target):
        return traversed
    return None


def _point_on_segment(start: LayoutPoint, end: LayoutPoint, target: LayoutPoint) -> bool:
    cross = (target.y - start.y) * (end.x - start.x) - (target.x - start.x) * (end.y - start.y)
    if abs(cross) > 1e-6:
        return False
    dot = (target.x - start.x) * (end.x - start.x) + (target.y - start.y) * (end.y - start.y)
    if dot < -1e-6:
        return False
    segment_length_sq = (end.x - start.x) ** 2 + (end.y - start.y) ** 2
    if dot - segment_length_sq > 1e-6:
        return False
    return True


def _extend_unique(target: list[LayoutPoint], additions: list[LayoutPoint]) -> None:
    for point in additions:
        if not target or not _same_point(target[-1], point):
            target.append(point)


def _dedupe_points(points: list[LayoutPoint]) -> list[LayoutPoint]:
    deduped: list[LayoutPoint] = []
    for point in points:
        if not deduped or not _same_point(deduped[-1], point):
            deduped.append(point)
    return deduped
