from __future__ import annotations

import json
from math import hypot
from pathlib import Path

from pydantic import BaseModel, Field

from fzed_shunting.demo.layout import LayoutPoint, RoutePolyline


class SchematicArea(BaseModel):
    area_id: str
    label: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> LayoutPoint:
        return LayoutPoint(x=self.x + self.width / 2.0, y=self.y + self.height / 2.0)


class SchematicTrackGeometry(BaseModel):
    track_code: str
    points: tuple[LayoutPoint, ...] = Field(default_factory=tuple)
    label_anchor: LayoutPoint
    center: LayoutPoint
    always_visible: bool = False
    is_mainline: bool = False


class SchematicLayout(BaseModel):
    canvas_width: float
    canvas_height: float
    areas: tuple[SchematicArea, ...] = Field(default_factory=tuple)
    track_geometries: dict[str, SchematicTrackGeometry] = Field(default_factory=dict)
    mainline_track_codes: tuple[str, ...] = Field(default_factory=tuple)


def load_schematic_layout(base_dir: Path) -> SchematicLayout:
    raw = json.loads((base_dir / "schematic_layout.json").read_text(encoding="utf-8"))
    mainline_track_codes = tuple(item["trackCode"] for item in raw["mainlineTracks"])
    mainline_track_code_set = set(mainline_track_codes)

    areas = tuple(
        SchematicArea(
            area_id=str(item["id"]),
            label=str(item["label"]),
            x=float(item["x"]),
            y=float(item["y"]),
            width=float(item["width"]),
            height=float(item["height"]),
        )
        for item in raw["areas"]
    )

    track_geometries: dict[str, SchematicTrackGeometry] = {}
    for track_code, spec in raw["tracks"].items():
        points = tuple(LayoutPoint(x=float(x), y=float(y)) for x, y in spec["points"])
        center = _polyline_midpoint(points)
        label_anchor = LayoutPoint(
            x=float(spec["labelAnchor"][0]),
            y=float(spec["labelAnchor"][1]),
        )
        track_geometries[track_code] = SchematicTrackGeometry(
            track_code=track_code,
            points=points,
            label_anchor=label_anchor,
            center=center,
            always_visible=bool(spec.get("alwaysVisible", False)),
            is_mainline=track_code in mainline_track_code_set,
        )

    return SchematicLayout(
        canvas_width=float(raw["canvas"]["width"]),
        canvas_height=float(raw["canvas"]["height"]),
        areas=areas,
        track_geometries=track_geometries,
        mainline_track_codes=mainline_track_codes,
    )


def build_schematic_route(layout: SchematicLayout, track_codes: list[str]) -> RoutePolyline:
    if not track_codes:
        return RoutePolyline()

    route_points: list[LayoutPoint] = []
    for index, track_code in enumerate(track_codes):
        geometry = layout.track_geometries.get(track_code)
        if geometry is None:
            continue
        previous_geometry = (
            layout.track_geometries.get(track_codes[index - 1])
            if index > 0
            else None
        )
        next_geometry = (
            layout.track_geometries.get(track_codes[index + 1])
            if index < len(track_codes) - 1
            else None
        )
        oriented_points = _orient_track_points(
            points=list(geometry.points),
            previous_geometry=previous_geometry,
            next_geometry=next_geometry,
        )
        _extend_unique(route_points, oriented_points)

    cumulative_lengths = [0.0]
    total_length_px = 0.0
    for start, end in zip(route_points, route_points[1:], strict=False):
        total_length_px += hypot(end.x - start.x, end.y - start.y)
        cumulative_lengths.append(total_length_px)
    return RoutePolyline(
        track_codes=list(track_codes),
        points=route_points,
        total_length_px=total_length_px,
        cumulative_lengths=cumulative_lengths,
    )


def _orient_track_points(
    *,
    points: list[LayoutPoint],
    previous_geometry: SchematicTrackGeometry | None,
    next_geometry: SchematicTrackGeometry | None,
) -> list[LayoutPoint]:
    if len(points) <= 1:
        return points

    previous_shared = _shared_endpoint(previous_geometry, points) if previous_geometry is not None else None
    next_shared = _shared_endpoint(next_geometry, points) if next_geometry is not None else None

    if previous_shared is not None:
        if _same_point(points[0], previous_shared):
            return points
        if _same_point(points[-1], previous_shared):
            return list(reversed(points))
    if next_shared is not None:
        if _same_point(points[-1], next_shared):
            return points
        if _same_point(points[0], next_shared):
            return list(reversed(points))
    return points


def _shared_endpoint(
    geometry: SchematicTrackGeometry,
    points: list[LayoutPoint],
) -> LayoutPoint | None:
    candidate_points = (geometry.points[0], geometry.points[-1])
    for candidate in candidate_points:
        if _same_point(candidate, points[0]) or _same_point(candidate, points[-1]):
            return candidate
    return None


def _same_point(left: LayoutPoint, right: LayoutPoint) -> bool:
    return abs(left.x - right.x) <= 1e-6 and abs(left.y - right.y) <= 1e-6


def _extend_unique(target: list[LayoutPoint], source: list[LayoutPoint]) -> None:
    for point in source:
        if target and _same_point(target[-1], point):
            continue
        target.append(point)


def _polyline_midpoint(points: tuple[LayoutPoint, ...]) -> LayoutPoint:
    if not points:
        return LayoutPoint(x=0.0, y=0.0)
    if len(points) == 1:
        return points[0]

    segment_lengths = [hypot(end.x - start.x, end.y - start.y) for start, end in zip(points, points[1:], strict=False)]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        return points[len(points) // 2]

    target_length = total_length / 2.0
    walked = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if walked + segment_length < target_length:
            walked += segment_length
            continue
        start = points[index]
        end = points[index + 1]
        ratio = (target_length - walked) / segment_length if segment_length > 1e-9 else 0.0
        return LayoutPoint(
            x=start.x + (end.x - start.x) * ratio,
            y=start.y + (end.y - start.y) * ratio,
        )
    return points[-1]
