from pathlib import Path
from math import inf

import pytest

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.tools.segmented_routes_svg import (
    ROUTE_LANE_SPACING,
    _track_text_boxes,
    build_continuous_network_layout,
    export_segmented_route_artifacts,
    load_segmented_physical_routes,
    render_segmented_routes_svg,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _track_span_x(layout, track_code: str) -> float:
    start, end = layout.track_geometries[track_code].endpoints
    return abs(end.x - start.x)


def _path_points(path_d: str) -> list[tuple[float, float]]:
    tokens = path_d.split()
    points: list[tuple[float, float]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command not in {"M", "L"}:
            raise AssertionError(f"unexpected path command {command}")
        points.append((float(tokens[index]), float(tokens[index + 1])))
        index += 2
    return points


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def orientation(
        start: tuple[float, float],
        middle: tuple[float, float],
        end: tuple[float, float],
    ) -> int:
        value = (
            (middle[0] - start[0]) * (end[1] - start[1])
            - (middle[1] - start[1]) * (end[0] - start[0])
        )
        if abs(value) < 1e-6:
            return 0
        return 1 if value > 0 else -1

    def on_segment(
        start: tuple[float, float],
        middle: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        return (
            min(start[0], end[0]) <= middle[0] <= max(start[0], end[0])
            and min(start[1], end[1]) <= middle[1] <= max(start[1], end[1])
        )

    first = orientation(first_start, first_end, second_start)
    second = orientation(first_start, first_end, second_end)
    third = orientation(second_start, second_end, first_start)
    fourth = orientation(second_start, second_end, first_end)
    if first == 0 and on_segment(first_start, second_start, first_end):
        return True
    if second == 0 and on_segment(first_start, second_end, first_end):
        return True
    if third == 0 and on_segment(second_start, first_start, second_end):
        return True
    if fourth == 0 and on_segment(second_start, first_end, second_end):
        return True
    return first != second and third != fourth


def _count_non_touching_crossings(layout) -> int:
    track_codes = list(layout.track_geometries)
    crossings = 0
    for index, first_code in enumerate(track_codes):
        first_geometry = layout.track_geometries[first_code]
        first_points = _path_points(first_geometry.path_d)
        for second_code in track_codes[index + 1 :]:
            second_geometry = layout.track_geometries[second_code]
            if set(first_geometry.endpoint_nodes) & set(second_geometry.endpoint_nodes):
                continue
            second_points = _path_points(second_geometry.path_d)
            found = False
            for first_segment_index in range(len(first_points) - 1):
                for second_segment_index in range(len(second_points) - 1):
                    if _segments_intersect(
                        first_points[first_segment_index],
                        first_points[first_segment_index + 1],
                        second_points[second_segment_index],
                        second_points[second_segment_index + 1],
                    ):
                        crossings += 1
                        found = True
                        break
                if found:
                    break
    return crossings


def _count_non_touching_parallel_near_overlaps(
    layout,
    *,
    clearance: float,
    min_shared_span: float,
) -> int:
    track_codes = list(layout.track_geometries)
    overlaps = 0
    for index, first_code in enumerate(track_codes):
        first_geometry = layout.track_geometries[first_code]
        first_points = _path_points(first_geometry.path_d)
        for second_code in track_codes[index + 1 :]:
            second_geometry = layout.track_geometries[second_code]
            if set(first_geometry.endpoint_nodes) & set(second_geometry.endpoint_nodes):
                continue
            second_points = _path_points(second_geometry.path_d)
            found = False
            for first_start, first_end in zip(first_points, first_points[1:]):
                first_min_x = min(first_start[0], first_end[0])
                first_max_x = max(first_start[0], first_end[0])
                first_min_y = min(first_start[1], first_end[1])
                first_max_y = max(first_start[1], first_end[1])
                first_dx = abs(first_end[0] - first_start[0])
                first_dy = abs(first_end[1] - first_start[1])
                for second_start, second_end in zip(second_points, second_points[1:]):
                    second_min_x = min(second_start[0], second_end[0])
                    second_max_x = max(second_start[0], second_end[0])
                    second_min_y = min(second_start[1], second_end[1])
                    second_max_y = max(second_start[1], second_end[1])
                    second_dx = abs(second_end[0] - second_start[0])
                    second_dy = abs(second_end[1] - second_start[1])
                    if first_dy < 1e-6 and second_dy < 1e-6:
                        shared_span = min(first_max_x, second_max_x) - max(first_min_x, second_min_x)
                        distance = abs(first_start[1] - second_start[1])
                    else:
                        continue
                    if shared_span < min_shared_span or distance >= clearance:
                        continue
                    overlaps += 1
                    found = True
                    break
                if found:
                    break
    return overlaps


def _closest_parallel_gap(
    layout,
    first_code: str,
    second_code: str,
    *,
    min_shared_span: float,
) -> float | None:
    first_points = _path_points(layout.track_geometries[first_code].path_d)
    second_points = _path_points(layout.track_geometries[second_code].path_d)
    closest_gap: float | None = None
    for first_start, first_end in zip(first_points, first_points[1:]):
        first_min_x = min(first_start[0], first_end[0])
        first_max_x = max(first_start[0], first_end[0])
        first_min_y = min(first_start[1], first_end[1])
        first_max_y = max(first_start[1], first_end[1])
        first_dx = abs(first_end[0] - first_start[0])
        first_dy = abs(first_end[1] - first_start[1])
        for second_start, second_end in zip(second_points, second_points[1:]):
            second_min_x = min(second_start[0], second_end[0])
            second_max_x = max(second_start[0], second_end[0])
            second_min_y = min(second_start[1], second_end[1])
            second_max_y = max(second_start[1], second_end[1])
            second_dx = abs(second_end[0] - second_start[0])
            second_dy = abs(second_end[1] - second_start[1])
            if first_dy < 1e-6 and second_dy < 1e-6:
                shared_span = min(first_max_x, second_max_x) - max(first_min_x, second_min_x)
                distance = abs(first_start[1] - second_start[1])
            else:
                continue
            if shared_span < min_shared_span:
                continue
            closest_gap = distance if closest_gap is None else min(closest_gap, distance)
    return closest_gap


def _dominant_horizontal_y(layout, track_code: str) -> float:
    points = _path_points(layout.track_geometries[track_code].path_d)
    best_length = -1.0
    best_y: float | None = None
    for start, end in zip(points, points[1:]):
        if abs(start[1] - end[1]) >= 1e-6:
            continue
        length = abs(end[0] - start[0])
        if length <= best_length:
            continue
        best_length = length
        best_y = start[1]
    if best_y is None:
        raise AssertionError(f"{track_code} has no horizontal segment")
    return best_y


def _count_label_overlaps(layout) -> int:
    boxes: list[tuple[str, float, float, float, float]] = []
    for track_code, geometry in layout.track_geometries.items():
        left, top, right, bottom = _track_text_boxes(track_code, geometry.label_anchor)[0]
        boxes.append((track_code, left, top, right, bottom))

    overlaps = 0
    for index, (_, left, top, right, bottom) in enumerate(boxes):
        for _, other_left, other_top, other_right, other_bottom in boxes[index + 1 :]:
            if min(right, other_right) <= max(left, other_left):
                continue
            if min(bottom, other_bottom) <= max(top, other_top):
                continue
            overlaps += 1
    return overlaps


def _label_gap(
    layout,
    *,
    first_track_code: str,
    second_track_code: str,
) -> float:
    first_left, first_top, first_right, first_bottom = _track_text_boxes(
        first_track_code,
        layout.track_geometries[first_track_code].label_anchor,
    )[0]
    second_left, second_top, second_right, second_bottom = _track_text_boxes(
        second_track_code,
        layout.track_geometries[second_track_code].label_anchor,
    )[0]
    dx = max(second_left - first_right, first_left - second_right, 0.0)
    dy = max(second_top - first_bottom, first_top - second_bottom, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = rect
    if abs(start[1] - end[1]) < 1e-6:
        y = start[1]
        seg_left = min(start[0], end[0])
        seg_right = max(start[0], end[0])
        return top <= y <= bottom and max(seg_left, left) < min(seg_right, right)
    if abs(start[0] - end[0]) < 1e-6:
        x = start[0]
        seg_top = min(start[1], end[1])
        seg_bottom = max(start[1], end[1])
        return left <= x <= right and max(seg_top, top) < min(seg_bottom, bottom)

    seg_left = min(start[0], end[0])
    seg_right = max(start[0], end[0])
    seg_top = min(start[1], end[1])
    seg_bottom = max(start[1], end[1])
    return max(seg_left, left) < min(seg_right, right) and max(seg_top, top) < min(seg_bottom, bottom)


def _point_to_rect_distance(
    point: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> float:
    left, top, right, bottom = rect
    dx = max(left - point[0], 0.0, point[0] - right)
    dy = max(top - point[1], 0.0, point[1] - bottom)
    return (dx * dx + dy * dy) ** 0.5


def _segment_rect_clearance(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
    *,
    samples: int = 400,
) -> float:
    best = inf
    for index in range(samples + 1):
        ratio = index / samples
        point = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        best = min(best, _point_to_rect_distance(point, rect))
    return best


def _label_clearance_to_track(
    layout,
    *,
    label_track_code: str,
    other_track_code: str,
) -> float:
    label_rect = _track_text_boxes(
        label_track_code,
        layout.track_geometries[label_track_code].label_anchor,
    )[0]
    points = _path_points(layout.track_geometries[other_track_code].path_d)
    return min(
        _segment_rect_clearance(start, end, label_rect)
        for start, end in zip(points, points[1:])
    )


def _count_partial_colinear_overlaps(layout) -> int:
    track_codes = list(layout.track_geometries)
    overlaps = 0
    for index, first_code in enumerate(track_codes):
        first_points = _path_points(layout.track_geometries[first_code].path_d)
        for second_code in track_codes[index + 1 :]:
            second_points = _path_points(layout.track_geometries[second_code].path_d)
            found = False
            for first_start, first_end in zip(first_points, first_points[1:]):
                for second_start, second_end in zip(second_points, second_points[1:]):
                    if abs(first_start[1] - first_end[1]) < 1e-6 and abs(second_start[1] - second_end[1]) < 1e-6:
                        if abs(first_start[1] - second_start[1]) >= 1e-6:
                            continue
                        first_left = min(first_start[0], first_end[0])
                        first_right = max(first_start[0], first_end[0])
                        second_left = min(second_start[0], second_end[0])
                        second_right = max(second_start[0], second_end[0])
                        if min(first_right, second_right) - max(first_left, second_left) > 5.0:
                            overlaps += 1
                            found = True
                            break
                    elif abs(first_start[0] - first_end[0]) < 1e-6 and abs(second_start[0] - second_end[0]) < 1e-6:
                        if abs(first_start[0] - second_start[0]) >= 1e-6:
                            continue
                        first_top = min(first_start[1], first_end[1])
                        first_bottom = max(first_start[1], first_end[1])
                        second_top = min(second_start[1], second_end[1])
                        second_bottom = max(second_start[1], second_end[1])
                        if min(first_bottom, second_bottom) - max(first_top, second_top) > 5.0:
                            overlaps += 1
                            found = True
                            break
                    else:
                        first_dx = first_end[0] - first_start[0]
                        first_dy = first_end[1] - first_start[1]
                        second_dx = second_end[0] - second_start[0]
                        second_dy = second_end[1] - second_start[1]
                        if abs(first_dx) < 1e-6 or abs(second_dx) < 1e-6:
                            continue
                        first_slope = first_dy / first_dx
                        second_slope = second_dy / second_dx
                        if abs(first_slope - second_slope) >= 1e-6:
                            continue
                        first_intercept = first_start[1] - first_slope * first_start[0]
                        second_intercept = second_start[1] - second_slope * second_start[0]
                        if abs(first_intercept - second_intercept) >= 1e-4:
                            continue
                        first_left = min(first_start[0], first_end[0])
                        first_right = max(first_start[0], first_end[0])
                        second_left = min(second_start[0], second_end[0])
                        second_right = max(second_start[0], second_end[0])
                        if min(first_right, second_right) - max(first_left, second_left) > 5.0:
                            overlaps += 1
                            found = True
                            break
                if found:
                    break
    return overlaps


def _count_group_overlaps(
    layout,
    *,
    track_codes: set[str],
) -> int:
    collisions = 0
    ordered_codes = sorted(track_codes)
    for index, first_code in enumerate(ordered_codes):
        first_geometry = layout.track_geometries[first_code]
        first_box = _track_text_boxes(first_code, first_geometry.label_anchor)[0]
        for second_code in ordered_codes[index + 1 :]:
            second_geometry = layout.track_geometries[second_code]
            second_box = _track_text_boxes(second_code, second_geometry.label_anchor)[0]
            if min(first_box[2], second_box[2]) <= max(first_box[0], second_box[0]):
                continue
            if min(first_box[3], second_box[3]) <= max(first_box[1], second_box[1]):
                continue
            collisions += 1
    return collisions


def test_segmented_routes_cover_expected_multi_segment_branches():
    master = load_master_data(DATA_DIR)

    routes = load_segmented_physical_routes(DATA_DIR, master)

    route = routes["L2-L12"]
    assert route.display_name == "存5北+存5南"
    assert [segment.track_code for segment in route.segments] == ["存5北", "存5南"]
    assert route.aggregate_physical_distance_m == pytest.approx(
        sum(segment.physical_distance_m for segment in route.segments),
        abs=0.1,
    )

    machine_route = routes["Z1-L8"]
    assert [segment.track_code for segment in machine_route.segments] == ["机北", "机棚"]

    repair_route = routes["L19-修1尽头"]
    assert [segment.track_code for segment in repair_route.segments] == ["修1库外", "修1库内"]

    for branch in routes.values():
        for segment in branch.segments:
            assert segment.track_code in master.tracks


def test_render_segmented_routes_svg_contains_expected_labels_and_paths():
    master = load_master_data(DATA_DIR)

    svg = render_segmented_routes_svg(DATA_DIR, master)

    assert svg.startswith("<svg")
    assert 'class="segmented-track"' in svg
    assert ">存5北<" in svg
    assert ">机棚<" in svg
    assert ">修1库内<" in svg
    assert "626.3m" in svg


def test_build_continuous_network_layout_keeps_anchor_tracks_on_expected_sides():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert layout.track_geometries["联6"].label_anchor.x > layout.track_geometries["存5北"].label_anchor.x
    assert layout.track_geometries["联6"].label_anchor.x > layout.track_geometries["机棚"].label_anchor.x
    assert layout.track_geometries["修1库内"].label_anchor.x < layout.track_geometries["联7"].label_anchor.x
    assert layout.track_geometries["修4库外"].label_anchor.x < layout.track_geometries["联7"].label_anchor.x


def test_build_continuous_network_layout_respects_ordered_branch_node_sequences():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert layout.node_positions["279"].x > layout.node_positions["L1"].x > layout.node_positions["L2"].x
    assert layout.node_positions["L6"].x > layout.node_positions["Z1"].x > layout.node_positions["Z2"].x
    assert layout.node_positions["Z2"].x > layout.node_positions["Z3"].x
    assert layout.node_positions["L7"].x > layout.node_positions["调棚北口"].x > layout.node_positions["调棚尽头"].x
    assert layout.node_positions["Z1"].x > layout.node_positions["机棚北口"].x > layout.node_positions["L8"].x
    assert layout.node_positions["L19"].x > layout.node_positions["修1门"].x > layout.node_positions["修1尽头"].x


def test_build_continuous_network_layout_shares_node_positions_for_connected_tracks():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    l1 = layout.node_positions["L1"]
    assert l1 in layout.track_geometries["联6"].endpoints
    assert l1 in layout.track_geometries["渡1"].endpoints

    storage_mid = layout.node_positions["存5中"]
    assert storage_mid in layout.track_geometries["存5北"].endpoints
    assert storage_mid in layout.track_geometries["存5南"].endpoints


def test_build_continuous_network_layout_prefers_horizontal_ribbon_shape():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    xs = [point.x for point in layout.node_positions.values()]
    ys = [point.y for point in layout.node_positions.values()]

    assert max(xs) - min(xs) > (max(ys) - min(ys)) * 2.0
    assert layout.canvas_width >= 4200.0


def test_build_continuous_network_layout_grows_height_and_cuts_crossings():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert layout.canvas_height > 760.0
    assert _count_non_touching_crossings(layout) <= 20


def test_build_continuous_network_layout_keeps_many_tracks_on_a_level_backbone():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    horizontal_track_count = 0
    for geometry in layout.track_geometries.values():
        points = _path_points(geometry.path_d)
        if any(
            abs(start[1] - end[1]) < 1e-6 and abs(start[0] - end[0]) >= 120.0
            for start, end in zip(points, points[1:])
        ):
            horizontal_track_count += 1

    assert horizontal_track_count >= 10


def test_build_continuous_network_layout_keeps_core_trunk_nodes_on_one_band():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    trunk_nodes = ["L13", "L14", "L15", "L16", "L17", "L18"]
    trunk_ys = [layout.node_positions[node_name].y for node_name in trunk_nodes]

    assert max(trunk_ys) - min(trunk_ys) <= 40.0


def test_build_continuous_network_layout_roughly_matches_reference_vertical_ordering():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert layout.track_geometries["存5北"].label_anchor.y < layout.track_geometries["存4北"].label_anchor.y
    assert layout.track_geometries["存4北"].label_anchor.y < layout.track_geometries["调北"].label_anchor.y
    assert layout.track_geometries["机北"].label_anchor.y < layout.track_geometries["调北"].label_anchor.y
    assert layout.track_geometries["修4库外"].label_anchor.y < layout.track_geometries["修3库外"].label_anchor.y
    assert layout.track_geometries["修3库外"].label_anchor.y < layout.track_geometries["修2库外"].label_anchor.y
    assert layout.track_geometries["修2库外"].label_anchor.y < layout.track_geometries["修1库外"].label_anchor.y
    assert layout.track_geometries["修1库外"].label_anchor.y < layout.track_geometries["轮"].label_anchor.y
    assert layout.track_geometries["洗北"].label_anchor.y > layout.track_geometries["机库"].label_anchor.y
    assert layout.track_geometries["油"].label_anchor.y > layout.track_geometries["机棚"].label_anchor.y


def test_build_continuous_network_layout_uses_compressed_length_sensitive_spans():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _track_span_x(layout, "联6") > 0.0
    assert _track_span_x(layout, "存5北") > _track_span_x(layout, "存5南")
    assert _track_span_x(layout, "调棚") > _track_span_x(layout, "调北")
    assert _track_span_x(layout, "修1库外") > _track_span_x(layout, "修1库内")
    assert _track_span_x(layout, "联6") < _track_span_x(layout, "存5北") < _track_span_x(layout, "联6") * 1.6


def test_build_continuous_network_layout_keeps_main_chain_x_monotonic():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    chain = ["轮", "渡11", "联7", "渡10", "渡9", "渡8", "存4南", "存4北", "渡1", "联6"]
    anchor_xs = [layout.track_geometries[track_code].label_anchor.x for track_code in chain]

    assert anchor_xs == sorted(anchor_xs)


def test_build_continuous_network_layout_separates_parallel_branches():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    left = layout.track_geometries["存2"].label_anchor
    right = layout.track_geometries["存3"].label_anchor

    assert abs(left.x - right.x) >= 120.0 or abs(left.y - right.y) >= 24.0


def test_render_segmented_routes_svg_uses_simple_orthogonal_branch_paths():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    svg = render_segmented_routes_svg(DATA_DIR, master)

    assert 'class="segmented-track"' in svg
    assert " C " not in svg
    assert layout.track_geometries["机北"].path_d.count(" L ") <= 3
    assert layout.track_geometries["调北"].path_d.count(" L ") <= 3


def test_build_continuous_network_layout_keeps_parallel_tracks_visually_separated():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _count_non_touching_parallel_near_overlaps(
        layout,
        clearance=14.0,
        min_shared_span=60.0,
    ) == 0


def test_build_continuous_network_layout_separates_known_middle_hotspots():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _closest_parallel_gap(layout, "存2", "机北", min_shared_span=120.0) >= 14.0
    assert _closest_parallel_gap(layout, "存1", "渡5", min_shared_span=60.0) >= 14.0
    assert _closest_parallel_gap(layout, "预修", "临4", min_shared_span=80.0) is None


def test_build_continuous_network_layout_aligns_machine_cluster_vertical_order():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    lane_ys = {
        track_code: _dominant_horizontal_y(layout, track_code)
        for track_code in ("预修", "存1", "机北", "机棚", "调北", "调棚", "机库", "油", "洗北", "洗南")
    }

    assert lane_ys["预修"] < lane_ys["存1"] < lane_ys["机北"] == lane_ys["机棚"]
    assert lane_ys["机北"] < lane_ys["调北"] == lane_ys["调棚"] < lane_ys["机库"]
    assert lane_ys["机库"] < lane_ys["油"] < lane_ys["洗北"] < lane_ys["洗南"]


def test_build_continuous_network_layout_uses_regular_machine_cluster_lane_steps():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    upper = _dominant_horizontal_y(layout, "预修")
    storage = _dominant_horizontal_y(layout, "存1")
    machine_north = _dominant_horizontal_y(layout, "机北")
    machine_shed = _dominant_horizontal_y(layout, "机棚")
    yard_north = _dominant_horizontal_y(layout, "调北")
    yard_shed = _dominant_horizontal_y(layout, "调棚")
    machine_house = _dominant_horizontal_y(layout, "机库")

    assert machine_north == pytest.approx(machine_shed)
    assert yard_north == pytest.approx(yard_shed)
    assert storage - upper == pytest.approx(ROUTE_LANE_SPACING * 2, abs=2.0)
    assert machine_north - storage == pytest.approx(ROUTE_LANE_SPACING * 2, abs=2.0)
    assert yard_north - machine_north == pytest.approx(ROUTE_LANE_SPACING * 2, abs=2.0)
    assert machine_house - yard_shed == pytest.approx(ROUTE_LANE_SPACING * 2, abs=2.0)


def test_build_continuous_network_layout_uses_wider_machine_cluster_level_gaps():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    upper = _dominant_horizontal_y(layout, "预修")
    storage = _dominant_horizontal_y(layout, "存1")
    machine = _dominant_horizontal_y(layout, "机北")
    yard = _dominant_horizontal_y(layout, "调北")
    machine_house = _dominant_horizontal_y(layout, "机库")

    assert storage - upper >= 52.0
    assert machine - storage >= 52.0
    assert yard - machine >= 52.0
    assert machine_house - yard >= 52.0


def test_build_continuous_network_layout_locks_machine_cluster_nodes_to_reference_levels():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    upper = _dominant_horizontal_y(layout, "预修")
    storage = _dominant_horizontal_y(layout, "存1")
    machine = _dominant_horizontal_y(layout, "机北")
    yard = _dominant_horizontal_y(layout, "调北")

    assert layout.node_positions["L13"].y == pytest.approx(upper, abs=1.0)
    assert layout.node_positions["Z3"].y == pytest.approx(upper, abs=1.0)
    assert layout.node_positions["L5"].y == pytest.approx(storage, abs=1.0)
    assert layout.node_positions["Z2"].y == pytest.approx(storage, abs=1.0)
    assert layout.node_positions["L6"].y == pytest.approx(machine, abs=1.0)
    assert layout.node_positions["Z1"].y == pytest.approx(machine, abs=1.0)
    assert layout.node_positions["机棚北口"].y == pytest.approx(machine, abs=1.0)
    assert layout.node_positions["L8"].y == pytest.approx(machine, abs=1.0)
    assert layout.node_positions["L7"].y == pytest.approx(yard, abs=1.0)
    assert layout.node_positions["调棚北口"].y == pytest.approx(yard, abs=1.0)


def test_build_continuous_network_layout_aligns_machine_cluster_service_nodes_on_one_pillar():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)
    pillar_xs = [
        layout.node_positions["机棚北口"].x,
        layout.node_positions["调棚北口"].x,
        layout.node_positions["机库尽头"].x,
    ]

    assert max(pillar_xs) - min(pillar_xs) <= 8.0


def test_build_continuous_network_layout_keeps_split_routes_on_one_fixed_lane():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _dominant_horizontal_y(layout, "存5北") == pytest.approx(_dominant_horizontal_y(layout, "存5南"))
    assert _dominant_horizontal_y(layout, "存4北") == pytest.approx(_dominant_horizontal_y(layout, "存4南"))
    assert _dominant_horizontal_y(layout, "机北") == pytest.approx(_dominant_horizontal_y(layout, "机棚"))
    assert _dominant_horizontal_y(layout, "调北") == pytest.approx(_dominant_horizontal_y(layout, "调棚"))


def test_build_continuous_network_layout_uses_direct_connectors_for_turnout_links():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    for track_code in layout.track_geometries:
        if not (track_code.startswith("渡") or track_code.startswith("临")):
            continue
        assert layout.track_geometries[track_code].path_d.count(" L ") == 1


def test_build_continuous_network_layout_avoids_duplicate_segments_at_l4_cluster():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert layout.track_geometries["渡3"].path_d != layout.track_geometries["存2"].path_d
    assert "L 3760.8 323.4 L 3760.8 330.3" not in layout.track_geometries["渡3"].path_d


def test_build_continuous_network_layout_avoids_partial_colinear_track_overlaps():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _count_partial_colinear_overlaps(layout) == 0


def test_build_continuous_network_layout_separates_track_labels():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _count_label_overlaps(layout) == 0


def test_build_continuous_network_layout_keeps_machine_cluster_text_clear_of_other_tracks():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _count_group_overlaps(
        layout,
        track_codes={"预修", "机棚", "机北", "调棚", "调北", "临4", "渡7", "渡6", "渡5", "临2", "机库", "渡4", "临3"},
    ) == 0


def test_build_continuous_network_layout_keeps_upper_storage_text_clear_of_other_tracks():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _count_group_overlaps(
        layout,
        track_codes={"存5南", "存5北", "存4南", "存4北", "存3", "存2", "渡8"},
    ) == 0


def test_build_continuous_network_layout_keeps_crowded_middle_labels_clear_of_neighbor_tracks():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _label_clearance_to_track(layout, label_track_code="临2", other_track_code="存1") >= 6.0
    assert _label_clearance_to_track(layout, label_track_code="渡6", other_track_code="存1") >= 6.0
    assert _label_clearance_to_track(layout, label_track_code="渡5", other_track_code="存1") >= 6.0
    assert _label_clearance_to_track(layout, label_track_code="油", other_track_code="洗北") >= 6.0
    assert _label_clearance_to_track(layout, label_track_code="临3", other_track_code="调棚") >= 6.0


def test_build_continuous_network_layout_polishes_right_middle_cluster_spacing():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _label_gap(layout, first_track_code="存1", second_track_code="临2") >= 40.0
    assert _label_clearance_to_track(layout, label_track_code="渡5", other_track_code="渡4") >= 10.0
    assert _label_clearance_to_track(layout, label_track_code="机北", other_track_code="渡7") >= 20.0


def test_build_continuous_network_layout_polishes_oil_and_wash_label_spacing():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _label_gap(layout, first_track_code="油", second_track_code="洗北") >= 12.0


def test_build_continuous_network_layout_polishes_repair_transfer_label_spacing():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _label_gap(layout, first_track_code="渡13", second_track_code="修2库外") >= 10.0


def test_build_continuous_network_layout_polishes_d4_d5_label_spacing():
    master = load_master_data(DATA_DIR)

    layout = build_continuous_network_layout(DATA_DIR, master)

    assert _label_gap(layout, first_track_code="渡4", second_track_code="渡5") >= 12.0


def test_export_segmented_route_artifacts_writes_json_and_svg(tmp_path: Path):
    master = load_master_data(DATA_DIR)

    result = export_segmented_route_artifacts(DATA_DIR, tmp_path, master)

    assert result["json_path"].exists()
    assert result["svg_path"].exists()
    assert result["json_path"].read_text(encoding="utf-8").strip().startswith("[")
    assert result["svg_path"].read_text(encoding="utf-8").strip().startswith("<svg")
