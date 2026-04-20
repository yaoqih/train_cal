from pathlib import Path
import json

from fzed_shunting.demo.layout import build_route_polyline, load_topology_layout
from fzed_shunting.domain.master_data import load_master_data


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"
SCHEMATIC_LAYOUT_PATH = DATA_DIR / "schematic_layout.json"


def test_load_topology_layout_builds_track_geometry_for_visible_tracks():
    master = load_master_data(DATA_DIR)

    layout = load_topology_layout(DATA_DIR, master)

    assert layout.canvas_width > 0
    assert layout.canvas_height > 0
    assert "L1" in layout.node_points
    assert "存5北" in layout.track_geometries
    assert len(layout.track_geometries["存5北"].points) >= 2
    assert layout.track_geometries["机库"].center.x > layout.track_geometries["存5北"].center.x


def test_load_topology_layout_preserves_explicit_polyline_and_label_anchor():
    master = load_master_data(DATA_DIR)

    layout = load_topology_layout(DATA_DIR, master)
    geometry = layout.track_geometries["存4北"]

    assert len(geometry.points) >= 3
    assert geometry.points[0] == layout.node_points["L2"]
    assert geometry.points[-1] == layout.node_points["Z4"]
    assert geometry.label_anchor != geometry.center
    assert geometry.label_anchor.y < geometry.center.y


def test_load_topology_layout_exposes_background_image_metadata():
    master = load_master_data(DATA_DIR)

    layout = load_topology_layout(DATA_DIR, master)

    assert layout.background_image is not None
    assert layout.background_image.path == "../../image.png"
    assert layout.background_image.crop_box == (300, 650, 5200, 2400)


def test_load_topology_layout_exposes_background_anchor_for_reference_image_replay():
    master = load_master_data(DATA_DIR)

    layout = load_topology_layout(DATA_DIR, master)
    geometry = layout.track_geometries["存5北"]
    loco_geometry = layout.track_geometries["机库"]

    assert geometry.background_anchor is not None
    assert geometry.background_anchor.x > 1800
    assert geometry.background_anchor.y < 120
    assert loco_geometry.background_anchor is not None


def test_build_route_polyline_uses_track_centers_and_shared_nodes():
    master = load_master_data(DATA_DIR)
    layout = load_topology_layout(DATA_DIR, master)

    route = build_route_polyline(
        layout,
        ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
    )

    assert route.track_codes == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]
    assert route.total_length_px > 0
    assert route.points[0] == layout.track_geometries["存5北"].center
    assert route.points[1] == layout.node_points["L2"]
    assert route.points[-1] == layout.track_geometries["机库"].center


def test_schematic_layout_json_defines_mainline_areas_and_key_endpoints():
    assert SCHEMATIC_LAYOUT_PATH.exists()

    payload = json.loads(SCHEMATIC_LAYOUT_PATH.read_text(encoding="utf-8"))

    assert payload["canvas"]["width"] > 0
    assert payload["canvas"]["height"] > 0
    assert isinstance(payload.get("areas", []), list)
    assert [item["trackCode"] for item in payload["mainlineTracks"]][:4] == ["联6", "渡2", "临1", "临2"]
    assert "机库" in payload["tracks"]
    assert "洗南" in payload["tracks"]
