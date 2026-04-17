from pathlib import Path

from fzed_shunting.demo.layout import build_route_polyline, load_topology_layout
from fzed_shunting.domain.master_data import load_master_data


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_load_topology_layout_builds_track_geometry_for_visible_tracks():
    master = load_master_data(DATA_DIR)

    layout = load_topology_layout(DATA_DIR, master)

    assert layout.canvas_width > 0
    assert layout.canvas_height > 0
    assert "L1" in layout.node_points
    assert "存5北" in layout.track_geometries
    assert len(layout.track_geometries["存5北"].points) >= 2
    assert layout.track_geometries["机库"].center.x > layout.track_geometries["存5北"].center.x


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
