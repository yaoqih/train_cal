from pathlib import Path

from fzed_shunting.domain.master_data import (
    BusinessRules,
    MasterData,
    PhysicalRoute,
    Track,
    load_master_data,
)
from fzed_shunting.domain.route_oracle import RouteOracle


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_route_oracle_expands_full_track_path_for_jiku_route():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    path_tracks = oracle.resolve_path_tracks("存5北", "机库")

    assert path_tracks == ["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"]

    result = oracle.validate_path(
        source_track="存5北",
        target_track="机库",
        path_tracks=path_tracks,
        train_length_m=50,
    )

    assert result.is_valid is True
    assert "L7-机库尽头" in result.branch_codes


def test_route_oracle_rejects_truncated_track_path_when_intermediate_tracks_exist():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="存5北",
        target_track="机库",
        path_tracks=["存5北", "机库"],
        train_length_m=50,
    )

    assert result.is_valid is False
    assert any("complete" in error.lower() for error in result.errors)


def test_route_oracle_accepts_oil_and_shot_branch_routes_when_missing_segment_is_40m_placeholder():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    shot_result = oracle.validate_path(
        source_track="联7",
        target_track="抛",
        path_tracks=["联7", "抛"],
        train_length_m=50,
    )
    oil_path = oracle.resolve_path_tracks("机库", "油")
    assert oil_path is not None
    oil_result = oracle.validate_path(
        source_track="机库",
        target_track="油",
        path_tracks=oil_path,
        train_length_m=50,
    )

    assert shot_result.is_valid is True
    assert oil_result.is_valid is True


def test_route_oracle_rejects_l1_length_overflow():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="存5北",
        target_track="存2",
        path_tracks=["存5北", "存2"],
        train_length_m=200,
    )

    assert result.is_valid is False
    assert any("190" in error for error in result.errors)


def test_route_oracle_uses_master_topology_for_custom_tracks():
    master = MasterData(
        tracks={
            "A": Track(
                code="A",
                name="A",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("A0", "J1"),
                connection_nodes=("J1",),
                terminal_branch="A0-J1",
            ),
            "B": Track(
                code="B",
                name="B",
                track_type="RUNNING",
                effective_length_m=20.0,
                allow_parking=False,
                allows_final_destination=False,
                endpoint_nodes=("J1", "J2"),
                connection_nodes=("J1", "J2"),
            ),
            "C": Track(
                code="C",
                name="C",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("J2", "C0"),
                connection_nodes=("J2",),
                terminal_branch="J2-C0",
            ),
        },
        physical_routes={
            "A0-J1": PhysicalRoute(
                code="A0-J1",
                total_length_m=40.0,
                status="已确认",
                left_node="A0",
                right_node="J1",
            ),
            "J1-J2": PhysicalRoute(
                code="J1-J2",
                total_length_m=20.0,
                status="已确认",
                left_node="J1",
                right_node="J2",
            ),
            "J2-C0": PhysicalRoute(
                code="J2-C0",
                total_length_m=40.0,
                status="已确认",
                left_node="J2",
                right_node="C0",
            ),
        },
    )
    oracle = RouteOracle(master)

    assert oracle.resolve_path_tracks("A", "C") == ["A", "B", "C"]

    route = oracle.resolve_route("A", "C")

    assert route is not None
    assert route.branch_codes == ["A0-J1", "J1-J2", "J2-C0"]


def test_route_oracle_returns_route_metrics():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="存5北",
        target_track="机库",
        path_tracks=["存5北", "渡1", "渡2", "临1", "临2", "渡4", "机库"],
        train_length_m=50,
    )

    assert result.is_valid is True
    assert result.total_length_m and result.total_length_m > 0
    assert result.uses_l1 is True


def test_route_oracle_rejects_terminal_branch_reverse_distance_overflow():
    master = MasterData(
        tracks={
            "A": Track(
                code="A",
                name="A",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("A0", "J1"),
                connection_nodes=("J1",),
            ),
            "B": Track(
                code="B",
                name="B",
                track_type="RUNNING",
                effective_length_m=20.0,
                allow_parking=False,
                allows_final_destination=False,
                endpoint_nodes=("J1", "J2"),
                connection_nodes=("J1", "J2"),
            ),
            "C": Track(
                code="C",
                name="C",
                track_type="WORK",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=False,
                endpoint_nodes=("J2", "C0"),
                connection_nodes=("J2",),
                terminal_branch="J2-C0",
            ),
        },
        physical_routes={
            "J1-J2": PhysicalRoute(
                code="J1-J2",
                total_length_m=20.0,
                status="已确认",
                left_node="J1",
                right_node="J2",
            ),
            "J2-C0": PhysicalRoute(
                code="J2-C0",
                total_length_m=30.0,
                status="已确认",
                left_node="J2",
                right_node="C0",
            ),
        },
        business_rules=BusinessRules(loco_length_m=20.0),
    )
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="A",
        target_track="C",
        path_tracks=["A", "B", "C"],
        train_length_m=15.0,
    )

    assert result.is_valid is False
    assert any("reverse" in error.lower() for error in result.errors)


def test_route_oracle_rejects_source_terminal_branch_reverse_distance_overflow():
    master = MasterData(
        tracks={
            "A": Track(
                code="A",
                name="A",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("A0", "J1"),
                connection_nodes=("J1",),
                terminal_branch="A0-J1",
            ),
            "B": Track(
                code="B",
                name="B",
                track_type="RUNNING",
                effective_length_m=20.0,
                allow_parking=False,
                allows_final_destination=False,
                endpoint_nodes=("J1", "J2"),
                connection_nodes=("J1", "J2"),
            ),
            "C": Track(
                code="C",
                name="C",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("J2", "C0"),
                connection_nodes=("J2",),
            ),
        },
        physical_routes={
            "A0-J1": PhysicalRoute(
                code="A0-J1",
                total_length_m=30.0,
                status="已确认",
                left_node="A0",
                right_node="J1",
            ),
            "J1-J2": PhysicalRoute(
                code="J1-J2",
                total_length_m=20.0,
                status="已确认",
                left_node="J1",
                right_node="J2",
            ),
        },
        business_rules=BusinessRules(loco_length_m=20.0),
    )
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="A",
        target_track="C",
        path_tracks=["A", "B", "C"],
        train_length_m=15.0,
    )

    assert result.is_valid is False
    assert "A0-J1" in result.reverse_branch_codes
    assert any("A0-J1" in error for error in result.errors)


def test_route_oracle_reports_source_and_target_reverse_branches():
    master = MasterData(
        tracks={
            "A": Track(
                code="A",
                name="A",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("A0", "J1"),
                connection_nodes=("J1",),
                terminal_branch="A0-J1",
            ),
            "B": Track(
                code="B",
                name="B",
                track_type="RUNNING",
                effective_length_m=20.0,
                allow_parking=False,
                allows_final_destination=False,
                endpoint_nodes=("J1", "J2"),
                connection_nodes=("J1", "J2"),
            ),
            "C": Track(
                code="C",
                name="C",
                track_type="WORK",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=False,
                endpoint_nodes=("J2", "C0"),
                connection_nodes=("J2",),
                terminal_branch="J2-C0",
            ),
        },
        physical_routes={
            "A0-J1": PhysicalRoute(
                code="A0-J1",
                total_length_m=40.0,
                status="已确认",
                left_node="A0",
                right_node="J1",
            ),
            "J1-J2": PhysicalRoute(
                code="J1-J2",
                total_length_m=20.0,
                status="已确认",
                left_node="J1",
                right_node="J2",
            ),
            "J2-C0": PhysicalRoute(
                code="J2-C0",
                total_length_m=40.0,
                status="已确认",
                left_node="J2",
                right_node="C0",
            ),
        },
        business_rules=BusinessRules(loco_length_m=20.0),
    )
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="A",
        target_track="C",
        path_tracks=["A", "B", "C"],
        train_length_m=15.0,
    )

    assert result.is_valid is True
    assert result.reverse_branch_codes == ["A0-J1", "J2-C0"]


def test_route_oracle_reports_interior_reverse_branch_from_track_metadata():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    path_tracks = oracle.resolve_path_tracks("调棚", "轮")

    assert path_tracks == ["调棚", "调北", "渡4", "渡5", "机北", "机棚", "临4", "渡10", "联7", "渡11", "轮"]

    result = oracle.validate_path(
        source_track="调棚",
        target_track="轮",
        path_tracks=path_tracks,
        train_length_m=50.0,
    )

    assert result.is_valid is True
    assert result.reverse_branch_codes == ["L7-调梁尽头", "Z1-L8", "L19-卸轮尽头"]


def test_route_oracle_reports_interior_reverse_branch_for_multiple_real_paths():
    master = load_master_data(DATA_DIR)
    oracle = RouteOracle(master)

    cases = [
        (
            "调棚",
            "修1库内",
            ["L7-调梁尽头", "Z1-L8", "L19-修1尽头"],
            True,
        ),
        (
            "机库",
            "油",
            ["L7-机库尽头", "Z1-L8", "L9-油漆尽头"],
            True,
        ),
        (
            "调棚",
            "油",
            ["L7-调梁尽头", "Z1-L8", "L9-油漆尽头"],
            True,
        ),
    ]

    for source_track, target_track, expected_reverse, expected_validity in cases:
        path_tracks = oracle.resolve_path_tracks(source_track, target_track)
        assert path_tracks is not None
        result = oracle.validate_path(
            source_track=source_track,
            target_track=target_track,
            path_tracks=path_tracks,
            train_length_m=50.0,
        )
        assert result.is_valid is expected_validity
        assert result.reverse_branch_codes == expected_reverse


def test_route_oracle_rejects_interior_reverse_branch_distance_overflow():
    master = MasterData(
        tracks={
            "A": Track(
                code="A",
                name="A",
                track_type="WORK",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=False,
                endpoint_nodes=("A0", "N1"),
                connection_nodes=("N1",),
                terminal_branch="A0-N1",
            ),
            "B": Track(
                code="B",
                name="B",
                track_type="WORK",
                effective_length_m=60.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("N1", "N2"),
                connection_nodes=("N1", "N2"),
                reverse_branches=("N1-N2",),
            ),
            "C": Track(
                code="C",
                name="C",
                track_type="STORAGE",
                effective_length_m=80.0,
                allow_parking=True,
                allows_final_destination=True,
                endpoint_nodes=("N2", "C0"),
                connection_nodes=("N2",),
                terminal_branch="N2-C0",
            ),
        },
        physical_routes={
            "A0-N1": PhysicalRoute(
                code="A0-N1",
                total_length_m=50.0,
                status="已确认",
                left_node="A0",
                right_node="N1",
            ),
            "N1-N2": PhysicalRoute(
                code="N1-N2",
                total_length_m=30.0,
                status="已确认",
                left_node="N1",
                right_node="N2",
            ),
            "N2-C0": PhysicalRoute(
                code="N2-C0",
                total_length_m=50.0,
                status="已确认",
                left_node="N2",
                right_node="C0",
            ),
        },
        business_rules=BusinessRules(loco_length_m=20.0),
    )
    oracle = RouteOracle(master)

    result = oracle.validate_path(
        source_track="A",
        target_track="C",
        path_tracks=["A", "B", "C"],
        train_length_m=15.0,
    )

    assert result.is_valid is False
    assert result.reverse_branch_codes == ["A0-N1", "N1-N2", "N2-C0"]
    assert any("N1-N2" in error for error in result.errors)
