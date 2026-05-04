from pathlib import Path

import pytest

from fzed_shunting.domain.master_data import load_master_data


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_load_master_data_contains_core_tracks():
    master = load_master_data(DATA_DIR)

    assert "机库" in master.tracks
    assert "存4南" in master.tracks
    assert "联6" in master.tracks


def test_track_semantics_match_business_rules():
    master = load_master_data(DATA_DIR)

    assert master.tracks["机库"].allows_final_destination is True
    assert master.tracks["存4南"].allows_final_destination is False
    assert master.tracks["联6"].allow_parking is False


def test_special_spots_and_areas_exist():
    master = load_master_data(DATA_DIR)

    assert "机库:WEIGH" in master.spots
    assert "大库:RANDOM" in master.areas
    assert "调棚:WORK" not in master.areas
    assert "调棚:PRE_REPAIR" not in master.areas
    assert "洗南:WORK" not in master.areas
    assert "油:WORK" not in master.areas
    assert "抛:WORK" not in master.areas
    assert not any(spot.category == "WORK_GROUP" for spot in master.spots.values())


def test_oil_and_shot_physical_routes_use_40m_missing_segment_without_changing_track_capacity():
    master = load_master_data(DATA_DIR)

    assert master.physical_routes["L9-油漆尽头"].status == "已确认"
    assert master.physical_routes["L9-油漆尽头"].total_length_m == 209.0
    assert master.physical_routes["L15-抛丸尽头"].status == "已确认"
    assert master.physical_routes["L15-抛丸尽头"].total_length_m == 129.8
    assert master.tracks["油"].effective_length_m == 124.0
    assert master.tracks["抛"].effective_length_m == 131.8


def test_master_data_contains_track_and_branch_topology():
    master = load_master_data(DATA_DIR)

    assert master.tracks["机库"].endpoint_nodes == ("L7", "机库尽头")
    assert master.tracks["机库"].connection_nodes == ("L7",)
    assert master.tracks["机库"].terminal_branch == "L7-机库尽头"
    assert master.physical_routes["L6-L7"].left_node == "L6"
    assert master.physical_routes["L6-L7"].right_node == "L7"
    assert master.business_rules.loco_length_m > 0
    assert master.business_rules.require_clear_intermediate_path_tracks is True


def test_invalid_master_data_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_master_data(DATA_DIR / "missing")
