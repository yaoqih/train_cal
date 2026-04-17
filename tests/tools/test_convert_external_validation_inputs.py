import json
from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.tools.convert_external_validation_inputs import (
    PAIR_SPECS,
    ExcelVehicleRow,
    build_shared_vehicle_scenario,
    build_vehicle_attributes,
    convert_external_validation_inputs,
    load_length_m_by_model,
    map_excel_track_name,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_map_excel_track_name_maps_known_aliases():
    assert map_excel_track_name("存5线北") == "存5北"
    assert map_excel_track_name("老预修") == "预修"
    assert map_excel_track_name("调梁库内") == "调棚"
    assert map_excel_track_name("喷漆库外") == "油"
    assert map_excel_track_name("存4线") == "存4北"


def test_build_vehicle_attributes_extracts_heavy_and_maps_empty_to_linxiu():
    heavy_repair_process, heavy_vehicle_attributes = build_vehicle_attributes("重")
    empty_repair_process, empty_vehicle_attributes = build_vehicle_attributes("空")

    assert heavy_repair_process == "段修"
    assert heavy_vehicle_attributes == "重车"
    assert empty_repair_process == "临修"
    assert empty_vehicle_attributes == ""


def test_load_length_m_by_model_reads_enabled_models_from_length_sheet():
    length_by_model = load_length_m_by_model(
        Path(__file__).resolve().parents[2] / "段内车型换长.xlsx"
    )

    assert length_by_model["C70E"] == 14.3
    assert length_by_model["C64K"] == 13.2
    assert "GQ70" in length_by_model


def test_normalize_plan_input_accepts_explicit_track_target_mode():
    master = load_master_data(DATA_DIR)
    payload = {
        "trackInfo": [
            {"trackName": "存5北", "trackDistance": 367},
            {"trackName": "调棚", "trackDistance": 174.3},
        ],
        "vehicleInfo": [
            {
                "trackName": "存5北",
                "order": "1",
                "vehicleModel": "C70E",
                "vehicleNo": "E1",
                "repairProcess": "段修",
                "vehicleLength": 16.9,
                "targetMode": "TRACK",
                "targetTrack": "调棚",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        ],
        "locoTrackName": "机库",
    }

    result = normalize_plan_input(payload, master)

    vehicle = result.vehicles[0]
    assert vehicle.goal.target_mode == "TRACK"
    assert vehicle.goal.target_track == "调棚"
    assert vehicle.goal.allowed_target_tracks == ["调棚"]


def test_build_shared_vehicle_scenario_keeps_all_shared_vehicles_and_records_excluded():
    master = load_master_data(DATA_DIR)
    length_by_model = {"C70E": 14.3, "P64K": 12.1}
    start_rows = [
        ExcelVehicleRow("存5线北", 1, "C70E", "A1", "段", ""),
        ExcelVehicleRow("老预修", 2, "P64K", "A2", "厂", ""),
        ExcelVehicleRow("存5线北", 3, "C70E", "DROP1", "段", ""),
    ]
    end_rows = [
        ExcelVehicleRow("调梁库内", 1, "C70E", "A1", "段", ""),
        ExcelVehicleRow("老预修", 2, "P64K", "A2", "厂", ""),
        ExcelVehicleRow("存4线", 1, "C70E", "ADD1", "段", ""),
    ]

    scenario, summary = build_shared_vehicle_scenario(
        pair_spec=PAIR_SPECS[0],
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    assert scenario["name"] == PAIR_SPECS[0].scenario_name
    assert {item["vehicleNo"] for item in scenario["payload"]["vehicleInfo"]} == {"A1", "A2"}
    by_no = {item["vehicleNo"]: item for item in scenario["payload"]["vehicleInfo"]}
    assert by_no["A1"]["targetMode"] == "TRACK"
    assert by_no["A1"]["targetTrack"] == "调棚"
    assert "targetAreaCode" not in by_no["A1"]
    assert by_no["A2"]["targetTrack"] == "预修"
    assert summary["added_vehicle_nos"] == ["ADD1"]
    assert summary["removed_vehicle_nos"] == ["DROP1"]
    assert "调棚" in {item["trackName"] for item in scenario["payload"]["trackInfo"]}


def test_build_shared_vehicle_scenario_records_duplicate_vehicle_rows():
    master = load_master_data(DATA_DIR)
    length_by_model = {"C70E": 14.3}
    start_rows = [
        ExcelVehicleRow("存5线北", 1, "C70E", "A1", "段", ""),
    ]
    end_rows = [
        ExcelVehicleRow("卸轮线", 1, "C70E", "A1", "空", ""),
        ExcelVehicleRow("存4线", 1, "C70E", "A1", "重", ""),
    ]

    scenario, summary = build_shared_vehicle_scenario(
        pair_spec=PAIR_SPECS[0],
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    vehicle = scenario["payload"]["vehicleInfo"][0]
    assert vehicle["targetTrack"] == "存4北"
    assert summary["end_sheet_duplicate_vehicle_nos"] == ["A1"]


def test_build_shared_vehicle_scenario_uses_aggregate_depot_target_for_inner_depot_rows():
    master = load_master_data(DATA_DIR)
    length_by_model = {"C70E": 14.3}
    start_rows = [
        ExcelVehicleRow("存5线北", 1, "C70E", "A1", "段", ""),
    ]
    end_rows = [
        ExcelVehicleRow("修2库内", 201, "C70E", "A1", "段", ""),
    ]

    scenario, _summary = build_shared_vehicle_scenario(
        pair_spec=PAIR_SPECS[0],
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    vehicle = scenario["payload"]["vehicleInfo"][0]
    assert vehicle["targetMode"] == "AREA"
    assert vehicle["targetTrack"] == "大库"
    assert vehicle["targetAreaCode"] == "大库:RANDOM"


def test_convert_external_validation_inputs_writes_duplicate_metadata_to_summary(tmp_path: Path):
    output_dir = tmp_path / "external_validation_inputs"

    summary = convert_external_validation_inputs(
        output_dir=output_dir,
        source_xlsx=Path(__file__).resolve().parents[2] / "标准化起点终点模板（9.4-9.8-9.9）.xlsx",
        length_xlsx=Path(__file__).resolve().parents[2] / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    scenario = next(item for item in summary["scenarios"] if item["scenario_name"] == "validation_2025-09-04_am_to_pm")
    assert scenario["end_sheet_duplicate_vehicle_nos"] == ["4206309", "4872388"]


def test_convert_external_validation_inputs_includes_all_master_tracks(tmp_path: Path):
    output_dir = tmp_path / "external_validation_inputs"

    convert_external_validation_inputs(
        output_dir=output_dir,
        source_xlsx=Path(__file__).resolve().parents[2] / "标准化起点终点模板（9.4-9.8-9.9）.xlsx",
        length_xlsx=Path(__file__).resolve().parents[2] / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    payload = json.loads(
        (output_dir / "validation_2025-09-04_am_to_pm.json").read_text(encoding="utf-8")
    )
    track_names = {item["trackName"] for item in payload["trackInfo"]}
    master = load_master_data(DATA_DIR)

    assert track_names == set(master.tracks)
