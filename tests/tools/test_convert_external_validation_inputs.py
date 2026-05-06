import json
from pathlib import Path
from shutil import copy2

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.tools.convert_external_validation_inputs import (
    PairSpec,
    ExcelVehicleRow,
    build_pair_spec_for_workbook,
    build_data_pair_spec_for_workbook,
    build_shared_vehicle_scenario,
    build_vehicle_attributes,
    convert_external_validation_inputs,
    convert_data_external_validation_inputs,
    convert_online_validation_inputs,
    discover_data_plan_workbooks,
    discover_monthly_plan_workbooks,
    discover_online_plan_workbooks,
    load_length_m_by_model,
    load_data_vehicle_rows_by_sheet,
    load_online_vehicle_rows_by_workbook,
    load_supplemental_length_m_by_model,
    load_vehicle_rows_by_sheet,
    map_excel_track_name,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "master"
MONTHLY_PLAN_ROOT = ROOT_DIR / "取送车计划"
SAMPLE_WORKBOOK = MONTHLY_PLAN_ROOT / "1月-取送车计划" / "取送车计划_20260103W.xlsx"
SAMPLE_WORKBOOK_WITH_TEMPLATE_SHEET = MONTHLY_PLAN_ROOT / "2月-取送车计划" / "取送车计划_20260206W.xlsx"
MARCH_Z_WORKBOOK = MONTHLY_PLAN_ROOT / "3月-取送车计划" / "取送车计划_20260302Z.xlsx"
MARCH_26_Z_WORKBOOK = MONTHLY_PLAN_ROOT / "3月-取送车计划" / "取送车计划_20260326Z.xlsx"
DATA_PLAN_ROOT = MONTHLY_PLAN_ROOT / "Data"
DATA_MAP_XLSX = DATA_PLAN_ROOT / "map.xlsx"
DATA_WORKBOOK_WITH_LEADING_BLANK_ROW = DATA_PLAN_ROOT / "2025-11-06-noon.xlsx"
DATA_WORKBOOK_WITH_EXTRA_REPAIR_COLUMN = DATA_PLAN_ROOT / "2025-11-11-noon.xlsx"
DATA_WORKBOOK_WITH_EXTRA_SHEETS = DATA_PLAN_ROOT / "2025-09-04-noon.xlsx"
DATA_WORKBOOK_WITH_BACKFILL = DATA_PLAN_ROOT / "2025-11-04-afternoon.xlsx"
ONLINE_PLAN_ROOT = MONTHLY_PLAN_ROOT / "new"
ONLINE_SAMPLE_WORKBOOK = ONLINE_PLAN_ROOT / "调车计划编制_20260401W.xlsx"
ONLINE_WORKBOOK_WITH_RESIDUAL_SHEET = ONLINE_PLAN_ROOT / "调车计划编制_20260402W.xlsx"


def test_map_excel_track_name_maps_known_aliases():
    assert map_excel_track_name("存5北") == "存5北"
    assert map_excel_track_name("存4") == "存4北"
    assert map_excel_track_name("修1") == "修1库内"
    assert map_excel_track_name("修4") == "修4库内"
    assert map_excel_track_name("存5线北") == "存5北"
    assert map_excel_track_name("老预修") == "预修"
    assert map_excel_track_name("调梁库内") == "调棚"
    assert map_excel_track_name("调梁库") == "调棚"
    assert map_excel_track_name("调梁线南") == "调棚"
    assert map_excel_track_name("调梁线北") == "调北"
    assert map_excel_track_name("喷漆库外") == "油"
    assert map_excel_track_name("喷漆库") == "油"
    assert map_excel_track_name("喷漆线") == "油"
    assert map_excel_track_name("存4线") == "存4北"
    assert map_excel_track_name("机走北") == "机北"
    assert map_excel_track_name("机走南") == "机棚"
    assert map_excel_track_name("洗罐线北") == "洗北"
    assert map_excel_track_name("洗罐线南") == "洗南"
    assert map_excel_track_name("洗罐库内") == "洗南"
    assert map_excel_track_name("洗罐库外") == "洗北"
    assert map_excel_track_name("洗罐线") == "洗南"


def test_build_vehicle_attributes_extracts_heavy_and_maps_empty_to_linxiu():
    heavy_repair_process, heavy_vehicle_attributes = build_vehicle_attributes("重")
    empty_repair_process, empty_vehicle_attributes = build_vehicle_attributes("空")
    weigh_repair_process, weigh_vehicle_attributes = build_vehicle_attributes("称重")
    direct_factory_repair_process, direct_factory_vehicle_attributes = build_vehicle_attributes("厂修")
    direct_stage_repair_process, direct_stage_vehicle_attributes = build_vehicle_attributes("段修")

    assert heavy_repair_process == "段修"
    assert heavy_vehicle_attributes == "重车"
    assert empty_repair_process == "临修"
    assert empty_vehicle_attributes == ""
    assert weigh_repair_process == "段修"
    assert weigh_vehicle_attributes == "称重"
    assert direct_factory_repair_process == "厂修"
    assert direct_factory_vehicle_attributes == ""
    assert direct_stage_repair_process == "段修"
    assert direct_stage_vehicle_attributes == ""


def test_load_length_m_by_model_reads_enabled_models_from_length_sheet():
    length_by_model = load_length_m_by_model(ROOT_DIR / "段内车型换长.xlsx")

    assert length_by_model["C70E"] == 14.3
    assert length_by_model["C64K"] == 13.2
    assert "GQ70" in length_by_model


def test_load_supplemental_length_m_by_model_reads_explicit_aliases_from_huanchang_xlsx():
    length_by_model = load_supplemental_length_m_by_model(
        ROOT_DIR / "换长.xlsx",
        alias_overrides={
            "P64G": "P64GK",
            "NX70AK": "NX70AF",
        },
        literal_overrides={
            "NX17": 1.5,
        },
    )

    assert length_by_model["P64G"] == 16.5
    assert length_by_model["NX70AK"] == 14.3
    assert length_by_model["NX17"] == 16.5


def test_discover_monthly_plan_workbooks_returns_all_monthly_xlsx():
    workbooks = discover_monthly_plan_workbooks(MONTHLY_PLAN_ROOT)

    assert len(workbooks) == 109
    assert workbooks[0].name == "取送车计划_20260103W.xlsx"
    assert workbooks[-1].name == "取送车计划_20260331Z.xlsx"


def test_discover_data_plan_workbooks_returns_all_data_xlsx_without_map():
    workbooks = discover_data_plan_workbooks(DATA_PLAN_ROOT)

    assert len(workbooks) == 18
    assert workbooks[0].name == "2025-09-04-noon.xlsx"
    assert workbooks[-1].name == "2025-12-09-noon.xlsx"
    assert DATA_MAP_XLSX not in workbooks


def test_discover_online_plan_workbooks_returns_new_xlsx_files():
    workbooks = discover_online_plan_workbooks(ONLINE_PLAN_ROOT)

    assert len(workbooks) == 6
    assert workbooks[0].name == "调车计划编制_20260401W.xlsx"
    assert workbooks[-1].name == "调车计划编制_20260403Z.xlsx"


def test_load_online_vehicle_rows_by_workbook_reads_two_column_snapshot_and_skips_residual_sheet():
    rows, summary = load_online_vehicle_rows_by_workbook(ONLINE_WORKBOOK_WITH_RESIDUAL_SHEET)

    assert len(rows) == 81
    assert summary["ignored_sheets"] == ["起点 (2)"]
    assert rows[0].track_name == "预修"
    assert rows[0].vehicle_no == "5337588"
    assert rows[0].target_track == "修4"
    assert rows[0].note == "拉走"
    assert rows[-1].track_name == "存1"
    assert rows[-1].vehicle_no == "4639021"
    assert rows[-1].target_track == ""


def test_build_pair_spec_for_workbook_ignores_template_sheet():
    pair_spec = build_pair_spec_for_workbook(SAMPLE_WORKBOOK_WITH_TEMPLATE_SHEET)

    assert pair_spec.start_sheet == "2.6-下午-起"
    assert pair_spec.end_sheet == "2.9-上午-终"
    assert pair_spec.scenario_name == "validation_20260206W"


def test_load_vehicle_rows_by_sheet_accepts_monthly_plan_column_variants():
    january_rows = load_vehicle_rows_by_sheet(SAMPLE_WORKBOOK)
    february_rows = load_vehicle_rows_by_sheet(
        MONTHLY_PLAN_ROOT / "2月-取送车计划" / "取送车计划_20260202W.xlsx"
    )

    january_first = january_rows["1.3-下午-起"][0]
    february_first = february_rows["0202W-起点"][0]

    assert january_first == ExcelVehicleRow("存5线北", 1, "NX17KF", "5267378", "段", "")
    assert february_first == ExcelVehicleRow("老预修", 1, "K13NK", "5508258", "", "")


def test_load_data_vehicle_rows_by_sheet_detects_header_offset_and_extra_columns():
    blank_row_rows = load_data_vehicle_rows_by_sheet(DATA_WORKBOOK_WITH_LEADING_BLANK_ROW)
    extra_column_rows = load_data_vehicle_rows_by_sheet(DATA_WORKBOOK_WITH_EXTRA_REPAIR_COLUMN)

    assert blank_row_rows["Start"][0] == ExcelVehicleRow("老预修", 2, "C70", "1662620", "预修", "")
    assert extra_column_rows["End"][0] == ExcelVehicleRow("老预修", 2, "NX17KF", "5265641", "段", "")


def test_load_data_vehicle_rows_by_sheet_ignores_map_and_presentation_sheets():
    rows_by_sheet = load_data_vehicle_rows_by_sheet(DATA_WORKBOOK_WITH_EXTRA_SHEETS)

    assert set(rows_by_sheet) == {"Start", "End"}
    assert rows_by_sheet["Start"][0] == ExcelVehicleRow("老预修", 1, "C70E", "1851493", "段", "")


def test_build_data_pair_spec_for_workbook_uses_stem_based_scenario_name():
    pair_spec = build_data_pair_spec_for_workbook(DATA_WORKBOOK_WITH_BACKFILL)

    assert pair_spec.start_sheet == "Start"
    assert pair_spec.end_sheet == "End"
    assert pair_spec.scenario_name == "validation_2025_11_04_afternoon"


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
    assert vehicle.goal.target_mode == "WORK_POSITION"
    assert vehicle.goal.target_track == "调棚"
    assert vehicle.goal.allowed_target_tracks == ["调棚"]
    assert vehicle.goal.work_position_kind == "FREE"


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
        pair_spec=PairSpec("start", "end", "validation_sample"),
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    assert scenario["name"] == "validation_sample"
    assert {item["vehicleNo"] for item in scenario["payload"]["vehicleInfo"]} == {"A1", "A2"}
    by_no = {item["vehicleNo"]: item for item in scenario["payload"]["vehicleInfo"]}
    assert by_no["A1"]["targetMode"] == "SNAPSHOT"
    assert by_no["A1"]["targetTrack"] == "调棚"
    assert by_no["A1"]["targetSource"] == "END_SNAPSHOT"
    assert by_no["A2"]["targetTrack"] == "预修"
    assert summary["added_vehicle_nos"] == ["ADD1"]
    assert summary["removed_vehicle_nos"] == ["DROP1"]
    assert "调棚" in {item["trackName"] for item in scenario["payload"]["trackInfo"]}


def test_build_shared_vehicle_scenario_preserves_cun4bei_as_hard_departure_target():
    master = load_master_data(DATA_DIR)
    length_by_model = {"C70E": 14.3}
    start_rows = [
        ExcelVehicleRow("存5线北", 1, "C70E", "A1", "段", ""),
    ]
    end_rows = [
        ExcelVehicleRow("存4线", 1, "C70E", "A1", "段", ""),
    ]

    scenario, _summary = build_shared_vehicle_scenario(
        pair_spec=PairSpec("start", "end", "validation_sample"),
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    vehicle = scenario["payload"]["vehicleInfo"][0]
    assert vehicle["targetMode"] == "TRACK"
    assert vehicle["targetTrack"] == "存4北"
    assert "targetSource" not in vehicle


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
        pair_spec=PairSpec("start", "end", "validation_sample"),
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    vehicle = scenario["payload"]["vehicleInfo"][0]
    assert vehicle["targetTrack"] == "存4北"
    assert summary["end_sheet_duplicate_vehicle_nos"] == ["A1"]


def test_build_shared_vehicle_scenario_uses_snapshot_depot_target_for_inner_depot_rows():
    master = load_master_data(DATA_DIR)
    length_by_model = {"C70E": 14.3}
    start_rows = [
        ExcelVehicleRow("存5线北", 1, "C70E", "A1", "段", ""),
    ]
    end_rows = [
        ExcelVehicleRow("修2库内", 201, "C70E", "A1", "段", ""),
    ]

    scenario, _summary = build_shared_vehicle_scenario(
        pair_spec=PairSpec("start", "end", "validation_sample"),
        start_rows=start_rows,
        end_rows=end_rows,
        length_m_by_model=length_by_model,
        master=master,
    )

    vehicle = scenario["payload"]["vehicleInfo"][0]
    assert vehicle["targetMode"] == "SNAPSHOT"
    assert vehicle["targetTrack"] == "大库"
    assert vehicle["targetAreaCode"] == "大库:RANDOM"
    assert vehicle["targetSource"] == "END_SNAPSHOT"


def test_convert_external_validation_inputs_converts_each_monthly_workbook_to_one_scenario(tmp_path: Path):
    source_root = tmp_path / "取送车计划"
    january_dir = source_root / "1月-取送车计划"
    february_dir = source_root / "2月-取送车计划"
    january_dir.mkdir(parents=True)
    february_dir.mkdir(parents=True)
    copy2(SAMPLE_WORKBOOK, january_dir / SAMPLE_WORKBOOK.name)
    copy2(
        SAMPLE_WORKBOOK_WITH_TEMPLATE_SHEET,
        february_dir / SAMPLE_WORKBOOK_WITH_TEMPLATE_SHEET.name,
    )
    output_dir = tmp_path / "external_validation_inputs"

    summary = convert_external_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    assert summary["source_root"] == "取送车计划"
    assert summary["scenario_count"] == 2
    assert [item["scenario_file"] for item in summary["scenarios"]] == [
        "validation_20260103W.json",
        "validation_20260206W.json",
    ]
    assert summary["scenarios"][0]["source_workbook"] == "1月-取送车计划/取送车计划_20260103W.xlsx"
    assert summary["scenarios"][1]["start_sheet"] == "2.6-下午-起"
    assert summary["scenarios"][1]["end_sheet"] == "2.9-上午-终"

    payload = json.loads((output_dir / "validation_20260103W.json").read_text(encoding="utf-8"))
    track_names = {item["trackName"] for item in payload["trackInfo"]}
    master = load_master_data(DATA_DIR)

    assert payload["vehicleInfo"]
    assert track_names == set(master.tracks)


def test_convert_data_external_validation_inputs_backfills_shared_fields_and_writes_separate_summary(
    tmp_path: Path,
):
    source_root = tmp_path / "Data"
    source_root.mkdir(parents=True)
    copy2(DATA_WORKBOOK_WITH_BACKFILL, source_root / DATA_WORKBOOK_WITH_BACKFILL.name)
    copy2(DATA_MAP_XLSX, source_root / DATA_MAP_XLSX.name)
    output_dir = tmp_path / "external_validation_inputs" / "data"

    summary = convert_data_external_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    assert summary["source_root"] == "Data"
    assert summary["scenario_count"] == 1
    assert summary["scenarios"][0]["scenario_file"] == "validation_2025_11_04_afternoon.json"
    assert summary["scenarios"][0]["source_workbook"] == "2025-11-04-afternoon.xlsx"
    assert summary["scenarios"][0]["end_sheet_backfilled_model_count"] == 10
    assert summary["scenarios"][0]["end_sheet_backfilled_repair_count"] == 10

    payload = json.loads(
        (output_dir / "validation_2025_11_04_afternoon.json").read_text(encoding="utf-8")
    )
    by_no = {item["vehicleNo"]: item for item in payload["vehicleInfo"]}

    assert by_no["1528734"]["vehicleModel"] == "C70EH"
    assert by_no["1528734"]["repairProcess"] == "段修"
    assert by_no["1528734"]["targetTrack"] == "存4北"
    assert (output_dir / "conversion_assumptions.md").exists()


def test_convert_online_validation_inputs_writes_single_snapshot_plan_payloads(tmp_path: Path):
    source_root = tmp_path / "new"
    source_root.mkdir(parents=True)
    copy2(ONLINE_SAMPLE_WORKBOOK, source_root / ONLINE_SAMPLE_WORKBOOK.name)
    output_dir = tmp_path / "online"

    summary = convert_online_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    assert summary["source_root"] == "new"
    assert summary["scenario_count"] == 1
    assert summary["scenarios"][0]["vehicle_count"] == 93
    assert summary["scenarios"][0]["empty_target_count"] == 33
    assert summary["scenarios"][0]["scenario_file"] == "validation_20260401W.json"

    payload = json.loads((output_dir / "validation_20260401W.json").read_text(encoding="utf-8"))
    by_no = {item["vehicleNo"]: item for item in payload["vehicleInfo"]}

    assert by_no["5331984"]["trackName"] == "预修"
    assert "targetMode" not in by_no["5331984"]
    assert by_no["5331984"]["targetTrack"] == "修4"
    assert "targetAreaCode" not in by_no["5331984"]
    assert by_no["5313995"]["targetTrack"] == "修1"
    assert by_no["5313995"]["isSpotting"] == "104"
    assert "targetAreaCode" not in by_no["5313995"]
    assert by_no["5741082"]["targetTrack"] == "调棚"
    assert by_no["5741082"]["isSpotting"] == "是"
    assert by_no["1676076"]["targetTrack"] == "存4北"
    assert "targetMode" not in by_no["1676076"]
    assert by_no["5329931"]["targetMode"] == "SNAPSHOT"
    assert by_no["5329931"]["targetTrack"] == "洗南"
    assert by_no["5329931"]["targetSource"] == "ONLINE_EMPTY_TARGET"
    assert by_no["4972204"]["vehicleAttributes"] == "重车"

    master = load_master_data(DATA_DIR)
    normalize_plan_input(payload, master)
    assert (output_dir / "conversion_assumptions.md").exists()


def test_convert_external_validation_inputs_removes_stale_validation_jsons(tmp_path: Path):
    source_root = tmp_path / "取送车计划"
    january_dir = source_root / "1月-取送车计划"
    january_dir.mkdir(parents=True)
    copy2(SAMPLE_WORKBOOK, january_dir / SAMPLE_WORKBOOK.name)
    output_dir = tmp_path / "external_validation_inputs"
    output_dir.mkdir()
    stale_path = output_dir / "validation_2025-09-04_am_to_pm.json"
    stale_note_path = output_dir / "conversion_assumptions.md"
    stale_path.write_text("{}", encoding="utf-8")
    stale_note_path.write_text("stale", encoding="utf-8")

    convert_external_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    assert not stale_path.exists()
    assert not stale_note_path.exists()
    assert [path.name for path in sorted(output_dir.glob("validation_*.json"))] == [
        "validation_20260103W.json"
    ]


def test_convert_external_validation_inputs_uses_huanchang_xlsx_for_explicit_alias_supplements(
    tmp_path: Path,
):
    source_root = tmp_path / "取送车计划"
    march_dir = source_root / "3月-取送车计划"
    march_dir.mkdir(parents=True)
    copy2(MARCH_Z_WORKBOOK, march_dir / MARCH_Z_WORKBOOK.name)
    copy2(MARCH_26_Z_WORKBOOK, march_dir / MARCH_26_Z_WORKBOOK.name)
    output_dir = tmp_path / "external_validation_inputs"

    summary = convert_external_validation_inputs(
        output_dir=output_dir,
        source_root=source_root,
        length_xlsx=ROOT_DIR / "段内车型换长.xlsx",
        master_dir=DATA_DIR,
    )

    missing_by_scenario = {
        item["scenario_name"]: item["missing_models"] for item in summary["scenarios"]
    }

    assert missing_by_scenario["validation_20260302Z"] == []
    assert missing_by_scenario["validation_20260326Z"] == []
