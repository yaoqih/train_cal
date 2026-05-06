from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

import typer

from fzed_shunting.domain.master_data import MasterData, load_master_data


app = typer.Typer(no_args_is_help=True)
DEFAULT_MASTER_DIR = Path(__file__).resolve().parents[3] / "data" / "master"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "取送车计划"
DEFAULT_DATA_SOURCE_ROOT = DEFAULT_SOURCE_ROOT / "Data"
DEFAULT_ONLINE_SOURCE_ROOT = DEFAULT_SOURCE_ROOT / "new"
DEFAULT_LENGTH_XLSX = Path(__file__).resolve().parents[3] / "段内车型换长.xlsx"
DEFAULT_SUPPLEMENTAL_LENGTH_XLSX = Path(__file__).resolve().parents[3] / "换长.xlsx"
MONTHLY_PLAN_SUBDIRS = ("1月-取送车计划", "2月-取送车计划", "3月-取送车计划")
MONTHLY_PLAN_TEMPLATE_MARKER = "模板"
MONTHLY_PLAN_HEADER_ROW_INDEX = 2
MONTHLY_PLAN_DATA_ROW_START_INDEX = 3
DATA_REQUIRED_SHEETS = ("Start", "End")
DATA_HEADER_SCAN_MAX_ROWS = 6
DATA_IGNORED_WORKBOOK_NAMES = {"map.xlsx"}
SUPPLEMENTAL_LENGTH_ALIASES = {
    "P64G": "P64GK",
    "NX70AK": "NX70AF",
}
SUPPLEMENTAL_LITERAL_LENGTHS = {
    "NX17": 1.5,
}


@dataclass(frozen=True)
class PairSpec:
    start_sheet: str
    end_sheet: str
    scenario_name: str


@dataclass(frozen=True)
class ExcelVehicleRow:
    track_name: str
    order: int
    vehicle_model: str
    vehicle_no: str
    repair: str
    note: str


@dataclass(frozen=True)
class OnlineVehicleRow(ExcelVehicleRow):
    target_track: str
    is_spotting: str
    vehicle_attributes: str


TRACK_ALIAS_MAP = {
    "存1": "存1",
    "存2": "存2",
    "存3": "存3",
    "存4": "存4北",
    "存4北": "存4北",
    "存5北": "存5北",
    "存5南": "存5南",
    "修1": "修1库内",
    "修2": "修2库内",
    "修3": "修3库内",
    "修4": "修4库内",
    "修1库内": "修1库内",
    "修2库内": "修2库内",
    "修3库内": "修3库内",
    "修4库内": "修4库内",
    "修1库外": "修1库外",
    "修2库外": "修2库外",
    "修3库外": "修3库外",
    "修4库外": "修4库外",
    "预修": "预修",
    "调棚": "调棚",
    "调北": "调北",
    "洗南": "洗南",
    "洗北": "洗北",
    "油": "油",
    "抛": "抛",
    "轮": "轮",
    "机棚": "机棚",
    "机库": "机库",
    "机北": "机北",
    "存1线": "存1",
    "存2线": "存2",
    "存3线": "存3",
    "存4线": "存4北",
    "存5线北": "存5北",
    "存5线南": "存5南",
    "机库线": "机库",
    "修1库内": "修1库内",
    "修2库内": "修2库内",
    "修3库内": "修3库内",
    "修4库内": "修4库内",
    "修1库外": "修1库外",
    "修2库外": "修2库外",
    "修3库外": "修3库外",
    "修4库外": "修4库外",
    "卸轮线": "轮",
    "抛丸线": "抛",
    "喷漆库内": "油",
    "喷漆库外": "油",
    "喷漆线": "油",
    "洗罐线内": "洗南",
    "洗罐线外": "洗北",
    "洗罐线北": "洗北",
    "洗罐线南": "洗南",
    "机走北": "机北",
    "机走南": "机棚",
    "机走预修": "机棚",
    "老预修": "预修",
    "调梁库": "调棚",
    "调梁库内": "调棚",
    "调梁库外": "调北",
    "调梁线北": "调北",
    "调梁线南": "调棚",
    "喷漆库": "油",
    "洗罐库内": "洗南",
    "洗罐库外": "洗北",
    "洗罐线": "洗南",
}

DEPOT_INNER_TRACKS = {"修1库内", "修2库内", "修3库内", "修4库内"}
ONLINE_SHORT_REPAIR_TARGETS = {"修1", "修2", "修3", "修4"}

REPAIR_PROCESS_MAP = {
    "段": "段修",
    "段修": "段修",
    "厂": "厂修",
    "厂修": "厂修",
    "临": "临修",
    "临修": "临修",
    "空": "临修",
    "重": "段修",
    "预修": "段修",
    "调梁": "段修",
    "机走": "段修",
    "卸轮": "段修",
    "认证": "段修",
    "鉴定": "段修",
    "待废": "段修",
    "代废": "段修",
    "对规车": "段修",
    "轮对": "段修",
}

VEHICLE_ATTRIBUTE_MAP = {
    "重": "重车",
    "称重": "称重",
}


def map_excel_track_name(track_name: str) -> str:
    try:
        return TRACK_ALIAS_MAP[track_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Excel track name: {track_name}") from exc


def map_online_target_track_name(track_name: str) -> str:
    if track_name in ONLINE_SHORT_REPAIR_TARGETS:
        return track_name
    return map_excel_track_name(track_name)


def build_vehicle_attributes(repair_value: str) -> tuple[str, str]:
    repair_value = (repair_value or "").strip()
    normalized = REPAIR_PROCESS_MAP.get(repair_value, "段修")
    return normalized, VEHICLE_ATTRIBUTE_MAP.get(repair_value, "")


def load_length_m_by_model(path: Path) -> dict[str, float]:
    rows = read_worksheet_rows(path)
    sheet_name = next(iter(rows))
    length_by_model: dict[str, float] = {}
    for row in rows[sheet_name][1:]:
        if len(row) < 5:
            continue
        model = row[1].strip()
        length_unit = row[2].strip()
        enabled = row[4].strip()
        if not model or enabled != "启用":
            continue
        length_by_model[model] = round(float(length_unit) * 11.0, 1)
    return length_by_model


def load_supplemental_length_m_by_model(
    path: Path,
    *,
    alias_overrides: dict[str, str],
    literal_overrides: dict[str, float],
) -> dict[str, float]:
    rows_by_sheet = read_worksheet_rows(path)
    best_by_model: dict[str, tuple[float, int]] = {}
    for rows in rows_by_sheet.values():
        for row in rows[1:]:
            if len(row) < 3:
                continue
            model = row[0].strip()
            length_unit = row[1].strip()
            count_raw = row[2].strip()
            if not model or not length_unit:
                continue
            try:
                length_value = float(length_unit)
                count_value = int(float(count_raw)) if count_raw else 0
            except ValueError:
                continue
            current = best_by_model.get(model)
            if current is None or count_value > current[1]:
                best_by_model[model] = (length_value, count_value)

    supplemental_lengths: dict[str, float] = {}
    for target_model, source_model in alias_overrides.items():
        if source_model not in best_by_model:
            continue
        supplemental_lengths[target_model] = round(best_by_model[source_model][0] * 11.0, 1)
    for target_model, length_unit in literal_overrides.items():
        supplemental_lengths[target_model] = round(length_unit * 11.0, 1)
    return supplemental_lengths


def build_shared_vehicle_scenario(
    *,
    pair_spec: PairSpec,
    start_rows: list[ExcelVehicleRow],
    end_rows: list[ExcelVehicleRow],
    length_m_by_model: dict[str, float],
    master: MasterData,
    merge_shared_fields_from_counterpart: bool = False,
) -> tuple[dict, dict]:
    start_duplicate_vehicle_nos = sorted(_duplicate_vehicle_nos(start_rows))
    end_duplicate_vehicle_nos = sorted(_duplicate_vehicle_nos(end_rows))
    start_by_no = {item.vehicle_no: item for item in start_rows}
    end_by_no = {item.vehicle_no: item for item in end_rows}
    shared_vehicle_nos = sorted(set(start_by_no) & set(end_by_no))
    added_vehicle_nos = sorted(set(end_by_no) - set(start_by_no))
    removed_vehicle_nos = sorted(set(start_by_no) - set(end_by_no))
    missing_models = sorted(
        {
            item.vehicle_model
            for item in list(start_rows) + list(end_rows)
            if item.vehicle_model and item.vehicle_model not in length_m_by_model
        }
    )
    start_sheet_backfilled_model_count = 0
    end_sheet_backfilled_model_count = 0
    start_sheet_backfilled_repair_count = 0
    end_sheet_backfilled_repair_count = 0
    skipped_vehicle_nos_missing_length: list[str] = []
    unresolved_shared_repair_vehicle_nos: list[str] = []

    track_codes: set[str] = set(master.tracks)
    vehicle_info: list[dict] = []
    for vehicle_no in shared_vehicle_nos:
        start = start_by_no[vehicle_no]
        end = end_by_no[vehicle_no]
        if merge_shared_fields_from_counterpart:
            start, start_backfills = _merge_excel_vehicle_row(start, end)
            end, end_backfills = _merge_excel_vehicle_row(end, start)
            start_sheet_backfilled_model_count += start_backfills["model"]
            start_sheet_backfilled_repair_count += start_backfills["repair"]
            end_sheet_backfilled_model_count += end_backfills["model"]
            end_sheet_backfilled_repair_count += end_backfills["repair"]
        current_track = map_excel_track_name(start.track_name)
        raw_target_track = map_excel_track_name(end.track_name)
        target_track = "大库" if raw_target_track in DEPOT_INNER_TRACKS else raw_target_track
        repair_process, vehicle_attributes = build_vehicle_attributes(start.repair)
        vehicle_length = length_m_by_model.get(start.vehicle_model)
        if not start.repair:
            unresolved_shared_repair_vehicle_nos.append(vehicle_no)
        if vehicle_length is None:
            skipped_vehicle_nos_missing_length.append(vehicle_no)
            continue
        target_mode = "TRACK" if raw_target_track == "存4北" else "SNAPSHOT"
        vehicle_payload = {
            "trackName": current_track,
            "order": str(start.order),
            "vehicleModel": start.vehicle_model,
            "vehicleNo": start.vehicle_no,
            "repairProcess": repair_process,
            "vehicleLength": vehicle_length,
            "targetMode": target_mode,
            "targetTrack": target_track,
            "isSpotting": "",
            "vehicleAttributes": vehicle_attributes,
        }
        if target_mode == "SNAPSHOT":
            vehicle_payload["targetSource"] = "END_SNAPSHOT"
        if raw_target_track in DEPOT_INNER_TRACKS:
            vehicle_payload["targetAreaCode"] = "大库:RANDOM"
        vehicle_info.append(vehicle_payload)
    track_info = [
        {
            "trackName": track_code,
            "trackDistance": master.tracks[track_code].effective_length_m,
        }
        for track_code in sorted(track_codes)
    ]
    scenario = {
        "name": pair_spec.scenario_name,
        "payload": {
            "trackInfo": track_info,
            "vehicleInfo": vehicle_info,
            "locoTrackName": "机库",
        },
    }
    summary = {
        "scenario_name": pair_spec.scenario_name,
        "start_sheet": pair_spec.start_sheet,
        "end_sheet": pair_spec.end_sheet,
        "shared_vehicle_count": len(vehicle_info),
        "added_vehicle_nos": added_vehicle_nos,
        "removed_vehicle_nos": removed_vehicle_nos,
        "start_sheet_duplicate_vehicle_nos": start_duplicate_vehicle_nos,
        "end_sheet_duplicate_vehicle_nos": end_duplicate_vehicle_nos,
        "missing_models": missing_models,
        "start_sheet_backfilled_model_count": start_sheet_backfilled_model_count,
        "end_sheet_backfilled_model_count": end_sheet_backfilled_model_count,
        "start_sheet_backfilled_repair_count": start_sheet_backfilled_repair_count,
        "end_sheet_backfilled_repair_count": end_sheet_backfilled_repair_count,
        "skipped_vehicle_nos_missing_length": skipped_vehicle_nos_missing_length,
        "unresolved_shared_repair_vehicle_nos": unresolved_shared_repair_vehicle_nos,
    }
    return scenario, summary


def build_online_vehicle_scenario(
    *,
    scenario_name: str,
    vehicle_rows: list[OnlineVehicleRow],
    length_m_by_model: dict[str, float],
    master: MasterData,
) -> tuple[dict, dict]:
    track_codes: set[str] = set(master.tracks)
    vehicle_info: list[dict] = []
    missing_models = sorted(
        {
            item.vehicle_model
            for item in vehicle_rows
            if item.vehicle_model and item.vehicle_model not in length_m_by_model
        }
    )
    skipped_vehicle_nos_missing_length: list[str] = []
    empty_target_vehicle_nos: list[str] = []
    normalized_target_counts: dict[str, int] = {}
    for vehicle_row in vehicle_rows:
        current_track = map_excel_track_name(vehicle_row.track_name)
        track_codes.add(current_track)
        target_source = "ONLINE_EXPLICIT_TARGET"
        if vehicle_row.target_track:
            raw_target_track = map_online_target_track_name(vehicle_row.target_track)
        else:
            raw_target_track = current_track
            target_source = "ONLINE_EMPTY_TARGET"
            empty_target_vehicle_nos.append(vehicle_row.vehicle_no)
        target_track = _online_payload_target_track(raw_target_track)
        repair_process, inferred_attributes = build_vehicle_attributes(vehicle_row.repair)
        vehicle_attributes = vehicle_row.vehicle_attributes or inferred_attributes
        vehicle_length = length_m_by_model.get(vehicle_row.vehicle_model)
        if vehicle_length is None:
            skipped_vehicle_nos_missing_length.append(vehicle_row.vehicle_no)
            continue
        normalized_target_counts[target_track] = normalized_target_counts.get(target_track, 0) + 1
        vehicle_payload = {
            "trackName": current_track,
            "order": str(vehicle_row.order),
            "vehicleModel": vehicle_row.vehicle_model,
            "vehicleNo": vehicle_row.vehicle_no,
            "repairProcess": repair_process,
            "vehicleLength": vehicle_length,
            "targetTrack": target_track,
            "isSpotting": vehicle_row.is_spotting,
            "vehicleAttributes": vehicle_attributes,
            "targetSource": target_source,
        }
        target_mode = _online_target_mode(raw_target_track, target_source)
        if target_mode is not None:
            vehicle_payload["targetMode"] = target_mode
        if target_track == "大库":
            vehicle_payload["targetAreaCode"] = "大库:RANDOM"
        vehicle_info.append(vehicle_payload)
    track_info = [
        {
            "trackName": track_code,
            "trackDistance": master.tracks[track_code].effective_length_m,
        }
        for track_code in sorted(track_codes)
    ]
    payload = {
        "trackInfo": track_info,
        "vehicleInfo": vehicle_info,
        "locoTrackName": "机库",
    }
    summary = {
        "scenario_name": scenario_name,
        "vehicle_count": len(vehicle_info),
        "empty_target_count": len(empty_target_vehicle_nos),
        "empty_target_vehicle_nos": empty_target_vehicle_nos,
        "missing_models": missing_models,
        "skipped_vehicle_nos_missing_length": skipped_vehicle_nos_missing_length,
        "normalized_target_counts": dict(sorted(normalized_target_counts.items())),
    }
    return payload, summary


def _online_payload_target_track(raw_target_track: str) -> str:
    return raw_target_track


def _online_target_mode(raw_target_track: str, target_source: str) -> str | None:
    if target_source == "ONLINE_EMPTY_TARGET":
        return "SNAPSHOT"
    return None


def discover_monthly_plan_workbooks(source_root: Path) -> list[Path]:
    workbooks: list[Path] = []
    for month_dir_name in MONTHLY_PLAN_SUBDIRS:
        month_dir = source_root / month_dir_name
        if not month_dir.exists():
            continue
        workbooks.extend(sorted(path for path in month_dir.glob("*.xlsx") if path.is_file()))
    return sorted(workbooks)


def build_pair_spec_for_workbook(path: Path, *, sheet_names: tuple[str, ...] | None = None) -> PairSpec:
    if sheet_names is None:
        sheet_names = tuple(read_worksheet_rows(path))
    data_sheet_names = tuple(
        name for name in sheet_names if MONTHLY_PLAN_TEMPLATE_MARKER not in name
    )
    if len(data_sheet_names) != 2:
        raise ValueError(
            f"{path.name} should contain exactly 2 non-template sheets, got {list(sheet_names)}"
        )
    scenario_suffix = path.stem.split("_", maxsplit=1)[-1]
    return PairSpec(
        start_sheet=data_sheet_names[0],
        end_sheet=data_sheet_names[1],
        scenario_name=f"validation_{scenario_suffix}",
    )


def discover_data_plan_workbooks(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.glob("*.xlsx")
        if path.is_file() and path.name not in DATA_IGNORED_WORKBOOK_NAMES
    )


def discover_online_plan_workbooks(source_root: Path) -> list[Path]:
    return sorted(path for path in source_root.glob("*.xlsx") if path.is_file())


def build_data_pair_spec_for_workbook(
    path: Path,
    *,
    sheet_names: tuple[str, ...] | None = None,
) -> PairSpec:
    if sheet_names is None:
        sheet_names = tuple(read_worksheet_rows(path))
    missing_sheet_names = [
        sheet_name for sheet_name in DATA_REQUIRED_SHEETS if sheet_name not in set(sheet_names)
    ]
    if missing_sheet_names:
        raise ValueError(
            f"{path.name} is missing required sheets: {', '.join(missing_sheet_names)}"
        )
    return PairSpec(
        start_sheet="Start",
        end_sheet="End",
        scenario_name=f"validation_{path.stem.replace('-', '_')}",
    )


def read_worksheet_rows(path: Path) -> dict[str, list[list[str]]]:
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_by_id = {item.attrib["Id"]: item.attrib["Target"] for item in rels}
        shared_strings = _read_shared_strings(archive, namespace)
        rows_by_sheet: dict[str, list[list[str]]] = {}
        for sheet in workbook.find("a:sheets", namespace):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            sheet_root = ET.fromstring(archive.read(f"xl/{rel_by_id[rid]}"))
            sheet_rows: list[list[str]] = []
            for row in sheet_root.findall(".//a:sheetData/a:row", namespace):
                values: dict[int, str] = {}
                for cell in row.findall("a:c", namespace):
                    ref = cell.attrib.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha())
                    values[_column_to_number(col)] = _read_cell_value(cell, shared_strings, namespace).strip()
                if values:
                    sheet_rows.append([values.get(index, "") for index in range(1, max(values) + 1)])
            rows_by_sheet[sheet.attrib["name"]] = sheet_rows
        return rows_by_sheet


def load_vehicle_rows_by_sheet(path: Path) -> dict[str, list[ExcelVehicleRow]]:
    rows_by_sheet = read_worksheet_rows(path)
    vehicles_by_sheet: dict[str, list[ExcelVehicleRow]] = {}
    for sheet_name, rows in rows_by_sheet.items():
        vehicles: list[ExcelVehicleRow] = []
        if len(rows) <= MONTHLY_PLAN_HEADER_ROW_INDEX:
            vehicles_by_sheet[sheet_name] = vehicles
            continue
        header_row = rows[MONTHLY_PLAN_HEADER_ROW_INDEX]
        if not header_row or len(header_row) % 2 != 0:
            raise ValueError(f"Unsupported monthly plan header layout in {path.name}::{sheet_name}")
        block_width = len(header_row) // 2
        block_headers = [
            [cell.strip() for cell in header_row[base_index:base_index + block_width]]
            for base_index in range(0, len(header_row), block_width)
        ]
        for row in rows[MONTHLY_PLAN_DATA_ROW_START_INDEX:]:
            for block_index, headers in enumerate(block_headers):
                base_index = block_index * block_width
                cells = (row[base_index:base_index + block_width] + [""] * block_width)[:block_width]
                vehicle = _build_vehicle_row_from_cells(
                    path=path,
                    sheet_name=sheet_name,
                    headers=headers,
                    cells=cells,
                )
                if vehicle is not None:
                    vehicles.append(vehicle)
        vehicles_by_sheet[sheet_name] = vehicles
    return vehicles_by_sheet


def load_data_vehicle_rows_by_sheet(path: Path) -> dict[str, list[ExcelVehicleRow]]:
    rows_by_sheet = read_worksheet_rows(path)
    vehicles_by_sheet: dict[str, list[ExcelVehicleRow]] = {}
    for sheet_name in DATA_REQUIRED_SHEETS:
        rows = rows_by_sheet.get(sheet_name)
        if rows is None:
            raise ValueError(f"{path.name} is missing required sheet: {sheet_name}")
        header_row_index, headers = _find_header_row(
            path=path,
            sheet_name=sheet_name,
            rows=rows,
        )
        vehicles: list[ExcelVehicleRow] = []
        for row in rows[header_row_index + 1:]:
            cells = (row + [""] * len(headers))[:len(headers)]
            vehicle = _build_vehicle_row_from_cells(
                path=path,
                sheet_name=sheet_name,
                headers=headers,
                cells=cells,
            )
            if vehicle is not None:
                vehicles.append(vehicle)
        vehicles_by_sheet[sheet_name] = vehicles
    return vehicles_by_sheet


def load_online_vehicle_rows_by_workbook(path: Path) -> tuple[list[OnlineVehicleRow], dict]:
    rows_by_sheet = read_worksheet_rows(path)
    expected_suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    vehicles: list[OnlineVehicleRow] = []
    ignored_sheets: list[str] = []
    used_sheets: list[str] = []
    for sheet_name, rows in rows_by_sheet.items():
        if not _online_sheet_matches_workbook_suffix(rows, expected_suffix):
            ignored_sheets.append(sheet_name)
            continue
        used_sheets.append(sheet_name)
        vehicles.extend(
            _load_online_vehicle_rows_from_sheet(
                path=path,
                sheet_name=sheet_name,
                rows=rows,
            )
        )
    duplicate_vehicle_nos = sorted(_duplicate_vehicle_nos(vehicles))
    if duplicate_vehicle_nos:
        raise ValueError(
            f"{path.name} has duplicate vehicleNo values after sheet filtering: "
            f"{', '.join(duplicate_vehicle_nos)}"
        )
    return vehicles, {
        "used_sheets": used_sheets,
        "ignored_sheets": ignored_sheets,
        "vehicle_count": len(vehicles),
    }


def convert_external_validation_inputs(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    length_xlsx: Path = DEFAULT_LENGTH_XLSX,
    supplemental_length_xlsx: Path = DEFAULT_SUPPLEMENTAL_LENGTH_XLSX,
    output_dir: Path,
    master_dir: Path = DEFAULT_MASTER_DIR,
) -> dict:
    master = load_master_data(master_dir)
    length_m_by_model = load_length_m_by_model(length_xlsx)
    if supplemental_length_xlsx.exists():
        length_m_by_model.update(
            load_supplemental_length_m_by_model(
                supplemental_length_xlsx,
                alias_overrides=SUPPLEMENTAL_LENGTH_ALIASES,
                literal_overrides=SUPPLEMENTAL_LITERAL_LENGTHS,
            )
        )
    workbooks = discover_monthly_plan_workbooks(source_root)
    if not workbooks:
        raise ValueError(f"No monthly plan workbooks found under {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("validation_*.json"):
        stale_path.unlink()
    for stale_path in (output_dir / "conversion_assumptions.md",):
        if stale_path.exists():
            stale_path.unlink()

    summaries: list[dict] = []
    for workbook in workbooks:
        rows_by_sheet = load_vehicle_rows_by_sheet(workbook)
        pair_spec = build_pair_spec_for_workbook(workbook, sheet_names=tuple(rows_by_sheet))
        scenario, summary = build_shared_vehicle_scenario(
            pair_spec=pair_spec,
            start_rows=rows_by_sheet[pair_spec.start_sheet],
            end_rows=rows_by_sheet[pair_spec.end_sheet],
            length_m_by_model=length_m_by_model,
            master=master,
        )
        scenario_path = output_dir / f"{pair_spec.scenario_name}.json"
        scenario_path.write_text(
            json.dumps(scenario["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(
            {
                **summary,
                "source_workbook": workbook.relative_to(source_root).as_posix(),
                "scenario_file": scenario_path.name,
            }
        )

    summary_payload = {
        "source_root": source_root.name,
        "length_xlsx": length_xlsx.name,
        "supplemental_length_xlsx": supplemental_length_xlsx.name if supplemental_length_xlsx.exists() else None,
        "scenario_count": len(summaries),
        "scenarios": summaries,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


def convert_online_validation_inputs(
    *,
    source_root: Path = DEFAULT_ONLINE_SOURCE_ROOT,
    length_xlsx: Path = DEFAULT_LENGTH_XLSX,
    supplemental_length_xlsx: Path = DEFAULT_SUPPLEMENTAL_LENGTH_XLSX,
    output_dir: Path,
    master_dir: Path = DEFAULT_MASTER_DIR,
) -> dict:
    master = load_master_data(master_dir)
    length_m_by_model = load_length_m_by_model(length_xlsx)
    if supplemental_length_xlsx.exists():
        length_m_by_model.update(
            load_supplemental_length_m_by_model(
                supplemental_length_xlsx,
                alias_overrides=SUPPLEMENTAL_LENGTH_ALIASES,
                literal_overrides=SUPPLEMENTAL_LITERAL_LENGTHS,
            )
        )
    workbooks = discover_online_plan_workbooks(source_root)
    if not workbooks:
        raise ValueError(f"No online plan workbooks found under {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("validation_*.json"):
        stale_path.unlink()
    for stale_path in (
        output_dir / "conversion_summary.json",
        output_dir / "conversion_assumptions.md",
    ):
        if stale_path.exists():
            stale_path.unlink()

    summaries: list[dict] = []
    for workbook in workbooks:
        rows, parse_summary = load_online_vehicle_rows_by_workbook(workbook)
        scenario_name = f"validation_{workbook.stem.rsplit('_', maxsplit=1)[-1]}"
        payload, summary = build_online_vehicle_scenario(
            scenario_name=scenario_name,
            vehicle_rows=rows,
            length_m_by_model=length_m_by_model,
            master=master,
        )
        scenario_path = output_dir / f"{scenario_name}.json"
        scenario_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(
            {
                **summary,
                **parse_summary,
                "source_workbook": workbook.relative_to(source_root).as_posix(),
                "scenario_file": scenario_path.name,
            }
        )

    summary_payload = {
        "source_root": source_root.name,
        "length_xlsx": length_xlsx.name,
        "supplemental_length_xlsx": supplemental_length_xlsx.name if supplemental_length_xlsx.exists() else None,
        "scenario_count": len(summaries),
        "scenarios": summaries,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "conversion_assumptions.md").write_text(
        _render_online_conversion_assumptions(summary_payload),
        encoding="utf-8",
    )
    return summary_payload


def convert_data_external_validation_inputs(
    *,
    source_root: Path = DEFAULT_DATA_SOURCE_ROOT,
    length_xlsx: Path = DEFAULT_LENGTH_XLSX,
    supplemental_length_xlsx: Path = DEFAULT_SUPPLEMENTAL_LENGTH_XLSX,
    output_dir: Path,
    master_dir: Path = DEFAULT_MASTER_DIR,
) -> dict:
    master = load_master_data(master_dir)
    length_m_by_model = load_length_m_by_model(length_xlsx)
    if supplemental_length_xlsx.exists():
        length_m_by_model.update(
            load_supplemental_length_m_by_model(
                supplemental_length_xlsx,
                alias_overrides=SUPPLEMENTAL_LENGTH_ALIASES,
                literal_overrides=SUPPLEMENTAL_LITERAL_LENGTHS,
            )
        )
    workbooks = discover_data_plan_workbooks(source_root)
    if not workbooks:
        raise ValueError(f"No data workbooks found under {source_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("validation_*.json"):
        stale_path.unlink()
    for stale_path in (
        output_dir / "conversion_summary.json",
        output_dir / "conversion_assumptions.md",
    ):
        if stale_path.exists():
            stale_path.unlink()

    summaries: list[dict] = []
    for workbook in workbooks:
        rows_by_sheet = load_data_vehicle_rows_by_sheet(workbook)
        pair_spec = build_data_pair_spec_for_workbook(workbook, sheet_names=tuple(rows_by_sheet))
        scenario, summary = build_shared_vehicle_scenario(
            pair_spec=pair_spec,
            start_rows=rows_by_sheet[pair_spec.start_sheet],
            end_rows=rows_by_sheet[pair_spec.end_sheet],
            length_m_by_model=length_m_by_model,
            master=master,
            merge_shared_fields_from_counterpart=True,
        )
        scenario_path = output_dir / f"{pair_spec.scenario_name}.json"
        scenario_path.write_text(
            json.dumps(scenario["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(
            {
                **summary,
                "source_workbook": workbook.relative_to(source_root).as_posix(),
                "scenario_file": scenario_path.name,
            }
        )

    summary_payload = {
        "source_root": source_root.name,
        "length_xlsx": length_xlsx.name,
        "supplemental_length_xlsx": supplemental_length_xlsx.name if supplemental_length_xlsx.exists() else None,
        "scenario_count": len(summaries),
        "scenarios": summaries,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "conversion_assumptions.md").write_text(
        _render_data_conversion_assumptions(summary_payload),
        encoding="utf-8",
    )
    return summary_payload


def _read_shared_strings(archive: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iterfind(".//a:t", namespace))
        for item in root.findall("a:si", namespace)
    ]


def _column_to_number(col: str) -> int:
    value = 0
    for char in col:
        if char.isalpha():
            value = value * 26 + ord(char.upper()) - 64
    return value


def _read_cell_value(cell: ET.Element, shared_strings: list[str], namespace: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        node = cell.find("a:is/a:t", namespace)
        return node.text if node is not None else ""
    value_node = cell.find("a:v", namespace)
    if value_node is None:
        return ""
    raw = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)]
    return raw


def _build_vehicle_row_from_cells(
    *,
    path: Path,
    sheet_name: str,
    headers: list[str],
    cells: list[str],
) -> ExcelVehicleRow | None:
    normalized_headers = {header.strip(): index for index, header in enumerate(headers)}
    required_headers = {"股道", "序号", "车型", "车号", "修程", "备注"}
    missing_headers = sorted(required_headers - set(normalized_headers))
    if missing_headers:
        raise ValueError(
            f"{path.name}::{sheet_name} is missing required headers: {', '.join(missing_headers)}"
    )
    vehicle_no = cells[normalized_headers["车号"]].strip()
    order_raw = cells[normalized_headers["序号"]].strip()
    if not vehicle_no or vehicle_no == "车号" or order_raw == "序号":
        return None
    if not order_raw:
        raise ValueError(f"{path.name}::{sheet_name} has vehicle {vehicle_no} without 序号")
    return ExcelVehicleRow(
        track_name=cells[normalized_headers["股道"]].strip(),
        order=int(float(order_raw)),
        vehicle_model=cells[normalized_headers["车型"]].strip(),
        vehicle_no=vehicle_no,
        repair=cells[normalized_headers["修程"]].strip(),
        note=cells[normalized_headers["备注"]].strip(),
    )


def _build_online_vehicle_row_from_cells(
    *,
    path: Path,
    sheet_name: str,
    headers: list[str],
    cells: list[str],
) -> OnlineVehicleRow | None:
    normalized_headers = {header.strip(): index for index, header in enumerate(headers)}
    required_headers = {"股道", "序号", "车型", "车号", "修程", "备注", "目标股道", "是否对位", "车辆属性"}
    missing_headers = sorted(required_headers - set(normalized_headers))
    if missing_headers:
        raise ValueError(
            f"{path.name}::{sheet_name} is missing required online headers: "
            f"{', '.join(missing_headers)}"
        )
    vehicle_no = cells[normalized_headers["车号"]].strip()
    order_raw = cells[normalized_headers["序号"]].strip()
    if not vehicle_no or vehicle_no == "车号" or order_raw == "序号":
        return None
    if not order_raw:
        raise ValueError(f"{path.name}::{sheet_name} has vehicle {vehicle_no} without 序号")
    return OnlineVehicleRow(
        track_name=cells[normalized_headers["股道"]].strip(),
        order=int(float(order_raw)),
        vehicle_model=cells[normalized_headers["车型"]].strip(),
        vehicle_no=vehicle_no,
        repair=cells[normalized_headers["修程"]].strip(),
        note=cells[normalized_headers["备注"]].strip(),
        target_track=cells[normalized_headers["目标股道"]].strip(),
        is_spotting=cells[normalized_headers["是否对位"]].strip(),
        vehicle_attributes=cells[normalized_headers["车辆属性"]].strip(),
    )


def _load_online_vehicle_rows_from_sheet(
    *,
    path: Path,
    sheet_name: str,
    rows: list[list[str]],
) -> list[OnlineVehicleRow]:
    header_row_index, headers = _find_online_header_row(
        path=path,
        sheet_name=sheet_name,
        rows=rows,
    )
    if len(headers) % 2 != 0:
        raise ValueError(f"Unsupported online plan header layout in {path.name}::{sheet_name}")
    block_width = len(headers) // 2
    block_headers = [
        [cell.strip() for cell in headers[base_index:base_index + block_width]]
        for base_index in range(0, len(headers), block_width)
    ]
    vehicles: list[OnlineVehicleRow] = []
    for row in rows[header_row_index + 1:]:
        for block_index, block_header in enumerate(block_headers):
            base_index = block_index * block_width
            cells = (row[base_index:base_index + block_width] + [""] * block_width)[:block_width]
            vehicle = _build_online_vehicle_row_from_cells(
                path=path,
                sheet_name=sheet_name,
                headers=block_header,
                cells=cells,
            )
            if vehicle is not None:
                vehicles.append(vehicle)
    return vehicles


def _find_online_header_row(
    *,
    path: Path,
    sheet_name: str,
    rows: list[list[str]],
) -> tuple[int, list[str]]:
    required_headers = {"股道", "序号", "车型", "车号", "修程", "备注", "目标股道", "是否对位", "车辆属性"}
    for row_index, row in enumerate(rows[:DATA_HEADER_SCAN_MAX_ROWS]):
        normalized_row = [cell.strip() for cell in row]
        row_values = {cell for cell in normalized_row if cell}
        if required_headers.issubset(row_values):
            return row_index, normalized_row
    raise ValueError(f"Unsupported online workbook header layout in {path.name}::{sheet_name}")


def _online_sheet_matches_workbook_suffix(rows: list[list[str]], expected_suffix: str) -> bool:
    title = rows[0][0].strip() if rows and rows[0] else ""
    if expected_suffix in title:
        return True
    if len(expected_suffix) != 9:
        return False
    date_part = expected_suffix[:8]
    period_code = expected_suffix[8]
    period = {"Z": "上午", "W": "下午"}.get(period_code)
    if period is None:
        return False
    dotted_date = f"{date_part[:4]}.{date_part[4:6]}.{date_part[6:8]}"
    return dotted_date in title and period in title


def _find_header_row(
    *,
    path: Path,
    sheet_name: str,
    rows: list[list[str]],
) -> tuple[int, list[str]]:
    required_headers = {"股道", "序号", "车型", "车号", "修程", "备注"}
    for row_index, row in enumerate(rows[:DATA_HEADER_SCAN_MAX_ROWS]):
        normalized_row = [cell.strip() for cell in row]
        if required_headers.issubset({cell for cell in normalized_row if cell}):
            return row_index, normalized_row
    raise ValueError(f"Unsupported data workbook header layout in {path.name}::{sheet_name}")


def _merge_excel_vehicle_row(
    primary: ExcelVehicleRow,
    counterpart: ExcelVehicleRow,
) -> tuple[ExcelVehicleRow, dict[str, int]]:
    backfill_counts = {
        "model": int(not primary.vehicle_model and bool(counterpart.vehicle_model)),
        "repair": int(not primary.repair and bool(counterpart.repair)),
    }
    return (
        ExcelVehicleRow(
            track_name=primary.track_name,
            order=primary.order,
            vehicle_model=primary.vehicle_model or counterpart.vehicle_model,
            vehicle_no=primary.vehicle_no,
            repair=primary.repair or counterpart.repair,
            note=primary.note or counterpart.note,
        ),
        backfill_counts,
    )


def _render_data_conversion_assumptions(summary_payload: dict) -> str:
    total_start_model_backfills = sum(
        item.get("start_sheet_backfilled_model_count", 0)
        for item in summary_payload.get("scenarios", [])
    )
    total_end_model_backfills = sum(
        item.get("end_sheet_backfilled_model_count", 0)
        for item in summary_payload.get("scenarios", [])
    )
    total_start_repair_backfills = sum(
        item.get("start_sheet_backfilled_repair_count", 0)
        for item in summary_payload.get("scenarios", [])
    )
    total_end_repair_backfills = sum(
        item.get("end_sheet_backfilled_repair_count", 0)
        for item in summary_payload.get("scenarios", [])
    )
    return "\n".join(
        [
            "# Data Conversion Assumptions",
            "",
            "- Only `Start` and `End` sheets are treated as scenario sources; `map` and presentation sheets are ignored.",
            "- The header row is detected by required columns `股道/序号/车型/车号/修程/备注` within the first six rows.",
            "- Old aliases are normalized as follows: `调梁库 -> 调棚`, `喷漆库 -> 油`, `洗罐库内 -> 洗南`, `洗罐库外 -> 洗北`, `洗罐线 -> 洗南`.",
            "- Raw repair values are normalized into the current contract with `称重 -> vehicleAttributes=称重`, `重 -> vehicleAttributes=重车`, direct `段修/厂修/临修` preserved, and other legacy labels folded into `段修`.",
            f"- Start-sheet model backfills from End: {total_start_model_backfills}",
            f"- End-sheet model backfills from Start: {total_end_model_backfills}",
            f"- Start-sheet repair backfills from End: {total_start_repair_backfills}",
            f"- End-sheet repair backfills from Start: {total_end_repair_backfills}",
        ]
    )


def _render_online_conversion_assumptions(summary_payload: dict) -> str:
    total_empty_targets = sum(
        item.get("empty_target_count", 0)
        for item in summary_payload.get("scenarios", [])
    )
    ignored_sheets = [
        f"{item.get('source_workbook')}: {', '.join(item.get('ignored_sheets', []))}"
        for item in summary_payload.get("scenarios", [])
        if item.get("ignored_sheets")
    ]
    lines = [
        "# Online Conversion Assumptions",
        "",
        "- Source workbooks under `取送车计划/new` are treated as single-snapshot online plan inputs, not paired Start/End snapshots.",
        "- Each data row with a `车号` becomes one `vehicleInfo` row; both left and right table blocks are parsed.",
        "- A sheet is used only when its title matches the workbook suffix, either directly as `20260402W` or by date/period such as `2026.04.02下午`; residual sheets with other dates are ignored.",
        "- Blank `目标股道` means the vehicle is kept in the payload as a current-track occupancy row with `targetMode=SNAPSHOT` and `targetSource=ONLINE_EMPTY_TARGET`.",
        "- Explicit online targets `修1/修2/修3/修4` are preserved as those short target names in JSON; normalization interprets them as the corresponding concrete inner-depot tracks.",
        "- `目标股道=存4` is normalized to `存4北` because `存4南` is not a legal final destination.",
        "- `车辆属性` from the workbook is preserved when present; otherwise legacy repair-derived attributes such as `重 -> 重车` are used.",
        f"- Total blank-target rows preserved as occupancy rows: {total_empty_targets}",
    ]
    if ignored_sheets:
        lines.extend(["", "## Ignored Sheets"])
        lines.extend(f"- {item}" for item in ignored_sheets)
    return "\n".join(lines)


def _duplicate_vehicle_nos(rows: list[ExcelVehicleRow]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row.vehicle_no in seen:
            duplicates.add(row.vehicle_no)
        seen.add(row.vehicle_no)
    return duplicates


@app.command("convert")
def convert_cmd(
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
    source_root: Path = typer.Option(DEFAULT_SOURCE_ROOT, exists=True, file_okay=False, dir_okay=True),
    length_xlsx: Path = typer.Option(DEFAULT_LENGTH_XLSX, exists=True, dir_okay=False),
    supplemental_length_xlsx: Path = typer.Option(DEFAULT_SUPPLEMENTAL_LENGTH_XLSX, exists=False, dir_okay=False),
    master_dir: Path = typer.Option(DEFAULT_MASTER_DIR, exists=True, file_okay=False, dir_okay=True),
):
    summary = convert_external_validation_inputs(
        source_root=source_root,
        length_xlsx=length_xlsx,
        supplemental_length_xlsx=supplemental_length_xlsx,
        output_dir=output_dir,
        master_dir=master_dir,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False))


@app.command("convert-data")
def convert_data_cmd(
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
    source_root: Path = typer.Option(DEFAULT_DATA_SOURCE_ROOT, exists=True, file_okay=False, dir_okay=True),
    length_xlsx: Path = typer.Option(DEFAULT_LENGTH_XLSX, exists=True, dir_okay=False),
    supplemental_length_xlsx: Path = typer.Option(DEFAULT_SUPPLEMENTAL_LENGTH_XLSX, exists=False, dir_okay=False),
    master_dir: Path = typer.Option(DEFAULT_MASTER_DIR, exists=True, file_okay=False, dir_okay=True),
):
    summary = convert_data_external_validation_inputs(
        source_root=source_root,
        length_xlsx=length_xlsx,
        supplemental_length_xlsx=supplemental_length_xlsx,
        output_dir=output_dir,
        master_dir=master_dir,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False))


@app.command("convert-online")
def convert_online_cmd(
    output_dir: Path = typer.Option(..., exists=False, file_okay=False, dir_okay=True),
    source_root: Path = typer.Option(DEFAULT_ONLINE_SOURCE_ROOT, exists=True, file_okay=False, dir_okay=True),
    length_xlsx: Path = typer.Option(DEFAULT_LENGTH_XLSX, exists=True, dir_okay=False),
    supplemental_length_xlsx: Path = typer.Option(DEFAULT_SUPPLEMENTAL_LENGTH_XLSX, exists=False, dir_okay=False),
    master_dir: Path = typer.Option(DEFAULT_MASTER_DIR, exists=True, file_okay=False, dir_okay=True),
):
    summary = convert_online_validation_inputs(
        source_root=source_root,
        length_xlsx=length_xlsx,
        supplemental_length_xlsx=supplemental_length_xlsx,
        output_dir=output_dir,
        master_dir=master_dir,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    app()
