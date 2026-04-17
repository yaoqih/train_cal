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
DEFAULT_SOURCE_XLSX = Path(__file__).resolve().parents[3] / "标准化起点终点模板（9.4-9.8-9.9）.xlsx"
DEFAULT_LENGTH_XLSX = Path(__file__).resolve().parents[3] / "段内车型换长.xlsx"


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


PAIR_SPECS = [
    PairSpec("9.4上午（起点）", "9.4下午（终点）", "validation_2025-09-04_am_to_pm"),
    PairSpec("9.8上午（起点）", "9.8下午（终点）", "validation_2025-09-08_am_to_pm"),
    PairSpec("9.8下午（起点）", "9.9上午（终点）", "validation_2025-09-08_pm_to_09-09_am"),
    PairSpec("9.9上午（起点）", "9.9下午（终点）", "validation_2025-09-09_am_to_pm"),
]

TRACK_ALIAS_MAP = {
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
    "洗罐线内": "洗南",
    "洗罐线外": "洗北",
    "机走预修": "机棚",
    "老预修": "预修",
    "调梁库内": "调棚",
    "调梁库外": "调北",
}

DEPOT_INNER_TRACKS = {"修1库内", "修2库内", "修3库内", "修4库内"}

REPAIR_PROCESS_MAP = {
    "段": "段修",
    "厂": "厂修",
    "临": "临修",
    "空": "临修",
    "重": "段修",
}


def map_excel_track_name(track_name: str) -> str:
    try:
        return TRACK_ALIAS_MAP[track_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Excel track name: {track_name}") from exc


def build_vehicle_attributes(repair_value: str) -> tuple[str, str]:
    normalized = REPAIR_PROCESS_MAP.get(repair_value, "段修")
    if repair_value == "重":
        return normalized, "重车"
    return normalized, ""


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


def build_shared_vehicle_scenario(
    *,
    pair_spec: PairSpec,
    start_rows: list[ExcelVehicleRow],
    end_rows: list[ExcelVehicleRow],
    length_m_by_model: dict[str, float],
    master: MasterData,
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

    track_codes: set[str] = set(master.tracks)
    vehicle_info: list[dict] = []
    for vehicle_no in shared_vehicle_nos:
        start = start_by_no[vehicle_no]
        end = end_by_no[vehicle_no]
        current_track = map_excel_track_name(start.track_name)
        raw_target_track = map_excel_track_name(end.track_name)
        target_track = "大库" if raw_target_track in DEPOT_INNER_TRACKS else raw_target_track
        repair_process, vehicle_attributes = build_vehicle_attributes(start.repair)
        vehicle_length = length_m_by_model.get(start.vehicle_model)
        if vehicle_length is None:
            continue
        target_mode = "AREA" if raw_target_track in DEPOT_INNER_TRACKS else "TRACK"
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
    }
    return scenario, summary


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
        for row in rows[3:]:
            for base_index in (0, 6):
                cells = (row[base_index:base_index + 6] + [""] * 6)[:6]
                track_name, order, vehicle_model, vehicle_no, repair, note = cells
                if not vehicle_no:
                    continue
                vehicles.append(
                    ExcelVehicleRow(
                        track_name=track_name,
                        order=int(order),
                        vehicle_model=vehicle_model,
                        vehicle_no=vehicle_no,
                        repair=repair,
                        note=note,
                    )
                )
        vehicles_by_sheet[sheet_name] = vehicles
    return vehicles_by_sheet


def convert_external_validation_inputs(
    *,
    source_xlsx: Path = DEFAULT_SOURCE_XLSX,
    length_xlsx: Path = DEFAULT_LENGTH_XLSX,
    output_dir: Path,
    master_dir: Path = DEFAULT_MASTER_DIR,
) -> dict:
    master = load_master_data(master_dir)
    length_m_by_model = load_length_m_by_model(length_xlsx)
    rows_by_sheet = load_vehicle_rows_by_sheet(source_xlsx)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for pair_spec in PAIR_SPECS:
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
                "scenario_file": scenario_path.name,
            }
        )

    summary_payload = {
        "source_xlsx": source_xlsx.name,
        "length_xlsx": length_xlsx.name,
        "scenario_count": len(PAIR_SPECS),
        "scenarios": summaries,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
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
    source_xlsx: Path = typer.Option(DEFAULT_SOURCE_XLSX, exists=True, dir_okay=False),
    length_xlsx: Path = typer.Option(DEFAULT_LENGTH_XLSX, exists=True, dir_okay=False),
    master_dir: Path = typer.Option(DEFAULT_MASTER_DIR, exists=True, file_okay=False, dir_okay=True),
):
    summary = convert_external_validation_inputs(
        source_xlsx=source_xlsx,
        length_xlsx=length_xlsx,
        output_dir=output_dir,
        master_dir=master_dir,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    app()
