"""Structural invariants for data/validation_inputs rule-scenario library."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "data" / "validation_inputs"
BASELINE_DIR = CORPUS_ROOT / "_baselines"
POSITIVE_DIR = CORPUS_ROOT / "positive"
NEGATIVE_DIR = CORPUS_ROOT / "negative"

ALLOWED_BASELINES = {
    "B1_clean",
    "B2_busy_storage",
    "B3_busy_yard_normal",
    "B4_busy_yard_inspection",
    "B5_busy_shed",
}

ALLOWED_ERROR_CATEGORIES = {
    "illegal_input",
    "capacity_overflow",
    "tow_limit_exceeded",
    "close_door_violation",
    "yard_allocation_violation",
}

FILENAME_RE = re.compile(
    r"^case_(\d+_\d+(?:_\d+)?)_([a-z0-9_]+)(?:_(\w+))?\.json$"
)
SPEC_SECTION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")


def _iter_cases(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("case_*.json"))


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("folder", [POSITIVE_DIR, NEGATIVE_DIR])
def test_case_filename_shape(folder: Path):
    for path in _iter_cases(folder):
        assert FILENAME_RE.match(path.name), (
            f"bad filename: {path.name}"
        )


@pytest.mark.parametrize("folder", [POSITIVE_DIR, NEGATIVE_DIR])
def test_case_metadata_shape(folder: Path):
    for path in _iter_cases(folder):
        payload = _load(path)
        assert isinstance(payload, dict), path
        meta = payload.get("metadata")
        assert isinstance(meta, dict), f"missing metadata in {path.name}"
        for field in ("spec_section", "rule_id", "variant", "baseline", "purpose"):
            assert field in meta, f"{path.name} missing metadata.{field}"
        assert SPEC_SECTION_RE.match(meta["spec_section"]), (
            f"{path.name} spec_section {meta['spec_section']} not NN.NN[.NN]"
        )
        assert meta["baseline"] in ALLOWED_BASELINES, (
            f"{path.name} references unknown baseline {meta['baseline']!r}"
        )


def test_positive_cases_have_expected_bounds():
    for path in _iter_cases(POSITIVE_DIR):
        meta = _load(path)["metadata"]
        assert "expected_bounds" in meta, (
            f"{path.name} missing metadata.expected_bounds"
        )
        assert "expected_error" not in meta, (
            f"{path.name} positive case must not have expected_error"
        )


def test_negative_cases_have_expected_error():
    for path in _iter_cases(NEGATIVE_DIR):
        meta = _load(path)["metadata"]
        assert "expected_error" in meta, (
            f"{path.name} missing metadata.expected_error"
        )
        err = meta["expected_error"]
        assert err.get("category") in ALLOWED_ERROR_CATEGORIES, (
            f"{path.name} unknown expected_error.category {err.get('category')!r}"
        )
        assert "expected_bounds" not in meta, (
            f"{path.name} negative case must not have expected_bounds"
        )


def test_vehicle_track_references_exist():
    for folder in (POSITIVE_DIR, NEGATIVE_DIR):
        for path in _iter_cases(folder):
            payload = _load(path)
            meta = payload["metadata"]
            if meta.get("allow_unknown_trackname") is True:
                continue
            tracks = {row["trackName"] for row in payload["trackInfo"]}
            missing = {
                row["trackName"]
                for row in payload["vehicleInfo"]
                if row.get("trackName") not in tracks
            }
            assert not missing, (
                f"{path.name} vehicleInfo refers to tracks not in trackInfo: {missing}"
            )


def test_case_identity_is_unique():
    seen: dict[tuple[str, str, str], Path] = {}
    for folder in (POSITIVE_DIR, NEGATIVE_DIR):
        for path in _iter_cases(folder):
            meta = _load(path)["metadata"]
            key = (meta["spec_section"], meta["rule_id"], meta["variant"])
            assert key not in seen, (
                f"duplicate case identity {key}: {seen[key]} vs {path}"
            )
            seen[key] = path


def test_baseline_files_exist_and_parse():
    assert BASELINE_DIR.exists(), "baselines dir must exist"
    for baseline in ALLOWED_BASELINES:
        path = BASELINE_DIR / f"{baseline}.json"
        assert path.exists(), f"baseline missing: {path}"
        payload = _load(path)
        assert "trackInfo" in payload and payload["trackInfo"], path
        assert "vehicleInfo" in payload, path
        assert payload.get("locoTrackName") is not None, path
