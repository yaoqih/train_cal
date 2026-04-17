from __future__ import annotations

from pathlib import Path
import json

from pydantic import BaseModel, Field


class Track(BaseModel):
    code: str
    name: str
    track_type: str
    effective_length_m: float
    allow_parking: bool
    allows_final_destination: bool
    description: str = ""
    endpoint_nodes: tuple[str, str] | tuple[()] = ()
    connection_nodes: tuple[str, ...] = ()
    terminal_branch: str | None = None
    reverse_branches: tuple[str, ...] = ()


class Spot(BaseModel):
    code: str
    track_code: str
    category: str
    capacity: int = 1


class Area(BaseModel):
    code: str
    track_code: str
    category: str
    spotting_required: bool = False


class PhysicalRoute(BaseModel):
    code: str
    total_length_m: float
    status: str
    description: str = ""
    left_node: str | None = None
    right_node: str | None = None


class BusinessRules(BaseModel):
    loco_length_m: float = 20.0
    require_clear_intermediate_path_tracks: bool = True


class MasterData(BaseModel):
    tracks: dict[str, Track] = Field(default_factory=dict)
    spots: dict[str, Spot] = Field(default_factory=dict)
    areas: dict[str, Area] = Field(default_factory=dict)
    physical_routes: dict[str, PhysicalRoute] = Field(default_factory=dict)
    business_rules: BusinessRules = Field(default_factory=BusinessRules)


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_master_data(base_dir: Path) -> MasterData:
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    tracks = {
        item["code"]: Track.model_validate(item)
        for item in _load_json(base_dir / "tracks.json")
    }
    spots = {
        item["code"]: Spot.model_validate(item)
        for item in _load_json(base_dir / "spots.json")
    }
    areas = {
        item["code"]: Area.model_validate(item)
        for item in _load_json(base_dir / "areas.json")
    }
    physical_routes = {
        item["code"]: PhysicalRoute.model_validate(item)
        for item in _load_json(base_dir / "physical_routes.json")
    }
    business_rules_path = base_dir / "business_rules.json"
    if business_rules_path.exists():
        business_rules = BusinessRules.model_validate(_load_json(business_rules_path))
    else:
        business_rules = BusinessRules()
    return MasterData(
        tracks=tracks,
        spots=spots,
        areas=areas,
        physical_routes=physical_routes,
        business_rules=business_rules,
    )
