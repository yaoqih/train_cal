from __future__ import annotations

from pydantic import BaseModel, Field


class HookAction(BaseModel):
    source_track: str
    target_track: str
    vehicle_nos: list[str] = Field(default_factory=list)
    path_tracks: list[str] = Field(default_factory=list)
    action_type: str = "PUT"


class SearchState(BaseModel):
    track_sequences: dict[str, list[str]] = Field(default_factory=dict)
    loco_track_name: str
