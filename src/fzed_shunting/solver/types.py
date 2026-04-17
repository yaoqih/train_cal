from __future__ import annotations

from pydantic import BaseModel, Field


class HookAction(BaseModel):
    source_track: str
    target_track: str
    vehicle_nos: list[str] = Field(default_factory=list)
    path_tracks: list[str] = Field(default_factory=list)
    action_type: str = "PUT"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HookAction):
            return NotImplemented
        return self._ordering_key() < other._ordering_key()

    def _ordering_key(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...], str]:
        return (
            self.source_track,
            self.target_track,
            tuple(self.vehicle_nos),
            tuple(self.path_tracks),
            self.action_type,
        )


class SearchState(BaseModel):
    track_sequences: dict[str, list[str]] = Field(default_factory=dict)
    loco_track_name: str
