from __future__ import annotations

from pydantic import BaseModel, Field

from fzed_shunting.domain.work_positions import WORK_POSITION_TRACKS
from fzed_shunting.domain.master_data import MasterData


class InputValidationError(ValueError):
    """Raised when the input contract is invalid."""


class GoalSpec(BaseModel):
    target_mode: str
    target_track: str
    allowed_target_tracks: list[str] = Field(default_factory=list)
    preferred_target_tracks: list[str] = Field(default_factory=list)
    fallback_target_tracks: list[str] = Field(default_factory=list)
    target_area_code: str | None = None
    target_spot_code: str | None = None
    target_source: str | None = None
    work_position_kind: str | None = None
    target_rank: int | None = None


class NormalizedVehicle(BaseModel):
    current_track: str
    order: int
    vehicle_model: str
    vehicle_no: str
    repair_process: str
    vehicle_length: float
    goal: GoalSpec
    need_weigh: bool = False
    is_heavy: bool = False
    is_close_door: bool = False
    raw_vehicle_attributes: str | None = None
    raw_is_spotting: str | None = None


class NormalizedTrackInfo(BaseModel):
    track_name: str
    track_distance: float


class NormalizedPlanInput(BaseModel):
    track_info: list[NormalizedTrackInfo] = Field(default_factory=list)
    vehicles: list[NormalizedVehicle] = Field(default_factory=list)
    loco_track_name: str
    yard_mode: str
    single_end_track_names: frozenset[str] = Field(default_factory=frozenset)


WORK_AREA_DEFAULTS = {
    "轮": "轮:OPERATE",
    "大库": "大库:RANDOM",
    "大库外": "大库外:RANDOM",
    "修1库内": "大库:RANDOM",
    "修2库内": "大库:RANDOM",
    "修3库内": "大库:RANDOM",
    "修4库内": "大库:RANDOM",
}

TRACK_MODE_ALLOWED = {
    "存1",
    "存2",
    "存3",
    "存5北",
    "存5南",
    "机北",
    "调北",
    "洗北",
    "预修",
    "机棚",
    "机库",
    "存4北",
    "修1库外",
    "修2库外",
    "修3库外",
    "修4库外",
}

AREA_ALLOWED_TRACKS = {
    "轮:OPERATE": ["轮"],
    "大库:RANDOM": ["修1库内", "修2库内", "修3库内", "修4库内"],
    "大库外:RANDOM": ["修1库外", "修2库外", "修3库外", "修4库外"],
    "接车:RANDOM": ["存5北", "存5南"],
    "存车:RANDOM": [
        "存1",
        "存2",
        "存3",
        "机北",
        "调北",
        "洗北",
    ],
    "预修:RANDOM": ["预修", "机棚", "调棚"],
    "洗罐:RANDOM": ["洗南", "洗北"],
    "调棚:SNAPSHOT": ["调棚"],
    "油:SNAPSHOT": ["油"],
    "抛:SNAPSHOT": ["抛"],
    "轮:SNAPSHOT": ["轮"],
}

SNAPSHOT_AREA_CODE_BY_TRACK = {
    "存5北": "接车:RANDOM",
    "存5南": "接车:RANDOM",
    "存1": "存车:RANDOM",
    "存2": "存车:RANDOM",
    "存3": "存车:RANDOM",
    "机北": "存车:RANDOM",
    "调北": "存车:RANDOM",
    "洗北": "存车:RANDOM",
    "修1库外": "大库外:RANDOM",
    "修2库外": "大库外:RANDOM",
    "修3库外": "大库外:RANDOM",
    "修4库外": "大库外:RANDOM",
    "预修": "预修:RANDOM",
    "机棚": "预修:RANDOM",
    "调棚": "预修:RANDOM",
    "洗南": "洗罐:RANDOM",
    "油": "油:SNAPSHOT",
    "抛": "抛:SNAPSHOT",
    "轮": "轮:SNAPSHOT",
    "修1库内": "大库:RANDOM",
    "修2库内": "大库:RANDOM",
    "修3库内": "大库:RANDOM",
    "修4库内": "大库:RANDOM",
}

DEPOT_INNER_PREFERRED_TRACKS_SHORT = ["修1库内", "修2库内"]
DEPOT_INNER_FALLBACK_TRACKS_SHORT = ["修3库内", "修4库内"]
DEPOT_INNER_PREFERRED_TRACKS_LONG = ["修3库内", "修4库内"]

VALID_IS_SPOTTING_LITERALS = {"", "否", "是", "迎检"}
FORBIDDEN_FINAL_AREA_CODES = {"机库:WEIGH"}
REMOVED_WORK_AREA_CODES = {
    "调棚:WORK",
    "调棚:PRE_REPAIR",
    "洗南:WORK",
    "油:WORK",
    "抛:WORK",
}


def _random_depot_track_preferences(vehicle_length: float) -> tuple[list[str], list[str]]:
    if vehicle_length >= 17.6:
        return (list(DEPOT_INNER_PREFERRED_TRACKS_LONG), [])
    return (list(DEPOT_INNER_PREFERRED_TRACKS_SHORT), list(DEPOT_INNER_FALLBACK_TRACKS_SHORT))


def normalize_plan_input(
    payload: dict,
    master: MasterData,
    *,
    allow_internal_loco_tracks: bool = False,
) -> NormalizedPlanInput:
    yard_mode = "NORMAL"
    loco_track_name = payload.get("locoTrackName") or "机库"
    if allow_internal_loco_tracks:
        if loco_track_name not in master.tracks:
            raise InputValidationError(f"Unknown locoTrackName: {loco_track_name}")
    elif loco_track_name not in {"机库", "机北"}:
        raise InputValidationError(f"Unsupported locoTrackName: {loco_track_name}")
    inspection_enabled = any(
        (item.get("isSpotting") or "").strip() == "迎检"
        for item in payload.get("vehicleInfo", [])
    )
    track_info = [
        NormalizedTrackInfo(
            track_name=item["trackName"],
            track_distance=float(item["trackDistance"]),
        )
        for item in payload.get("trackInfo", [])
    ]
    vehicles: list[NormalizedVehicle] = []
    seen_orders: set[tuple[str, int]] = set()
    for raw in payload.get("vehicleInfo", []):
        order = int(raw["order"])
        order_key = (raw["trackName"], order)
        if order_key in seen_orders:
            raise InputValidationError(
                f"Duplicate order on track {raw['trackName']}: {order}"
            )
        seen_orders.add(order_key)
        if raw["trackName"] not in master.tracks:
            raise InputValidationError(f"Unknown source track: {raw['trackName']}")
        source_track = master.tracks[raw["trackName"]]
        if source_track.track_type == "RUNNING":
            raise InputValidationError(
                f"Running track cannot be source track: {raw['trackName']}"
            )
        goal, vehicle_yard_mode = _normalize_goal(raw, master, inspection_enabled=inspection_enabled)
        yard_mode = _merge_yard_mode(yard_mode, vehicle_yard_mode)
        need_weigh, is_heavy, is_close_door = _parse_vehicle_attribute(
            raw.get("vehicleAttributes", "")
        )
        vehicles.append(
            NormalizedVehicle(
                current_track=raw["trackName"],
                order=order,
                vehicle_model=raw["vehicleModel"],
                vehicle_no=raw["vehicleNo"],
                repair_process=raw["repairProcess"],
                vehicle_length=float(raw["vehicleLength"]),
                goal=goal,
                need_weigh=need_weigh,
                is_heavy=is_heavy,
                is_close_door=is_close_door,
                raw_vehicle_attributes=raw.get("vehicleAttributes", ""),
                raw_is_spotting=raw.get("isSpotting", ""),
            )
        )
    return NormalizedPlanInput(
        track_info=track_info,
        vehicles=vehicles,
        loco_track_name=loco_track_name,
        yard_mode=yard_mode,
        single_end_track_names=frozenset(
            code for code, track in master.tracks.items() if len(track.connection_nodes) == 1
        ),
    )


def _validate_is_spotting_literal(raw_value: str) -> None:
    value = (raw_value or "").strip()
    if not value:
        return
    if value in VALID_IS_SPOTTING_LITERALS:
        return
    if value.isdigit():
        return
    raise InputValidationError(f"Unsupported isSpotting value: {value}")


def _normalize_goal(
    raw: dict,
    master: MasterData,
    inspection_enabled: bool = False,
) -> tuple[GoalSpec, str]:
    _validate_is_spotting_literal(raw.get("isSpotting", ""))
    explicit_mode = (raw.get("targetMode") or "").strip().upper()
    explicit_area_code = (raw.get("targetAreaCode") or "").strip() or None
    explicit_spot_code = (raw.get("targetSpotCode") or "").strip() or None
    explicit_source = (raw.get("targetSource") or "").strip() or None
    if explicit_mode:
        return _normalize_explicit_goal(
            raw=raw,
            master=master,
            explicit_mode=explicit_mode,
            explicit_area_code=explicit_area_code,
            explicit_spot_code=explicit_spot_code,
            explicit_source=explicit_source,
            inspection_enabled=inspection_enabled,
        )

    target_track = raw["targetTrack"]
    if target_track != "大库" and target_track != "大库外" and target_track not in master.tracks:
        raise InputValidationError(f"Unknown target track: {target_track}")

    spotting = (raw.get("isSpotting") or "").strip()
    vehicle_yard_mode = "NORMAL"

    if target_track in master.tracks:
        track = master.tracks[target_track]
        if track.track_type == "RUNNING":
            raise InputValidationError(f"Running track cannot be target: {target_track}")
        if (
            track.allows_final_destination is False
            and target_track not in WORK_AREA_DEFAULTS
            and target_track not in WORK_POSITION_TRACKS
        ):
            raise InputValidationError(
                f"Track {target_track} cannot be a final target (allows_final_destination=False)"
            )

    if target_track in WORK_POSITION_TRACKS:
        if spotting in ("", "否"):
            return (_work_position_goal(target_track, "FREE"), vehicle_yard_mode)
        if spotting == "是":
            return (_work_position_goal(target_track, "SPOTTING"), vehicle_yard_mode)
        if spotting.isdigit():
            return (
                _work_position_goal(
                    target_track,
                    "EXACT_NORTH_RANK",
                    target_rank=_parse_work_position_rank(spotting),
                ),
                vehicle_yard_mode,
            )

    if spotting in ("", "否"):
        if target_track in WORK_AREA_DEFAULTS:
            area_code = WORK_AREA_DEFAULTS[target_track]
            preferred_target_tracks: list[str] = []
            fallback_target_tracks: list[str] = []
            if area_code == "大库:RANDOM":
                preferred_target_tracks, fallback_target_tracks = _random_depot_track_preferences(
                    float(raw["vehicleLength"])
                )
            return (
                GoalSpec(
                    target_mode="AREA",
                    target_track=_area_track(area_code, target_track),
                    allowed_target_tracks=AREA_ALLOWED_TRACKS[area_code],
                    preferred_target_tracks=preferred_target_tracks or list(AREA_ALLOWED_TRACKS[area_code]),
                    fallback_target_tracks=fallback_target_tracks,
                    target_area_code=area_code,
                ),
                vehicle_yard_mode,
            )
        if target_track not in TRACK_MODE_ALLOWED:
            raise InputValidationError(f"Track target not allowed: {target_track}")
        return (
            GoalSpec(
                target_mode="TRACK",
                target_track=target_track,
                allowed_target_tracks=[target_track],
            ),
            vehicle_yard_mode,
        )

    if spotting == "是":
        if target_track == "轮":
            return (
                GoalSpec(
                    target_mode="AREA",
                    target_track=target_track,
                    allowed_target_tracks=[target_track],
                    target_area_code=WORK_AREA_DEFAULTS[target_track],
                ),
                vehicle_yard_mode,
            )
        raise InputValidationError(f"AREA request invalid for target: {target_track}")

    if spotting == "迎检":
        if target_track != "大库":
            raise InputValidationError("迎检 only allowed for 大库")
        preferred_target_tracks, fallback_target_tracks = _random_depot_track_preferences(
            float(raw["vehicleLength"])
        )
        return (
            GoalSpec(
                target_mode="AREA",
                target_track="修1库内",
                allowed_target_tracks=AREA_ALLOWED_TRACKS["大库:RANDOM"],
                preferred_target_tracks=preferred_target_tracks,
                fallback_target_tracks=fallback_target_tracks,
                target_area_code="大库:RANDOM",
            ),
            "INSPECTION",
        )

    if spotting.isdigit():
        if len(spotting) != 3:
            raise InputValidationError(f"Invalid spot code: {spotting}")
        if spotting[1:] in {"06", "07"} and not inspection_enabled:
            raise InputValidationError(f"Spot code not allowed in NORMAL mode: {spotting}")
        depot_track = _spot_to_track(spotting)
        if target_track not in {"大库", depot_track}:
            raise InputValidationError(
                f"Spot code {spotting} inconsistent with target {target_track}"
            )
        return (
            GoalSpec(
                target_mode="SPOT",
                target_track=depot_track,
                allowed_target_tracks=[depot_track],
                target_spot_code=spotting,
            ),
            vehicle_yard_mode,
        )

    raise InputValidationError(f"Unsupported isSpotting value: {spotting}")


def _normalize_explicit_goal(
    *,
    raw: dict,
    master: MasterData,
    explicit_mode: str,
    explicit_area_code: str | None,
    explicit_spot_code: str | None,
    explicit_source: str | None,
    inspection_enabled: bool,
) -> tuple[GoalSpec, str]:
    target_track = raw["targetTrack"]
    if target_track not in master.tracks and target_track not in {"大库", "大库外"}:
        raise InputValidationError(f"Unknown target track: {target_track}")
    if explicit_mode == "TRACK":
        if target_track in WORK_POSITION_TRACKS:
            return (
                _work_position_goal(
                    target_track,
                    "FREE",
                    target_source=explicit_source,
                ),
                "INSPECTION" if inspection_enabled else "NORMAL",
            )
        if target_track not in master.tracks:
            raise InputValidationError(f"TRACK goal requires concrete track: {target_track}")
        track = master.tracks[target_track]
        if track.track_type == "RUNNING":
            raise InputValidationError(f"Running track cannot be target: {target_track}")
        if (
            track.allows_final_destination is False
            and target_track not in WORK_AREA_DEFAULTS
            and target_track not in WORK_POSITION_TRACKS
        ):
            raise InputValidationError(
                f"Track {target_track} cannot be a final target (allows_final_destination=False)"
            )
        return (
            GoalSpec(
                target_mode="TRACK",
                target_track=target_track,
                allowed_target_tracks=[target_track],
                target_area_code=explicit_area_code,
                target_spot_code=explicit_spot_code,
                target_source=explicit_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    if explicit_mode == "SNAPSHOT":
        return _normalize_snapshot_goal(
            target_track=target_track,
            target_source=explicit_source,
            master=master,
            vehicle_length=float(raw["vehicleLength"]),
            inspection_enabled=inspection_enabled,
        )
    if explicit_mode == "AREA":
        if explicit_area_code is None:
            default_area_code = WORK_AREA_DEFAULTS.get(target_track)
            if default_area_code is None:
                raise InputValidationError(
                    f"AREA goal requires targetAreaCode (no default for {target_track})"
                )
            explicit_area_code = default_area_code
        if explicit_area_code in REMOVED_WORK_AREA_CODES:
            raise InputValidationError(
                f"Removed work area code {explicit_area_code}; use targetTrack with isSpotting or numeric north rank"
            )
        if explicit_area_code in FORBIDDEN_FINAL_AREA_CODES:
            raise InputValidationError(
                f"{explicit_area_code} cannot be an upstream final target"
            )
        allowed_tracks = AREA_ALLOWED_TRACKS.get(explicit_area_code)
        if allowed_tracks is None:
            if target_track not in master.tracks:
                raise InputValidationError(f"AREA goal requires concrete track: {target_track}")
            allowed_tracks = [target_track]
        return (
            GoalSpec(
                target_mode="AREA",
                target_track=target_track,
                allowed_target_tracks=list(allowed_tracks),
                preferred_target_tracks=(
                    _random_depot_track_preferences(float(raw["vehicleLength"]))[0]
                    if explicit_area_code == "大库:RANDOM"
                    else list(allowed_tracks)
                ),
                fallback_target_tracks=(
                    _random_depot_track_preferences(float(raw["vehicleLength"]))[1]
                    if explicit_area_code == "大库:RANDOM"
                    else []
                ),
                target_area_code=explicit_area_code,
                target_spot_code=explicit_spot_code,
                target_source=explicit_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    if explicit_mode == "SPOT":
        if explicit_spot_code is None:
            raise InputValidationError("SPOT goal requires targetSpotCode")
        if target_track in WORK_POSITION_TRACKS:
            return (
                _work_position_goal(
                    target_track,
                    "EXACT_NORTH_RANK",
                    target_rank=_parse_work_position_rank(explicit_spot_code),
                    target_source=explicit_source,
                ),
                "INSPECTION" if inspection_enabled else "NORMAL",
            )
        if target_track not in master.tracks and target_track != "大库":
            raise InputValidationError(f"SPOT goal requires depot track: {target_track}")
        if explicit_spot_code[1:] in {"06", "07"} and not inspection_enabled:
            raise InputValidationError(f"Spot code not allowed in NORMAL mode: {explicit_spot_code}")
        depot_track = _spot_to_track(explicit_spot_code)
        if target_track not in {"大库", depot_track}:
            raise InputValidationError(
                f"Spot code {explicit_spot_code} inconsistent with target {target_track}"
            )
        return (
            GoalSpec(
                target_mode="SPOT",
                target_track=depot_track,
                allowed_target_tracks=[depot_track],
                target_area_code=explicit_area_code,
                target_spot_code=explicit_spot_code,
                target_source=explicit_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    raise InputValidationError(f"Unsupported explicit targetMode: {explicit_mode}")


def _normalize_snapshot_goal(
    *,
    target_track: str,
    target_source: str | None,
    master: MasterData,
    vehicle_length: float,
    inspection_enabled: bool,
) -> tuple[GoalSpec, str]:
    if target_track == "大库":
        preferred, fallback = _random_depot_track_preferences(vehicle_length)
        return (
            GoalSpec(
                target_mode="SNAPSHOT",
                target_track="修1库内",
                allowed_target_tracks=AREA_ALLOWED_TRACKS["大库:RANDOM"],
                preferred_target_tracks=preferred,
                fallback_target_tracks=fallback,
                target_area_code="大库:RANDOM",
                target_source=target_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    if target_track == "大库外":
        allowed = AREA_ALLOWED_TRACKS["大库外:RANDOM"]
        return (
            GoalSpec(
                target_mode="SNAPSHOT",
                target_track="修1库外",
                allowed_target_tracks=list(allowed),
                preferred_target_tracks=list(allowed),
                target_area_code="大库外:RANDOM",
                target_source=target_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    if target_track not in master.tracks:
        raise InputValidationError(f"Unknown target track: {target_track}")
    track = master.tracks[target_track]
    if track.track_type == "RUNNING" or not track.allow_parking:
        raise InputValidationError(f"SNAPSHOT target must be parkable: {target_track}")
    if target_track == "存4北":
        return (
            GoalSpec(
                target_mode="TRACK",
                target_track=target_track,
                allowed_target_tracks=[target_track],
                target_source=target_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )

    area_code = SNAPSHOT_AREA_CODE_BY_TRACK.get(target_track)
    if area_code is None:
        return (
            GoalSpec(
                target_mode="SNAPSHOT",
                target_track=target_track,
                allowed_target_tracks=[target_track],
                preferred_target_tracks=[target_track],
                target_source=target_source,
            ),
            "INSPECTION" if inspection_enabled else "NORMAL",
        )
    allowed = list(AREA_ALLOWED_TRACKS[area_code])
    preferred = [target_track] if target_track in allowed else []
    fallback = [track for track in allowed if track != target_track]
    return (
        GoalSpec(
            target_mode="SNAPSHOT",
            target_track=target_track,
            allowed_target_tracks=allowed,
            preferred_target_tracks=preferred,
            fallback_target_tracks=fallback,
            target_area_code=area_code,
            target_source=target_source,
        ),
        "INSPECTION" if inspection_enabled else "NORMAL",
    )


def _area_track(area_code: str, fallback_track: str) -> str:
    if area_code == "大库:RANDOM":
        return "修1库内"
    if area_code == "大库外:RANDOM":
        return "修1库外"
    if ":" in area_code:
        return area_code.split(":", 1)[0]
    return fallback_track


def _work_position_goal(
    target_track: str,
    kind: str,
    *,
    target_rank: int | None = None,
    target_source: str | None = None,
) -> GoalSpec:
    return GoalSpec(
        target_mode="WORK_POSITION",
        target_track=target_track,
        allowed_target_tracks=[target_track],
        target_area_code=None,
        target_spot_code=None,
        target_source=target_source,
        work_position_kind=kind,
        target_rank=target_rank,
    )


def _parse_work_position_rank(raw_value: str) -> int:
    value = (raw_value or "").strip()
    if not value.isdigit():
        raise InputValidationError("Work track targetSpotCode must be a positive north rank")
    rank = int(value)
    if rank <= 0:
        raise InputValidationError("Work track targetSpotCode must be a positive north rank")
    return rank


def _spot_to_track(spot_code: str) -> str:
    first = spot_code[0]
    if first not in {"1", "2", "3", "4"}:
        raise InputValidationError(f"Unsupported spot prefix: {spot_code}")
    return f"修{first}库内"


def _merge_yard_mode(current: str, incoming: str) -> str:
    if current == incoming:
        return current
    if current == "NORMAL" and incoming == "INSPECTION":
        return incoming
    if current == "INSPECTION" and incoming == "NORMAL":
        return current
    raise InputValidationError(f"Conflicting yard modes: {current} vs {incoming}")


def _parse_vehicle_attribute(value: str) -> tuple[bool, bool, bool]:
    value = (value or "").strip()
    if value == "称重":
        return True, False, False
    if value == "重车":
        return False, True, False
    if value == "关门车":
        return False, False, True
    return False, False, False
