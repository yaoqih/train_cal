from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from fzed_shunting.domain.depot_spots import list_track_spots
from fzed_shunting.domain.master_data import MasterData
from fzed_shunting.io.normalize_input import NormalizedPlanInput, normalize_plan_input

OPERATION_MODE_L7_CLOSED_TOPOLOGY = "L7_CLOSED_TOPOLOGY"
L7_BLOCKED_BRANCHES = ("L15-L16",)
JI_BUFFER_TRACKS = ("机南", "机棚", "机北1", "机北2", "机北3")
WASH_CONFLICT_TRACKS = frozenset({"洗北", "洗南", "油"})
DEPOT_TARGET_TRACKS = frozenset({"修1", "修2", "修3", "修4", "轮"})
DEPOT_OUTER_TRACKS = frozenset({"修1库外", "修2库外", "修3库外", "修4库外"})
STORAGE_TRACKS = frozenset({"存1", "存2", "存3", "存4北", "存4南", "存5北", "存5南"})
PHASE1_NON_DEPOT_REGION_TRACKS = frozenset({
    "抛",
    "洗南",
    "洗北",
    "油",
    "预修",
    "调棚",
    "调北",
    "机棚",
    "机南",
    "机北1",
    "机北3",
    "机北2",
    "存1",
    "存2",
    "存3",
    "存4北",
    "存4南",
    "存5北",
    "存5南",
})
PHASE1_OPENING_MAX_PLANS = 2
PHASE1_OPENING_BUDGET_M = 72.0
PHASE1_LOCAL_FINISH_MAX_PLANS = 4
PHASE1_LOCAL_FINISH_BUDGET_M = 72.0
PHASE1_LOCAL_FINISH_SEGMENT_LENGTH_M = 42.0
PHASE1_UNIT_MAX_LENGTH_M = 48.0
PHASE1_LOCAL_HIGH_PRESSURE_SOURCE_TRACKS = (
    "存5北",
    "调棚",
    "预修",
    "存5南",
    "油",
    "抛",
    "存3",
    "存2",
    "存1",
    "调北",
    "洗南",
    "洗北",
)
PHASE1_USABLE_BUFFER_CAPACITY_M = {
    "机北1": 81.4,
    "机北2": 55.7,
    "机北3": 52.0,
    "机棚": 82.0,
    "机南": 60.0,
}
PHASE1_MAIN_BUFFER_TRACKS = ("机棚", "机南")
PHASE1_SUPPORT_BUFFER_TRACKS = ("机北1", "机北2", "机北3")
PHASE1_BUFFER_TRACK_ROLES = {
    "机棚": "main",
    "机南": "main",
    "机北1": "support",
    "机北2": "support",
    "机北3": "special",
}
PHASE1_TEMP_PARKING_TRACKS = ("存1", "存2", "存3", "存5南", "存5北", "调北", "预修", "调棚")
PHASE1_BLOCKER_BUCKET_WORK = "PHASE1_BLOCKER_BUCKET_WORK"
PHASE1_BLOCKER_BUCKET_YARD = "PHASE1_BLOCKER_BUCKET_YARD"
PHASE1_WORK_BUCKET_TRACKS = ("调棚", "预修", "调北", "存5北", "存5南", "存3", "存2", "存1")
PHASE1_YARD_BUCKET_TRACKS = ("存3", "存2", "存1", "存5北", "存5南", "调北", "预修", "调棚")
PHASE1_DEPOT_PACKAGE_MAX_REQUIRED_M = max(PHASE1_USABLE_BUFFER_CAPACITY_M.values())
PHASE1_TEMP_PACKAGE_MAX_LENGTH_M = 36.0
PHASE1_MAX_ACTIVE_HOT_SOURCE_TRACKS = 4
PHASE1_MAX_ACTIVE_STORAGE_SOURCE_TRACKS = 2
PHASE1_MAX_SELECTED_PACKAGES = 18
PHASE1_MAX_TOTAL_ACTIVE_VEHICLES = 32
PHASE1_MAX_OPTIONAL_CLEANUP_PACKAGES = 4
PHASE1_PRIMARY_BACKBONE_SLOT_COUNT = 3
PHASE1_ELASTIC_BACKBONE_SLOT_COUNT = 1
PHASE2_L1_TRANSFER_MAX_LENGTH_M = 193.0
PHASE3_DYNAMIC_CURRENT_HOLD = "PHASE3_DYNAMIC_CURRENT_HOLD"
FIXED_DEPOT_RESIDENT_SOURCE = "FIXED_DEPOT_RESIDENT"


@dataclass(frozen=True)
class VehicleStageFacts:
    vehicle_no: str
    current_track: str
    current_order: int
    vehicle_length: float
    repair_process: str
    raw_goal: dict[str, Any]
    final_target_track: str
    final_target_spot: str
    final_allowed_tracks: tuple[str, ...]
    final_preferred_tracks: tuple[str, ...]
    final_fallback_tracks: tuple[str, ...]
    final_target_area_code: str
    final_target_mode: str
    final_target_source: str
    final_family: str
    current_zone: str
    needs_depot_batch: bool
    is_depot_area_vehicle: bool
    is_current_final_track: bool
    is_fixed_depot_resident: bool
    is_wash_conflict: bool
    is_cun4bei_final: bool
    needs_stage2_depot_collect: bool
    need_weigh: bool
    is_heavy: bool
    is_close_door: bool


@dataclass(frozen=True)
class Phase1OpeningPlan:
    plan_id: str
    source_track: str
    blocker_vehicle_nos: tuple[str, ...]
    released_vehicle_nos: tuple[str, ...]
    blocker_total_length: float
    released_total_length: float
    released_final_families: tuple[str, ...]
    blocker_target_tracks: tuple[str, ...]


@dataclass(frozen=True)
class Phase1LocalFinishPlan:
    plan_id: str
    source_track: str
    target_track: str
    vehicle_nos: tuple[str, ...]
    total_length: float
    priority_tag: str
    cluster_kind: str
    completion_gain: int
    released_candidate_gain: int
    source_pending_pressure: int
    source_priority: int


@dataclass(frozen=True)
class Phase1TaskPackage:
    unit_id: str
    chain_id: str
    unit_type: str
    source_track: str
    vehicle_nos: tuple[str, ...]
    stage_target_track: str
    stage_target_source: str
    uses_buffer: bool
    total_length_m: float
    final_family: str
    repair_process_profile: tuple[str, ...]
    min_spot_priority: int
    source_order_start: int
    source_order_end: int
    entry_order_key: tuple[Any, ...]
    dependency_tags: tuple[str, ...]
    release_gain: int
    completion_gain: int
    released_candidate_gain: int
    topology_risk: int
    buffer_preference: tuple[str, ...]
    is_mandatory_clearance: bool
    segment_index: int
    segment_role: str
    segment_class: str


@dataclass(frozen=True)
class Phase1PackageEdge:
    from_unit_id: str
    to_unit_id: str
    reason: str


@dataclass(frozen=True)
class Phase1PackagePlan:
    selected_packages: tuple[Phase1TaskPackage, ...]
    deferred_vehicle_nos: frozenset[str]
    buffer_assignment: dict[str, str]
    target_rank_by_vehicle: dict[str, int]
    package_order: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase1TrackChain:
    chain_id: str
    source_track: str
    source_role: str
    packages: tuple[Phase1TaskPackage, ...]
    local_vehicle_count: int
    buffer_vehicle_count: int
    hidden_candidate_count: int
    candidate_count: int
    candidate_length_m: float


@dataclass(frozen=True)
class Phase1Plan:
    selected_vehicle_nos: frozenset[str]
    deferred_vehicle_nos: frozenset[str]
    buffer_assignment: dict[str, str]
    target_rank_by_vehicle: dict[str, int]
    goal_overrides: dict[str, tuple[str, str]]
    wave_plans: tuple["Phase1WavePlan", ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase1LayoutTemplate:
    template_name: str
    tracks: tuple[str, ...]
    preferred_open_order: tuple[str, ...]
    support_tracks: frozenset[str]


@dataclass(frozen=True)
class Phase1Block:
    block_id: str
    source_track: str
    block_type: str
    vehicle_nos: tuple[str, ...]
    total_length_m: float
    target_track: str | None
    target_source: str
    uses_buffer: bool
    buffer_preference: tuple[str, ...]
    source_order_start: int
    source_order_end: int
    final_family: str
    phase3_rank_key: tuple[Any, ...]
    released_depot_vehicle_count: int
    released_finish_vehicle_count: int
    required_predecessor_ids: tuple[str, ...]
    layout_role: str
    topology_zone: str
    throat_group: str
    pressure_gain: int
    coupling_degree: int


@dataclass(frozen=True)
class SourceTrackPlan:
    source_track: str
    blocks: tuple[Phase1Block, ...]
    reachable_depot_vehicle_nos: tuple[str, ...]
    reachable_finish_vehicle_nos: tuple[str, ...]
    cun4_clear_required: bool
    buffer_demand_m: float
    source_priority_score: tuple[int, ...]


@dataclass(frozen=True)
class Phase1BackbonePlan:
    selected_block_ids: tuple[str, ...]
    selected_source_tracks: tuple[str, ...]
    reserved_buffer_by_track: dict[str, float]
    selected_buffer_assignment: dict[str, str]
    target_rank_by_vehicle: dict[str, int]
    layout_template_name: str
    opened_buffer_tracks: tuple[str, ...]
    depot_slot_limit: int | None = None
    depot_slot_limited: bool = False


@dataclass(frozen=True)
class Phase1FinishPlan:
    selected_block_ids: tuple[str, ...]
    goal_overrides: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class Phase1WavePlan:
    wave_name: str
    wave_role: str
    wave_type: str
    selected_source_track: str
    selected_block_ids: tuple[str, ...]
    required_predecessor_ids: tuple[str, ...]
    selected_vehicle_nos: frozenset[str]
    buffer_assignment: dict[str, str]
    goal_overrides: dict[str, tuple[str, str]]
    target_rank_by_vehicle: dict[str, int]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase1MacroTask:
    task_id: str
    source_track: str
    source_role: str
    wave_chunks: tuple[tuple[str, tuple[str, ...]], ...]
    block_ids: tuple[str, ...]
    required_predecessor_ids: tuple[str, ...]
    vehicle_nos: tuple[str, ...]
    buffer_vehicle_count: int
    cleanup_vehicle_count: int
    released_depot_vehicle_count: int
    released_finish_vehicle_count: int
    pressure_gain: int
    topology_zones: tuple[str, ...]
    throat_groups: tuple[str, ...]
    score_key: tuple[Any, ...]


@dataclass(frozen=True)
class Phase1TrackFacts:
    source_track: str
    candidate_vehicle_nos: tuple[str, ...]
    candidate_count: int
    candidate_length_m: float
    must_clear_count: int
    local_finish_count: int
    opening_release_count: int
    hidden_candidate_count: int
    bottleneck_tags: tuple[str, ...]
    source_role: str
    source_priority_score: tuple[int, ...]


@dataclass(frozen=True)
class Phase1StructurePlan:
    track_facts: tuple[Phase1TrackFacts, ...]
    package_source_tracks: tuple[str, ...]


@dataclass(frozen=True)
class Phase1DemandSummary:
    depot_vehicle_nos: frozenset[str]
    cun4_vehicle_nos: frozenset[str]
    ji_non_depot_vehicle_nos: frozenset[str]
    total_non_depot_region_vehicle_nos: frozenset[str]
    depot_demand_total_length_m: float
    ji_capacity_total_m: float
    ji_overflow_m: float


@dataclass(frozen=True)
class Phase1LayoutPackage:
    package_id: str
    chain_id: str
    package_kind: str
    source_track: str
    vehicle_nos: tuple[str, ...]
    total_length_m: float
    target_track: str
    target_source: str
    final_family: str
    min_spot_priority: int
    source_order_start: int
    source_order_end: int
    buffer_preference: tuple[str, ...]
    uses_buffer: bool
    pressure_cut: str
    reason_tags: tuple[str, ...]
    execution_layer: str
    complexity_cost: int
    source_chain_role: str
    is_required_for_backbone: bool
    segment_role: str
    source_segment_index: int
    source_segment_count: int
    source_total_vehicle_count: int
    requires_previous_segment: bool


@dataclass(frozen=True)
class Phase1SourceOpeningSummary:
    source_track: str
    source_chain_role: str
    backbone_packages: tuple[Phase1LayoutPackage, ...]
    required_cleanup_packages: tuple[Phase1LayoutPackage, ...]
    optional_cleanup_packages: tuple[Phase1LayoutPackage, ...]
    backbone_vehicle_count: int
    required_cleanup_vehicle_count: int
    required_clearance_gain_units: int
    opening_cost_units: int
    opening_gain_units: int
    opening_score: int


@dataclass(frozen=True)
class Phase1SourceAdmission:
    source_track: str
    source_chain_role: str
    backbone_packages: tuple[Phase1LayoutPackage, ...]
    required_cleanup_packages: tuple[Phase1LayoutPackage, ...]
    optional_cleanup_packages: tuple[Phase1LayoutPackage, ...]
    backbone_vehicle_count: int
    required_cleanup_vehicle_count: int
    required_clearance_gain_units: int
    opening_cost_units: int
    opening_gain_units: int
    opening_score: int
    primary_package_ids: tuple[str, ...]
    primary_vehicle_count: int
    primary_required_length_m: float
    admission_tier: str


@dataclass(frozen=True)
class Phase1AdmissionPlan:
    admitted_source_tracks: tuple[str, ...]
    primary_source_tracks: tuple[str, ...]
    elastic_source_tracks: tuple[str, ...]
    companion_source_tracks: tuple[str, ...]
    deferred_source_tracks: tuple[str, ...]
    slot_index_by_source: dict[str, int]
    slot_type_by_source: dict[str, str]
    rejection_reason_by_source: dict[str, str]


@dataclass(frozen=True)
class Phase1LayoutResult:
    selected_depot_packages: tuple[Phase1LayoutPackage, ...]
    non_buffer_packages: tuple[Phase1LayoutPackage, ...]
    deferred_vehicle_nos: frozenset[str]
    buffer_assignment: dict[str, str]
    target_rank_by_vehicle: dict[str, int]
    goal_overrides: dict[str, tuple[str, str]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TopologyStagePlan:
    phase1_plan: Phase1Plan
    stage2_plan: "Stage2Plan"


@dataclass(frozen=True)
class Phase2DemandSummary:
    depot_stay_vehicle_nos: frozenset[str]
    cun4_final_vehicle_nos: frozenset[str]
    outbound_vehicle_nos: frozenset[str]
    fixed_depot_resident_vehicle_nos: frozenset[str]


@dataclass(frozen=True)
class Phase2OutboundGroup:
    group_id: str
    group_kind: str
    vehicle_nos: tuple[str, ...]
    current_track: str
    source_order_start: int
    final_target_track: str
    final_family: str
    repair_process_profile: tuple[str, ...]
    total_length_m: float
    rank_key: tuple[Any, ...]


@dataclass(frozen=True)
class Phase2TrackLayer:
    source_track: str
    layer_index: int
    group_ids: tuple[str, ...]
    vehicle_nos: tuple[str, ...]
    total_length_m: float
    outbound_vehicle_nos: tuple[str, ...]
    cun4_final_vehicle_nos: tuple[str, ...]
    exposed_prefix_vehicle_nos: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase2ExecutionPlan:
    execution_name: str
    track_layers: tuple[Phase2TrackLayer, ...]
    collection_batches: tuple[tuple[str, ...], ...]
    predecessor_unlock_vehicle_nos: tuple[str, ...]
    must_pull_vehicle_nos: tuple[str, ...]
    unlocking_optional_vehicle_nos: tuple[str, ...]
    phase3_clearance_vehicle_nos: tuple[str, ...]
    pure_batch_optional_vehicle_nos: tuple[str, ...]
    optional_cun4_vehicle_nos: tuple[str, ...]
    deferred_tail_vehicle_nos: tuple[str, ...]
    transfer_vehicle_nos: tuple[str, ...]
    reserved_cun4_capacity_m: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Phase2WavePlan:
    wave_name: str
    wave_role: str
    collected_vehicle_nos: tuple[str, ...]
    attach_units: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Stage2Plan:
    demand_summary: Phase2DemandSummary
    outbound_groups: tuple[Phase2OutboundGroup, ...]
    track_layers: tuple[Phase2TrackLayer, ...]
    execution_plan: Phase2ExecutionPlan | None
    wave_plans: tuple[Phase2WavePlan, ...]
    diagnostics: dict[str, Any]


PHASE4_DYNAMIC_CURRENT_HOLD = "PHASE4_DYNAMIC_CURRENT_HOLD"
PHASE4_RESIDUAL_CLEANUP = "PHASE4_RESIDUAL_CLEANUP"


def is_l7_closed_topology_mode(payload: dict[str, Any]) -> bool:
    return str(payload.get("operationMode") or "").strip().upper() == OPERATION_MODE_L7_CLOSED_TOPOLOGY


def build_l7_closed_topology_workflow_payload(
    master: MasterData,
    payload: dict[str, Any],
    *,
    allow_internal_loco_tracks: bool = False,
    phase1_respect_existing_buffer_occupancy: bool = True,
    phase1_buffer_occupancy_exempt_vehicle_nos: frozenset[str] | None = None,
) -> dict[str, Any]:
    normalized = normalize_plan_input(
        payload,
        master,
        allow_internal_loco_tracks=allow_internal_loco_tracks,
    )
    vehicle_facts = _build_vehicle_stage_facts(payload, normalized)
    if phase1_respect_existing_buffer_occupancy and phase1_buffer_occupancy_exempt_vehicle_nos is None:
        phase1_buffer_occupancy_exempt_vehicle_nos = frozenset(
            facts.vehicle_no
            for facts in vehicle_facts
            if facts.current_track in JI_BUFFER_TRACKS and facts.needs_depot_batch
        )
    stage_plan = _build_topology_stage_plan(
        vehicle_facts,
        master,
        normalized.yard_mode,
        phase1_respect_existing_buffer_occupancy=phase1_respect_existing_buffer_occupancy,
        phase1_buffer_occupancy_exempt_vehicle_nos=phase1_buffer_occupancy_exempt_vehicle_nos,
    )
    final_goal_by_vehicle = {
        str(item["vehicleNo"]): _final_goal_payload(item)
        for item in payload.get("vehicleInfo", [])
    }
    initial_vehicle_info = [
        _initial_vehicle_info_item(item)
        for item in payload.get("vehicleInfo", [])
    ]
    track_info = _with_required_phase_buffer_track_info(
        master=master,
        track_info=[dict(item) for item in payload.get("trackInfo", [])],
    )
    fixed_depot_resident_vehicle_nos = sorted(
        facts.vehicle_no for facts in vehicle_facts if facts.is_fixed_depot_resident
    )

    return {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": track_info,
        "initialVehicleInfo": initial_vehicle_info,
        "locoTrackName": payload.get("locoTrackName") or "机库",
        "workflowStages": [
            {
                "name": "phase1_pre_repair_buffering",
                "description": "联7封锁下，先形成上游入库骨架，并只做必要清障",
                "routePolicy": {"blockedBranches": list(L7_BLOCKED_BRANCHES)},
                "stagePolicy": {
                    "stageMode": "PHASE1_PRE_REPAIR_BUFFERING",
                    "bufferTracks": list(JI_BUFFER_TRACKS),
                    "fixedDepotResidentVehicleNos": fixed_depot_resident_vehicle_nos,
                    "phase1OriginalGoalRows": [
                        {
                            "vehicleNo": facts.vehicle_no,
                            "targetTrack": str(facts.raw_goal.get("targetTrack") or ""),
                            "isSpotting": str(facts.raw_goal.get("isSpotting") or ""),
                        }
                        for facts in vehicle_facts
                    ],
                    "phase1InitialBufferVehicleNos": sorted(
                        facts.vehicle_no
                        for facts in vehicle_facts
                        if facts.current_track in JI_BUFFER_TRACKS and facts.needs_depot_batch
                    ),
                    "packageAssignments": dict(stage_plan.phase1_plan.buffer_assignment),
                    "layoutAssignments": dict(stage_plan.phase1_plan.buffer_assignment),
                    "packageTargetRanks": dict(stage_plan.phase1_plan.target_rank_by_vehicle),
                    "layoutTargetRanks": dict(stage_plan.phase1_plan.target_rank_by_vehicle),
                    "deferredVehicleNos": sorted(stage_plan.phase1_plan.deferred_vehicle_nos),
                    "phase1WavePlans": [
                        _phase1_wave_stage_policy(
                            wave_plan,
                            stage_plan=stage_plan,
                            vehicle_facts=vehicle_facts,
                        )
                        for wave_plan in stage_plan.phase1_plan.wave_plans
                    ],
                    "phase1Diagnostics": dict(stage_plan.phase1_plan.diagnostics),
                },
                "vehicleGoals": [
                    _phase1_goal(facts, stage_plan=stage_plan)
                    for facts in vehicle_facts
                ],
            },
            {
                "name": "phase2_depot_area_marshalling",
                "description": "联7开放后，先在大库区内部形成出库链，再尽量少次整列拉到存4北",
                "routePolicy": {},
                "stagePolicy": {
                    "stageMode": "PHASE2_DEPOT_AREA_MARSHALLING",
                    "bufferTracks": list(JI_BUFFER_TRACKS),
                    "exchangeTrack": "存4北",
                    "fixedDepotResidentVehicleNos": fixed_depot_resident_vehicle_nos,
                    "fixedDepotResidentVehicles": sorted(stage_plan.stage2_plan.demand_summary.fixed_depot_resident_vehicle_nos),
                    "depotStayVehicles": sorted(stage_plan.stage2_plan.demand_summary.depot_stay_vehicle_nos),
                    "cun4FinalVehicles": sorted(stage_plan.stage2_plan.demand_summary.cun4_final_vehicle_nos),
                    "depotOutboundVehicles": sorted(stage_plan.stage2_plan.demand_summary.outbound_vehicle_nos),
                    "phase2OutboundGroups": [
                        {
                            "groupId": group.group_id,
                            "groupKind": group.group_kind,
                            "vehicleNos": list(group.vehicle_nos),
                            "currentTrack": group.current_track,
                            "sourceOrderStart": group.source_order_start,
                            "finalTargetTrack": group.final_target_track,
                            "finalFamily": group.final_family,
                            "repairProcessProfile": list(group.repair_process_profile),
                            "totalLengthM": group.total_length_m,
                        }
                        for group in stage_plan.stage2_plan.outbound_groups
                    ],
                    "phase2TrackLayers": [
                        {
                            "sourceTrack": layer.source_track,
                            "layerIndex": layer.layer_index,
                            "groupIds": list(layer.group_ids),
                            "vehicleNos": list(layer.vehicle_nos),
                            "outboundVehicleNos": list(layer.outbound_vehicle_nos),
                            "cun4FinalVehicleNos": list(layer.cun4_final_vehicle_nos),
                            "exposedPrefixVehicleNos": list(layer.exposed_prefix_vehicle_nos),
                            "totalLengthM": layer.total_length_m,
                            "diagnostics": dict(layer.diagnostics),
                        }
                        for layer in stage_plan.stage2_plan.track_layers
                    ],
                    "phase2ExecutionPlan": _phase2_execution_stage_policy(
                        stage_plan.stage2_plan.execution_plan,
                    ),
                    "phase2WavePlans": [
                        {
                            "waveName": wave.wave_name,
                            "waveRole": wave.wave_role,
                            "phase2CollectedVehicleNos": list(wave.collected_vehicle_nos),
                            "phase2AttachUnits": [dict(item) for item in wave.attach_units],
                            "activeGoalsByVehicle": dict(wave.diagnostics.get("activeGoalsByVehicle") or {}),
                            "waveDiagnostics": dict(wave.diagnostics),
                        }
                        for wave in stage_plan.stage2_plan.wave_plans
                    ],
                    "phase2Diagnostics": dict(stage_plan.stage2_plan.diagnostics),
                },
                "vehicleGoals": [
                    _phase2_goal(facts, stage_plan=stage_plan)
                    for facts in vehicle_facts
                ],
            },
            {
                "name": "phase3_ji_to_depot_allocation",
                "description": "将已成形的入库骨架直接推进到大库最终精确位置",
                "routePolicy": {},
                "stagePolicy": {
                    "stageMode": "PHASE3_JI_TO_DEPOT_ALLOCATION",
                    "bufferTracks": list(JI_BUFFER_TRACKS),
                    "fixedDepotResidentVehicleNos": fixed_depot_resident_vehicle_nos,
                },
                "vehicleGoals": [
                    _phase3_goal(
                        facts,
                        final_goal_by_vehicle=final_goal_by_vehicle,
                        stage_plan=stage_plan,
                    )
                    for facts in vehicle_facts
                ],
            },
            {
                "name": "final_exact_settle_and_cleanup",
                "description": "冻结第3阶段已完成的大库结果，仅处理剩余未归位车辆与少量收尾清理",
                "routePolicy": {},
                "stagePolicy": {
                    "stageMode": PHASE4_RESIDUAL_CLEANUP,
                    "fixedDepotResidentVehicleNos": fixed_depot_resident_vehicle_nos,
                },
                "vehicleGoals": [
                    _phase4_goal(
                        facts,
                        final_goal_by_vehicle=final_goal_by_vehicle,
                    )
                    for facts in vehicle_facts
                ],
            },
        ],
    }


def _with_required_phase_buffer_track_info(
    *,
    master: MasterData,
    track_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {str(item.get("trackName") or ""): dict(item) for item in track_info}
    required_tracks = tuple(dict.fromkeys(JI_BUFFER_TRACKS + PHASE1_TEMP_PARKING_TRACKS + tuple(sorted(DEPOT_OUTER_TRACKS))))
    for track_name in required_tracks:
        if track_name in by_name or track_name not in master.tracks:
            continue
        by_name[track_name] = {
            "trackName": track_name,
            "trackDistance": round(float(master.tracks[track_name].effective_length_m), 1),
        }
    original_order = [str(item.get("trackName") or "") for item in track_info]
    remaining = [track for track in required_tracks if track not in original_order and track in by_name]
    return [by_name[name] for name in original_order if name in by_name] + [by_name[name] for name in remaining]


def _build_vehicle_stage_facts(
    payload: dict[str, Any],
    normalized: NormalizedPlanInput,
) -> list[VehicleStageFacts]:
    raw_by_vehicle = {
        str(item["vehicleNo"]): dict(item)
        for item in payload.get("vehicleInfo", [])
    }
    facts: list[VehicleStageFacts] = []
    for vehicle in normalized.vehicles:
        raw_goal = raw_by_vehicle[vehicle.vehicle_no]
        allowed_tracks = tuple(vehicle.goal.allowed_target_tracks)
        final_target_track = vehicle.goal.target_track
        final_target_spot = vehicle.goal.target_spot_code or str(raw_goal.get("isSpotting", "") or "")
        needs_depot_batch = bool(DEPOT_TARGET_TRACKS.intersection(allowed_tracks))
        is_depot_area_vehicle = vehicle.current_track in DEPOT_TARGET_TRACKS or vehicle.current_track in DEPOT_OUTER_TRACKS
        is_cun4bei_final = "存4北" in allowed_tracks
        final_family = _primary_final_track_from_allowed(allowed_tracks, final_target_track)
        current_zone = _track_zone(vehicle.current_track)
        is_current_final_track = vehicle.current_track in allowed_tracks
        is_fixed_depot_resident = _is_fixed_depot_resident_goal(
            current_track=vehicle.current_track,
            raw_target_track=str(raw_goal.get("targetTrack") or ""),
            raw_is_spotting=str(raw_goal.get("isSpotting") or ""),
        )
        facts.append(
            VehicleStageFacts(
                vehicle_no=vehicle.vehicle_no,
                current_track=vehicle.current_track,
                current_order=vehicle.order,
                vehicle_length=vehicle.vehicle_length,
                repair_process=vehicle.repair_process,
                raw_goal=raw_goal,
                final_target_track=final_target_track,
                final_target_spot=final_target_spot,
                final_allowed_tracks=allowed_tracks,
                final_preferred_tracks=tuple(vehicle.goal.preferred_target_tracks),
                final_fallback_tracks=tuple(vehicle.goal.fallback_target_tracks),
                final_target_area_code=vehicle.goal.target_area_code or "",
                final_target_mode=vehicle.goal.target_mode,
                final_target_source=vehicle.goal.target_source or "",
                final_family=final_family,
                current_zone=current_zone,
                needs_depot_batch=needs_depot_batch,
                is_depot_area_vehicle=is_depot_area_vehicle,
                is_current_final_track=is_current_final_track,
                is_fixed_depot_resident=is_fixed_depot_resident,
                is_wash_conflict=vehicle.current_track in WASH_CONFLICT_TRACKS and needs_depot_batch,
                is_cun4bei_final=is_cun4bei_final,
                needs_stage2_depot_collect=_needs_stage2_depot_collect(
                    current_track=vehicle.current_track,
                    final_allowed_tracks=allowed_tracks,
                    needs_depot_batch=needs_depot_batch,
                    is_depot_area_vehicle=is_depot_area_vehicle,
                    is_current_final_track=is_current_final_track,
                ),
                need_weigh=vehicle.need_weigh,
                is_heavy=vehicle.is_heavy,
                is_close_door=vehicle.is_close_door,
            )
        )
    facts.sort(key=lambda item: (topology_pressure_key(item), final_sequence_key(item), item.vehicle_no))
    return facts


def _is_fixed_depot_resident_goal(
    *,
    current_track: str,
    raw_target_track: str,
    raw_is_spotting: str,
) -> bool:
    if current_track == "轮":
        return raw_target_track == "轮"
    if current_track not in {"修1", "修2", "修3", "修4"}:
        return False
    if raw_target_track == current_track:
        return True
    if raw_target_track != "大库":
        return False
    # Empty/否 means the exported final depot slot did not request a different
    # alignment. The current depot slot is therefore a hard business anchor.
    return raw_is_spotting.strip() in {"", "否"}


def final_sequence_key(facts: VehicleStageFacts) -> tuple[Any, ...]:
    return (
        _final_family_priority(facts.final_family),
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
        -facts.vehicle_length,
        facts.vehicle_no,
    )


def topology_pressure_key(facts: VehicleStageFacts) -> tuple[int, ...]:
    return (
        0 if facts.is_wash_conflict else 1,
        _pressure_cut_priority(_pressure_cut_name(facts)),
        _zone_priority(facts.current_zone),
        _final_family_priority(facts.final_family),
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
        -facts.vehicle_length,
    )


def _stage2_other_collect_order_key(facts: VehicleStageFacts) -> tuple[Any, ...]:
    return (
        _phase2_other_target_priority(facts.final_target_track),
        _final_family_priority(facts.final_family),
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
        facts.current_track,
        facts.current_order,
        facts.vehicle_no,
    )


def _stage2_cun4bei_final_order_key(facts: VehicleStageFacts) -> tuple[Any, ...]:
    return (
        _phase2_depot_track_priority(facts.current_track),
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
        0 if facts.is_close_door else 1,
        facts.current_order,
        facts.vehicle_no,
    )


def _phase2_other_target_priority(track_name: str) -> int:
    return {
        "存1": 0,
        "存2": 1,
        "存3": 2,
        "存4南": 3,
        "存5南": 4,
        "存5北": 5,
        "机北3": 6,
        "调北": 7,
        "洗北": 8,
        "洗南": 9,
        "油": 10,
        "抛": 11,
        "调棚": 12,
        "预修": 13,
        "机棚": 14,
        "机南": 15,
    }.get(track_name, 99)


def _phase2_depot_track_priority(track_name: str) -> int:
    return {
        "修4": 0,
        "修3": 1,
        "修2": 2,
        "修1": 3,
        "轮": 4,
        "修4库外": 5,
        "修3库外": 6,
        "修2库外": 7,
        "修1库外": 8,
    }.get(track_name, 99)


def _build_topology_stage_plan(
    facts_list: list[VehicleStageFacts],
    master: MasterData,
    yard_mode: str,
    *,
    phase1_respect_existing_buffer_occupancy: bool = True,
    phase1_buffer_occupancy_exempt_vehicle_nos: frozenset[str] | None = None,
) -> TopologyStagePlan:
    phase1_plan = _build_phase1_plan(
        facts_list,
        master,
        respect_existing_buffer_occupancy=phase1_respect_existing_buffer_occupancy,
        buffer_occupancy_exempt_vehicle_nos=phase1_buffer_occupancy_exempt_vehicle_nos,
    )
    stage2_plan = _build_stage2_plan(facts_list, yard_mode=yard_mode)
    return TopologyStagePlan(
        phase1_plan=phase1_plan,
        stage2_plan=stage2_plan,
    )


def _build_stage2_plan(
    facts_list: list[VehicleStageFacts],
    *,
    yard_mode: str,
) -> Stage2Plan:
    depot_stay_members: list[VehicleStageFacts] = []
    cun4_final_members: list[VehicleStageFacts] = []
    outbound_members: list[VehicleStageFacts] = []
    fixed_depot_resident_vehicle_nos = frozenset(
        facts.vehicle_no for facts in facts_list if facts.is_fixed_depot_resident
    )
    for facts in facts_list:
        if not facts.is_depot_area_vehicle:
            continue
        if facts.is_fixed_depot_resident:
            depot_stay_members.append(facts)
            continue
        if facts.needs_depot_batch:
            depot_stay_members.append(facts)
            continue
        if facts.is_cun4bei_final:
            cun4_final_members.append(facts)
            continue
        if facts.needs_stage2_depot_collect:
            outbound_members.append(facts)
            continue
        depot_stay_members.append(facts)

    demand_summary = Phase2DemandSummary(
        depot_stay_vehicle_nos=frozenset(facts.vehicle_no for facts in depot_stay_members),
        cun4_final_vehicle_nos=frozenset(facts.vehicle_no for facts in cun4_final_members),
        outbound_vehicle_nos=frozenset(facts.vehicle_no for facts in outbound_members),
        fixed_depot_resident_vehicle_nos=fixed_depot_resident_vehicle_nos,
    )
    outbound_groups = _build_phase2_outbound_groups(
        outbound_members=outbound_members,
        cun4_final_members=cun4_final_members,
    )
    track_layers = _build_phase2_track_layers(
        outbound_groups=outbound_groups,
    )
    execution_plan = _build_phase2_execution_plan(
        track_layers=track_layers,
        vehicle_traits={
            facts.vehicle_no: {
                "need_weigh": facts.need_weigh,
                "is_heavy": facts.is_heavy,
                "is_close_door": facts.is_close_door,
                "vehicle_length": facts.vehicle_length,
            }
            for facts in facts_list
        },
    )
    wave_plans = _build_phase2_wave_plans(
        facts_list=facts_list,
        demand_summary=demand_summary,
        yard_mode=yard_mode,
        vehicle_traits={
            facts.vehicle_no: {
                "need_weigh": facts.need_weigh,
                "is_heavy": facts.is_heavy,
                "is_close_door": facts.is_close_door,
                "vehicle_length": facts.vehicle_length,
            }
            for facts in facts_list
        },
    )
    diagnostics = {
        "depotStayVehicleNos": sorted(demand_summary.depot_stay_vehicle_nos),
        "depotStayVehicleCount": len(demand_summary.depot_stay_vehicle_nos),
        "cun4FinalVehicleNos": sorted(demand_summary.cun4_final_vehicle_nos),
        "cun4FinalVehicleCount": len(demand_summary.cun4_final_vehicle_nos),
        "depotOutboundVehicleNos": sorted(demand_summary.outbound_vehicle_nos),
        "depotOutboundVehicleCount": len(demand_summary.outbound_vehicle_nos),
        "fixedDepotResidentVehicleNos": sorted(demand_summary.fixed_depot_resident_vehicle_nos),
        "fixedDepotResidentVehicleCount": len(demand_summary.fixed_depot_resident_vehicle_nos),
        "outboundGroupCount": len(outbound_groups),
        "outboundGroups": [
            {
                "groupId": group.group_id,
                "groupKind": group.group_kind,
                "vehicleNos": list(group.vehicle_nos),
                "currentTrack": group.current_track,
                "sourceOrderStart": group.source_order_start,
                "finalTargetTrack": group.final_target_track,
                "finalFamily": group.final_family,
                "repairProcessProfile": list(group.repair_process_profile),
                "totalLengthM": group.total_length_m,
            }
            for group in outbound_groups
        ],
        "trackLayerCount": len(track_layers),
        "trackLayers": [
            {
                "sourceTrack": layer.source_track,
                "layerIndex": layer.layer_index,
                "groupIds": list(layer.group_ids),
                "vehicleNos": list(layer.vehicle_nos),
                "outboundVehicleNos": list(layer.outbound_vehicle_nos),
                "cun4FinalVehicleNos": list(layer.cun4_final_vehicle_nos),
                "exposedPrefixVehicleNos": list(layer.exposed_prefix_vehicle_nos),
                "totalLengthM": layer.total_length_m,
            }
            for layer in track_layers
        ],
        "executionPlan": None if execution_plan is None else dict(execution_plan.diagnostics),
        "wavePlanCount": len(wave_plans),
        "wavePlans": [
            {
                "waveName": wave.wave_name,
                "waveRole": wave.wave_role,
                "collectedVehicleNos": list(wave.collected_vehicle_nos),
                "attachUnits": [dict(item) for item in wave.attach_units],
                "diagnostics": dict(wave.diagnostics),
            }
            for wave in wave_plans
        ],
    }
    return Stage2Plan(
        demand_summary=demand_summary,
        outbound_groups=outbound_groups,
        track_layers=track_layers,
        execution_plan=execution_plan,
        wave_plans=wave_plans,
        diagnostics=diagnostics,
    )


def _build_phase2_outbound_groups(
    *,
    outbound_members: list[VehicleStageFacts],
    cun4_final_members: list[VehicleStageFacts],
) -> tuple[Phase2OutboundGroup, ...]:
    groups: list[Phase2OutboundGroup] = []
    group_index = 1
    active_members = sorted(
        [*outbound_members, *cun4_final_members],
        key=lambda facts: (facts.current_track, facts.current_order, facts.vehicle_no),
    )
    by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in active_members:
        by_track[facts.current_track].append(facts)
    for current_track in sorted(by_track, key=_phase2_depot_track_priority):
        track_members = by_track[current_track]
        current_run: list[VehicleStageFacts] = []
        current_kind = ""
        for facts in track_members:
            group_kind = "CUN4_FINAL" if facts.is_cun4bei_final else "DEPOT_OUTBOUND"
            if not current_run:
                current_run = [facts]
                current_kind = group_kind
                continue
            previous = current_run[-1]
            if group_kind == current_kind and _phase2_should_extend_run(
                previous=previous,
                current=facts,
                kind=current_kind,
            ):
                current_run.append(facts)
                continue
            groups.append(
                _build_phase2_outbound_group(
                    members=current_run,
                    group_kind=current_kind,
                    group_id=f"G{group_index:03d}",
                )
            )
            group_index += 1
            current_run = [facts]
            current_kind = group_kind
        if current_run:
            groups.append(
                _build_phase2_outbound_group(
                    members=current_run,
                    group_kind=current_kind,
                    group_id=f"G{group_index:03d}",
                )
            )
            group_index += 1
    return tuple(groups)


def _phase2_member_runs(
    members: list[VehicleStageFacts],
    *,
    kind: str,
) -> list[list[VehicleStageFacts]]:
    if not members:
        return []
    runs: list[list[VehicleStageFacts]] = []
    current_run: list[VehicleStageFacts] = []
    for facts in members:
        if not current_run:
            current_run = [facts]
            continue
        previous = current_run[-1]
        if _phase2_should_extend_run(previous=previous, current=facts, kind=kind):
            current_run.append(facts)
            continue
        runs.append(current_run)
        current_run = [facts]
    if current_run:
        runs.append(current_run)
    return runs


def _phase2_should_extend_run(
    *,
    previous: VehicleStageFacts,
    current: VehicleStageFacts,
    kind: str,
) -> bool:
    if current.current_track != previous.current_track:
        return False
    if current.current_order != previous.current_order + 1:
        return False
    if kind == "CUN4_FINAL":
        return (
            current.final_target_track == previous.final_target_track
            and current.repair_process == previous.repair_process
        )
    return (
        current.final_target_track == previous.final_target_track
        and current.final_family == previous.final_family
        and current.repair_process == previous.repair_process
    )


def _build_phase2_outbound_group(
    *,
    members: list[VehicleStageFacts],
    group_kind: str,
    group_id: str,
) -> Phase2OutboundGroup:
    head = members[0]
    if group_kind == "CUN4_FINAL":
        rank_key = _stage2_cun4bei_final_order_key(head)
    else:
        rank_key = _stage2_other_collect_order_key(head)
    return Phase2OutboundGroup(
        group_id=group_id,
        group_kind=group_kind,
        vehicle_nos=tuple(item.vehicle_no for item in members),
        current_track=head.current_track,
        source_order_start=head.current_order,
        final_target_track=head.final_target_track,
        final_family=head.final_family,
        repair_process_profile=tuple(item.repair_process for item in members),
        total_length_m=round(sum(item.vehicle_length for item in members), 1),
        rank_key=rank_key,
    )


def _build_phase2_track_layers(
    *,
    outbound_groups: tuple[Phase2OutboundGroup, ...],
) -> tuple[Phase2TrackLayer, ...]:
    if not outbound_groups:
        return ()
    groups_by_track: dict[str, list[Phase2OutboundGroup]] = defaultdict(list)
    for group in outbound_groups:
        groups_by_track[group.current_track].append(group)
    for track_groups in groups_by_track.values():
        track_groups.sort(key=lambda group: group.source_order_start)
    layers: list[Phase2TrackLayer] = []
    for current_track in sorted(groups_by_track, key=_phase2_depot_track_priority):
        track_groups = groups_by_track[current_track]
        prefix_vehicle_nos: list[str] = []
        prefix_group_ids: list[str] = []
        prefix_total_length_m = 0.0
        for depth, group in enumerate(track_groups, start=1):
            prefix_group_ids.append(group.group_id)
            prefix_vehicle_nos.extend(group.vehicle_nos)
            prefix_total_length_m += group.total_length_m
            outbound_vehicle_nos = tuple(group.vehicle_nos) if group.group_kind != "CUN4_FINAL" else ()
            cun4_final_vehicle_nos = tuple(group.vehicle_nos) if group.group_kind == "CUN4_FINAL" else ()
            layers.append(
                Phase2TrackLayer(
                    source_track=current_track,
                    layer_index=depth,
                    group_ids=(group.group_id,),
                    vehicle_nos=tuple(group.vehicle_nos),
                    total_length_m=group.total_length_m,
                    outbound_vehicle_nos=outbound_vehicle_nos,
                    cun4_final_vehicle_nos=cun4_final_vehicle_nos,
                    exposed_prefix_vehicle_nos=tuple(prefix_vehicle_nos),
                    diagnostics={
                        "sourceTrack": current_track,
                        "layerIndex": depth,
                        "groupIds": [group.group_id],
                        "vehicleNos": list(group.vehicle_nos),
                        "outboundVehicleNos": list(outbound_vehicle_nos),
                        "cun4FinalVehicleNos": list(cun4_final_vehicle_nos),
                        "totalLengthM": group.total_length_m,
                        "exposedPrefixVehicleNos": list(prefix_vehicle_nos),
                        "exposedPrefixLengthM": round(prefix_total_length_m, 1),
                    },
                )
            )
    return tuple(layers)


def _build_phase2_wave_plans(
    *,
    facts_list: list[VehicleStageFacts],
    demand_summary: Phase2DemandSummary,
    yard_mode: str,
    vehicle_traits: dict[str, dict[str, Any]],
) -> tuple[Phase2WavePlan, ...]:
    facts_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    active_vehicle_nos = [
        facts.vehicle_no
        for facts in sorted(
            facts_list,
            key=lambda item: (
                _phase2_depot_track_priority(item.current_track),
                item.current_order,
                item.vehicle_no,
            ),
        )
        if facts.is_depot_area_vehicle
        and not facts.is_fixed_depot_resident
        and (facts.is_cun4bei_final or (not facts.is_cun4bei_final and not facts.needs_depot_batch))
    ]
    if not active_vehicle_nos:
        return ()
    if len(active_vehicle_nos) > 20:
        return ()
    if not _phase2_hook_group_is_valid(tuple(active_vehicle_nos), vehicle_traits=vehicle_traits):
        return ()
    if _phase2_vehicle_length_m(tuple(active_vehicle_nos), vehicle_traits=vehicle_traits) > PHASE2_L1_TRANSFER_MAX_LENGTH_M:
        return ()

    grouped_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        if facts.is_depot_area_vehicle:
            grouped_by_track[facts.current_track].append(facts)
    for items in grouped_by_track.values():
        items.sort(key=lambda item: (item.current_order, item.vehicle_no))

    reorder_active_goals: dict[str, dict[str, Any]] = {}
    outbound_attach_units: list[dict[str, Any]] = []
    cun4_attach_units: list[dict[str, Any]] = []
    collected_vehicle_nos: list[str] = []
    reorder_needed = False
    for source_track in sorted(grouped_by_track, key=_phase2_depot_track_priority):
        track_items = grouped_by_track[source_track]
        fixed_suffix_start = len(track_items)
        for index, facts in enumerate(track_items):
            if facts.is_fixed_depot_resident:
                fixed_suffix_start = index
                break
        if any(
            not facts.is_fixed_depot_resident
            for facts in track_items[fixed_suffix_start:]
        ):
            return ()
        movable_prefix = [facts for facts in track_items[:fixed_suffix_start] if facts.vehicle_no in active_vehicle_nos]
        if not movable_prefix:
            continue
        ordered_prefix = sorted(
            movable_prefix,
            key=lambda facts: (
                0 if not facts.is_cun4bei_final else 1,
                facts.current_order,
                facts.vehicle_no,
            ),
        )
        if [facts.vehicle_no for facts in ordered_prefix] != [facts.vehicle_no for facts in movable_prefix]:
            reorder_needed = True
        if source_track in DEPOT_TARGET_TRACKS:
            available_spots = list_track_spots(source_track, yard_mode)
            if len(available_spots) < len(track_items):
                return ()
            for slot_index, facts in enumerate(ordered_prefix, start=1):
                reorder_active_goals[facts.vehicle_no] = {
                    "vehicleNo": facts.vehicle_no,
                    "targetTrack": source_track,
                    "targetMode": "SPOT",
                    "targetSpotCode": available_spots[slot_index - 1],
                    "targetSource": "PHASE2_REORDER_PREFIX",
                    "isSpotting": available_spots[slot_index - 1],
                }
        ordered_outbound_vehicle_nos = [
            facts.vehicle_no
            for facts in ordered_prefix
            if not facts.is_cun4bei_final
        ]
        ordered_cun4_vehicle_nos = [
            facts.vehicle_no
            for facts in ordered_prefix
            if facts.is_cun4bei_final
        ]
        if ordered_outbound_vehicle_nos:
            outbound_attach_units.append(
                {
                    "sourceTrack": source_track,
                    "vehicleNos": ordered_outbound_vehicle_nos,
                }
            )
            collected_vehicle_nos.extend(ordered_outbound_vehicle_nos)
        if ordered_cun4_vehicle_nos:
            cun4_attach_units.append(
                {
                    "sourceTrack": source_track,
                    "vehicleNos": ordered_cun4_vehicle_nos,
                }
            )
            collected_vehicle_nos.extend(ordered_cun4_vehicle_nos)

    attach_units = [*outbound_attach_units, *cun4_attach_units]
    if not attach_units:
        return ()
    waves: list[Phase2WavePlan] = []
    if reorder_needed and reorder_active_goals:
        waves.append(
            Phase2WavePlan(
                wave_name="phase2_reorder_prefix",
                wave_role="REORDER",
                collected_vehicle_nos=tuple(),
                attach_units=tuple(),
                diagnostics={
                    "activeGoalsByVehicle": dict(sorted(reorder_active_goals.items())),
                    "reorderVehicleNos": sorted(reorder_active_goals),
                    "outboundVehicleNos": sorted(
                        facts.vehicle_no
                        for facts in facts_by_vehicle.values()
                        if facts.vehicle_no in reorder_active_goals and not facts.is_cun4bei_final
                    ),
                    "cun4VehicleNos": sorted(
                        facts.vehicle_no
                        for facts in facts_by_vehicle.values()
                        if facts.vehicle_no in reorder_active_goals and facts.is_cun4bei_final
                    ),
                },
            )
        )
    waves.append(
        Phase2WavePlan(
            wave_name="phase2_export_single_wave",
            wave_role="EXPORT",
            collected_vehicle_nos=tuple(collected_vehicle_nos),
            attach_units=tuple(attach_units),
                diagnostics={
                    "activeGoalsByVehicle": {},
                    "collectedVehicleNos": list(collected_vehicle_nos),
                    "attachUnits": [dict(item) for item in attach_units],
                    "outboundAttachUnits": [dict(item) for item in outbound_attach_units],
                    "cun4AttachUnits": [dict(item) for item in cun4_attach_units],
                "outboundPrefixVehicleNos": sorted(
                    facts.vehicle_no
                    for facts in facts_by_vehicle.values()
                    if facts.vehicle_no in collected_vehicle_nos and not facts.is_cun4bei_final
                ),
                "cun4VehicleNos": sorted(
                    facts.vehicle_no
                    for facts in facts_by_vehicle.values()
                    if facts.vehicle_no in collected_vehicle_nos and facts.is_cun4bei_final
                ),
            },
        )
    )
    return tuple(waves)


def _build_phase2_execution_plan(
    *,
    track_layers: tuple[Phase2TrackLayer, ...],
    vehicle_traits: dict[str, dict[str, Any]],
) -> Phase2ExecutionPlan | None:
    if not track_layers:
        return None
    ordered_layers = tuple(
        sorted(
            track_layers,
            key=lambda layer: (
                _phase2_depot_track_priority(layer.source_track),
                layer.layer_index,
            ),
        )
    )
    must_pull_vehicle_nos = tuple(
        vehicle_no
        for layer in ordered_layers
        for vehicle_no in layer.outbound_vehicle_nos
    )
    must_pull_set = set(must_pull_vehicle_nos)
    predecessor_unlock_vehicle_nos = tuple(
        vehicle_no
        for source_track in {layer.source_track for layer in ordered_layers}
        for source_layers in (tuple(layer for layer in ordered_layers if layer.source_track == source_track),)
        for index, layer in enumerate(source_layers)
        if layer.outbound_vehicle_nos
        for previous_layer in source_layers[:index]
        for vehicle_no in previous_layer.vehicle_nos
        if vehicle_no not in must_pull_set
    )
    predecessor_unlock_set = set(predecessor_unlock_vehicle_nos)
    unlocking_optional_track_set = {
        layer.source_track
        for layer in ordered_layers
        if layer.outbound_vehicle_nos
    }
    optional_layers = [layer for layer in ordered_layers if layer.cun4_final_vehicle_nos]
    unlocking_optional_vehicle_nos = tuple(
        vehicle_no
        for layer in optional_layers
        if layer.source_track in unlocking_optional_track_set
        for vehicle_no in layer.cun4_final_vehicle_nos
    )
    pure_batch_optional_vehicle_nos = tuple(
        vehicle_no
        for layer in optional_layers
        if layer.source_track not in unlocking_optional_track_set
        for vehicle_no in layer.cun4_final_vehicle_nos
    )
    phase3_clearance_vehicle_nos = tuple(
        vehicle_no
        for layer in optional_layers
        if layer.source_track in {"修1", "修2", "修3", "修4"}
        for vehicle_no in layer.cun4_final_vehicle_nos
    )
    phase3_clearance_set = set(phase3_clearance_vehicle_nos)
    optional_vehicle_nos = tuple((*unlocking_optional_vehicle_nos, *pure_batch_optional_vehicle_nos))

    def _layer_selection_key(layer: Phase2TrackLayer) -> tuple[int, int, int, str]:
        layer_vehicle_nos = layer.vehicle_nos
        if any(vehicle_no in predecessor_unlock_set for vehicle_no in layer_vehicle_nos):
            tier = 0
        elif any(vehicle_no in must_pull_set for vehicle_no in layer_vehicle_nos):
            tier = 1
        elif any(vehicle_no in phase3_clearance_set for vehicle_no in layer_vehicle_nos):
            tier = 2
        else:
            tier = 3
        return (
            tier,
            _phase2_attach_selection_priority(layer.source_track),
            layer.layer_index,
            layer.source_track,
        )
    required_layer_vehicle_nos = predecessor_unlock_set | must_pull_set | phase3_clearance_set
    planned_layers = tuple(
        sorted(
            (
                layer
                for layer in ordered_layers
                if any(vehicle_no in required_layer_vehicle_nos for vehicle_no in layer.vehicle_nos)
            ),
            key=_layer_selection_key,
        )
    )
    selected_layers = _split_phase2_layers_for_hook_constraints(
        planned_layers,
        vehicle_traits=vehicle_traits,
    )
    selected_vehicle_nos = [
        vehicle_no
        for layer in selected_layers
        for vehicle_no in layer.vehicle_nos
    ]
    selected_vehicle_set = set(selected_vehicle_nos)
    deferred_tail_vehicle_nos = [
        vehicle_no
        for layer in ordered_layers
        for vehicle_no in layer.vehicle_nos
        if vehicle_no not in selected_vehicle_set
    ]
    reserved_cun4_capacity_m = 60.0
    selected_length_m = sum(layer.total_length_m for layer in selected_layers)
    collection_batches = _build_phase2_collection_batches(
        selected_layers,
        vehicle_traits=vehicle_traits,
    )
    transfer_vehicle_nos = tuple(selected_vehicle_nos)
    deferred_tail_tuple = tuple(deferred_tail_vehicle_nos)
    return Phase2ExecutionPlan(
        execution_name="phase2_collect_then_transfer",
        track_layers=tuple(selected_layers),
        collection_batches=collection_batches,
        predecessor_unlock_vehicle_nos=predecessor_unlock_vehicle_nos,
        must_pull_vehicle_nos=must_pull_vehicle_nos,
        unlocking_optional_vehicle_nos=unlocking_optional_vehicle_nos,
        phase3_clearance_vehicle_nos=phase3_clearance_vehicle_nos,
        pure_batch_optional_vehicle_nos=pure_batch_optional_vehicle_nos,
        optional_cun4_vehicle_nos=optional_vehicle_nos,
        deferred_tail_vehicle_nos=deferred_tail_tuple,
        transfer_vehicle_nos=transfer_vehicle_nos,
        reserved_cun4_capacity_m=reserved_cun4_capacity_m,
        diagnostics={
            "executionName": "phase2_collect_then_transfer",
            "sourceTrackCount": len(selected_layers),
            "sourceTracks": [layer.source_track for layer in selected_layers],
            "collectionBatches": [list(batch) for batch in collection_batches],
            "collectionBatchCount": len(collection_batches),
            "predecessorUnlockVehicleNos": list(predecessor_unlock_vehicle_nos),
            "predecessorUnlockVehicleCount": len(predecessor_unlock_vehicle_nos),
            "mustPullVehicleNos": list(must_pull_vehicle_nos),
            "mustPullVehicleCount": len(must_pull_vehicle_nos),
            "unlockingOptionalVehicleNos": list(unlocking_optional_vehicle_nos),
            "unlockingOptionalVehicleCount": len(unlocking_optional_vehicle_nos),
            "phase3ClearanceVehicleNos": list(phase3_clearance_vehicle_nos),
            "phase3ClearanceVehicleCount": len(phase3_clearance_vehicle_nos),
            "pureBatchOptionalVehicleNos": list(pure_batch_optional_vehicle_nos),
            "pureBatchOptionalVehicleCount": len(pure_batch_optional_vehicle_nos),
            "optionalCun4VehicleNos": list(optional_vehicle_nos),
            "optionalCun4VehicleCount": len(optional_vehicle_nos),
            "deferredTailVehicleNos": list(deferred_tail_tuple),
            "deferredTailVehicleCount": len(deferred_tail_tuple),
            "transferVehicleNos": list(transfer_vehicle_nos),
            "transferVehicleCount": len(transfer_vehicle_nos),
            "reservedCun4CapacityM": reserved_cun4_capacity_m,
            "selectedLengthM": round(selected_length_m, 1),
            "trackLayers": [dict(layer.diagnostics) for layer in selected_layers],
        },
    )


def _split_phase2_layers_for_hook_constraints(
    layers: tuple[Phase2TrackLayer, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> tuple[Phase2TrackLayer, ...]:
    split_layers: list[Phase2TrackLayer] = []
    for layer in layers:
        chunks = _phase2_vehicle_chunks_for_hook_constraints(
            layer.vehicle_nos,
            vehicle_traits=vehicle_traits,
        )
        if len(chunks) == 1 and chunks[0] == layer.vehicle_nos:
            split_layers.append(layer)
            continue
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_set = set(chunk)
            outbound_vehicle_nos = tuple(
                vehicle_no for vehicle_no in layer.outbound_vehicle_nos if vehicle_no in chunk_set
            )
            cun4_final_vehicle_nos = tuple(
                vehicle_no for vehicle_no in layer.cun4_final_vehicle_nos if vehicle_no in chunk_set
            )
            total_length_m = round(
                sum(float(vehicle_traits[vehicle_no].get("vehicle_length") or 0.0) for vehicle_no in chunk),
                1,
            )
            split_layers.append(
                Phase2TrackLayer(
                    source_track=layer.source_track,
                    layer_index=layer.layer_index,
                    group_ids=tuple(f"{group_id}::P{chunk_index}" for group_id in layer.group_ids),
                    vehicle_nos=chunk,
                    total_length_m=total_length_m,
                    outbound_vehicle_nos=outbound_vehicle_nos,
                    cun4_final_vehicle_nos=cun4_final_vehicle_nos,
                    exposed_prefix_vehicle_nos=chunk,
                    diagnostics={
                        **dict(layer.diagnostics),
                        "vehicleNos": list(chunk),
                        "outboundVehicleNos": list(outbound_vehicle_nos),
                        "cun4FinalVehicleNos": list(cun4_final_vehicle_nos),
                        "totalLengthM": total_length_m,
                        "splitForPhase2HookConstraints": True,
                        "splitPartIndex": chunk_index,
                    },
                )
            )
    return tuple(split_layers)


def _phase2_vehicle_chunks_for_hook_constraints(
    vehicle_nos: tuple[str, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    chunks: list[tuple[str, ...]] = []
    current: list[str] = []
    for vehicle_no in vehicle_nos:
        candidate = tuple([*current, vehicle_no])
        if current and not _phase2_transfer_group_is_valid(candidate, vehicle_traits=vehicle_traits):
            chunks.append(tuple(current))
            current = [vehicle_no]
            if not _phase2_transfer_group_is_valid(tuple(current), vehicle_traits=vehicle_traits):
                raise ValueError(f"phase2 vehicle cannot form legal transfer group: {vehicle_no}")
            continue
        current.append(vehicle_no)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _build_phase2_collection_batches(
    layers: tuple[Phase2TrackLayer, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    for layer in layers:
        layer_vehicle_nos = list(layer.vehicle_nos)
        candidate = tuple([*current, *layer_vehicle_nos])
        if current and not _phase2_transfer_group_is_valid(candidate, vehicle_traits=vehicle_traits):
            batches.append(tuple(current))
            current = layer_vehicle_nos
            if not _phase2_transfer_group_is_valid(tuple(current), vehicle_traits=vehicle_traits):
                raise ValueError(f"phase2 layer cannot form legal transfer group: {layer.vehicle_nos}")
            continue
        current.extend(layer_vehicle_nos)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _phase2_transfer_group_is_valid(
    vehicle_nos: tuple[str, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> bool:
    if not _phase2_hook_group_is_valid(vehicle_nos, vehicle_traits=vehicle_traits):
        return False
    return _phase2_vehicle_length_m(vehicle_nos, vehicle_traits=vehicle_traits) <= PHASE2_L1_TRANSFER_MAX_LENGTH_M


def _phase2_vehicle_length_m(
    vehicle_nos: tuple[str, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> float:
    return round(sum(float(vehicle_traits[vehicle_no].get("vehicle_length") or 0.0) for vehicle_no in vehicle_nos), 1)


def _phase2_hook_group_is_valid(
    vehicle_nos: tuple[str, ...],
    *,
    vehicle_traits: dict[str, dict[str, Any]],
) -> bool:
    if not vehicle_nos:
        return True
    heavy_count = sum(1 for vehicle_no in vehicle_nos if vehicle_traits[vehicle_no].get("is_heavy"))
    empty_count = len(vehicle_nos) - heavy_count
    equivalent_empty_count = empty_count + 4 * heavy_count
    if heavy_count == 0 and empty_count > 20:
        return False
    if heavy_count > 2:
        return False
    if heavy_count > 0 and equivalent_empty_count > 20:
        return False
    if (
        len(vehicle_nos) >= 2
        and vehicle_traits[vehicle_nos[0]].get("is_close_door")
        and any(vehicle_traits[vehicle_no].get("is_heavy") for vehicle_no in vehicle_nos[1:])
    ):
        return False
    weigh_count = sum(1 for vehicle_no in vehicle_nos if vehicle_traits[vehicle_no].get("need_weigh"))
    if weigh_count > 1:
        return False
    if weigh_count == 1 and not vehicle_traits[vehicle_nos[-1]].get("need_weigh"):
        return False
    return True


def _build_phase1_plan(
    facts_list: list[VehicleStageFacts],
    master: MasterData,
    *,
    respect_existing_buffer_occupancy: bool = True,
    buffer_occupancy_exempt_vehicle_nos: frozenset[str] | None = None,
) -> Phase1Plan:
    source_plans = _build_phase1_source_track_plans(facts_list)
    reachable_depot_set = _build_phase1_reachable_depot_set(source_plans)
    backbone_plan = _solve_phase1_backbone_plan(
        facts_list=facts_list,
        source_plans=source_plans,
        reachable_depot_set=reachable_depot_set,
        depot_slot_limit=None,
        master=master,
        respect_existing_buffer_occupancy=respect_existing_buffer_occupancy,
        buffer_occupancy_exempt_vehicle_nos=buffer_occupancy_exempt_vehicle_nos,
    )
    wave_plans, finish_plan, finish_goal_overrides = _build_phase1_wave_plans(
        facts_list=facts_list,
        source_plans=source_plans,
        backbone_plan=backbone_plan,
        master=master,
    )
    deferred_vehicle_nos = frozenset(
        vehicle_no
        for vehicle_no in reachable_depot_set
        if vehicle_no not in backbone_plan.selected_buffer_assignment
    )
    _validate_phase1_staging_contract(
        facts_list=facts_list,
        reachable_depot_set=reachable_depot_set,
        buffer_assignment=backbone_plan.selected_buffer_assignment,
        goal_overrides=finish_goal_overrides,
    )
    diagnostics = _build_phase1_block_diagnostics(
        facts_list=facts_list,
        source_plans=source_plans,
        reachable_depot_set=reachable_depot_set,
        backbone_plan=backbone_plan,
        finish_plan=finish_plan,
        deferred_vehicle_nos=deferred_vehicle_nos,
        goal_overrides=finish_goal_overrides,
        buffer_assignment=backbone_plan.selected_buffer_assignment,
        target_rank_by_vehicle=backbone_plan.target_rank_by_vehicle,
        master=master,
    )
    selected_vehicle_nos = frozenset(
        vehicle_no
        for vehicle_no, track_name in backbone_plan.selected_buffer_assignment.items()
        if track_name in JI_BUFFER_TRACKS
    )
    return Phase1Plan(
        selected_vehicle_nos=selected_vehicle_nos,
        deferred_vehicle_nos=deferred_vehicle_nos,
        buffer_assignment=dict(backbone_plan.selected_buffer_assignment),
        target_rank_by_vehicle=dict(backbone_plan.target_rank_by_vehicle),
        goal_overrides=dict(finish_goal_overrides),
        wave_plans=wave_plans,
        diagnostics=dict(diagnostics),
    )


def _phase2_attach_selection_priority(source_track: str) -> int:
    return {
        "轮": 0,
        "修4库外": 1,
        "修3库外": 2,
        "修2库外": 3,
        "修1库外": 4,
        "修4": 5,
        "修3": 6,
        "修2": 7,
        "修1": 8,
    }.get(source_track, 99)


def _build_phase1_demand_summary(
    *,
    facts_list: list[VehicleStageFacts],
    master: MasterData,
) -> Phase1DemandSummary:
    depot_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.needs_depot_batch and not facts.is_depot_area_vehicle
    )
    cun4_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in {"存4北", "存4南"}
    )
    ji_non_depot_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch
    )
    total_non_depot_region_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in PHASE1_NON_DEPOT_REGION_TRACKS and not facts.is_depot_area_vehicle
    )
    depot_demand_total_length_m = round(
        sum(facts.vehicle_length for facts in facts_list if facts.vehicle_no in depot_vehicle_nos),
        1,
    )
    ji_capacity_total_m = round(
        sum(
            min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track])
            for track in JI_BUFFER_TRACKS
        ),
        1,
    )
    return Phase1DemandSummary(
        depot_vehicle_nos=depot_vehicle_nos,
        cun4_vehicle_nos=cun4_vehicle_nos,
        ji_non_depot_vehicle_nos=ji_non_depot_vehicle_nos,
        total_non_depot_region_vehicle_nos=total_non_depot_region_vehicle_nos,
        depot_demand_total_length_m=depot_demand_total_length_m,
        ji_capacity_total_m=ji_capacity_total_m,
        ji_overflow_m=round(max(0.0, depot_demand_total_length_m - ji_capacity_total_m), 1),
    )


def _build_phase1_layout_packages(
    *,
    facts_list: list[VehicleStageFacts],
    master: MasterData,
) -> list[Phase1LayoutPackage]:
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        if facts.is_depot_area_vehicle:
            continue
        facts_by_track[facts.current_track].append(facts)
    package_index = 1
    packages: list[Phase1LayoutPackage] = []
    for source_track, members in sorted(facts_by_track.items()):
        ordered = sorted(members, key=lambda item: (item.current_order, item.vehicle_no))
        staged_members = _build_phase1_source_stage_members(source_track=source_track, ordered_members=ordered)
        chunks: list[tuple[tuple[str, str, str], list[tuple[VehicleStageFacts, str, str, str]]]] = []
        current_signature: tuple[str, str, str] | None = None
        current_chunk: list[tuple[VehicleStageFacts, str, str, str]] = []
        current_length = 0.0
        for item in staged_members:
            facts, package_kind, target_track, target_source = item
            signature = (package_kind, target_track, target_source)
            length_limit = (
                PHASE1_DEPOT_PACKAGE_MAX_REQUIRED_M
                if package_kind == "depot_batch"
                else PHASE1_TEMP_PACKAGE_MAX_LENGTH_M
            )
            projected_length = current_length + facts.vehicle_length
            if (
                current_chunk
                and (
                    signature != current_signature
                    or _phase1_should_split_layout_chunk(
                        current_chunk=current_chunk,
                        next_facts=facts,
                        target_track=target_track,
                        projected_length=projected_length,
                        length_limit=length_limit,
                    )
                )
            ):
                chunks.append((current_signature or ("", "", ""), current_chunk))
                current_signature = None
                current_chunk = []
                current_length = 0.0
            current_chunk.append(item)
            current_signature = signature
            current_length += facts.vehicle_length
        if current_chunk:
            chunks.append((current_signature or ("", "", ""), current_chunk))
        chain_id = f"CHAIN::{source_track}"
        source_backbone_chunk_count = sum(
            1
            for signature, _ in chunks
            if signature[0] == "depot_batch"
        )
        source_backbone_vehicle_count = sum(
            len(chunk)
            for signature, chunk in chunks
            if signature[0] == "depot_batch"
        )
        backbone_chunk_index = 0
        for signature, chunk in chunks:
            package_kind, target_track, target_source = signature
            chunk_members = [facts for facts, *_ in chunk]
            uses_buffer = package_kind == "depot_batch"
            if uses_buffer:
                backbone_chunk_index += 1
            segment_role = _phase1_backbone_segment_role(
                uses_buffer=uses_buffer,
                segment_index=backbone_chunk_index,
                segment_count=source_backbone_chunk_count,
            )
            final_family = chunk_members[0].final_family
            min_spot_priority = min(_final_spot_priority(item.final_target_spot) for item in chunk_members)
            reason_tags = tuple(
                sorted(
                    {
                        _pressure_cut_name(item)
                        for item in chunk_members
                    }
                    | {
                        package_kind,
                        segment_role,
                    }
                    | set(_phase1_special_reason_tags(chunk_members, target_track))
                )
            )
            execution_layer = _phase1_execution_layer(
                package_kind=package_kind,
                target_source=target_source,
            )
            source_chain_role = _phase1_source_chain_role(source_track=source_track)
            packages.append(
                Phase1LayoutPackage(
                    package_id=f"P{package_index:03d}",
                    chain_id=chain_id,
                    package_kind=package_kind,
                    source_track=source_track,
                    vehicle_nos=tuple(item.vehicle_no for item in chunk_members),
                    total_length_m=round(sum(item.vehicle_length for item in chunk_members), 1),
                    target_track=target_track,
                    target_source=target_source,
                    final_family=final_family,
                    min_spot_priority=min_spot_priority,
                    source_order_start=min(item.current_order for item in chunk_members),
                    source_order_end=max(item.current_order for item in chunk_members),
                    buffer_preference=(
                        _phase1_buffer_preference(
                            final_family=final_family,
                            source_track=source_track,
                            current_track=chunk_members[0].current_track,
                        )
                        if uses_buffer
                        else tuple()
                    ),
                    uses_buffer=uses_buffer,
                    pressure_cut=_pressure_cut_name(chunk_members[0]),
                    reason_tags=reason_tags,
                    execution_layer=execution_layer,
                    complexity_cost=_phase1_package_complexity_cost(
                        package_kind=package_kind,
                        source_track=source_track,
                        vehicle_count=len(chunk_members),
                        uses_buffer=uses_buffer,
                        execution_layer=execution_layer,
                    ),
                    source_chain_role=source_chain_role,
                    is_required_for_backbone=execution_layer in {"L1_BACKBONE", "L2_REQUIRED_CLEAR"},
                    segment_role=segment_role,
                    source_segment_index=backbone_chunk_index if uses_buffer else 0,
                    source_segment_count=source_backbone_chunk_count if uses_buffer else 0,
                    source_total_vehicle_count=source_backbone_vehicle_count if uses_buffer else 0,
                    requires_previous_segment=uses_buffer and backbone_chunk_index >= 2,
                )
            )
            package_index += 1
    packages.sort(
        key=lambda item: (
            _phase1_execution_layer_priority(item.execution_layer),
            _depot_topology_entry_priority(item.final_family),
            _phase1_source_chain_priority(item.source_chain_role),
            item.min_spot_priority,
            _phase1_track_priority(item.source_track),
            item.source_order_start,
            item.package_id,
        )
    )
    return packages


def _phase1_should_split_layout_chunk(
    *,
    current_chunk: list[tuple[VehicleStageFacts, str, str, str]],
    next_facts: VehicleStageFacts,
    target_track: str,
    projected_length: float,
    length_limit: float,
) -> bool:
    projected_count = len(current_chunk) + 1
    projected_required_length = projected_length + max(1.0, projected_count - 1)
    if projected_required_length > length_limit:
        return True
    current_members = [facts for facts, *_ in current_chunk]
    if next_facts.need_weigh or any(item.need_weigh for item in current_members):
        return True
    if target_track == "存4北" and (
        next_facts.is_close_door or any(item.is_close_door for item in current_members)
    ):
        return True
    heavy_count = sum(1 for item in current_members if item.is_heavy)
    if next_facts.is_heavy and heavy_count >= 2:
        return True
    if next_facts.is_heavy and any(item.is_close_door for item in current_members):
        return True
    if next_facts.is_close_door and any(item.is_heavy for item in current_members):
        return True
    return False


def _phase1_special_reason_tags(
    members: list[VehicleStageFacts],
    target_track: str,
) -> tuple[str, ...]:
    tags: list[str] = []
    if any(item.need_weigh for item in members):
        tags.append("need_weigh_singleton")
    if any(item.is_heavy for item in members):
        tags.append("heavy_cap")
    if target_track == "存4北" and any(item.is_close_door for item in members):
        tags.append("close_door_cun4_singleton")
    return tuple(tags)


def _phase1_backbone_segment_role(
    *,
    uses_buffer: bool,
    segment_index: int,
    segment_count: int,
) -> str:
    if not uses_buffer:
        return "cleanup"
    if segment_count <= 1 or segment_index <= 1:
        return "core"
    if segment_index == 2:
        return "follow"
    return "tail"


def _phase1_execution_layer(
    *,
    package_kind: str,
    target_source: str,
) -> str:
    if package_kind == "depot_batch":
        return "L1_BACKBONE"
    if target_source in {
        "PHASE1_CLEAR_CUN4",
        "PHASE1_CLEAR_JI",
        PHASE1_BLOCKER_BUCKET_WORK,
        PHASE1_BLOCKER_BUCKET_YARD,
    }:
        return "L2_REQUIRED_CLEAR"
    return "L3_OPTIONAL_CLEANUP"


def _phase1_execution_layer_priority(layer: str) -> int:
    return {
        "L1_BACKBONE": 0,
        "L2_REQUIRED_CLEAR": 1,
        "L3_OPTIONAL_CLEANUP": 2,
    }.get(layer, 9)


def _phase1_source_chain_role(*, source_track: str) -> str:
    if source_track in WASH_CONFLICT_TRACKS:
        return "wash_gate"
    if source_track in {"调棚", "预修"}:
        return "work_gate"
    if source_track in {"抛", "调北", "机棚"}:
        return "work_support"
    if source_track in {"存5北", "存5南"}:
        return "receiving_storage"
    if source_track in {"存1", "存2", "存3"}:
        return "yard_storage"
    if source_track in {"存4北", "存4南"}:
        return "cun4_clear"
    if source_track in JI_BUFFER_TRACKS:
        return "ji_clear"
    return "other"


def _phase1_source_chain_priority(role: str) -> int:
    return {
        "wash_gate": 0,
        "work_gate": 1,
        "work_support": 2,
        "receiving_storage": 3,
        "yard_storage": 4,
        "cun4_clear": 5,
        "ji_clear": 6,
        "other": 9,
    }.get(role, 9)


def _phase1_is_hot_source_role(role: str) -> bool:
    return role in {"wash_gate", "work_gate", "work_support"}


def _phase1_is_storage_source_role(role: str) -> bool:
    return role in {"receiving_storage", "yard_storage", "cun4_clear", "ji_clear"}


def _phase1_package_complexity_cost(
    *,
    package_kind: str,
    source_track: str,
    vehicle_count: int,
    uses_buffer: bool,
    execution_layer: str,
) -> int:
    cost = vehicle_count
    if uses_buffer:
        cost += 2
    if package_kind == "temp_repark":
        cost += 2
    if execution_layer == "L3_OPTIONAL_CLEANUP":
        cost += 2
    if source_track in WASH_CONFLICT_TRACKS:
        cost += 1
    return cost


def _build_phase1_source_stage_members(
    *,
    source_track: str,
    ordered_members: list[VehicleStageFacts],
) -> list[tuple[VehicleStageFacts, str, str, str]]:
    first_depot_index: int | None = None
    for index, facts in enumerate(ordered_members):
        if facts.needs_depot_batch:
            first_depot_index = index
            break
    staged: list[tuple[VehicleStageFacts, str, str, str]] = []
    for index, facts in enumerate(ordered_members):
        if facts.current_track in JI_BUFFER_TRACKS and facts.needs_depot_batch:
            staged.append((facts, "depot_batch", facts.current_track, "PHASE1_BACKBONE_PLACE"))
            continue
        if facts.needs_depot_batch:
            staged.append((facts, "depot_batch", "", "PHASE1_BACKBONE_PLACE"))
            continue
        has_depot_suffix = first_depot_index is not None and index < first_depot_index
        staged_target = _phase1_non_depot_stage_target(
            facts=facts,
            source_track=source_track,
            has_depot_suffix=has_depot_suffix,
        )
        if staged_target is None:
            continue
        staged.append((facts, staged_target[0], staged_target[1], staged_target[2]))
    return staged


def _phase1_non_depot_stage_target(
    *,
    facts: VehicleStageFacts,
    source_track: str,
    has_depot_suffix: bool,
) -> tuple[str, str, str] | None:
    if facts.is_depot_area_vehicle:
        return None
    if facts.current_track in {"存4北", "存4南"}:
        if (
            not facts.needs_depot_batch
            and facts.final_target_track not in {"存4北", "存4南"}
            and facts.final_target_track not in DEPOT_TARGET_TRACKS
            and facts.final_target_track not in DEPOT_OUTER_TRACKS
            and facts.final_target_track != facts.current_track
        ):
            return ("local_finish", facts.final_target_track, "PHASE1_CLEAR_CUN4")
        return ("temp_repark", "", "PHASE1_CLEAR_CUN4")
    if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch:
        if (
            facts.final_target_track not in {"存4北", "存4南"}
            and facts.final_target_track not in DEPOT_TARGET_TRACKS
            and facts.final_target_track not in DEPOT_OUTER_TRACKS
            and facts.final_target_track != facts.current_track
        ):
            return ("local_finish", facts.final_target_track, "PHASE1_CLEAR_JI")
        return ("temp_repark", "", "PHASE1_CLEAR_JI")
    can_leave_source = (
        not facts.needs_depot_batch
        and facts.final_target_track not in DEPOT_TARGET_TRACKS
        and facts.final_target_track not in DEPOT_OUTER_TRACKS
        and facts.final_target_track != facts.current_track
    )
    if has_depot_suffix:
        return ("temp_repark", "", _phase1_blocker_bucket_target_source(facts=facts, source_track=source_track))
    if can_leave_source and _phase1_should_keep_optional_local_finish(facts=facts, source_track=source_track):
        return ("local_finish", facts.final_target_track, "PHASE1_LOCAL_FINISH")
    return None


def _phase1_blocker_bucket_target_source(
    *,
    facts: VehicleStageFacts,
    source_track: str,
) -> str:
    if (
        facts.final_target_track in STORAGE_TRACKS
        or facts.final_target_track in JI_BUFFER_TRACKS
        or source_track in STORAGE_TRACKS
        or source_track in JI_BUFFER_TRACKS
    ):
        return PHASE1_BLOCKER_BUCKET_YARD
    return PHASE1_BLOCKER_BUCKET_WORK


def _phase1_should_keep_optional_local_finish(
    *,
    facts: VehicleStageFacts,
    source_track: str,
) -> bool:
    return source_track in {"存4北", "存4南"} or facts.current_track in JI_BUFFER_TRACKS


def _phase1_buffer_preference(
    *,
    final_family: str,
    source_track: str,
    current_track: str,
) -> tuple[str, ...]:
    if current_track in JI_BUFFER_TRACKS:
        ordered = [current_track] + [track for track in JI_BUFFER_TRACKS if track != current_track]
        return tuple(ordered)
    if final_family == "轮":
        return ("机北3", "机北2", "机北1", "机棚", "机南")
    if final_family in {"修1", "修2"}:
        return ("机南", "机棚", "机北1", "机北2", "机北3")
    return ("机棚", "机北1", "机北2", "机北3", "机南")


def _build_phase1_source_opening_summaries(
    *,
    packages: list[Phase1LayoutPackage],
) -> list[Phase1SourceOpeningSummary]:
    admissions = _build_phase1_source_admissions(packages=packages)
    return [
        Phase1SourceOpeningSummary(
            source_track=admission.source_track,
            source_chain_role=admission.source_chain_role,
            backbone_packages=admission.backbone_packages,
            required_cleanup_packages=admission.required_cleanup_packages,
            optional_cleanup_packages=admission.optional_cleanup_packages,
            backbone_vehicle_count=admission.backbone_vehicle_count,
            required_cleanup_vehicle_count=admission.required_cleanup_vehicle_count,
            required_clearance_gain_units=admission.required_clearance_gain_units,
            opening_cost_units=admission.opening_cost_units,
            opening_gain_units=admission.opening_gain_units,
            opening_score=admission.opening_score,
        )
        for admission in admissions
    ]


def _build_phase1_source_admissions(
    *,
    packages: list[Phase1LayoutPackage],
) -> list[Phase1SourceAdmission]:
    packages_by_source: dict[str, list[Phase1LayoutPackage]] = defaultdict(list)
    for package in packages:
        packages_by_source[package.source_track].append(package)
    admissions: list[Phase1SourceAdmission] = []
    for source_track, source_packages in packages_by_source.items():
        backbone_packages = tuple(package for package in source_packages if package.uses_buffer)
        required_cleanup_packages = tuple(
            package
            for package in source_packages
            if not package.uses_buffer and package.execution_layer == "L2_REQUIRED_CLEAR"
        )
        optional_cleanup_packages = tuple(
            package
            for package in source_packages
            if not package.uses_buffer and package.execution_layer == "L3_OPTIONAL_CLEANUP"
        )
        if backbone_packages:
            role = backbone_packages[0].source_chain_role
        elif required_cleanup_packages:
            role = required_cleanup_packages[0].source_chain_role
        else:
            role = optional_cleanup_packages[0].source_chain_role if optional_cleanup_packages else "other"
        backbone_vehicle_count = sum(len(package.vehicle_nos) for package in backbone_packages)
        required_cleanup_vehicle_count = sum(len(package.vehicle_nos) for package in required_cleanup_packages)
        required_clearance_gain_units = sum(
            max(
                len(package.vehicle_nos),
                int(getattr(package, "complexity_cost", 0) or 0),
            )
            for package in required_cleanup_packages
            if package.source_chain_role in {"receiving_storage", "yard_storage", "cun4_clear"}
        )
        opening_cost_units = len(backbone_packages) + (
            _phase1_source_opening_cleanup_multiplier(role) * len(required_cleanup_packages)
        )
        opening_gain_units = backbone_vehicle_count * 2 + required_clearance_gain_units
        opening_score = opening_gain_units - opening_cost_units
        primary_packages = _phase1_primary_backbone_packages(backbone_packages)
        primary_package_ids = tuple(package.package_id for package in primary_packages)
        primary_vehicle_count = sum(len(package.vehicle_nos) for package in primary_packages)
        primary_required_length_m = round(
            sum(package.total_length_m + max(1.0, len(package.vehicle_nos) - 1) for package in primary_packages),
            1,
        )
        admissions.append(
            Phase1SourceAdmission(
                source_track=source_track,
                source_chain_role=role,
                backbone_packages=backbone_packages,
                required_cleanup_packages=required_cleanup_packages,
                optional_cleanup_packages=optional_cleanup_packages,
                backbone_vehicle_count=backbone_vehicle_count,
                required_cleanup_vehicle_count=required_cleanup_vehicle_count,
                required_clearance_gain_units=required_clearance_gain_units,
                opening_cost_units=opening_cost_units,
                opening_gain_units=opening_gain_units,
                opening_score=opening_score,
                primary_package_ids=primary_package_ids,
                primary_vehicle_count=primary_vehicle_count,
                primary_required_length_m=primary_required_length_m,
                admission_tier=_phase1_source_admission_tier(
                    source_chain_role=role,
                    backbone_vehicle_count=backbone_vehicle_count,
                    required_clearance_gain_units=required_clearance_gain_units,
                    opening_score=opening_score,
                ),
            )
        )
    admissions.sort(key=_phase1_source_admission_priority)
    return admissions


def _phase1_primary_backbone_packages(
    backbone_packages: tuple[Phase1LayoutPackage, ...],
) -> tuple[Phase1LayoutPackage, ...]:
    if not backbone_packages:
        return ()
    core_packages = tuple(package for package in backbone_packages if package.segment_role == "core")
    if core_packages:
        return core_packages
    packages_by_family: dict[str, list[Phase1LayoutPackage]] = defaultdict(list)
    for package in backbone_packages:
        packages_by_family[package.final_family].append(package)
    primary_packages: list[Phase1LayoutPackage] = []
    for family_packages in packages_by_family.values():
        primary_packages.append(
            max(
                family_packages,
                key=lambda item: (
                    len(item.vehicle_nos),
                    item.total_length_m + max(1.0, len(item.vehicle_nos) - 1),
                    -item.source_order_start,
                    item.package_id,
                ),
            )
        )
    primary_packages.sort(
        key=lambda item: (
            _depot_topology_entry_priority(item.final_family),
            item.min_spot_priority,
            item.source_order_start,
            item.package_id,
        ),
    )
    return tuple(primary_packages)


def _phase1_source_admission_tier(
    *,
    source_chain_role: str,
    backbone_vehicle_count: int,
    required_clearance_gain_units: int,
    opening_score: int,
) -> str:
    if backbone_vehicle_count <= 0 and required_clearance_gain_units > 0:
        if source_chain_role in {"receiving_storage", "yard_storage", "cun4_clear"}:
            return "clearance_required"
        return "weak"
    if backbone_vehicle_count <= 0:
        return "defer"
    if source_chain_role in {"wash_gate", "work_gate", "ji_clear"}:
        return "core"
    if source_chain_role in {"work_support", "other"} and backbone_vehicle_count >= 2:
        return "strong"
    threshold = 0
    if source_chain_role in {"receiving_storage", "yard_storage", "cun4_clear"}:
        threshold = 1
    return "strong" if opening_score >= threshold else "weak"


def _phase1_source_admission_priority(admission: Phase1SourceAdmission) -> tuple[Any, ...]:
    return (
        {"core": 0, "strong": 1, "clearance_required": 2, "weak": 3, "defer": 4}.get(admission.admission_tier, 9),
        _phase1_source_chain_priority(admission.source_chain_role),
        -admission.backbone_vehicle_count,
        -admission.required_clearance_gain_units,
        -admission.primary_vehicle_count,
        admission.primary_required_length_m,
        len(admission.required_cleanup_packages),
        _phase1_track_priority(admission.source_track),
        admission.source_track,
    )


def _phase1_source_is_companion_eligible(admission: Phase1SourceAdmission) -> bool:
    if admission.source_chain_role not in {"receiving_storage", "yard_storage", "cun4_clear"}:
        return False
    if admission.required_cleanup_packages:
        return False
    if len(admission.backbone_packages) != 1:
        return False
    return admission.backbone_vehicle_count <= 1 and admission.opening_score >= 1


def _select_phase1_admission_plan(
    *,
    admissions: list[Phase1SourceAdmission],
    remaining_buffer: dict[str, float],
) -> Phase1AdmissionPlan:
    eligible = [admission for admission in admissions if admission.backbone_packages]
    total_capacity = round(sum(remaining_buffer.values()), 1)
    strong_candidates = [
        admission
        for admission in eligible
        if admission.admission_tier in {"core", "strong", "clearance_required"}
    ]
    fallback_candidates = [
        admission
        for admission in eligible
        if admission.admission_tier == "weak"
    ]
    ordered_candidates = strong_candidates + fallback_candidates
    best_score: tuple[int, int, int, int, int, float, int] | None = None
    best_selected: tuple[str, ...] = ()

    def dfs(index: int, selected: list[Phase1SourceAdmission]) -> None:
        nonlocal best_score, best_selected
        selected_slot_sources = [item for item in selected if not _phase1_source_is_companion_eligible(item)]
        if len(selected_slot_sources) > PHASE1_PRIMARY_BACKBONE_SLOT_COUNT + PHASE1_ELASTIC_BACKBONE_SLOT_COUNT:
            return
        if index >= len(ordered_candidates):
            if not selected:
                return
            selected_source_tracks = tuple(item.source_track for item in selected)
            selected_vehicle_count = sum(item.backbone_vehicle_count for item in selected)
            selected_primary_vehicle_count = sum(item.primary_vehicle_count for item in selected)
            selected_required_cleanup = sum(len(item.required_cleanup_packages) for item in selected)
            selected_primary_required_length = round(sum(item.primary_required_length_m for item in selected), 1)
            core_count = sum(1 for item in selected if item.admission_tier == "core")
            strong_count = sum(1 for item in selected if item.admission_tier == "strong")
            clearance_required_count = sum(1 for item in selected if item.admission_tier == "clearance_required")
            weak_count = sum(1 for item in selected if item.admission_tier == "weak")
            score = (
                core_count,
                strong_count,
                clearance_required_count,
                -weak_count,
                selected_vehicle_count,
                selected_primary_vehicle_count,
                -selected_required_cleanup,
                -selected_primary_required_length,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_selected = selected_source_tracks
            return

        remaining_candidates = ordered_candidates[index:]
        current_core = sum(1 for item in selected if item.admission_tier == "core")
        remaining_core = sum(1 for item in remaining_candidates if item.admission_tier == "core")
        if best_score is not None and current_core + remaining_core < best_score[0]:
            return

        candidate = ordered_candidates[index]
        next_primary_required = round(
            sum(item.primary_required_length_m for item in selected) + candidate.primary_required_length_m,
            1,
        )
        selected_slot_count = sum(
            1
            for item in selected
            if not _phase1_source_is_companion_eligible(item)
        )
        selected_strong_or_core = sum(
            1
            for item in selected
            if item.admission_tier in {"core", "strong"} and not _phase1_source_is_companion_eligible(item)
        )
        weak_selected = sum(1 for item in selected if item.admission_tier == "weak")
        can_take = next_primary_required <= total_capacity
        if _phase1_source_is_companion_eligible(candidate):
            can_take = can_take
        elif candidate.admission_tier in {"core", "strong", "clearance_required"}:
            can_take = can_take and selected_slot_count < (
                PHASE1_PRIMARY_BACKBONE_SLOT_COUNT + PHASE1_ELASTIC_BACKBONE_SLOT_COUNT
            )
        elif candidate.admission_tier == "weak":
            can_take = (
                can_take
                and selected_strong_or_core >= PHASE1_PRIMARY_BACKBONE_SLOT_COUNT
                and weak_selected < PHASE1_ELASTIC_BACKBONE_SLOT_COUNT
                and candidate.backbone_vehicle_count >= 2
                and candidate.opening_score >= 0
            )
        else:
            can_take = False
        if can_take:
            selected.append(candidate)
            dfs(index + 1, selected)
            selected.pop()
        dfs(index + 1, selected)

    dfs(0, [])
    admitted_source_tracks = best_selected
    admission_by_source = {admission.source_track: admission for admission in admissions}
    admitted_slot_sources = [
        source_track
        for source_track in admitted_source_tracks
        if not _phase1_source_is_companion_eligible(admission_by_source[source_track])
    ]
    companion_source_tracks = tuple(
        source_track
        for source_track in admitted_source_tracks
        if source_track not in admitted_slot_sources
    )
    primary_source_tracks = tuple(admitted_slot_sources[:PHASE1_PRIMARY_BACKBONE_SLOT_COUNT])
    elastic_source_tracks = tuple(admitted_slot_sources[PHASE1_PRIMARY_BACKBONE_SLOT_COUNT:])
    slot_index_by_source: dict[str, int] = {}
    slot_type_by_source: dict[str, str] = {}
    for index, source_track in enumerate(primary_source_tracks, start=1):
        slot_index_by_source[source_track] = index
        slot_type_by_source[source_track] = "primary"
    for offset, source_track in enumerate(elastic_source_tracks, start=len(primary_source_tracks) + 1):
        slot_index_by_source[source_track] = offset
        slot_type_by_source[source_track] = "elastic"
    for source_track in companion_source_tracks:
        slot_index_by_source[source_track] = len(primary_source_tracks) + len(elastic_source_tracks)
        slot_type_by_source[source_track] = "companion"
    rejection_reason_by_source: dict[str, str] = {}
    admitted_set = set(admitted_source_tracks)
    for admission in admissions:
        if admission.source_track in admitted_set:
            continue
        if not admission.backbone_packages:
            if admission.required_clearance_gain_units > 0:
                rejection_reason_by_source[admission.source_track] = "clearance_not_admitted"
            else:
                rejection_reason_by_source[admission.source_track] = "no_backbone"
        elif admission.admission_tier == "weak":
            rejection_reason_by_source[admission.source_track] = "weak_source"
        else:
            rejection_reason_by_source[admission.source_track] = "backbone_slot_limit"
    deferred_source_tracks = tuple(
        admission.source_track
        for admission in admissions
        if admission.source_track not in admitted_set
    )
    return Phase1AdmissionPlan(
        admitted_source_tracks=admitted_source_tracks,
        primary_source_tracks=primary_source_tracks,
        elastic_source_tracks=elastic_source_tracks,
        companion_source_tracks=companion_source_tracks,
        deferred_source_tracks=deferred_source_tracks,
        slot_index_by_source=slot_index_by_source,
        slot_type_by_source=slot_type_by_source,
        rejection_reason_by_source=rejection_reason_by_source,
    )


def _phase1_source_opening_cleanup_multiplier(role: str) -> int:
    if role in {"wash_gate", "work_gate", "ji_clear"}:
        return 1
    if role in {"work_support", "other"}:
        return 2
    return 3


def _phase1_source_opening_priority(summary: Phase1SourceOpeningSummary) -> tuple[Any, ...]:
    return (
        _phase1_source_chain_priority(summary.source_chain_role),
        -summary.opening_score,
        -summary.backbone_vehicle_count,
        len(summary.required_cleanup_packages),
        _phase1_track_priority(summary.source_track),
        summary.source_track,
    )


def _solve_phase1_layout(
    *,
    facts_list: list[VehicleStageFacts],
    demand_summary: Phase1DemandSummary,
    packages: list[Phase1LayoutPackage],
    master: MasterData,
) -> Phase1LayoutResult:
    facts_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    source_admissions = _build_phase1_source_admissions(packages=packages)
    buffer_assignment: dict[str, str] = {}
    deferred_vehicle_nos: set[str] = set()
    remaining_buffer = {
        track: min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track])
        for track in JI_BUFFER_TRACKS
    }
    buffer_track_sources = {
        track: set()
        for track in JI_BUFFER_TRACKS
    }
    buffer_track_families = {
        track: set()
        for track in JI_BUFFER_TRACKS
    }
    candidate_depot_packages: list[Phase1LayoutPackage] = []
    selected_source_tracks: set[str] = set()
    selected_hot_source_tracks: set[str] = set()
    selected_storage_source_tracks: set[str] = set()
    selected_package_ids: set[str] = set()
    budget_hit_reasons: Counter[str] = Counter()
    total_active_vehicle_count = 0
    optional_cleanup_package_count = 0
    source_open_rejections: Counter[str] = Counter()
    admission_plan = _select_phase1_admission_plan(
        admissions=source_admissions,
        remaining_buffer=remaining_buffer,
    )
    admission_by_source = {
        admission.source_track: admission
        for admission in source_admissions
    }
    for source_track, reason in admission_plan.rejection_reason_by_source.items():
        admission = admission_by_source[source_track]
        source_open_rejections[reason] += 1
        deferred_vehicle_nos.update(
            vehicle_no
            for package in admission.backbone_packages
            for vehicle_no in package.vehicle_nos
        )
    for source_track in admission_plan.admitted_source_tracks:
        admission = admission_by_source[source_track]
        for package in sorted(
            admission.backbone_packages,
            key=lambda item: (
                admission_plan.slot_index_by_source.get(item.source_track, 99),
                _depot_topology_entry_priority(item.final_family),
                item.min_spot_priority,
                _depot_repair_process_priority(facts_by_vehicle[item.vehicle_nos[0]].repair_process),
                item.source_order_start,
                item.package_id,
            ),
        ):
            allowed, reason = _phase1_budget_allows_package(
                package=package,
                selected_source_tracks=selected_source_tracks,
                selected_hot_source_tracks=selected_hot_source_tracks,
                selected_storage_source_tracks=selected_storage_source_tracks,
                selected_package_count=len(selected_package_ids),
                total_active_vehicle_count=total_active_vehicle_count,
                optional_cleanup_package_count=optional_cleanup_package_count,
            )
            if not allowed:
                budget_hit_reasons[reason] += 1
                deferred_vehicle_nos.update(package.vehicle_nos)
                continue
            candidate_depot_packages.append(package)
            selected_package_ids.add(package.package_id)
            selected_source_tracks.add(package.source_track)
            if _phase1_is_hot_source_role(package.source_chain_role):
                selected_hot_source_tracks.add(package.source_track)
            elif _phase1_is_storage_source_role(package.source_chain_role):
                selected_storage_source_tracks.add(package.source_track)
            total_active_vehicle_count += len(package.vehicle_nos)
    selected_depot_packages, dropped_depot_packages = _assign_phase1_backbone_packages_globally(
        packages=candidate_depot_packages,
        remaining_buffer=remaining_buffer,
        slot_index_by_source=admission_plan.slot_index_by_source,
    )
    selected_depot_packages = list(selected_depot_packages)
    dropped_depot_packages = list(dropped_depot_packages)
    dropped_depot_ids = {package.package_id for package in dropped_depot_packages}
    if dropped_depot_packages:
        budget_hit_reasons["buffer_capacity"] += len(dropped_depot_packages)
        deferred_vehicle_nos.update(
            vehicle_no
            for package in dropped_depot_packages
            for vehicle_no in package.vehicle_nos
        )
        selected_package_ids -= dropped_depot_ids
        selected_source_tracks = {
            package.source_track
            for package in selected_depot_packages
        }
        selected_hot_source_tracks = {
            package.source_track
            for package in selected_depot_packages
            if _phase1_is_hot_source_role(package.source_chain_role)
        }
        selected_storage_source_tracks = {
            package.source_track
            for package in selected_depot_packages
            if _phase1_is_storage_source_role(package.source_chain_role)
        }
        total_active_vehicle_count = sum(len(package.vehicle_nos) for package in selected_depot_packages)
    for package in selected_depot_packages:
        target_track = package.target_track
        required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
        for vehicle_no in package.vehicle_nos:
            buffer_assignment[vehicle_no] = target_track
        remaining_buffer[target_track] -= required
        buffer_track_sources[target_track].add(package.source_track)
        buffer_track_families[target_track].add(package.final_family)
    required_non_buffer_packages = [
        package
        for admission in source_admissions
        for package in admission.required_cleanup_packages
        if _phase1_should_select_required_cleanup_package(
            package=package,
            selected_source_tracks=selected_source_tracks,
        )
    ]
    optional_non_buffer_packages = [
        package
        for admission in source_admissions
        if admission.source_track in selected_source_tracks
        for package in admission.optional_cleanup_packages
    ]
    chosen_required_non_buffer: list[Phase1LayoutPackage] = []
    chosen_optional_non_buffer: list[Phase1LayoutPackage] = []
    for package in sorted(
        required_non_buffer_packages + optional_non_buffer_packages,
        key=lambda item: (
            _phase1_execution_layer_priority(item.execution_layer),
            0 if item.source_track in selected_source_tracks else 1,
            _phase1_source_chain_priority(item.source_chain_role),
            item.complexity_cost,
            _phase1_track_priority(item.source_track),
            item.source_order_start,
            item.package_id,
        ),
    ):
        allowed, reason = _phase1_budget_allows_package(
            package=package,
            selected_source_tracks=selected_source_tracks,
            selected_hot_source_tracks=selected_hot_source_tracks,
            selected_storage_source_tracks=selected_storage_source_tracks,
            selected_package_count=len(selected_package_ids),
            total_active_vehicle_count=total_active_vehicle_count,
            optional_cleanup_package_count=optional_cleanup_package_count,
        )
        if not allowed:
            budget_hit_reasons[reason] += 1
            continue
        selected_package_ids.add(package.package_id)
        selected_source_tracks.add(package.source_track)
        if _phase1_is_hot_source_role(package.source_chain_role):
            selected_hot_source_tracks.add(package.source_track)
        elif _phase1_is_storage_source_role(package.source_chain_role):
            selected_storage_source_tracks.add(package.source_track)
        total_active_vehicle_count += len(package.vehicle_nos)
        if package.execution_layer == "L2_REQUIRED_CLEAR":
            chosen_required_non_buffer.append(package)
        else:
            chosen_optional_non_buffer.append(package)
            optional_cleanup_package_count += 1
    selected_non_buffer_candidates = chosen_required_non_buffer + chosen_optional_non_buffer
    temp_assignments = _assign_phase1_temp_targets(
        facts_list=facts_list,
        packages=selected_non_buffer_candidates,
        buffer_assignment=buffer_assignment,
        master=master,
    )
    goal_overrides: dict[str, tuple[str, str]] = {}
    selected_non_buffer_packages: list[Phase1LayoutPackage] = []
    for package in selected_non_buffer_candidates:
        if package.package_kind == "local_finish":
            for vehicle_no in package.vehicle_nos:
                goal_overrides[vehicle_no] = (package.target_track, package.target_source)
            selected_non_buffer_packages.append(package)
            continue
        if package.package_kind != "temp_repark":
            continue
        assigned_track = temp_assignments.get(package.package_id)
        if assigned_track is None:
            continue
        for vehicle_no in package.vehicle_nos:
            goal_overrides[vehicle_no] = (assigned_track, package.target_source)
        selected_non_buffer_packages.append(
            Phase1LayoutPackage(
                package_id=package.package_id,
                chain_id=package.chain_id,
                package_kind=package.package_kind,
                source_track=package.source_track,
                vehicle_nos=package.vehicle_nos,
                total_length_m=package.total_length_m,
                target_track=assigned_track,
                target_source=package.target_source,
                final_family=package.final_family,
                min_spot_priority=package.min_spot_priority,
                source_order_start=package.source_order_start,
                source_order_end=package.source_order_end,
                    buffer_preference=package.buffer_preference,
                    uses_buffer=False,
                    pressure_cut=package.pressure_cut,
                    reason_tags=package.reason_tags,
                execution_layer=package.execution_layer,
                complexity_cost=package.complexity_cost,
                source_chain_role=package.source_chain_role,
                is_required_for_backbone=package.is_required_for_backbone,
                segment_role=package.segment_role,
                source_segment_index=package.source_segment_index,
                source_segment_count=package.source_segment_count,
                source_total_vehicle_count=package.source_total_vehicle_count,
                requires_previous_segment=package.requires_previous_segment,
                )
            )
    target_rank_by_vehicle = _build_phase1_target_ranks_from_assignment(
        facts_list=facts_list,
        buffer_assignment=buffer_assignment,
    )
    diagnostics = _build_phase1_layout_diagnostics(
        facts_list=facts_list,
        demand_summary=demand_summary,
        all_packages=packages,
        selected_depot_packages=tuple(selected_depot_packages),
        selected_non_buffer_packages=tuple(selected_non_buffer_packages),
        deferred_vehicle_nos=frozenset(deferred_vehicle_nos),
        buffer_assignment=buffer_assignment,
        target_rank_by_vehicle=target_rank_by_vehicle,
        master=master,
        budget_hit_reasons=dict(sorted((budget_hit_reasons + source_open_rejections).items())),
        selected_source_tracks=sorted(selected_source_tracks),
        selected_hot_source_tracks=sorted(selected_hot_source_tracks),
        selected_storage_source_tracks=sorted(selected_storage_source_tracks),
        total_active_vehicle_count=total_active_vehicle_count,
        optional_cleanup_package_count=optional_cleanup_package_count,
        source_admissions=source_admissions,
        admission_plan=admission_plan,
    )
    return Phase1LayoutResult(
        selected_depot_packages=tuple(selected_depot_packages),
        non_buffer_packages=tuple(selected_non_buffer_packages),
        deferred_vehicle_nos=frozenset(deferred_vehicle_nos),
        buffer_assignment=buffer_assignment,
        target_rank_by_vehicle=target_rank_by_vehicle,
        goal_overrides=goal_overrides,
        diagnostics=diagnostics,
    )


def _choose_phase1_buffer_track_for_package(
    *,
    package: Phase1LayoutPackage,
    remaining_buffer: dict[str, float],
    buffer_track_sources: dict[str, set[str]],
    buffer_track_families: dict[str, set[str]],
) -> str | None:
    required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
    feasible = [track for track, remaining in remaining_buffer.items() if remaining >= required]
    if not feasible:
        return None
    preferred_order = {track: index for index, track in enumerate(package.buffer_preference)}

    def sort_key(track: str) -> tuple[int, int, float, str]:
        source_set = buffer_track_sources.get(track, set())
        family_set = buffer_track_families.get(track, set())
        if source_set == {package.source_track}:
            occupancy_class = 0
        elif family_set == {package.final_family}:
            occupancy_class = 1
        elif not source_set:
            occupancy_class = 2
        else:
            occupancy_class = 3
        return (
            occupancy_class,
            preferred_order.get(track, len(package.buffer_preference) + 10),
            -remaining_buffer.get(track, 0.0),
            track,
        )

    return min(feasible, key=sort_key)


def _assign_phase1_backbone_packages_globally(
    *,
    packages: list[Phase1LayoutPackage],
    remaining_buffer: dict[str, float],
    slot_index_by_source: dict[str, int],
) -> tuple[tuple[Phase1LayoutPackage, ...], tuple[Phase1LayoutPackage, ...]]:
    ordered_packages = sorted(
        packages,
        key=lambda item: (
            slot_index_by_source.get(item.source_track, 99),
            0 if item.segment_role == "core" else 1 if item.segment_role == "follow" else 2,
            _depot_topology_entry_priority(item.final_family),
            item.min_spot_priority,
            _phase1_source_chain_priority(item.source_chain_role),
            item.source_order_start,
            item.package_id,
        ),
    )
    required_length_by_package = {
        package.package_id: package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
        for package in ordered_packages
    }
    core_packages = [
        package
        for package in ordered_packages
        if package.segment_role == "core" or not package.requires_previous_segment
    ]
    extension_packages = [
        package
        for package in ordered_packages
        if package not in core_packages
    ]
    first_extension_required_by_source: dict[str, float] = {}
    for package in extension_packages:
        required = required_length_by_package[package.package_id]
        first_extension_required_by_source.setdefault(package.source_track, required)
    best_score: tuple[int, int, int, int, int, int, float, int, int] | None = None
    best_assignment: dict[str, str] | None = None

    def dfs(
        index: int,
        remaining: dict[str, float],
        assignment: dict[str, str],
        track_sources: dict[str, set[str]],
        track_families: dict[str, set[str]],
    ) -> None:
        nonlocal best_score, best_assignment
        if index >= len(core_packages):
            selected_packages = [package for package in core_packages if package.package_id in assignment]
            selected_vehicle_count = sum(len(package.vehicle_nos) for package in selected_packages)
            selected_length = round(sum(package.total_length_m for package in selected_packages), 1)
            used_tracks = {track for track in assignment.values()}
            selected_source_tracks = {package.source_track for package in selected_packages}
            extension_ready_count = sum(
                1
                for package in selected_packages
                if (
                    first_extension_required_by_source.get(package.source_track) is not None
                    and remaining.get(assignment[package.package_id], 0.0)
                    >= first_extension_required_by_source[package.source_track]
                )
            )
            split_source_penalty = sum(
                max(
                    0,
                    sum(1 for package in selected_packages if package.source_track == source_track) - 1,
                )
                for source_track in selected_source_tracks
            )
            source_mix_penalty = sum(
                max(0, len(track_sources[track]) - 1)
                for track in used_tracks
            )
            family_mix_penalty = sum(
                max(0, len(track_families[track]) - 1)
                for track in used_tracks
            )
            preference_penalty = sum(
                package.buffer_preference.index(assignment[package.package_id])
                if assignment[package.package_id] in package.buffer_preference
                else len(package.buffer_preference) + 5
                for package in selected_packages
            )
            score = (
                selected_vehicle_count,
                extension_ready_count,
                -source_mix_penalty,
                -len(used_tracks),
                -split_source_penalty,
                -len(selected_source_tracks),
                selected_length,
                -family_mix_penalty,
                -preference_penalty,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = dict(assignment)
            return

        package = core_packages[index]
        remaining_vehicle_possible = sum(len(item.vehicle_nos) for item in core_packages[index:])
        selected_vehicle_now = sum(
            len(item.vehicle_nos)
            for item in core_packages[:index]
            if item.package_id in assignment
        )
        if best_score is not None and selected_vehicle_now + remaining_vehicle_possible < best_score[0]:
            return

        required = required_length_by_package[package.package_id]
        feasible_tracks = [
            track
            for track in JI_BUFFER_TRACKS
            if remaining.get(track, 0.0) >= required
        ]
        preferred_order = {track: idx for idx, track in enumerate(package.buffer_preference)}
        feasible_tracks.sort(
            key=lambda track: (
                0
                if (
                    first_extension_required_by_source.get(package.source_track) is not None
                    and remaining.get(track, 0.0) - required
                    >= first_extension_required_by_source[package.source_track]
                )
                else 1,
                0 if track_sources[track] == {package.source_track} else 1,
                0 if track_families[track] == {package.final_family} else 1 if not track_families[track] else 2,
                0 if not track_sources[track] else 1,
                preferred_order.get(track, len(package.buffer_preference) + 10),
                -remaining[track],
                track,
            )
        )
        for track in feasible_tracks:
            next_remaining = dict(remaining)
            next_remaining[track] -= required
            next_assignment = dict(assignment)
            next_assignment[package.package_id] = track
            next_track_sources = {name: set(values) for name, values in track_sources.items()}
            next_track_sources[track].add(package.source_track)
            next_track_families = {name: set(values) for name, values in track_families.items()}
            next_track_families[track].add(package.final_family)
            dfs(
                index + 1,
                next_remaining,
                next_assignment,
                next_track_sources,
                next_track_families,
            )

        dfs(index + 1, remaining, assignment, track_sources, track_families)

    dfs(
        0,
        dict(remaining_buffer),
        {},
        {track: set() for track in JI_BUFFER_TRACKS},
        {track: set() for track in JI_BUFFER_TRACKS},
    )
    assignment = dict(best_assignment or {})
    remaining_after_core = dict(remaining_buffer)
    track_sources = {track: set() for track in JI_BUFFER_TRACKS}
    track_families = {track: set() for track in JI_BUFFER_TRACKS}
    selected_segment_indexes_by_source: dict[str, set[int]] = defaultdict(set)
    source_package_tracks: dict[str, set[str]] = defaultdict(set)
    for package in core_packages:
        target_track = assignment.get(package.package_id)
        if target_track is None:
            continue
        remaining_after_core[target_track] -= required_length_by_package[package.package_id]
        track_sources[target_track].add(package.source_track)
        track_families[target_track].add(package.final_family)
        selected_segment_indexes_by_source[package.source_track].add(package.source_segment_index)
        source_package_tracks[package.source_track].add(target_track)
    for package in extension_packages:
        if package.source_track not in selected_segment_indexes_by_source:
            continue
        if package.requires_previous_segment and (
            package.source_segment_index - 1 not in selected_segment_indexes_by_source[package.source_track]
        ):
            continue
        target_track = _choose_phase1_extension_track_for_package(
            package=package,
            remaining_buffer=remaining_after_core,
            track_sources=track_sources,
            track_families=track_families,
            required_length_by_package=required_length_by_package,
            source_package_tracks=source_package_tracks,
        )
        if target_track is None:
            continue
        assignment[package.package_id] = target_track
        remaining_after_core[target_track] -= required_length_by_package[package.package_id]
        track_sources[target_track].add(package.source_track)
        track_families[target_track].add(package.final_family)
        selected_segment_indexes_by_source[package.source_track].add(package.source_segment_index)
        source_package_tracks[package.source_track].add(target_track)
    selected: list[Phase1LayoutPackage] = []
    dropped: list[Phase1LayoutPackage] = []
    for package in ordered_packages:
        target_track = assignment.get(package.package_id)
        if target_track is None:
            dropped.append(package)
            continue
        selected.append(
            Phase1LayoutPackage(
                package_id=package.package_id,
                chain_id=package.chain_id,
                package_kind=package.package_kind,
                source_track=package.source_track,
                vehicle_nos=package.vehicle_nos,
                total_length_m=package.total_length_m,
                target_track=target_track,
                target_source=package.target_source,
                final_family=package.final_family,
                min_spot_priority=package.min_spot_priority,
                source_order_start=package.source_order_start,
                source_order_end=package.source_order_end,
                buffer_preference=package.buffer_preference,
                uses_buffer=package.uses_buffer,
                pressure_cut=package.pressure_cut,
                reason_tags=package.reason_tags,
                execution_layer=package.execution_layer,
                complexity_cost=package.complexity_cost,
                source_chain_role=package.source_chain_role,
                is_required_for_backbone=package.is_required_for_backbone,
                segment_role=package.segment_role,
                source_segment_index=package.source_segment_index,
                source_segment_count=package.source_segment_count,
                source_total_vehicle_count=package.source_total_vehicle_count,
                requires_previous_segment=package.requires_previous_segment,
            )
        )
    return tuple(selected), tuple(dropped)


def _choose_phase1_extension_track_for_package(
    *,
    package: Phase1LayoutPackage,
    remaining_buffer: dict[str, float],
    track_sources: dict[str, set[str]],
    track_families: dict[str, set[str]],
    required_length_by_package: dict[str, float],
    source_package_tracks: dict[str, set[str]],
) -> str | None:
    required = required_length_by_package[package.package_id]
    source_tracks = source_package_tracks.get(package.source_track, set())
    feasible = [
        track
        for track in source_tracks
        if remaining_buffer.get(track, 0.0) >= required
    ]
    if not feasible:
        return None
    preferred_order = {track: idx for idx, track in enumerate(package.buffer_preference)}

    def sort_key(track: str) -> tuple[int, int, float, str]:
        family_penalty = 0 if not track_families[track] or track_families[track] == {package.final_family} else 1
        return (
            family_penalty,
            preferred_order.get(track, len(package.buffer_preference) + 10),
            -remaining_buffer.get(track, 0.0),
            track,
        )

    ordered = sorted(feasible, key=sort_key)
    return ordered[0] if ordered else None


def _phase1_budget_allows_package(
    *,
    package: Phase1LayoutPackage,
    selected_source_tracks: set[str],
    selected_hot_source_tracks: set[str],
    selected_storage_source_tracks: set[str],
    selected_package_count: int,
    total_active_vehicle_count: int,
    optional_cleanup_package_count: int,
) -> tuple[bool, str]:
    if selected_package_count >= PHASE1_MAX_SELECTED_PACKAGES:
        return False, "max_selected_packages"
    next_source_tracks = set(selected_source_tracks)
    next_source_tracks.add(package.source_track)
    next_hot_source_tracks = set(selected_hot_source_tracks)
    next_storage_source_tracks = set(selected_storage_source_tracks)
    if _phase1_is_hot_source_role(package.source_chain_role):
        next_hot_source_tracks.add(package.source_track)
    elif _phase1_is_storage_source_role(package.source_chain_role):
        next_storage_source_tracks.add(package.source_track)
    if len(next_hot_source_tracks) > PHASE1_MAX_ACTIVE_HOT_SOURCE_TRACKS:
        return False, "max_active_hot_source_tracks"
    if len(next_storage_source_tracks) > PHASE1_MAX_ACTIVE_STORAGE_SOURCE_TRACKS:
        return False, "max_active_storage_source_tracks"
    if total_active_vehicle_count + len(package.vehicle_nos) > PHASE1_MAX_TOTAL_ACTIVE_VEHICLES:
        return False, "max_total_active_vehicles"
    if package.execution_layer == "L3_OPTIONAL_CLEANUP":
        if optional_cleanup_package_count >= PHASE1_MAX_OPTIONAL_CLEANUP_PACKAGES:
            return False, "max_optional_cleanup_packages"
    return True, "ok"


def _phase1_should_select_required_cleanup_package(
    *,
    package: Phase1LayoutPackage,
    selected_source_tracks: set[str],
) -> bool:
    if package.target_source in {"PHASE1_CLEAR_CUN4", "PHASE1_CLEAR_JI"}:
        return True
    return package.source_track in selected_source_tracks


def _assign_phase1_temp_targets(
    *,
    facts_list: list[VehicleStageFacts],
    packages: list[Phase1LayoutPackage],
    buffer_assignment: dict[str, str],
    master: MasterData,
) -> dict[str, str]:
    facts_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    moving_vehicle_nos = set(buffer_assignment)
    moving_vehicle_nos.update(
        vehicle_no
        for package in packages
        for vehicle_no in package.vehicle_nos
        if package.package_kind in {"local_finish", "temp_repark"}
    )
    remaining_temp = {}
    for track in PHASE1_TEMP_PARKING_TRACKS:
        staying_length = sum(
            facts.vehicle_length
            for facts in facts_list
            if facts.current_track == track and facts.vehicle_no not in moving_vehicle_nos
        )
        remaining_temp[track] = round(float(master.tracks[track].effective_length_m) - staying_length, 1)
    temp_track_sources = {
        track: set()
        for track in PHASE1_TEMP_PARKING_TRACKS
    }
    source_temp_track: dict[str, str] = {}
    assignments: dict[str, str] = {}
    temp_packages = [package for package in packages if package.package_kind == "temp_repark"]
    temp_packages.sort(
        key=lambda item: (
            0 if item.target_source == "PHASE1_CLEAR_JI" else 1,
            0 if item.target_source == "PHASE1_CLEAR_CUN4" else 1,
            0 if item.target_source == PHASE1_BLOCKER_BUCKET_WORK else 1,
            _phase1_track_priority(item.source_track),
            item.source_order_start,
            item.package_id,
        )
    )
    for package in temp_packages:
        target_track = _choose_phase1_temp_track_for_package(
            package=package,
            remaining_temp=remaining_temp,
            facts_by_vehicle=facts_by_vehicle,
            temp_track_sources=temp_track_sources,
            source_temp_track=source_temp_track,
        )
        if target_track is None:
            continue
        assignments[package.package_id] = target_track
        required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
        remaining_temp[target_track] -= required
        temp_track_sources[target_track].add(package.source_track)
        source_temp_track.setdefault(package.source_track, target_track)
    return assignments


def _choose_phase1_temp_track_for_package(
    *,
    package: Phase1LayoutPackage,
    remaining_temp: dict[str, float],
    facts_by_vehicle: dict[str, VehicleStageFacts],
    temp_track_sources: dict[str, set[str]],
    source_temp_track: dict[str, str],
) -> str | None:
    required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
    first_facts = facts_by_vehicle[package.vehicle_nos[0]]
    preferred: list[str] = []
    if (
        first_facts.final_target_track in PHASE1_TEMP_PARKING_TRACKS
        and first_facts.final_target_track != package.source_track
    ):
        preferred.append(first_facts.final_target_track)
    for track in _phase1_temp_bucket_tracks(package=package, first_facts=first_facts):
        if track == package.source_track or track in preferred:
            continue
        preferred.append(track)
    feasible = [track for track in preferred if remaining_temp.get(track, 0.0) >= required]
    if not feasible:
        return None
    preferred_track = source_temp_track.get(package.source_track)

    def sort_key(track: str) -> tuple[int, int, int, float, str]:
        source_set = temp_track_sources.get(track, set())
        if preferred_track == track:
            reuse_class = 0
        elif source_set == {package.source_track}:
            reuse_class = 1
        elif not source_set:
            reuse_class = 2
        else:
            reuse_class = 3
        return (
            reuse_class,
            preferred.index(track),
            0 if track == first_facts.final_target_track else 1,
            -remaining_temp.get(track, 0.0),
            track,
        )

    return min(feasible, key=sort_key)


def _phase1_temp_bucket_tracks(
    *,
    package: Phase1LayoutPackage,
    first_facts: VehicleStageFacts,
) -> tuple[str, ...]:
    if package.target_source == PHASE1_BLOCKER_BUCKET_WORK:
        return PHASE1_WORK_BUCKET_TRACKS
    if package.target_source == PHASE1_BLOCKER_BUCKET_YARD:
        return PHASE1_YARD_BUCKET_TRACKS
    if (
        first_facts.final_target_track in STORAGE_TRACKS
        or first_facts.final_target_track in JI_BUFFER_TRACKS
    ):
        return PHASE1_YARD_BUCKET_TRACKS
    return PHASE1_WORK_BUCKET_TRACKS


def _build_phase1_target_ranks_from_assignment(
    *,
    facts_list: list[VehicleStageFacts],
    buffer_assignment: dict[str, str],
) -> dict[str, int]:
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        target_track = buffer_assignment.get(facts.vehicle_no)
        if target_track is None:
            continue
        facts_by_track[target_track].append(facts)
    target_rank_by_vehicle: dict[str, int] = {}
    for track, items in facts_by_track.items():
        ordered = sorted(
            items,
            key=lambda facts: (
                _depot_topology_entry_priority(facts.final_family),
                _final_spot_priority(facts.final_target_spot),
                _depot_repair_process_priority(facts.repair_process),
                _phase1_track_priority(facts.current_track),
                facts.current_order,
                facts.vehicle_no,
            ),
        )
        for rank, facts in enumerate(ordered, start=1):
            target_rank_by_vehicle[facts.vehicle_no] = rank
    return target_rank_by_vehicle


def _build_phase1_layout_diagnostics(
    *,
    facts_list: list[VehicleStageFacts],
    demand_summary: Phase1DemandSummary,
    all_packages: list[Phase1LayoutPackage],
    selected_depot_packages: list[Phase1LayoutPackage],
    selected_non_buffer_packages: list[Phase1LayoutPackage],
    deferred_vehicle_nos: frozenset[str],
    buffer_assignment: dict[str, str],
    target_rank_by_vehicle: dict[str, int],
    master: MasterData,
    budget_hit_reasons: dict[str, int],
    selected_source_tracks: list[str],
    selected_hot_source_tracks: list[str],
    selected_storage_source_tracks: list[str],
    total_active_vehicle_count: int,
    optional_cleanup_package_count: int,
    source_admissions: list[Phase1SourceAdmission],
    admission_plan: Phase1AdmissionPlan,
) -> dict[str, Any]:
    selected_vehicle_nos = frozenset(buffer_assignment)
    goal_override_vehicle_nos: set[str] = set()
    goal_overrides = {
        vehicle_no: (package.target_track, package.target_source)
        for package in selected_non_buffer_packages
        for vehicle_no in package.vehicle_nos
    }
    goal_override_vehicle_nos.update(goal_overrides)
    region_completion = _phase1_non_depot_region_completion(
        facts_list=facts_list,
        selected_vehicle_nos=selected_vehicle_nos,
        goal_overrides=goal_overrides,
        deferred_vehicle_nos=deferred_vehicle_nos,
    )
    depot_compiled_vehicle_nos = sorted(vehicle_no for vehicle_no in demand_summary.depot_vehicle_nos if vehicle_no in buffer_assignment)
    cun4_cleared_vehicle_nos = sorted(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in {"存4北", "存4南"} and facts.vehicle_no in (goal_override_vehicle_nos | set(buffer_assignment))
    )
    ji_cleared_vehicle_nos = sorted(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch and facts.vehicle_no in goal_overrides
    )
    buffer_lengths: dict[str, float] = {}
    buffer_required_lengths: dict[str, float] = {}
    buffer_source_tracks: dict[str, set[str]] = {}
    for vehicle_no, track in buffer_assignment.items():
        buffer_lengths[track] = round(buffer_lengths.get(track, 0.0) + facts_by_vehicle_length(facts_list, vehicle_no), 1)
    selected_depot_by_vehicle = {
        vehicle_no: package.source_track
        for package in selected_depot_packages
        for vehicle_no in package.vehicle_nos
    }
    for package in selected_depot_packages:
        track = buffer_assignment.get(package.vehicle_nos[0])
        if track is None:
            continue
        required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
        buffer_required_lengths[track] = round(buffer_required_lengths.get(track, 0.0) + required, 1)
    for vehicle_no, track in buffer_assignment.items():
        source_track = selected_depot_by_vehicle.get(vehicle_no)
        if source_track is None:
            continue
        buffer_source_tracks.setdefault(track, set()).add(source_track)
    selected_packages = selected_depot_packages + selected_non_buffer_packages
    selected_layer_counts = Counter(package.execution_layer for package in selected_packages)
    cleanup_by_source: dict[str, dict[str, int]] = {}
    temp_tracks_by_source: dict[str, list[str]] = {}
    for package in selected_non_buffer_packages:
        source_row = cleanup_by_source.setdefault(
            package.source_track,
            {
                "requiredLocalFinishCount": 0,
                "requiredTempReparkCount": 0,
                "optionalLocalFinishCount": 0,
                "optionalTempReparkCount": 0,
            },
        )
        key = (
            "requiredLocalFinishCount"
            if package.execution_layer == "L2_REQUIRED_CLEAR" and package.package_kind == "local_finish"
            else "requiredTempReparkCount"
            if package.execution_layer == "L2_REQUIRED_CLEAR"
            else "optionalLocalFinishCount"
            if package.package_kind == "local_finish"
            else "optionalTempReparkCount"
        )
        source_row[key] += 1
        if package.package_kind == "temp_repark" and package.target_track:
            temp_tracks_by_source.setdefault(package.source_track, []).append(package.target_track)
    return {
        "selectedPackageIds": [package.package_id for package in selected_packages],
        "selectedChainIds": sorted({package.chain_id for package in selected_packages}),
        "selectedPackageCount": len(selected_packages),
        "selectedPackageSourceTracks": sorted({package.source_track for package in selected_packages}),
        "selectedVehicleNos": sorted(selected_vehicle_nos),
        "selectedVehicleCount": len(selected_vehicle_nos),
        "selectedLocalVehicleNos": sorted(goal_overrides),
        "selectedLocalVehicleCount": len(goal_overrides),
        "activeVehicleCount": total_active_vehicle_count,
        "backboneVehicleCount": len(selected_vehicle_nos),
        "cleanupVehicleCount": len(goal_overrides),
        "selectedSourceTrackCount": len(selected_source_tracks),
        "selectedSourceTracks": list(selected_source_tracks),
        "admittedSourceTracks": list(admission_plan.admitted_source_tracks),
        "primaryBackboneSourceTracks": list(admission_plan.primary_source_tracks),
        "elasticBackboneSourceTracks": list(admission_plan.elastic_source_tracks),
        "companionBackboneSourceTracks": list(admission_plan.companion_source_tracks),
        "deferredBackboneSourceTracks": list(admission_plan.deferred_source_tracks),
        "selectedHotSourceTrackCount": len(selected_hot_source_tracks),
        "selectedHotSourceTracks": list(selected_hot_source_tracks),
        "selectedStorageSourceTrackCount": len(selected_storage_source_tracks),
        "selectedStorageSourceTracks": list(selected_storage_source_tracks),
        "optionalCleanupPackageCount": optional_cleanup_package_count,
        "selectedExecutionLayerCounts": dict(sorted(selected_layer_counts.items())),
        "selectedCleanupBySource": {
            source_track: counts
            for source_track, counts in sorted(cleanup_by_source.items())
        },
        "selectedTempTracksBySource": {
            source_track: sorted(dict.fromkeys(tracks))
            for source_track, tracks in sorted(temp_tracks_by_source.items())
        },
        "budgetHitReasons": dict(budget_hit_reasons),
        "sourceOpenSummaries": [
            {
                "sourceTrack": admission.source_track,
                "sourceChainRole": admission.source_chain_role,
                "backbonePackageCount": len(admission.backbone_packages),
                "backboneVehicleCount": admission.backbone_vehicle_count,
                "requiredCleanupPackageCount": len(admission.required_cleanup_packages),
                "requiredCleanupVehicleCount": admission.required_cleanup_vehicle_count,
                "requiredClearanceGainUnits": admission.required_clearance_gain_units,
                "optionalCleanupPackageCount": len(admission.optional_cleanup_packages),
                "openingCostUnits": admission.opening_cost_units,
                "openingGainUnits": admission.opening_gain_units,
                "openingScore": admission.opening_score,
                "primaryPackageIds": list(admission.primary_package_ids),
                "primaryVehicleCount": admission.primary_vehicle_count,
                "primaryRequiredLengthM": admission.primary_required_length_m,
                "admissionTier": admission.admission_tier,
                "selectedForBackbone": admission.source_track in selected_source_tracks,
                "admissionDecision": (
                    "primary"
                    if admission.source_track in admission_plan.primary_source_tracks
                    else "elastic"
                    if admission.source_track in admission_plan.elastic_source_tracks
                    else "companion"
                    if admission.source_track in admission_plan.companion_source_tracks
                    else "deferred"
                ),
                "slotIndex": admission_plan.slot_index_by_source.get(admission.source_track),
                "rejectionReason": admission_plan.rejection_reason_by_source.get(admission.source_track),
            }
            for admission in source_admissions
        ],
        "selectedTotalLengthM": round(sum(facts_by_vehicle_length(facts_list, vehicle_no) for vehicle_no in selected_vehicle_nos), 1),
        "deferredVehicleNos": sorted(deferred_vehicle_nos),
        "depotDemandVehicleCount": len(demand_summary.depot_vehicle_nos),
        "depotDemandTotalLengthM": demand_summary.depot_demand_total_length_m,
        "depotCompiledVehicleCount": len(depot_compiled_vehicle_nos),
        "depotCompiledVehicleNos": depot_compiled_vehicle_nos,
        "depotCompileRatio": round(len(depot_compiled_vehicle_nos) / len(demand_summary.depot_vehicle_nos), 4) if demand_summary.depot_vehicle_nos else 1.0,
        "uncompiledDepotVehicleNos": sorted(demand_summary.depot_vehicle_nos - set(buffer_assignment)),
        "cun4VehicleCount": len(demand_summary.cun4_vehicle_nos),
        "cun4ClearedVehicleCount": len(cun4_cleared_vehicle_nos),
        "cun4ClearedVehicleNos": cun4_cleared_vehicle_nos,
        "cun4ClearRatio": round(len(cun4_cleared_vehicle_nos) / len(demand_summary.cun4_vehicle_nos), 4) if demand_summary.cun4_vehicle_nos else 1.0,
        "remainingCun4VehicleNos": sorted(demand_summary.cun4_vehicle_nos - set(cun4_cleared_vehicle_nos)),
        "jiNonDepotVehicleCount": len(demand_summary.ji_non_depot_vehicle_nos),
        "jiNonDepotClearedVehicleCount": len(ji_cleared_vehicle_nos),
        "jiNonDepotClearedVehicleNos": ji_cleared_vehicle_nos,
        "jiPurityRatio": round(len(ji_cleared_vehicle_nos) / len(demand_summary.ji_non_depot_vehicle_nos), 4) if demand_summary.ji_non_depot_vehicle_nos else 1.0,
        "remainingJiNonDepotVehicleNos": sorted(demand_summary.ji_non_depot_vehicle_nos - set(ji_cleared_vehicle_nos)),
        "nonDepotRegionCompletion": region_completion,
        "jiCapacityTotalM": demand_summary.ji_capacity_total_m,
        "jiOverflowM": demand_summary.ji_overflow_m,
        "bufferLengthsM": buffer_lengths,
        "bufferRequiredLengthsM": buffer_required_lengths,
        "bufferSourceTracks": {
            track: sorted(source_tracks)
            for track, source_tracks in sorted(buffer_source_tracks.items())
        },
        "mixedBufferTrackCount": sum(1 for source_tracks in buffer_source_tracks.values() if len(source_tracks) >= 2),
        "bufferCapacityM": {
            track: round(min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track]), 1)
            for track in JI_BUFFER_TRACKS
        },
        "targetRankByVehicle": dict(sorted(target_rank_by_vehicle.items())),
        "taskPackages": [
            {
                "packageId": package.package_id,
                "chainId": package.chain_id,
                "packageKind": package.package_kind,
                "sourceTrack": package.source_track,
                "vehicleNos": list(package.vehicle_nos),
                "targetTrack": (
                    buffer_assignment.get(package.vehicle_nos[0], package.target_track)
                    if package.uses_buffer
                    else package.target_track
                ),
                "targetSource": package.target_source,
                "usesBuffer": package.uses_buffer,
                "totalLengthM": package.total_length_m,
                "finalFamily": package.final_family,
                "minSpotPriority": package.min_spot_priority,
                "sourceOrderStart": package.source_order_start,
                "sourceOrderEnd": package.source_order_end,
                "bufferPreference": list(package.buffer_preference),
                "pressureCut": package.pressure_cut,
                "reasonTags": list(package.reason_tags),
                "executionLayer": package.execution_layer,
                "complexityCost": package.complexity_cost,
                "sourceChainRole": package.source_chain_role,
                "isRequiredForBackbone": package.is_required_for_backbone,
                "segmentRole": package.segment_role,
                "sourceSegmentIndex": package.source_segment_index,
                "sourceSegmentCount": package.source_segment_count,
                "sourceTotalVehicleCount": package.source_total_vehicle_count,
                "requiresPreviousSegment": package.requires_previous_segment,
                "selected": (
                    package.vehicle_nos[0] in buffer_assignment
                    if package.uses_buffer
                    else package.package_id in {item.package_id for item in selected_non_buffer_packages}
                ),
                "bufferTrack": buffer_assignment.get(package.vehicle_nos[0]) if package.uses_buffer else None,
            }
            for package in all_packages
        ],
    }


def facts_by_vehicle_length(
    facts_list: list[VehicleStageFacts],
    vehicle_no: str,
) -> float:
    for facts in facts_list:
        if facts.vehicle_no == vehicle_no:
            return facts.vehicle_length
    return 0.0


def _build_phase1_source_track_plans(
    facts_list: list[VehicleStageFacts],
) -> list[SourceTrackPlan]:
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        if (
            facts.current_track in PHASE1_NON_DEPOT_REGION_TRACKS
            or _needs_phase1_depot_staging(facts)
            or (facts.current_track in JI_BUFFER_TRACKS and not _needs_phase1_depot_staging(facts))
        ):
            facts_by_track[facts.current_track].append(facts)
    block_index = 1
    plans: list[SourceTrackPlan] = []
    for source_track, members in sorted(facts_by_track.items()):
        ordered = sorted(members, key=lambda item: (item.current_order, item.vehicle_no))
        blocks, block_index = _build_single_source_track_blocks(
            source_track=source_track,
            ordered_members=ordered,
            block_index=block_index,
        )
        reachable_depot = tuple(
            vehicle_no
            for block in blocks
            if block.block_type in {"bridge_to_depot", "depot_batch"}
            for vehicle_no in block.vehicle_nos
        )
        reachable_finish = tuple(
            vehicle_no
            for block in blocks
            if block.block_type in {"clear_cun4", "prefix_clear", "tail_finish"}
            for vehicle_no in block.vehicle_nos
        )
        plans.append(
            SourceTrackPlan(
                source_track=source_track,
                blocks=tuple(blocks),
                reachable_depot_vehicle_nos=reachable_depot,
                reachable_finish_vehicle_nos=reachable_finish,
                cun4_clear_required=any(block.block_type == "clear_cun4" for block in blocks),
                buffer_demand_m=round(
                    sum(block.total_length_m for block in blocks if block.uses_buffer),
                    1,
                ),
                source_priority_score=_phase1_source_plan_priority(
                    source_track=source_track,
                    blocks=blocks,
                    reachable_depot_count=len(reachable_depot),
                ),
            )
        )
    plans.sort(key=lambda item: (item.source_priority_score, item.source_track))
    return plans


def _build_single_source_track_blocks(
    *,
    source_track: str,
    ordered_members: list[VehicleStageFacts],
    block_index: int,
) -> tuple[list[Phase1Block], int]:
    depot_seen = False
    hard_blocked_before_depot = False
    needs_bridge_prefix = False
    staged: list[tuple[str, VehicleStageFacts]] = []
    future_depot_suffix: list[bool] = [False] * len(ordered_members)
    future_has_depot = False
    for index in range(len(ordered_members) - 1, -1, -1):
        future_depot_suffix[index] = future_has_depot
        if _needs_phase1_depot_staging(ordered_members[index]):
            future_has_depot = True
    for facts_index, facts in enumerate(ordered_members):
        local_target = _phase1_local_finish_target_track(facts)
        if local_target is not None:
            if future_depot_suffix[facts_index] and not depot_seen:
                staged.append(("bucket", facts))
            else:
                staged.append(("local", facts))
            needs_bridge_prefix = True
            continue
        front_evict_target = (
            _phase1_front_evict_target_track(source_track=source_track, facts=facts)
            if future_depot_suffix[facts_index] and not depot_seen
            else None
        )
        if front_evict_target is not None:
            staged.append(("bucket", facts))
            needs_bridge_prefix = True
            continue
        if _needs_phase1_depot_staging(facts):
            if facts.current_track in JI_BUFFER_TRACKS:
                continue
            block_kind = "bridge" if not depot_seen and needs_bridge_prefix else "depot"
            staged.append((block_kind, facts))
            depot_seen = True
            continue
        if depot_seen:
            staged.append(("blocked", facts))
        else:
            hard_blocked_before_depot = True
            needs_bridge_prefix = True
            staged.append(("blocked", facts))
    blocks: list[Phase1Block] = []
    chunk_members: list[VehicleStageFacts] = []
    chunk_kind: str | None = None
    chunk_target: str | None = None
    first_depot_started = False
    all_relevant = [
        facts
        for kind, facts in staged
        if kind in {"local", "bucket", "bridge", "depot"}
    ]
    for kind, facts in staged:
        if kind == "blocked":
            if chunk_members:
                block = _make_phase1_block(
                    block_id=f"U{block_index:04d}",
                    source_track=source_track,
                    kind=chunk_kind or "tail_finish",
                    members=chunk_members,
                    first_depot_started=first_depot_started,
                    all_relevant_members=all_relevant,
                    blocked_before_depot=hard_blocked_before_depot,
                )
                blocks.append(block)
                if block.block_type in {"bridge_to_depot", "depot_batch"}:
                    first_depot_started = True
                block_index += 1
                chunk_members = []
                chunk_kind = None
                chunk_target = None
            continue
        current_target = (
            _phase1_local_finish_target_track(facts)
            if kind == "local"
            else _phase1_blocker_bucket_target_source(facts=facts, source_track=source_track)
            if kind == "bucket"
            else None
        )
        should_split = bool(
            chunk_members
            and (
                kind != chunk_kind
                or current_target != chunk_target
                or _phase1_should_split_block_chunk(
                    source_track=source_track,
                    members=chunk_members,
                    next_facts=facts,
                    kind=kind,
                )
            )
        )
        if should_split:
            block = _make_phase1_block(
                block_id=f"U{block_index:04d}",
                source_track=source_track,
                kind=chunk_kind or "tail_finish",
                members=chunk_members,
                first_depot_started=first_depot_started,
                all_relevant_members=all_relevant,
                blocked_before_depot=hard_blocked_before_depot,
            )
            blocks.append(block)
            if block.block_type in {"bridge_to_depot", "depot_batch"}:
                first_depot_started = True
            block_index += 1
            chunk_members = []
        chunk_members.append(facts)
        chunk_kind = kind
        chunk_target = current_target
    if chunk_members:
        block = _make_phase1_block(
            block_id=f"U{block_index:04d}",
            source_track=source_track,
            kind=chunk_kind or "tail_finish",
            members=chunk_members,
            first_depot_started=first_depot_started,
            all_relevant_members=all_relevant,
            blocked_before_depot=hard_blocked_before_depot,
        )
        blocks.append(block)
        block_index += 1
    with_predecessors: list[Phase1Block] = []
    predecessor_ids: list[str] = []
    source_buffer_block_index = 0
    for block in blocks:
        layout_role = "cleanup"
        if block.uses_buffer:
            source_buffer_block_index += 1
            layout_role = "core" if source_buffer_block_index == 1 else "spill"
        with_predecessors.append(
            Phase1Block(
                block_id=block.block_id,
                source_track=block.source_track,
                block_type=block.block_type,
                vehicle_nos=block.vehicle_nos,
                total_length_m=block.total_length_m,
                target_track=block.target_track,
                target_source=block.target_source,
                uses_buffer=block.uses_buffer,
                buffer_preference=block.buffer_preference,
                source_order_start=block.source_order_start,
                source_order_end=block.source_order_end,
                final_family=block.final_family,
                phase3_rank_key=block.phase3_rank_key,
                released_depot_vehicle_count=block.released_depot_vehicle_count,
                released_finish_vehicle_count=block.released_finish_vehicle_count,
                required_predecessor_ids=tuple(predecessor_ids[-1:]),
                layout_role=layout_role,
                topology_zone=block.topology_zone,
                throat_group=block.throat_group,
                pressure_gain=block.pressure_gain,
                coupling_degree=block.coupling_degree,
            )
        )
        predecessor_ids.append(block.block_id)
    return with_predecessors, block_index


def _phase1_should_split_block_chunk(
    *,
    source_track: str,
    members: list[VehicleStageFacts],
    next_facts: VehicleStageFacts,
    kind: str,
) -> bool:
    length_limit = (
        PHASE1_LOCAL_FINISH_SEGMENT_LENGTH_M
        if kind in {"local", "bucket"} and source_track in {"存5北", "存5南", "存3", "存2", "存1"}
        else PHASE1_UNIT_MAX_LENGTH_M
    )
    projected_length = sum(item.vehicle_length for item in members) + next_facts.vehicle_length
    projected_count = len(members) + 1
    projected_required_length = projected_length + max(1.0, projected_count - 1)
    if projected_required_length > length_limit:
        return True
    if next_facts.need_weigh or any(item.need_weigh for item in members):
        return True
    if next_facts.is_heavy and sum(1 for item in members if item.is_heavy) >= 2:
        return True
    if next_facts.is_heavy and any(item.is_close_door for item in members):
        return True
    if next_facts.is_close_door and any(item.is_heavy for item in members):
        return True
    if next_facts.final_target_track == "存4北" and (
        next_facts.is_close_door or any(item.is_close_door for item in members)
    ):
        return True
    return False


def _make_phase1_block(
    *,
    block_id: str,
    source_track: str,
    kind: str,
    members: list[VehicleStageFacts],
    first_depot_started: bool,
    all_relevant_members: list[VehicleStageFacts],
    blocked_before_depot: bool,
) -> Phase1Block:
    total_length_m = round(sum(item.vehicle_length for item in members), 1)
    max_order = max(item.current_order for item in members)
    future_relevant = [
        item
        for item in all_relevant_members
        if item.current_order > max_order
    ]
    future_depot = [item for item in future_relevant if _is_phase1_candidate(item)]
    future_finish = [item for item in future_relevant if _phase1_local_finish_target_track(item) is not None]
    if kind in {"local", "bucket"}:
        target_track = None if kind == "bucket" else _phase1_local_finish_target_track(members[0])
        target_source = None
        if source_track in {"存4北", "存4南"} and target_track is None:
            target_source = "PHASE1_CLEAR_CUN4"
        elif source_track in JI_BUFFER_TRACKS and target_track is None:
            target_source = "PHASE1_CLEAR_JI"
        elif target_track is None:
            target_source = _phase1_blocker_bucket_target_source(
                facts=members[0],
                source_track=source_track,
            )
        if source_track in {"存4北", "存4南"}:
            block_type = "clear_cun4"
        elif not first_depot_started and future_depot:
            block_type = "prefix_clear"
        else:
            block_type = "tail_finish"
        if target_source is None:
            target_source = _phase1_stage_target_source_for_local_clear(members[0])
        uses_buffer = False
        buffer_preference: tuple[str, ...] = tuple()
        final_family = target_track or members[0].final_family
    else:
        target_track = None
        target_source = "PHASE1_BACKBONE_PLACE"
        uses_buffer = not blocked_before_depot
        block_type = "bridge_to_depot" if kind == "bridge" else "depot_batch"
        final_family = members[0].final_family
        buffer_preference = _phase1_buffer_preference(
            final_family=final_family,
            source_track=source_track,
            current_track=members[0].current_track,
        ) if uses_buffer else tuple()
    return Phase1Block(
        block_id=block_id,
        source_track=source_track,
        block_type=block_type,
        vehicle_nos=tuple(item.vehicle_no for item in members),
        total_length_m=total_length_m,
        target_track=target_track,
        target_source=target_source,
        uses_buffer=uses_buffer,
        buffer_preference=buffer_preference,
        source_order_start=min(item.current_order for item in members),
        source_order_end=max(item.current_order for item in members),
        final_family=final_family,
        phase3_rank_key=(
            _depot_topology_entry_priority(final_family),
            min(_final_spot_priority(item.final_target_spot) for item in members),
            _depot_repair_process_priority(members[0].repair_process),
            min(item.current_order for item in members),
        ),
        released_depot_vehicle_count=len(future_depot),
        released_finish_vehicle_count=len(future_finish),
        required_predecessor_ids=tuple(),
        layout_role="cleanup",
        topology_zone=_phase1_topology_zone(source_track),
        throat_group=_phase1_block_throat_group(
            source_track=source_track,
            target_track=target_track,
            final_family=final_family,
        ),
        pressure_gain=_phase1_block_pressure_gain(
            source_track=source_track,
            target_track=target_track,
            future_depot_count=len(future_depot),
            member_count=len(members),
        ),
        coupling_degree=_phase1_block_coupling_degree(
            source_track=source_track,
            target_track=target_track,
            uses_buffer=uses_buffer,
        ),
    )


def _phase1_topology_zone(track_name: str) -> str:
    if track_name in {"存5北", "存5南"}:
        return "receiving"
    if track_name in {"存1", "存2", "存3", "调北", "洗北"}:
        return "yard"
    if track_name in {"预修", "机棚"}:
        return "pre_repair"
    if track_name in {"调棚", "洗南", "油", "抛", "轮"}:
        return "functional"
    if track_name in {"机南", "机北1", "机北2", "机北3", "存4北", "存4南"}:
        return "buffer"
    return "other"


def _phase1_throat_group(track_name: str) -> str:
    if track_name in {"机南", "机棚", "预修", "洗北", "洗南", "油", "机北1", "机北2", "机北3"}:
        return "G_L8"
    if track_name in {"调北", "调棚", "机库"}:
        return "G_L7"
    if track_name in {"存5北", "存5南", "存4北", "存4南", "存1", "存2", "存3"}:
        return "G_STORAGE"
    if track_name in {"抛", "轮"}:
        return "G_L15"
    return "G_OTHER"


def _phase1_block_throat_group(
    *,
    source_track: str,
    target_track: str | None,
    final_family: str,
) -> str:
    source_group = _phase1_throat_group(source_track)
    if target_track is None:
        if final_family == "轮":
            return "G_L15"
        return source_group
    target_group = _phase1_throat_group(target_track)
    if source_group == target_group:
        return source_group
    return f"{source_group}->{target_group}"


def _phase1_block_pressure_gain(
    *,
    source_track: str,
    target_track: str | None,
    future_depot_count: int,
    member_count: int,
) -> int:
    zone = _phase1_topology_zone(source_track)
    zone_base = {
        "receiving": 18,
        "functional": 14,
        "pre_repair": 12,
        "yard": 10,
        "buffer": 8,
        "other": 6,
    }.get(zone, 6)
    direct_finish_bonus = 4 if target_track is not None else 0
    return zone_base + future_depot_count * 4 + member_count + direct_finish_bonus


def _phase1_block_coupling_degree(
    *,
    source_track: str,
    target_track: str | None,
    uses_buffer: bool,
) -> int:
    source_group = _phase1_throat_group(source_track)
    target_group = _phase1_throat_group(target_track) if target_track is not None else source_group
    if uses_buffer:
        return 3
    if source_group == target_group:
        return 1
    if "G_OTHER" in {source_group, target_group}:
        return 2
    return 3


def _phase1_source_plan_priority(
    *,
    source_track: str,
    blocks: list[Phase1Block],
    reachable_depot_count: int,
) -> tuple[int, ...]:
    return (
        0 if any(block.block_type == "clear_cun4" for block in blocks) else 1,
        0 if source_track in WASH_CONFLICT_TRACKS else 1,
        0 if source_track in {"调棚", "预修", "抛", "调北"} else 1,
        -reachable_depot_count,
        -sum(1 for block in blocks if block.block_type in {"prefix_clear", "tail_finish"}),
        _phase1_local_source_priority(source_track),
    )


def _build_phase1_reachable_depot_set(
    source_plans: list[SourceTrackPlan],
) -> frozenset[str]:
    return frozenset(
        vehicle_no
        for plan in source_plans
        for vehicle_no in plan.reachable_depot_vehicle_nos
    )


def _phase1_backbone_templates(
    *,
    source_plans: list[SourceTrackPlan],
) -> tuple[Phase1LayoutTemplate, ...]:
    return (
        Phase1LayoutTemplate(
            template_name="pure_depot_staging",
            tracks=JI_BUFFER_TRACKS,
            preferred_open_order=("机棚", "机南", "机北1", "机北2", "机北3"),
            support_tracks=frozenset({"机北1", "机北2", "机北3"}),
        ),
    )


def _phase1_depot_insert_slot_limit(facts_list: list[VehicleStageFacts]) -> int:
    track_slot_capacity = {track: 5 for track in ("修1", "修2", "修3", "修4")}
    occupied_by_track: Counter[str] = Counter()
    for facts in facts_list:
        if facts.current_track not in track_slot_capacity:
            continue
        if facts.needs_depot_batch:
            occupied_by_track[facts.current_track] += 1
    return max(
        0,
        sum(
            max(0, capacity - occupied_by_track.get(track, 0))
            for track, capacity in track_slot_capacity.items()
        ),
    )


def _phase1_assigned_repair_depot_vehicle_count(
    assignment_by_block: dict[str, str],
    *,
    block_by_id: dict[str, Phase1Block],
) -> int:
    return sum(
        _phase1_repair_depot_vehicle_count(block_by_id[block_id])
        for block_id in assignment_by_block
        if block_id in block_by_id
    )


def _phase1_selected_repair_depot_vehicle_count(
    assignment_by_vehicle: dict[str, str],
    *,
    block_by_id: dict[str, Phase1Block],
) -> int:
    selected_vehicle_nos = set(assignment_by_vehicle)
    return sum(
        1
        for block in block_by_id.values()
        if block.final_family in {"修1", "修2", "修3", "修4"}
        for vehicle_no in block.vehicle_nos
        if vehicle_no in selected_vehicle_nos
    )


def _phase1_repair_depot_vehicle_count(block: Phase1Block) -> int:
    if block.final_family not in {"修1", "修2", "修3", "修4"}:
        return 0
    return len(block.vehicle_nos)


def _phase1_required_predecessor_closure_ids(
    *,
    block: Phase1Block,
    block_by_id: dict[str, Phase1Block],
) -> tuple[str, ...]:
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def visit(current: Phase1Block) -> None:
        for predecessor_id in current.required_predecessor_ids:
            predecessor = block_by_id.get(predecessor_id)
            if predecessor is None or predecessor.block_id in seen:
                continue
            seen.add(predecessor.block_id)
            visit(predecessor)
            ordered_ids.append(predecessor.block_id)

    visit(block)
    return tuple(ordered_ids)


def _phase1_block_assignment_order(block: Phase1Block, source_priority: tuple[int, ...]) -> tuple[Any, ...]:
    return (
        0 if block.layout_role == "core" else 1,
        0 if block.block_type == "bridge_to_depot" else 1,
        -block.released_depot_vehicle_count,
        -len(block.vehicle_nos),
        source_priority,
        block.phase3_rank_key,
        block.source_track,
        block.block_id,
    )


def _phase1_template_track_candidates(
    *,
    block: Phase1Block,
    template: Phase1LayoutTemplate,
    remaining: dict[str, float],
    source_tracks: dict[str, set[str]],
    track_sources: dict[str, set[str]],
    track_families: dict[str, set[str]],
) -> list[str]:
    source_used_tracks = source_tracks.get(block.source_track, set())
    existing_source_tracks = [
        track for track in template.tracks
        if track in source_used_tracks and _phase1_block_fits_buffer(block, track, remaining)
    ]
    if block.layout_role == "spill" and existing_source_tracks:
        reusable = sorted(
            existing_source_tracks,
            key=lambda track: (
                0 if track_families[track] == {block.final_family} else 1 if not track_families[track] else 2,
                template.preferred_open_order.index(track),
                -remaining[track],
                track,
            ),
        )
    else:
        reusable = []

    feasible = [
        track
        for track in template.tracks
        if _phase1_block_fits_buffer(block, track, remaining)
        and (
            block.layout_role == "core"
            or not source_used_tracks
            or (
                len(source_used_tracks) == 1
                and PHASE1_BUFFER_TRACK_ROLES.get(track) in {"main", "support"}
            )
            or (
                len(source_used_tracks) == 2
                and track in template.support_tracks
            )
        )
    ]
    if not feasible:
        return reusable
    ordered = sorted(
        feasible,
        key=lambda track: (
            0 if block.final_family == "轮" and track == "机北3" else 1,
            0 if track in source_used_tracks else 1,
            0 if PHASE1_BUFFER_TRACK_ROLES.get(track) == "main" else 1,
            0 if track_families[track] == {block.final_family} else 1 if not track_families[track] else 2,
            0 if not track_sources[track] else 1,
            template.preferred_open_order.index(track),
            -remaining[track],
            track,
        ),
    )
    return list(dict.fromkeys(reusable + ordered))


def _phase1_template_assignment_score(
    *,
    ordered_blocks: list[Phase1Block],
    assignment: dict[str, str],
    template: Phase1LayoutTemplate,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    selected_blocks = [block for block in ordered_blocks if block.block_id in assignment]
    source_tracks: dict[str, set[str]] = defaultdict(set)
    family_tracks: dict[str, set[str]] = defaultdict(set)
    track_sources: dict[str, set[str]] = defaultdict(set)
    for block in selected_blocks:
        track = assignment[block.block_id]
        source_tracks[block.source_track].add(track)
        family_tracks[block.final_family].add(track)
        track_sources[track].add(block.source_track)
    opened_tracks = {assignment[block.block_id] for block in selected_blocks}
    source_split_count = sum(max(0, len(tracks) - 1) for tracks in source_tracks.values())
    family_split_count = sum(max(0, len(tracks) - 1) for tracks in family_tracks.values())
    support_track_count = sum(1 for track in opened_tracks if track in template.support_tracks)
    spill_block_count = sum(1 for block in selected_blocks if block.layout_role == "spill")
    source_mix_penalty = sum(max(0, len(sources) - 1) for sources in track_sources.values())
    core_vehicle_count = sum(len(block.vehicle_nos) for block in selected_blocks if block.layout_role == "core")
    total_vehicle_count = sum(len(block.vehicle_nos) for block in selected_blocks)
    preference_penalty = sum(
        block.buffer_preference.index(assignment[block.block_id])
        if assignment[block.block_id] in block.buffer_preference
        else len(block.buffer_preference) + 5
        for block in selected_blocks
    )
    return (
        core_vehicle_count,
        total_vehicle_count,
        -source_mix_penalty,
        -support_track_count,
        -len(opened_tracks),
        -source_split_count,
        -family_split_count,
        -spill_block_count,
        -preference_penalty,
    )


def _solve_phase1_template_backbone_plan(
    *,
    source_plans: list[SourceTrackPlan],
    reserved_buffer: dict[str, float],
    template: Phase1LayoutTemplate,
    depot_slot_limit: int | None,
) -> Phase1BackbonePlan:
    ordered_blocks = sorted(
        [
            block
            for plan in source_plans
            for block in plan.blocks
            if block.uses_buffer and block.block_type in {"bridge_to_depot", "depot_batch"}
        ],
        key=lambda block: _phase1_block_assignment_order(
            block,
            next(plan.source_priority_score for plan in source_plans if plan.source_track == block.source_track),
        ),
    )
    block_by_id = {
        block.block_id: block
        for plan in source_plans
        for block in plan.blocks
    }
    if len(ordered_blocks) > 12:
        return _solve_phase1_template_backbone_plan_greedily(
            source_plans=source_plans,
            reserved_buffer=reserved_buffer,
            template=template,
            depot_slot_limit=depot_slot_limit,
        )
    best_score: tuple[int, int, int, int, int, int, int, int, int] | None = None
    best_assignment: dict[str, str] | None = None

    def dfs(
        index: int,
        remaining: dict[str, float],
        assignment: dict[str, str],
        source_tracks: dict[str, set[str]],
        track_sources: dict[str, set[str]],
        track_families: dict[str, set[str]],
    ) -> None:
        nonlocal best_score, best_assignment
        if index >= len(ordered_blocks):
            if len(assignment) != len(ordered_blocks):
                return
            score = _phase1_template_assignment_score(
                ordered_blocks=ordered_blocks,
                assignment=assignment,
                template=template,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = dict(assignment)
            return

        block = ordered_blocks[index]
        if not _phase1_backbone_predecessors_ready(
            block=block,
            block_by_id=block_by_id,
            selected_block_ids=set(assignment),
        ):
            return
        if (
            depot_slot_limit is not None
            and _phase1_assigned_repair_depot_vehicle_count(
                assignment,
                block_by_id=block_by_id,
            ) + _phase1_repair_depot_vehicle_count(block) > depot_slot_limit
        ):
            return

        candidates = _phase1_template_track_candidates(
            block=block,
            template=template,
            remaining=remaining,
            source_tracks=source_tracks,
            track_sources=track_sources,
            track_families=track_families,
        )
        for track in candidates:
            next_remaining = dict(remaining)
            next_remaining[track] -= _phase1_buffer_required_length(block)
            next_assignment = dict(assignment)
            next_assignment[block.block_id] = track
            next_source_tracks = {name: set(values) for name, values in source_tracks.items()}
            next_source_tracks.setdefault(block.source_track, set()).add(track)
            next_track_sources = {name: set(values) for name, values in track_sources.items()}
            next_track_sources[track].add(block.source_track)
            next_track_families = {name: set(values) for name, values in track_families.items()}
            next_track_families[track].add(block.final_family)
            dfs(index + 1, next_remaining, next_assignment, next_source_tracks, next_track_sources, next_track_families)

    dfs(
        0,
        {track: reserved_buffer[track] for track in template.tracks},
        {},
        {},
        {track: set() for track in template.tracks},
        {track: set() for track in template.tracks},
    )
    if best_assignment is None and ordered_blocks:
        raise ValueError(
            "phase1 cannot place all depot staging blocks in JI buffers: "
            + ", ".join(
                f"{block.block_id}:{block.source_track}:{list(block.vehicle_nos)}"
                for block in ordered_blocks
            )
        )
    assignment_by_block = best_assignment or {}
    selected_block_ids: list[str] = []
    selected_source_tracks: list[str] = []
    assignment: dict[str, str] = {}
    for block in ordered_blocks:
        target_track = assignment_by_block.get(block.block_id)
        if target_track is None:
            continue
        if block.source_track not in selected_source_tracks:
            selected_source_tracks.append(block.source_track)
        selected_block_id_set = set(selected_block_ids)
        for predecessor_id in _phase1_required_predecessor_closure_ids(
            block=block,
            block_by_id=block_by_id,
        ):
            predecessor = block_by_id.get(predecessor_id)
            if predecessor is None or predecessor.uses_buffer or predecessor_id in selected_block_id_set:
                continue
            selected_block_ids.append(predecessor_id)
            selected_block_id_set.add(predecessor_id)
        selected_block_ids.append(block.block_id)
        for vehicle_no in block.vehicle_nos:
            assignment[vehicle_no] = target_track
    target_rank_by_vehicle = _build_phase1_target_ranks_from_blocks(
        source_plans=source_plans,
        selected_block_ids=frozenset(selected_block_ids),
        buffer_assignment=assignment,
    )
    opened_tracks = tuple(
        track for track in template.preferred_open_order
        if track in {assignment_by_block[block_id] for block_id in assignment_by_block}
    )
    return Phase1BackbonePlan(
        selected_block_ids=tuple(selected_block_ids),
        selected_source_tracks=tuple(selected_source_tracks),
        reserved_buffer_by_track=reserved_buffer,
        selected_buffer_assignment=assignment,
        target_rank_by_vehicle=target_rank_by_vehicle,
        layout_template_name=template.template_name,
        opened_buffer_tracks=opened_tracks,
        depot_slot_limit=depot_slot_limit,
        depot_slot_limited=(
            depot_slot_limit is not None
            and _phase1_selected_repair_depot_vehicle_count(
                assignment,
                block_by_id=block_by_id,
            ) >= depot_slot_limit
            and len(ordered_blocks) > len(assignment_by_block)
        ),
    )


def _solve_phase1_backbone_plan(
    *,
    facts_list: list[VehicleStageFacts],
    source_plans: list[SourceTrackPlan],
    reachable_depot_set: frozenset[str],
    depot_slot_limit: int | None,
    master: MasterData,
    respect_existing_buffer_occupancy: bool,
    buffer_occupancy_exempt_vehicle_nos: frozenset[str] | None,
) -> Phase1BackbonePlan:
    admitted_source_plans = list(source_plans)
    reserved_buffer = (
        _build_phase1_runtime_available_buffer_budget(
            facts_list=facts_list,
            master=master,
            exempt_vehicle_nos=buffer_occupancy_exempt_vehicle_nos,
        )
        if respect_existing_buffer_occupancy
        else _build_phase1_buffer_budget(master)
    )
    templates = _phase1_backbone_templates(source_plans=admitted_source_plans)
    template_plans = [
        _solve_phase1_template_backbone_plan(
            source_plans=admitted_source_plans,
            reserved_buffer=reserved_buffer,
            template=template,
            depot_slot_limit=depot_slot_limit,
        )
        for template in templates
    ]
    best_plan = max(
        template_plans,
        key=lambda plan: _phase1_template_assignment_score(
            ordered_blocks=[
                block
                for source_plan in admitted_source_plans
                for block in source_plan.blocks
                if block.uses_buffer and block.block_type in {"bridge_to_depot", "depot_batch"}
            ],
            assignment={
                block_id: plan.selected_buffer_assignment[vehicle_no]
                for block_id, vehicle_no in (
                    (
                        block.block_id,
                        block.vehicle_nos[0],
                    )
                    for source_plan in admitted_source_plans
                    for block in source_plan.blocks
                    if block.block_id in set(plan.selected_block_ids) and block.uses_buffer
                )
            },
            template=next(template for template in templates if template.template_name == plan.layout_template_name),
        ),
    )
    return Phase1BackbonePlan(
        selected_block_ids=best_plan.selected_block_ids,
        selected_source_tracks=best_plan.selected_source_tracks,
        reserved_buffer_by_track=best_plan.reserved_buffer_by_track,
        selected_buffer_assignment=best_plan.selected_buffer_assignment,
        target_rank_by_vehicle=best_plan.target_rank_by_vehicle,
        layout_template_name=best_plan.layout_template_name,
        opened_buffer_tracks=best_plan.opened_buffer_tracks,
        depot_slot_limit=depot_slot_limit,
        depot_slot_limited=(
            depot_slot_limit is not None
            and _phase1_selected_repair_depot_vehicle_count(
                best_plan.selected_buffer_assignment,
                block_by_id={
                    block.block_id: block
                    for source_plan in admitted_source_plans
                    for block in source_plan.blocks
                },
            ) >= depot_slot_limit
            and len(reachable_depot_set) > depot_slot_limit
        ),
    )


def _build_phase1_buffer_budget(master: MasterData) -> dict[str, float]:
    return {
        track: min(
            float(master.tracks[track].effective_length_m),
            PHASE1_USABLE_BUFFER_CAPACITY_M[track],
        )
        for track in JI_BUFFER_TRACKS
    }


def _build_phase1_runtime_available_buffer_budget(
    *,
    facts_list: list[VehicleStageFacts],
    master: MasterData,
    exempt_vehicle_nos: frozenset[str] | None = None,
) -> dict[str, float]:
    budget = _build_phase1_buffer_budget(master)
    exempt_vehicle_nos = exempt_vehicle_nos or frozenset()
    existing_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        if facts.current_track not in JI_BUFFER_TRACKS:
            continue
        if not facts.needs_depot_batch:
            continue
        if facts.vehicle_no in exempt_vehicle_nos:
            continue
        existing_by_track[facts.current_track].append(facts)
    for track_name, members in existing_by_track.items():
        ordered = sorted(members, key=lambda item: (item.current_order, item.vehicle_no))
        occupied = round(
            sum(item.vehicle_length for item in ordered) + max(1.0, len(ordered) - 1),
            1,
        )
        budget[track_name] = round(max(0.0, budget[track_name] - occupied), 1)
    return budget


def _solve_phase1_template_backbone_plan_greedily(
    *,
    source_plans: list[SourceTrackPlan],
    reserved_buffer: dict[str, float],
    template: Phase1LayoutTemplate,
    depot_slot_limit: int | None,
) -> Phase1BackbonePlan:
    remaining = {track: reserved_buffer[track] for track in template.tracks}
    selected_block_ids: list[str] = []
    selected_source_tracks: list[str] = []
    assignment: dict[str, str] = {}
    source_tracks: dict[str, set[str]] = defaultdict(set)
    block_by_id = {
        block.block_id: block
        for plan in source_plans
        for block in plan.blocks
    }
    selected_set: set[str] = set()
    while len([
        block
        for plan in source_plans
        for block in plan.blocks
        if block.uses_buffer and block.block_type in {"bridge_to_depot", "depot_batch"}
    ]) > len([
        block_id
        for block_id in selected_set
        if block_by_id.get(block_id) is not None and block_by_id[block_id].uses_buffer
    ]):
        next_choice = _choose_next_phase1_backbone_block(
            source_plans=source_plans,
            block_by_id=block_by_id,
            selected_block_ids=selected_set,
            remaining=remaining,
            template=template,
            source_tracks=source_tracks,
            depot_slot_limit=depot_slot_limit,
            selected_vehicle_count=_phase1_selected_repair_depot_vehicle_count(
                assignment,
                block_by_id=block_by_id,
            ),
        )
        if next_choice is None:
            remaining_blocks = [
                block
                for plan in source_plans
                for block in plan.blocks
                if block.uses_buffer
                and block.block_type in {"bridge_to_depot", "depot_batch"}
                and block.block_id not in selected_set
            ]
            raise ValueError(
                "phase1 cannot greedily place all depot staging blocks in JI buffers: "
                + ", ".join(
                    f"{block.block_id}:{block.source_track}:{list(block.vehicle_nos)}"
                    for block in remaining_blocks
                )
            )
        plan, block, target_track = next_choice
        if plan.source_track not in selected_source_tracks:
            selected_source_tracks.append(plan.source_track)
        for predecessor_id in _phase1_required_predecessor_closure_ids(
            block=block,
            block_by_id=block_by_id,
        ):
            predecessor = block_by_id.get(predecessor_id)
            if predecessor is None or predecessor.uses_buffer or predecessor_id in selected_set:
                continue
            selected_set.add(predecessor_id)
            selected_block_ids.append(predecessor_id)
        selected_set.add(block.block_id)
        selected_block_ids.append(block.block_id)
        source_tracks[block.source_track].add(target_track)
        remaining[target_track] -= _phase1_buffer_required_length(block)
        for vehicle_no in block.vehicle_nos:
            assignment[vehicle_no] = target_track
    target_rank_by_vehicle = _build_phase1_target_ranks_from_blocks(
        source_plans=source_plans,
        selected_block_ids=frozenset(selected_block_ids),
        buffer_assignment=assignment,
    )
    return Phase1BackbonePlan(
        selected_block_ids=tuple(selected_block_ids),
        selected_source_tracks=tuple(selected_source_tracks),
        reserved_buffer_by_track=reserved_buffer,
        selected_buffer_assignment=assignment,
        target_rank_by_vehicle=target_rank_by_vehicle,
        layout_template_name=template.template_name,
        opened_buffer_tracks=tuple(
            track for track in template.preferred_open_order if track in set(assignment.values())
        ),
        depot_slot_limit=depot_slot_limit,
        depot_slot_limited=(
            depot_slot_limit is not None
            and _phase1_selected_repair_depot_vehicle_count(
                assignment,
                block_by_id=block_by_id,
            ) >= depot_slot_limit
            and any(
                block.uses_buffer
                and block.block_type in {"bridge_to_depot", "depot_batch"}
                and block.block_id not in selected_set
                for plan in source_plans
                for block in plan.blocks
            )
        ),
    )


def _phase1_block_fits_buffer(
    block: Phase1Block,
    track: str,
    remaining: dict[str, float],
) -> bool:
    required = _phase1_buffer_required_length(block)
    return remaining.get(track, 0.0) >= required


def _phase1_buffer_required_length(block: Phase1Block) -> float:
    return block.total_length_m + max(1.0, len(block.vehicle_nos) - 1)


def _choose_next_phase1_backbone_block(
    *,
    source_plans: list[SourceTrackPlan],
    block_by_id: dict[str, Phase1Block],
    selected_block_ids: set[str],
    remaining: dict[str, float],
    template: Phase1LayoutTemplate,
    source_tracks: dict[str, set[str]],
    depot_slot_limit: int | None,
    selected_vehicle_count: int,
) -> tuple[SourceTrackPlan, Phase1Block, str] | None:
    candidates: list[tuple[tuple[Any, ...], SourceTrackPlan, Phase1Block, str]] = []
    track_sources: dict[str, set[str]] = defaultdict(set)
    track_families: dict[str, set[str]] = defaultdict(set)
    for plan in source_plans:
        for block in plan.blocks:
            if block.block_id in selected_block_ids:
                continue
            if block.block_type not in {"bridge_to_depot", "depot_batch"} or not block.uses_buffer:
                continue
            if (
                depot_slot_limit is not None
                and selected_vehicle_count + _phase1_repair_depot_vehicle_count(block) > depot_slot_limit
            ):
                continue
            if not _phase1_backbone_predecessors_ready(
                block=block,
                block_by_id=block_by_id,
                selected_block_ids=selected_block_ids,
            ):
                continue
            for target_track in _phase1_template_track_candidates(
                block=block,
                template=template,
                remaining=remaining,
                source_tracks=source_tracks,
                track_sources=track_sources,
                track_families=track_families,
            )[:2]:
                candidates.append((
                    (
                        0 if block.layout_role == "core" else 1,
                        0 if target_track in source_tracks.get(block.source_track, set()) else 1,
                        0 if target_track not in template.support_tracks else 1,
                        0 if block.block_type == "bridge_to_depot" else 1,
                        -block.released_depot_vehicle_count,
                        -len(block.vehicle_nos),
                        plan.source_priority_score,
                        block.phase3_rank_key,
                        block.source_track,
                        block.block_id,
                    ),
                    plan,
                    block,
                    target_track,
                ))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, plan, block, target_track = candidates[0]
    return plan, block, target_track


def _phase1_backbone_predecessors_ready(
    *,
    block: Phase1Block,
    block_by_id: dict[str, Phase1Block],
    selected_block_ids: set[str],
) -> bool:
    for predecessor_id in block.required_predecessor_ids:
        predecessor = block_by_id.get(predecessor_id)
        if predecessor is None:
            continue
        if predecessor.uses_buffer and predecessor_id not in selected_block_ids:
            return False
    return True


def _build_phase1_target_ranks_from_blocks(
    *,
    source_plans: list[SourceTrackPlan],
    selected_block_ids: frozenset[str],
    buffer_assignment: dict[str, str],
) -> dict[str, int]:
    units_by_track: dict[str, list[Phase1Block]] = defaultdict(list)
    for plan in source_plans:
        for block in plan.blocks:
            if block.block_id not in selected_block_ids or not block.uses_buffer:
                continue
            target_track = buffer_assignment.get(block.vehicle_nos[0])
            if target_track is not None:
                units_by_track[target_track].append(block)
    result: dict[str, int] = {}
    for track, blocks in units_by_track.items():
        ordered = sorted(
            blocks,
            key=lambda block: (
                block.phase3_rank_key,
                block.source_track,
                block.source_order_start,
            ),
        )
        rank = 1
        for block in ordered:
            for offset, vehicle_no in enumerate(block.vehicle_nos):
                result[vehicle_no] = rank + offset
            rank += len(block.vehicle_nos)
    return result


def _build_phase1_wave_plans(
    *,
    facts_list: list[VehicleStageFacts],
    source_plans: list[SourceTrackPlan],
    backbone_plan: Phase1BackbonePlan,
    master: MasterData,
) -> tuple[tuple[Phase1WavePlan, ...], Phase1FinishPlan, dict[str, tuple[str, str]]]:
    all_selected_ids = _phase1_wave_a_block_ids(source_plans=source_plans, backbone_plan=backbone_plan)
    block_by_id = {
        block.block_id: block
        for plan in source_plans
        for block in plan.blocks
    }
    selected_set = set(all_selected_ids)
    macro_tasks = _build_phase1_macro_tasks(
        source_plans=source_plans,
        selected_block_ids=selected_set,
    )
    wave_definitions: list[tuple[str, str, str, tuple[str, ...], Phase1MacroTask]] = []
    wave_index = 1
    for macro_task in macro_tasks:
        for wave_role, block_ids in macro_task.wave_chunks:
            wave_definitions.append(
                (
                    f"wave_{wave_index:02d}_{macro_task.source_track}_{block_ids[0]}",
                    wave_role,
                    macro_task.source_track,
                    block_ids,
                    macro_task,
                )
            )
            wave_index += 1
    cumulative_ids: list[str] = []
    wave_plans: list[Phase1WavePlan] = []
    for wave_name, wave_role, selected_source_track, added_ids, macro_task in wave_definitions:
        cumulative_ids.extend(added_ids)
        required_predecessor_ids = tuple(
            dict.fromkeys(
                predecessor_id
                for block_id in added_ids
                for predecessor_id in block_by_id[block_id].required_predecessor_ids
                if predecessor_id in selected_set and predecessor_id not in added_ids
            )
        )
        wave_finish_plan = Phase1FinishPlan(
            selected_block_ids=tuple(added_ids),
            goal_overrides={},
        )
        goal_overrides = _materialize_phase1_finish_goal_overrides(
            facts_list=facts_list,
            source_plans=source_plans,
            finish_plan=wave_finish_plan,
            buffer_assignment=backbone_plan.selected_buffer_assignment,
            master=master,
        )
        wave_active_buffer_assignment = _phase1_wave_buffer_assignment(
            source_plans=source_plans,
            selected_block_ids=frozenset(added_ids),
            buffer_assignment=backbone_plan.selected_buffer_assignment,
        )
        wave_target_ranks = _build_phase1_target_ranks_from_blocks(
            source_plans=source_plans,
            selected_block_ids=frozenset(added_ids),
            buffer_assignment=wave_active_buffer_assignment,
        )
        wave_selected_vehicle_nos = frozenset(set(wave_active_buffer_assignment) | set(goal_overrides))
        wave_plans.append(
            Phase1WavePlan(
                wave_name=wave_name,
                wave_role=wave_role,
                wave_type=wave_role,
                selected_source_track=selected_source_track,
                selected_block_ids=tuple(added_ids),
                required_predecessor_ids=required_predecessor_ids,
                selected_vehicle_nos=wave_selected_vehicle_nos,
                buffer_assignment=wave_active_buffer_assignment,
                goal_overrides=dict(goal_overrides),
                target_rank_by_vehicle=wave_target_ranks,
                diagnostics={
                    **_build_phase1_wave_diagnostics(
                    source_plans=source_plans,
                    wave_name=wave_name,
                    wave_role=wave_role,
                    selected_source_track=selected_source_track,
                    added_block_ids=tuple(added_ids),
                    cumulative_block_ids=tuple(dict.fromkeys(cumulative_ids)),
                    ),
                    "macroTaskId": macro_task.task_id,
                    "macroTaskBlockIds": list(macro_task.block_ids),
                    "macroTaskRequiredPredecessorIds": list(macro_task.required_predecessor_ids),
                    "macroTaskVehicleNos": list(macro_task.vehicle_nos),
                    "macroTaskSourceRole": macro_task.source_role,
                    "macroTaskTopologyZones": list(macro_task.topology_zones),
                    "macroTaskThroatGroups": list(macro_task.throat_groups),
                    "macroTaskScoreKey": list(macro_task.score_key),
                    "macroTaskBufferVehicleCount": macro_task.buffer_vehicle_count,
                    "macroTaskCleanupVehicleCount": macro_task.cleanup_vehicle_count,
                    "macroTaskReleasedDepotVehicleCount": macro_task.released_depot_vehicle_count,
                    "macroTaskReleasedFinishVehicleCount": macro_task.released_finish_vehicle_count,
                    "macroTaskPressureGain": macro_task.pressure_gain,
                },
            )
        )
    merged_finish_plan = Phase1FinishPlan(
        selected_block_ids=tuple(dict.fromkeys(cumulative_ids)),
        goal_overrides={},
    )
    final_goal_overrides = _materialize_phase1_finish_goal_overrides(
        facts_list=facts_list,
        source_plans=source_plans,
        finish_plan=merged_finish_plan,
        buffer_assignment=backbone_plan.selected_buffer_assignment,
        master=master,
    )
    return tuple(wave_plans), merged_finish_plan, final_goal_overrides


def _phase1_group_cleanup_wave_block_ids(
    *,
    source_role: str,
    cleanup_blocks: list[Phase1Block],
) -> tuple[tuple[str, ...], ...]:
    if not cleanup_blocks:
        return tuple()
    if source_role not in {"receiving_storage", "yard_storage"}:
        return tuple((block.block_id,) for block in cleanup_blocks)
    groups: list[tuple[str, ...]] = []
    current_group: list[str] = []
    current_vehicle_count = 0
    current_length_m = 0.0
    for block in cleanup_blocks:
        vehicle_count = len(block.vehicle_nos)
        projected_vehicle_count = current_vehicle_count + vehicle_count
        projected_length_m = current_length_m + block.total_length_m
        if current_group and (
            len(current_group) >= 2
            or projected_vehicle_count > 4
            or projected_length_m > 42.0
        ):
            groups.append(tuple(current_group))
            current_group = []
            current_vehicle_count = 0
            current_length_m = 0.0
        current_group.append(block.block_id)
        current_vehicle_count += vehicle_count
        current_length_m += block.total_length_m
    if current_group:
        groups.append(tuple(current_group))
    return tuple(groups)


def _build_phase1_macro_tasks(
    *,
    source_plans: list[SourceTrackPlan],
    selected_block_ids: set[str],
) -> tuple[Phase1MacroTask, ...]:
    tasks: list[Phase1MacroTask] = []
    for plan in source_plans:
        selected_blocks = [block for block in plan.blocks if block.block_id in selected_block_ids]
        if not selected_blocks:
            continue
        source_role = _phase1_source_chain_role(source_track=plan.source_track)
        first_buffer_index = next(
            (index for index, block in enumerate(selected_blocks) if block.uses_buffer),
            None,
        )
        wave_chunks: list[tuple[str, tuple[str, ...]]] = []
        cleanup_buffer: list[Phase1Block] = []
        seen_buffer = False

        def flush_cleanup() -> None:
            nonlocal cleanup_buffer
            if not cleanup_buffer:
                return
            wave_role = "source_local_finish" if seen_buffer else "source_clearance"
            for cleanup_ids in _phase1_group_cleanup_wave_block_ids(
                source_role=source_role,
                cleanup_blocks=cleanup_buffer,
            ):
                wave_chunks.append((wave_role, cleanup_ids))
            cleanup_buffer = []

        for block in selected_blocks:
            if block.uses_buffer:
                flush_cleanup()
                seen_buffer = True
                wave_chunks.append(("source_marshalling", (block.block_id,)))
            else:
                cleanup_buffer.append(block)
        flush_cleanup()
        if not wave_chunks:
            continue
        tasks.append(
            _make_phase1_macro_task(
                plan=plan,
                source_role=source_role,
                wave_chunks=tuple(wave_chunks),
            )
        )
    tasks.sort(key=lambda task: task.score_key)
    return tuple(tasks)


def _make_phase1_macro_task(
    *,
    plan: SourceTrackPlan,
    source_role: str,
    wave_chunks: tuple[tuple[str, tuple[str, ...]], ...],
) -> Phase1MacroTask:
    block_by_id = {block.block_id: block for block in plan.blocks}
    block_ids = tuple(
        dict.fromkeys(
            block_id
            for _wave_role, chunk_ids in wave_chunks
            for block_id in chunk_ids
            if block_id in block_by_id
        )
    )
    blocks = [block_by_id[block_id] for block_id in block_ids]
    vehicle_nos = tuple(
        dict.fromkeys(
            vehicle_no
            for block in blocks
            for vehicle_no in block.vehicle_nos
        )
    )
    required_predecessor_ids = tuple(
        dict.fromkeys(
            predecessor_id
            for block in blocks
            for predecessor_id in block.required_predecessor_ids
            if predecessor_id not in block_ids
        )
    )
    buffer_vehicle_count = sum(len(block.vehicle_nos) for block in blocks if block.uses_buffer)
    cleanup_vehicle_count = sum(len(block.vehicle_nos) for block in blocks if not block.uses_buffer)
    released_depot = sum(block.released_depot_vehicle_count for block in blocks)
    released_finish = sum(block.released_finish_vehicle_count for block in blocks)
    pressure_gain = sum(block.pressure_gain for block in blocks)
    topology_zones = tuple(dict.fromkeys(block.topology_zone for block in blocks))
    throat_groups = tuple(dict.fromkeys(block.throat_group for block in blocks))
    source_opening = any(
        not block.uses_buffer and block.released_depot_vehicle_count > 0
        for block in blocks
    )
    has_buffer = buffer_vehicle_count > 0
    cleanup_before_buffer = (
        bool(wave_chunks)
        and wave_chunks[0][0] == "source_clearance"
        and any(block_by_id[block_id].uses_buffer for block_id in block_ids)
    )
    score_key = (
        0 if cleanup_before_buffer else 1,
        0 if source_opening else 1,
        0 if source_role in {"wash_gate", "work_gate", "work_support"} else 1,
        0 if has_buffer else 1,
        -released_depot,
        -buffer_vehicle_count,
        -pressure_gain,
        _phase1_source_chain_priority(source_role),
        len(wave_chunks),
        cleanup_vehicle_count,
        *_phase1_flat_sort_key(plan.source_priority_score),
        plan.source_track,
    )
    return Phase1MacroTask(
        task_id=f"MT::{plan.source_track}::{block_ids[0]}",
        source_track=plan.source_track,
        source_role=source_role,
        wave_chunks=wave_chunks,
        block_ids=block_ids,
        required_predecessor_ids=required_predecessor_ids,
        vehicle_nos=vehicle_nos,
        buffer_vehicle_count=buffer_vehicle_count,
        cleanup_vehicle_count=cleanup_vehicle_count,
        released_depot_vehicle_count=released_depot,
        released_finish_vehicle_count=released_finish,
        pressure_gain=pressure_gain,
        topology_zones=topology_zones,
        throat_groups=throat_groups,
        score_key=score_key,
    )


def _phase1_flat_sort_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(item for nested in value for item in _phase1_flat_sort_key(nested))
    return (value,)


def _phase1_wave_buffer_assignment(
    *,
    source_plans: list[SourceTrackPlan],
    selected_block_ids: frozenset[str],
    buffer_assignment: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for plan in source_plans:
        for block in plan.blocks:
            if block.block_id not in selected_block_ids or not block.uses_buffer:
                continue
            target_track = buffer_assignment.get(block.vehicle_nos[0])
            if target_track is None:
                continue
            for vehicle_no in block.vehicle_nos:
                result[vehicle_no] = target_track
    return result


def _phase1_wave_a_block_ids(
    *,
    source_plans: list[SourceTrackPlan],
    backbone_plan: Phase1BackbonePlan,
) -> tuple[str, ...]:
    selected_ids: list[str] = []
    selected_backbone = set(backbone_plan.selected_block_ids)
    for plan in source_plans:
        for block in plan.blocks:
            if block.block_id in selected_backbone:
                selected_ids.append(block.block_id)
            elif block.block_type == "clear_cun4":
                selected_ids.append(block.block_id)
            elif block.target_source == "PHASE1_CLEAR_JI":
                selected_ids.append(block.block_id)
    selected_ids.extend(
        _phase1_required_clearance_closure_block_ids(
            source_plans=source_plans,
            backbone_plan=backbone_plan,
        )
    )
    return tuple(dict.fromkeys(selected_ids))


def _phase1_required_clearance_closure_block_ids(
    *,
    source_plans: list[SourceTrackPlan],
    backbone_plan: Phase1BackbonePlan,
) -> tuple[str, ...]:
    selected_backbone = set(backbone_plan.selected_block_ids)
    chosen_ids: list[str] = []
    candidate_closures: list[tuple[tuple[Any, ...], list[str]]] = []
    block_by_id = {
        block.block_id: block
        for plan in source_plans
        for block in plan.blocks
    }
    for plan in source_plans:
        if any(block.block_id in selected_backbone and block.uses_buffer for block in plan.blocks):
            continue
        if _phase1_source_chain_role(source_track=plan.source_track) not in {"receiving_storage", "yard_storage"}:
            continue
        cleanup_blocks = [
            block
            for block in plan.blocks
            if not block.uses_buffer and block.target_source == "PHASE1_LOCAL_FINISH"
        ]
        if not cleanup_blocks:
            continue
        best_prefix_ids: tuple[str, ...] = tuple()
        best_prefix_score: tuple[Any, ...] | None = None
        for anchor_block in cleanup_blocks:
            closure_ids = _phase1_cleanup_predecessor_closure_ids(
                anchor_block=anchor_block,
                block_by_id=block_by_id,
            )
            prefix_ids, prefix_score = _phase1_best_storage_enabling_prefix_ids(
                closure_ids=closure_ids,
                block_by_id=block_by_id,
            )
            if not prefix_ids:
                continue
            if best_prefix_score is None or prefix_score > best_prefix_score:
                best_prefix_score = prefix_score
                best_prefix_ids = prefix_ids
        prefix_blocks = [
            block_by_id[block_id]
            for block_id in best_prefix_ids
            if block_id in block_by_id
        ]
        released_depot = max(
            (block.released_depot_vehicle_count for block in prefix_blocks),
            default=0,
        )
        released_finish = max(
            (block.released_finish_vehicle_count for block in prefix_blocks),
            default=0,
        )
        if released_depot <= 0 and released_finish <= 0:
            continue
        candidate_closures.append(
            (
                (
                    -released_depot,
                    -released_finish,
                    _phase1_track_priority(plan.source_track),
                    plan.source_track,
                ),
                list(best_prefix_ids),
            )
        )
    for _score, block_ids in sorted(candidate_closures, key=lambda item: item[0])[:2]:
        chosen_ids.extend(block_ids)
    return tuple(chosen_ids)


def _phase1_best_storage_enabling_prefix_ids(
    *,
    closure_ids: tuple[str, ...],
    block_by_id: dict[str, Phase1Block],
) -> tuple[tuple[str, ...], tuple[Any, ...]] | tuple[tuple[()], None]:
    best_ids: tuple[str, ...] = tuple()
    best_score: tuple[Any, ...] | None = None
    prefix_ids: list[str] = []
    prefix_vehicle_count = 0
    prefix_length_m = 0.0
    for block_id in closure_ids:
        block = block_by_id.get(block_id)
        if block is None:
            continue
        prefix_ids.append(block_id)
        prefix_vehicle_count += len(block.vehicle_nos)
        prefix_length_m += block.total_length_m
        score = (
            block.released_depot_vehicle_count,
            block.released_finish_vehicle_count,
            block.pressure_gain,
            -len(prefix_ids),
            -prefix_vehicle_count,
            -round(prefix_length_m, 1),
            -block.source_order_end,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_ids = tuple(prefix_ids)
    if best_score is None:
        return tuple(), None
    return best_ids, best_score


def _phase1_cleanup_predecessor_closure_ids(
    *,
    anchor_block: Phase1Block,
    block_by_id: dict[str, Phase1Block],
) -> tuple[str, ...]:
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def visit(block: Phase1Block) -> None:
        for predecessor_id in block.required_predecessor_ids:
            predecessor = block_by_id.get(predecessor_id)
            if predecessor is None or predecessor.uses_buffer:
                continue
            if predecessor.block_id in seen:
                continue
            seen.add(predecessor.block_id)
            visit(predecessor)
            ordered_ids.append(predecessor.block_id)

    visit(anchor_block)
    if anchor_block.block_id not in seen:
        ordered_ids.append(anchor_block.block_id)
    return tuple(dict.fromkeys(ordered_ids))


def _phase1_wave_b_block_ids(
    *,
    source_plans: list[SourceTrackPlan],
    already_selected_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return ()


def _phase1_wave_c_block_ids(
    *,
    source_plans: list[SourceTrackPlan],
    already_selected_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return ()


def _build_phase1_wave_diagnostics(
    *,
    source_plans: list[SourceTrackPlan],
    wave_name: str,
    wave_role: str,
    selected_source_track: str,
    added_block_ids: tuple[str, ...],
    cumulative_block_ids: tuple[str, ...],
) -> dict[str, Any]:
    block_by_id = {
        block.block_id: block
        for plan in source_plans
        for block in plan.blocks
    }
    added_blocks = [block_by_id[block_id] for block_id in added_block_ids if block_id in block_by_id]
    cumulative_blocks = [block_by_id[block_id] for block_id in cumulative_block_ids if block_id in block_by_id]
    pressure_cut_counts = Counter(
        _phase1_block_pressure_cut(
            {
                "blockType": block.block_type,
                "sourceTrack": block.source_track,
            }
        )
        for block in added_blocks
    )
    return {
        "waveName": wave_name,
        "waveRole": wave_role,
        "waveType": wave_role,
        "selectedSourceTrack": selected_source_track,
        "selectedSourceRole": (
            _phase1_source_chain_role(source_track=selected_source_track)
            if selected_source_track
            else ""
        ),
        "addedBlockIds": list(added_block_ids),
        "requiredPredecessorIds": list(
            dict.fromkeys(
                predecessor_id
                for block in added_blocks
                for predecessor_id in block.required_predecessor_ids
                if predecessor_id not in added_block_ids
            )
        ),
        "addedBlockCount": len(added_blocks),
        "addedVehicleCount": sum(len(block.vehicle_nos) for block in added_blocks),
        "addedBufferVehicleCount": sum(len(block.vehicle_nos) for block in added_blocks if block.uses_buffer),
        "addedCleanupVehicleCount": sum(len(block.vehicle_nos) for block in added_blocks if not block.uses_buffer),
        "addedSourceTracks": sorted({block.source_track for block in added_blocks}),
        "addedThroatGroups": dict(sorted(Counter(block.throat_group for block in added_blocks).items())),
        "pressureCutCounts": dict(sorted(pressure_cut_counts.items())),
        "releasedDepotVehicleCount": sum(block.released_depot_vehicle_count for block in added_blocks),
        "releasedFinishVehicleCount": sum(block.released_finish_vehicle_count for block in added_blocks),
        "pressureGain": sum(block.pressure_gain for block in added_blocks),
        "containsBufferMove": any(block.uses_buffer for block in added_blocks),
        "maxRequiredPredecessorDepth": max(
            (len(block.required_predecessor_ids) for block in added_blocks),
            default=0,
        ),
        "cumulativeBlockCount": len(cumulative_blocks),
        "cumulativeVehicleCount": sum(len(block.vehicle_nos) for block in cumulative_blocks),
    }


def _phase1_optional_finish_order(block: Phase1Block) -> tuple[Any, ...]:
    return (
        0 if block.source_track in WASH_CONFLICT_TRACKS else 1,
        0 if block.source_track in {"调棚", "预修", "调北", "抛"} else 1,
        _phase1_local_source_priority(block.source_track),
        0 if block.block_type == "prefix_clear" and block.released_depot_vehicle_count > 0 else 1,
        0 if block.block_type == "tail_finish" else 1,
        block.phase3_rank_key,
        block.source_order_start,
        block.block_id,
    )


def _phase1_pressure_relief_order(block: Phase1Block) -> tuple[Any, ...]:
    return (
        -block.pressure_gain,
        block.coupling_degree,
        0 if block.topology_zone == "receiving" else 1,
        0 if block.throat_group in {"G_STORAGE", "G_L7"} else 1,
        _phase1_local_source_priority(block.source_track),
        block.phase3_rank_key,
        block.source_order_start,
        block.block_id,
    )


def _phase1_cleanup_wave_order(block: Phase1Block) -> tuple[Any, ...]:
    return (
        block.coupling_degree,
        -block.pressure_gain,
        _phase1_local_source_priority(block.source_track),
        block.phase3_rank_key,
        block.source_order_start,
        block.block_id,
    )


def _materialize_phase1_finish_goal_overrides(
    *,
    facts_list: list[VehicleStageFacts],
    source_plans: list[SourceTrackPlan],
    finish_plan: Phase1FinishPlan,
    buffer_assignment: dict[str, str],
    master: MasterData,
) -> dict[str, tuple[str, str]]:
    selected_finish_ids = set(finish_plan.selected_block_ids)
    selected_finish_blocks = [
        block
        for plan in source_plans
        for block in plan.blocks
        if block.block_id in selected_finish_ids and not block.uses_buffer
    ]
    goal_overrides: dict[str, tuple[str, str]] = {}
    temp_packages: list[Phase1LayoutPackage] = []
    for block in selected_finish_blocks:
        if block.target_track is not None:
            for vehicle_no in block.vehicle_nos:
                goal_overrides[vehicle_no] = (block.target_track, block.target_source)
            continue
        temp_packages.append(
            Phase1LayoutPackage(
                package_id=block.block_id,
                chain_id=f"CHAIN::{block.source_track}",
                package_kind="temp_repark",
                source_track=block.source_track,
                vehicle_nos=block.vehicle_nos,
                total_length_m=block.total_length_m,
                target_track="",
                target_source=block.target_source,
                final_family=block.final_family,
                min_spot_priority=999,
                source_order_start=block.source_order_start,
                source_order_end=block.source_order_end,
                buffer_preference=tuple(),
                uses_buffer=False,
                pressure_cut=_phase1_block_pressure_cut({
                    "blockType": block.block_type,
                    "sourceTrack": block.source_track,
                }),
                reason_tags=tuple(),
                execution_layer="L2_REQUIRED_CLEAR",
                complexity_cost=max(1, len(block.vehicle_nos)),
                source_chain_role=_phase1_source_chain_role(source_track=block.source_track),
                is_required_for_backbone=True,
                segment_role="cleanup",
                source_segment_index=0,
                source_segment_count=0,
                source_total_vehicle_count=0,
                requires_previous_segment=False,
            )
        )
    temp_assignments = _assign_phase1_temp_targets(
        facts_list=facts_list,
        packages=temp_packages,
        buffer_assignment=buffer_assignment,
        master=master,
    )
    for package in temp_packages:
        target_track = temp_assignments.get(package.package_id)
        if target_track is None:
            continue
        for vehicle_no in package.vehicle_nos:
            goal_overrides[vehicle_no] = (target_track, package.target_source)
    return goal_overrides


def _build_phase1_block_diagnostics(
    *,
    facts_list: list[VehicleStageFacts],
    source_plans: list[SourceTrackPlan],
    reachable_depot_set: frozenset[str],
    backbone_plan: Phase1BackbonePlan,
    finish_plan: Phase1FinishPlan,
    deferred_vehicle_nos: frozenset[str],
    goal_overrides: dict[str, tuple[str, str]],
    buffer_assignment: dict[str, str],
    target_rank_by_vehicle: dict[str, int],
    master: MasterData,
) -> dict[str, Any]:
    all_blocks = [block for plan in source_plans for block in plan.blocks]
    facts_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    block_by_id = {block.block_id: block for block in all_blocks}
    selected_backbone = [block_by_id[block_id] for block_id in backbone_plan.selected_block_ids]
    selected_finish = [
        block_by_id[block_id]
        for block_id in finish_plan.selected_block_ids
        if block_id in block_by_id and not block_by_id[block_id].uses_buffer
    ]
    selected_all_source_tracks = sorted({
        block.source_track
        for block in selected_backbone + selected_finish
    })
    selected_buffer_blocks = [block for block in selected_backbone if block.uses_buffer]
    selected_buffer_vehicle_nos = frozenset(
        vehicle_no
        for block in selected_buffer_blocks
        for vehicle_no in block.vehicle_nos
    )
    selected_finish_vehicle_nos = frozenset(
        vehicle_no
        for block in selected_finish
        for vehicle_no in block.vehicle_nos
    )
    region_completion = _phase1_non_depot_region_completion(
        facts_list=facts_list,
        selected_vehicle_nos=selected_buffer_vehicle_nos,
        goal_overrides=goal_overrides,
        deferred_vehicle_nos=deferred_vehicle_nos,
    )
    all_depot_candidates = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if _is_phase1_candidate(facts)
    )
    cun4_pending = [
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in {"存4北", "存4南"}
        and not facts.is_cun4bei_final
        and facts.vehicle_no not in goal_overrides
    ]
    unselected_bridge_blocks = [
        {
            "blockId": block.block_id,
            "sourceTrack": block.source_track,
            "vehicleNos": list(block.vehicle_nos),
            "releasedDepotVehicleCount": block.released_depot_vehicle_count,
        }
        for block in all_blocks
        if block.block_type == "bridge_to_depot" and block.block_id not in backbone_plan.selected_block_ids
    ]
    block_projection = [
        {
            "blockId": block.block_id,
            "sourceTrack": block.source_track,
            "blockType": block.block_type,
            "vehicleNos": list(block.vehicle_nos),
            "targetTrack": block.target_track,
            "targetSource": block.target_source,
            "usesBuffer": block.uses_buffer,
            "bufferPreference": list(block.buffer_preference),
            "releasedDepotVehicleCount": block.released_depot_vehicle_count,
            "releasedFinishVehicleCount": block.released_finish_vehicle_count,
            "requiredPredecessorIds": list(block.required_predecessor_ids),
            "layoutRole": block.layout_role,
            "selectedBackbone": block.block_id in backbone_plan.selected_block_ids,
            "selectedFinish": block.block_id in finish_plan.selected_block_ids,
            "bufferTrack": (
                buffer_assignment.get(block.vehicle_nos[0])
                if block.uses_buffer and block.vehicle_nos
                else None
            ),
        }
        for block in all_blocks
    ]
    source_summary_projection = []
    for plan in source_plans:
        backbone_blocks = [block for block in plan.blocks if block.uses_buffer]
        cleanup_blocks = [block for block in plan.blocks if not block.uses_buffer]
        opening_gain_units = sum(len(block.vehicle_nos) * 2 for block in backbone_blocks)
        opening_cost_units = len(backbone_blocks) + sum(len(block.vehicle_nos) for block in cleanup_blocks)
        opening_score = opening_gain_units - opening_cost_units
        selected_for_backbone = any(
            block.block_id in backbone_plan.selected_block_ids and block.uses_buffer
            for block in plan.blocks
        )
        source_summary_projection.append(
            {
                "sourceTrack": plan.source_track,
                "sourceChainRole": _phase1_source_chain_role(source_track=plan.source_track),
                "backbonePackageCount": len(backbone_blocks),
                "backboneVehicleCount": sum(len(block.vehicle_nos) for block in backbone_blocks),
                "requiredCleanupPackageCount": len(cleanup_blocks),
                "requiredCleanupVehicleCount": sum(len(block.vehicle_nos) for block in cleanup_blocks),
                "optionalCleanupPackageCount": 0,
                "openingCostUnits": opening_cost_units,
                "openingGainUnits": opening_gain_units,
                "openingScore": opening_score,
                "primaryPackageIds": [block.block_id for block in backbone_blocks[:1]],
                "primaryVehicleCount": len(backbone_blocks[0].vehicle_nos) if backbone_blocks else 0,
                "primaryRequiredLengthM": (
                    _phase1_buffer_required_length(backbone_blocks[0]) if backbone_blocks else 0.0
                ),
                "admissionTier": "core" if opening_score > 0 else "weak",
                "selectedForBackbone": selected_for_backbone,
                "admissionDecision": "primary" if selected_for_backbone else "deferred",
                "slotIndex": (
                    selected_all_source_tracks.index(plan.source_track) + 1
                    if plan.source_track in selected_all_source_tracks
                    else None
                ),
                "rejectionReason": None if selected_for_backbone else ("weak_source" if opening_score <= 0 else "backbone_slot_limit"),
            }
        )
    source_plan_projection = [
        {
            "sourceTrack": plan.source_track,
            "reachableDepotVehicleNos": list(plan.reachable_depot_vehicle_nos),
            "reachableFinishVehicleNos": list(plan.reachable_finish_vehicle_nos),
            "cun4ClearRequired": plan.cun4_clear_required,
            "bufferDemandM": plan.buffer_demand_m,
            "sourcePriorityScore": list(plan.source_priority_score),
            "blockIds": [block.block_id for block in plan.blocks],
            "candidateCount": len(plan.reachable_depot_vehicle_nos),
            "candidateLengthM": round(
                sum(block.total_length_m for block in plan.blocks if block.uses_buffer),
                1,
            ),
            "localFinishCount": sum(
                1 for block in plan.blocks if block.block_type in {"clear_cun4", "prefix_clear", "tail_finish"}
            ),
            "hiddenCandidateCount": 0,
            "sourceRole": _phase1_legacy_source_role(plan),
        }
        for plan in source_plans
    ]
    local_finish_plans = [
        {
            "planId": block.block_id,
            "sourceTrack": block.source_track,
            "targetTrack": block.target_track,
            "vehicleNos": list(block.vehicle_nos),
            "totalLengthM": block.total_length_m,
            "priorityTag": block.block_type,
            "clusterKind": block.block_type,
            "completionGain": len(block.vehicle_nos),
            "releasedCandidateGain": block.released_depot_vehicle_count,
            "sourcePendingPressure": block.released_finish_vehicle_count,
        }
        for block in all_blocks
        if block.block_type in {"clear_cun4", "prefix_clear", "tail_finish"}
    ]
    package_edges = [
        {
            "from": predecessor,
            "to": block.block_id,
            "reason": "same_source_contiguous",
        }
        for block in all_blocks
        for predecessor in block.required_predecessor_ids
    ]
    macro_tasks = _build_phase1_macro_tasks(
        source_plans=source_plans,
        selected_block_ids=set(finish_plan.selected_block_ids),
    )
    opened_source_tracks = {
        block.source_track
        for block in selected_finish
        if block.block_type == "prefix_clear" and block.released_depot_vehicle_count > 0
    }
    released_vehicle_nos = sorted(
        {
            vehicle_no
            for block in selected_buffer_blocks
            if block.source_track in opened_source_tracks
            for vehicle_no in block.vehicle_nos
        }
    )
    hidden_vehicle_nos = sorted(
        vehicle_no
        for vehicle_no in reachable_depot_set
        if vehicle_no not in selected_buffer_vehicle_nos
    )
    buffer_block_order: dict[str, list[str]] = defaultdict(list)
    for block in selected_buffer_blocks:
        track = buffer_assignment.get(block.vehicle_nos[0])
        if track is not None:
            buffer_block_order[track].append(block.block_id)
    selected_temp_tracks_by_source: dict[str, list[str]] = defaultdict(list)
    selected_cleanup_by_source: dict[str, dict[str, int]] = {}
    for block in selected_finish:
        source_row = selected_cleanup_by_source.setdefault(
            block.source_track,
            {
                "requiredLocalFinishCount": 0,
                "requiredTempReparkCount": 0,
                "optionalLocalFinishCount": 0,
                "optionalTempReparkCount": 0,
            },
        )
        resolved_target_track = goal_overrides.get(block.vehicle_nos[0], (block.target_track, block.target_source))[0]
        if resolved_target_track is not None and block.target_source in {
            "PHASE1_BLOCKER_BUCKET_WORK",
            "PHASE1_BLOCKER_BUCKET_YARD",
            "PHASE1_CLEAR_JI",
        }:
            source_row["requiredTempReparkCount"] += 1
            selected_temp_tracks_by_source[block.source_track].append(resolved_target_track)
        else:
            source_row["requiredLocalFinishCount"] += 1
    task_package_projection = []
    for block in block_projection:
        block_facts = [facts_by_vehicle[vehicle_no] for vehicle_no in block["vehicleNos"] if vehicle_no in facts_by_vehicle]
        target_source = str(block["targetSource"])
        package_kind = (
            "depot_batch"
            if block["usesBuffer"]
            else "local_finish"
            if target_source not in {PHASE1_BLOCKER_BUCKET_WORK, PHASE1_BLOCKER_BUCKET_YARD, "PHASE1_CLEAR_JI"}
            else "temp_repark"
        )
        special_tags = set()
        if any(facts.need_weigh for facts in block_facts):
            special_tags.add("need_weigh_singleton")
        if any(facts.is_heavy for facts in block_facts):
            special_tags.add("heavy_cap")
        if any(facts.is_close_door for facts in block_facts) and any(
            facts.final_target_track == "存4北" for facts in block_facts
        ):
            special_tags.add("close_door_cun4_singleton")
        source_chain_role = _phase1_source_chain_role(source_track=str(block["sourceTrack"]))
        block_copy = dict(block)
        block_copy["packageId"] = block["blockId"]
        block_copy["chainId"] = f"CHAIN::{block['sourceTrack']}"
        block_copy["packageKind"] = package_kind
        block_copy["pressureCut"] = _phase1_block_pressure_cut(block_copy)
        block_copy["reasonTags"] = sorted(
            special_tags
            | {str(block["blockType"]), str(block_copy["pressureCut"])}
        )
        block_copy["executionLayer"] = "L1_BACKBONE" if block["usesBuffer"] else "L2_REQUIRED_CLEAR"
        block_copy["complexityCost"] = max(1, len(block["vehicleNos"]))
        block_copy["sourceChainRole"] = source_chain_role
        block_copy["isRequiredForBackbone"] = True
        block_copy["segmentRole"] = "core" if block["usesBuffer"] else "cleanup"
        block_copy["sourceSegmentIndex"] = 1
        block_copy["sourceSegmentCount"] = 1
        block_copy["sourceTotalVehicleCount"] = len(block["vehicleNos"])
        block_copy["requiresPreviousSegment"] = False
        block_copy["selected"] = bool(block["selectedBackbone"] or block["selectedFinish"])
        task_package_projection.append(block_copy)
    depot_compiled_vehicle_nos = sorted(vehicle_no for vehicle_no in reachable_depot_set if vehicle_no in buffer_assignment)
    cun4_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in {"存4北", "存4南"}
    )
    cun4_cleared_vehicle_nos = sorted(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in {"存4北", "存4南"}
        and (facts.vehicle_no in goal_overrides or facts.vehicle_no in buffer_assignment)
    )
    ji_non_depot_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch
    )
    ji_cleared_vehicle_nos = sorted(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch and facts.vehicle_no in goal_overrides
    )
    selected_layer_counts = Counter(
        "L1_BACKBONE" if block.uses_buffer else "L2_REQUIRED_CLEAR"
        for block in selected_backbone + selected_finish
    )
    budget_hit_reasons = Counter(
        str(summary["rejectionReason"])
        for summary in source_summary_projection
        if summary["rejectionReason"]
    )
    selected_hot_source_tracks = sorted(
        source_track
        for source_track in selected_all_source_tracks
        if _phase1_is_hot_source_role(_phase1_source_chain_role(source_track=source_track))
    )
    selected_storage_source_tracks = sorted(
        source_track
        for source_track in selected_all_source_tracks
        if _phase1_is_storage_source_role(_phase1_source_chain_role(source_track=source_track))
    )
    mixed_buffer_track_count = sum(
        1
        for track in JI_BUFFER_TRACKS
        if len({
            block.source_track
            for block in selected_buffer_blocks
            if buffer_assignment.get(block.vehicle_nos[0]) == track
        }) >= 2
    )
    return {
        "selectedPackageIds": [block.block_id for block in selected_backbone + selected_finish],
        "selectedChainIds": sorted({f"CHAIN::{block.source_track}" for block in selected_backbone + selected_finish}),
        "selectedPackageCount": len({block.block_id for block in selected_backbone + selected_finish}),
        "selectedPackageSourceTracks": selected_all_source_tracks,
        "selectedVehicleNos": sorted(selected_buffer_vehicle_nos),
        "selectedVehicleCount": len(selected_buffer_vehicle_nos),
        "depotInsertSlotLimit": backbone_plan.depot_slot_limit,
        "depotInsertSlotLimited": backbone_plan.depot_slot_limited,
        "selectedRepairDepotInsertVehicleCount": sum(
            1
            for vehicle_no in selected_buffer_vehicle_nos
            if facts_by_vehicle[vehicle_no].final_family in {"修1", "修2", "修3", "修4"}
        ),
        "selectedLocalVehicleNos": sorted(selected_finish_vehicle_nos - selected_buffer_vehicle_nos),
        "selectedLocalVehicleCount": len(selected_finish_vehicle_nos - selected_buffer_vehicle_nos),
        "activeVehicleCount": len(selected_buffer_vehicle_nos) + len(selected_finish_vehicle_nos - selected_buffer_vehicle_nos),
        "backboneVehicleCount": len(selected_buffer_vehicle_nos),
        "cleanupVehicleCount": len(selected_finish_vehicle_nos - selected_buffer_vehicle_nos),
        "selectedSourceTrackCount": len(selected_all_source_tracks),
        "selectedSourceTracks": selected_all_source_tracks,
        "admittedSourceTracks": selected_all_source_tracks,
        "primaryBackboneSourceTracks": sorted({block.source_track for block in selected_backbone if block.uses_buffer}),
        "elasticBackboneSourceTracks": [],
        "companionBackboneSourceTracks": [],
        "deferredBackboneSourceTracks": sorted(
            plan.source_track
            for plan in source_plans
            if plan.source_track not in {block.source_track for block in selected_backbone if block.uses_buffer}
            and plan.reachable_depot_vehicle_nos
        ),
        "selectedHotSourceTrackCount": len(selected_hot_source_tracks),
        "selectedHotSourceTracks": selected_hot_source_tracks,
        "selectedStorageSourceTrackCount": len(selected_storage_source_tracks),
        "selectedStorageSourceTracks": selected_storage_source_tracks,
        "optionalCleanupPackageCount": 0,
        "selectedExecutionLayerCounts": dict(sorted(selected_layer_counts.items())),
        "selectedCleanupBySource": dict(sorted(selected_cleanup_by_source.items())),
        "selectedTempTracksBySource": {
            source_track: sorted(dict.fromkeys(tracks))
            for source_track, tracks in sorted(selected_temp_tracks_by_source.items())
        },
        "budgetHitReasons": dict(sorted(budget_hit_reasons.items())),
        "sourceOpenSummaries": source_summary_projection,
        "selectedTotalLengthM": round(sum(facts.vehicle_length for facts in facts_list if facts.vehicle_no in selected_buffer_vehicle_nos), 1),
        "depotDemandVehicleCount": len(reachable_depot_set),
        "depotDemandTotalLengthM": round(
            sum(facts.vehicle_length for facts in facts_list if facts.vehicle_no in reachable_depot_set),
            1,
        ),
        "depotCompiledVehicleCount": len(depot_compiled_vehicle_nos),
        "depotCompiledVehicleNos": depot_compiled_vehicle_nos,
        "depotCompileRatio": round(
            len(depot_compiled_vehicle_nos) / len(reachable_depot_set),
            4,
        ) if reachable_depot_set else 1.0,
        "uncompiledDepotVehicleNos": sorted(reachable_depot_set - set(buffer_assignment)),
        "cun4VehicleCount": len(cun4_vehicle_nos),
        "cun4ClearedVehicleCount": len(cun4_cleared_vehicle_nos),
        "cun4ClearedVehicleNos": cun4_cleared_vehicle_nos,
        "cun4ClearRatio": round(len(cun4_cleared_vehicle_nos) / len(cun4_vehicle_nos), 4) if cun4_vehicle_nos else 1.0,
        "remainingCun4VehicleNos": sorted(cun4_vehicle_nos - set(cun4_cleared_vehicle_nos)),
        "jiNonDepotVehicleCount": len(ji_non_depot_vehicle_nos),
        "jiNonDepotClearedVehicleCount": len(ji_cleared_vehicle_nos),
        "jiNonDepotClearedVehicleNos": ji_cleared_vehicle_nos,
        "jiPurityRatio": round(len(ji_cleared_vehicle_nos) / len(ji_non_depot_vehicle_nos), 4) if ji_non_depot_vehicle_nos else 1.0,
        "remainingJiNonDepotVehicleNos": sorted(ji_non_depot_vehicle_nos - set(ji_cleared_vehicle_nos)),
        "jiCapacityTotalM": round(
            sum(min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track]) for track in JI_BUFFER_TRACKS),
            1,
        ),
        "jiOverflowM": round(
            max(
                0.0,
                sum(facts.vehicle_length for facts in facts_list if facts.vehicle_no in reachable_depot_set)
                - sum(min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track]) for track in JI_BUFFER_TRACKS),
            ),
            1,
        ),
        "bufferRequiredLengthsM": {
            track: round(
                sum(
                    _phase1_buffer_required_length(block)
                    for block in selected_buffer_blocks
                    if buffer_assignment.get(block.vehicle_nos[0]) == track
                ),
                1,
            )
            for track in JI_BUFFER_TRACKS
            if any(buffer_assignment.get(block.vehicle_nos[0]) == track for block in selected_buffer_blocks)
        },
        "bufferSourceTracks": {
            track: sorted({
                block.source_track
                for block in selected_buffer_blocks
                if buffer_assignment.get(block.vehicle_nos[0]) == track
            })
            for track in JI_BUFFER_TRACKS
            if any(buffer_assignment.get(block.vehicle_nos[0]) == track for block in selected_buffer_blocks)
        },
        "mixedBufferTrackCount": mixed_buffer_track_count,
        "cun4Cleared": not cun4_pending,
        "cun4PendingVehicleNos": sorted(cun4_pending),
        "reachableDepotVehicleCount": len(reachable_depot_set),
        "bufferedDepotVehicleCount": len(selected_buffer_vehicle_nos),
        "depotCompileRatioAll": round(
            len(selected_buffer_vehicle_nos) / len(all_depot_candidates),
            4,
        ) if all_depot_candidates else 1.0,
        "depotCompileRatioReachable": round(
            len(selected_buffer_vehicle_nos) / len(reachable_depot_set),
            4,
        ) if reachable_depot_set else 1.0,
        "nonDepotRegionCompletion": region_completion,
        "selectedBackboneBlockIds": list(backbone_plan.selected_block_ids),
        "selectedFinishBlockIds": list(finish_plan.selected_block_ids),
        "selectedSourceTracks": selected_all_source_tracks,
        "unselectedBridgeBlocks": unselected_bridge_blocks,
        "bufferBudgetUsage": {
            "reserved": dict(backbone_plan.reserved_buffer_by_track),
            "used": {
                track: round(
                    sum(
                        _phase1_buffer_required_length(block)
                        for block in selected_buffer_blocks
                        if buffer_assignment.get(block.vehicle_nos[0]) == track
                    ),
                    1,
                )
                for track in JI_BUFFER_TRACKS
            },
            "residual": {
                track: round(
                    backbone_plan.reserved_buffer_by_track[track]
                    - sum(
                        _phase1_buffer_required_length(block)
                        for block in selected_buffer_blocks
                        if buffer_assignment.get(block.vehicle_nos[0]) == track
                    ),
                    1,
                )
                for track in JI_BUFFER_TRACKS
            },
        },
        "phase3AlignmentPenalty": _estimate_phase3_branch_penalty_from_blocks(
            selected_buffer_blocks=selected_buffer_blocks,
            buffer_assignment=buffer_assignment,
        ),
        "layoutTemplateName": backbone_plan.layout_template_name,
        "openedBufferTrackCount": len(backbone_plan.opened_buffer_tracks),
        "openedBufferTracks": list(backbone_plan.opened_buffer_tracks),
        "mainBufferTrackCount": sum(1 for track in backbone_plan.opened_buffer_tracks if PHASE1_BUFFER_TRACK_ROLES.get(track) == "main"),
        "supportBufferTrackCount": sum(1 for track in backbone_plan.opened_buffer_tracks if PHASE1_BUFFER_TRACK_ROLES.get(track) != "main"),
        "sourceSplitCount": sum(
            max(0, len({
                buffer_assignment.get(block.vehicle_nos[0])
                for block in selected_buffer_blocks
                if block.source_track == source_track and buffer_assignment.get(block.vehicle_nos[0]) is not None
            }) - 1)
            for source_track in {block.source_track for block in selected_buffer_blocks}
        ),
        "familySplitCount": sum(
            max(0, len({
                buffer_assignment.get(block.vehicle_nos[0])
                for block in selected_buffer_blocks
                if block.final_family == family and buffer_assignment.get(block.vehicle_nos[0]) is not None
            }) - 1)
            for family in {block.final_family for block in selected_buffer_blocks}
        ),
        "spillBlockCount": sum(1 for block in selected_buffer_blocks if block.layout_role == "spill"),
        "highConflictTrackUsageCount": sum(1 for track in backbone_plan.opened_buffer_tracks if track in {"机北1", "机北3"}),
        "deferredVehicleNos": sorted(deferred_vehicle_nos),
        "hiddenVehicleNos": hidden_vehicle_nos,
        "releasedVehicleNos": released_vehicle_nos,
        "selectedOpeningPlanIds": sorted(opened_source_tracks),
        "openingReleasedVehicleCount": len(released_vehicle_nos),
        "primaryPackageIds": [
            block.block_id
            for block in selected_backbone
            if block.block_type in {"clear_cun4", "bridge_to_depot", "depot_batch"}
        ],
        "targetRankByVehicle": dict(sorted(target_rank_by_vehicle.items())),
        "bufferAssignments": dict(sorted(buffer_assignment.items())),
        "bufferBlockOrder": dict(buffer_block_order),
        "bufferCapacityM": {
            track: round(min(float(master.tracks[track].effective_length_m), PHASE1_USABLE_BUFFER_CAPACITY_M[track]), 1)
            for track in JI_BUFFER_TRACKS
        },
        "localFinishPlanCount": len(local_finish_plans),
        "localFinishPlans": local_finish_plans,
        "trackFacts": source_plan_projection,
        "sourceTrackPlans": source_plan_projection,
        "phase1Blocks": block_projection,
        "phase1MacroTasks": [
            {
                "taskId": task.task_id,
                "sourceTrack": task.source_track,
                "sourceRole": task.source_role,
                "waveChunks": [
                    {
                        "waveRole": wave_role,
                        "blockIds": list(block_ids),
                    }
                    for wave_role, block_ids in task.wave_chunks
                ],
                "blockIds": list(task.block_ids),
                "requiredPredecessorIds": list(task.required_predecessor_ids),
                "vehicleNos": list(task.vehicle_nos),
                "bufferVehicleCount": task.buffer_vehicle_count,
                "cleanupVehicleCount": task.cleanup_vehicle_count,
                "releasedDepotVehicleCount": task.released_depot_vehicle_count,
                "releasedFinishVehicleCount": task.released_finish_vehicle_count,
                "pressureGain": task.pressure_gain,
                "topologyZones": list(task.topology_zones),
                "throatGroups": list(task.throat_groups),
                "scoreKey": list(task.score_key),
            }
            for task in macro_tasks
        ],
        "taskPackages": task_package_projection,
        "packageEdges": package_edges,
    }


def _estimate_phase3_branch_penalty_from_blocks(
    *,
    selected_buffer_blocks: list[Phase1Block],
    buffer_assignment: dict[str, str],
) -> int:
    family_tracks: dict[str, set[str]] = defaultdict(set)
    for block in selected_buffer_blocks:
        target_track = buffer_assignment.get(block.vehicle_nos[0])
        if target_track is not None:
            family_tracks[block.final_family].add(target_track)
    return sum(max(0, len(tracks) - 1) for tracks in family_tracks.values())


def _phase1_block_pressure_cut(block: dict[str, Any]) -> str:
    if block["blockType"] == "prefix_clear" and int(block.get("releasedDepotVehicleCount") or 0) > 0:
        return "opening_release_to_ji"
    source_track = str(block["sourceTrack"])
    if source_track in WASH_CONFLICT_TRACKS:
        return "wash_to_ji"
    if source_track in {"调棚", "预修", "抛", "调北"}:
        return "work_to_ji"
    return "storage_to_ji"


def _phase1_legacy_source_role(plan: SourceTrackPlan) -> str:
    if any(block.block_type == "clear_cun4" for block in plan.blocks):
        return "clearance"
    if plan.source_track in WASH_CONFLICT_TRACKS:
        return "wash_backbone"
    if plan.source_track in {"调棚", "预修", "抛", "调北"}:
        return "work_backbone"
    if any(block.uses_buffer for block in plan.blocks):
        return "storage_backbone"
    return "local_finish"


def _solve_phase1_structure_plan(
    *,
    facts_list: list[VehicleStageFacts],
    candidate_facts: tuple[VehicleStageFacts, ...],
    hidden_vehicle_nos: frozenset[str],
    local_finish_plans: tuple[Phase1LocalFinishPlan, ...],
    opening_plans: tuple[Phase1OpeningPlan, ...],
) -> Phase1StructurePlan:
    track_facts = _build_phase1_track_facts(
        facts_list=facts_list,
        candidate_facts=candidate_facts,
        hidden_vehicle_nos=hidden_vehicle_nos,
        local_finish_plans=local_finish_plans,
        opening_plans=opening_plans,
    )
    return Phase1StructurePlan(
        track_facts=tuple(track_facts),
        package_source_tracks=_select_phase1_package_source_tracks(track_facts),
    )


def _build_phase1_track_facts(
    *,
    facts_list: list[VehicleStageFacts],
    candidate_facts: tuple[VehicleStageFacts, ...],
    hidden_vehicle_nos: frozenset[str],
    local_finish_plans: tuple[Phase1LocalFinishPlan, ...],
    opening_plans: tuple[Phase1OpeningPlan, ...],
) -> list[Phase1TrackFacts]:
    candidate_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in candidate_facts:
        candidate_by_track[facts.current_track].append(facts)
    local_finish_by_track: Counter[str] = Counter(plan.source_track for plan in local_finish_plans)
    opening_release_by_track: Counter[str] = Counter(
        {plan.source_track: len(plan.released_vehicle_nos) for plan in opening_plans}
    )
    hidden_by_track: Counter[str] = Counter()
    all_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    for vehicle_no in hidden_vehicle_nos:
        facts = all_by_vehicle.get(vehicle_no)
        if facts is not None:
            hidden_by_track[facts.current_track] += 1
    track_names = sorted(
        set(candidate_by_track)
        | set(local_finish_by_track)
        | set(opening_release_by_track)
        | set(hidden_by_track)
    )
    result: list[Phase1TrackFacts] = []
    for track in track_names:
        track_candidates = sorted(candidate_by_track.get(track, []), key=lambda item: (item.current_order, item.vehicle_no))
        candidate_count = len(track_candidates)
        candidate_length_m = round(sum(item.vehicle_length for item in track_candidates), 1)
        must_clear_count = sum(1 for item in track_candidates if _phase1_must_move(item))
        bottleneck_tags: list[str] = []
        if track in WASH_CONFLICT_TRACKS:
            bottleneck_tags.append("wash_gate")
        if track in {"调棚", "预修"}:
            bottleneck_tags.append("work_gate")
        if track in {"存1", "存2", "存3", "抛", "调北"}:
            bottleneck_tags.append("backbone_storage")
        if track in {"存4北", "存4南"}:
            bottleneck_tags.append("cun4_clear")
        if opening_release_by_track[track] > 0:
            source_role = "clearance"
        elif track in WASH_CONFLICT_TRACKS:
            source_role = "wash_backbone"
        elif track in {"调棚", "预修"}:
            source_role = "work_backbone"
        elif candidate_count > 0:
            source_role = "storage_backbone"
        elif local_finish_by_track[track] > 0:
            source_role = "local_finish"
        else:
            source_role = "defer"
        source_priority_score = (
            0 if opening_release_by_track[track] > 0 else 1,
            0 if track in WASH_CONFLICT_TRACKS else 1,
            0 if source_role in {"work_backbone", "storage_backbone"} else 1,
            -candidate_count,
            -int(round(candidate_length_m)),
            -opening_release_by_track[track],
            -hidden_by_track[track],
            _phase1_local_source_priority(track),
        )
        result.append(
            Phase1TrackFacts(
                source_track=track,
                candidate_vehicle_nos=tuple(item.vehicle_no for item in track_candidates),
                candidate_count=candidate_count,
                candidate_length_m=candidate_length_m,
                must_clear_count=must_clear_count,
                local_finish_count=local_finish_by_track[track],
                opening_release_count=opening_release_by_track[track],
                hidden_candidate_count=hidden_by_track[track],
                bottleneck_tags=tuple(bottleneck_tags),
                source_role=source_role,
                source_priority_score=source_priority_score,
            )
        )
    result.sort(key=lambda item: (item.source_priority_score, item.source_track))
    return result


def _select_phase1_package_source_tracks(
    track_facts: list[Phase1TrackFacts],
) -> tuple[str, ...]:
    ordered = sorted(track_facts, key=lambda item: (item.source_priority_score, item.source_track))
    selected: list[str] = []
    for facts in ordered:
        if (
            facts.candidate_count <= 0
            and facts.local_finish_count <= 0
            and facts.hidden_candidate_count <= 0
            and facts.opening_release_count <= 0
        ):
            continue
        selected.append(facts.source_track)
    return tuple(selected)




def _build_phase1_task_packages(
    *,
    facts_list: list[VehicleStageFacts],
    candidate_facts: tuple[VehicleStageFacts, ...],
    hidden_vehicle_nos: frozenset[str],
    local_finish_plans: tuple[Phase1LocalFinishPlan, ...],
    opening_plans: tuple[Phase1OpeningPlan, ...],
    selected_opening_plans: tuple[Phase1OpeningPlan, ...],
    track_facts: tuple[Phase1TrackFacts, ...],
    package_source_tracks: tuple[str, ...],
) -> list[Phase1TaskPackage]:
    del candidate_facts
    selected_opening_by_track = {
        plan.source_track: plan
        for plan in selected_opening_plans
    }
    track_facts_by_source = {
        facts.source_track: facts
        for facts in track_facts
    }
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        facts_by_track[facts.current_track].append(facts)
    unit_index = 1
    units: list[Phase1TaskPackage] = []
    for source_track in package_source_tracks:
        ordered_members = sorted(
            facts_by_track.get(source_track, []),
            key=lambda item: (item.current_order, item.vehicle_no),
        )
        if not ordered_members:
            continue
        source_track_facts = track_facts_by_source.get(source_track)
        source_role = _phase1_source_role_for_track(
            source_track=source_track,
            candidate_members=[item for item in ordered_members if _is_phase1_candidate(item)],
            local_finish_plans=local_finish_plans,
            opening_plans=opening_plans,
            package_source_tracks=package_source_tracks,
        )
        chain_id = f"C_{source_track}"
        chunk: list[VehicleStageFacts] = []
        chunk_signature: tuple[Any, ...] | None = None
        chunk_length = 0.0
        segment_index = 0
        for facts in ordered_members:
            package_kind = _phase1_package_kind(facts)
            if package_kind is None:
                if chunk:
                    segment_index += 1
                    units.append(
                        _make_phase1_task_package(
                            unit_id=f"U{unit_index:03d}",
                            chain_id=chain_id,
                            members=chunk,
                            package_kind=chunk_signature[0],
                            source_role=source_role,
                            selected_opening_plan=selected_opening_by_track.get(source_track),
                            source_track_facts=source_track_facts,
                            hidden_vehicle_nos=hidden_vehicle_nos,
                            is_selected_package_source=source_track in package_source_tracks,
                            segment_index=segment_index,
                            future_members=ordered_members,
                        )
                    )
                    unit_index += 1
                    chunk = []
                    chunk_length = 0.0
                    chunk_signature = None
                continue
            signature = _phase1_package_signature(
                facts,
                package_kind=package_kind,
                source_role=source_role,
            )
            length_limit = (
                PHASE1_LOCAL_FINISH_SEGMENT_LENGTH_M
                if package_kind == "local_clear" and source_track in {"存5北", "存5南", "存3", "存2", "存1"}
                else PHASE1_UNIT_MAX_LENGTH_M
            )
            projected_length = chunk_length + facts.vehicle_length
            should_split = bool(
                chunk
                and (
                    signature != chunk_signature
                    or projected_length > length_limit
                )
            )
            if should_split:
                segment_index += 1
                units.append(
                    _make_phase1_task_package(
                        unit_id=f"U{unit_index:03d}",
                        chain_id=chain_id,
                        members=chunk,
                        package_kind=chunk_signature[0],
                        source_role=source_role,
                        selected_opening_plan=selected_opening_by_track.get(source_track),
                        source_track_facts=source_track_facts,
                        hidden_vehicle_nos=hidden_vehicle_nos,
                        is_selected_package_source=source_track in package_source_tracks,
                        segment_index=segment_index,
                        future_members=ordered_members,
                    )
                )
                unit_index += 1
                chunk = []
                chunk_length = 0.0
                chunk_signature = None
            chunk.append(facts)
            chunk_length += facts.vehicle_length
            chunk_signature = signature
        if chunk:
            segment_index += 1
            units.append(
                _make_phase1_task_package(
                    unit_id=f"U{unit_index:03d}",
                    chain_id=chain_id,
                    members=chunk,
                    package_kind=chunk_signature[0],
                    source_role=source_role,
                    selected_opening_plan=selected_opening_by_track.get(source_track),
                    source_track_facts=source_track_facts,
                    hidden_vehicle_nos=hidden_vehicle_nos,
                    is_selected_package_source=source_track in package_source_tracks,
                    segment_index=segment_index,
                    future_members=ordered_members,
                )
            )
            unit_index += 1
    units.sort(key=_phase1_task_package_order_key)
    return units


def _phase1_source_role_for_track(
    *,
    source_track: str,
    candidate_members: list[VehicleStageFacts],
    local_finish_plans: tuple[Phase1LocalFinishPlan, ...],
    opening_plans: tuple[Phase1OpeningPlan, ...],
    package_source_tracks: tuple[str, ...],
) -> str:
    if candidate_members and all(item.needs_depot_batch for item in candidate_members):
        if source_track in WASH_CONFLICT_TRACKS or source_track in {"调棚", "预修", "抛", "调北"}:
            return "main_backbone"
        if source_track in package_source_tracks:
            return "storage_support"
    if source_track in WASH_CONFLICT_TRACKS or source_track in {"调棚", "预修", "抛", "调北"}:
        return "main_backbone"
    if source_track in package_source_tracks and candidate_members:
        return "storage_support"
    if any(plan.source_track == source_track for plan in local_finish_plans):
        return "clearance"
    if candidate_members:
        return "storage_support"
    return "defer"


def _phase1_package_kind(facts: VehicleStageFacts) -> str | None:
    if _phase1_local_finish_target_track(facts) is not None:
        return "local_clear"
    if _is_phase1_candidate(facts):
        return "depot_batch"
    return None


def _phase1_package_signature(
    facts: VehicleStageFacts,
    *,
    package_kind: str,
    source_role: str,
) -> tuple[Any, ...]:
    if package_kind == "local_clear":
        return (
            "local_clear",
            _phase1_local_finish_target_track(facts),
            _phase1_local_finish_kind(facts),
            facts.current_track,
        )
    return (
        "depot_batch",
        source_role,
        facts.final_family,
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
    )


def _make_phase1_task_package(
    *,
    unit_id: str,
    chain_id: str,
    members: list[VehicleStageFacts],
    package_kind: str,
    source_role: str,
    selected_opening_plan: Phase1OpeningPlan | None,
    source_track_facts: Phase1TrackFacts | None,
    hidden_vehicle_nos: frozenset[str],
    is_selected_package_source: bool,
    segment_index: int,
    future_members: list[VehicleStageFacts],
) -> Phase1TaskPackage:
    source_track = members[0].current_track
    total_length_m = round(sum(item.vehicle_length for item in members), 1)
    stage_target_track = (
        _phase1_local_finish_target_track(members[0])
        if package_kind == "local_clear"
        else ""
    )
    stage_target_source = (
        _phase1_stage_target_source_for_local_clear(members[0])
        if package_kind == "local_clear"
        else "PHASE1_BACKBONE_PLACE"
    )
    uses_buffer = package_kind == "depot_batch"
    final_family = members[0].final_family if uses_buffer else (stage_target_track or members[0].final_family)
    repair_profile = tuple(sorted({item.repair_process for item in members}))
    min_spot_priority = min(_final_spot_priority(item.final_target_spot) for item in members)
    dependency_tags: list[str] = []
    release_gain = 0
    released_candidate_gain = 0
    if selected_opening_plan is not None:
        dependency_tags.append("opening_released")
        release_gain += len(selected_opening_plan.released_vehicle_nos)
    max_member_order = max(item.current_order for item in members)
    future_relevant_members = [
        item
        for item in future_members
        if item.current_order > max_member_order and _phase1_package_kind(item) is not None
    ]
    future_buffer_members = [
        item for item in future_relevant_members
        if _phase1_package_kind(item) == "depot_batch"
    ]
    if uses_buffer and segment_index == 1 and source_track_facts is not None:
        released_candidate_gain += source_track_facts.hidden_candidate_count
    if uses_buffer:
        released_candidate_gain += len(future_relevant_members)
        if released_candidate_gain > 0:
            dependency_tags.append("bridge_release")
    if package_kind == "local_clear":
        released_candidate_gain += len(future_buffer_members)
        if released_candidate_gain > 0:
            dependency_tags.append("prefix_release")
    if any(item.vehicle_no in hidden_vehicle_nos for item in members):
        dependency_tags.append("hidden_candidate")
    if source_track in WASH_CONFLICT_TRACKS:
        dependency_tags.append("wash_gate")
    elif source_track in {"调棚", "预修", "抛", "调北"}:
        dependency_tags.append("work_gate")
    elif source_track in STORAGE_TRACKS:
        dependency_tags.append("storage_support")
    if is_selected_package_source:
        dependency_tags.append("selected_package_source")
    unit_type = "clearance" if package_kind == "local_clear" else (
        source_role if source_role in {"main_backbone", "storage_support", "clearance"} else "storage_support"
    )
    topology_risk = 0 if source_track in WASH_CONFLICT_TRACKS else (1 if source_track in {"调棚", "预修", "抛", "调北"} else 2)
    segment_role = "chain_start" if segment_index == 1 else "chain_extension"
    if package_kind == "local_clear":
        segment_class = "prefix_clear" if future_buffer_members else "tail_clear"
    else:
        segment_class = "bridge_to_depot" if future_relevant_members else "depot_batch"
    return Phase1TaskPackage(
        unit_id=unit_id,
        chain_id=chain_id,
        unit_type=unit_type,
        source_track=source_track,
        vehicle_nos=tuple(item.vehicle_no for item in members),
        stage_target_track=stage_target_track,
        stage_target_source=stage_target_source,
        uses_buffer=uses_buffer,
        total_length_m=total_length_m,
        final_family=final_family,
        repair_process_profile=repair_profile,
        min_spot_priority=min_spot_priority,
        source_order_start=min(item.current_order for item in members),
        source_order_end=max(item.current_order for item in members),
        entry_order_key=(
            _phase1_unit_type_priority(unit_type),
            _depot_topology_entry_priority(final_family),
            min_spot_priority,
            _depot_repair_process_priority(members[0].repair_process),
            members[0].current_order,
        ),
        dependency_tags=tuple(dependency_tags),
        release_gain=release_gain,
        completion_gain=len(members),
        released_candidate_gain=released_candidate_gain,
        topology_risk=topology_risk,
        buffer_preference=(
            _buffer_track_preference_for_unit_type(
                unit_type=unit_type,
                final_family=final_family,
                source_track=source_track,
            )
            if uses_buffer
            else tuple()
        ),
        is_mandatory_clearance=(unit_type == "clearance"),
        segment_index=segment_index,
        segment_role=segment_role,
        segment_class=segment_class,
    )


def _phase1_unit_type_priority(unit_type: str) -> int:
    return {
        "clearance": 0,
        "main_backbone": 1,
        "storage_support": 2,
    }.get(unit_type, 9)


def _buffer_track_preference_for_unit_type(
    *,
    unit_type: str,
    final_family: str,
    source_track: str,
) -> tuple[str, ...]:
    if unit_type == "clearance":
        return ("机棚", "机南", "机北2", "机北3")
    if source_track in WASH_CONFLICT_TRACKS:
        if final_family == "轮":
            return ("机北3", "机北2", "机棚", "机南")
        return ("机北3", "机北2", "机南", "机棚")
    if source_track in {"调棚", "预修", "抛", "调北"}:
        if final_family in {"修1", "修2"}:
            return ("机南", "机棚", "机北2", "机北3")
        return ("机南", "机北3", "机北2", "机棚")
    if final_family == "轮":
        return ("机北3", "机北2", "机棚", "机南")
    if final_family in {"修1", "修2"}:
        return ("机棚", "机南", "机北2", "机北3")
    return ("机南", "机北3", "机北2", "机棚")


def _build_phase1_unit_dependencies(
    units: list[Phase1TaskPackage],
) -> list[Phase1PackageEdge]:
    edges: list[Phase1PackageEdge] = []
    by_source: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    clearance_units = [unit for unit in units if unit.unit_type == "clearance"]
    for unit in units:
        by_source[unit.source_track].append(unit)
    for items in by_source.values():
        ordered = sorted(items, key=lambda item: (item.source_order_start, item.source_order_end, item.unit_id))
        for left, right in zip(ordered, ordered[1:]):
            edges.append(
                Phase1PackageEdge(
                    from_unit_id=left.unit_id,
                    to_unit_id=right.unit_id,
                    reason="same_source_contiguous",
                )
            )
    for clearance_unit in clearance_units:
        if "opening_released" not in clearance_unit.dependency_tags:
            continue
        for unit in units:
            if unit.source_track == clearance_unit.source_track and unit.unit_id != clearance_unit.unit_id:
                edges.append(
                    Phase1PackageEdge(
                        from_unit_id=clearance_unit.unit_id,
                        to_unit_id=unit.unit_id,
                        reason="clearance_before_release",
                    )
                )
    for family, family_units in _group_units_by_family(units).items():
        ordered = sorted(
            family_units,
            key=lambda item: (item.min_spot_priority, item.source_order_start, item.unit_id),
        )
        for left, right in zip(ordered, ordered[1:]):
            if left.source_track == right.source_track:
                continue
            if left.unit_type != right.unit_type:
                continue
            if left.min_spot_priority >= 999 or right.min_spot_priority >= 999:
                continue
            if left.min_spot_priority == right.min_spot_priority:
                continue
            edges.append(
                Phase1PackageEdge(
                    from_unit_id=left.unit_id,
                    to_unit_id=right.unit_id,
                    reason="inner_before_outer",
                )
            )
    return _dedupe_phase1_package_edges(edges)


def _group_units_by_family(
    units: list[Phase1TaskPackage],
) -> dict[str, list[Phase1TaskPackage]]:
    result: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    for unit in units:
        result[unit.final_family].append(unit)
    return result


def _dedupe_phase1_package_edges(
    edges: list[Phase1PackageEdge],
) -> list[Phase1PackageEdge]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Phase1PackageEdge] = []
    for edge in edges:
        signature = (edge.from_unit_id, edge.to_unit_id, edge.reason)
        if signature in seen or edge.from_unit_id == edge.to_unit_id:
            continue
        seen.add(signature)
        result.append(edge)
    return result


def _build_phase1_track_chains(
    units: list[Phase1TaskPackage],
) -> list[Phase1TrackChain]:
    units_by_chain: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    for unit in units:
        units_by_chain[unit.chain_id].append(unit)
    chains: list[Phase1TrackChain] = []
    for chain_id, chain_units in sorted(units_by_chain.items()):
        ordered = sorted(chain_units, key=lambda unit: (unit.segment_index, unit.source_order_start, unit.unit_id))
        chains.append(
            Phase1TrackChain(
                chain_id=chain_id,
                source_track=ordered[0].source_track,
                source_role=ordered[0].unit_type,
                packages=tuple(ordered),
                local_vehicle_count=sum(len(unit.vehicle_nos) for unit in ordered if not unit.uses_buffer),
                buffer_vehicle_count=sum(len(unit.vehicle_nos) for unit in ordered if unit.uses_buffer),
                hidden_candidate_count=max(unit.released_candidate_gain for unit in ordered),
                candidate_count=sum(len(unit.vehicle_nos) for unit in ordered),
                candidate_length_m=round(sum(unit.total_length_m for unit in ordered), 1),
            )
        )
    chains.sort(key=_phase1_track_chain_order_key)
    return chains


def _solve_phase1_package_plan(
    *,
    facts_list: list[VehicleStageFacts],
    candidate_facts: tuple[VehicleStageFacts, ...],
    units: tuple[Phase1TaskPackage, ...],
    package_edges: tuple[Phase1PackageEdge, ...],
    goal_overrides: dict[str, tuple[str, str]],
    hidden_vehicle_nos: frozenset[str],
    released_vehicle_nos: frozenset[str],
    blocker_vehicle_count: int,
    master: MasterData,
) -> Phase1PackagePlan:
    selected_packages = _select_phase1_task_packages(
        units=list(units),
        chains=_build_phase1_track_chains(list(units)),
        package_edges=list(package_edges),
        blocker_vehicle_count=blocker_vehicle_count,
        master=master,
    )
    buffer_assignment = _solve_phase1_buffer_layout(
        selected_packages=selected_packages,
        master=master,
    )
    target_rank_by_vehicle = _build_phase1_target_ranks_from_packages(
        selected_packages=selected_packages,
        buffer_assignment=buffer_assignment,
    )
    selected_vehicle_nos = frozenset(
        vehicle_no
        for package in selected_packages
        for vehicle_no in package.vehicle_nos
    )
    deferred_vehicle_nos = frozenset(
        facts.vehicle_no
        for facts in facts_list
        if (
            _is_phase1_candidate(facts)
            and facts.vehicle_no not in selected_vehicle_nos
            and facts.vehicle_no not in released_vehicle_nos
            and facts.vehicle_no not in goal_overrides
        )
    )
    diagnostics = {
        "selectedPackageIds": [package.unit_id for package in selected_packages],
        "packageOrder": [package.unit_id for package in selected_packages],
        "selectedChainIds": sorted({package.chain_id for package in selected_packages}),
        "selectedPackageCount": len(selected_packages),
        "selectedMainBackboneCount": sum(
            1 for package in selected_packages if package.unit_type == "main_backbone"
        ),
        "selectedStorageSupportCount": sum(
            1 for package in selected_packages if package.unit_type == "storage_support"
        ),
        "selectedClearanceCount": sum(1 for package in selected_packages if package.unit_type == "clearance"),
        "phase3BranchEstimate": _estimate_phase3_branch_penalty(selected_packages, buffer_assignment),
        "hiddenVehicleNos": sorted(hidden_vehicle_nos),
    }
    return Phase1PackagePlan(
        selected_packages=tuple(selected_packages),
        deferred_vehicle_nos=deferred_vehicle_nos,
        buffer_assignment=buffer_assignment,
        target_rank_by_vehicle=target_rank_by_vehicle,
        package_order=tuple(package.unit_id for package in selected_packages),
        diagnostics=diagnostics,
    )


def _select_phase1_task_packages(
    *,
    units: list[Phase1TaskPackage],
    chains: list[Phase1TrackChain],
    package_edges: list[Phase1PackageEdge],
    blocker_vehicle_count: int,
    master: MasterData,
) -> list[Phase1TaskPackage]:
    predecessor_map: dict[str, set[str]] = defaultdict(set)
    for edge in package_edges:
        predecessor_map[edge.to_unit_id].add(edge.from_unit_id)
    remaining = {
        track: min(
            float(master.tracks[track].effective_length_m),
            PHASE1_USABLE_BUFFER_CAPACITY_M[track],
        )
        for track in JI_BUFFER_TRACKS
    }
    chain_by_id = {chain.chain_id: chain for chain in chains}
    selected_by_chain: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    started_chain_ids: set[str] = set()
    selected: list[Phase1TaskPackage] = []
    selected_ids: set[str] = set()
    while True:
        extension_candidates: list[Phase1TaskPackage] = []
        start_candidates: list[Phase1TaskPackage] = []
        for chain in chains:
            selected_chain_units = selected_by_chain.get(chain.chain_id, [])
            next_index = len(selected_chain_units)
            if next_index >= len(chain.packages):
                continue
            candidate = chain.packages[next_index]
            if candidate.unit_id in selected_ids:
                continue
            if not predecessor_map.get(candidate.unit_id, set()).issubset(selected_ids):
                continue
            if chain.chain_id in started_chain_ids:
                extension_candidates.append(candidate)
            else:
                start_candidates.append(candidate)
        next_unit = _choose_phase1_next_unit(
            extension_candidates=extension_candidates,
            start_candidates=start_candidates,
            remaining=remaining,
            selected_by_chain=selected_by_chain,
            chain_by_id=chain_by_id,
            blocker_vehicle_count=blocker_vehicle_count,
        )
        if next_unit is None:
            break
        if next_unit.uses_buffer:
            target_track = _choose_buffer_track_for_unit(unit=next_unit, remaining=remaining)
            required = next_unit.total_length_m + max(1.0, len(next_unit.vehicle_nos) - 1)
            if remaining.get(target_track, 0.0) < required:
                break
        selected.append(next_unit)
        selected_ids.add(next_unit.unit_id)
        selected_by_chain[next_unit.chain_id].append(next_unit)
        started_chain_ids.add(next_unit.chain_id)
        if next_unit.uses_buffer:
            remaining[target_track] -= next_unit.total_length_m
    return selected


def _phase1_task_package_order_key(unit: Phase1TaskPackage) -> tuple[Any, ...]:
    return (
        _phase1_segment_class_priority(unit.segment_class),
        0 if unit.segment_role == "chain_extension" else 1,
        _phase1_unit_type_priority(unit.unit_type),
        unit.topology_risk,
        -unit.released_candidate_gain,
        -unit.release_gain,
        -unit.completion_gain,
        _depot_topology_entry_priority(unit.final_family),
        unit.min_spot_priority,
        unit.source_track,
        unit.source_order_start,
        unit.unit_id,
    )


def _phase1_track_chain_order_key(chain: Phase1TrackChain) -> tuple[Any, ...]:
    lead = chain.packages[0]
    return (
        _phase1_segment_class_priority(lead.segment_class),
        _phase1_unit_type_priority(lead.unit_type),
        lead.topology_risk,
        -lead.released_candidate_gain,
        -lead.release_gain,
        -chain.buffer_vehicle_count,
        -chain.local_vehicle_count,
        -chain.candidate_count,
        -int(round(chain.candidate_length_m)),
        chain.source_track,
        lead.source_order_start,
        chain.chain_id,
    )


def _choose_phase1_next_unit(
    *,
    extension_candidates: list[Phase1TaskPackage],
    start_candidates: list[Phase1TaskPackage],
    remaining: dict[str, float],
    selected_by_chain: dict[str, list[Phase1TaskPackage]],
    chain_by_id: dict[str, Phase1TrackChain],
    blocker_vehicle_count: int,
) -> Phase1TaskPackage | None:
    feasible_extensions = [
        unit
        for unit in extension_candidates
        if _phase1_unit_fits_remaining(unit=unit, remaining=remaining)
    ]
    feasible_starts = [
        unit
        for unit in start_candidates
        if _phase1_unit_fits_remaining(unit=unit, remaining=remaining)
    ]
    if feasible_extensions:
        extension = sorted(
            feasible_extensions,
            key=lambda unit: _phase1_extension_order_key(
                unit=unit,
                selected_by_chain=selected_by_chain,
                chain_by_id=chain_by_id,
                blocker_vehicle_count=blocker_vehicle_count,
            ),
        )[0]
        if not feasible_starts:
            return extension
    else:
        extension = None
    if feasible_starts:
        starter = sorted(
            feasible_starts,
            key=lambda unit: _phase1_chain_start_order_key(
                unit=unit,
                chain_by_id=chain_by_id,
                blocker_vehicle_count=blocker_vehicle_count,
            ),
        )[0]
        if extension is None:
            return starter
        if _phase1_should_prefer_extension(
            extension=extension,
            starter=starter,
            chain_by_id=chain_by_id,
        ):
            return extension
        return starter
    return extension


def _phase1_unit_fits_remaining(
    *,
    unit: Phase1TaskPackage,
    remaining: dict[str, float],
) -> bool:
    if not unit.uses_buffer:
        return True
    target_track = _choose_buffer_track_for_unit(unit=unit, remaining=remaining)
    required = unit.total_length_m + max(1.0, len(unit.vehicle_nos) - 1)
    return remaining.get(target_track, 0.0) >= required


def _phase1_chain_start_order_key(
    *,
    unit: Phase1TaskPackage,
    chain_by_id: dict[str, Phase1TrackChain],
    blocker_vehicle_count: int,
) -> tuple[Any, ...]:
    chain = chain_by_id[unit.chain_id]
    return (
        _phase1_segment_class_priority(unit.segment_class),
        0 if unit.source_track in WASH_CONFLICT_TRACKS else 1,
        0 if blocker_vehicle_count > 0 and unit.source_track in WASH_CONFLICT_TRACKS else 1,
        0 if unit.uses_buffer else 1,
        -_phase1_chain_remaining_buffer_vehicle_count(chain, unit.segment_index),
        _phase1_chain_buffer_distance(chain, unit.segment_index),
        _phase1_unit_type_priority(unit.unit_type),
        unit.topology_risk,
        -unit.released_candidate_gain,
        -unit.release_gain,
        -chain.buffer_vehicle_count,
        -chain.local_vehicle_count,
        _depot_topology_entry_priority(unit.final_family),
        unit.min_spot_priority,
        unit.source_track,
        unit.unit_id,
    )


def _phase1_extension_order_key(
    *,
    unit: Phase1TaskPackage,
    selected_by_chain: dict[str, list[Phase1TaskPackage]],
    chain_by_id: dict[str, Phase1TrackChain],
    blocker_vehicle_count: int,
) -> tuple[Any, ...]:
    chain = chain_by_id[unit.chain_id]
    already_selected = selected_by_chain.get(unit.chain_id, [])
    return (
        _phase1_segment_class_priority(unit.segment_class),
        0 if blocker_vehicle_count == 0 else 1,
        0 if unit.source_track in WASH_CONFLICT_TRACKS else 1,
        0 if unit.uses_buffer else 1,
        -len(already_selected),
        -_phase1_chain_remaining_buffer_vehicle_count(chain, unit.segment_index),
        -unit.released_candidate_gain,
        -unit.completion_gain,
        -chain.hidden_candidate_count,
        _depot_topology_entry_priority(unit.final_family),
        unit.min_spot_priority,
        unit.segment_index,
        unit.unit_id,
    )


def _phase1_should_prefer_extension(
    *,
    extension: Phase1TaskPackage,
    starter: Phase1TaskPackage,
    chain_by_id: dict[str, Phase1TrackChain],
) -> bool:
    extension_chain = chain_by_id[extension.chain_id]
    starter_chain = chain_by_id[starter.chain_id]
    extension_score = (
        (12 - _phase1_segment_class_priority(extension.segment_class) * 3)
        + extension.released_candidate_gain * 4
        + extension.completion_gain * 3
        + _phase1_chain_remaining_buffer_vehicle_count(extension_chain, extension.segment_index) * 5
        + extension_chain.hidden_candidate_count * 2
        + max(0, 4 - extension.topology_risk)
    )
    starter_score = (
        (12 - _phase1_segment_class_priority(starter.segment_class) * 3)
        + starter.released_candidate_gain * 3
        + starter.release_gain * 2
        + _phase1_chain_remaining_buffer_vehicle_count(starter_chain, starter.segment_index) * 5
        + starter_chain.candidate_count
    )
    return extension_score >= starter_score


def _phase1_segment_class_priority(segment_class: str) -> int:
    return {
        "bridge_to_depot": 0,
        "depot_batch": 1,
        "prefix_clear": 2,
        "tail_clear": 3,
    }.get(segment_class, 9)


def _phase1_chain_remaining_buffer_vehicle_count(
    chain: Phase1TrackChain,
    segment_index: int,
) -> int:
    return sum(
        len(unit.vehicle_nos)
        for unit in chain.packages
        if unit.segment_index >= segment_index and unit.uses_buffer
    )


def _phase1_chain_buffer_distance(
    chain: Phase1TrackChain,
    segment_index: int,
) -> int:
    for unit in chain.packages:
        if unit.segment_index < segment_index:
            continue
        if unit.uses_buffer:
            return unit.segment_index - segment_index
    return 99


def _choose_buffer_track_for_unit(
    *,
    unit: Phase1TaskPackage,
    remaining: dict[str, float],
) -> str:
    if not unit.buffer_preference:
        return JI_BUFFER_TRACKS[0]
    required = unit.total_length_m + max(1.0, len(unit.vehicle_nos) - 1)
    for track in unit.buffer_preference:
        if remaining.get(track, 0.0) >= required:
            return track
    return max(unit.buffer_preference, key=lambda track: remaining.get(track, 0.0))


def _solve_phase1_buffer_layout(
    *,
    selected_packages: list[Phase1TaskPackage],
    master: MasterData,
) -> dict[str, str]:
    remaining = {
        track: min(
            float(master.tracks[track].effective_length_m),
            PHASE1_USABLE_BUFFER_CAPACITY_M[track],
        )
        for track in JI_BUFFER_TRACKS
    }
    assignment: dict[str, str] = {}
    preferred_track_by_source: dict[str, str] = {}
    buffer_packages = [package for package in selected_packages if package.uses_buffer]
    for package in sorted(buffer_packages, key=_phase1_task_package_order_key):
        target_track = preferred_track_by_source.get(package.source_track)
        required = package.total_length_m + max(1.0, len(package.vehicle_nos) - 1)
        if target_track is None or remaining.get(target_track, 0.0) < required:
            target_track = _choose_buffer_track_for_unit(unit=package, remaining=remaining)
        for vehicle_no in package.vehicle_nos:
            assignment[vehicle_no] = target_track
        remaining[target_track] -= package.total_length_m
        preferred_track_by_source[package.source_track] = target_track
    return assignment


def _build_phase1_target_ranks_from_packages(
    *,
    selected_packages: list[Phase1TaskPackage],
    buffer_assignment: dict[str, str],
) -> dict[str, int]:
    units_by_track: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    for package in selected_packages:
        if not package.uses_buffer:
            continue
        track = buffer_assignment[package.vehicle_nos[0]]
        units_by_track[track].append(package)
    result: dict[str, int] = {}
    for track, units_for_track in units_by_track.items():
        ordered = sorted(
            units_for_track,
            key=lambda package: (
                _depot_topology_entry_priority(package.final_family),
                package.min_spot_priority,
                package.source_order_start,
                package.unit_id,
            ),
        )
        rank = 1
        for package in ordered:
            for offset, vehicle_no in enumerate(package.vehicle_nos):
                result[vehicle_no] = rank + offset
            rank += len(package.vehicle_nos)
    return result


def _estimate_phase3_branch_penalty(
    selected_packages: list[Phase1TaskPackage],
    buffer_assignment: dict[str, str],
) -> int:
    family_tracks: dict[str, set[str]] = defaultdict(set)
    for package in selected_packages:
        if not package.uses_buffer:
            continue
        family_tracks[package.final_family].add(buffer_assignment[package.vehicle_nos[0]])
    return sum(max(0, len(tracks) - 1) for tracks in family_tracks.values())


def _phase1_unit_pressure_cut(unit: Phase1TaskPackage) -> str:
    if "opening_released" in unit.dependency_tags:
        return "opening_release_to_ji"
    if unit.source_track in WASH_CONFLICT_TRACKS:
        return "wash_to_ji"
    if unit.source_track in {"调棚", "预修", "抛", "调北"}:
        return "work_to_ji"
    return "storage_to_ji"


def _build_phase1_track_facts_projection(
    all_task_packages: list[Phase1TaskPackage],
    local_finish_plans: list[Phase1LocalFinishPlan],
    opening_plans: list[Phase1OpeningPlan],
) -> list[dict[str, Any]]:
    units_by_track: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    for unit in all_task_packages:
        units_by_track[unit.source_track].append(unit)
    local_finish_by_track = Counter(plan.source_track for plan in local_finish_plans)
    opening_by_track = Counter(plan.source_track for plan in opening_plans)
    rows: list[dict[str, Any]] = []
    for source_track, items in sorted(units_by_track.items()):
        vehicle_nos = [vehicle_no for unit in items for vehicle_no in unit.vehicle_nos]
        candidate_length = round(sum(unit.total_length_m for unit in items), 1)
        unit_type = items[0].unit_type
        source_role = {
            "main_backbone": "wash_backbone" if source_track in WASH_CONFLICT_TRACKS else "work_backbone",
            "storage_support": "storage_backbone",
            "clearance": "clearance",
        }.get(unit_type, "defer")
        rows.append(
            {
                "sourceTrack": source_track,
                "candidateVehicleNos": vehicle_nos,
                "candidateCount": len(vehicle_nos),
                "candidateLengthM": candidate_length,
                "mustClearCount": sum(1 for unit in items if unit.is_mandatory_clearance),
                "localFinishCount": local_finish_by_track[source_track],
                "openingReleaseCount": opening_by_track[source_track],
                "hiddenCandidateCount": 0,
                "bottleneckTags": list(sorted({tag for unit in items for tag in unit.dependency_tags if tag.endswith("gate")})),
                "sourceRole": source_role,
                "sourcePriorityScore": [],
            }
        )
    return rows


def _build_phase1_chain_projection(
    *,
    all_task_packages: list[Phase1TaskPackage],
    selected_package_ids: set[str],
    buffer_assignment: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Phase1TaskPackage]] = defaultdict(list)
    for package in all_task_packages:
        grouped[package.chain_id].append(package)
    rows: list[dict[str, Any]] = []
    for chain_id, packages in sorted(grouped.items()):
        ordered = sorted(packages, key=lambda unit: (unit.segment_index, unit.unit_id))
        rows.append(
            {
                "chainId": chain_id,
                "sourceTrack": ordered[0].source_track,
                "packageType": ordered[0].unit_type,
                "selectedDepth": sum(1 for unit in ordered if unit.unit_id in selected_package_ids),
                "segmentCount": len(ordered),
                "localVehicleCount": sum(len(unit.vehicle_nos) for unit in ordered if not unit.uses_buffer),
                "bufferVehicleCount": sum(len(unit.vehicle_nos) for unit in ordered if unit.uses_buffer),
                "releasedCandidateGain": sum(unit.released_candidate_gain for unit in ordered),
                "vehicleCount": sum(len(unit.vehicle_nos) for unit in ordered),
                "totalLengthM": round(sum(unit.total_length_m for unit in ordered), 1),
                "segments": [
                    {
                        "packageId": unit.unit_id,
                        "segmentIndex": unit.segment_index,
                        "segmentRole": unit.segment_role,
                        "segmentClass": unit.segment_class,
                        "vehicleNos": list(unit.vehicle_nos),
                        "stageTargetTrack": unit.stage_target_track,
                        "stageTargetSource": unit.stage_target_source,
                        "usesBuffer": unit.uses_buffer,
                        "selected": unit.unit_id in selected_package_ids,
                        "bufferTrack": buffer_assignment.get(unit.vehicle_nos[0]) if unit.unit_id in selected_package_ids else None,
                    }
                    for unit in ordered
                ],
            }
        )
    return rows


def _is_phase1_candidate(facts: VehicleStageFacts) -> bool:
    return _needs_phase1_depot_staging(facts) and facts.current_track not in JI_BUFFER_TRACKS


def _needs_phase1_depot_staging(facts: VehicleStageFacts) -> bool:
    return facts.needs_depot_batch and not facts.is_depot_area_vehicle


def _validate_phase1_staging_contract(
    *,
    facts_list: list[VehicleStageFacts],
    reachable_depot_set: frozenset[str],
    buffer_assignment: dict[str, str],
    goal_overrides: dict[str, tuple[str, str]],
) -> None:
    facts_by_vehicle = {facts.vehicle_no: facts for facts in facts_list}
    missing = sorted(vehicle_no for vehicle_no in reachable_depot_set if vehicle_no not in buffer_assignment)
    if missing:
        raise ValueError(f"phase1 depot staging did not cover vehicles: {missing}")

    invalid_buffered = sorted(
        vehicle_no
        for vehicle_no, target_track in buffer_assignment.items()
        if target_track not in JI_BUFFER_TRACKS
        or not _needs_phase1_depot_staging(facts_by_vehicle[vehicle_no])
    )
    if invalid_buffered:
        raise ValueError(f"phase1 staging buffers contain non-staging vehicles: {invalid_buffered}")

    uncleared_ji = sorted(
        facts.vehicle_no
        for facts in facts_list
        if facts.current_track in JI_BUFFER_TRACKS
        and not _needs_phase1_depot_staging(facts)
        and facts.vehicle_no not in goal_overrides
    )
    if uncleared_ji:
        raise ValueError(f"phase1 JI buffers still contain non-staging vehicles: {uncleared_ji}")


def _build_phase1_accessibility(
    facts_list: list[VehicleStageFacts],
    *,
    locally_released_vehicle_nos: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    candidate_facts: list[VehicleStageFacts] = []
    hidden_vehicle_nos: list[str] = []
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        facts_by_track[facts.current_track].append(facts)
    for track_facts in facts_by_track.values():
        ordered = sorted(track_facts, key=lambda item: (item.current_order, item.vehicle_no))
        blocker_seen = False
        for facts in ordered:
            if facts.vehicle_no in locally_released_vehicle_nos:
                continue
            if not _is_phase1_candidate(facts):
                blocker_seen = True
                continue
            if blocker_seen:
                hidden_vehicle_nos.append(facts.vehicle_no)
                continue
            candidate_facts.append(facts)
    candidate_facts.sort(
        key=lambda item: (
            _pressure_cut_priority(_pressure_cut_name(item)),
            item.current_track,
            item.current_order,
            final_sequence_key(item),
        )
    )
    return {
        "candidate_facts": tuple(candidate_facts),
        "hidden_vehicle_nos": tuple(sorted(hidden_vehicle_nos)),
    }


def _build_phase1_opening_plans(
    *,
    facts_list: list[VehicleStageFacts],
    hidden_vehicle_nos: frozenset[str],
) -> list[Phase1OpeningPlan]:
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    for facts in facts_list:
        facts_by_track[facts.current_track].append(facts)
    plans: list[Phase1OpeningPlan] = []
    plan_index = 1
    for current_track, track_facts in sorted(facts_by_track.items()):
        if current_track in JI_BUFFER_TRACKS or current_track in DEPOT_TARGET_TRACKS:
            continue
        ordered = sorted(track_facts, key=lambda item: (item.current_order, item.vehicle_no))
        prefix_blockers: list[VehicleStageFacts] = []
        release_candidates: list[VehicleStageFacts] = []
        for facts in ordered:
            if facts.vehicle_no in hidden_vehicle_nos and _is_phase1_candidate(facts):
                release_candidates.append(facts)
                continue
            if release_candidates:
                break
            if _is_phase1_opening_blocker(facts):
                prefix_blockers.append(facts)
                continue
            if _is_phase1_candidate(facts):
                break
            prefix_blockers = []
            break
        if not prefix_blockers or not release_candidates:
            continue
        blocker_total_length = round(sum(item.vehicle_length for item in prefix_blockers), 1)
        released_total_length = round(sum(item.vehicle_length for item in release_candidates), 1)
        if len(prefix_blockers) > 3:
            continue
        if blocker_total_length > PHASE1_OPENING_BUDGET_M:
            continue
        if len(release_candidates) < 2 and released_total_length < blocker_total_length + 20.0:
            continue
        if released_total_length <= blocker_total_length:
            continue
        blocker_targets = tuple(
            _phase1_opening_target_track(item)
            for item in prefix_blockers
        )
        if any(track == item.current_track for track, item in zip(blocker_targets, prefix_blockers)):
            continue
        released_families = tuple(
            sorted({item.final_family for item in release_candidates}, key=_final_family_priority)
        )
        plans.append(
            Phase1OpeningPlan(
                plan_id=f"O{plan_index:03d}",
                source_track=current_track,
                blocker_vehicle_nos=tuple(item.vehicle_no for item in prefix_blockers),
                released_vehicle_nos=tuple(item.vehicle_no for item in release_candidates),
                blocker_total_length=blocker_total_length,
                released_total_length=released_total_length,
                released_final_families=released_families,
                blocker_target_tracks=blocker_targets,
            )
        )
        plan_index += 1
    plans.sort(key=_phase1_opening_plan_order_key)
    return plans


def _build_phase1_local_finish_plans(
    facts_list: list[VehicleStageFacts],
    *,
    hidden_vehicle_nos: frozenset[str],
) -> list[Phase1LocalFinishPlan]:
    candidates: list[Phase1LocalFinishPlan] = []
    facts_by_track: dict[str, list[VehicleStageFacts]] = defaultdict(list)
    pending_pressure_by_track: Counter[str] = Counter()
    for facts in facts_list:
        facts_by_track[facts.current_track].append(facts)
        if _phase1_local_finish_target_track(facts) is not None:
            pending_pressure_by_track[facts.current_track] += 1
    plan_index = 1
    for current_track, track_facts in sorted(facts_by_track.items()):
        ordered = sorted(track_facts, key=lambda item: (item.current_order, item.vehicle_no))
        hidden_candidates = [
            facts
            for facts in ordered
            if facts.vehicle_no in hidden_vehicle_nos and _is_phase1_candidate(facts)
        ]
        first_hidden_order = min((facts.current_order for facts in hidden_candidates), default=None)
        cluster_members: list[VehicleStageFacts] = []
        cluster_target: str | None = None
        cluster_kind: str | None = None
        for facts in ordered:
            target_track = _phase1_local_finish_target_track(facts)
            if target_track is None:
                if cluster_members:
                    chunked_members = _split_phase1_local_finish_members(
                        source_track=current_track,
                        members=cluster_members,
                    )
                    for chunk in chunked_members:
                        candidates.append(
                            _make_phase1_local_finish_plan(
                                plan_id=f"L{plan_index:03d}",
                                source_track=current_track,
                                target_track=cluster_target or current_track,
                                members=chunk,
                                source_pending_pressure=pending_pressure_by_track[current_track],
                                cluster_kind=_phase1_local_finish_cluster_kind(
                                    chunk=chunk,
                                    default_kind=cluster_kind or "regional_finish",
                                    first_hidden_order=first_hidden_order,
                                ),
                                released_candidate_gain=_phase1_local_finish_released_candidate_gain(
                                    chunk=chunk,
                                    hidden_candidates=hidden_candidates,
                                ),
                            )
                        )
                        plan_index += 1
                    cluster_members = []
                    cluster_target = None
                    cluster_kind = None
                continue
            candidate_kind = _phase1_local_finish_kind(facts)
            if cluster_members and (
                target_track != cluster_target
                or candidate_kind != cluster_kind
            ):
                chunked_members = _split_phase1_local_finish_members(
                    source_track=current_track,
                    members=cluster_members,
                )
                for chunk in chunked_members:
                    candidates.append(
                        _make_phase1_local_finish_plan(
                            plan_id=f"L{plan_index:03d}",
                            source_track=current_track,
                            target_track=cluster_target or current_track,
                            members=chunk,
                            source_pending_pressure=pending_pressure_by_track[current_track],
                            cluster_kind=_phase1_local_finish_cluster_kind(
                                chunk=chunk,
                                default_kind=cluster_kind or "regional_finish",
                                first_hidden_order=first_hidden_order,
                            ),
                            released_candidate_gain=_phase1_local_finish_released_candidate_gain(
                                chunk=chunk,
                                hidden_candidates=hidden_candidates,
                            ),
                        )
                    )
                    plan_index += 1
                cluster_members = []
                cluster_target = None
                cluster_kind = None
            cluster_members.append(facts)
            cluster_target = target_track
            cluster_kind = candidate_kind
        if cluster_members:
            chunked_members = _split_phase1_local_finish_members(
                source_track=current_track,
                members=cluster_members,
            )
            for chunk in chunked_members:
                candidates.append(
                    _make_phase1_local_finish_plan(
                        plan_id=f"L{plan_index:03d}",
                        source_track=current_track,
                        target_track=cluster_target or current_track,
                        members=chunk,
                        source_pending_pressure=pending_pressure_by_track[current_track],
                        cluster_kind=_phase1_local_finish_cluster_kind(
                            chunk=chunk,
                            default_kind=cluster_kind or "regional_finish",
                            first_hidden_order=first_hidden_order,
                        ),
                        released_candidate_gain=_phase1_local_finish_released_candidate_gain(
                            chunk=chunk,
                            hidden_candidates=hidden_candidates,
                        ),
                    )
                )
                plan_index += 1
    candidates.sort(
        key=lambda plan: (
            _phase1_local_finish_priority(plan),
            plan.source_priority,
            -plan.released_candidate_gain,
            -plan.completion_gain,
            -plan.source_pending_pressure,
            plan.total_length,
            plan.source_track,
            plan.plan_id,
        )
    )
    return candidates


def _phase1_local_finish_target_track(facts: VehicleStageFacts) -> str | None:
    if facts.is_depot_area_vehicle:
        return None
    if facts.current_track in {"存4北", "存4南"} and not facts.is_cun4bei_final:
        if (
            not facts.needs_depot_batch
            and facts.final_target_track not in {"存4北", "存4南"}
            and facts.final_target_track not in DEPOT_TARGET_TRACKS
            and facts.final_target_track not in DEPOT_OUTER_TRACKS
            and facts.final_target_track != facts.current_track
        ):
            return facts.final_target_track
        return None
    if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch:
        if (
            facts.final_target_track not in {"存4北", "存4南"}
            and facts.final_target_track not in DEPOT_TARGET_TRACKS
            and facts.final_target_track not in DEPOT_OUTER_TRACKS
            and facts.final_target_track != facts.current_track
        ):
            return facts.final_target_track
        return None
    if (
        not facts.needs_depot_batch
        and facts.final_target_track not in DEPOT_TARGET_TRACKS
        and facts.final_target_track not in DEPOT_OUTER_TRACKS
    ):
        if facts.is_close_door and facts.final_target_track == "存4北":
            return None
        if facts.final_target_track == facts.current_track:
            return None
        return facts.final_target_track
    return None


def _phase1_stage_target_source_for_local_clear(facts: VehicleStageFacts) -> str:
    if facts.current_track in {"存4北", "存4南"}:
        return "PHASE1_CLEAR_CUN4"
    if facts.current_track in JI_BUFFER_TRACKS and not facts.needs_depot_batch:
        return "PHASE1_CLEAR_JI"
    return "PHASE1_LOCAL_FINISH"


def _make_phase1_local_finish_plan(
    *,
    plan_id: str,
    source_track: str,
    target_track: str,
    members: list[VehicleStageFacts],
    source_pending_pressure: int,
    cluster_kind: str,
    released_candidate_gain: int,
) -> Phase1LocalFinishPlan:
    total_length = round(sum(item.vehicle_length for item in members), 1)
    if source_track in {"存4北", "存4南"}:
        priority_tag = "clear_cun4"
    elif source_track in JI_BUFFER_TRACKS:
        priority_tag = "ji_evict"
    else:
        priority_tag = "regional_finish"
    return Phase1LocalFinishPlan(
        plan_id=plan_id,
        source_track=source_track,
        target_track=target_track,
        vehicle_nos=tuple(item.vehicle_no for item in members),
        total_length=total_length,
        priority_tag=priority_tag,
        cluster_kind=cluster_kind,
        completion_gain=len(members),
        released_candidate_gain=released_candidate_gain,
        source_pending_pressure=source_pending_pressure,
        source_priority=_phase1_local_source_priority(source_track),
    )


def _split_phase1_local_finish_members(
    *,
    source_track: str,
    members: list[VehicleStageFacts],
) -> list[list[VehicleStageFacts]]:
    if source_track not in {"存5北", "存5南", "存3", "存2", "存1"}:
        return [members]
    chunks: list[list[VehicleStageFacts]] = []
    current_chunk: list[VehicleStageFacts] = []
    current_length = 0.0
    for facts in members:
        projected = current_length + facts.vehicle_length
        if current_chunk and projected > PHASE1_LOCAL_FINISH_SEGMENT_LENGTH_M:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0.0
        current_chunk.append(facts)
        current_length += facts.vehicle_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _phase1_local_finish_cluster_kind(
    *,
    chunk: list[VehicleStageFacts],
    default_kind: str,
    first_hidden_order: int | None,
) -> str:
    if first_hidden_order is None:
        return default_kind
    max_order = max(item.current_order for item in chunk)
    return "prefix_release" if max_order < first_hidden_order else default_kind


def _phase1_local_finish_released_candidate_gain(
    *,
    chunk: list[VehicleStageFacts],
    hidden_candidates: list[VehicleStageFacts],
) -> int:
    if not hidden_candidates:
        return 0
    max_order = max(item.current_order for item in chunk)
    return sum(1 for item in hidden_candidates if item.current_order > max_order)


def _select_phase1_local_finish_plans(
    local_finish_plans: list[Phase1LocalFinishPlan],
) -> list[Phase1LocalFinishPlan]:
    selected: list[Phase1LocalFinishPlan] = []
    used_vehicle_nos: set[str] = set()
    used_source_track_counts: Counter[str] = Counter()
    total_length = 0.0
    release_length = 0.0
    primary_pass = [
        plan
        for plan in local_finish_plans
        if _phase1_local_source_priority(plan.source_track) < 5
    ]
    secondary_pass = [
        plan
        for plan in local_finish_plans
        if _phase1_local_source_priority(plan.source_track) >= 5
    ]
    for plan in primary_pass + secondary_pass:
        is_release_plan = plan.released_candidate_gain > 0
        plan_limit = PHASE1_LOCAL_FINISH_MAX_PLANS + (2 if is_release_plan else 0)
        if len(selected) >= plan_limit:
            break
        if any(vehicle_no in used_vehicle_nos for vehicle_no in plan.vehicle_nos):
            continue
        if (
            plan.priority_tag == "regional_finish"
            and used_source_track_counts[plan.source_track] >= _phase1_local_source_plan_limit(plan)
            and plan.cluster_kind != "prefix_release"
        ):
            continue
        if is_release_plan:
            if release_length + plan.total_length > PHASE1_LOCAL_FINISH_BUDGET_M + 36.0:
                continue
        elif total_length + plan.total_length > PHASE1_LOCAL_FINISH_BUDGET_M:
            continue
        if (
            plan.priority_tag == "regional_finish"
            and _phase1_local_source_priority(plan.source_track) >= 5
            and any(
                item.source_track == plan.source_track
                for item in selected
            )
        ):
            continue
        selected.append(plan)
        used_vehicle_nos.update(plan.vehicle_nos)
        used_source_track_counts[plan.source_track] += 1
        total_length += plan.total_length
        if is_release_plan:
            release_length += plan.total_length
    return selected


def _build_phase1_local_finish_goal_overrides(
    local_finish_plans: list[Phase1LocalFinishPlan],
) -> dict[str, tuple[str, str]]:
    overrides: dict[str, tuple[str, str]] = {}
    for plan in local_finish_plans:
        source = "PHASE1_LOCAL_FINISH"
        if plan.priority_tag == "clear_cun4":
            source = "PHASE1_CLEAR_CUN4"
        elif plan.priority_tag == "ji_evict":
            source = "PHASE1_BLOCKING_EVICT"
        for vehicle_no in plan.vehicle_nos:
            overrides[vehicle_no] = (plan.target_track, source)
    return overrides


def _is_phase1_opening_blocker(facts: VehicleStageFacts) -> bool:
    if facts.needs_depot_batch:
        return False
    if facts.is_depot_area_vehicle:
        return False
    return facts.current_track in WASH_CONFLICT_TRACKS or facts.current_track in STORAGE_TRACKS or facts.current_zone == "WORK"


def _phase1_opening_target_track(facts: VehicleStageFacts) -> str:
    if facts.is_cun4bei_final:
        return "存4北"
    if facts.current_track == "存4北":
        return "存4南"
    if facts.current_track == "存4南":
        return "存4北"
    return "存4南"


def _phase1_local_finish_kind(facts: VehicleStageFacts) -> str:
    if facts.current_track in {"存4北", "存4南"}:
        return "cun4_clear"
    if facts.current_track in JI_BUFFER_TRACKS:
        return "ji_evict"
    if facts.current_track in WASH_CONFLICT_TRACKS or facts.current_zone == "WORK":
        return "prefix_release"
    return "regional_finish"


def _phase1_local_finish_priority(plan: Phase1LocalFinishPlan) -> int:
    return {
        "clear_cun4": 0,
        "ji_evict": 1,
        "regional_finish": 2,
    }.get(plan.priority_tag, 9)


def _phase1_local_source_priority(track_name: str) -> int:
    try:
        return PHASE1_LOCAL_HIGH_PRESSURE_SOURCE_TRACKS.index(track_name)
    except ValueError:
        return len(PHASE1_LOCAL_HIGH_PRESSURE_SOURCE_TRACKS) + _phase1_track_priority(track_name)


def _phase1_local_source_plan_limit(plan: Phase1LocalFinishPlan) -> int:
    if (
        plan.source_track in {"存5北", "调棚", "预修"}
        and plan.source_pending_pressure >= 10
    ):
        return 2
    return 1


def _phase1_opening_plan_order_key(plan: Phase1OpeningPlan) -> tuple[Any, ...]:
    gain = plan.released_total_length - plan.blocker_total_length
    family_count = len(plan.released_final_families)
    return (
        -gain,
        family_count,
        plan.blocker_total_length,
        plan.source_track,
        plan.plan_id,
    )


def _select_phase1_opening_plans(
    opening_plans: list[Phase1OpeningPlan],
) -> list[Phase1OpeningPlan]:
    selected: list[Phase1OpeningPlan] = []
    used_source_tracks: set[str] = set()
    used_blockers: set[str] = set()
    total_blocker_length = 0.0
    for plan in opening_plans:
        if len(selected) >= PHASE1_OPENING_MAX_PLANS:
            break
        if plan.source_track in used_source_tracks:
            continue
        if any(vehicle_no in used_blockers for vehicle_no in plan.blocker_vehicle_nos):
            continue
        if total_blocker_length + plan.blocker_total_length > PHASE1_OPENING_BUDGET_M:
            continue
        selected.append(plan)
        used_source_tracks.add(plan.source_track)
        used_blockers.update(plan.blocker_vehicle_nos)
        total_blocker_length += plan.blocker_total_length
    return selected


def _build_phase1_opening_goal_overrides(
    opening_plans: list[Phase1OpeningPlan],
) -> dict[str, tuple[str, str]]:
    overrides: dict[str, tuple[str, str]] = {}
    for plan in opening_plans:
        for vehicle_no, target_track in zip(plan.blocker_vehicle_nos, plan.blocker_target_tracks):
            overrides[vehicle_no] = (target_track, "PHASE1_LOCAL_FINISH")
    return overrides


def _build_phase1_diagnostics(
    *,
    local_finish_plans: list[Phase1LocalFinishPlan],
    opening_plans: list[Phase1OpeningPlan],
    selected_opening_plans: list[Phase1OpeningPlan],
    selected_package_source_tracks: list[str],
    selected_facts: list[VehicleStageFacts],
    facts_list: list[VehicleStageFacts],
    deferred_vehicle_nos: frozenset[str],
    buffer_assignment: dict[str, str],
    target_rank_by_vehicle: dict[str, int],
    hidden_vehicle_nos: frozenset[str],
    released_vehicle_nos: frozenset[str],
    goal_overrides: dict[str, tuple[str, str]],
    master: MasterData,
    task_packages: list[Phase1TaskPackage],
    all_task_packages: list[Phase1TaskPackage],
    package_edges: list[Phase1PackageEdge],
    package_plan_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    task_packages = list(task_packages)
    all_task_packages = list(all_task_packages)
    package_edges = list(package_edges)
    package_plan_diagnostics = dict(package_plan_diagnostics)
    selected_package_ids = {unit.unit_id for unit in task_packages}
    buffer_loads = Counter(buffer_assignment.values())
    buffer_lengths: dict[str, float] = {}
    for facts in selected_facts:
        track = buffer_assignment[facts.vehicle_no]
        buffer_lengths[track] = round(buffer_lengths.get(track, 0.0) + facts.vehicle_length, 1)
    pressure_counts = Counter(_phase1_unit_pressure_cut(unit) for unit in task_packages)
    all_pressure_counts = Counter(_phase1_unit_pressure_cut(unit) for unit in all_task_packages)
    deferred_by_family = Counter(
        unit.final_family
        for unit in all_task_packages
        if unit.unit_id not in selected_package_ids
    )
    selected_opening_plan_ids = {plan.plan_id for plan in selected_opening_plans}
    ordered_blocks_by_buffer: dict[str, list[str]] = {}
    buffer_family_mix: dict[str, dict[str, int]] = {}
    selected_local_vehicle_nos = {
        vehicle_no
        for unit in task_packages
        if not unit.uses_buffer
        for vehicle_no in unit.vehicle_nos
    }
    region_completion = _phase1_non_depot_region_completion(
        facts_list=facts_list,
        selected_vehicle_nos=frozenset(facts.vehicle_no for facts in selected_facts),
        goal_overrides=goal_overrides,
        deferred_vehicle_nos=deferred_vehicle_nos,
    )
    ordered_blocks_by_buffer = defaultdict(list)
    for unit in task_packages:
        track = buffer_assignment.get(unit.vehicle_nos[0])
        if track is None:
            continue
        ordered_blocks_by_buffer[track].append(unit.unit_id)
    for track, unit_ids in list(ordered_blocks_by_buffer.items()):
        units_for_track = [
            unit for unit in task_packages if buffer_assignment.get(unit.vehicle_nos[0]) == track
        ]
        units_for_track.sort(key=_phase1_task_package_order_key)
        ordered_blocks_by_buffer[track] = [unit.unit_id for unit in units_for_track]
        family_counts: Counter[str] = Counter()
        for unit in units_for_track:
            family_counts[unit.final_family] += len(unit.vehicle_nos)
        buffer_family_mix[track] = dict(sorted(family_counts.items()))
    track_facts_projection = _build_phase1_track_facts_projection(
        all_task_packages,
        local_finish_plans,
        opening_plans,
    )
    package_edge_projection = [
        {
            "from": edge.from_unit_id,
            "to": edge.to_unit_id,
            "reason": edge.reason,
        }
        for edge in package_edges
    ]
    task_package_projection = [
        {
            "packageId": unit.unit_id,
            "chainId": unit.chain_id,
            "packageType": unit.unit_type,
            "sourceTrack": unit.source_track,
            "vehicleNos": list(unit.vehicle_nos),
            "stageTargetTrack": unit.stage_target_track,
            "stageTargetSource": unit.stage_target_source,
            "usesBuffer": unit.uses_buffer,
            "totalLengthM": unit.total_length_m,
            "finalFamily": unit.final_family,
            "repairProcessProfile": list(unit.repair_process_profile),
            "minSpotPriority": unit.min_spot_priority,
            "sourceOrderStart": unit.source_order_start,
            "sourceOrderEnd": unit.source_order_end,
            "dependencyTags": list(unit.dependency_tags),
            "releaseGain": unit.release_gain,
            "completionGain": unit.completion_gain,
            "releasedCandidateGain": unit.released_candidate_gain,
            "topologyRisk": unit.topology_risk,
            "bufferPreference": list(unit.buffer_preference),
            "isMandatoryClearance": unit.is_mandatory_clearance,
            "segmentIndex": unit.segment_index,
            "segmentRole": unit.segment_role,
            "segmentClass": unit.segment_class,
            "pressureCut": _phase1_unit_pressure_cut(unit),
            "selected": unit.unit_id in selected_package_ids,
            "bufferTrack": (
                buffer_assignment.get(unit.vehicle_nos[0])
                if unit.unit_id in selected_package_ids
                else None
            ),
        }
        for unit in all_task_packages
    ]
    return {
        "totalTaskPackageCount": len(all_task_packages),
        "selectedPackageIds": [unit.unit_id for unit in task_packages],
        "selectedChainIds": sorted({unit.chain_id for unit in task_packages}),
        "selectedPackageCount": len(task_packages),
        "primaryPackageIds": [
            unit.unit_id for unit in task_packages if unit.unit_type in {"main_backbone", "clearance"}
        ],
        "selectedVehicleNos": sorted(facts.vehicle_no for facts in selected_facts),
        "selectedVehicleCount": len(selected_facts),
        "selectedLocalVehicleNos": sorted(selected_local_vehicle_nos),
        "selectedLocalVehicleCount": len(selected_local_vehicle_nos),
        "selectedTotalLengthM": round(sum(facts.vehicle_length for facts in selected_facts), 1),
        "deferredVehicleNos": sorted(deferred_vehicle_nos),
        "hiddenVehicleNos": sorted(hidden_vehicle_nos),
        "releasedVehicleNos": sorted(released_vehicle_nos),
        "openingPlanCount": len(opening_plans),
        "localFinishPlanCount": len(local_finish_plans),
        "selectedOpeningPlanIds": [plan.plan_id for plan in selected_opening_plans],
        "selectedPackageSourceTracks": list(selected_package_source_tracks),
        "trackFacts": track_facts_projection,
        "packageEdges": package_edge_projection,
        "commitmentState": {
            "selectedPackageIds": sorted(selected_package_ids),
            "selectedChainIds": sorted({unit.chain_id for unit in task_packages}),
            "selectedPackageSourceTracks": sorted({unit.source_track for unit in task_packages}),
            "bufferLockPolicy": "phase1_task_package_first",
        },
        "packagePlanDiagnostics": dict(package_plan_diagnostics),
        "openingReleasedVehicleCount": sum(len(plan.released_vehicle_nos) for plan in selected_opening_plans),
        "openingBlockerVehicleCount": sum(len(plan.blocker_vehicle_nos) for plan in selected_opening_plans),
        "deferredByFinalFamily": dict(sorted(deferred_by_family.items())),
        "pressureCutCounts": dict(sorted(pressure_counts.items())),
        "allPressureCutCounts": dict(sorted(all_pressure_counts.items())),
        "bufferLengthsM": buffer_lengths,
        "bufferVehicleCounts": dict(sorted(buffer_loads.items())),
        "bufferBlockOrder": ordered_blocks_by_buffer,
        "bufferFamilyMix": buffer_family_mix,
        "phase3BranchCount": sum(max(0, len(families) - 1) for families in buffer_family_mix.values()),
        "nonDepotRegionCompletion": region_completion,
        "targetRankByVehicle": dict(sorted(target_rank_by_vehicle.items())),
        "bufferCapacityM": {
            track: round(float(master.tracks[track].effective_length_m), 1)
            for track in JI_BUFFER_TRACKS
        },
        "localFinishPlans": [
            {
                "planId": plan.plan_id,
                "sourceTrack": plan.source_track,
                "targetTrack": plan.target_track,
                "vehicleNos": list(plan.vehicle_nos),
                "totalLengthM": plan.total_length,
                "priorityTag": plan.priority_tag,
                "clusterKind": plan.cluster_kind,
                "completionGain": plan.completion_gain,
                "releasedCandidateGain": plan.released_candidate_gain,
                "sourcePendingPressure": plan.source_pending_pressure,
            }
            for plan in local_finish_plans
        ],
        "openingPlans": [
            {
                "planId": plan.plan_id,
                "sourceTrack": plan.source_track,
                "blockerVehicleNos": list(plan.blocker_vehicle_nos),
                "releasedVehicleNos": list(plan.released_vehicle_nos),
                "blockerTotalLengthM": plan.blocker_total_length,
                "releasedTotalLengthM": plan.released_total_length,
                "releasedFinalFamilies": list(plan.released_final_families),
                "blockerTargetTracks": list(plan.blocker_target_tracks),
                "selected": plan.plan_id in selected_opening_plan_ids,
            }
            for plan in opening_plans
        ],
        "taskPackages": task_package_projection,
        "trackChains": _build_phase1_chain_projection(
            all_task_packages=all_task_packages,
            selected_package_ids=selected_package_ids,
            buffer_assignment=buffer_assignment,
        ),
    }


def _phase1_non_depot_region_completion(
    *,
    facts_list: list[VehicleStageFacts],
    selected_vehicle_nos: frozenset[str],
    goal_overrides: dict[str, tuple[str, str]],
    deferred_vehicle_nos: frozenset[str],
) -> dict[str, Any]:
    total = 0
    local_finished: list[str] = []
    precompiled: list[str] = []
    already_ok: list[str] = []
    pending: list[str] = []
    completed_by_track: Counter[str] = Counter()
    pending_by_track: Counter[str] = Counter()
    for facts in facts_list:
        if facts.current_track not in PHASE1_NON_DEPOT_REGION_TRACKS:
            continue
        if facts.is_depot_area_vehicle:
            continue
        total += 1
        override = goal_overrides.get(facts.vehicle_no)
        if override is not None:
            local_finished.append(facts.vehicle_no)
            completed_by_track[facts.current_track] += 1
            continue
        if facts.vehicle_no in selected_vehicle_nos:
            precompiled.append(facts.vehicle_no)
            completed_by_track[facts.current_track] += 1
            continue
        if _is_phase1_region_already_ok(facts):
            already_ok.append(facts.vehicle_no)
            completed_by_track[facts.current_track] += 1
            continue
        pending.append(facts.vehicle_no)
        pending_by_track[facts.current_track] += 1
    completed_count = len(local_finished) + len(precompiled) + len(already_ok)
    pending_by_reason = _phase1_pending_reason_breakdown(
        facts_list=facts_list,
        pending_vehicle_nos=frozenset(pending),
        deferred_vehicle_nos=deferred_vehicle_nos,
    )
    return {
        "totalVehicleCount": total,
        "completedVehicleCount": completed_count,
        "completionRatio": round(completed_count / total, 4) if total else 1.0,
        "localFinishedVehicleNos": sorted(local_finished),
        "precompiledVehicleNos": sorted(precompiled),
        "alreadyOkVehicleNos": sorted(already_ok),
        "pendingVehicleNos": sorted(pending),
        "pendingDeferredVehicleNos": sorted(
            vehicle_no for vehicle_no in pending if vehicle_no in deferred_vehicle_nos
        ),
        "pendingByReason": pending_by_reason,
        "completedByTrack": dict(sorted(completed_by_track.items())),
        "pendingByTrack": dict(sorted(pending_by_track.items())),
        "counts": {
            "localFinished": len(local_finished),
            "precompiled": len(precompiled),
            "alreadyOk": len(already_ok),
            "pending": len(pending),
        },
    }


def _is_phase1_region_already_ok(facts: VehicleStageFacts) -> bool:
    if facts.current_track in {"存4北", "存4南"}:
        return facts.is_cun4bei_final and not facts.needs_depot_batch
    if facts.current_track in JI_BUFFER_TRACKS:
        return not facts.needs_depot_batch
    return (not facts.needs_depot_batch) and facts.is_current_final_track


def _phase1_pending_reason_breakdown(
    *,
    facts_list: list[VehicleStageFacts],
    pending_vehicle_nos: frozenset[str],
    deferred_vehicle_nos: frozenset[str],
) -> dict[str, Any]:
    pending_by_vehicle = {
        facts.vehicle_no: facts
        for facts in facts_list
        if facts.vehicle_no in pending_vehicle_nos
    }
    hidden_pending: list[str] = []
    released_not_selected: list[str] = []
    local_doable: list[str] = []
    topology_conflict: list[str] = []
    for vehicle_no, facts in pending_by_vehicle.items():
        if vehicle_no in deferred_vehicle_nos and facts.needs_depot_batch:
            released_not_selected.append(vehicle_no)
            continue
        if facts.current_track in WASH_CONFLICT_TRACKS or facts.current_zone == "WORK":
            local_doable.append(vehicle_no)
            continue
        if facts.current_track in STORAGE_TRACKS and facts.needs_depot_batch:
            hidden_pending.append(vehicle_no)
            continue
        topology_conflict.append(vehicle_no)
    return {
        "hiddenBlockedVehicleNos": sorted(hidden_pending),
        "releasedNotSelectedVehicleNos": sorted(released_not_selected),
        "localDoableVehicleNos": sorted(local_doable),
        "topologyConflictVehicleNos": sorted(topology_conflict),
        "counts": {
            "hiddenBlocked": len(hidden_pending),
            "releasedNotSelected": len(released_not_selected),
            "localDoable": len(local_doable),
            "topologyConflict": len(topology_conflict),
        },
    }


def _phase1_backbone_order_key(facts: VehicleStageFacts) -> tuple[int, int, int, int, float]:
    return (
        _pressure_cut_priority(_pressure_cut_name(facts)),
        _depot_topology_entry_priority(facts.final_family),
        _depot_repair_process_priority(facts.repair_process),
        _final_spot_priority(facts.final_target_spot),
        _phase1_track_priority(facts.current_track),
        -facts.vehicle_length,
    )


def _phase1_member_order_bucket(facts: VehicleStageFacts) -> int:
    spot_priority = _final_spot_priority(facts.final_target_spot)
    if spot_priority <= 199:
        return 0
    if spot_priority <= 299:
        return 1
    if spot_priority <= 399:
        return 2
    return 9


def _depot_topology_entry_priority(final_track: str) -> int:
    return {
        "轮": 0,
        "修1": 1,
        "修2": 2,
        "修3": 3,
        "修4": 4,
    }.get(final_track, 9)


def _depot_repair_process_priority(repair_process: str) -> int:
    # 厂修车应更靠里，随机大库落位和 phase1 入库顺序都要给它更深的优先级。
    return 1 if repair_process == "厂修" else 0


def _pressure_cut_name(facts: VehicleStageFacts) -> str:
    if facts.current_track in {"洗北", "洗南", "油"}:
        return "wash_to_ji"
    if facts.current_track in {"调棚", "预修"}:
        return "work_to_ji"
    if facts.current_track in {"机南", "机棚", "机北3"}:
        return "ji_internal"
    return "storage_to_ji"


def _pressure_cut_priority(pressure_cut: str) -> int:
    return {
        "wash_to_ji": 0,
        "opening_release_to_ji": 1,
        "work_to_ji": 2,
        "ji_internal": 3,
        "storage_to_ji": 4,
    }.get(pressure_cut, 9)


def _phase1_must_move(facts: VehicleStageFacts) -> bool:
    return facts.current_track in {"洗北", "洗南", "油", "机棚"}


def _phase1_track_priority(track_name: str) -> int:
    return {
        "洗北": 0,
        "洗南": 1,
        "油": 2,
        "机棚": 3,
        "调棚": 4,
        "预修": 5,
        "机南": 6,
        "机北3": 7,
    }.get(track_name, 9)


def _needs_stage2_depot_collect(
    *,
    current_track: str,
    final_allowed_tracks: tuple[str, ...],
    needs_depot_batch: bool,
    is_depot_area_vehicle: bool,
    is_current_final_track: bool,
) -> bool:
    if needs_depot_batch:
        return False
    if not is_depot_area_vehicle:
        return False
    if "存4北" in final_allowed_tracks:
        return True
    return not is_current_final_track and current_track != "存4北"


def _track_zone(track_name: str) -> str:
    if track_name in WASH_CONFLICT_TRACKS:
        return "WASH"
    if track_name in JI_BUFFER_TRACKS:
        return "JI_BUFFER"
    if track_name == "存4北":
        return "CUN4BEI"
    if track_name == "存4南":
        return "CUN4NAN"
    if track_name in DEPOT_TARGET_TRACKS or track_name in DEPOT_OUTER_TRACKS:
        return "DEPOT"
    if track_name in STORAGE_TRACKS:
        return "STORAGE"
    if track_name in {"调棚", "预修", "调北"}:
        return "WORK"
    return "OTHER"


def _zone_priority(zone: str) -> int:
    return {
        "WASH": 0,
        "WORK": 1,
        "JI_BUFFER": 2,
        "CUN4BEI": 3,
        "STORAGE": 4,
        "DEPOT": 5,
        "CUN4NAN": 6,
        "OTHER": 9,
    }.get(zone, 9)


def _phase1_goal(
    facts: VehicleStageFacts,
    *,
    stage_plan: TopologyStagePlan,
) -> dict[str, Any]:
    if facts.is_fixed_depot_resident:
        return _fixed_depot_resident_goal(facts)
    phase1 = stage_plan.phase1_plan
    goal_override = phase1.goal_overrides.get(facts.vehicle_no)
    if goal_override is not None:
        target_track, source = goal_override
        return _stage_track_goal(
            facts.vehicle_no,
            target_track,
            source,
        )
    if facts.vehicle_no in phase1.selected_vehicle_nos:
        return _stage_track_goal(
            facts.vehicle_no,
            phase1.buffer_assignment[facts.vehicle_no],
            "PHASE1_BACKBONE_PLACE",
        )
    return _hold_current_track_goal(facts)


def _phase1_wave_stage_policy(
    wave_plan: Phase1WavePlan,
    *,
    stage_plan: TopologyStagePlan,
    vehicle_facts: list[VehicleStageFacts],
) -> dict[str, Any]:
    return {
        "waveName": wave_plan.wave_name,
        "waveRole": wave_plan.wave_role,
        "waveType": wave_plan.wave_type,
        "selectedSourceTrack": wave_plan.selected_source_track,
        "selectedBlockIds": list(wave_plan.selected_block_ids),
        "requiredPredecessorIds": list(wave_plan.required_predecessor_ids),
        "selectedVehicleNos": sorted(wave_plan.selected_vehicle_nos),
        "packageAssignments": dict(wave_plan.buffer_assignment),
        "layoutAssignments": dict(wave_plan.buffer_assignment),
        "packageTargetRanks": dict(wave_plan.target_rank_by_vehicle),
        "layoutTargetRanks": dict(wave_plan.target_rank_by_vehicle),
        "vehicleGoals": [
            _phase1_wave_goal(facts, wave_plan=wave_plan, stage_plan=stage_plan)
            for facts in vehicle_facts
        ],
        "waveDiagnostics": dict(wave_plan.diagnostics),
    }


def _phase2_execution_stage_policy(
    execution_plan: Phase2ExecutionPlan | None,
) -> dict[str, Any] | None:
    if execution_plan is None:
        return None
    return {
        "executionName": execution_plan.execution_name,
        "sourceTracks": [layer.source_track for layer in execution_plan.track_layers],
        "collectionBatches": [list(batch) for batch in execution_plan.collection_batches],
        "predecessorUnlockVehicleNos": list(execution_plan.predecessor_unlock_vehicle_nos),
        "mustPullVehicleNos": list(execution_plan.must_pull_vehicle_nos),
        "unlockingOptionalVehicleNos": list(execution_plan.unlocking_optional_vehicle_nos),
        "phase3ClearanceVehicleNos": list(execution_plan.phase3_clearance_vehicle_nos),
        "pureBatchOptionalVehicleNos": list(execution_plan.pure_batch_optional_vehicle_nos),
        "optionalCun4VehicleNos": list(execution_plan.optional_cun4_vehicle_nos),
        "deferredTailVehicleNos": list(execution_plan.deferred_tail_vehicle_nos),
        "transferVehicleNos": list(execution_plan.transfer_vehicle_nos),
        "reservedCun4CapacityM": execution_plan.reserved_cun4_capacity_m,
        "trackLayers": [
            {
                "sourceTrack": layer.source_track,
                "layerIndex": layer.layer_index,
                "groupIds": list(layer.group_ids),
                "vehicleNos": list(layer.vehicle_nos),
                "outboundVehicleNos": list(layer.outbound_vehicle_nos),
                "cun4FinalVehicleNos": list(layer.cun4_final_vehicle_nos),
                "exposedPrefixVehicleNos": list(layer.exposed_prefix_vehicle_nos),
                "totalLengthM": layer.total_length_m,
                "diagnostics": dict(layer.diagnostics),
            }
            for layer in execution_plan.track_layers
        ],
        "executionDiagnostics": dict(execution_plan.diagnostics),
    }


def rebuild_phase2_execution_policy_for_runtime(
    *,
    stage_payload: dict[str, Any],
    track_sequences: dict[str, list[str] | tuple[str, ...]],
) -> dict[str, Any] | None:
    stage_policy = dict(stage_payload.get("stagePolicy") or {})
    depot_stay_vehicle_nos = {
        str(item)
        for item in stage_policy.get("depotStayVehicles") or ()
    }
    fixed_depot_resident_vehicle_nos = {
        str(item)
        for item in stage_policy.get("fixedDepotResidentVehicleNos") or ()
    }
    cun4_final_vehicle_nos = {
        str(item)
        for item in stage_policy.get("cun4FinalVehicles") or ()
    }
    outbound_vehicle_nos = {
        str(item)
        for item in stage_policy.get("depotOutboundVehicles") or ()
    }
    depot_stay_vehicle_nos |= fixed_depot_resident_vehicle_nos
    cun4_final_vehicle_nos -= fixed_depot_resident_vehicle_nos
    outbound_vehicle_nos -= fixed_depot_resident_vehicle_nos
    active_vehicle_nos = cun4_final_vehicle_nos | outbound_vehicle_nos
    if not active_vehicle_nos:
        return None

    vehicle_rows = {
        str(item["vehicleNo"]): dict(item)
        for item in stage_payload.get("vehicleInfo") or ()
    }
    vehicle_traits = {
        vehicle_no: _phase2_vehicle_traits_from_row(row)
        for vehicle_no, row in vehicle_rows.items()
    }
    group_meta_by_vehicle: dict[str, dict[str, Any]] = {}
    for group in stage_policy.get("phase2OutboundGroups") or ():
        meta = {
            "groupKind": str(group.get("groupKind") or ""),
            "finalTargetTrack": str(group.get("finalTargetTrack") or ""),
            "finalFamily": str(group.get("finalFamily") or ""),
        }
        for vehicle_no in group.get("vehicleNos") or ():
            group_meta_by_vehicle[str(vehicle_no)] = meta

    runtime_layers: list[Phase2TrackLayer] = []
    runtime_blocked_tails: list[dict[str, Any]] = []
    runtime_deferred_tail_vehicle_nos: list[str] = []
    ordered_tracks = sorted(
        (
            str(track_name)
            for track_name, sequence in track_sequences.items()
            if sequence and _phase2_depot_track_priority(str(track_name)) < 99
        ),
        key=_phase2_depot_track_priority,
    )
    for source_track in ordered_tracks:
        source_seq = [str(vehicle_no) for vehicle_no in track_sequences.get(source_track, ())]
        prefix_vehicle_nos: list[str] = []
        prefix_total_length_m = 0.0
        blocked_tail_vehicle_nos: list[str] = []
        run_vehicle_nos: list[str] = []
        run_kind = ""
        run_target_track = ""
        run_final_family = ""
        layer_index = 1
        anchor_vehicle_no = ""

        def flush_run() -> None:
            nonlocal run_vehicle_nos, run_kind, run_target_track, run_final_family, prefix_total_length_m, layer_index
            if not run_vehicle_nos:
                return
            total_length_m = round(
                sum(float(vehicle_rows[vehicle_no]["vehicleLength"]) for vehicle_no in run_vehicle_nos),
                1,
            )
            prefix_vehicle_nos.extend(run_vehicle_nos)
            prefix_total_length_m = round(prefix_total_length_m + total_length_m, 1)
            outbound_run_vehicle_nos = tuple(run_vehicle_nos) if run_kind == "DEPOT_OUTBOUND" else ()
            cun4_run_vehicle_nos = tuple(run_vehicle_nos) if run_kind == "CUN4_FINAL" else ()
            runtime_layers.append(
                Phase2TrackLayer(
                    source_track=source_track,
                    layer_index=layer_index,
                    group_ids=(f"RT::{source_track}::{layer_index}",),
                    vehicle_nos=tuple(run_vehicle_nos),
                    total_length_m=total_length_m,
                    outbound_vehicle_nos=outbound_run_vehicle_nos,
                    cun4_final_vehicle_nos=cun4_run_vehicle_nos,
                    exposed_prefix_vehicle_nos=tuple(prefix_vehicle_nos),
                    diagnostics={
                        "sourceTrack": source_track,
                        "layerIndex": layer_index,
                        "groupIds": [f"RT::{source_track}::{layer_index}"],
                        "vehicleNos": list(run_vehicle_nos),
                        "outboundVehicleNos": list(outbound_run_vehicle_nos),
                        "cun4FinalVehicleNos": list(cun4_run_vehicle_nos),
                        "totalLengthM": total_length_m,
                        "exposedPrefixVehicleNos": list(prefix_vehicle_nos),
                        "exposedPrefixLengthM": prefix_total_length_m,
                        "runtimeRebuilt": True,
                        "finalTargetTrack": run_target_track,
                        "finalFamily": run_final_family,
                    },
                )
            )
            layer_index += 1
            run_vehicle_nos = []
            run_kind = ""
            run_target_track = ""
            run_final_family = ""

        for index, vehicle_no in enumerate(source_seq):
            if vehicle_no in fixed_depot_resident_vehicle_nos:
                flush_run()
                if any(tail_vehicle_no in active_vehicle_nos for tail_vehicle_no in source_seq[index + 1:]):
                    anchor_vehicle_no = vehicle_no
                continue
            if vehicle_no in depot_stay_vehicle_nos or vehicle_no not in active_vehicle_nos:
                tail_has_outbound = any(
                    tail_vehicle_no in outbound_vehicle_nos
                    for tail_vehicle_no in source_seq[index + 1:]
                )
                if tail_has_outbound:
                    if run_vehicle_nos and run_kind != "PREDECESSOR_UNLOCK":
                        flush_run()
                    if not run_vehicle_nos:
                        run_kind = "PREDECESSOR_UNLOCK"
                        run_target_track = ""
                        run_final_family = ""
                    run_vehicle_nos.append(vehicle_no)
                    continue
                flush_run()
                if vehicle_no in depot_stay_vehicle_nos and not anchor_vehicle_no:
                    anchor_vehicle_no = vehicle_no
                continue
            if anchor_vehicle_no:
                blocked_tail_vehicle_nos.append(vehicle_no)
                runtime_deferred_tail_vehicle_nos.append(vehicle_no)
                continue
            meta = group_meta_by_vehicle.get(vehicle_no) or {}
            vehicle_kind = "CUN4_FINAL" if vehicle_no in cun4_final_vehicle_nos else "DEPOT_OUTBOUND"
            vehicle_target_track = str(meta.get("finalTargetTrack") or "")
            vehicle_final_family = str(meta.get("finalFamily") or vehicle_target_track)
            if (
                run_vehicle_nos
                and (
                    vehicle_kind != run_kind
                    or vehicle_target_track != run_target_track
                    or vehicle_final_family != run_final_family
                )
            ):
                flush_run()
            if not run_vehicle_nos:
                run_kind = vehicle_kind
                run_target_track = vehicle_target_track
                run_final_family = vehicle_final_family
            run_vehicle_nos.append(vehicle_no)
        flush_run()
        if blocked_tail_vehicle_nos:
            runtime_blocked_tails.append(
                {
                    "sourceTrack": source_track,
                    "anchorVehicleNo": anchor_vehicle_no,
                    "blockedTailVehicleNos": list(blocked_tail_vehicle_nos),
                }
            )

    base_execution_plan = _build_phase2_execution_plan(
        track_layers=tuple(runtime_layers),
        vehicle_traits=vehicle_traits,
    )
    if base_execution_plan is None:
        if not runtime_deferred_tail_vehicle_nos:
            return None
        return {
            "executionName": "phase2_collect_then_transfer",
            "sourceTracks": [],
            "collectionBatches": [],
            "predecessorUnlockVehicleNos": [],
            "mustPullVehicleNos": [],
            "unlockingOptionalVehicleNos": [],
            "pureBatchOptionalVehicleNos": [],
            "optionalCun4VehicleNos": [],
            "deferredTailVehicleNos": list(dict.fromkeys(runtime_deferred_tail_vehicle_nos)),
            "transferVehicleNos": [],
            "reservedCun4CapacityM": 60.0,
            "trackLayers": [],
            "executionDiagnostics": {
                "runtimeRebuilt": True,
                "runtimeBlockedTails": runtime_blocked_tails,
                "deferredTailVehicleNos": list(dict.fromkeys(runtime_deferred_tail_vehicle_nos)),
            },
        }

    merged_deferred_tail_vehicle_nos = tuple(
        dict.fromkeys(
            [
                *base_execution_plan.deferred_tail_vehicle_nos,
                *runtime_deferred_tail_vehicle_nos,
            ]
        )
    )
    rebuilt_execution_plan = Phase2ExecutionPlan(
        execution_name=base_execution_plan.execution_name,
        track_layers=base_execution_plan.track_layers,
        collection_batches=base_execution_plan.collection_batches,
        predecessor_unlock_vehicle_nos=base_execution_plan.predecessor_unlock_vehicle_nos,
        must_pull_vehicle_nos=base_execution_plan.must_pull_vehicle_nos,
        unlocking_optional_vehicle_nos=base_execution_plan.unlocking_optional_vehicle_nos,
        phase3_clearance_vehicle_nos=base_execution_plan.phase3_clearance_vehicle_nos,
        pure_batch_optional_vehicle_nos=base_execution_plan.pure_batch_optional_vehicle_nos,
        optional_cun4_vehicle_nos=base_execution_plan.optional_cun4_vehicle_nos,
        deferred_tail_vehicle_nos=merged_deferred_tail_vehicle_nos,
        transfer_vehicle_nos=base_execution_plan.transfer_vehicle_nos,
        reserved_cun4_capacity_m=base_execution_plan.reserved_cun4_capacity_m,
        diagnostics={
            **base_execution_plan.diagnostics,
            "runtimeRebuilt": True,
            "runtimeBlockedTails": runtime_blocked_tails,
            "deferredTailVehicleNos": list(merged_deferred_tail_vehicle_nos),
            "deferredTailVehicleCount": len(merged_deferred_tail_vehicle_nos),
        },
    )
    return _phase2_execution_stage_policy(rebuilt_execution_plan)


def rebuild_phase1_stage_for_runtime(
    *,
    master: MasterData,
    track_info: list[dict[str, Any]],
    current_vehicle_info: list[dict[str, Any]],
    loco_track_name: str,
    original_goal_rows: list[dict[str, Any]],
    initial_buffer_vehicle_nos: frozenset[str] | None = None,
) -> dict[str, Any]:
    goal_row_by_vehicle = {
        str(item["vehicleNo"]): dict(item)
        for item in original_goal_rows
    }
    runtime_vehicle_rows: list[dict[str, Any]] = []
    for item in current_vehicle_info:
        vehicle_no = str(item["vehicleNo"])
        goal_row = goal_row_by_vehicle.get(vehicle_no)
        if goal_row is None:
            raise ValueError(f"missing original phase1 goal row for vehicle {vehicle_no}")
        runtime_vehicle_rows.append(
            {
                "trackName": str(item["trackName"]),
                "order": str(item["order"]),
                "vehicleModel": str(item["vehicleModel"]),
                "vehicleNo": vehicle_no,
                "repairProcess": str(item["repairProcess"]),
                "vehicleLength": item["vehicleLength"],
                "vehicleAttributes": str(item.get("vehicleAttributes", "")),
                "targetTrack": str(goal_row.get("targetTrack") or ""),
                "isSpotting": str(goal_row.get("isSpotting") or ""),
            }
        )
    runtime_payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": [dict(item) for item in track_info],
        "vehicleInfo": runtime_vehicle_rows,
        "locoTrackName": loco_track_name,
    }
    workflow_payload = build_l7_closed_topology_workflow_payload(
        master,
        runtime_payload,
        allow_internal_loco_tracks=True,
        phase1_respect_existing_buffer_occupancy=True,
        phase1_buffer_occupancy_exempt_vehicle_nos=initial_buffer_vehicle_nos,
    )
    return dict(workflow_payload["workflowStages"][0])


def _phase2_vehicle_traits_from_row(row: dict[str, Any]) -> dict[str, Any]:
    attributes = str(row.get("vehicleAttributes") or "").strip()
    return {
        "need_weigh": attributes == "称重",
        "is_heavy": attributes == "重车",
        "is_close_door": attributes == "关门车",
        "vehicle_length": float(row.get("vehicleLength") or 0.0),
    }


def _phase1_wave_goal(
    facts: VehicleStageFacts,
    *,
    wave_plan: Phase1WavePlan,
    stage_plan: TopologyStagePlan,
) -> dict[str, Any]:
    if facts.is_fixed_depot_resident:
        return _fixed_depot_resident_goal(facts)
    goal_override = wave_plan.goal_overrides.get(facts.vehicle_no)
    if goal_override is not None:
        target_track, source = goal_override
        return _stage_track_goal(facts.vehicle_no, target_track, source)
    if facts.vehicle_no in wave_plan.selected_vehicle_nos:
        return _stage_track_goal(
            facts.vehicle_no,
            wave_plan.buffer_assignment[facts.vehicle_no],
            "PHASE1_BACKBONE_PLACE",
        )
    return _hold_current_track_goal(facts)


def _phase2_goal(
    facts: VehicleStageFacts,
    *,
    stage_plan: TopologyStagePlan,
) -> dict[str, Any]:
    if facts.is_fixed_depot_resident:
        return _fixed_depot_resident_goal(facts)
    if facts.vehicle_no in stage_plan.phase1_plan.selected_vehicle_nos:
        return _snapshot_track_goal(
            vehicle_no=facts.vehicle_no,
            target_track=stage_plan.phase1_plan.buffer_assignment[facts.vehicle_no],
            source="PHASE2_HOLD_PHASE1_BACKBONE",
            vehicle_length=facts.vehicle_length,
        )
    execution_plan = stage_plan.stage2_plan.execution_plan
    transfer_vehicle_nos = frozenset(execution_plan.transfer_vehicle_nos) if execution_plan is not None else frozenset()
    if facts.vehicle_no in transfer_vehicle_nos:
        return _stage_track_goal(facts.vehicle_no, "存4北", "PHASE2_TRANSFER_TO_CUN4")
    if execution_plan is not None and facts.vehicle_no in execution_plan.deferred_tail_vehicle_nos:
        return _hold_current_track_goal(facts)
    if facts.vehicle_no in stage_plan.stage2_plan.demand_summary.depot_stay_vehicle_nos:
        return _snapshot_track_goal(
            vehicle_no=facts.vehicle_no,
            target_track=_phase2_resting_track(facts),
            source="PHASE2_DEPOT_STAY",
            vehicle_length=facts.vehicle_length,
        )
    return _snapshot_track_goal(
        vehicle_no=facts.vehicle_no,
        target_track=_phase2_resting_track(facts),
        source="STAGE_HOLD",
        vehicle_length=facts.vehicle_length,
    )


def _phase3_goal(
    facts: VehicleStageFacts,
    *,
    final_goal_by_vehicle: dict[str, dict[str, Any]],
    stage_plan: TopologyStagePlan,
) -> dict[str, Any]:
    if facts.is_fixed_depot_resident:
        return _fixed_depot_resident_goal(facts)
    if facts.needs_depot_batch:
        return _normalized_final_goal_payload(facts)
    return _snapshot_track_goal(
        vehicle_no=facts.vehicle_no,
        target_track=facts.current_track,
        source=PHASE3_DYNAMIC_CURRENT_HOLD,
        vehicle_length=facts.vehicle_length,
    )


def _fixed_depot_resident_goal(facts: VehicleStageFacts) -> dict[str, Any]:
    return _snapshot_track_goal(
        vehicle_no=facts.vehicle_no,
        target_track=facts.current_track,
        source=FIXED_DEPOT_RESIDENT_SOURCE,
        vehicle_length=facts.vehicle_length,
    )


def _hold_current_track_goal(facts: VehicleStageFacts) -> dict[str, Any]:
    return _snapshot_track_goal(
        vehicle_no=facts.vehicle_no,
        target_track=facts.current_track,
        source="HOLD_CURRENT",
        vehicle_length=facts.vehicle_length,
    )


def _snapshot_track_goal(
    *,
    vehicle_no: str,
    target_track: str,
    source: str,
    vehicle_length: float,
) -> dict[str, Any]:
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": target_track,
        "targetMode": "SNAPSHOT",
        "targetSource": source,
        "isSpotting": "",
        "vehicleLength": vehicle_length,
    }


def _stage_track_goal(vehicle_no: str, target_track: str, source: str) -> dict[str, Any]:
    return {
        "vehicleNo": vehicle_no,
        "targetTrack": target_track,
        "targetMode": "AREA",
        "targetAreaCode": f"STAGE::{source}",
        "targetSource": source,
        "isSpotting": "",
    }


def _initial_vehicle_info_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trackName": str(item["trackName"]),
        "order": str(item["order"]),
        "vehicleModel": str(item["vehicleModel"]),
        "vehicleNo": str(item["vehicleNo"]),
        "repairProcess": str(item["repairProcess"]),
        "vehicleLength": item["vehicleLength"],
        "vehicleAttributes": str(item.get("vehicleAttributes", "")),
    }


def _final_goal_payload(item: dict[str, Any]) -> dict[str, Any]:
    goal = {
        "vehicleNo": str(item["vehicleNo"]),
        "targetTrack": str(item["targetTrack"]),
        "isSpotting": str(item.get("isSpotting", "")),
    }
    for key in ("targetMode", "targetAreaCode", "targetSpotCode", "targetSource"):
        value = item.get(key)
        if value not in (None, ""):
            goal[key] = value
    return goal


def _normalized_final_goal_payload(facts: VehicleStageFacts) -> dict[str, Any]:
    goal = {
        "vehicleNo": facts.vehicle_no,
        "targetTrack": facts.final_target_track,
        "targetMode": facts.final_target_mode,
        "isSpotting": facts.final_target_spot,
    }
    if facts.final_target_area_code:
        goal["targetAreaCode"] = facts.final_target_area_code
    if facts.final_target_source:
        goal["targetSource"] = facts.final_target_source
    if facts.final_preferred_tracks:
        goal["preferredTargetTracks"] = list(facts.final_preferred_tracks)
    if facts.final_fallback_tracks:
        goal["fallbackTargetTracks"] = list(facts.final_fallback_tracks)
    if facts.final_allowed_tracks:
        goal["allowedTargetTracks"] = list(facts.final_allowed_tracks)
    return goal


def _primary_final_track_from_allowed(
    allowed_tracks: tuple[str, ...],
    final_target_track: str,
) -> str:
    for track in allowed_tracks:
        if track in DEPOT_TARGET_TRACKS:
            return track
    return final_target_track


def _phase2_resting_track(facts: VehicleStageFacts) -> str:
    return facts.current_track


def _phase4_goal(
    facts: VehicleStageFacts,
    *,
    final_goal_by_vehicle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if facts.is_fixed_depot_resident:
        return _fixed_depot_resident_goal(facts)
    if facts.needs_depot_batch:
        return {
            "vehicleNo": facts.vehicle_no,
            "targetTrack": facts.final_target_track,
            "targetMode": "SNAPSHOT",
            "targetSource": PHASE4_DYNAMIC_CURRENT_HOLD,
            "isSpotting": "",
        }
    return dict(final_goal_by_vehicle[facts.vehicle_no])


def _phase1_non_depot_ji_evict_track(facts: VehicleStageFacts) -> str:
    if facts.is_cun4bei_final:
        return "存4北"
    return "存4南"


def _phase1_cun4_clear_target_track(current_track: str) -> str:
    return "存4南" if current_track == "存4北" else "存4北"


def _phase1_front_evict_target_track(
    *,
    source_track: str,
    facts: VehicleStageFacts,
) -> str | None:
    if facts.needs_depot_batch:
        return None
    if facts.final_target_track != facts.current_track:
        return None
    for track in PHASE1_TEMP_PARKING_TRACKS:
        if track != source_track:
            return track
    return None


def _final_family_priority(final_track: str) -> int:
    return {
        "轮": 0,
        "修1": 1,
        "修2": 2,
        "修3": 3,
        "修4": 4,
        "存4北": 5,
    }.get(final_track, 9)


def _final_spot_priority(final_spot: str) -> int:
    if not final_spot:
        return 999
    if final_spot.isdigit():
        return int(final_spot)
    return 998
