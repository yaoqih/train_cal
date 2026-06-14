from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import (
    FIXED_DEPOT_RESIDENT_SOURCE,
    OPERATION_MODE_L7_CLOSED_TOPOLOGY,
    Phase1BackbonePlan,
    Phase1Block,
    SourceTrackPlan,
    Phase1LayoutPackage,
    PHASE3_DYNAMIC_CURRENT_HOLD,
    PHASE4_DYNAMIC_CURRENT_HOLD,
    PHASE4_RESIDUAL_CLEANUP,
    _build_phase1_source_admissions,
    _phase1_wave_a_block_ids,
    build_l7_closed_topology_workflow_payload,
    rebuild_phase1_stage_for_runtime,
    rebuild_phase2_execution_policy_for_runtime,
)
from fzed_shunting.workflow.runner import (
    _resolve_dynamic_stage,
    _build_phase1_runtime_frontier_wave,
    _build_phase1_clearance_wave,
    _phase1_check_rolling_candidate_executable,
    _phase1_filter_executable_rolling_candidates,
    _phase1_rolling_candidates_with_route_clearance,
    _phase1_source_prefix_blocking_vehicle_nos,
    solve_workflow,
)
from fzed_shunting.workflow.phase1_rolling_planner import Phase1RollingCandidate
from fzed_shunting.workflow.phase1_rolling_planner import (
    _candidate_target_tracks,
    build_phase1_rolling_candidates,
)
from fzed_shunting.verify.replay import ReplayState


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(
    *,
    track_name: str,
    order: str,
    vehicle_no: str,
    target_track: str,
    is_spotting: str = "",
    vehicle_length: float = 14.3,
    repair_process: str = "段修",
    vehicle_attributes: str = "",
) -> dict:
    return {
        "trackName": track_name,
        "order": order,
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": repair_process,
        "vehicleLength": vehicle_length,
        "vehicleAttributes": vehicle_attributes,
        "targetTrack": target_track,
        "isSpotting": is_spotting,
    }


def _base_track_info() -> list[dict]:
    return [
        {"trackName": "洗南", "trackDistance": 88.7},
        {"trackName": "洗北", "trackDistance": 88.7},
        {"trackName": "油", "trackDistance": 109.0},
        {"trackName": "调棚", "trackDistance": 174.3},
        {"trackName": "预修", "trackDistance": 140.0},
        {"trackName": "调北", "trackDistance": 142.0},
        {"trackName": "机南", "trackDistance": 90.1},
        {"trackName": "机棚", "trackDistance": 106.8},
        {"trackName": "机北1", "trackDistance": 81.4},
        {"trackName": "机北2", "trackDistance": 55.7},
        {"trackName": "机北3", "trackDistance": 69.1},
        {"trackName": "存1", "trackDistance": 185.3},
        {"trackName": "存2", "trackDistance": 181.9},
        {"trackName": "存3", "trackDistance": 181.9},
        {"trackName": "存4北", "trackDistance": 317.8},
        {"trackName": "存4南", "trackDistance": 154.5},
        {"trackName": "存5南", "trackDistance": 220.0},
        {"trackName": "存5北", "trackDistance": 220.0},
        {"trackName": "修1", "trackDistance": 151.7},
        {"trackName": "修2", "trackDistance": 151.7},
        {"trackName": "修3", "trackDistance": 151.7},
        {"trackName": "修4", "trackDistance": 151.7},
        {"trackName": "轮", "trackDistance": 47.3},
        {"trackName": "修1库外", "trackDistance": 100.0},
    ]


def test_build_l7_closed_topology_workflow_payload_creates_four_topology_stages():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="修2"),
            _vehicle(track_name="存4北", order="1", vehicle_no="C", target_track="存1"),
            _vehicle(track_name="机南", order="1", vehicle_no="D", target_track="存2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)

    assert workflow_payload["operationMode"] == OPERATION_MODE_L7_CLOSED_TOPOLOGY
    assert [stage["name"] for stage in workflow_payload["workflowStages"]] == [
        "phase1_pre_repair_buffering",
        "phase2_depot_area_marshalling",
        "phase3_ji_to_depot_allocation",
        "final_exact_settle_and_cleanup",
    ]
    assert workflow_payload["workflowStages"][0]["routePolicy"]["blockedBranches"] == ["L15-L16"]
    assert workflow_payload["workflowStages"][1]["routePolicy"] == {}
    assert workflow_payload["workflowStages"][2]["routePolicy"] == {}
    assert workflow_payload["workflowStages"][3]["routePolicy"] == {}


def test_phase1_runtime_frontier_waves_remain_single_source():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A1", target_track="修1"),
            _vehicle(track_name="洗南", order="2", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="调棚", order="1", vehicle_no="B1", target_track="修2"),
            _vehicle(track_name="预修", order="1", vehicle_no="C1", target_track="修3"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    stage = workflow_payload["workflowStages"][0]
    runtime_stage = rebuild_phase1_stage_for_runtime(
        master=master,
        track_info=workflow_payload["trackInfo"],
        current_vehicle_info=workflow_payload["initialVehicleInfo"],
        loco_track_name=workflow_payload["locoTrackName"],
        original_goal_rows=list(stage["stagePolicy"]["phase1OriginalGoalRows"]),
        initial_buffer_vehicle_nos=frozenset(stage["stagePolicy"]["phase1InitialBufferVehicleNos"]),
    )
    wave = _build_phase1_runtime_frontier_wave(runtime_stage=runtime_stage)

    assert wave is not None
    assert str(wave.get("selectedSourceTrack") or "")
    assert str(wave.get("waveType") or "")
    assert len((wave.get("waveDiagnostics") or {}).get("selectedSourceTracks") or []) <= 1


def test_phase1_wave_plans_split_clearance_and_marshalling_by_source():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A1", target_track="存1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="洗南", order="1", vehicle_no="B1", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    waves = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1WavePlans"]

    assert waves
    for wave in waves:
        assert str(wave.get("waveType") or "")
        assert str(wave.get("selectedSourceTrack") or "")
    assert any(str(wave.get("waveType")) == "source_marshalling" for wave in waves)


def test_phase1_macro_tasks_bind_source_clearance_to_marshalling_intent():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A1", target_track="存1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="预修", order="1", vehicle_no="B1", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase1_policy = workflow_payload["workflowStages"][0]["stagePolicy"]
    diagnostics = phase1_policy["phase1Diagnostics"]
    macro_tasks = diagnostics["phase1MacroTasks"]
    waves = phase1_policy["phase1WavePlans"]

    assert macro_tasks
    task_by_source = {task["sourceTrack"]: task for task in macro_tasks}
    assert "调棚" in task_by_source
    tiaopeng_task = task_by_source["调棚"]
    assert [chunk["waveRole"] for chunk in tiaopeng_task["waveChunks"]] == [
        "source_clearance",
        "source_marshalling",
    ]
    assert set(tiaopeng_task["vehicleNos"]) == {"A1", "A2"}

    task_waves = [
        wave
        for wave in waves
        if (wave.get("waveDiagnostics") or {}).get("macroTaskId") == tiaopeng_task["taskId"]
    ]
    assert [wave["waveType"] for wave in task_waves] == [
        "source_clearance",
        "source_marshalling",
    ]
    assert all(
        (wave.get("waveDiagnostics") or {}).get("macroTaskBlockIds") == tiaopeng_task["blockIds"]
        for wave in task_waves
    )


def test_phase1_rolling_candidates_follow_macro_task_frontier():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A1", target_track="存1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="预修", order="1", vehicle_no="B1", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    stage = workflow_payload["workflowStages"][0]
    runtime_stage = rebuild_phase1_stage_for_runtime(
        master=master,
        track_info=workflow_payload["trackInfo"],
        current_vehicle_info=workflow_payload["initialVehicleInfo"],
        loco_track_name=workflow_payload["locoTrackName"],
        original_goal_rows=list(stage["stagePolicy"]["phase1OriginalGoalRows"]),
        initial_buffer_vehicle_nos=frozenset(stage["stagePolicy"]["phase1InitialBufferVehicleNos"]),
    )

    candidates = build_phase1_rolling_candidates(runtime_stage=runtime_stage)

    assert candidates
    first_wave = candidates[0].wave
    first_diag = first_wave["waveDiagnostics"]
    assert first_diag["runtimeFrontierStrategy"] == "macro_task_frontier"
    assert first_diag["macroTaskId"]
    assert first_wave["selectedBlockIds"][0] in first_diag["macroTaskBlockIds"]


def test_phase1_route_clearance_wave_inherits_macro_task_context():
    reason_wave = {
        "waveName": "phase1_roll_调棚_U0001_预修",
        "selectedSourceTrack": "调棚",
        "waveDiagnostics": {
            "targetTrack": "预修",
            "macroTaskId": "MT::调棚::U0001",
            "macroTaskBlockIds": ["U0001", "U0002"],
            "macroTaskWaveChunks": [
                {"waveRole": "source_clearance", "blockIds": ["U0001"]},
                {"waveRole": "source_marshalling", "blockIds": ["U0002"]},
            ],
            "macroTaskSourceRole": "work_gate",
            "macroTaskScoreKey": [0, 0, 0],
        },
    }

    wave = _build_phase1_clearance_wave(
        source_track="存5北",
        target_track="存5南",
        vehicle_nos=["B1"],
        goal_by_vehicle={},
        reason_wave=reason_wave,
    )

    diagnostics = wave["waveDiagnostics"]
    assert diagnostics["runtimeFrontierStrategy"] == "rolling_route_clearance"
    assert diagnostics["macroTaskId"] == "MT::调棚::U0001"
    assert diagnostics["macroTaskBlockIds"] == ["U0001", "U0002"]
    assert diagnostics["routeClearanceForMacroTask"] is True


def test_phase1_buffer_candidate_targets_include_runtime_alternatives():
    targets = _candidate_target_tracks(
        block={
            "usesBuffer": True,
            "vehicleNos": ["A"],
            "bufferPreference": ["机南", "机棚", "机北1"],
        },
        source_track="预修",
        goal_by_vehicle={"A": {"targetTrack": "机南"}},
    )

    assert targets[:3] == ("机南", "机棚", "机北1")


def test_phase1_executable_check_reports_source_prefix_mismatch():
    master = load_master_data(DATA_DIR)
    state = ReplayState(
        track_sequences={"调棚": ["X", "A"], "机南": []},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    wave = {
        "waveName": "phase1_roll_调棚_U0001_机南",
        "selectedSourceTrack": "调棚",
        "selectedVehicleNos": ["A"],
        "selectedBlockIds": ["U0001"],
        "waveDiagnostics": {"targetTrack": "机南"},
    }

    check = _phase1_check_rolling_candidate_executable(
        candidate_wave=wave,
        state=state,
        master=master,
    )

    assert check.ok is False
    assert check.reason == "source_prefix_mismatch"
    assert check.source_prefix[:2] == ("X", "A")


def test_phase1_filter_keeps_route_clearance_and_rejects_blocked_direct_candidate():
    master = load_master_data(DATA_DIR)
    state = ReplayState(
        track_sequences={"调棚": ["X", "A"], "机南": [], "存5北": ["B"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    direct = Phase1RollingCandidate(
        wave={
            "waveName": "phase1_roll_调棚_U0001_机南",
            "selectedSourceTrack": "调棚",
            "selectedVehicleNos": ["A"],
            "selectedBlockIds": ["U0001"],
            "waveDiagnostics": {"targetTrack": "机南"},
        },
        score=(0,),
    )
    clearance = Phase1RollingCandidate(
        wave={
            "waveName": "phase1_clear_route_存5北_存5南",
            "selectedSourceTrack": "存5北",
            "selectedVehicleNos": ["B"],
            "selectedBlockIds": ["ROUTE_CLEAR::存5北"],
            "waveDiagnostics": {
                "targetTrack": "存5南",
                "runtimeFrontierStrategy": "rolling_route_clearance",
            },
        },
        score=(1,),
    )

    filtered, rejected = _phase1_filter_executable_rolling_candidates(
        candidates=(direct, clearance),
        state=state,
        master=master,
    )

    assert filtered == (clearance,)
    assert rejected[0]["error"] == "source_prefix_mismatch"
    assert rejected[0]["sourcePrefix"][:2] == ["X", "A"]


def test_phase1_source_prefix_blocking_vehicle_nos_returns_only_front_blockers():
    state = ReplayState(
        track_sequences={"调棚": ["X1", "X2", "A", "B", "Y"]},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    wave = {
        "selectedSourceTrack": "调棚",
        "selectedVehicleNos": ["A", "B"],
        "waveDiagnostics": {"targetTrack": "机南"},
    }

    blockers = _phase1_source_prefix_blocking_vehicle_nos(
        candidate_wave=wave,
        state=state,
    )

    assert blockers == ("X1", "X2")


def test_phase1_route_clearance_generates_source_prefix_clearance_wave():
    master = load_master_data(DATA_DIR)
    state = ReplayState(
        track_sequences={"调棚": ["X", "A"], "机南": [], "存5北": []},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    direct = Phase1RollingCandidate(
        wave={
            "waveName": "phase1_roll_调棚_U0001_机南",
            "selectedSourceTrack": "调棚",
            "selectedVehicleNos": ["A"],
            "selectedBlockIds": ["U0001"],
            "waveDiagnostics": {
                "targetTrack": "机南",
                "macroTaskId": "MT::调棚::U0001",
                "macroTaskBlockIds": ["U0001"],
            },
        },
        score=(0,),
    )

    candidates = _phase1_rolling_candidates_with_route_clearance(
        runtime_stage={"vehicleGoals": [{"vehicleNo": "X", "targetTrack": "存5北"}]},
        base_candidates=(direct,),
        state=state,
        master=master,
        vehicle_meta={"X": {"vehicleLength": 10.0}, "A": {"vehicleLength": 10.0}},
    )

    clearance_waves = [
        candidate.wave
        for candidate in candidates
        if candidate.wave["selectedBlockIds"] == ["ROUTE_CLEAR::调棚"]
    ]
    assert clearance_waves
    assert clearance_waves[0]["selectedVehicleNos"] == ["X"]
    assert clearance_waves[0]["waveDiagnostics"]["runtimeFrontierStrategy"] == "rolling_route_clearance"
    assert clearance_waves[0]["waveDiagnostics"]["macroTaskId"] == "MT::调棚::U0001"


def test_phase1_route_clearance_generation_does_not_hide_non_active_macro_blockers():
    master = load_master_data(DATA_DIR)
    state = ReplayState(
        track_sequences={"调棚": ["X", "A"], "机南": [], "存5北": []},
        loco_track_name="机库",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    direct = Phase1RollingCandidate(
        wave={
            "waveName": "phase1_roll_调棚_U0001_机南",
            "selectedSourceTrack": "调棚",
            "selectedVehicleNos": ["A"],
            "selectedBlockIds": ["U0001"],
            "waveDiagnostics": {
                "targetTrack": "机南",
                "macroTaskId": "MT::调棚::U0001",
                "macroTaskBlockIds": ["U0001"],
            },
        },
        score=(0,),
    )

    candidates = _phase1_rolling_candidates_with_route_clearance(
        runtime_stage={"vehicleGoals": [{"vehicleNo": "X", "targetTrack": "存5北"}]},
        base_candidates=(direct,),
        state=state,
        master=master,
        vehicle_meta={"X": {"vehicleLength": 10.0}, "A": {"vehicleLength": 10.0}},
        active_macro_task_id="MT::other",
    )

    assert any(
        candidate.wave["selectedBlockIds"] == ["ROUTE_CLEAR::调棚"]
        for candidate in candidates
    )


def test_phase1_route_clearance_uses_executable_check_blockers():
    master = load_master_data(DATA_DIR)
    state = ReplayState(
        track_sequences={
            "存2": ["P"],
            "机棚": ["X1", "X2", "X3"],
            "洗南": ["A1", "A2", "A3"],
            "机南": ["M"],
            "调北": [],
        },
        loco_track_name="存2",
        loco_node="Z3",
        weighed_vehicle_nos=set(),
        spot_assignments={},
    )
    blocked = Phase1RollingCandidate(
        wave={
            "waveName": "phase1_roll_洗南_U0001_机棚",
            "selectedSourceTrack": "洗南",
            "selectedVehicleNos": ["A1", "A2", "A3"],
            "selectedBlockIds": ["U0001"],
            "waveDiagnostics": {"targetTrack": "机棚"},
        },
        score=(0,),
    )

    candidates = _phase1_rolling_candidates_with_route_clearance(
        runtime_stage={"vehicleGoals": []},
        base_candidates=(blocked,),
        state=state,
        master=master,
        vehicle_meta={
            "X1": {"vehicleLength": 10.0},
            "X2": {"vehicleLength": 10.0},
            "X3": {"vehicleLength": 10.0},
            "A1": {"vehicleLength": 10.0},
            "A2": {"vehicleLength": 10.0},
            "A3": {"vehicleLength": 10.0},
        },
    )

    assert any(
        candidate.wave["selectedBlockIds"] == ["ROUTE_CLEAR::机棚"]
        for candidate in candidates
    )


def test_phase1_runtime_frontier_prefers_shallow_clearance_wave():
    runtime_stage = {
        "stagePolicy": {
            "phase1WavePlans": [
                {
                    "waveName": "w1",
                    "waveType": "source_clearance",
                    "waveRole": "source_clearance",
                    "selectedSourceTrack": "存1",
                    "selectedVehicleNos": ["A1"],
                    "vehicleGoals": [{"vehicleNo": "A1", "targetTrack": "油"}],
                    "selectedBlockIds": ["S1"],
                    "requiredPredecessorIds": [],
                    "waveDiagnostics": {
                        "selectedSourceRole": "yard_storage",
                        "maxRequiredPredecessorDepth": 0,
                        "releasedDepotVehicleCount": 1,
                        "pressureGain": 1,
                        "pressureCutCounts": {"opening_release_to_ji": 1},
                    },
                },
                {
                    "waveName": "w2",
                    "waveType": "source_clearance",
                    "waveRole": "source_clearance",
                    "selectedSourceTrack": "调棚",
                    "selectedVehicleNos": ["B1", "B2"],
                    "vehicleGoals": [
                        {"vehicleNo": "B1", "targetTrack": "预修"},
                        {"vehicleNo": "B2", "targetTrack": "预修"},
                    ],
                    "selectedBlockIds": ["S2"],
                    "requiredPredecessorIds": [],
                    "waveDiagnostics": {
                        "selectedSourceRole": "work_gate",
                        "maxRequiredPredecessorDepth": 2,
                        "releasedDepotVehicleCount": 3,
                        "pressureGain": 4,
                        "pressureCutCounts": {"opening_release_to_ji": 1, "work_to_ji": 1},
                    },
                },
            ]
        }
    }

    wave = _build_phase1_runtime_frontier_wave(runtime_stage=runtime_stage)

    assert wave is not None
    assert wave["waveName"] == "w1"
    assert wave["selectedSourceTrack"] == "存1"


def test_phase1_receiving_storage_clearance_can_be_admitted():
    admissions = _build_phase1_source_admissions(
        packages=[
            Phase1LayoutPackage(
                package_id="P1",
                chain_id="CHAIN::存5北",
                package_kind="local_finish",
                source_track="存5北",
                vehicle_nos=("V1", "V2"),
                total_length_m=28.0,
                target_track="预修",
                target_source="PHASE1_LOCAL_FINISH",
                final_family="预修",
                min_spot_priority=999,
                source_order_start=1,
                source_order_end=2,
                buffer_preference=tuple(),
                uses_buffer=False,
                pressure_cut="storage_to_ji",
                reason_tags=tuple(),
                execution_layer="L2_REQUIRED_CLEAR",
                complexity_cost=2,
                source_chain_role="receiving_storage",
                is_required_for_backbone=True,
                segment_role="cleanup",
                source_segment_index=1,
                source_segment_count=1,
                source_total_vehicle_count=2,
                requires_previous_segment=False,
            )
        ]
    )

    assert len(admissions) == 1
    assert admissions[0].required_clearance_gain_units > 0
    assert admissions[0].admission_tier == "clearance_required"


def test_phase1_wave_a_includes_required_storage_clearance_prefix():
    storage_block = Phase1Block(
        block_id="S1",
        source_track="存5北",
        block_type="tail_finish",
        vehicle_nos=("V1", "V2"),
        total_length_m=28.0,
        target_track="预修",
        target_source="PHASE1_LOCAL_FINISH",
        uses_buffer=False,
        buffer_preference=tuple(),
        source_order_start=1,
        source_order_end=2,
        final_family="预修",
        phase3_rank_key=(0, 0, 0, 1),
        released_depot_vehicle_count=0,
        released_finish_vehicle_count=10,
        required_predecessor_ids=tuple(),
        layout_role="cleanup",
        topology_zone="receiving",
        throat_group="G_STORAGE",
        pressure_gain=5,
        coupling_degree=1,
    )
    source_plan = SourceTrackPlan(
        source_track="存5北",
        blocks=(storage_block,),
        reachable_depot_vehicle_nos=tuple(),
        reachable_finish_vehicle_nos=("V1", "V2"),
        cun4_clear_required=False,
        buffer_demand_m=0.0,
        source_priority_score=(9,),
    )
    backbone_plan = Phase1BackbonePlan(
        selected_block_ids=tuple(),
        selected_source_tracks=tuple(),
        reserved_buffer_by_track={},
        selected_buffer_assignment={},
        target_rank_by_vehicle={},
        layout_template_name="t",
        opened_buffer_tracks=tuple(),
    )

    selected_ids = _phase1_wave_a_block_ids(
        source_plans=[source_plan],
        backbone_plan=backbone_plan,
    )

    assert "S1" in selected_ids


def test_phase1_wave_a_selects_minimum_storage_enabling_prefix():
    blocks = (
        Phase1Block(
            block_id="S1",
            source_track="存5北",
            block_type="tail_finish",
            vehicle_nos=("V1", "V2"),
            total_length_m=28.0,
            target_track="预修",
            target_source="PHASE1_LOCAL_FINISH",
            uses_buffer=False,
            buffer_preference=tuple(),
            source_order_start=1,
            source_order_end=2,
            final_family="预修",
            phase3_rank_key=(0, 0, 0, 1),
            released_depot_vehicle_count=0,
            released_finish_vehicle_count=14,
            required_predecessor_ids=tuple(),
            layout_role="cleanup",
            topology_zone="receiving",
            throat_group="G_STORAGE",
            pressure_gain=5,
            coupling_degree=1,
        ),
        Phase1Block(
            block_id="S2",
            source_track="存5北",
            block_type="tail_finish",
            vehicle_nos=("V3", "V4"),
            total_length_m=28.0,
            target_track="预修",
            target_source="PHASE1_LOCAL_FINISH",
            uses_buffer=False,
            buffer_preference=tuple(),
            source_order_start=3,
            source_order_end=4,
            final_family="预修",
            phase3_rank_key=(0, 0, 0, 3),
            released_depot_vehicle_count=0,
            released_finish_vehicle_count=12,
            required_predecessor_ids=("S1",),
            layout_role="cleanup",
            topology_zone="receiving",
            throat_group="G_STORAGE",
            pressure_gain=4,
            coupling_degree=1,
        ),
    )
    source_plan = SourceTrackPlan(
        source_track="存5北",
        blocks=blocks,
        reachable_depot_vehicle_nos=tuple(),
        reachable_finish_vehicle_nos=("V1", "V2", "V3", "V4"),
        cun4_clear_required=False,
        buffer_demand_m=0.0,
        source_priority_score=(9,),
    )
    backbone_plan = Phase1BackbonePlan(
        selected_block_ids=tuple(),
        selected_source_tracks=tuple(),
        reserved_buffer_by_track={},
        selected_buffer_assignment={},
        target_rank_by_vehicle={},
        layout_template_name="t",
        opened_buffer_tracks=tuple(),
    )

    selected_ids = _phase1_wave_a_block_ids(
        source_plans=[source_plan],
        backbone_plan=backbone_plan,
    )

    assert selected_ids == ("S1",)


def test_phase1_wave_plans_keep_storage_clearance_at_frontier_granularity():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="存5北", order="1", vehicle_no="A1", target_track="调棚", vehicle_length=22.0),
            _vehicle(track_name="存5北", order="2", vehicle_no="A2", target_track="调棚", vehicle_length=22.0),
            _vehicle(track_name="存5北", order="3", vehicle_no="A3", target_track="调棚", vehicle_length=22.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    waves = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1WavePlans"]
    clearance_waves = [
        wave
        for wave in waves
        if str(wave.get("selectedSourceTrack") or "") == "存5北"
        and str(wave.get("waveType") or "") == "source_clearance"
    ]

    assert len(clearance_waves) >= 1
    assert all(len(list(wave.get("selectedBlockIds") or [])) == 1 for wave in clearance_waves)


def test_phase1_runtime_frontier_skips_wave_with_unfinished_predecessor():
    runtime_stage = {
        "stagePolicy": {
            "phase1WavePlans": [
                {
                    "waveName": "w_after",
                    "waveType": "source_clearance",
                    "waveRole": "source_clearance",
                    "selectedSourceTrack": "存5北",
                    "selectedBlockIds": ["S2"],
                    "requiredPredecessorIds": ["S1"],
                    "selectedVehicleNos": ["B1"],
                    "vehicleGoals": [{"vehicleNo": "B1", "targetTrack": "调棚"}],
                    "waveDiagnostics": {
                        "releasedDepotVehicleCount": 4,
                        "pressureGain": 5,
                    },
                },
                {
                    "waveName": "w_before",
                    "waveType": "source_clearance",
                    "waveRole": "source_clearance",
                    "selectedSourceTrack": "存5北",
                    "selectedBlockIds": ["S1"],
                    "requiredPredecessorIds": [],
                    "selectedVehicleNos": ["A1"],
                    "vehicleGoals": [{"vehicleNo": "A1", "targetTrack": "调棚"}],
                    "waveDiagnostics": {
                        "releasedDepotVehicleCount": 1,
                        "pressureGain": 1,
                    },
                },
            ],
            "phase1Diagnostics": {
                "phase1Blocks": [
                    {"blockId": "S1", "selectedFinish": True, "selectedBackbone": False},
                    {"blockId": "S2", "selectedFinish": True, "selectedBackbone": False},
                ]
            },
        }
    }

    wave = _build_phase1_runtime_frontier_wave(runtime_stage=runtime_stage)

    assert wave is not None
    assert wave["waveName"] == "w_before"


def test_phase1_compiles_depot_cars_and_clears_cun4_and_ji():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="修2"),
            _vehicle(track_name="存4北", order="1", vehicle_no="C", target_track="存1"),
            _vehicle(track_name="机南", order="1", vehicle_no="D", target_track="存2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]

    assert goals["A"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["B"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["A"]["targetTrack"] in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert goals["B"]["targetTrack"] in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert goals["C"]["targetSource"] == "PHASE1_CLEAR_CUN4"
    assert goals["C"]["targetTrack"] == "存1"
    assert goals["D"]["targetSource"] == "PHASE1_CLEAR_JI"
    assert goals["D"]["targetTrack"] in {"存1", "存2", "存3", "存5南", "存5北", "调北", "预修", "调棚"}
    assert diagnostics["depotCompileRatio"] == 1.0
    assert diagnostics["cun4ClearRatio"] == 1.0
    assert diagnostics["jiPurityRatio"] == 1.0
    assert diagnostics["remainingCun4VehicleNos"] == []
    assert diagnostics["remainingJiNonDepotVehicleNos"] == []
    assert diagnostics["selectedExecutionLayerCounts"]["L1_BACKBONE"] == 2
    assert diagnostics["selectedExecutionLayerCounts"]["L2_REQUIRED_CLEAR"] == 2


def test_phase1_staging_contract_covers_all_depot_cars_and_keeps_ji_pure():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="存1", order="1", vehicle_no="B", target_track="修2"),
            _vehicle(track_name="机棚", order="1", vehicle_no="C", target_track="修3"),
            _vehicle(track_name="机南", order="1", vehicle_no="D", target_track="存2"),
            _vehicle(track_name="机北1", order="1", vehicle_no="E", target_track="调棚"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]

    assert goals["A"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["B"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["A"]["targetTrack"] in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert goals["B"]["targetTrack"] in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert goals["C"]["targetSource"] == "HOLD_CURRENT"
    assert goals["D"]["targetSource"] == "PHASE1_CLEAR_JI"
    assert goals["E"]["targetSource"] == "PHASE1_CLEAR_JI"
    assert goals["D"]["targetTrack"] not in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert goals["E"]["targetTrack"] not in {"机南", "机棚", "机北1", "机北2", "机北3"}
    assert diagnostics["depotDemandVehicleCount"] == 2
    assert diagnostics["depotCompiledVehicleCount"] == 2
    assert diagnostics["uncompiledDepotVehicleNos"] == []
    assert diagnostics["remainingJiNonDepotVehicleNos"] == []


def test_phase1_uses_jibei1_as_legal_buffer_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="轮", vehicle_length=50.0),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="轮", vehicle_length=50.0),
            _vehicle(track_name="调棚", order="1", vehicle_no="C", target_track="修3", vehicle_length=50.0),
            _vehicle(track_name="预修", order="1", vehicle_no="D", target_track="修4", vehicle_length=50.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    buffer_tracks = {
        item["targetTrack"]
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
        if item["targetSource"] == "PHASE1_BACKBONE_PLACE"
    }

    assert "机北1" in buffer_tracks


def test_phase1_uses_split_source_budgets_for_hot_and_storage_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1", vehicle_length=38.0),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="修1", vehicle_length=38.0),
            _vehicle(track_name="调棚", order="1", vehicle_no="C", target_track="修2", vehicle_length=38.0),
            _vehicle(track_name="预修", order="1", vehicle_no="D", target_track="修2", vehicle_length=38.0),
            _vehicle(track_name="存5北", order="1", vehicle_no="E", target_track="修3", vehicle_length=38.0),
            _vehicle(track_name="存5南", order="1", vehicle_no="F", target_track="修4", vehicle_length=38.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]

    assert diagnostics["depotDemandVehicleCount"] == 6
    assert diagnostics["depotCompiledVehicleCount"] == 6
    assert diagnostics["depotCompileRatio"] == 1.0
    assert diagnostics["uncompiledDepotVehicleNos"] == []
    assert diagnostics["selectedHotSourceTrackCount"] <= 4
    assert diagnostics["selectedStorageSourceTrackCount"] <= 2
    assert diagnostics["selectedHotSourceTracks"] == ["油", "洗南", "调棚", "预修"]
    assert diagnostics["selectedStorageSourceTracks"] == ["存5北", "存5南"]


def test_phase1_puts_factory_repair_deeper_than_depot_repair():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="DUAN", target_track="修1", repair_process="段修"),
            _vehicle(track_name="调棚", order="2", vehicle_no="CHANG", target_track="修1", repair_process="厂修"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    target_ranks = workflow_payload["workflowStages"][0]["stagePolicy"]["packageTargetRanks"]

    assert target_ranks["DUAN"] < target_ranks["CHANG"]


def test_phase1_uses_temp_repark_for_non_depot_blocker_before_depot_batch():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="预修", order="1", vehicle_no="A", target_track="预修"),
            _vehicle(track_name="预修", order="2", vehicle_no="B", target_track="修1"),
            _vehicle(track_name="预修", order="3", vehicle_no="C", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }

    assert goals["A"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_WORK"
    assert goals["A"]["targetTrack"] in {"存1", "存2", "存3", "存5南", "存5北", "调北", "调棚"}
    assert goals["A"]["targetTrack"] not in {"存4北", "存4南"}
    assert goals["B"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["C"]["targetSource"] == "PHASE1_BACKBONE_PLACE"


def test_phase1_keeps_need_weigh_vehicle_as_singleton_package():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="预修", order="1", vehicle_no="W1", target_track="修1", vehicle_attributes="称重"),
            _vehicle(track_name="预修", order="2", vehicle_no="N1", target_track="修1"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    packages = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]["taskPackages"]
    weight_packages = [package for package in packages if "W1" in package["vehicleNos"]]
    normal_packages = [package for package in packages if "N1" in package["vehicleNos"]]

    assert len(weight_packages) == 1
    assert weight_packages[0]["vehicleNos"] == ["W1"]
    assert "need_weigh_singleton" in weight_packages[0]["reasonTags"]
    assert weight_packages[0]["executionLayer"] == "L1_BACKBONE"
    assert len(normal_packages) == 1
    assert normal_packages[0]["vehicleNos"] == ["N1"]


def test_phase1_caps_heavy_vehicle_package_size():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="预修", order="1", vehicle_no="H1", target_track="修1", vehicle_attributes="重车", vehicle_length=13.0),
            _vehicle(track_name="预修", order="2", vehicle_no="H2", target_track="修1", vehicle_attributes="重车", vehicle_length=13.0),
            _vehicle(track_name="预修", order="3", vehicle_no="H3", target_track="修1", vehicle_attributes="重车", vehicle_length=13.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    packages = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]["taskPackages"]
    heavy_packages = [package for package in packages if package["usesBuffer"]]

    assert [package["vehicleNos"] for package in heavy_packages] == [["H1", "H2"], ["H3"]]
    assert all("heavy_cap" in package["reasonTags"] for package in heavy_packages)
    assert all(package["executionLayer"] == "L1_BACKBONE" for package in heavy_packages)
    assert all(package["selected"] for package in heavy_packages)
    assert len({package["bufferTrack"] for package in heavy_packages}) == 1


def test_phase1_block_layout_compiles_same_source_across_minimal_tracks():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="预修", order="1", vehicle_no="P1", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="2", vehicle_no="P2", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="3", vehicle_no="P3", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="4", vehicle_no="P4", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="5", vehicle_no="P5", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="6", vehicle_no="P6", target_track="修1", vehicle_length=13.0),
            _vehicle(track_name="预修", order="7", vehicle_no="P7", target_track="修1", vehicle_length=13.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
    heavy_packages = [
        package
        for package in diagnostics["taskPackages"]
        if package["usesBuffer"]
    ]
    selected_packages = [package for package in heavy_packages if package["selected"]]
    selected_tracks = {
        package["bufferTrack"]
        for package in selected_packages
    }

    assert diagnostics["depotCompiledVehicleCount"] == 7
    assert diagnostics["depotDemandVehicleCount"] == 7
    assert len(selected_packages) == 3
    assert selected_tracks == {"机南", "机棚"}
    assert all(package["sourceTrack"] == "预修" for package in selected_packages)


def test_phase1_skips_optional_close_door_cun4_move_from_main_backbone():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="存5北", order="1", vehicle_no="CD1", target_track="存4北", vehicle_attributes="关门车"),
            _vehicle(track_name="存5北", order="2", vehicle_no="N1", target_track="存4北"),
            _vehicle(track_name="存5北", order="3", vehicle_no="N2", target_track="存4北"),
            _vehicle(track_name="存5北", order="4", vehicle_no="N3", target_track="存4北"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    packages = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]["taskPackages"]
    close_door_packages = [package for package in packages if "CD1" in package["vehicleNos"]]

    assert close_door_packages == []


def test_phase1_backbone_contract_covers_all_depot_staging_vehicles():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A1", target_track="修1"),
            _vehicle(track_name="油", order="1", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="调棚", order="1", vehicle_no="A3", target_track="修2"),
            _vehicle(track_name="预修", order="1", vehicle_no="A4", target_track="修2"),
            _vehicle(track_name="存5北", order="1", vehicle_no="A5", target_track="修3"),
            _vehicle(track_name="存1", order="1", vehicle_no="A6", target_track="修3"),
            _vehicle(track_name="存2", order="1", vehicle_no="A7", target_track="修4"),
            _vehicle(track_name="存3", order="1", vehicle_no="A8", target_track="修4"),
            _vehicle(track_name="抛", order="1", vehicle_no="A9", target_track="轮"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]

    assert diagnostics["depotDemandVehicleCount"] == 9
    assert diagnostics["depotCompiledVehicleCount"] == 9
    assert diagnostics["depotCompileRatio"] == 1.0
    assert diagnostics["uncompiledDepotVehicleNos"] == []
    assert diagnostics["deferredVehicleNos"] == []


def test_phase1_includes_low_yield_storage_source_when_it_has_depot_staging_vehicle():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="存5北", order="1", vehicle_no="B1", target_track="存1"),
            _vehicle(track_name="存5北", order="2", vehicle_no="B2", target_track="存1"),
            _vehicle(track_name="存5北", order="3", vehicle_no="B3", target_track="存2"),
            _vehicle(track_name="存5北", order="4", vehicle_no="D1", target_track="修3"),
            _vehicle(track_name="预修", order="1", vehicle_no="D2", target_track="修1"),
            _vehicle(track_name="调棚", order="1", vehicle_no="D3", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }

    assert goals["D2"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["D3"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["D1"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["B1"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_YARD"
    assert goals["B1"]["targetTrack"] in {"存1", "存2", "存3", "存5北", "存5南"}
    assert goals["B2"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_YARD"
    assert goals["B2"]["targetTrack"] in {"存1", "存2", "存3", "存5北", "存5南"}
    assert goals["B3"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_YARD"
    assert goals["B3"]["targetTrack"] in {"存1", "存2", "存3", "存5北", "存5南"}
    source_summaries = {
        row["sourceTrack"]: row
        for row in diagnostics["sourceOpenSummaries"]
    }
    assert source_summaries["存5北"]["openingScore"] < 0
    assert source_summaries["存5北"]["selectedForBackbone"] is True
    assert source_summaries["存5北"]["admissionDecision"] == "primary"
    assert source_summaries["存5北"]["rejectionReason"] is None
    assert diagnostics["uncompiledDepotVehicleNos"] == []
    assert diagnostics["selectedCleanupBySource"]["存5北"]["requiredTempReparkCount"] == 2


def test_phase1_block_layout_preserves_full_compile_without_source_mixing():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="油", order="1", vehicle_no="Y1", target_track="修1"),
            _vehicle(track_name="油", order="2", vehicle_no="Y2", target_track="修1"),
            _vehicle(track_name="油", order="3", vehicle_no="Y3", target_track="修1"),
            _vehicle(track_name="预修", order="1", vehicle_no="P1", target_track="修2"),
            _vehicle(track_name="预修", order="2", vehicle_no="P2", target_track="修2"),
            _vehicle(track_name="预修", order="3", vehicle_no="P3", target_track="修2"),
            _vehicle(track_name="预修", order="4", vehicle_no="P4", target_track="修2"),
            _vehicle(track_name="预修", order="5", vehicle_no="P5", target_track="修2"),
            _vehicle(track_name="预修", order="6", vehicle_no="P6", target_track="修2"),
            _vehicle(track_name="预修", order="7", vehicle_no="P7", target_track="修2"),
            _vehicle(track_name="预修", order="8", vehicle_no="P8", target_track="修2"),
            _vehicle(track_name="预修", order="9", vehicle_no="P9", target_track="修2"),
            _vehicle(track_name="预修", order="10", vehicle_no="P10", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }

    assert diagnostics["depotDemandVehicleCount"] == 13
    assert diagnostics["depotCompiledVehicleCount"] == 13
    assert diagnostics["depotCompileRatio"] == 1.0
    assert goals["P1"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert goals["P10"]["targetSource"] == "PHASE1_BACKBONE_PLACE"
    assert diagnostics["bufferSourceTracks"]["机棚"] == ["预修"]
    assert diagnostics["bufferSourceTracks"]["机北1"] == ["预修"]
    assert diagnostics["bufferSourceTracks"]["机北2"] == ["预修"]


def test_phase1_avoids_mixing_sources_on_same_buffer_when_empty_track_exists():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗北", order="1", vehicle_no="XB1", target_track="修1", vehicle_length=18.0),
            _vehicle(track_name="洗北", order="2", vehicle_no="XB2", target_track="修1", vehicle_length=18.0),
            _vehicle(track_name="洗南", order="1", vehicle_no="XN1", target_track="修2", vehicle_length=18.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }

    assert goals["XB1"]["targetTrack"] == goals["XB2"]["targetTrack"]
    assert goals["XN1"]["targetTrack"] != goals["XB1"]["targetTrack"]
    assert diagnostics["mixedBufferTrackCount"] == 0


def test_phase1_reuses_same_temp_track_for_same_source_cleanup_when_capacity_allows():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A", target_track="调棚", vehicle_length=20.0),
            _vehicle(track_name="调棚", order="2", vehicle_no="B", target_track="调棚", vehicle_length=20.0),
            _vehicle(track_name="调棚", order="3", vehicle_no="C", target_track="修3", vehicle_length=20.0),
            _vehicle(track_name="调棚", order="4", vehicle_no="D", target_track="修4", vehicle_length=20.0),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    diagnostics = workflow_payload["workflowStages"][0]["stagePolicy"]["phase1Diagnostics"]
    goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }

    assert goals["A"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_WORK"
    assert goals["B"]["targetSource"] == "PHASE1_BLOCKER_BUCKET_WORK"
    assert goals["A"]["targetTrack"] == goals["B"]["targetTrack"]
    assert diagnostics["selectedTempTracksBySource"]["调棚"] == [goals["A"]["targetTrack"]]


def test_phase2_builds_cun4_staging_and_final_segments():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="修1", order="1", vehicle_no="B", target_track="修1"),
            _vehicle(track_name="修1库外", order="1", vehicle_no="C", target_track="存1"),
            _vehicle(track_name="修2", order="1", vehicle_no="D", target_track="存4北"),
            _vehicle(track_name="修3", order="1", vehicle_no="E", target_track="存4北"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase1_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][0]["vehicleGoals"]
    }
    phase2_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][1]["vehicleGoals"]
    }
    phase2_diag = workflow_payload["workflowStages"][1]["stagePolicy"]["phase2Diagnostics"]
    phase2_policy = workflow_payload["workflowStages"][1]["stagePolicy"]

    assert phase2_goals["A"]["targetTrack"] == phase1_goals["A"]["targetTrack"]
    assert phase2_goals["A"]["targetSource"] == "PHASE2_HOLD_PHASE1_BACKBONE"
    assert phase2_goals["B"]["targetTrack"] == "修1"
    assert phase2_goals["B"]["targetSource"] == FIXED_DEPOT_RESIDENT_SOURCE
    assert phase2_goals["B"]["targetMode"] == "SNAPSHOT"
    assert phase2_goals["C"]["targetTrack"] == "存4北"
    assert phase2_goals["C"]["targetSource"] == "PHASE2_TRANSFER_TO_CUN4"
    assert phase2_goals["C"]["targetMode"] == "AREA"
    assert phase2_goals["D"]["targetTrack"] == "存4北"
    assert phase2_goals["D"]["targetSource"] == "PHASE2_TRANSFER_TO_CUN4"
    assert phase2_goals["E"]["targetTrack"] == "存4北"
    assert phase2_goals["E"]["targetSource"] == "PHASE2_TRANSFER_TO_CUN4"
    assert phase2_policy["depotStayVehicles"] == ["B"]
    assert phase2_policy["fixedDepotResidentVehicleNos"] == ["B"]
    assert phase2_policy["fixedDepotResidentVehicles"] == ["B"]
    assert phase2_policy["depotOutboundVehicles"] == ["C"]
    assert phase2_policy["cun4FinalVehicles"] == ["D", "E"]
    assert [layer["sourceTrack"] for layer in phase2_policy["phase2TrackLayers"]] == ["修3", "修2", "修1库外"]
    assert phase2_policy["phase2ExecutionPlan"]["sourceTracks"] == ["修1库外", "修3", "修2"]
    assert phase2_policy["phase2ExecutionPlan"]["phase3ClearanceVehicleNos"] == ["E", "D"]
    assert phase2_policy["phase2ExecutionPlan"]["transferVehicleNos"] == ["C", "E", "D"]
    assert phase2_policy["phase2ExecutionPlan"]["deferredTailVehicleNos"] == []
    assert phase2_diag["depotStayVehicleNos"] == ["B"]
    assert phase2_diag["depotOutboundVehicleNos"] == ["C"]
    assert phase2_diag["outboundGroupCount"] == 3
    assert phase2_diag["trackLayerCount"] == 3


def test_fixed_depot_resident_is_not_cleared_or_reallocated():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="修2", order="1", vehicle_no="FIX", target_track="修2"),
            _vehicle(track_name="修3", order="1", vehicle_no="C4", target_track="存4北"),
            _vehicle(track_name="修1库外", order="1", vehicle_no="OUT", target_track="存1"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_policy = workflow_payload["workflowStages"][1]["stagePolicy"]
    phase2_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][1]["vehicleGoals"]
    }
    phase3_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][2]["vehicleGoals"]
    }

    assert phase2_policy["fixedDepotResidentVehicleNos"] == ["FIX"]
    assert phase2_policy["fixedDepotResidentVehicles"] == ["FIX"]
    assert phase2_goals["FIX"]["targetSource"] == FIXED_DEPOT_RESIDENT_SOURCE
    assert phase2_goals["FIX"]["targetTrack"] == "修2"
    assert phase3_goals["FIX"]["targetSource"] == FIXED_DEPOT_RESIDENT_SOURCE
    assert phase3_goals["FIX"]["targetTrack"] == "修2"
    execution_plan = phase2_policy["phase2ExecutionPlan"]
    assert "FIX" not in execution_plan["transferVehicleNos"]
    assert "FIX" not in execution_plan["phase3ClearanceVehicleNos"]


def test_depot_random_goal_on_current_repair_slot_is_fixed_depot_resident():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="修2", order="1", vehicle_no="RANDOM", target_track="大库"),
            _vehicle(track_name="修1库外", order="1", vehicle_no="OUT", target_track="存1"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_policy = workflow_payload["workflowStages"][1]["stagePolicy"]
    phase3_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][2]["vehicleGoals"]
    }

    assert phase2_policy["fixedDepotResidentVehicleNos"] == ["RANDOM"]
    assert phase2_policy["fixedDepotResidentVehicles"] == ["RANDOM"]
    assert phase3_goals["RANDOM"].get("targetSource") == FIXED_DEPOT_RESIDENT_SOURCE


def test_phase2_runtime_rebuild_pulls_depot_stay_anchor_when_it_blocks_outbound():
    stage_payload = {
        "vehicleInfo": [
            _vehicle(track_name="修2", order="1", vehicle_no="A", target_track="存4北"),
            _vehicle(track_name="修2", order="2", vehicle_no="B", target_track="轮"),
            _vehicle(track_name="修2", order="3", vehicle_no="C", target_track="存4北"),
            _vehicle(track_name="修2", order="4", vehicle_no="D", target_track="油", repair_process="厂修"),
        ],
        "stagePolicy": {
            "depotStayVehicles": ["B"],
            "cun4FinalVehicles": ["A", "C"],
            "depotOutboundVehicles": ["D"],
            "phase2OutboundGroups": [
                {
                    "groupId": "G1",
                    "groupKind": "CUN4_FINAL",
                    "vehicleNos": ["A"],
                    "currentTrack": "修2",
                    "sourceOrderStart": 1,
                    "finalTargetTrack": "存4北",
                    "finalFamily": "存4北",
                    "repairProcessProfile": ["段修"],
                    "totalLengthM": 14.3,
                },
                {
                    "groupId": "G2",
                    "groupKind": "CUN4_FINAL",
                    "vehicleNos": ["C"],
                    "currentTrack": "修2",
                    "sourceOrderStart": 3,
                    "finalTargetTrack": "存4北",
                    "finalFamily": "存4北",
                    "repairProcessProfile": ["段修"],
                    "totalLengthM": 14.3,
                },
                {
                    "groupId": "G3",
                    "groupKind": "DEPOT_OUTBOUND",
                    "vehicleNos": ["D"],
                    "currentTrack": "修2",
                    "sourceOrderStart": 4,
                    "finalTargetTrack": "油",
                    "finalFamily": "油",
                    "repairProcessProfile": ["厂修"],
                    "totalLengthM": 14.3,
                },
            ],
        },
    }

    policy = rebuild_phase2_execution_policy_for_runtime(
        stage_payload=stage_payload,
        track_sequences={"修2": ["A", "B", "C", "D"]},
    )

    assert policy is not None
    assert policy["transferVehicleNos"] == ["A", "B", "C", "D"]
    assert policy["mustPullVehicleNos"] == ["D"]
    assert policy["predecessorUnlockVehicleNos"] == ["A", "B", "C"]
    assert policy["deferredTailVehicleNos"] == []
    assert policy["collectionBatches"] == [["A", "B", "C", "D"]]
    assert [layer["vehicleNos"] for layer in policy["trackLayers"]] == [["A"], ["B"], ["C"], ["D"]]
    assert policy["executionDiagnostics"]["runtimeBlockedTails"] == []


def test_phase2_runtime_rebuild_does_not_pull_fixed_depot_resident_anchor():
    stage_payload = {
        "vehicleInfo": [
            _vehicle(track_name="修2", order="1", vehicle_no="FIX", target_track="修2"),
            _vehicle(track_name="修2", order="2", vehicle_no="C", target_track="存4北"),
            _vehicle(track_name="修2", order="3", vehicle_no="D", target_track="油", repair_process="厂修"),
        ],
        "stagePolicy": {
            "fixedDepotResidentVehicleNos": ["FIX"],
            "depotStayVehicles": ["FIX"],
            "cun4FinalVehicles": ["C"],
            "depotOutboundVehicles": ["D"],
            "phase2OutboundGroups": [
                {
                    "groupId": "G1",
                    "groupKind": "CUN4_FINAL",
                    "vehicleNos": ["C"],
                    "currentTrack": "修2",
                    "sourceOrderStart": 2,
                    "finalTargetTrack": "存4北",
                    "finalFamily": "存4北",
                    "repairProcessProfile": ["段修"],
                    "totalLengthM": 14.3,
                },
                {
                    "groupId": "G2",
                    "groupKind": "DEPOT_OUTBOUND",
                    "vehicleNos": ["D"],
                    "currentTrack": "修2",
                    "sourceOrderStart": 3,
                    "finalTargetTrack": "油",
                    "finalFamily": "油",
                    "repairProcessProfile": ["厂修"],
                    "totalLengthM": 14.3,
                },
            ],
        },
    }

    policy = rebuild_phase2_execution_policy_for_runtime(
        stage_payload=stage_payload,
        track_sequences={"修2": ["FIX", "C", "D"]},
    )

    assert policy is not None
    assert policy["transferVehicleNos"] == []
    assert policy["predecessorUnlockVehicleNos"] == []
    assert policy["deferredTailVehicleNos"] == ["C", "D"]
    assert policy["executionDiagnostics"]["runtimeBlockedTails"] == [
        {
            "sourceTrack": "修2",
            "anchorVehicleNo": "FIX",
            "blockedTailVehicleNos": ["C", "D"],
        }
    ]


def test_phase2_splits_required_outbound_into_legal_collection_batches():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(track_name="修1库外", order=str(index), vehicle_no=f"O{index:02d}", target_track="存1")
        for index in range(1, 22)
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    execution_plan = workflow_payload["workflowStages"][1]["stagePolicy"]["phase2ExecutionPlan"]

    assert execution_plan["transferVehicleNos"] == [f"O{index:02d}" for index in range(1, 22)]
    assert execution_plan["deferredTailVehicleNos"] == []
    assert execution_plan["collectionBatches"] == [
        [f"O{index:02d}" for index in range(1, 14)],
        [f"O{index:02d}" for index in range(14, 22)],
    ]


def test_phase2_splits_required_outbound_for_heavy_equivalent_limit():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(
            track_name="修1库外",
            order=str(index),
            vehicle_no=f"H{index:02d}",
            target_track="存1",
            vehicle_attributes="重车" if index in {1, 2} else "",
        )
        for index in range(1, 16)
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    execution_plan = workflow_payload["workflowStages"][1]["stagePolicy"]["phase2ExecutionPlan"]

    assert execution_plan["collectionBatches"] == [
        [f"H{index:02d}" for index in range(1, 14)],
        ["H14", "H15"],
    ]


def test_phase2_splits_required_outbound_when_close_door_precedes_heavy():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(
            track_name="修1库外",
            order="1",
            vehicle_no="CLOSE",
            target_track="存1",
            vehicle_attributes="关门车",
        ),
        _vehicle(
            track_name="修1库外",
            order="2",
            vehicle_no="HEAVY",
            target_track="存1",
            vehicle_attributes="重车",
        ),
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    execution_plan = workflow_payload["workflowStages"][1]["stagePolicy"]["phase2ExecutionPlan"]

    assert execution_plan["collectionBatches"] == [["CLOSE"], ["HEAVY"]]


def test_phase2_splits_required_outbound_for_weigh_tail_constraint():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(
            track_name="修1库外",
            order="1",
            vehicle_no="W1",
            target_track="存1",
            vehicle_attributes="称重",
        ),
        _vehicle(track_name="修1库外", order="2", vehicle_no="N1", target_track="存1"),
        _vehicle(
            track_name="修1库外",
            order="3",
            vehicle_no="W2",
            target_track="存1",
            vehicle_attributes="称重",
        ),
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    execution_plan = workflow_payload["workflowStages"][1]["stagePolicy"]["phase2ExecutionPlan"]

    assert execution_plan["collectionBatches"] == [["W1"], ["N1", "W2"]]


def test_phase2_splits_collection_batches_for_l1_transfer_length_limit():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(
            track_name="修1库外",
            order=str(index),
            vehicle_no=f"L{index}",
            target_track="存1",
            vehicle_length=50.0,
        )
        for index in range(1, 6)
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_payload = {
        "trackInfo": workflow_payload["trackInfo"],
        "initialVehicleInfo": workflow_payload["initialVehicleInfo"],
        "locoTrackName": workflow_payload["locoTrackName"],
        "workflowStages": workflow_payload["workflowStages"][:2],
    }
    result = solve_workflow(
        master,
        phase2_payload,
        solver="beam",
        heuristic_weight=1.0,
        beam_width=4,
        time_budget_ms=10_000,
        use_validation_recovery=True,
    )

    phase2_view = result.stages[1].view
    phase2_plan = result.stages[1].input_payload["stagePolicy"]["phase2ExecutionPlan"]
    detach_hooks = [
        hook
        for hook in phase2_view.hook_plan
        if hook.action_type == "DETACH" and hook.target_track == "存4北"
    ]

    assert phase2_view.summary.is_valid is True
    assert phase2_plan["collectionBatches"] == [["L1", "L2", "L3"], ["L4", "L5"]]
    assert [hook.vehicle_nos for hook in detach_hooks] == phase2_plan["collectionBatches"]


def test_phase2_runner_executes_split_batches_and_leaves_loco_empty():
    master = load_master_data(DATA_DIR)
    vehicles = [
        _vehicle(track_name="修1库外", order=str(index), vehicle_no=f"O{index:02d}", target_track="存1")
        for index in range(1, 22)
    ]
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": vehicles,
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_payload = {
        "trackInfo": workflow_payload["trackInfo"],
        "initialVehicleInfo": workflow_payload["initialVehicleInfo"],
        "locoTrackName": workflow_payload["locoTrackName"],
        "workflowStages": workflow_payload["workflowStages"][:2],
    }
    result = solve_workflow(
        master,
        phase2_payload,
        solver="beam",
        heuristic_weight=1.0,
        beam_width=4,
        time_budget_ms=10_000,
        use_validation_recovery=True,
    )

    phase2_view = result.stages[1].view
    phase2_plan = result.stages[1].input_payload["stagePolicy"]["phase2ExecutionPlan"]
    detach_hooks = [
        hook
        for hook in phase2_view.hook_plan
        if hook.action_type == "DETACH" and hook.target_track == "存4北"
    ]

    assert phase2_view.summary.is_valid is True
    assert phase2_plan["collectionBatches"] == [
        [f"O{index:02d}" for index in range(1, 14)],
        [f"O{index:02d}" for index in range(14, 22)],
    ]
    assert [hook.vehicle_nos for hook in detach_hooks] == phase2_plan["collectionBatches"]
    assert set(phase2_view.steps[-1].track_sequences["存4北"]) == {
        f"O{index:02d}" for index in range(1, 22)
    }
    assert phase2_view.steps[-1].loco_carry_vehicle_nos == []


def test_phase2_single_wave_prefers_global_outbound_prefix_before_cun4_block():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="修4", order="1", vehicle_no="C4A", target_track="存4北"),
            _vehicle(track_name="修4", order="2", vehicle_no="O4A", target_track="油"),
            _vehicle(track_name="修3", order="1", vehicle_no="C3A", target_track="存4北"),
            _vehicle(track_name="修3", order="2", vehicle_no="O3A", target_track="调棚"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_policy = workflow_payload["workflowStages"][1]["stagePolicy"]
    wave_plans = phase2_policy["phase2WavePlans"]

    assert len(wave_plans) == 2
    export_wave = wave_plans[-1]
    assert export_wave["waveRole"] == "EXPORT"
    assert export_wave["waveDiagnostics"]["outboundAttachUnits"] == [
        {"sourceTrack": "修4", "vehicleNos": ["O4A"]},
        {"sourceTrack": "修3", "vehicleNos": ["O3A"]},
    ]
    assert export_wave["waveDiagnostics"]["cun4AttachUnits"] == [
        {"sourceTrack": "修4", "vehicleNos": ["C4A"]},
        {"sourceTrack": "修3", "vehicleNos": ["C3A"]},
    ]

    phase2_payload = {
        "trackInfo": workflow_payload["trackInfo"],
        "initialVehicleInfo": workflow_payload["initialVehicleInfo"],
        "locoTrackName": workflow_payload["locoTrackName"],
        "workflowStages": workflow_payload["workflowStages"][:2],
    }
    result = solve_workflow(
        master,
        phase2_payload,
        solver="beam",
        heuristic_weight=1.0,
        beam_width=4,
        time_budget_ms=10_000,
        use_validation_recovery=True,
    )

    final_cun4 = result.stages[1].view.steps[-1].track_sequences["存4北"]
    assert final_cun4[:2] == ["O3A", "C3A"] or final_cun4[:2] == ["O4A", "O3A"]
    outbound_positions = {vehicle_no: final_cun4.index(vehicle_no) for vehicle_no in ("O4A", "O3A")}
    cun4_positions = {vehicle_no: final_cun4.index(vehicle_no) for vehicle_no in ("C4A", "C3A")}
    assert max(outbound_positions.values()) < min(cun4_positions.values())


def test_phase2_reorder_uses_only_depot_outer_or_storage_buffers():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="修4", order="1", vehicle_no="C4A", target_track="存4北"),
            _vehicle(track_name="修4", order="2", vehicle_no="O4A", target_track="油"),
            _vehicle(track_name="修3", order="1", vehicle_no="C3A", target_track="存4北"),
            _vehicle(track_name="修3", order="2", vehicle_no="O3A", target_track="调棚"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase2_payload = {
        "trackInfo": workflow_payload["trackInfo"],
        "initialVehicleInfo": workflow_payload["initialVehicleInfo"],
        "locoTrackName": workflow_payload["locoTrackName"],
        "workflowStages": workflow_payload["workflowStages"][:2],
    }
    result = solve_workflow(
        master,
        phase2_payload,
        solver="beam",
        heuristic_weight=1.0,
        beam_width=4,
        time_budget_ms=10_000,
        use_validation_recovery=True,
    )

    allowed_buffers = {
        "修1库外",
        "修2库外",
        "修3库外",
        "修4库外",
        "存1",
        "存2",
        "存3",
        "存4南",
        "存5北",
        "存5南",
    }
    reorder_detach_targets = [
        hook.target_track
        for hook in result.stages[1].view.hook_plan
        if hook.action_type == "DETACH"
        and hook.target_track not in {"修3", "修4", "存4北"}
    ]

    assert reorder_detach_targets
    assert set(reorder_detach_targets).issubset(allowed_buffers)
    assert set(reorder_detach_targets).isdisjoint({"机北3", "调北", "预修", "机棚"})
    assert len(set(reorder_detach_targets)) >= 2


def test_phase3_restores_exact_depot_goals_and_phase4_downgrades_to_dynamic_hold():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="轮"),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="修1", is_spotting="101"),
            _vehicle(track_name="存4北", order="1", vehicle_no="C", target_track="存1"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase3_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][2]["vehicleGoals"]
    }
    phase4_goals = {
        item["vehicleNo"]: item
        for item in workflow_payload["workflowStages"][3]["vehicleGoals"]
    }

    assert phase3_goals["A"]["targetTrack"] == "轮"
    assert phase3_goals["A"]["isSpotting"] == ""
    assert phase3_goals["B"]["targetTrack"] == "修1"
    assert phase3_goals["B"]["isSpotting"] == "101"
    assert phase3_goals["C"]["targetSource"] == PHASE3_DYNAMIC_CURRENT_HOLD
    assert phase3_goals["C"]["targetTrack"] == "存4北"
    assert workflow_payload["workflowStages"][3]["stagePolicy"]["stageMode"] == PHASE4_RESIDUAL_CLEANUP
    assert phase4_goals["A"]["targetSource"] == PHASE4_DYNAMIC_CURRENT_HOLD
    assert phase4_goals["A"]["targetMode"] == "SNAPSHOT"
    assert phase4_goals["C"]["targetTrack"] == "存1"


def test_phase3_builds_wave_plans_for_multi_block_depot_push():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A1", target_track="修1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="A2", target_track="修1"),
            _vehicle(track_name="预修", order="1", vehicle_no="B1", target_track="修2"),
            _vehicle(track_name="预修", order="2", vehicle_no="B2", target_track="修2"),
            _vehicle(track_name="修3", order="1", vehicle_no="HOLD3", target_track="修3"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase3_stage = _resolve_dynamic_stage(
        stage=workflow_payload["workflowStages"][2],
        track_info=payload["trackInfo"],
        current_vehicle_info=payload["vehicleInfo"],
        current_state=ReplayState(
            track_sequences={
                "调棚": ["A1", "A2"],
                "预修": ["B1", "B2"],
                "修3": ["HOLD3"],
            },
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )
    phase3_policy = phase3_stage["stagePolicy"]
    wave_plans = list(phase3_policy.get("phase3WavePlans") or [])

    assert phase3_policy["stageMode"] == "PHASE3_JI_TO_DEPOT_ALLOCATION"
    assert len(wave_plans) == 2
    waves_by_target = {
        wave["waveTargetTrack"]: set(wave["activeGoalsByVehicle"])
        for wave in wave_plans
    }
    assert waves_by_target == {
        "修1": {"A1", "A2"},
        "修2": {"B1", "B2"},
    }


def test_phase3_compacts_same_source_blocks_into_preflighted_tail_run():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="B", target_track="修2"),
            _vehicle(track_name="调棚", order="3", vehicle_no="C", target_track="修2"),
            _vehicle(track_name="修3", order="1", vehicle_no="HOLD3", target_track="修3"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase3_stage = _resolve_dynamic_stage(
        stage=workflow_payload["workflowStages"][2],
        track_info=payload["trackInfo"],
        current_vehicle_info=payload["vehicleInfo"],
        current_state=ReplayState(
            track_sequences={
                "调棚": ["A", "B", "C"],
                "修3": ["HOLD3"],
            },
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )
    phase3_policy = phase3_stage["stagePolicy"]
    wave_plans = list(phase3_policy.get("phase3WavePlans") or [])

    assert len(wave_plans) == 1
    assert wave_plans[0]["requiresExplicitPlan"] is True
    assert wave_plans[0]["waveRole"] == "PHASE3_SOURCE_TAIL_RUN_TO_DEPOT"
    assert wave_plans[0]["waveTargetRuns"] == [
        {"targetTrack": "修1", "vehicleNos": ["A"]},
        {"targetTrack": "修2", "vehicleNos": ["B", "C"]},
    ]
    assert phase3_policy["phase3ExecutionPlanDiagnostics"]["enabled"] is True
    assert phase3_policy["phase3ExecutionPlanDiagnostics"]["plannedHookCount"] == 3


def test_phase3_does_not_enable_partial_wave_plans_when_active_vehicle_is_hidden():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="调棚", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="调棚", order="2", vehicle_no="HOLD", target_track="调棚"),
            _vehicle(track_name="调棚", order="3", vehicle_no="B", target_track="修2"),
        ],
        "locoTrackName": "机库",
    }

    workflow_payload = build_l7_closed_topology_workflow_payload(master, payload)
    phase3_stage = _resolve_dynamic_stage(
        stage=workflow_payload["workflowStages"][2],
        track_info=payload["trackInfo"],
        current_vehicle_info=payload["vehicleInfo"],
        current_state=ReplayState(
            track_sequences={"调棚": ["A", "HOLD", "B"]},
            loco_track_name="机库",
            weighed_vehicle_nos=set(),
            spot_assignments={},
        ),
        master=master,
    )
    phase3_policy = phase3_stage["stagePolicy"]
    diagnostics = phase3_policy["phase3BlockPlanDiagnostics"]

    assert diagnostics["coveredVehicleNos"] == ["A"]
    assert diagnostics["hiddenActiveVehicleNos"] == ["B"]
    assert diagnostics["allActiveCoveredByFrontier"] is False
    assert phase3_policy["phase3ExecutionPlanDiagnostics"]["enabled"] is False
    assert phase3_policy["phase3ExecutionPlanDiagnostics"]["reason"] == "active_vehicle_hidden_behind_hold"
    assert "phase3WavePlans" not in phase3_policy


def test_solve_workflow_auto_expands_l7_mode_and_applies_stage_route_overlay(monkeypatch):
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="修1"),
            _vehicle(track_name="存1", order="1", vehicle_no="B", target_track="存4北"),
        ],
        "locoTrackName": "机库",
    }
    seen_stage_names: list[str] = []
    seen_branch_status: list[str] = []

    def fake_build_demo_view_model(
        stage_master,
        stage_payload,
        plan_payload=None,
        solver="beam",
        heuristic_weight=1.0,
        beam_width=None,
        time_budget_ms=None,
        initial_state_override=None,
        use_validation_recovery=True,
        diagnose_front_search_only=False,
    ):
        seen_stage_names.append(stage_payload["workflowStageName"])
        seen_branch_status.append(stage_master.physical_routes["L15-L16"].status)
        track_sequences: dict[str, list[str]] = {}
        for item in stage_payload["vehicleInfo"]:
            track_sequences.setdefault(str(item["targetTrack"]), []).append(str(item["vehicleNo"]))
        return SimpleNamespace(
            summary=SimpleNamespace(
                is_valid=True,
                hook_count=0,
                final_tracks=sorted(track_sequences),
            ),
            final_spot_assignments={},
            final_work_position_assignments={},
            failed_hook_nos=[],
            verifier_errors=[],
            steps=[
                SimpleNamespace(
                    track_sequences=track_sequences,
                    loco_track_name=stage_payload["locoTrackName"],
                    weighed_vehicle_nos=[],
                    spot_assignments={},
                )
            ],
        )

    monkeypatch.setattr(
        "fzed_shunting.demo.view_model.build_demo_view_model",
        fake_build_demo_view_model,
    )

    result = solve_workflow(master, payload)

    assert result.stage_count == 4
    phase1_call_count = seen_stage_names.count("phase1_pre_repair_buffering")
    assert phase1_call_count >= 1
    assert seen_stage_names[:phase1_call_count] == [
        "phase1_pre_repair_buffering",
    ] * phase1_call_count
    assert seen_stage_names[phase1_call_count] == "phase2_depot_area_marshalling"
    assert seen_stage_names[-1] == "final_exact_settle_and_cleanup"
    assert set(seen_stage_names[phase1_call_count + 1:-1]) == {
        "phase3_ji_to_depot_allocation",
    }
    assert seen_branch_status[:phase1_call_count] == ["阶段封锁"] * phase1_call_count
    assert seen_branch_status[phase1_call_count:] == ["已确认"] * (len(seen_branch_status) - phase1_call_count)
    assert result.stages[0].input_payload["stagePolicy"]["stageMode"] == "PHASE1_PRE_REPAIR_BUFFERING"
    assert len(result.stages[0].input_payload["stagePolicy"]["phase1WavePlans"]) == phase1_call_count
