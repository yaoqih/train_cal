from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import (
    FIXED_DEPOT_RESIDENT_SOURCE,
    OPERATION_MODE_L7_CLOSED_TOPOLOGY,
    PHASE3_DYNAMIC_CURRENT_HOLD,
    PHASE4_DYNAMIC_CURRENT_HOLD,
    PHASE4_RESIDUAL_CLEANUP,
    build_l7_closed_topology_workflow_payload,
    rebuild_phase2_execution_policy_for_runtime,
)
from fzed_shunting.workflow.runner import solve_workflow


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


def test_phase1_uses_jibei1_as_legal_buffer_track():
    master = load_master_data(DATA_DIR)
    payload = {
        "operationMode": OPERATION_MODE_L7_CLOSED_TOPOLOGY,
        "trackInfo": _base_track_info(),
        "vehicleInfo": [
            _vehicle(track_name="洗南", order="1", vehicle_no="A", target_track="轮", vehicle_length=70.0),
            _vehicle(track_name="油", order="1", vehicle_no="B", target_track="轮", vehicle_length=70.0),
            _vehicle(track_name="调棚", order="1", vehicle_no="C", target_track="修3", vehicle_length=70.0),
            _vehicle(track_name="预修", order="1", vehicle_no="D", target_track="修4", vehicle_length=70.0),
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


def test_phase1_applies_backbone_budget_before_full_compile():
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

    assert diagnostics["selectedHotSourceTrackCount"] <= 4
    assert diagnostics["selectedStorageSourceTrackCount"] <= 2
    assert diagnostics["selectedPackageCount"] <= 18
    assert diagnostics["budgetHitReasons"]


def test_phase1_keeps_low_yield_storage_source_out_of_backbone_but_still_finishes_direct_tail():
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
    assert goals["D1"]["targetSource"] == "STAGE_HOLD"
    assert goals["B1"]["targetSource"] == "PHASE1_LOCAL_FINISH"
    assert goals["B1"]["targetTrack"] == "存1"
    assert goals["B2"]["targetSource"] == "PHASE1_LOCAL_FINISH"
    assert goals["B2"]["targetTrack"] == "存1"
    assert goals["B3"]["targetSource"] == "PHASE1_LOCAL_FINISH"
    assert goals["B3"]["targetTrack"] == "存2"
    source_summaries = {
        row["sourceTrack"]: row
        for row in diagnostics["sourceOpenSummaries"]
    }
    assert source_summaries["存5北"]["openingScore"] < 0
    assert source_summaries["存5北"]["selectedForBackbone"] is False
    assert source_summaries["存5北"]["admissionDecision"] == "deferred"
    assert source_summaries["存5北"]["rejectionReason"] == "weak_source"
    assert diagnostics["budgetHitReasons"]["weak_source"] >= 1
    assert diagnostics["selectedCleanupBySource"]["存5北"]["requiredLocalFinishCount"] == 2


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
    assert phase2_goals["B"]["targetMode"] == "AREA"
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
    assert seen_stage_names[:4] == [
        "phase1_pre_repair_buffering",
        "phase1_pre_repair_buffering",
        "phase1_pre_repair_buffering",
        "phase2_depot_area_marshalling",
    ]
    assert seen_stage_names[-1] == "final_exact_settle_and_cleanup"
    assert set(seen_stage_names[4:-1]) == {
        "phase3_ji_to_depot_allocation",
    }
    assert seen_branch_status[:3] == ["阶段封锁", "阶段封锁", "阶段封锁"]
    assert seen_branch_status[3:] == ["已确认"] * (len(seen_branch_status) - 3)
    assert result.stages[0].input_payload["stagePolicy"]["stageMode"] == "PHASE1_PRE_REPAIR_BUFFERING"
    assert len(result.stages[0].input_payload["stagePolicy"]["phase1WavePlans"]) == 3
