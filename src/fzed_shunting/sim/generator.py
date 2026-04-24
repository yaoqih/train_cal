from __future__ import annotations

import random

from fzed_shunting.domain.master_data import MasterData


MODELS = ["罐车", "棚车", "敞车", "平车"]
PROCESSES = ["段修", "厂修", "临修"]
ATTRS = ["", "", "", "称重", "重车", "关门车"]
PROFILE_CONFIGS = {
    "micro": {
        "start_tracks": ["存5北", "存5南", "存1", "存2", "存3"],
        "targets": ["机库", "存4北", "调棚", "洗南", "修1库外", "大库"],
        "direct_targets": ["机库", "存4北", "修1库外", "修2库外", "修3库外", "修4库外"],
        "default_tags": ["micro"],
    },
    "medium": {
        "start_tracks": ["存5北", "存5南", "存1", "存2", "存3", "机北", "调北", "洗北"],
        "targets": ["机库", "存4北", "调棚", "洗南", "修1库外", "修2库外", "修3库外", "修4库外", "大库"],
        "direct_targets": ["机库", "存4北", "修1库外", "修2库外", "修3库外", "修4库外"],
        "default_tags": ["medium", "mixed_goals"],
    },
    "large": {
        "start_tracks": ["存5北", "存5南", "存1", "存2", "存3", "机北", "调北", "洗北", "修1库外", "修2库外"],
        "targets": ["机库", "存4北", "调棚", "洗南", "修1库外", "修2库外", "修3库外", "修4库外", "大库"],
        "direct_targets": ["机库", "存4北", "修1库外", "修2库外", "修3库外", "修4库外"],
        "default_tags": ["large", "mixed_goals", "congested"],
    },
}


def generate_micro_scenario(
    master: MasterData,
    seed: int,
    vehicle_count: int = 6,
    direct_only: bool = False,
) -> dict:
    return generate_scenario(
        master,
        seed=seed,
        vehicle_count=vehicle_count,
        profile="micro",
        direct_only=direct_only,
    )


def generate_scenario(
    master: MasterData,
    seed: int,
    vehicle_count: int = 6,
    profile: str = "micro",
    direct_only: bool = False,
) -> dict:
    if profile == "adversarial":
        return _generate_adversarial_scenario(master, seed=seed, vehicle_count=vehicle_count)
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"Unsupported scenario profile: {profile}")

    rng = random.Random(seed)
    config = PROFILE_CONFIGS[profile]
    track_orders: dict[str, int] = {}
    vehicles: list[dict] = []
    used_track_infos: set[str] = set()
    targets = config["direct_targets"] if direct_only else config["targets"]
    for idx in range(vehicle_count):
        track = rng.choice(config["start_tracks"])
        target = rng.choice(targets)
        vehicle = _build_vehicle(
            rng=rng,
            seed=seed,
            idx=idx,
            track=track,
            target=target,
            order=_next_order(track_orders, track),
        )
        vehicles.append(vehicle)
        used_track_infos.add(track)
        used_track_infos.update(_required_track_infos_for_vehicle(vehicle))

    if profile in {"medium", "large"} and vehicle_count >= 2:
        _inject_front_blocker(vehicles, seed=seed)
        if len(vehicles) > vehicle_count:
            del vehicles[vehicle_count:]
        used_track_infos.add("临1")

    return _build_scenario(
        master=master,
        seed=seed,
        profile=profile,
        vehicles=vehicles,
        used_track_infos=used_track_infos,
        tags=list(config["default_tags"]) + (["direct_only"] if direct_only else []),
        rng=rng,
    )


def generate_typical_suite(master: MasterData) -> dict:
    scenarios = [
        {
            "name": "single_direct",
            "description": "单车直接送达最终目标",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP001",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "机库",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "single_direct"]},
            },
        },
        {
            "name": "weigh_then_store",
            "description": "称重后再落最终目标",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP002",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存4北",
                        "isSpotting": "",
                        "vehicleAttributes": "称重",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "weigh_then_store"]},
            },
        },
        {
            "name": "front_blocker",
            "description": "源股道前挡车，需要临停清障",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "临1", "trackDistance": master.tracks["临1"].effective_length_m},
                    {"trackName": "临3", "trackDistance": master.tracks["临3"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP003A",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存5北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "2",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP003B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "机库",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 2, "tags": ["typical", "front_blocker"]},
            },
        },
        {
            "name": "path_blocker",
            "description": "中间路径被占，需要临停清障",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "存5南", "trackDistance": master.tracks["存5南"].effective_length_m},
                    {"trackName": "临1", "trackDistance": master.tracks["临1"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP004A",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "targetTrack": "大库",
                        "isSpotting": "101",
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5南",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP004B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存5南",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 2, "tags": ["typical", "path_blocker"]},
            },
        },
        {
            "name": "cun4nan_staging",
            "description": "使用存4南作为过程临停清障，最终仍回到正式目标线",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "存4南", "trackDistance": master.tracks["存4南"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP004C",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存5北",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "2",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP004D",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "机库",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical",
                    "seed": 0,
                    "vehicleCount": 2,
                    "tags": ["typical", "cun4nan_staging", "temporary_staging"],
                },
            },
        },
        {
            "name": "dispatch_work_spot",
            "description": "调棚作业工位分配",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP005",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "调棚",
                        "isSpotting": "是",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "dispatch_work_spot"]},
            },
        },
        {
            "name": "dispatch_pre_repair",
            "description": "调棚预修区默认落位",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP006",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "调棚",
                        "isSpotting": "",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "dispatch_pre_repair"]},
            },
        },
        {
            "name": "inspection_depot",
            "description": "迎检模式下的大库随机落位",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "修2库内", "trackDistance": master.tracks["修2库内"].effective_length_m},
                    {"trackName": "修3库内", "trackDistance": master.tracks["修3库内"].effective_length_m},
                    {"trackName": "修4库内", "trackDistance": master.tracks["修4库内"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP007",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "targetTrack": "大库",
                        "isSpotting": "迎检",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "inspection"]},
            },
        },
        {
            "name": "close_door_non_cun4bei",
            "description": "非存4北场景下的大钩关门车顺位约束",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "存3", "trackDistance": master.tracks["存3"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": str(idx + 1),
                        "vehicleModel": "棚车",
                        "vehicleNo": f"TYP008{idx + 1:02d}",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "存3",
                        "isSpotting": "",
                        "vehicleAttributes": "关门车" if idx == 0 else "",
                    }
                    for idx in range(11)
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 11, "tags": ["typical", "close_door"]},
            },
        },
        {
            "name": "wash_work_area",
            "description": "洗南作业区对位场景",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "洗南", "trackDistance": master.tracks["洗南"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "罐车",
                        "vehicleNo": "TYP009",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "targetTrack": "洗南",
                        "isSpotting": "是",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "wash"]},
            },
        },
        {
            "name": "wheel_operate",
            "description": "轮线运营作业区场景",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "轮", "trackDistance": master.tracks["轮"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "敞车",
                        "vehicleNo": "TYP010",
                        "repairProcess": "临修",
                        "vehicleLength": 14.3,
                        "targetTrack": "轮",
                        "isSpotting": "是",
                        "vehicleAttributes": "重车",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {"profile": "typical", "seed": 0, "vehicleCount": 1, "tags": ["typical", "wheel"]},
            },
        },
        {
            "name": "paint_work_area",
            "description": "油漆作业区对位场景",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "油", "trackDistance": master.tracks["油"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP011",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "targetTrack": "油",
                        "isSpotting": "是",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "paint"],
                },
            },
        },
        {
            "name": "shot_work_area",
            "description": "抛丸作业区对位场景",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "抛", "trackDistance": master.tracks["抛"].effective_length_m},
                ],
                "vehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "TYP012",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "targetTrack": "抛",
                        "isSpotting": "是",
                        "vehicleAttributes": "",
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "shot"],
                },
            },
        },
    ]
    return {"suite": "typical", "scenarioCount": len(scenarios), "scenarios": scenarios}


def generate_typical_workflow_suite(master: MasterData) -> dict:
    scenarios = [
        {
            "name": "dispatch_depot_departure",
            "description": "调棚作业后入大库，再修竣进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_001",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "dispatch_work",
                        "description": "先送调棚作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_001", "targetTrack": "调棚", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "depot_spot",
                        "description": "作业后送入大库精准台位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_001", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "修竣后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_001", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "dispatch", "depot", "departure"],
                },
            },
        },
        {
            "name": "weigh_then_departure",
            "description": "先在机库完成称重，再送存4北集结",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_002",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "称重",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "weigh_stage",
                        "description": "先在机库完成称重",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_002", "targetTrack": "机库", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "称重后送入存4北集结",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_002", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "weigh", "departure"],
                },
            },
        },
        {
            "name": "weigh_jiku_final",
            "description": "称重车直接在机库完成称重并以机库作为最终存车线",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_002B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "称重",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "jiku_final",
                        "description": "直接送机库完成称重并终到",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_002B", "targetTrack": "机库", "isSpotting": ""}
                        ],
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "weigh", "jiku_final"],
                },
            },
        },
        {
            "name": "multi_vehicle_dispatch_merge",
            "description": "两车先分流，一车调棚入库，另一车原地待命，最后共同并入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_003A",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "2",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_003B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                ],
                "workflowStages": [
                    {
                        "name": "dispatch_work",
                        "description": "前车送调棚作业，后车留在原线",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_003A", "targetTrack": "调棚", "isSpotting": "是"},
                            {"vehicleNo": "WF_TYP_003B", "targetTrack": "存5北", "isSpotting": ""},
                        ],
                    },
                    {
                        "name": "depot_hold",
                        "description": "前车作业后送入大库，后车继续待命",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_003A", "targetTrack": "大库", "isSpotting": "101"},
                            {"vehicleNo": "WF_TYP_003B", "targetTrack": "存5北", "isSpotting": ""},
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "两车最终并入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_003A", "targetTrack": "存4北", "isSpotting": ""},
                            {"vehicleNo": "WF_TYP_003B", "targetTrack": "存4北", "isSpotting": ""},
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 2,
                    "tags": ["typical", "workflow", "multi_vehicle", "dispatch", "merge"],
                },
            },
        },
        {
            "name": "wash_depot_departure",
            "description": "罐车先入洗南，再送大库，最后进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "洗南", "trackDistance": master.tracks["洗南"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "罐车",
                        "vehicleNo": "WF_TYP_004",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "wash_work",
                        "description": "先送洗南作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_004", "targetTrack": "洗南", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "depot_spot",
                        "description": "洗南后送入大库精准台位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_004", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "修竣后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_004", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "wash", "depot", "departure"],
                },
            },
        },
        {
            "name": "wheel_departure",
            "description": "重车先入轮线作业，再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "轮", "trackDistance": master.tracks["轮"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "敞车",
                        "vehicleNo": "WF_TYP_005",
                        "repairProcess": "临修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "重车",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "wheel_operate",
                        "description": "先送轮线作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005", "targetTrack": "轮", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "作业后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "wheel", "departure"],
                },
            },
        },
        {
            "name": "paint_depot_departure",
            "description": "先入油漆作业区，再送大库，最后进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "油", "trackDistance": master.tracks["油"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_005B",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "paint_work",
                        "description": "先送油漆作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005B", "targetTrack": "油", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "depot_spot",
                        "description": "油漆后送入大库精准台位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005B", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "工序完成后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005B", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "paint", "depot", "departure"],
                },
            },
        },
        {
            "name": "shot_depot_departure",
            "description": "先入抛丸作业区，再送大库，最后进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "抛", "trackDistance": master.tracks["抛"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_005C",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "shot_work",
                        "description": "先送抛丸作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005C", "targetTrack": "抛", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "depot_spot",
                        "description": "抛丸后送入大库精准台位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005C", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "工序完成后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_005C", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "shot", "depot", "departure"],
                },
            },
        },
        {
            "name": "outer_depot_departure",
            "description": "先送大库外随机线，再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "修1库外", "trackDistance": master.tracks["修1库外"].effective_length_m},
                    {"trackName": "修2库外", "trackDistance": master.tracks["修2库外"].effective_length_m},
                    {"trackName": "修3库外", "trackDistance": master.tracks["修3库外"].effective_length_m},
                    {"trackName": "修4库外", "trackDistance": master.tracks["修4库外"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_006",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "outer_random",
                        "description": "先送大库外随机线",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_006", "targetTrack": "大库外", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "阶段停放后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_006", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "outer_depot", "departure"],
                },
            },
        },
        {
            "name": "dispatch_jiku_final",
            "description": "先送调棚作业，再以机库作为最终存车线",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                    {"trackName": "机库", "trackDistance": master.tracks["机库"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_007",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "dispatch_work",
                        "description": "先送调棚作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007", "targetTrack": "调棚", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "jiku_final",
                        "description": "作业后以机库作为最终存车线",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007", "targetTrack": "机库", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "dispatch", "jiku_final"],
                },
            },
        },
        {
            "name": "pre_repair_departure",
            "description": "先送调棚尽头预修位，再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "调棚", "trackDistance": master.tracks["调棚"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_007B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "pre_repair",
                        "description": "先送调棚尽头预修位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007B", "targetTrack": "调棚", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "预修后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007B", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "pre_repair", "departure"],
                },
            },
        },
        {
            "name": "main_pre_repair_departure",
            "description": "先送主预修线，再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "预修", "trackDistance": master.tracks["预修"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_007C",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "main_pre_repair",
                        "description": "先送主预修线",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007C", "targetTrack": "预修", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "预修后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007C", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "main_pre_repair", "departure"],
                },
            },
        },
        {
            "name": "jipeng_departure",
            "description": "先送机棚补充预修线，再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "机棚", "trackDistance": master.tracks["机棚"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_007D",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "jipeng_hold",
                        "description": "先送机棚补充预修",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007D", "targetTrack": "机棚", "isSpotting": ""}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "补充预修后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007D", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "jipeng", "departure"],
                },
            },
        },
        {
            "name": "depot_wheel_departure",
            "description": "先入大库精准台位，再到轮线作业，最后进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "轮", "trackDistance": master.tracks["轮"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "敞车",
                        "vehicleNo": "WF_TYP_007E",
                        "repairProcess": "临修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "重车",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "depot_spot",
                        "description": "先送大库精准台位",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007E", "targetTrack": "大库", "isSpotting": "101"}
                        ],
                    },
                    {
                        "name": "wheel_operate",
                        "description": "后续送轮线运营作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007E", "targetTrack": "轮", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "作业完成后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_007E", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "depot", "wheel", "departure"],
                },
            },
        },
        {
            "name": "tank_wash_direct_departure",
            "description": "临修罐车先入洗南，再直接进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "洗南", "trackDistance": master.tracks["洗南"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "罐车",
                        "vehicleNo": "WF_TYP_008",
                        "repairProcess": "临修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "wash_work",
                        "description": "先送洗南作业区",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_008", "targetTrack": "洗南", "isSpotting": "是"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "临修洗罐后直接进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_008", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "tank", "wash", "direct_departure"],
                },
            },
        },
        {
            "name": "close_door_departure",
            "description": "关门车与普通车共同进入存4北，需满足 top-3 顺位限制",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_009A",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "2",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_009B",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "3",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_009C",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    },
                    {
                        "trackName": "存5北",
                        "order": "4",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_009D",
                        "repairProcess": "段修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "关门车",
                    },
                ],
                "workflowStages": [
                    {
                        "name": "departure",
                        "description": "全部车辆共同进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_009A", "targetTrack": "存4北", "isSpotting": ""},
                            {"vehicleNo": "WF_TYP_009B", "targetTrack": "存4北", "isSpotting": ""},
                            {"vehicleNo": "WF_TYP_009C", "targetTrack": "存4北", "isSpotting": ""},
                            {"vehicleNo": "WF_TYP_009D", "targetTrack": "存4北", "isSpotting": ""},
                        ],
                    }
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 4,
                    "tags": ["typical", "workflow", "close_door", "departure"],
                },
            },
        },
        {
            "name": "inspection_departure",
            "description": "迎检入大库后再进入存4北",
            "payload": {
                "trackInfo": [
                    {"trackName": "存5北", "trackDistance": master.tracks["存5北"].effective_length_m},
                    {"trackName": "修1库内", "trackDistance": master.tracks["修1库内"].effective_length_m},
                    {"trackName": "修2库内", "trackDistance": master.tracks["修2库内"].effective_length_m},
                    {"trackName": "修3库内", "trackDistance": master.tracks["修3库内"].effective_length_m},
                    {"trackName": "修4库内", "trackDistance": master.tracks["修4库内"].effective_length_m},
                    {"trackName": "存4北", "trackDistance": master.tracks["存4北"].effective_length_m},
                ],
                "initialVehicleInfo": [
                    {
                        "trackName": "存5北",
                        "order": "1",
                        "vehicleModel": "棚车",
                        "vehicleNo": "WF_TYP_010",
                        "repairProcess": "厂修",
                        "vehicleLength": 14.3,
                        "vehicleAttributes": "",
                    }
                ],
                "workflowStages": [
                    {
                        "name": "inspection_depot",
                        "description": "先按迎检模式进入大库",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_010", "targetTrack": "大库", "isSpotting": "迎检"}
                        ],
                    },
                    {
                        "name": "departure",
                        "description": "迎检后进入存4北",
                        "vehicleGoals": [
                            {"vehicleNo": "WF_TYP_010", "targetTrack": "存4北", "isSpotting": ""}
                        ],
                    },
                ],
                "locoTrackName": "机库",
                "scenarioMeta": {
                    "profile": "typical_workflow",
                    "seed": 0,
                    "vehicleCount": 1,
                    "tags": ["typical", "workflow", "inspection", "depot", "departure"],
                },
            },
        },
    ]
    return {"suite": "typical_workflow", "scenarioCount": len(scenarios), "scenarios": scenarios}


def _generate_adversarial_scenario(
    master: MasterData,
    seed: int,
    vehicle_count: int,
) -> dict:
    rng = random.Random(seed)
    vehicles = [
        {
            "trackName": "存5北",
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": f"ADV{seed:03d}000",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "存5北",
            "isSpotting": "",
            "vehicleAttributes": "",
        },
        {
            "trackName": "存5北",
            "order": "2",
            "vehicleModel": "棚车",
            "vehicleNo": f"ADV{seed:03d}001",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": "机库",
            "isSpotting": "",
            "vehicleAttributes": "",
        },
    ]
    used_track_infos = {"存5北", "机库", "临1"}
    if vehicle_count >= 4:
        vehicles.extend(
            [
                {
                    "trackName": "存5南",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": f"ADV{seed:03d}002",
                    "repairProcess": "段修",
                    "vehicleLength": 14.3,
                    "targetTrack": "存5南",
                    "isSpotting": "",
                    "vehicleAttributes": "",
                },
                {
                    "trackName": "存2",
                    "order": "1",
                    "vehicleModel": "棚车",
                    "vehicleNo": f"ADV{seed:03d}003",
                    "repairProcess": "厂修",
                    "vehicleLength": 14.3,
                    "targetTrack": "大库",
                    "isSpotting": "101",
                    "vehicleAttributes": "",
                },
            ]
        )
        used_track_infos.update({"存5南", "存2", "修1库内"})
    next_idx = len(vehicles)
    extra_tracks = ["存1", "存3", "机北", "调北"]
    extra_targets = ["存4北", "机库", "修1库外", "修2库外"]
    while len(vehicles) < vehicle_count:
        track = extra_tracks[(len(vehicles) - next_idx) % len(extra_tracks)]
        target = extra_targets[(len(vehicles) - next_idx) % len(extra_targets)]
        vehicles.append(
            _build_vehicle(
                rng=rng,
                seed=seed,
                idx=len(vehicles),
                track=track,
                target=target,
                order=_order_for_existing_track(vehicles, track) + 1,
            )
        )
        used_track_infos.add(track)
        used_track_infos.update(_required_track_infos_for_vehicle(vehicles[-1]))
    return _build_scenario(
        master=master,
        seed=seed,
        profile="adversarial",
        vehicles=vehicles,
        used_track_infos=used_track_infos,
        tags=["adversarial", "front_blocker", "path_blocker"],
        rng=rng,
    )


def _build_scenario(
    master: MasterData,
    seed: int,
    profile: str,
    vehicles: list[dict],
    used_track_infos: set[str],
    tags: list[str],
    rng: random.Random,
) -> dict:
    track_info = []
    for code in sorted(used_track_infos):
        if code in master.tracks:
            track_info.append(
                {
                    "trackName": code,
                    "trackDistance": master.tracks[code].effective_length_m,
                }
            )
    return {
        "trackInfo": track_info,
        "vehicleInfo": vehicles,
        "locoTrackName": rng.choice(["机库", "机北"]),
        "scenarioMeta": {
            "profile": profile,
            "seed": seed,
            "vehicleCount": len(vehicles),
            "tags": tags,
        },
    }


def _build_vehicle(
    rng: random.Random,
    seed: int,
    idx: int,
    track: str,
    target: str,
    order: int,
) -> dict:
    is_spotting = ""
    if target == "调棚" and rng.random() < 0.5:
        is_spotting = "是"
    elif target == "大库" and rng.random() < 0.35:
        is_spotting = rng.choice(["迎检", "101", "301"])
    vehicle_attributes = rng.choice(ATTRS)
    if vehicle_attributes == "关门车" and target == "存4北":
        vehicle_attributes = ""
    return {
        "trackName": track,
        "order": str(order),
        "vehicleModel": rng.choice(MODELS),
        "vehicleNo": f"SIM{seed:03d}{idx:03d}",
        "repairProcess": rng.choice(PROCESSES),
        "vehicleLength": round(rng.uniform(12.0, 18.0), 1),
        "targetTrack": target,
        "isSpotting": is_spotting,
        "vehicleAttributes": vehicle_attributes,
    }


def _required_track_infos_for_vehicle(vehicle: dict) -> set[str]:
    track_infos = {vehicle["trackName"]}
    target = vehicle["targetTrack"]
    spotting = vehicle.get("isSpotting", "")
    if target == "大库":
        if spotting == "迎检" or spotting == "":
            track_infos.update({"修1库内", "修2库内", "修3库内", "修4库内"})
        elif spotting.isdigit() and len(spotting) == 3:
            track_infos.add(_spot_to_track(spotting))
        else:
            track_infos.update({"修1库内", "修2库内", "修3库内", "修4库内"})
    elif target == "大库外":
        track_infos.update({"修1库外", "修2库外", "修3库外", "修4库外"})
    else:
        track_infos.add(target)
    if vehicle["vehicleAttributes"] == "称重":
        track_infos.add("机库")
    return track_infos


def _spot_to_track(spot_code: str) -> str:
    return {
        "1": "修1库内",
        "2": "修2库内",
        "3": "修3库内",
        "4": "修4库内",
    }[spot_code[0]]


def _inject_front_blocker(vehicles: list[dict], seed: int) -> None:
    source_track = "存5北"
    front_vehicle = None
    blocked_vehicle = None
    for vehicle in vehicles:
        if vehicle["trackName"] == source_track and vehicle["order"] == "1":
            front_vehicle = vehicle
        if vehicle["trackName"] == source_track and vehicle["order"] == "2":
            blocked_vehicle = vehicle
    if front_vehicle is None:
        front_vehicle = {
            "trackName": source_track,
            "order": "1",
            "vehicleModel": "棚车",
            "vehicleNo": f"SIM{seed:03d}900",
            "repairProcess": "段修",
            "vehicleLength": 14.3,
            "targetTrack": source_track,
            "isSpotting": "",
            "vehicleAttributes": "",
        }
        vehicles.append(front_vehicle)
    else:
        front_vehicle["targetTrack"] = source_track
        front_vehicle["isSpotting"] = ""
        front_vehicle["vehicleAttributes"] = ""
    if blocked_vehicle is None:
        vehicles.append(
            {
                "trackName": source_track,
                "order": "2",
                "vehicleModel": "棚车",
                "vehicleNo": f"SIM{seed:03d}901",
                "repairProcess": "段修",
                "vehicleLength": 14.3,
                "targetTrack": "机库",
                "isSpotting": "",
                "vehicleAttributes": "",
            }
        )
    else:
        blocked_vehicle["targetTrack"] = "机库"
        blocked_vehicle["isSpotting"] = ""


def _next_order(track_orders: dict[str, int], track: str) -> int:
    order = track_orders.get(track, 0) + 1
    track_orders[track] = order
    return order


def _order_for_existing_track(vehicles: list[dict], track: str) -> int:
    orders = [int(vehicle["order"]) for vehicle in vehicles if vehicle["trackName"] == track]
    return max(orders, default=0)
