from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.sim.generator import (
    generate_micro_scenario,
    generate_scenario,
    generate_typical_suite,
    generate_typical_workflow_suite,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def test_generate_micro_scenario_returns_valid_shape():
    master = load_master_data(DATA_DIR)

    scenario = generate_micro_scenario(master, seed=7, vehicle_count=6)

    assert "trackInfo" in scenario
    assert "vehicleInfo" in scenario
    assert len(scenario["vehicleInfo"]) == 6
    assert scenario["locoTrackName"] in {"机库", "机北"}


def test_generated_vehicle_orders_are_unique_per_track():
    master = load_master_data(DATA_DIR)

    scenario = generate_micro_scenario(master, seed=13, vehicle_count=8)

    orders_by_track = {}
    for vehicle in scenario["vehicleInfo"]:
        track = vehicle["trackName"]
        orders_by_track.setdefault(track, set())
        assert vehicle["order"] not in orders_by_track[track]
        orders_by_track[track].add(vehicle["order"])


def test_generate_medium_scenario_includes_profile_metadata():
    master = load_master_data(DATA_DIR)

    scenario = generate_scenario(master, seed=23, vehicle_count=10, profile="medium")

    assert scenario["scenarioMeta"]["profile"] == "medium"
    assert scenario["scenarioMeta"]["seed"] == 23
    assert len(scenario["vehicleInfo"]) == 10


def test_generate_adversarial_scenario_contains_front_blocker_pattern():
    master = load_master_data(DATA_DIR)

    scenario = generate_scenario(master, seed=31, vehicle_count=4, profile="adversarial")

    assert scenario["scenarioMeta"]["profile"] == "adversarial"
    assert "front_blocker" in scenario["scenarioMeta"]["tags"]
    grouped = {}
    for vehicle in scenario["vehicleInfo"]:
        grouped.setdefault(vehicle["trackName"], []).append(vehicle)
    assert any(
        len(vehicles) >= 2
        and vehicles[0]["targetTrack"] == vehicles[0]["trackName"]
        and vehicles[1]["targetTrack"] != vehicles[1]["trackName"]
        for vehicles in grouped.values()
    )


def test_generated_scenarios_avoid_close_door_targeting_cun4bei():
    master = load_master_data(DATA_DIR)

    for seed in range(10, 30):
        scenario = generate_scenario(master, seed=seed, vehicle_count=20, profile="large")
        assert all(
            not (vehicle["vehicleAttributes"] == "关门车" and vehicle["targetTrack"] == "存4北")
            for vehicle in scenario["vehicleInfo"]
        )


def test_generated_scenario_includes_jiku_track_info_for_weigh_vehicle():
    master = load_master_data(DATA_DIR)

    found = False
    for seed in range(1, 50):
        scenario = generate_scenario(master, seed=seed, vehicle_count=12, profile="medium")
        if any(vehicle["vehicleAttributes"] == "称重" for vehicle in scenario["vehicleInfo"]):
            found = True
            assert any(track["trackName"] == "机库" for track in scenario["trackInfo"])
            break

    assert found is True


def test_generated_scenario_includes_all_depot_inner_tracks_for_random_depot_goal():
    master = load_master_data(DATA_DIR)

    found = False
    for seed in range(1, 80):
        scenario = generate_scenario(master, seed=seed, vehicle_count=12, profile="medium")
        if any(vehicle["targetTrack"] == "大库" and vehicle["isSpotting"] in {"", "迎检"} for vehicle in scenario["vehicleInfo"]):
            found = True
            track_names = {track["trackName"] for track in scenario["trackInfo"]}
            assert {"修1库内", "修2库内", "修3库内", "修4库内"} <= track_names
            break

    assert found is True


def test_generated_scenario_includes_exact_inner_track_for_exact_depot_spot():
    master = load_master_data(DATA_DIR)

    found = False
    for seed in range(1, 120):
        scenario = generate_scenario(master, seed=seed, vehicle_count=12, profile="medium")
        exact_spot_vehicle = next(
            (
                vehicle
                for vehicle in scenario["vehicleInfo"]
                if vehicle["targetTrack"] == "大库" and vehicle["isSpotting"] in {"101", "301"}
            ),
            None,
        )
        if exact_spot_vehicle is not None:
            found = True
            track_names = {track["trackName"] for track in scenario["trackInfo"]}
            expected = "修1库内" if exact_spot_vehicle["isSpotting"] == "101" else "修3库内"
            assert expected in track_names
            break

    assert found is True


def test_generate_typical_suite_contains_named_scenarios():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)

    assert len(suite["scenarios"]) >= 8
    names = {item["name"] for item in suite["scenarios"]}
    assert {
        "single_direct",
        "weigh_then_store",
        "front_blocker",
        "path_blocker",
        "cun4nan_staging",
        "dispatch_work_spot",
        "dispatch_pre_repair",
        "inspection_depot",
        "close_door_non_cun4bei",
    } <= names


def test_generate_typical_suite_contains_inspection_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    inspection = next(item for item in suite["scenarios"] if item["name"] == "inspection_depot")
    payload = inspection["payload"]

    assert payload["vehicleInfo"][0]["isSpotting"] == "迎检"
    assert payload["vehicleInfo"][0]["targetTrack"] == "大库"
    assert "inspection" in payload["scenarioMeta"]["tags"]
    track_names = {track["trackName"] for track in payload["trackInfo"]}
    assert {"修1库内", "修2库内", "修3库内", "修4库内"} <= track_names


def test_generate_typical_suite_contains_dispatch_pre_repair_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    pre_repair = next(item for item in suite["scenarios"] if item["name"] == "dispatch_pre_repair")
    payload = pre_repair["payload"]

    assert payload["vehicleInfo"][0]["targetTrack"] == "调棚"
    assert payload["vehicleInfo"][0]["isSpotting"] == ""


def test_generate_typical_suite_contains_cun4nan_staging_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    staging = next(item for item in suite["scenarios"] if item["name"] == "cun4nan_staging")
    payload = staging["payload"]

    track_names = {track["trackName"] for track in payload["trackInfo"]}
    assert {"存5北", "存4南", "机库"} <= track_names
    assert payload["vehicleInfo"][0]["targetTrack"] == "存5北"
    assert payload["vehicleInfo"][1]["targetTrack"] == "机库"
    assert "cun4nan_staging" in payload["scenarioMeta"]["tags"]


def test_generate_typical_suite_contains_close_door_non_cun4bei_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    close_door = next(item for item in suite["scenarios"] if item["name"] == "close_door_non_cun4bei")
    payload = close_door["payload"]

    assert len(payload["vehicleInfo"]) >= 11
    assert payload["vehicleInfo"][0]["vehicleAttributes"] == "关门车"
    assert payload["vehicleInfo"][0]["targetTrack"] != "存4北"
    assert payload["vehicleInfo"][0]["targetTrack"] == "存3"


def test_generate_typical_suite_contains_wash_area_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    wash = next(item for item in suite["scenarios"] if item["name"] == "wash_work_area")
    payload = wash["payload"]

    assert payload["vehicleInfo"][0]["targetTrack"] == "洗南"
    assert payload["vehicleInfo"][0]["isSpotting"] == "是"
    assert "wash" in payload["scenarioMeta"]["tags"]
    track_names = {track["trackName"] for track in payload["trackInfo"]}
    assert {"存5北", "洗南"} <= track_names


def test_generate_typical_suite_contains_wheel_operate_scenario():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    wheel = next(item for item in suite["scenarios"] if item["name"] == "wheel_operate")
    payload = wheel["payload"]

    assert payload["vehicleInfo"][0]["targetTrack"] == "轮"
    assert payload["vehicleInfo"][0]["isSpotting"] == "是"
    assert payload["vehicleInfo"][0]["vehicleAttributes"] == "重车"
    assert "wheel" in payload["scenarioMeta"]["tags"]


def test_generate_typical_suite_contains_paint_and_shot_work_area_scenarios():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_suite(master)
    names = {item["name"] for item in suite["scenarios"]}

    assert {"paint_work_area", "shot_work_area"} <= names


def test_generate_typical_workflow_suite_contains_dispatch_depot_departure_chain():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_workflow_suite(master)

    assert suite["scenarioCount"] >= 1
    workflow = next(item for item in suite["scenarios"] if item["name"] == "dispatch_depot_departure")
    payload = workflow["payload"]
    assert len(payload["workflowStages"]) == 3
    assert payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "调棚"
    assert payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert payload["workflowStages"][2]["vehicleGoals"][0]["targetTrack"] == "存4北"


def test_generate_typical_workflow_suite_contains_multi_vehicle_and_weigh_flows():
    master = load_master_data(DATA_DIR)

    suite = generate_typical_workflow_suite(master)

    names = {item["name"] for item in suite["scenarios"]}
    assert {
        "dispatch_depot_departure",
        "weigh_then_departure",
        "multi_vehicle_dispatch_merge",
        "wash_depot_departure",
        "wheel_departure",
        "paint_depot_departure",
        "shot_depot_departure",
        "outer_depot_departure",
        "dispatch_jiku_final",
        "tank_wash_direct_departure",
        "close_door_departure",
        "inspection_departure",
        "weigh_jiku_final",
        "pre_repair_departure",
        "main_pre_repair_departure",
        "jipeng_departure",
        "depot_wheel_departure",
    } <= names

    weigh = next(item for item in suite["scenarios"] if item["name"] == "weigh_then_departure")
    weigh_payload = weigh["payload"]
    assert len(weigh_payload["workflowStages"]) == 2
    assert weigh_payload["initialVehicleInfo"][0]["vehicleAttributes"] == "称重"
    assert weigh_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "机库"
    assert weigh_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    multi = next(item for item in suite["scenarios"] if item["name"] == "multi_vehicle_dispatch_merge")
    multi_payload = multi["payload"]
    assert len(multi_payload["initialVehicleInfo"]) == 2
    assert len(multi_payload["workflowStages"]) == 3
    assert multi_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "调棚"
    assert multi_payload["workflowStages"][0]["vehicleGoals"][1]["targetTrack"] == "存5北"
    assert all(goal["targetTrack"] == "存4北" for goal in multi_payload["workflowStages"][2]["vehicleGoals"])

    wash = next(item for item in suite["scenarios"] if item["name"] == "wash_depot_departure")
    wash_payload = wash["payload"]
    assert wash_payload["initialVehicleInfo"][0]["vehicleModel"] == "罐车"
    assert len(wash_payload["workflowStages"]) == 3
    assert wash_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "洗南"
    assert wash_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert wash_payload["workflowStages"][2]["vehicleGoals"][0]["targetTrack"] == "存4北"

    wheel = next(item for item in suite["scenarios"] if item["name"] == "wheel_departure")
    wheel_payload = wheel["payload"]
    assert wheel_payload["initialVehicleInfo"][0]["vehicleAttributes"] == "重车"
    assert len(wheel_payload["workflowStages"]) == 2
    assert wheel_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "轮"
    assert wheel_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    paint = next(item for item in suite["scenarios"] if item["name"] == "paint_depot_departure")
    paint_payload = paint["payload"]
    assert len(paint_payload["workflowStages"]) == 3
    assert paint_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "油"
    assert paint_payload["workflowStages"][0]["vehicleGoals"][0]["isSpotting"] == "是"
    assert paint_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert paint_payload["workflowStages"][1]["vehicleGoals"][0]["isSpotting"] == "101"
    assert paint_payload["workflowStages"][2]["vehicleGoals"][0]["targetTrack"] == "存4北"

    shot = next(item for item in suite["scenarios"] if item["name"] == "shot_depot_departure")
    shot_payload = shot["payload"]
    assert len(shot_payload["workflowStages"]) == 3
    assert shot_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "抛"
    assert shot_payload["workflowStages"][0]["vehicleGoals"][0]["isSpotting"] == "是"
    assert shot_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert shot_payload["workflowStages"][1]["vehicleGoals"][0]["isSpotting"] == "101"
    assert shot_payload["workflowStages"][2]["vehicleGoals"][0]["targetTrack"] == "存4北"

    outer = next(item for item in suite["scenarios"] if item["name"] == "outer_depot_departure")
    outer_payload = outer["payload"]
    assert len(outer_payload["workflowStages"]) == 2
    assert outer_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "大库外"
    assert outer_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    jiku = next(item for item in suite["scenarios"] if item["name"] == "dispatch_jiku_final")
    jiku_payload = jiku["payload"]
    assert len(jiku_payload["workflowStages"]) == 2
    assert jiku_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "调棚"
    assert jiku_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "机库"

    tank = next(item for item in suite["scenarios"] if item["name"] == "tank_wash_direct_departure")
    tank_payload = tank["payload"]
    assert tank_payload["initialVehicleInfo"][0]["vehicleModel"] == "罐车"
    assert tank_payload["initialVehicleInfo"][0]["repairProcess"] == "临修"
    assert len(tank_payload["workflowStages"]) == 2
    assert tank_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "洗南"
    assert tank_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    close_door = next(item for item in suite["scenarios"] if item["name"] == "close_door_departure")
    close_payload = close_door["payload"]
    assert len(close_payload["initialVehicleInfo"]) == 4
    assert sum(1 for item in close_payload["initialVehicleInfo"] if item["vehicleAttributes"] == "关门车") == 1
    assert len(close_payload["workflowStages"]) == 1
    assert all(goal["targetTrack"] == "存4北" for goal in close_payload["workflowStages"][0]["vehicleGoals"])

    inspection = next(item for item in suite["scenarios"] if item["name"] == "inspection_departure")
    inspection_payload = inspection["payload"]
    assert len(inspection_payload["workflowStages"]) == 2
    assert inspection_payload["workflowStages"][0]["vehicleGoals"][0]["isSpotting"] == "迎检"
    assert inspection_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert inspection_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    weigh_jiku = next(item for item in suite["scenarios"] if item["name"] == "weigh_jiku_final")
    weigh_jiku_payload = weigh_jiku["payload"]
    assert weigh_jiku_payload["initialVehicleInfo"][0]["vehicleAttributes"] == "称重"
    assert len(weigh_jiku_payload["workflowStages"]) == 1
    assert weigh_jiku_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "机库"

    pre_repair = next(item for item in suite["scenarios"] if item["name"] == "pre_repair_departure")
    pre_repair_payload = pre_repair["payload"]
    assert len(pre_repair_payload["workflowStages"]) == 2
    assert pre_repair_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "调棚"
    assert pre_repair_payload["workflowStages"][0]["vehicleGoals"][0]["isSpotting"] == ""
    assert pre_repair_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    main_pre_repair = next(item for item in suite["scenarios"] if item["name"] == "main_pre_repair_departure")
    main_pre_repair_payload = main_pre_repair["payload"]
    assert len(main_pre_repair_payload["workflowStages"]) == 2
    assert main_pre_repair_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "预修"
    assert main_pre_repair_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    jipeng = next(item for item in suite["scenarios"] if item["name"] == "jipeng_departure")
    jipeng_payload = jipeng["payload"]
    assert len(jipeng_payload["workflowStages"]) == 2
    assert jipeng_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "机棚"
    assert jipeng_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "存4北"

    depot_wheel = next(item for item in suite["scenarios"] if item["name"] == "depot_wheel_departure")
    depot_wheel_payload = depot_wheel["payload"]
    assert depot_wheel_payload["initialVehicleInfo"][0]["vehicleAttributes"] == "重车"
    assert len(depot_wheel_payload["workflowStages"]) == 3
    assert depot_wheel_payload["workflowStages"][0]["vehicleGoals"][0]["targetTrack"] == "大库"
    assert depot_wheel_payload["workflowStages"][0]["vehicleGoals"][0]["isSpotting"] == "101"
    assert depot_wheel_payload["workflowStages"][1]["vehicleGoals"][0]["targetTrack"] == "轮"
    assert depot_wheel_payload["workflowStages"][1]["vehicleGoals"][0]["isSpotting"] == "是"
    assert depot_wheel_payload["workflowStages"][2]["vehicleGoals"][0]["targetTrack"] == "存4北"
