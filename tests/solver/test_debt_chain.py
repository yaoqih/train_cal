from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.domain.route_oracle import RouteOracle
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.debt_chain import analyze_debt_chains
from fzed_shunting.verify.replay import build_initial_state


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "master"


def _vehicle(vehicle_no: str, source: str, target: str, *, order: int = 1, spotting: str = "") -> dict:
    return {
        "trackName": source,
        "order": str(order),
        "vehicleModel": "棚车",
        "vehicleNo": vehicle_no,
        "repairProcess": "段修",
        "vehicleLength": 14.3,
        "targetTrack": target,
        "isSpotting": spotting,
        "vehicleAttributes": "",
    }


def test_debt_chain_analyzer_separates_independent_tracks():
    master = load_master_data(DATA_DIR)
    normalized = normalize_plan_input(
        {
            "trackInfo": [
                {"trackName": "机库", "trackDistance": 71.6},
                {"trackName": "调棚", "trackDistance": 174.3},
                {"trackName": "存5北", "trackDistance": 367.0},
                {"trackName": "存1", "trackDistance": 113.0},
            ],
            "vehicleInfo": [
                _vehicle("SPOT_A", "存5北", "调棚", order=1, spotting="是"),
                _vehicle("CAP_A", "存1", "存1", order=1),
                _vehicle("CAP_B", "存1", "存4北", order=2),
            ],
            "locoTrackName": "机库",
        },
        master,
        allow_internal_loco_tracks=True,
    )
    state = build_initial_state(normalized)

    summary = analyze_debt_chains(normalized, state, route_oracle=RouteOracle(master))

    assert summary.chain_count == 2
    assert {chain.anchor_track for chain in summary.chains} == {"调棚", "存1"}
    assert any(chain.track_names == ("存1",) for chain in summary.chains)
    assert any(chain.track_names == ("存5北", "调棚") for chain in summary.chains)
