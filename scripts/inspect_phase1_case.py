from __future__ import annotations

import argparse
import json
from pathlib import Path

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.workflow.l7_closed_topology_mode import build_l7_closed_topology_workflow_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--master-dir", type=Path, default=Path("data/master"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    payload["operationMode"] = "L7_CLOSED_TOPOLOGY"
    master = load_master_data(args.master_dir)
    workflow = build_l7_closed_topology_workflow_payload(master, payload)
    stage1 = workflow["workflowStages"][0]
    print(json.dumps(stage1["stagePolicy"], ensure_ascii=False, indent=2))
    print(json.dumps(stage1["vehicleGoals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
