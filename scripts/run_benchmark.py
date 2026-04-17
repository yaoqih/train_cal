from pathlib import Path
import json

from fzed_shunting.benchmark.runner import run_benchmark


if __name__ == "__main__":
    report = run_benchmark(
        Path("artifacts/benchmark"),
        scenario_count=10,
        vehicle_count=4,
        seed_start=0,
        direct_only=True,
        profile="micro",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
