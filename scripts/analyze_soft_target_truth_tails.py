from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from fzed_shunting.domain.master_data import load_master_data
from fzed_shunting.io.normalize_input import normalize_plan_input
from fzed_shunting.solver.soft_target_template import compute_soft_target_template
from fzed_shunting.verify.replay import build_initial_state


def _nearest_percentile(values: list[int], pct: int) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct / 100)
    return ordered[max(0, min(index, len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    solved = [result for result in summary["results"] if result.get("solved")]
    p75 = _nearest_percentile([int(result["hook_count"]) for result in solved], 75)
    tail_results = [result for result in solved if int(result["hook_count"]) > p75]
    master = load_master_data(Path(args.master_dir))

    cases: list[dict] = []
    for result in sorted(tail_results, key=lambda item: item["hook_count"], reverse=True):
        scenario_path = Path(args.input_dir) / result["scenario"]
        normalized = normalize_plan_input(json.loads(scenario_path.read_text()), master)
        initial_state = build_initial_state(normalized)
        template = compute_soft_target_template(normalized, initial_state)
        summary_dict = template.to_summary_dict()
        shape = (result.get("debug_stats") or {}).get("plan_shape_metrics") or {}
        cases.append(
            {
                "scenario": result["scenario"],
                "hook_count": result["hook_count"],
                "fallback_stage": result.get("fallback_stage"),
                "staging_hook_count": shape.get("staging_hook_count"),
                "staging_to_staging_hook_count": shape.get("staging_to_staging_hook_count"),
                "max_vehicle_touch_count": shape.get("max_vehicle_touch_count"),
                "soft_target_summary": summary_dict,
            }
        )

    random_counts = [
        case["soft_target_summary"]["random_depot_template_count"]
        for case in cases
    ]
    output = {
        "summary_path": args.summary,
        "scenario_count": len(summary["results"]),
        "solved_count": len(solved),
        "p75_hook_count": p75,
        "tail_count": len(cases),
        "mean_random_depot_template_count": round(mean(random_counts), 3)
        if random_counts
        else 0.0,
        "cases": cases,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    lines = [
        "# Truth P75 Soft Target Template Wave11",
        "",
        f"Summary: `{args.summary}`",
        f"Solved: {len(solved)} / {len(summary['results'])}",
        f"p75 hook count: {p75}",
        f"Tail cases (>p75): {len(cases)}",
        f"Mean random depot template count: {output['mean_random_depot_template_count']}",
        "",
        "| Scenario | Hooks | Stage | Staging | S2S | Touch | Templates | Random | 存4北 prefix | Preferred tracks |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        tpl = case["soft_target_summary"]
        preferred = "; ".join(
            f"{track}={count}"
            for track, count in sorted(
                tpl["preferred_track_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        )
        lines.append(
            f"| {case['scenario']} | {case['hook_count']} | {case['fallback_stage']} | "
            f"{case['staging_hook_count']} | {case['staging_to_staging_hook_count']} | "
            f"{case['max_vehicle_touch_count']} | {tpl['vehicle_template_count']} | "
            f"{tpl['random_depot_template_count']} | "
            f"{tpl['cun4bei_template']['required_plain_prefix_count']} | {preferred} |"
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
