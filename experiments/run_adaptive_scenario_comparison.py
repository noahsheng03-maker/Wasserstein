from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_paper_style_experiment import build_bundle, evaluate_closed_loop, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive-vs-fixed scenario comparisons.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "experiments" / "adaptive_compare_smoke.yaml"),
        help="Path to adaptive scenario comparison config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "adaptive_compare_smoke"),
        help="Directory for adaptive comparison outputs.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only one named scenario from scenario_library.",
    )
    parser.add_argument(
        "--fixed-epsilon",
        type=float,
        default=None,
        help="Run only one fixed epsilon instead of the whole fixed_epsilons list.",
    )
    parser.add_argument(
        "--skip-adaptive",
        action="store_true",
        help="Skip the adaptive controller and run only fixed-epsilon comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    output_path = out_dir / "adaptive_scenario_comparison.csv"
    scenario_names = config["adaptive_compare"]["scenarios"]
    if args.scenario is not None:
        scenario_names = [args.scenario]
    fixed_epsilons = config["adaptive_compare"]["fixed_epsilons"]
    if args.fixed_epsilon is not None:
        fixed_epsilons = [float(args.fixed_epsilon)]

    for scenario_name in scenario_names:
        scenario_cfg = copy.deepcopy(config)
        scenario_cfg["scenario"] = config["scenario_library"][scenario_name]
        bundle = build_bundle(scenario_cfg)

        if not args.skip_adaptive:
            print(f"[adaptive-compare] running scenario={scenario_name} mode=adaptive")
            adaptive = evaluate_closed_loop(scenario_cfg, mode="adaptive", bundle=bundle)
            with (out_dir / f"{scenario_name}_adaptive_trace.json").open("w", encoding="utf-8") as handle:
                json.dump(adaptive, handle, indent=2)
            records.append(
                {
                    "scenario": scenario_name,
                    "controller_mode": "adaptive",
                    "epsilon_setting": adaptive["epsilon_initial"],
                    "tracking_error": adaptive["tracking_error"],
                    "violation_rate": adaptive["violation_rate"],
                    "tail_violation_score": adaptive["tail_violation_score"],
                    "epsilon_final": adaptive["epsilon_final"],
                    "offline_batches": adaptive["offline_batches"],
                }
            )

        for epsilon in fixed_epsilons:
            print(f"[adaptive-compare] running scenario={scenario_name} mode=fixed epsilon={epsilon}")
            fixed = evaluate_closed_loop(scenario_cfg, mode="fixed", fixed_epsilon=epsilon, bundle=bundle)
            records.append(
                {
                    "scenario": scenario_name,
                    "controller_mode": f"fixed_{epsilon}",
                    "epsilon_setting": epsilon,
                    "tracking_error": fixed["tracking_error"],
                    "violation_rate": fixed["violation_rate"],
                    "tail_violation_score": fixed["tail_violation_score"],
                    "epsilon_final": fixed["epsilon_final"],
                    "offline_batches": fixed["offline_batches"],
                }
            )

    new_frame = pd.DataFrame(records)
    if output_path.exists():
        old_frame = pd.read_csv(output_path)
        frame = pd.concat([old_frame, new_frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["scenario", "controller_mode"], keep="last")
        frame = frame.sort_values(["scenario", "controller_mode"]).reset_index(drop=True)
    else:
        frame = new_frame

    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
