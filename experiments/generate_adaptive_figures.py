from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for adaptive DR-DeePC comparisons.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files with adaptive scenario comparison results.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "paper" / "figures"),
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def tidy_label(name: str) -> str:
    if name == "adaptive":
        return "Adaptive"
    if name.startswith("fixed_"):
        return f"Fixed {name.split('_', 1)[1]}"
    return name


def plot_scenario(frame: pd.DataFrame, scenario: str, output_dir: Path) -> None:
    frame = frame.copy()
    frame["label"] = frame["controller_mode"].map(tidy_label)
    order = ["adaptive", "fixed_0.0001", "fixed_0.003", "fixed_0.02"]
    frame["order"] = frame["controller_mode"].apply(lambda x: order.index(x) if x in order else len(order))
    frame = frame.sort_values("order")

    metrics = [
        ("tracking_error", "Tracking Error"),
        ("violation_rate", "Violation Rate"),
        ("tail_violation_score", "Tail Violation Score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    colors = ["#1b6ca8" if mode == "adaptive" else "#d9a441" for mode in frame["controller_mode"]]

    for ax, (column, title) in zip(axes, metrics):
        ax.bar(frame["label"], frame[column], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(f"Full-Scale Adaptive DR-DeePC Comparison: {scenario}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"adaptive_compare_{scenario}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames = [pd.read_csv(Path(path)) for path in args.inputs]
    all_rows = pd.concat(frames, ignore_index=True)
    for scenario in sorted(all_rows["scenario"].unique()):
        scenario_frame = all_rows[all_rows["scenario"] == scenario]
        plot_scenario(scenario_frame, scenario, output_dir)


if __name__ == "__main__":
    main()
