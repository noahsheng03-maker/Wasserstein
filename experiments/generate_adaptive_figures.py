from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_TITLES = {
    "noise_jump": "Noise Jump",
    "recovery": "Recovery",
    "bias_shift": "Bias Shift",
}

METRICS = [
    ("tracking_error", "Tracking Error"),
    ("violation_rate", "Violation Rate"),
    ("tail_violation_score", "Tail Violation Score"),
]


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


def _fixed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = frame[frame["controller_mode"].str.startswith("fixed_")].copy()
    fixed["epsilon"] = fixed["epsilon_setting"].astype(float)
    return fixed.sort_values("epsilon")


def _adaptive_row(frame: pd.DataFrame) -> pd.Series:
    return frame[frame["controller_mode"] == "adaptive"].iloc[0]


def plot_scenario_curve(frame: pd.DataFrame, scenario: str, output_dir: Path) -> None:
    fixed = _fixed_rows(frame)
    adaptive = _adaptive_row(frame)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    for ax, (column, ylabel) in zip(axes, METRICS):
        ax.plot(
            fixed["epsilon"],
            fixed[column],
            color="#d68c1f",
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label="Fixed-radius sweep",
        )
        ax.axhline(
            adaptive[column],
            color="#155d8b",
            linestyle="--",
            linewidth=2.0,
            label="Adaptive radius",
        )
        ax.scatter(
            [adaptive["epsilon_setting"]],
            [adaptive[column]],
            color="#155d8b",
            marker="D",
            s=36,
            zorder=4,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"Wasserstein Radius $\epsilon$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"Full-Scale Adaptive DR-DeePC: {SCENARIO_TITLES.get(scenario, scenario)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"adaptive_compare_{scenario}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview(all_rows: pd.DataFrame, output_dir: Path) -> None:
    scenarios = [s for s in ["noise_jump", "recovery", "bias_shift"] if s in set(all_rows["scenario"])]
    if not scenarios:
        return

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(9.2, 3.2 * len(scenarios)))
    if len(scenarios) == 1:
        axes = [axes]

    for row_axes, scenario in zip(axes, scenarios):
        frame = all_rows[all_rows["scenario"] == scenario]
        fixed = _fixed_rows(frame)
        adaptive = _adaptive_row(frame)

        row_axes[0].plot(
            fixed["epsilon"],
            fixed["tracking_error"],
            color="#d68c1f",
            marker="o",
            linewidth=2.0,
            markersize=5,
        )
        row_axes[0].axhline(adaptive["tracking_error"], color="#155d8b", linestyle="--", linewidth=2.0)
        row_axes[0].scatter([adaptive["epsilon_setting"]], [adaptive["tracking_error"]], color="#155d8b", marker="D", s=34)
        row_axes[0].set_xscale("log")
        row_axes[0].set_ylabel(f"{SCENARIO_TITLES.get(scenario, scenario)}\nTracking Error", fontsize=9)
        row_axes[0].grid(True, which="both", alpha=0.25, linewidth=0.6)

        row_axes[1].plot(
            fixed["epsilon"],
            fixed["violation_rate"],
            color="#d68c1f",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Fixed-radius sweep",
        )
        row_axes[1].axhline(adaptive["violation_rate"], color="#155d8b", linestyle="--", linewidth=2.0, label="Adaptive radius")
        row_axes[1].scatter([adaptive["epsilon_setting"]], [adaptive["violation_rate"]], color="#155d8b", marker="D", s=34)
        row_axes[1].set_xscale("log")
        row_axes[1].set_ylabel("Violation Rate", fontsize=9)
        row_axes[1].grid(True, which="both", alpha=0.25, linewidth=0.6)

        for ax in row_axes:
            ax.set_xlabel(r"Wasserstein Radius $\epsilon$", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_axisbelow(True)

    axes[0][1].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("Adaptive Radius vs Fixed Radius Across Nonstationary Scenarios", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "adaptive_compare_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "adaptive_compare_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames = [pd.read_csv(Path(path)) for path in args.inputs]
    all_rows = pd.concat(frames, ignore_index=True)

    for scenario in sorted(all_rows["scenario"].unique()):
        scenario_frame = all_rows[all_rows["scenario"] == scenario]
        if "adaptive" not in set(scenario_frame["controller_mode"]):
            continue
        plot_scenario_curve(scenario_frame, scenario, output_dir)

    plot_overview(all_rows, output_dir)


if __name__ == "__main__":
    main()
