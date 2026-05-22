from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for fixed-radius baseline studies.")
    parser.add_argument(
        "--nominal",
        default=str(REPO_ROOT / "outputs" / "full_scale" / "paper_style_nominal_comparison_full.csv"),
        help="Nominal full-scale comparison CSV.",
    )
    parser.add_argument(
        "--n-comparison",
        default=str(REPO_ROOT / "outputs" / "n_comparison_nominal.csv"),
        help="N comparison CSV.",
    )
    parser.add_argument(
        "--tini",
        default=str(REPO_ROOT / "outputs" / "full_scale_tini3" / "paper_style_tini_sweep.csv"),
        help="T_ini sweep CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "paper" / "figures"),
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def _nominal_fixed(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = frame[frame["controller_mode"].str.startswith("fixed_")].copy()
    fixed["epsilon"] = fixed["epsilon_setting"].astype(float)
    return fixed.sort_values("epsilon")


def _nominal_adaptive(frame: pd.DataFrame) -> pd.Series:
    return frame[frame["controller_mode"] == "adaptive"].iloc[0]


def plot_nominal_epsilon(frame: pd.DataFrame, output_dir: Path) -> None:
    fixed = _nominal_fixed(frame)
    adaptive = _nominal_adaptive(frame)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    metrics = [
        ("tracking_error", "Tracking Error"),
        ("violation_rate", "Violation Rate"),
        ("tail_violation_score", "Tail Violation Score"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(fixed["epsilon"], fixed[col], color="#c97b16", marker="o", linewidth=2.0)
        ax.axhline(adaptive[col], color="#155d8b", linestyle="--", linewidth=2.0)
        ax.scatter([adaptive["epsilon_setting"]], [adaptive[col]], color="#155d8b", marker="D", s=34)
        ax.set_xscale("log")
        ax.set_xlabel(r"Wasserstein Radius $\epsilon$", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Full-Scale Nominal Radius Sensitivity", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "baseline_nominal_epsilon.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "baseline_nominal_epsilon.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_n_comparison(frame: pd.DataFrame, output_dir: Path) -> None:
    keep = frame[frame["epsilon_setting"].isin([0.0001, 0.001, 0.003, 0.01, 0.02])]
    pivots = {}
    for metric in ["tracking_error", "violation_rate", "tail_violation_score"]:
        pivots[metric] = keep.pivot_table(index="epsilon_setting", columns="dataset", values=metric, aggfunc="mean").sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("tracking_error", "Tracking Error"),
            ("violation_rate", "Violation Rate"),
            ("tail_violation_score", "Tail Violation Score"),
        ],
    ):
        pivot = pivots[metric]
        for dataset, color, marker in [("N1", "#7f3c8d", "s"), ("N10", "#11a579", "o")]:
            if dataset in pivot.columns:
                ax.plot(pivot.index, pivot[dataset], color=color, marker=marker, linewidth=2.0, label=dataset)
        ax.set_xscale("log")
        ax.set_xlabel(r"Fixed Radius $\epsilon$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle("Effect of Offline Batch Count in Nominal Comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "baseline_n_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "baseline_n_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tini(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame.sort_values("t_ini")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))
    for ax, (col, ylabel, color) in zip(
        axes,
        [
            ("tracking_error", "Tracking Error", "#155d8b"),
            ("violation_rate", "Violation Rate", "#c97b16"),
            ("tail_violation_score", "Tail Violation Score", "#2f6f3e"),
        ],
    ):
        ax.plot(frame["t_ini"], frame[col], color=color, marker="o", linewidth=2.0)
        ax.set_xlabel(r"Initialization Window $T_{\mathrm{ini}}$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(r"Full-Scale Sensitivity to $T_{\mathrm{ini}}$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "baseline_tini_sweep.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "baseline_tini_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nominal = pd.read_csv(Path(args.nominal))
    n_comparison = pd.read_csv(Path(args.n_comparison))
    tini = pd.read_csv(Path(args.tini))

    plot_nominal_epsilon(nominal, output_dir)
    plot_n_comparison(n_comparison, output_dir)
    plot_tini(tini, output_dir)


if __name__ == "__main__":
    main()
