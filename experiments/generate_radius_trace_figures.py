from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_TITLES = {
    "noise_jump": "Noise Jump",
    "recovery": "Recovery",
    "bias_shift": "Bias Shift",
    "nominal": "Nominal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive radius trace figures.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Adaptive trace JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "paper" / "figures"),
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def infer_scenario_name(path: Path) -> str:
    name = path.stem
    if name.endswith("_adaptive_trace"):
        return name[: -len("_adaptive_trace")]
    return name


def plot_trace(trace: dict, scenario: str, output_dir: Path) -> None:
    epsilon = trace.get("epsilon_trace", [])
    pred = trace.get("prediction_error_trace", [])
    stress = trace.get("stress_trace", [])

    steps_eps = list(range(len(epsilon)))
    steps_sig = list(range(1, len(pred) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.8), sharex=True)

    axes[0].plot(steps_eps, epsilon, color="#155d8b", linewidth=2.0)
    axes[0].set_ylabel(r"Radius $\epsilon_t$", fontsize=9)
    axes[0].grid(True, alpha=0.25, linewidth=0.6)

    axes[1].plot(steps_sig, pred, color="#c97b16", linewidth=1.8)
    axes[1].set_ylabel("Prediction Mismatch", fontsize=9)
    axes[1].grid(True, alpha=0.25, linewidth=0.6)

    axes[2].plot(steps_sig, stress, color="#2f6f3e", linewidth=1.8)
    axes[2].set_ylabel("Constraint Stress", fontsize=9)
    axes[2].set_xlabel("Closed-Loop Step", fontsize=9)
    axes[2].grid(True, alpha=0.25, linewidth=0.6)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Adaptive Radius Evolution: {SCENARIO_TITLES.get(scenario, scenario)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"adaptive_radius_trace_{scenario}"
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for input_path in args.inputs:
        path = Path(input_path)
        trace = json.loads(path.read_text())
        scenario = infer_scenario_name(path)
        plot_trace(trace, scenario, output_dir)


if __name__ == "__main__":
    main()
