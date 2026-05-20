from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def summarize_nominal(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["controller_mode"]).copy()
    ordered["rank_tracking"] = ordered["tracking_error"].rank(method="min")
    ordered["rank_violation"] = ordered["violation_rate"].rank(method="min")
    ordered["rank_tail"] = ordered["tail_violation_score"].rank(method="min")
    return ordered


def summarize_epsilon(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_value, group in frame.groupby("N"):
        best_tracking = group.loc[group["tracking_error"].idxmin()]
        best_safe = group.loc[group["tail_violation_score"].idxmin()]
        rows.append(
            {
                "N": int(n_value),
                "best_tracking_epsilon": float(best_tracking["epsilon"]),
                "best_tracking_error": float(best_tracking["tracking_error"]),
                "best_safe_epsilon": float(best_safe["epsilon"]),
                "best_safe_tail_score": float(best_safe["tail_violation_score"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_horizon_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["matrix_type", "offline_horizon"]).reset_index(drop=True)


def summarize_tini(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["t_ini"]).reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-style experiment CSV outputs.")
    parser.add_argument(
        "--outputs",
        default=str(REPO_ROOT / "outputs"),
        help="Directory containing paper-style CSV outputs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.outputs)
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    nominal = load_csv(out_dir / "paper_style_nominal_comparison.csv")
    if nominal is not None:
        nominal_summary = summarize_nominal(nominal)
        nominal_summary.to_csv(summary_dir / "nominal_summary.csv", index=False)
        print("\n=== nominal_summary ===")
        print(nominal_summary)

    epsilon = load_csv(out_dir / "paper_style_epsilon_sweep.csv")
    if epsilon is not None:
        epsilon_summary = summarize_epsilon(epsilon)
        epsilon_summary.to_csv(summary_dir / "epsilon_summary.csv", index=False)
        print("\n=== epsilon_summary ===")
        print(epsilon_summary)

    horizon_matrix = load_csv(out_dir / "paper_style_horizon_matrix_sweep.csv")
    if horizon_matrix is not None:
        horizon_matrix_summary = summarize_horizon_matrix(horizon_matrix)
        horizon_matrix_summary.to_csv(summary_dir / "horizon_matrix_summary.csv", index=False)
        print("\n=== horizon_matrix_summary ===")
        print(horizon_matrix_summary)

    tini = load_csv(out_dir / "paper_style_tini_sweep.csv")
    if tini is not None:
        tini_summary = summarize_tini(tini)
        tini_summary.to_csv(summary_dir / "tini_summary.csv", index=False)
        print("\n=== tini_summary ===")
        print(tini_summary)


if __name__ == "__main__":
    main()
