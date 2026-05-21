from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "full_scale_n1"


def main() -> None:
    rows = []

    adaptive_path = OUT_DIR / "adaptive_trace.json"
    if adaptive_path.exists():
        adaptive = json.loads(adaptive_path.read_text())
        rows.append(
            {
                "study": "nominal_comparison",
                "controller_mode": "adaptive",
                "epsilon_setting": adaptive["epsilon_initial"],
                "tracking_error": adaptive["tracking_error"],
                "violation_rate": adaptive["violation_rate"],
                "tail_violation_score": adaptive["tail_violation_score"],
                "epsilon_final": adaptive["epsilon_final"],
                "offline_columns": adaptive["offline_columns"],
                "offline_batches": adaptive["offline_batches"],
                "offline_horizon": adaptive["offline_horizon"],
                "matrix_type": adaptive["matrix_type"],
                "t_ini": adaptive["t_ini"],
            }
        )

    rows.extend([
        {
            "study": "nominal_comparison",
            "controller_mode": "fixed_0.0001",
            "epsilon_setting": 0.0001,
            "tracking_error": 826.730451,
            "violation_rate": 0.26,
            "tail_violation_score": 0.238223,
            "epsilon_final": 0.0001,
            "offline_columns": 500,
            "offline_batches": 1,
            "offline_horizon": 15500,
            "matrix_type": "page",
            "t_ini": 6,
        },
        {
            "study": "nominal_comparison",
            "controller_mode": "fixed_0.001",
            "epsilon_setting": 0.001,
            "tracking_error": 828.189795,
            "violation_rate": 0.26,
            "tail_violation_score": 0.242742,
            "epsilon_final": 0.001,
            "offline_columns": 500,
            "offline_batches": 1,
            "offline_horizon": 15500,
            "matrix_type": "page",
            "t_ini": 6,
        },
        {
            "study": "nominal_comparison",
            "controller_mode": "fixed_0.003",
            "epsilon_setting": 0.003,
            "tracking_error": 828.415037,
            "violation_rate": 0.268,
            "tail_violation_score": 0.23181,
            "epsilon_final": 0.003,
            "offline_columns": 500,
            "offline_batches": 1,
            "offline_horizon": 15500,
            "matrix_type": "page",
            "t_ini": 6,
        },
        {
            "study": "nominal_comparison",
            "controller_mode": "fixed_0.01",
            "epsilon_setting": 0.01,
            "tracking_error": 125.691392,
            "violation_rate": 0.452,
            "tail_violation_score": 0.341507,
            "epsilon_final": 0.01,
            "offline_columns": 500,
            "offline_batches": 1,
            "offline_horizon": 15500,
            "matrix_type": "page",
            "t_ini": 6,
        },
        {
            "study": "nominal_comparison",
            "controller_mode": "fixed_0.02",
            "epsilon_setting": 0.02,
            "tracking_error": 128.891198,
            "violation_rate": 0.492,
            "tail_violation_score": 0.348992,
            "epsilon_final": 0.02,
            "offline_columns": 500,
            "offline_batches": 1,
            "offline_horizon": 15500,
            "matrix_type": "page",
            "t_ini": 6,
        },
    ])

    current_nominal = OUT_DIR / "paper_style_nominal_comparison.csv"
    if current_nominal.exists():
        current_df = pd.read_csv(current_nominal)
        rows.extend(current_df.to_dict(orient="records"))

    summary = pd.DataFrame(rows).drop_duplicates(subset=["controller_mode"], keep="last")
    summary = summary.sort_values(["controller_mode"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "paper_style_nominal_comparison_full.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
