from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
N10_PATH = REPO_ROOT / "outputs" / "full_scale" / "paper_style_nominal_comparison_full.csv"
N1_PATH = REPO_ROOT / "outputs" / "full_scale_n1" / "paper_style_nominal_comparison_full.csv"
OUT_PATH = REPO_ROOT / "outputs" / "n_comparison_nominal.csv"


def main() -> None:
    n10 = pd.read_csv(N10_PATH).assign(dataset="N10")
    n1 = pd.read_csv(N1_PATH).assign(dataset="N1")
    merged = pd.concat([n1, n10], ignore_index=True)
    merged = merged.sort_values(["controller_mode", "dataset"]).reset_index(drop=True)
    merged.to_csv(OUT_PATH, index=False)
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
