#!/bin/zsh
set -euo pipefail

CONFIG="${1:-experiments/paper_base.yaml}"

for STUDY in nominal epsilon horizon-matrix tini; do
  echo "Running ${STUDY} with ${CONFIG}"
  python3 experiments/run_paper_style_experiment.py --config "$CONFIG" --study "$STUDY"
done

echo "Summarizing outputs"
python3 experiments/summarize_paper_results.py
