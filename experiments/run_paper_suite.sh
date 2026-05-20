#!/bin/zsh
set -euo pipefail

CONFIG="${1:-experiments/paper_base.yaml}"

echo "Running nominal comparison with $CONFIG"
python3 experiments/run_paper_style_experiment.py --config "$CONFIG" --study nominal

echo "Running epsilon sweep with $CONFIG"
python3 experiments/run_paper_style_experiment.py --config "$CONFIG" --study epsilon

echo "Running horizon/matrix sweep with $CONFIG"
python3 experiments/run_paper_style_experiment.py --config "$CONFIG" --study horizon-matrix

echo "Running T_ini sweep with $CONFIG"
python3 experiments/run_paper_style_experiment.py --config "$CONFIG" --study tini
