# Wasserstein

Research repository for adaptive Wasserstein radius selection in distributionally robust data-enabled predictive control (DR-DeePC).

## Current Focus

This project studies how to replace a fixed Wasserstein ambiguity radius with an adaptive radius update mechanism in DR-DeePC. The goal is to improve the performance-safety trade-off under nonstationary uncertainty while reducing manual tuning effort.

## Repository Structure

- `paper/`: manuscript drafts and LaTeX sources
- `notes/`: research notes, experiment plans, and outlines
- `src/deepc/`: baseline DeePC and DR-DeePC components
- `src/adaptive_radius/`: adaptive Wasserstein radius logic
- `src/utils/`: shared utilities
- `experiments/`: experiment scripts organized by scenario

## Planned Experiment Scenarios

- `nominal/`: stationary uncertainty
- `noise_jump/`: abrupt variance increase
- `distribution_shift/`: nonstationary or biased uncertainty
- `recovery/`: disturbance increase followed by recovery

## Immediate Next Steps

1. Finalize the paper draft structure.
2. Implement a fixed-radius DR-DeePC baseline.
3. Add adaptive radius update rules and safeguards.
4. Build reproducible experiment scripts and result logging.
