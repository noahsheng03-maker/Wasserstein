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

## Paper-Aligned Experiment Protocol

The repository is being organized to preserve the full experiment logic of the
reference paper before adding the adaptive-radius extension. The current target
is to reproduce the same protocol dimensions discussed in the paper:

- nominal closed-loop tracking with a fixed Wasserstein radius
- sensitivity to `epsilon`
- sensitivity to the number of repeated batches `N`
- sensitivity to offline data length / column count
- comparison between Page and Hankel data matrices
- sensitivity to the initialization horizon `T_ini`

The adaptive-radius method is evaluated on top of this paper-style protocol
instead of replacing it with a simplified benchmark.

## Current Implementation Status

- `experiments/run_paper_style_experiment.py` now runs the paper-style study
  structure and logs the outputs as CSV/JSON artifacts.
- `experiments/run_full_scale_studies.sh` runs the four fixed-radius paper-style
  studies sequentially and then builds summary tables.
- The current system backend is a hover-like linear surrogate with the same
  input/output dimensions and horizon scales as the paper.
- A higher-fidelity nonlinear quadcopter backend still needs to be integrated
  before claiming a full reproduction of the original numerical study.

## Immediate Next Steps

1. Run and validate the full paper-style study suite.
2. Add structured result tables and plotting scripts.
3. Integrate a higher-fidelity quadcopter-style backend.
4. Extend the reproduced protocol with adaptive-radius comparisons.
