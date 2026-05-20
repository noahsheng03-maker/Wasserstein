# Paper Protocol Alignment

This note tracks how the repository experiments align with
_Distributionally Robust Chance Constrained Data-Enabled Predictive Control_.

## Original Paper Experiment Dimensions

The original paper studies the following axes in its numerical section:

1. Nominal quadcopter closed-loop trajectory under DR-DeePC.
2. Sensitivity to the Wasserstein radius `epsilon`.
3. Effect of the number of repeated experiments `N`.
4. Effect of offline data length / number of columns.
5. Comparison between Page and Hankel matrices.
6. Sensitivity to the initialization horizon `T_ini`.

## Repository Mapping

- `experiments/paper_base.yaml`
  Centralizes the paper-aligned default parameters, including `T_ini = 6`,
  `T_f = 25`, `alpha = 0.1`, repeated offline batches, and the paper-style
  sweep grids.

- `experiments/run_paper_style_experiment.py`
  Runs the paper-style protocol and currently exposes:
  - `--study nominal`
  - `--study epsilon`
  - `--study horizon-matrix`
  - `--study tini`
  - `--study all`

- `experiments/paper_smoke.yaml`
  Keeps the same protocol structure but shortens the closed-loop rollout and
  grid sizes so the implementation can be debugged without changing the full
  paper-aligned configuration.

- `experiments/run_paper_suite.sh`
  Sequential launcher for the four paper-style studies.

- `experiments/run_full_scale_studies.sh`
  Sequential launcher for the full paper-aligned configuration followed by CSV
  summary generation.

- `experiments/summarize_paper_results.py`
  Builds compact summary tables from the raw study CSV outputs.

## What Is Already Aligned

- Same experiment logic and sweep dimensions as the paper.
- Same role of repeated offline input batches.
- Same use of Page/Hankel alternatives.
- Same emphasis on `epsilon`, `N`, offline horizon, and `T_ini`.

## What Still Needs to Be Upgraded

- The current closed-loop backend is a structured linear hover surrogate rather
  than the paper's high-fidelity nonlinear quadcopter simulator.
- Plotting scripts still need to be added so the results can be compared with
  the original figures more directly.
- The adaptive-radius extension still needs to be run against every paper-style
  study axis after the fixed-radius reproduction is stable.
