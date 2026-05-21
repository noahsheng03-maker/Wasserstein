# Full-Scale Run Log

This note tracks the execution status of the paper-style fixed-radius studies
under `experiments/paper_base.yaml`.

## Studies

- `nominal`
  - status: nominal epsilon sweep completed
  - note: solver profile updated to `SCS_fast` on 2026-05-21 after profiling.
  - current stage: full-scale adaptive and fixed-radius nominal points completed for the epsilon grid tracked in `outputs/full_scale/paper_style_nominal_comparison_full.csv`

- `epsilon`
  - status: nominal epsilon sweep completed for the tracked fixed-radius grid
  - note: results are consolidated in `outputs/full_scale/paper_style_nominal_comparison_full.csv`

- `horizon-matrix`
  - status: 3/4 cases completed; `hankel, 15407` full closed-loop case remains computationally prohibitive
  - note: `hankel, 15407` is solvable in one-step full-scale evaluation, but requires hundreds of seconds per step

- `tini`
  - status: completed for `T_ini = 1,...,10`
  - note: results are stored in `outputs/full_scale_tini3/paper_style_tini_sweep.csv`

- `adaptive-scenarios`
  - status: started
  - note: the first full-scale `noise_jump` comparison (`adaptive` vs `fixed_0.0001`) completed and is stored in `outputs/adaptive_compare_full_noisejump/adaptive_scenario_comparison.csv`

## Logging Convention

For each study, record:

- launch time
- completion status
- key output files
- summary observations
- solver warnings or feasibility issues
