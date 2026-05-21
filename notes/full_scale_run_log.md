# Full-Scale Run Log

This note tracks the execution status of the paper-style fixed-radius studies
under `experiments/paper_base.yaml`.

## Studies

- `nominal`
  - status: nominal epsilon sweep completed
  - note: solver profile updated to `SCS_fast` on 2026-05-21 after profiling.
  - current stage: full-scale adaptive and fixed-radius nominal points completed for the epsilon grid tracked in `outputs/full_scale/paper_style_nominal_comparison_full.csv`

- `epsilon`
  - status: pending

- `horizon-matrix`
  - status: pending

- `tini`
  - status: pending

## Logging Convention

For each study, record:

- launch time
- completion status
- key output files
- summary observations
- solver warnings or feasibility issues
