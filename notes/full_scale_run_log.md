# Full-Scale Run Log

This note tracks the execution status of the paper-style fixed-radius studies
under `experiments/paper_base.yaml`.

## Studies

- `nominal`
  - status: running
  - note: solver profile updated to `SCS_fast` on 2026-05-21 after profiling.
  - current stage: profiling-based restart path for adaptive and per-epsilon resumable runs

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
