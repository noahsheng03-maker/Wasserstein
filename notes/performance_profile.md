# Performance Profile

This note records the first timing measurements for the paper-style
full-scale configuration in `experiments/paper_base.yaml`.

## Initial Timing Results

Measured on 2026-05-21:

- `build_bundle(config)` with full-scale parameters:
  - about `0.8 s`
  - interpretation: offline data assembly is not the main bottleneck

- `evaluate_closed_loop(..., mode="adaptive")` with `horizon_sim = 3`:
  - about `61.5 s` with the earlier default solver path
  - interpretation: roughly `20 s` per closed-loop step on average

- smoke-scale single solve benchmark:
  - `CLARABEL_default`: about `3.74 s`
  - `SCS_default`: about `7.85 s`
  - `SCS_fast` (`max_iters=2000`, `eps=1e-3`): about `0.08 s`
  - `SCS_faster` (`max_iters=1000`, `eps=5e-3`): about `0.08 s`

- smoke-scale adaptive rollout with `horizon_sim = 3`:
  - about `8.84 s` before the solver-profile update
  - about `3.25 s` after switching to `SCS_fast`

- full-scale adaptive rollout with `horizon_sim = 3`:
  - about `31.1 s` after switching to `SCS_fast`

## Main Bottleneck

The dominant cost is repeated DR-DeePC solves inside the closed-loop rollout,
not offline matrix construction.

This has a direct implication for the nominal paper-style study:

- adaptive nominal run:
  - around `250` repeated solves
- fixed-radius nominal sweep:
  - `len(epsilon_grid)` repeated closed-loop runs
  - each closed-loop run again contains around `250` solves

Therefore, the current Python/CVXPY prototype is expected to take many hours
for the full `nominal` study even before the remaining paper-style studies are
run.

## Optimization Direction

The next optimization work should target:

1. repeated solve cost
2. solver configuration
3. checkpointing and resumable study execution
4. avoiding repeated trajectories that do not change the data bundle

## Execution Strategy Update

To make the full-scale nominal study operational without changing the paper
protocol, the nominal runner now supports resumable execution:

- run adaptive nominal alone
- run only one fixed-radius nominal grid point at a time

This keeps the paper-aligned study content intact while avoiding a single
multi-hour monolithic command.
