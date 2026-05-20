# Research Outline

## Topic

Adaptive Wasserstein Radius Selection for Distributionally Robust Data-Enabled Predictive Control

## Core Question

How should the Wasserstein ambiguity radius be selected and updated so that DR-DeePC remains safe under changing uncertainty while avoiding unnecessary conservatism?

## Main Idea

- Keep the inner DR-DeePC formulation unchanged.
- Add an outer-loop ambiguity-radius adaptation mechanism.
- Drive the update using prediction mismatch, constraint stress, and distribution shift indicators.
- Enforce a safeguarded lower bound to avoid over-optimistic radius shrinkage.

## Short-Term Tasks

1. Complete the draft paper sections.
2. Define the adaptive update law precisely.
3. Implement the fixed-radius baseline.
4. Build reproducible experiments.
