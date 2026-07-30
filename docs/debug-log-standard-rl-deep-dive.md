# Debugging log for standard-RL deep dives

Make the RL deep-dive procedure produce reproducible verdicts for dataset-backed runs that have no agent harness or trial artifacts.

## Initial status

The deep-dive skill requires rollout artifacts and refers to agentic per-trial timing. The diagnostics runbook has no standard-RL mode or evidence substitutions, so a healthy standard run can produce an `ERROR` gate.

## Hypothesis 1

An explicit mode declaration plus mode-specific rollout and duty-cycle gates will remove the unsatisfiable requirements while preserving the common restart, liveness, resource, and dynamics gates.

## Changes to make

Define agentic and standard evidence for rollout quality, liveness, demand, and timing. Require `parse_skyrl_metrics.py --format standard` for dataset-backed runs and mark per-trial duty cycle as not applicable there.

## Results

The revised skill requires an explicit mode from the launched configuration. The runbook now maps
standard rollout, liveness, demand, and phase evidence without requiring agentic artifacts. The
report marks per-trial duty cycle `N/A` for standard runs, and both documents require the matching
metrics parser format.

## Future work

- [x] Review the skill and runbook together for contradictory gate language.
