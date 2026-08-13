# Debugging log for Harbor analysis schema

Make RL analysis consume current Harbor trial results and ATIF trajectories without dropping scored fields or
replicates.

## Initial status

The loader treats every `result.json` as a self-contained trace. Current Harbor trials store rewards and errors
under nested result fields and steps in `agent/trajectory.json`; the run also contains one job aggregate result.

## Hypothesis 1

Joining trial results to their sibling trajectories and explicitly normalizing Harbor fields will restore reward,
turn, context, error, usage, and task identity analysis.

## Changes to make

Add current-schema fixtures, distinguish job aggregates from trials, define context as peak request prompt tokens,
and aggregate matched replicates by task instead of overwriting them.

## Results

The regression tests initially failed because Harbor rewards, trajectories, structured task IDs, and
replicate-aware statistics were not implemented. After normalization, the reported live corpus loads 208/208 scored trials with
reward sum 110, mean reward 0.528846, and non-empty turn, peak-context, and cumulative-usage fields for every
trial. The 139-test infra suite passes.

## Future work

- [ ] None.
