# Debug log: shaping-invariant pass@k

## Reported behavior

`reward/avg_pass_at_8`, and pass@k generally, changes when reward shaping changes even when the underlying verifier outcomes do not.

## Hypotheses

1. **Confirmed by code reading:** `get_metrics_from_generator_output` derives both the optimization-score mean and pass@k from the post-shaping `rewards` channel.
2. **Confirmed by code reading:** the TerminalBench generator computes `original_reward` before shaping but does not preserve it in `GeneratorOutput`.

## Contract

- `reward/avg_raw_reward` continues to describe the reward used for optimization.
- pass@k uses the unshaped verifier outcome when the generator provides it.
- Generators without a separate unshaped channel retain the existing rewards-based behavior.
- Invalidated trajectories must have both shaped and unshaped rewards reset to zero.

## Experiments

1. Added a regression test whose shaped rewards are all positive but whose unshaped outcomes contain only one successful prompt group.
   - Before the fix: **failed**, reporting pass@2 = 1.0 instead of 0.5.
   - This isolates the metric calculation from any particular shaper implementation.

## Change under test

- Add an optional `unshaped_rewards` generator-output channel.
- Make pass@k prefer that channel while keeping `mean_raw_reward` on the optimization reward.
- Preserve TerminalBench's verifier outcome through shaping and reset it whenever error handling invalidates a trajectory.

## Results

- Focused generator metrics and concatenation tests: **20 passed**.
- Full trainer CPU suite: **965 passed, 20 skipped**.
- Changed-file lint and formatting: **passed**.

The regression now evaluates two different shaped reward vectors over the same verifier outcomes. Their optimization-score means differ, while both report the same pass@2 value.

## Advisory review

- Fixed a propagation gap in dynamic replacement and filtering so sampled trajectories keep their corresponding unshaped outcomes.
- Consolidated failed-trajectory clearing into one operation so training and metric reward channels reset together.
- Retained the established `reward/avg_raw_reward` metric name to avoid breaking existing dashboards. Its docstring now distinguishes that optimization-reward mean from the new unshaped outcome channel.
