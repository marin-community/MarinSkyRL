# Debugging log for Megatron PPO diagnostics

Determine whether Megatron's zero clipping and log-ratio metrics are missing diagnostics or a genuine unit policy ratio, then make that state explicit to operators.

## Initial status

Three Megatron runs published zero clipping and log-ratio metrics with `log_ratio_diagnostics_failed=0`; an FSDP2 control published nonzero values.

## Hypothesis 1

Megatron drops the metrics produced by the policy objective or log-ratio monitor.

## Changes to make

Inspect the policy loss, pipeline closure, worker aggregation, and the exact revisions used by the reported runs.

## Results

Refuted. Since commit `4c47f42b`, the Megatron pipeline closure computes clipping from its training-forward log probabilities, accumulates log-ratio diagnostics, broadcasts them from the last pipeline stage, and reduces them across workers. A CPU pipeline test already produces nonzero log-ratio metrics from a synthetic delta. The affected `tt-x9` run used that commit and reported `log_ratio_diagnostics_failed=0`.

## Hypothesis 2

The reported Megatron ratio is genuinely one because the old-logprob inference forward and the first/only training forward use identical weights, tokens, precision, and deterministic model execution.

## Changes to make

Compare the run geometry and forward modes, then expose the exact-unit condition independently of zero-valued clipping metrics and warn when it makes the configured bounds inert.

## Results

Confirmed by code and run evidence. Both paths run before the optimizer step; Megatron switches from eval to train, but the model dropout is zero. The affected fully-async configuration has one update epoch and one mini-batch for the whole train batch. Its nonzero TIS ratio simultaneously proves that rollout-to-training drift is measured while recomputed-old-to-training drift is exactly zero. The FSDP2 control's nonzero recomputation delta is backend numerical variation, not an optimizer update between the two forwards.

## Future work

- [ ] Validate the new unit-ratio metric and warning on the next Megatron GPU run.
