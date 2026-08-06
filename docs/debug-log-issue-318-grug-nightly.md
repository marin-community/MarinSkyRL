# Debugging log for issue 318 Grug nightly

Determine why scheduled run 31096906530 failed and restore the Grug GB200 nightly lane.

## Initial status

The `gsm8k-h100` job passed. Iris scheduled the four-GB200 Grug job, resolved the frozen environment, and ran
the lifecycle test. Pytest failed in `_assert_engine_weights` with `KeyError: 'ep_rank'` while checking serving
expert ownership.

## Hypothesis 1

PR #276 added the ownership assertion against the wrong level of the existing weight-readback schema. Each
worker result stores its rank metadata in `rank_values["__ranks__"]`; individual named-weight entries contain
weight metadata but no `ep_rank`. The failure was introduced after the last successful Grug run at commit
`5974892a`, and the producer still returns `ep_rank` in the worker-level `__ranks__` mapping.

## Changes to make

Read the worker's serving EP rank once from `rank_values["__ranks__"]` and attribute each present expert to that
rank. Keep the exact serving-weight and single-owner assertions unchanged.

## Results

Run 31096906530 completed the first rollout, training, checkpoint resume, and weight broadcast on commit
`4725d66e`. It then failed while recording the first serving expert's owner. Serving-weight inspection did not
complete, and the second rollout did not start. A local synthetic two-rank schema probe passed after the change:
expert 0 was attributed to serving EP rank 0 and expert 4 to serving EP rank 1. Pytest collected all four tests in
`test_grug_fsdp2_rl_cycle.py`, the CPU nightly script test passed, and the changed-file Ruff checks passed.

Branch validation run 31099078221 was canceled at the workflow's 100-minute timeout before the GB200 job
reached pytest. It does not qualify the lifecycle fix.

## Future work

- [ ] Run the four-GB200 lifecycle gate from the published branch.
