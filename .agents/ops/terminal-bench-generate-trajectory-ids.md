# Debugging log for terminal-bench generation trajectory IDs

Restore the standalone Terminal-Bench generation entrypoint after the trajectory-runner request contract changed.

## Initial status

The Iris A0 base-model evaluation failed on all four attempts before producing trials. The first application error
was `KeyError: 'trajectory_ids'` in `HarborTrajectoryRunner._run`; the later zombie-process messages came from Ray
teardown.

## Hypothesis 1

`TerminalBenchGenerateExp.run` constructs a one-prompt-per-task request directly and omits the trajectory IDs that
the Harbor runner requires. The training path already has a request builder that repeats prompts and assigns one
`TrajectoryID` per sample.

## Changes to make

Add a regression test that observes the request passed to the trajectory runner. It must contain eight repetitions
per task, stable task/repetition IDs, and evaluation batch metadata. Confirm that the test fails before changing the
entrypoint.

## Results

The regression test failed before the fix because the entrypoint sent two prompts instead of sixteen and omitted
the corresponding trajectory IDs. Routing the standalone entrypoint through `prepare_trajectory_request` produced
eight repetitions per task, stable task/repetition IDs, repeated environment metadata, backend sampling parameters,
and evaluation batch metadata. The focused regression test passes.

## Future work

- [ ] Relaunch the A0 evaluation after this fix is deployed.
