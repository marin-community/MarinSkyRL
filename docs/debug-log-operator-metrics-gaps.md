# Debugging log for operator metrics gaps

Reproduce and fix east-08a log synchronization and the missing agentic reward EMA table.

## Initial status

`sync_rl_logs.py --cluster cw-us-east-08a` is rejected by argparse because the cluster is absent from the supported cluster mapping. `parse_skyrl_metrics.py --format agentic` parses rewards but only computes and prints checkpoint EMA data in standard mode.

## Hypothesis 1

Adding east-08a to the log sync cluster mapping will make the existing finelog path accept the cluster while preserving the shared CoreWeave object-store source.

## Changes to make

Extend the cluster contract test before changing the mapping.

## Results

The regression test failed because `KCFG` contained only `cw-us-east-02a` and `cw-rno2a`. Adding `cw-us-east-08a` makes argparse accept the cluster while retaining the shared CoreWeave object-store credentials.

## Hypothesis 2

Checkpoint selection can consume the metrics already parsed for either log format. Computing it after parsing will emit the same EMA table for agentic and standard runs without maintaining two extraction paths.

## Changes to make

Add an agentic CLI regression test that asserts the printed trailing-5 EMA table, then generalize selection over parsed metrics and wire it into both formats.

## Results

The agentic CLI regression test parsed three reward steps but found no selector heading or table. Selection now consumes the metrics already parsed by either format, prints the EMA table for both, and writes the same table into both report variants. The two focused suites pass with 14 tests.

## Future work

- [ ] None.
