# Debugging log for distributed debug mode

Build one launch-time diagnostic contract for distributed RL jobs and prove that healthy and failed multi-node
runs terminate with complete durable evidence.

## Initial status

Distributed diagnostics are configured independently in launcher flags, RL YAML `extra_env`, Ray runtime
environment construction, inference-engine actor propagation, and cluster-specific artifact sync. A launch can
therefore claim to be instrumented while omitting timing, stack capture, phase records, nested inference workers,
or durable copies of node-local dumps.

## Hypothesis 1

A typed trainer debug mode can be expanded before Ray initialization and reused by every launcher. Cluster
launchers only need to select the mode and, when necessary, provide a node-local staging root that their runtime
already knows how to sync.

## Changes to make

- Add an `off | distributed` trainer setting and one dependency-light resolver for its environment and paths.
- Make Iris expose the same setting and stage debug files beneath a job-scoped node-local directory.
- Propagate the resolved environment into Ray policy and inference-engine workers.
- Persist structured startup and collective-phase records, and sync Iris staging directories with bounded I/O.
- Add CPU behavior tests and an opt-in two-run Jupiter artifact contract.

## Results

- The first full CPU pass exposed two contract gaps before GPU work: the no-comments check rejected explanatory
  YAML comments, and the strict additive-config test rejected an undeclared new trainer field. The comments are
  gone and `debug_mode: off` is now part of that test's explicit additive-field set.
- Ruff removed an implicit re-export from `nccl_diagnostics.py`; the resulting full-suite import failures proved
  that the public surface needed an explicit same-name re-export. The corrected import surface passes collection.
- Local behavior tests pass for config and CLI resolution, normal-mode non-activation, worker timeout precedence,
  inference-worker projection, process receipts, collective-phase JSONL output, bounded Iris artifact sync, and
  prospective environment-variable enforcement.
- Full local suites: `cloud/iris/tests` passed 200 tests with 1 skip; `skyrl-train/tests/cpu` passed 973 tests with
  20 skips.
- Jupiter job 1247290 validated the healthy two-node run and its complete artifact inventory. Its failure arm
  correctly failed the test: withholding a rank from the first collective blocked in lazy NCCL communicator
  creation, produced no `WorkNCCL` timeout, and required the independent 150-second controller to reap it. The
  contract now warms the communicator before injecting non-arrival so it tests the production-shaped timeout
  and flight-recorder path rather than communicator bootstrap.
- Jupiter job 1247515 completed both warmed collectives' intended behavior. The healthy gang exited normally.
  In the fault gang, rank 0 timed out on sequence 2 after rank 1 stopped at sequence 1; both ranks dumped flight
  recorders, rank 0 exited by `SIGABRT`, and torchrun plus Slurm reaped the gang in 134 seconds. The test's final
  assertion was wrong because it expected rank 0 to return to Python after the fatal abort. The controller
  manifest, process manifest, and flight-recorder dump now form that rank's durable failure receipt.
- Jupiter job 1247555 passed the corrected contract at commit `5085e637`: one test passed in 135.81 seconds and
  the batch completed with exit code 0. The healthy step finished in 30 seconds. The deliberate non-arrival step
  failed in 1 minute 45 seconds, and the controller treated that bounded failure as the expected result. GPFS
  contains both process manifests and NCCL logs for each arm, both healthy completion receipts, the withheld-rank
  receipt, both failed-arm flight-recorder dumps, and a controller artifact manifest for each arm under
  `/e/scratch/jureap59/feuer1/debug-contracts/5085e637/slurm-1247555/`.

## Future work

- [ ] Evaluate whether other failure families need separate presets instead of adding high-overhead controls to
      the distributed preset.
