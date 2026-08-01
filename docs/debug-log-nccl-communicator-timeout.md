# NCCL communicator timeout debug log

## Problem

An expert-parallel training run deadlocked with live ranks waiting in different distributed phases. MarinSkyRL
already configured a finite process-group timeout, asynchronous NCCL error handling, and the ProcessGroupNCCL
monitor, but the gang remained alive well beyond the intended bound.

The opt-in four-GPU fault suite originally mixed setup time with the fault deadline and used faults that could
complete without proving non-arrival. It was revised to wait for every rank to finish setup before injecting one
of three explicit faults: a missing EP-subgroup participant, a missing WORLD participant, or a rank exit during a
WORLD collective. Controller teardown is independently bounded so a failed contract cannot hang the test runner.

## Evidence

On a four-GPU GH200 node, rank exit terminated through torchrun supervision. Both non-arrival modes remained
hung with asynchronous error handling, blocking wait, and a shortened monitor dump grace period. PyTorch's
configuration banners confirmed that the requested process-group watchdog settings reached the workers, but no
timeout was detected.

Enabling `TORCH_NCCL_USE_COMM_NONBLOCKING=1` and setting `TORCH_NCCL_NONBLOCKING_TIMEOUT` to the same eight-second
deadline as the process group made both non-arrival modes terminate within the fault-suite bound. This behavior
is reproducible, but the precise blocked NCCL frame was not captured; attributing the gap to communicator work
that precedes creation of a watchdog-visible `WorkNCCL` remains an inference from the discriminating test.

## Fix

Ray workers now enable nonblocking NCCL communicator operations and derive their timeout from the canonical
worker collective timeout. The GPU fault suite imports the same environment builder, preventing the test and
production configuration from drifting apart.

## Verification

- CPU tests assert that the runtime environment enables communicator nonblocking mode and keeps the communicator
  timeout aligned with both the default and a configured worker collective timeout.
- The opt-in GPU test asserts bounded teardown for WORLD and EP-subgroup non-arrival as well as rank exit.
- The repository lint and review passes cover the implementation and test changes.
