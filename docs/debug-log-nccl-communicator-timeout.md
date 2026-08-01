# NCCL communicator timeout debug log

## Initial problem

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

## Initial fix

PR #264 enabled nonblocking NCCL communicator operations on every Ray worker and derived their timeout from the
canonical worker collective timeout. The GPU fault suite imported the same environment builder.

## Regression

Eight TaskTrove training arms subsequently failed before banking a step. Every failure reached the
expert-parallel token dispatch and raised NCCL error 7 (`ncclInProgress`) from `all_to_all_single`. Removing #264
from the deployed revision allowed replacement arms to reach policy training.

The original fault suite covered only failing `all_reduce` calls. It established that nonblocking communicator
mode bounded those injected failures, but did not establish that ordinary training collectives remained valid.
NCCL documents that nonblocking communicators may return `ncclInProgress` from NCCL calls and require callers to
poll for completion. The training path treats `all_to_all_single` as synchronous, so enabling this communicator
mode globally violated its API assumption.

## Corrective fix

Production runtime environments no longer enable or forward `TORCH_NCCL_USE_COMM_NONBLOCKING` and
`TORCH_NCCL_NONBLOCKING_TIMEOUT`. Process-group timeouts, asynchronous error handling, monitoring, heartbeat
limits, and flight-recorder configuration remain enabled.

The opt-in fault suite retains nonblocking communicators only inside its subprocess environment. It now also
runs a healthy EP-subgroup `all_to_all_single` with production communicator settings and validates every value
received by every rank. This separates two contracts that #264 had conflated: bounded teardown under an injected
communicator fault, and successful execution of the healthy training collective.

## Verification

- A CPU regression test fails if production runtime construction enables or forwards communicator nonblocking.
- The opt-in GPU suite checks a healthy EP all-to-all using production communicator settings, then separately
  asserts bounded teardown for WORLD and EP-subgroup non-arrival and rank exit under its diagnostic settings.
- The repository lint and review passes cover the implementation and test changes.
