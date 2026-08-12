# Debugging log for batch-invariant kernels

Add one config switch that enables the pinned vLLM batch-invariant kernels in both rollout and trainer processes, while leaving the default execution path unchanged.

## Initial status

`trainer.algorithm.batch_invariant` does not exist. MarinSkyRL neither exports `VLLM_BATCH_INVARIANT` to Ray workers nor initializes the vLLM override module in trainer workers. The pinned Marin vLLM does initialize its own GPU workers from that environment variable.

## Hypothesis 1

The existing vLLM implementation can be shared by both sides if MarinSkyRL propagates its environment gate to every inference worker and explicitly initializes it in each trainer worker before model construction.

## Changes to make

- Add behavior tests for the default, Ray environment propagation, nested vLLM actor propagation, trainer initialization, and unsupported generator configurations.
- Add the default-off config key and validation.
- Add a small trainer-side adapter around the pinned vLLM initializer and call it during trainer-worker construction.
- Forward `VLLM_BATCH_INVARIANT` explicitly to nested vLLM actors and log the registered CUDA overrides.

## Results

The focused tests first failed at collection because no MarinSkyRL activation module existed. After adding the shared adapter and configuration wiring, all five focused CPU tests pass. Inspection of the pinned vLLM source confirms `init_worker_distributed_environment()` calls `init_batch_invariance()` inside every vLLM GPU worker; explicit nested-actor environment forwarding is therefore sufficient for the generator side. The trainer adapter calls the same initializer after actor device pinning and before process-group and model initialization. Calling it before device pinning would let vLLM's device-capability query initialize CUDA against an incorrect visible-device set.

## Future work

- [ ] Add batch-invariant `_grouped_mm` support separately; the pinned implementation does not override it.
- [ ] Measure throughput and numerical parity on the production GPU topology before considering a default-on policy.
