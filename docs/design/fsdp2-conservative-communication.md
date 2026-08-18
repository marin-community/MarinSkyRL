# Conservative FSDP2 communication mode

Status: reviewed implementation plan

## Decision

Add `trainer.policy.fsdp_config.communication_mode` and the corresponding critic setting as a `StrEnum` with two values:

- `standard`: retain the current FSDP2 behavior.
- `conservative`: reduce simultaneous work on the FSDP and expert-parallel communication paths.

The mode is an operational mitigation for a recurring multi-node stall. It is not the default. It must not alter model numerics, sharding geometry, gradient accumulation, or `reshard_after_forward`.

The initial conservative mode has two controls:

1. Set `CUDA_DEVICE_MAX_CONNECTIONS=1` in the selected policy or critic actor process before CUDA initialization.
2. Suppress FSDP2's inferred backward prefetch schedule using a pinned-Torch compatibility helper that is verified by a GPU trace test.

Two further controls are harness ablations, not part of the production mode until measurements justify them:

3. Block host submission at the backward boundary until the current stream completes.
4. Cap NCCL CTAs to test whether lower per-collective GPU and network pressure improves progress across communicators.

Production configuration exposes the single mode, not permanent booleans for each experiment. The diagnostic harness selects each control independently and records physical time intervals, so the final conservative composition can be reduced to the smallest effective set.

## Evidence and hypothesis

The preserved flight-recorder dump does not show a collective-order mismatch. All members of one inter-node FSDP group enqueue the same approximately 151-million-element reduce-scatter. Some ranks do not complete it, while the other FSDP groups finish and enter the next expert-parallel all-to-all. The waiting expert-parallel collectives are downstream symptoms.

The production-shaped 16-GPU tests have completed the same FSDP and expert-parallel schedule. That evidence argues against a deterministic ordering defect. Assuming the fabric is healthy, a remaining software hypothesis is a progress failure caused by concurrent CUDA streams or communicators under production timing and load.

FSDP2 creates separate high-priority CUDA streams for all-gather copy-in, all-gather, and reduce-scatter. Its default backward schedule overlaps the next parameter all-gather, current gradient computation, and an earlier reduce-scatter. PyTorch documents this overlap and the public API for overriding backward prefetch. The FSDP paper reports that backward prefetch improved GPT-175B throughput by about 18 percent, so disabling it has a material expected cost and belongs behind an opt-in mode. See the [FSDP2 communication documentation](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.fsdp.fully_shard.md) and [PyTorch FSDP paper](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf).

NVIDIA documents that `CUDA_DEVICE_MAX_CONNECTIONS` controls compute and copy work queues. Setting it below the number of active streams can create false dependencies and serialize work. A value of one does not guarantee that NCCL collectives run serially; it is a scheduling hypothesis that deliberately reduces the runtime's opportunities for concurrent progress. See the [CUDA environment-variable documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections).

NCCL's `NCCL_MAX_CTAS` caps the GPU resources that one collective can consume. Lower values can reduce peak throughput as well as contention, so the harness must sweep values rather than embedding an arbitrary cap in production mode. See the [NCCL environment-variable documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-max-ctas).

## Communication sites

| Site | Current behavior | Planned experiment | Intended observation |
| --- | --- | --- | --- |
| Policy or critic actor creation | Actors inherit the job environment | Attach `CUDA_DEVICE_MAX_CONNECTIONS=1` to only the conservative role's Ray actor `runtime_env` | Whether fewer CUDA work queues reduce time overlap between FSDP and EP collectives |
| FSDP hierarchy finalization | FSDP2 infers the next backward all-gather from reverse post-forward order | Suppress inferred backward prefetch on every FSDP module | Whether removing the next all-gather reduces the three-way all-gather, compute, and reduce-scatter overlap |
| Backward host boundary | The current stream waits on the final reduce-scatter event, but the host may enqueue the next microbatch's work | In the harness, call `torch.cuda.current_stream().synchronize()` after backward | Whether limiting host queue depth changes the amount of pending cross-stream work; this is not a correctness fix |
| NCCL kernel resources | NCCL selects CTA count for throughput | In fresh harness processes, sweep `NCCL_MAX_CTAS` values including unset | Whether a less aggressive collective leaves enough GPU or transport resources for another communicator to progress |
| Gradient accumulation | FSDP reduce-scatters every microbatch | No change | Avoid retaining full unsharded gradients, especially with CPU offload |
| Forward resharding | Controlled by `reshard_after_forward` | No change | Avoid silently trading communication for a large memory increase |
| Expert dispatch backend | Torch all-to-all or the configured backend | No change | Keep this plan independent of DeepEP and routing changes |

## Pinned-Torch prefetch behavior

Pinned Torch 2.11 does not treat an empty explicit list as “disable.” Its pre-backward hook selects the implicit default whenever the stored list is empty. Therefore, `set_modules_to_backward_prefetch([])` preserves the behavior that this plan needs to suppress.

The candidate compatibility helper calls `module.set_modules_to_backward_prefetch([module])`. The nonempty list suppresses the default path. The explicit self-unshard should then be a no-op because the current module was just unsharded. This behavior is an inference from the pinned implementation, not a documented PyTorch disable contract.

Before production code uses the helper, a real two-rank GPU trace must prove that:

- the next module's all-gather is absent from the current module's pre-backward window;
- the self-target does not issue another all-gather;
- losses, gradients, and optimizer deltas match standard mode.

The helper must assert the supported Torch version. A Torch upgrade must fail the compatibility test and require revalidation. If the trace does not prove the behavior, omit this control and open an upstream request for an explicit disable API rather than depending on private FSDP state.

## Configuration and actor scope

Define `FSDPCommunicationMode(StrEnum)`. Add it to policy and critic FSDP configuration with `standard` as the base default. Conservative mode is invalid unless the role uses FSDP2. The reference model has no backward pass, so it does not receive this setting.

Do not modify `prepare_runtime_environment()`: it supplies the job-level Ray environment and would also affect vLLM, reference, rollout, and Harbor actors. Instead:

1. Resolve a role-specific actor environment in `trainer.build_models()`.
2. Pass it through `PPORayActorGroup`.
3. Merge its `env_vars` into the inherited job runtime environment without replacing the worker setup hook or other runtime fields, then attach it to every policy or critic `ray_actor_type.options(runtime_env=...)` call before actor import and CUDA initialization.
4. Leave reference and inference actors unchanged.

Use the existing environment-management abstraction to own the setting. Detect a conflicting inherited value before applying the role overlay and fail with both values in the error. Actor startup must assert and log the observed value. Policy and critic may select modes independently.

Log the resolved mode and effective controls unconditionally once per actor. When distributed debug mode is active, write a role-specific process manifest after the role is known; do not rely on the earlier bootstrap manifest.

## Implementation plan

1. Add and validate `FSDPCommunicationMode` in the root Hydra configuration.
2. Add role-specific actor runtime environments to `PPORayActorGroup` and preserve the current actor options when no override is present.
3. Implement the prefetch compatibility helper after the GPU spike proves the self-target behavior. Apply it after expert and root sharding are complete.
4. Add structured observability for mode, CUDA connection count, prefetch policy, Torch version, and harness-only controls.
5. Extend the production-shaped harness with independent connection-limit, prefetch, host-drain, and CTA-cap switches. Run connection and CTA variants in fresh actor processes because environment variables are not reliable after CUDA initialization.
6. Select the production composition from measured overlap and throughput. Do not automatically promote the host drain or CTA cap.

## Alternatives not included

### Custom reduce-scatter wrapper

`FSDPModule.set_custom_reduce_scatter()` changes allocation and collective implementation, but FSDP2 invokes it inside its reduce-scatter stream. A wrapper around the default collective does not by itself reduce cross-stream concurrency. The test harness may use a wrapper for timestamps, not as a mitigation.

### Gradient accumulation without synchronization

`set_requires_gradient_sync(False)` suppresses reduce-scatter on intermediate microbatches but retains unsharded gradients. That increases device memory substantially and conflicts with CPU-offloaded configurations. It is a separate memory-throughput feature.

### `reshard_after_forward=false`

Keeping full parameters resident removes backward all-gathers but increases peak memory. Earlier no-reshard experiments did not establish that it prevents the stall. The existing setting remains an independent ablation.

### NCCL barriers and private FSDP streams

An extra NCCL barrier adds another collective and can create a new dependency cycle across communicators. Replacing FSDP2's private streams would depend on unsupported internals. Neither belongs in the first implementation.

### Device-wide synchronization

FSDP2 defaults `is_last_backward=True` and makes the current stream wait on the pending reduce-scatter event after each backward. `torch.cuda.synchronize()` is not needed for correctness. The narrower harness ablation waits for the current stream from the host; it tests launch pacing without claiming a new device dependency.

## Test plan

CPU behavior tests:

- `standard` produces no actor-specific runtime environment and retains current actor construction.
- `conservative` resolves `CUDA_DEVICE_MAX_CONNECTIONS=1` for only the selected FSDP2 policy or critic actors.
- Reference and inference actors do not inherit the FSDP-only override.
- Conservative mode is rejected for FSDP1, Megatron, and DeepSpeed.
- A conflicting environment value fails with both values in the error.
- Structured startup metadata records the effective role policy.

GPU behavior tests:

- A two-rank trace proves the pinned-version prefetch behavior using real FSDP2 modules and public custom collective callbacks. A test of helper call shape is insufficient.
- A gradient-accumulation test compares loss, expert, router, and dense gradients, and optimizer-step parameter deltas between modes.
- A memory test verifies that conservative mode does not retain full gradients across microbatches.
- Boundary behavior is tested by observing queued CUDA work, not by mocking a synchronization call.
- The 16-GPU FSDP4/EP4 harness runs standard mode plus each isolated control and the candidate combination.

Extend `WorkInfo` or an equivalent recorder with communicator role, rank, dtype, and input/output bytes. Obtain physical start, finish, and active duration from CUDA events or profiler timestamps, then compute interval intersections between FSDP and EP operations. CPU completion-hook timestamps and enqueue/completion sequence numbers do not prove physical overlap.

Instrumentation must not be assumed neutral. Prefer existing work timing and module-boundary ranges where they supply the required data. If custom collective callbacks are necessary, compare a wrapped standard run with an unwrapped standard run before interpreting the mitigation variants.

## Acceptance and rollout

The PR may merge with the mode defaulting to `standard` only after the pinned-Torch trace and CPU tests pass. It must not become a Jupiter campaign default until the production-shaped run shows all of the following:

- the candidate has lower measured FSDP/EP interval overlap than standard mode;
- loss, gradients, and optimizer deltas remain equivalent;
- peak device and host memory stay within defined benchmark tolerances;
- the throughput cost is measured over repeated runs with confidence intervals;
- the diagnostic recorder identifies which control changes the schedule.

Run matched production arms across more than one node set with flight recording enabled. Compare wedge incidence per aggregate node-hour rather than treating one surviving arm as proof. If the mode still wedges at the same reduce-scatter rate, retain it as a diagnostic option and do not promote it. If it improves the rate, remove controls that do not contribute before considering a new default.
