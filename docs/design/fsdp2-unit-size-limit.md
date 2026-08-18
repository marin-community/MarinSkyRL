# Maximum FSDP2 unit size

Status: reviewed independent implementation plan

## Decision

Add a user-configurable bounded FSDP2 unit policy with separate ceilings for all-gather and reduce-scatter payloads:

```yaml
fsdp_config:
  unit_size_policy:
    mode: unbounded
    max_all_gather_bytes: 268435456       # 256 MiB when bounded
    max_reduce_scatter_bytes: 402653184   # 384 MiB when bounded
    min_target_all_gather_bytes: 134217728  # 128 MiB balancing heuristic
```

The initial bounded policy uses a 256 MiB all-gather ceiling and a 384 MiB reduce-scatter ceiling. The repository default remains `unbounded` until supported-model inventory and Jupiter validation are complete. When an operator selects `bounded`, the values above are the defaults.

The ceilings describe padded FSDP collective inputs, not NCCL packet sizes. NCCL chooses transport chunks and InfiniBand packets below this layer. Smaller FSDP units create more collective calls; they do not directly create 128 or 256 MiB wire packets.

The policy never silently exempts an oversized atomic unit. If a model adapter cannot split a unit without changing its computation or checkpoint contract, model construction fails before distributed collectives start and reports the unit, both directional payloads, and the required adapter.

## Why two ceilings are necessary

FSDP2 all-gathers parameters in `param_dtype` and reduce-scatters gradients in `reduce_dtype`. MarinSkyRL defaults to BF16 parameter communication and FP32 gradient reduction. A unit with 151 million elements is therefore approximately 302 MB for all-gather and 604 MB for reduce-scatter before padding. One byte limit cannot express both costs without forcing BF16 all-gather partitions below the useful-size region.

For each proposed unit, the planner reports:

- `all_gather_bytes`: the EP-coordinate-local, FSDP-unsharded parameter elements after FSDP padding, multiplied by the parameter communication dtype size;
- `reduce_scatter_bytes`: the padded unsharded gradient elements multiplied by the reduction dtype size.

The exact buffer sizes must be verified against a supported custom collective recorder or profiler. A DTensor's global `numel()` is not the FSDP communicator payload.

The preserved 151-million-element event is consistent with the grouped-expert holder, but that mapping and the event dtype must be proved by the inventory. Earlier stalls have also involved smaller shapes, so a unit limit cannot explain or fix every observed wedge.

## Research and capacity basis

The [PyTorch FSDP paper](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf) holds total communication constant and varies all-gather size. Communication time rises rapidly below 33 million FP32 elements, approximately 132 MB on the paper's hardware. This is an experimental reference point for all-gather balancing, not a universal lower bound and not a reduce-scatter result.

Jupiter Booster nodes expose four GH200 GPUs and four NDR200 ConnectX-7 interfaces. NDR200 provides 200 Gbit/s, or 25 GB/s raw, per interface. The run must capture NCCL topology output to verify GPU-to-HCA affinity rather than infer affinity from device counts. See the [JUPITER Booster configuration](https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html).

For FSDP group size `p` and padded collective input `U`, a ring reduce-scatter or all-gather moves approximately `(p - 1) / p * U` bytes per rank, excluding protocol, staging, and contention:

| Padded collective input | FSDP4 bytes per rank | FSDP4 raw time at 25 GB/s | FSDP8 bytes per rank | FSDP8 raw time at 25 GB/s |
| --- | ---: | ---: | ---: | ---: |
| 128 MiB | 96 MiB | 4.0 ms | 112 MiB | 4.7 ms |
| 256 MiB | 192 MiB | 8.1 ms | 224 MiB | 9.4 ms |
| 384 MiB | 288 MiB | 12.1 ms | 336 MiB | 14.1 ms |
| 512 MiB | 384 MiB | 16.1 ms | 448 MiB | 18.8 ms |

For the 151-million-element example under BF16/FP32, two balanced chunks are approximately 151 MB per all-gather and 302 MB per reduce-scatter. Both exceed the paper's all-gather reference point or stay below the directional ceilings. This motivates the 256/384 MiB initial values. It does not establish them as globally optimal.

## Configuration contract

Define `FSDPUnitSizeMode(StrEnum)` with `unbounded` and `bounded`. Centralize the policy defaults once and validate:

- bounded mode requires positive directional ceilings;
- `min_target_all_gather_bytes <= max_all_gather_bytes`;
- the policy is used only with FSDP2;
- irrelevant bounded fields are rejected or clearly reported when mode is unbounded;
- every FSDP2 role enforces the all-gather ceiling, while only trainable units in training roles enforce the reduce-scatter ceiling;
- the reference role does not split a unit solely because its hypothetical FP32 gradient payload would exceed the reduce-scatter ceiling;
- policy, critic, and reference may choose different policies because their models, meshes, and dtypes may differ.

The minimum target is only a balancing heuristic for partitions introduced by this policy. Naturally small existing units are valid and are not merged or rejected.

## Feasibility and inventory phase

Before structural adaptation, inventory the complete supported model matrix under configured EP/FSDP geometry and dtypes. Include grouped experts, transformer layers, embeddings, output heads, tied weights, and residual root parameters. For every predicted unit, record parameter names, local shapes, padding, both payloads, and adapter type.

Structural adaptation occurs before the full-state snapshot and before EP distribution:

1. Predict EP-coordinate-local shapes from the unsharded model and configured meshes.
2. Produce a deterministic partition plan and verify expert-count divisibility across EP and FSDP.
3. Rewrite supported model structure.
4. Take the canonical full-state snapshot.
5. Apply the selected EP backend to each adapted child.
6. Compose FSDP sharding on each child, then shard parents and root bottom-up.
7. Load the canonical checkpoint into the adapted structure.
8. Compare predicted payloads with observed collective buffers.

Today, `apply_ep()` distributes an expert holder and immediately applies `fully_shard()`. The implementation must separate planning/adaptation from that lifecycle; adapting a composed DTensor afterward is too late.

The first implementation remains opt-in unless every model used by launcher configurations either satisfies the ceilings naturally or has a validated adapter. Default promotion requires the complete supported-model inventory, not only the affected MoE.

## Partitioning rules

For a splittable candidate, choose the smallest partition count that satisfies every applicable directional ceiling after padding. All-gather applies to all FSDP units. Reduce-scatter applies only when the unit has trainable gradients in a training role. Produce balanced partitions near equal collective cost; do not greedily fill one ceiling and leave a small tail. Prefer introduced all-gather partitions at or above the minimum target when the ceilings and structural divisibility allow it.

The directional ceilings and structural divisibility are hard requirements. The minimum target is not. When no legal plan keeps every introduced partition above the target, choose the plan that minimizes the number and total bytes of undersized partitions, then log those partitions for the benchmark. Parameter order and the plan must be identical on every rank.

## Grouped-expert adapter

The primary adapter splits the global expert axis into EP-aligned compute chunks before EP distribution. EP then gives every rank balanced local shards of those chunks. It preserves these invariants:

- each MoE layer performs exactly one EP token dispatch and one combine;
- chunking happens after dispatch and before combine;
- routed rows are partitioned by local expert chunk, processed by the existing grouped matrix multiplication, and restored to dispatch order;
- empty-token chunks and uneven routing are valid;
- selected expert IDs and router replay retain their existing global-to-local mapping;
- canonical checkpoint and exported weight names remain `w1`, `w2`, and `w3`.

Do not implement each chunk as an independently dispatched expert module; that would multiply EP all-to-all operations. Do not use projection holders unless a distributed prototype proves safe FSDP weight lifetimes and shows that it reduces collective occupancy. `GroupedExperts.forward()` needs all three projections, so a structural projection split is not assumed viable.

The adapter must teach Torch EP and any supported DeepEP path to distribute each child. Unsupported combinations fail validation rather than silently bypassing the ceiling.

## Implementation plan

1. Add immutable unit descriptors and a pure directional-payload planner beside `fsdp_utils.py`.
2. Implement the supported-model dry-run inventory and deterministic cross-rank validation.
3. Add the grouped-expert feasibility spike before changing the initialization lifecycle. Measure two expert-axis chunks, empty-token behavior, and grouped-GEMM efficiency.
4. Adapt model structure before the state snapshot, then update EP and FSDP composition to consume the plan.
5. Preserve canonical state keys through full-state load, DCP, HF export, and streamed inference weight extraction.
6. Assert observed collective input sizes against the plan and emit unit-count and payload distributions.
7. Run the benchmark matrix and leave the base mode unbounded until the acceptance gates pass.

## Cost of smaller units

- More all-gather and reduce-scatter calls increase NCCL launch, rendezvous, and host scheduling overhead.
- Units below the bandwidth-saturation region can turn a bandwidth-bound workload into a latency-bound workload.
- More hooks and allocation lifetimes can increase allocator fragmentation and peak memory.
- Backward prefetch can make several units resident simultaneously, offsetting the intended reduction in per-unit memory and network occupancy.
- Additional collective boundaries can increase FSDP/EP interleaving.
- Expert-axis chunks add grouped-GEMM launches and may reduce compute efficiency.
- More structural units complicate checkpoints, weight synchronization, and model-specific maintenance.

The policy therefore balances partitions and measures both communication overlap and grouped-GEMM performance. It does not assume that smaller is always safer.

## Test plan

CPU behavior tests:

- directional byte accounting uses EP-local shapes, FSDP padding, `param_dtype`, and `reduce_dtype`;
- the BF16/FP32 151-million-element example produces two balanced partitions under the default bounded policy;
- partitions obey expert divisibility and never create an avoidable small tail;
- oversized atomic units fail before process-group collectives and name the missing adapter;
- every rank derives the same plan;
- unbounded mode produces the current hierarchy exactly;
- full-state load and canonical external keys survive adaptation.

GPU behavior tests:

- predicted all-gather and reduce-scatter inputs match sizes observed through public collective callbacks or profiling;
- one dispatch and one combine occur per MoE layer before and after chunking;
- forward, backward, and multiple optimizer steps match for logits, routed expert IDs, loss, expert/router/dense gradients, and parameter deltas;
- router replay crosses chunk boundaries under gradient checkpointing;
- uneven routing, zero-token chunks, gradient accumulation, and CPU offload are covered;
- full-state initialization, DCP model and optimizer resume, HF export, and streamed inference weight sync preserve values and ordering;
- Torch EP and DeepEP pass, or unsupported combinations fail before launch;
- FSDP4 and FSDP8 production-shaped harnesses assert all applicable observed directional ceilings.

Use NCCL profiler or Kineto/NVTX timestamps to measure interval overlap. Flight-recorder enqueue and completion sequences alone do not prove physical link concurrency.

## Jupiter benchmark matrix

| Variant | Purpose |
| --- | --- |
| Unbounded current layout | Throughput and wedge-rate baseline |
| 512/512 MiB ceilings | Large-unit control |
| 256/384 MiB ceilings | Candidate bounded defaults |
| 128/256 MiB ceilings | Expose launch and small-unit penalties |
| Candidate bounded policy plus conservative communication | Measure interaction without coupling the configurations |

Record step time, tokens/s, collective latency and overlap distributions, grouped-GEMM time, GPU and host memory, collective count, NCCL topology, and flight-recorder completion sequences.

## Acceptance and default promotion

Bounded mode may merge as opt-in after correctness tests pass. Promote it to the repository default only if:

- every supported launcher model passes inventory or has an adapter;
- all observed payloads satisfy their applicable hard ceilings;
- numerics, checkpoints, exports, and inference weight sync pass;
- repeated benchmarks put the steady-state throughput regression within 5 percent with a reported confidence interval, unless a larger stability benefit is explicitly accepted;
- GPU and host memory tolerances are defined and met;
- multiple matched runs across node sets provide enough aggregate exposure that the Poisson 95 percent upper bound for the candidate wedge rate, approximately `3 / zero-failure exposure`, is below the measured baseline rate.

If the candidate values create excessive launch overhead, grouped-GEMM loss, or FSDP/EP interleaving, keep the user control but select different directional defaults or leave the repository unbounded. Do not retain 256/384 MiB only because they were the initial estimates.
