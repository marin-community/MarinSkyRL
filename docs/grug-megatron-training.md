# Grug Megatron training

`trainer.strategy=megatron` trains Grug through Megatron-Core with pipeline
parallelism as the primary geometry. Snowball's 26 layers split evenly across
PP2 or PP13; TP must stay at one because the model has five KV heads, and
expert parallelism may be layered on top of PP for the 256 experts. Sample
packing is not yet validated and should stay disabled.

## Hero architecture

The same provider also accepts Hero schema-v2 exports. Hero retains the stored
maximum KV-head count in its checkpoint, then selects the leading local or
global head count for each layer. Its local layers use interleaved half-RoPE;
global layers skip RoPE. The provider reads the global-layer period from the
checkpoint instead of assuming every fourth layer.

Hero's routed experts operate on normalized latent projections. The router and
shared experts still receive the full-width input. Separate shared experts are
added in checkpoint order after the routed output returns to full width.
Schema-v2 expert tensors keep their individual names on import and export.

ShortConv runs before the key norm and after the attention and MLP output
projections. Packed documents have independent convolution histories. With
context parallelism, ranks exchange only the history tails needed by the
kernel and preserve Megatron's two-chunk sequence layout. Local attention uses
Transformer Engine's `a2a` context communication because its `p2p` path rejects
sliding windows; global attention uses `p2p`. Choose TP and CP so both local and
global attention head geometry remains valid. Shared-expert overlap and MLP
chunking are currently unsupported when the Hero modules require them disabled.

The eager Grug model supports Snowball only and rejects Hero configuration. It
must not be used as a Hero reference. The tiny Hero worker test exercises all
Hero parameter families, packed training, repeated updates, and checkpoint
continuation; passing it alone does not establish full-Hero capacity or parity
with Levanter.

The frozen `megatron` runtime includes ARM64 Transformer Engine and
FlashAttention wheels for GB200, built against the same Torch 2.13/CUDA 13.2
versions as H100. See [native wheel builds](../scripts/wheels/README.md).
Transformer Engine 2.11 disables deterministic fused-attention training on
SM100. Deterministic GB200 runs use `trainer.flash_attn=true` with the pinned
FlashAttention 2.8.3 wheel. When using local attention's `a2a` context exchange,
both the query and KV head counts after TP must divide evenly by CP; Hero's
12 local KV heads permit CP4 at TP1, but not CP8.

Megatron's precision-aware AdamW can consume BF16 gradient buffers while keeping
FP32 master weights and moments. Native CPU offload can move a fraction of those
optimizer states out of GPU memory. Configure these together under
`trainer.policy.megatron_config`:

```yaml
ddp_config:
  grad_reduce_in_fp32: false
optimizer_checkpoint_sharding_type: dp_reshardable
optimizer_config_kwargs:
  use_precision_aware_optimizer: true
  store_param_remainders: false
  optimizer_cpu_offload: true
  optimizer_offload_fraction: 0.5
  overlap_cpu_optimizer_d2h_h2d: false
```

This uses fresh AdamW, not Hero's pretraining optimizer history. CPU offload
requires host memory for master weights, moments, and temporary gradients.
The checkpoint integration restores the inner CPU/GPU master weights and Adam
step counters, including subsequent GPU updates after resume. The pinned TE
norm and clipping kernels avoid Megatron's gradient-sized Torch temporary.
Tiny Hero with half of its optimizer offloaded passes exact continuation at
512 and 65,536 tokens on H100. Full-model measurements appear below.

For long sequences, enable Megatron's native weighted SwiGLU fusion under the
same `trainer.policy.megatron_config` section:

```yaml
transformer_config_kwargs:
  bias_activation_fusion: true
  recompute_granularity: full
  recompute_method: uniform
  recompute_num_layers: 1
```

The fused activation avoids the unfused path's large FP32 intermediate before
casting routed outputs back to BF16. It changes intermediate BF16 rounding, so
qualify training and serving comparisons with the selected setting. It does
not require FP8. CPU offload still needs enough host memory for optimizer state
and staging; increasing its fraction can move an out-of-memory failure from
the GPU to the host.

### Full Hero capacity

The 535B step-108000 BF16 export completed repeated AdamW updates through
65,536 tokens on both layouts below. Each layout also saved and restored the
full model and optimizer, then reproduced the next update bit for bit at
65,536 tokens. Query bias stayed frozen. Resume used the same parallel layout.

| Setting | H100 | GB200 |
| --- | --- | --- |
| GPUs / hosts | 256 / 32 | 64 / 16 |
| TP / PP / EP / CP | 1 / 8 / 32 / 4 | 1 / 4 / 16 / 4 |
| CPU optimizer fraction | 1.0 | 0.5 |
| Host memory request per node | 768 GiB | 768 GiB |
| OpenMP threads per worker | 4 | 8 |
| Placement | CP groups within each host | CP groups within each host; one NVLink domain |

Both runs used the precision-aware optimizer and recomputation settings above,
BF16 gradient reduction, overlapping gradient reduction and parameter gathering,
deterministic FlashAttention 2.8.3, and `NCCL_NVLS_ENABLE=0`. Training and scoring
micro-batches were one sequence per worker. The qualification driver used the
packed worker path with one document per row, a 256-token response, and fixed
synthetic trajectories. It did not measure a live rollout workload.

The table reports the second and third updates at each batch size. Timings
include the worker call and memory telemetry. Peak memory is the maximum
PyTorch allocation across ranks during the three measured updates; it excludes
allocations made outside PyTorch.

| Hardware | Context | Global batch | Update seconds | Peak GPU GiB |
| --- | ---: | ---: | ---: | ---: |
| H100 | 32,768 | 8 | 27.605 / 25.092 | 38.339 |
| H100 | 65,536 | 8 | 42.461 / 41.777 | 63.147 |
| H100 | 65,536 | 16 | 46.891 / 42.599 | 64.079 |
| GB200 | 32,768 | 4 | 13.792 / 13.693 | 138.257 |
| GB200 | 65,536 | 4 | 22.654 / 22.738 | 154.022 |
| GB200 | 65,536 | 16 | 37.637 / 36.974 | 145.511 |

Batch 16 ran after checkpoint restore, which changes optimizer and allocator
lifetimes. Its lower GB200 peak does not establish that larger batches need
less memory. The H100 32K measurement used 512 GiB/node; that run later failed
the Ray host-memory guard during 65K restore. The successful restore retry
used 768 GiB/node. CPU offload therefore needs a host-memory budget for both
training and checkpoint loading.

Across the measured training and resume lifecycle, peak container memory was
629.6 GiB/node on H100 and reached the 768 GiB/node limit on GB200. These
container peaks include reclaimable file cache and shared memory; they are not
just optimizer allocations. The largest individual worker peak RSS was
63.3 GiB on H100 and 143.0 GiB on GB200. The successful GB200 run therefore
does not justify reducing its host-memory request.

Each full model-plus-optimizer checkpoint occupies about 6.82 TiB in S3.
H100 save/restore took 676/737 seconds; GB200 took 659/1,406 seconds. These
measurements establish long-context training and restart capacity. They do not
establish 65K vLLM generation capacity or end-to-end RL iteration throughput.

For planning only, the BF16 checkpoint implies about 40.6 GiB of weights per
vLLM rank at TP1/EP64: 25.4 GiB of replicated weights plus 15.2 GiB of routed
expert weights. One 65K sequence adds about 2.67 GiB of BF16 KV data if local
layers retain only their 2,048-token windows. Cache padding, activations,
ShortConv state, kernel workspace, and allocator overhead need additional room.
These are storage estimates, not a tested serving batch limit. Whole-cluster
RL capacity also depends on rollout lengths, scoring, policy epochs, and how
often weights and checkpoints are transferred.

The immutable reports are under
`s3://marin-us-east-02a/marin/users/romain/hero-megatron-01a0bca4/`:
`full-h100-a9/report.json` for H100 32K,
`full-h100-a10/report.json` for H100 65K and exact resume, and
`full-gb200-a6/report.json` for GB200. H100 a10 used source `d386f521`; GB200
used the sealed `7dc88a2b` source bundle. The GB200 report's descriptive source
field says `development checkout`; its validated runtime identity and archived
bundle establish the revision. Source bundles and launch arguments are stored
under `evidence/qualification-20260920/` and
`evidence/qualification-20260920b/` at the same S3 prefix.

### Controlled publication cycle

Full Hero completed a controlled cycle on the 256-H100 layout above with
64 colocated vLLM ranks at TP1/EP64. It published the initial weights,
generated eight four-token responses, scored and trained with captured routes,
saved a checkpoint, made another update, restored, published the restored
weights, and generated again. Readback matched the selected weight families
and all 48 FP32 router biases exactly before and after training. The biases
stayed frozen. Route replay hit and executed-route match fractions were both
1.0, router gradient norm was 4.185, and the training log-ratio maximum was zero.

Initial and updated publication took 178 and 279 seconds. The replay update
took 67 seconds; checkpoint save and restore took 671 and 623 seconds. Serving
used a 128-token maximum context and GPU memory utilization 0.58. These timings
do not measure long-context generation or steady-state rollout throughput.

The serving/training log-probability gap was 0.3623 maximum and 0.0324 mean.
This cycle establishes the training and publication lifecycle; cross-backend
numerical qualification remains open. A later 64-GB200 cycle (Megatron
TP1/PP4/EP16/CP4, serving TP1/PP1/EP64/DP64) completed route replay, backward
and one AdamW step, checkpoint save/restore, updated optimizer offload, updated
weight publication, and final generation. Replay hit and executed-route match
were both 1.0, and checked updated weights matched exactly. The full-prefix
diagnostic reduced the serving/training log-probability gap from 0.0507 maximum
and 0.0174 mean to 0.0233 maximum and 0.0104 mean. That residual remains a
numerical qualification gap; the cycle proves the lifecycle, not parity.

The successful GB200 report is
`s3://marin-us-east-02a/marin/users/romain/hero-megatron-01a0bca4/full-cycle-gb200-a4/report.json`,
from SkyRL `00625398` and vLLM `9ff94e459611`. It used the frozen query-bias
policy. The older H100 cycle above remains useful layout evidence, but its
larger aggregate gap is not a parity bound.

The successful H100 report is `full-cycle-h100-a6/report.json` under the S3
prefix above, from SkyRL `bf37e425` and vLLM `9ff94e459611`. Its source bundle,
launch arguments, and focused memory-lifetime regressions are under
`evidence/bf37e425/`.

The port lives in two modules:

- `skyrl_train.models.grug_megatron` holds the Megatron-Core modules that a
  stock GPT spec cannot express: the gated RMS norms, the weightless QK norm,
  attention with the per-layer query scale, XSA, and the per-head output gate,
  the biased top-(k+1) sigmoid router, and a `GPTModel` subclass that applies
  the gated embedding norm on the first pipeline stage.
- `skyrl_train.models.grug_megatron_bridge` registers the Megatron-Bridge
  provider and weight mappings. Importing the Megatron worker registers the
  bridge so `AutoBridge.from_hf_pretrained` resolves Grug checkpoints.

Sliding-window attention on local layers, no RoPE on long layers, half-RoPE,
grouped-GEMM experts with a shared expert, and GQA use Megatron-Core settings
chosen by the provider (`window_size`, `window_attn_skip_freq`, `no_rope_freq`,
`rotary_percent=0.5`, `moe_grouped_gemm`, `moe_shared_expert_intermediate_size`).
Attention runs through Transformer Engine's fused backend; `trainer.flash_attn`
selects the flash backend instead.

## Weights

The HF checkpoint keeps its stacked `[E, ...]` expert tensors. The bridge maps
each Megatron per-expert grouped-GEMM weight to one slice of the stacked tensor
on import and re-stacks on export, so exported checkpoints and weight sync use
the same names as FSDP2 training and vLLM serving. The router bias becomes
Megatron's persistent fp32 `expert_bias` buffer and is sent to vLLM in fp32 in
its own weight-sync bucket; every other tensor is sent in the generator dtype.

Colocated Megatron publication allocates its CUDA IPC packing buffers with
expandable segments disabled, then immediately restores the training allocator
settings. In the pinned Torch 2.13 runtime, GB200 fabric-handle cleanup raises
`std::get: wrong index for variant` after an expandable buffer crosses IPC.
Model and optimizer allocations can still use expandable segments.

When serving uses fewer GPUs than training, only training ranks with a
colocated receiver create IPC handles. All ranks still participate in export
collectives. Creating handles on unused ranks leaves their packing buffers
resident because no receiver can release the IPC reference.

Reloading model weights onto the GPU releases their pinned CPU staging copy
and clears unused pinned allocations after the transfer finishes. Otherwise
that obsolete copy competes with optimizer offload during the next rollout.

The pinned vLLM reload loader discards inputs for experts owned by other ranks
and copies retained weight views into compact storage. Otherwise a small view
can keep an entire IPC bucket alive until its layer finishes loading, exhausting
the memory needed to gather the next expert tensor.

Re-stacking gathers every expert of a layer onto each rank before the tensor
is sent, which needs a few GiB of headroom beyond the resident model, gradient
buffers, and optimizer state. On the 67B-A2B snowball checkpoint at PP2 x EP8 x
DP2 that headroom does not exist once the optimizer state is materialized, and a
colocated reference model needs the same room for its forward, so disaggregated
runs set `trainer.offload_optimizer_during_rollouts=true` to keep the optimizer
state and gradient buffers on CPU from each policy update until the next one.

## Numerics

Two Megatron behaviours break the on-policy contract that the recomputed old
log-probabilities equal the training forward, which FSDP2 satisfies exactly:

- Megatron's unfused unpermute combines the top-k expert outputs with an atomic
  scatter-add. For top-2 routing the two-term sum is order-independent, but Grug
  routes top-4, so the forward was not reproducible run to run. The bridge forces
  `moe_permute_fusion`, whose Transformer Engine kernels reduce in a fixed order.
- cuBLAS selects kernels per GEMM shape, and at Grug's width the per-row rounding
  differs between kernels. The log-probability forward and the training forward
  must therefore use the same micro-batch size
  (`micro_forward_batch_size_per_gpu == micro_train_batch_size_per_gpu`);
  `validate_cfg` rejects Megatron runs where they differ.

`test_grug_megatron_train_forward_matches_eval_forward` guards both on a toy
model and on a Snowball-shaped tiny model (256 experts, top-4, 20/5 heads,
2048-token window) with variable-length rows at long sequences, and
`test_grug_megatron_eval_forward_is_independent_of_peer_rank_batch` checks that
expert-parallel co-batching does not leak between ranks.

## Memory

The Snowball configs select Megatron's `dp_reshardable` optimizer format, which
writes data-parallel-local shards without gathering optimizer state onto
data-parallel rank zero. Resume must keep the tensor, pipeline, context, and
expert geometry fixed.

S3 checkpoints aggregate each rank's torch-dist items into one multipart object.
Each rank overlaps up to four 64MiB `UploadPart` requests while tensor copy-ahead
is bounded at 1GiB. Including the current multipart buffer, the explicit staging
budget is 1.3125GiB per rank, or 10.5GiB on an eight-GPU host, plus transient
serialization and client overhead. Only small control files such as `common.pt`,
`metadata.json`, and the Hugging Face configuration use a local temporary
directory. Local-path checkpoints retain Megatron Core's standard writer.

S3 restores stage only control files locally. PyTorch DCP reads the tensor
ranges needed by each rank directly from the rank objects, so every node does
not need disk space for the full model and optimizer checkpoint. Parameter
gathering completes before checkpointing, HF export, validation snapshots, and
vLLM publication, including when normal training overlaps those gathers with
the next forward pass.

Megatron Core 0.18's Multi-Storage Client path remains disabled because its
object writer buffers each complete remote file in `BytesIO` before uploading
it. PyTorch 2.11's one-file-per-rank loop retains all resolved tensors until the
file closes, even though that dictionary is only needed for safetensors. SkyRL's
writer omits that retention for torch serialization and streams the bounded
copy-ahead window into one rank object.

On the GPU, the last pipeline stage holds the vocab-sized logits; the loss
computes entropy under no_grad unless an entropy loss is configured, which
avoids saving two vocab-sized copies for backward. The Snowball configs also
set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Throughput

With 1024-token prompts, 8192-token generations, 64 prompts x 8 samples per
step, four policy nodes at PP2 x EP8 x DP16 and four vLLM nodes at DP8 x EP8,
against the FSDP2 trainer at the same geometry:

| phase | Megatron | FSDP2 |
| --- | --- | --- |
| step | 190-196s | 634s |
| policy_train | 26-27s | 457s |
| generate | 134-139s | 131s |
| fwd_logprobs | 6-7s | 32s |
| sync_weights | 15-17s | 11.3s |

With equal micro-batch sizes the recomputed old log-probs and the training
forward agree exactly at this scale: `policy/log_ratio_abs_max` is 0 and
`policy/ppo_ratio_exact_unit_fraction` is 1.0 on every step.

The Megatron numbers use `cloud/iris/configs/snowball_megatron_full.yaml`,
which overlaps gradient reduction and parameter gathering with compute and
reduces gradients in bf16. Generation takes about 70% of the step, so further
gains come from the generator rather than the trainer.

## Query bias

Megatron supports the `frozen` and `loss_free` query-bias modes. Frozen bias
steers expert selection exactly as in the HF model and is never updated. The
`loss_free` mode requires Grug route replay so the worker can count the routes
that actually executed; the update is applied after the optimizer step. The
trained H100 replay gate covers this policy and observes a nonzero bias change.
The full-Hero GB200 lifecycle above used frozen bias, so it does not qualify
`loss_free` at 535B. The `interpolate` and `replace` quantile-balancing modes
remain FSDP2-only.

## Validation

`skyrl-train/tests/gpu/test_grug_megatron.py` covers HF log-probability parity
at PP1, PP2, and PP2+EP2, a PP2 training step with an export round trip, and a
four-H100 disaggregated cycle with Marin vLLM. Run it on Iris with
`skyrl-train/ci/marin_nightly/run_grug_megatron.sh`, which resolves the frozen
`megatron` runtime profile.
