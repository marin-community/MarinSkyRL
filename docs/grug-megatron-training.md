# Grug Megatron training

`trainer.strategy=megatron` trains Grug through Megatron-Core with pipeline
parallelism as the primary geometry. Grug's 26 layers split evenly across
PP2 or PP13; TP must stay at one because the model has five KV heads, and
expert parallelism may be layered on top of PP for the 256 experts. Sample
packing is not yet validated and should stay disabled.

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

Policy nodes for the 67B-A2B snowball checkpoint need 1800GB of host memory
and 1000GB of disk to save a Megatron checkpoint: the writer stages about 146GB
per rank in host RAM, then leaves a 46GB shard per rank in `/tmp` beside the
staged HF checkpoint until the upload finishes. On the GPU, the last pipeline
stage holds the vocab-sized logits; the loss computes entropy under no_grad
unless an entropy loss is configured, which avoids saving two vocab-sized
copies for backward. The snowball configs also set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

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

Only the frozen query-bias mode is supported on Megatron. The bias steers
expert selection exactly as in the HF model but is never updated; the
`loss_free`, `interpolate`, and `replace` modes remain FSDP2-only.

## Validation

`skyrl-train/tests/gpu/test_grug_megatron.py` covers HF log-probability parity
at PP1, PP2, and PP2+EP2, a PP2 training step with an export round trip, and a
four-H100 disaggregated cycle with Marin vLLM. Run it on Iris with
`skyrl-train/ci/marin_nightly/run_grug_megatron.sh`, which resolves the frozen
`megatron` runtime profile.
