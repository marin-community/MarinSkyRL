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
DP2 that headroom does not exist once the optimizer state is materialized, so
disaggregated runs set `trainer.offload_optimizer_for_weight_sync=true` to move
the optimizer state and gradient buffers to CPU for the duration of each sync.

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
