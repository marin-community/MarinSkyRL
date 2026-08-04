# Grug FSDP2 training

This is a policy-only, EP1 FSDP2 implementation. The trainer uses the
canonical PyTorch model in `skyrl_train.models.grug_moe`; vLLM serves the same
HF checkpoint. Eager attention remains the correctness reference. Policy
training also supports `flash_attention_2`, selected with
`trainer.flash_attn=true` or `trainer.attn_backend=flash_attention_2`.
Unsupported fused requests fail instead of falling back to eager attention.

The fused path requires BF16 or FP16 on a supported CUDA GPU. It preserves
Grug's 20-query/5-KV-head GQA, half-RoPE on local layers, full causal long
layers, 2,048-token local window, QK scaling, XSA, and per-head gating. Dense
unpacked batches may contain left or right padding; FlashAttention unpads valid
tokens, and model outputs are defined at valid query positions. Sample packing,
trainer EP/CP, R3/router replay, grouped MoE, LoRA/4-bit loading, and PKO remain
unsupported.

## Runtime support

Grug serving uses the Marin vLLM wheels selected by the root `uv.lock`. The lock
chooses immutable x86_64 and aarch64 assets for the H100 and GB200 execution
platforms, respectively. The standard Iris environment verifies `vllm._C`, the
cuMem allocator, and `GrugMoeForCausalLM` before training starts.

The eager policy path does not require FlashAttention. Selecting the fused policy
path still requires a compiled FlashAttention build compatible with the locked
Torch and CUDA ABI.

## Query bias

For every optimizer window, each rank counts non-padding tokens and uses
`q = max(1, floor(tokens * top_k / num_experts))`. Each router retains only its
per-expert top-q values of `unbiased_logit - biased_(K+1)th_logit`; concatenating
and top-k reducing these candidates is exactly equivalent to retaining the full
token-by-expert matrix. The q-th values are averaged across ranks. After a real
optimizer step, the next persistent FP32 bias is `center(-beta)`. A skipped
non-finite step discards the observation and preserves the previous bias.

This padding exclusion is the RL adaptation of Levanter's fixed, padding-free
batch geometry. The strict cross-framework fixture is padding-free.
