# Grug FSDP2 training

This is a policy-only, EP1 FSDP2 implementation. The trainer uses the
canonical PyTorch model in `skyrl_train.models.grug_moe`; vLLM serves the same
HF checkpoint. Packing, FlashAttention, trainer EP/CP, R3/router
replay, grouped MoE, LoRA/4-bit loading, and PKO are intentionally rejected.

## Runtime image

Grug serving requires the Marin vLLM fork at commit `4b55591306c9`. Resolve the
cluster's standard image from `cloud/iris/gpu_rl_images.py` and verify that it
contains this fork. If it does not, pass an explicit verified image by immutable
digest.

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
