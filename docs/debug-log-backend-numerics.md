# Backend numerics divergence debug log

## Problem

TaskTrove runs diverged immediately between FSDP2 and Megatron. The surviving explanations were an incorrect
gradient norm, different log-probability backwards, or direction-changing MoE or attention gradients.

## Hypotheses

1. `fsdp2_clip_grad_norm_` miscounts expert shards while the underlying gradients are correct.
2. FSDP2 expert gradients are summed across the replicated EP batch instead of averaged.
3. One log-probability or model kernel changes gradient direction between backends.

## Experiments

Three standalone diagnostics under `skyrl-train/tests/gpu/diagnostics/` ran on one Jupiter GH200 node:

- T1 compares FSDP2 EP=4 and Megatron gradient norms with an unsharded FP32 oracle. It also records each raw
  gradient's cosine, norm ratio, fitted scale, scale-removed residual, and simulated first-Adam-step delta.
- T2 compares FSDP2 eager/compiled and Megatron TP=1/TP=2 chunked/unchunked log-probability gradients with an
  FP64 oracle, including temperature scaling and the inference-only contract.
- T3 compares isolated MoE gradients in FP32 and eager-versus-FlashAttention2 attention gradients in BF16.

Runtime: four NVIDIA GH200 120GB GPUs, Torch `2.9.0a0+50eac811a6.nv25.09`, CUDA 13.0, Transformers 5.10.1.
Artifacts are in `/e/scratch/jureap59/feuer1/codex/backend-numerics-results-5bdeeedf` on Jupiter.

## Results before the fix

T1 reported FSDP2 global norm `0.00880580` versus oracle `0.00223208`, a 294.51% relative error. Expert tensors
had cosines from 0.999983 to 0.999985, norm ratios from 3.99983 to 4.00173, fitted scales from 3.99976 to
4.00166, and scale-removed residuals below 0.58%. Router and dense ratios remained approximately one.

This rejects a norm-accounting-only bug because the reconstructed raw expert gradients themselves were 4x.
It also rejects meaningful direction contamination: the error is the EP-size scale. TorchTitan dispatch sends
each rank's replicated logical batch to the owning expert and its backward sums the four contributions.

## Fix

Supported FSDP2 expert holders now implement a typed expert-gradient-averaging contract. `apply_ep` records the
EP size on both grouped and Grug expert holders and attaches `1 / ep_size` hooks only to their parameters.
Router, dense, and shared-expert parameters are unchanged. Hook refresh is idempotent and runs again after
`load_state_dict(assign=True)` replaces parameters during checkpoint loading.

## Results after the fix

All three Jupiter diagnostics pass:

```text
t1=0
t2-fsdp=0
t2-megatron-tp1=0
t2-megatron-tp2=0
t3=0
```

FSDP2's global norm is now `0.00223259` versus oracle `0.00223208`, a 0.0229% relative error. Expert raw-gradient
norm ratios are 0.99996–1.00043 with the same near-unit cosines. Their simulated first-Adam-step delta norm
ratios are 0.99989–0.99996. Megatron's global relative error is `2.55e-9`.

T2's maximum absolute gradient error is `1.79e-7` for FSDP2 and `5.96e-8` for every Megatron TP/chunking case,
well below `1e-5`. T3's FP32 MoE cosine floor is 0.99999985 with norm ratios 0.99982–1.00000. Its BF16 attention
cosine floor is 0.99995738 with norm ratios 0.99904–1.00009.

## Conclusion

The FSDP2/Megatron norm gap was a real FSDP2 expert-gradient scale defect: missing `1 / EP` normalization, not
`fsdp2_clip_grad_norm_` overcounting correct shards. The fix restores the logical single-batch gradient scale.
The standalone diagnostics find no material log-probability, isolated MoE-direction, or attention-direction
disagreement at their stated tolerances.
