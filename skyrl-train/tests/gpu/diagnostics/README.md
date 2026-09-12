# Backend numerics diagnostics

These operator-run tests localize FSDP2 and Megatron training divergence. They are deliberately outside
`gpu_ci` and use filenames without the `test_` prefix, so ordinary pytest discovery and PR CI do not run them.
Every command writes `discrepancies.csv` and `results.json` below `SKYRL_NUMERICS_ARTIFACT_DIR`.

Run all three diagnostics on one four-GPU node with the production Megatron runtime:

```bash
sbatch tests/gpu/diagnostics/run_backend_numerics_jupiter.sbatch
```

T1 runs the same serialized toy-MoE batch through an fp32 oracle, the production TorchTitan EP wrapper at
EP=4, and Megatron Core's production gradient-norm helper. Its per-tensor output includes raw-gradient cosine,
norm ratio, a fitted scale and scale-removed residual, and the cosine and norm ratio of a simulated first Adam
step. T2 checks eager, compiled, chunked, inference-only, TP=1, and TP=2 log-probability variants against an
fp64 `log_softmax` oracle. T3 reports per-tensor gradient cosines and norm ratios for the SkyRL for-loop and
grouped-MM experts, Megatron Core's Transformer Engine grouped experts, Hugging Face FlashAttention2, and
Transformer Engine attention.

The checked-in Jupiter batch file runs a frozen GPU environment resolved from this checkout's `uv.lock`. The
Iris frozen-CUDA activation supplies the wheel-backed CUDA runtime and the exact Torch, Megatron Core, and
Transformer Engine closure under test. The job records the checkout revision, GPU topology, imported package
versions, commands, and test output alongside the numerical artifacts.

## FP16 gradient storage

`fp16_gradient_native.py` is an opt-in CPU/Gloo diagnostic requiring the locked MCore 0.18.0
package. Run it explicitly with `python -m pytest skyrl-train/tests/gpu/diagnostics/fp16_gradient_native.py`
from the repository root. It redirects CUDA allocation at the hardware boundary and exercises native
gradient-buffer views, the backward scheduler, dynamic scaler and optimizer step on four CPU ranks in two DP groups.
A one-rank overflow after reduction must also skip the other DP group. It does not qualify CUDA kernels,
NCCL reduce-scatter, Transformer Engine Adam, the full model provider or checkpoint resume.


`fp16_gradient_cuda.py` is the next hardware qualification, invoked as
`torchrun --standalone --nproc-per-node=4 skyrl-train/tests/gpu/diagnostics/fp16_gradient_cuda.py
--source-commit <full-sha> --output <fresh-absolute-directory>` in the frozen Megatron runtime.
It runs TP2/DP2 on one node with native NCCL reduce-scatter, TE Linear backward, native
DistributedOptimizer and TE Adam. Three exact finite updates surround one deliberately injected
post-reduction overflow on rank 3; every rank must skip it and the LR scheduler must pause.
The diagnostic checks FP32 masters/moments, BF16 model storage, FP16 buffer views and unanimous
scale events. Its pass line is `E71B_FP16_CUDA_KERNEL_PASS`. It supplies no Qwen quality, full
model construction, production batch/rollout-loop, inter-node or checkpoint-resume qualification.
